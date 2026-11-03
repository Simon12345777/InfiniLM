"""Mooncake transfer scheduling and reply handling without GPU hardware."""

import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np


class MooncakeReceiveTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import msgspec
            from mooncake.engine import TransferEngine  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"Mooncake test dependencies unavailable: {exc}")
        from infinilm.kv_connector.mooncake import mooncake_connector_worker as rx

        cls.rx = rx
        cls.msgpack = msgspec.msgpack
        cls.status = rx.MooncakeXferResponseStatus
        cls.result = rx.MooncakeXferReqStatus

    def setUp(self):
        rx = self.rx
        worker = rx.MooncakeConnectorWorker.__new__(rx.MooncakeConnectorWorker)
        worker.shutdown = lambda: None  # No TransferEngine, sockets or threads to stop.
        worker.hostname, worker.rpc_port = "localhost", 0
        worker.tp_size, worker.tp_rank = 1, 0
        worker.kv_caches_base_addr, worker.kv_flag_ptrs = [], []
        worker.kv_flag = np.zeros(2, dtype=np.uint8)
        worker._encoder = self.msgpack.Encoder()
        worker._xfer_resp_decoder = self.msgpack.Decoder(rx.MooncakeXferResponse)
        worker.finished_recving_reqs = set()
        worker.xfer_failed_recving_reqs_ids = set()
        worker.timeout_reqs_to_recv = {}
        worker.async_zmq_ctx = None
        worker._prefix_limit = worker._prefix_inflight = 0
        worker._prefix_waiters = set()
        self.worker = worker
        self.pulls = {
            rid: rx.PullReqMeta(rid, "x-" + rid, [i], "P", "", pull_tasks_count=1)
            for i, rid in enumerate(("short", "long"))
        }
        self.replies = asyncio.Queue()
        self.socket = MagicMock()
        self.socket.send = AsyncMock()
        self.socket.recv = self.replies.get
        factory = patch.object(rx, "make_zmq_socket")
        factory.start().return_value.__enter__.return_value = self.socket
        self.addCleanup(factory.stop)

    def reply(self, status, rid, result):
        response = self.rx.MooncakeXferResponse(status, [rid], [result])
        self.replies.put_nowait(self.msgpack.encode(response))

    async def exercise_chunk_transfer(self, cancel=False, fail_prefix=False, two_steps=False):
        from infinilm.config.kv_transfer import KVTransferConfig
        from infinilm.kv_connector.mooncake.mooncake_connector_scheduler import (
            MooncakeConnectorScheduler,
        )
        from infinilm.llm.request import RequestStatus

        w = self.worker
        w.sender_loop = asyncio.get_running_loop()
        w._sender_executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(w._sender_executor.shutdown)
        w.reqs_need_send, w.finished_sending_reqs = {}, set()
        w._active_pulls = set()
        w.kv_caches_base_addr, w.block_len = [1000], 16
        w.kv_flag_src_ptrs = list(range(100))
        started, release = asyncio.Event(), threading.Event()
        second_write = asyncio.Event()
        writes = []

        def write(session, src, dst, lengths):
            writes.append((list(src), list(dst), list(lengths)))
            if len(writes) == 1:
                w.sender_loop.call_soon_threadsafe(started.set)
                if not release.wait(3):
                    raise TimeoutError("prefix writer not released")
                return int(fail_prefix)
            if len(writes) == 2:
                w.sender_loop.call_soon_threadsafe(second_write.set)
            return 0

        w.engine = SimpleNamespace(batch_transfer_sync_write=write)
        cfg = KVTransferConfig(kv_connector="MooncakeConnector", kv_role="kv_producer",
                               kv_connector_extra_config={"chunk_transfer_overlap": True})
        scheduler = MooncakeConnectorScheduler(cfg, "P")
        req = SimpleNamespace(request_id="p", num_computed_tokens=16 if two_steps else 32,
                              block_table=[4] if two_steps else [4, 5], get_prompt_length=lambda: 35,
                              sampling_params=SimpleNamespace(max_tokens=1),
                              kv_transfer_params={"transfer_id": "x", "do_remote_decode": True})
        scheduler.update_state_after_alloc(req, req.block_table, 0, 16)
        scheduler.publish_computed_chunk(req, 16)
        await w.record_send_reqs(scheduler.build_connector_meta())
        dest = self.rx.MooncakeXferMetadata("D", 42, 1, 0, {"d": ("x", [6, 7, 8])},
                                            [2000], list(range(300, 310)))
        sock = AsyncMock()
        task = asyncio.create_task(w.send_kv_to_decode(b"id", sock, dest))
        try:
            await asyncio.wait_for(started.wait(), 2)
            with self.assertRaisesRegex(ValueError, "Concurrent Mooncake pulls"):
                await w.send_kv_to_decode(b"duplicate", sock, dest)
            if two_steps:
                req.block_table.append(5)
                req.num_computed_tokens = 32
                scheduler.publish_computed_chunk(req, 16)
                await w.record_send_reqs(scheduler.build_connector_meta())
                release.set()
                await asyncio.wait_for(second_write.wait(), 2)
            # The model can finish its partial tail while the prefix write is in flight.
            req.block_table.append(9)
            req.status = RequestStatus.CANCELED if cancel else RequestStatus.FINISHED
            delay_free, _ = scheduler.request_finished(req, req.block_table, 16)
            self.assertTrue(delay_free)
            await w.record_send_reqs(scheduler.build_connector_meta())
            self.assertFalse(w.finished_sending_reqs)
            sock.send_multipart.assert_not_called()
            release.set()
            await asyncio.wait_for(task, 2)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        response = self.msgpack.decode(sock.send_multipart.call_args.args[0][1],
                                       type=self.rx.MooncakeXferResponse)
        self.assertEqual(w.finished_sending_reqs, {"p"})
        if two_steps:
            self.assertEqual(writes.pop(0), ([1064], [2096], [16]))
            self.assertEqual(writes[0], ([1080], [2112], [16]))
        else:
            self.assertEqual(writes[0], ([1064], [2096], [32]))
        if cancel:
            self.assertEqual(len(writes), 1)
            self.assertEqual(response.reqs_statues, [self.result.XFER_FAIL])
        else:
            expected = ([1064, 1144], [2096, 2128], [32, 16]) if fail_prefix else (
                [1144], [2128], [16]
            )
            self.assertEqual(writes[1], expected)
            self.assertEqual(writes[2], ([6, 7, 8], [306, 307, 308], [1, 1, 1]))
            self.assertEqual(response.reqs_statues, [self.result.SUCCESS])

    async def test_chunk_transfer_final_writes_tail_and_flags(self):
        await self.exercise_chunk_transfer()

    async def test_chunk_arriving_during_write_only_sends_new_pages(self):
        await self.exercise_chunk_transfer(two_steps=True)

    async def test_cancel_waits_for_inflight_prefix(self):
        await self.exercise_chunk_transfer(cancel=True)

    async def test_failed_prefix_is_rewritten_before_success(self):
        await self.exercise_chunk_transfer(fail_prefix=True)

    async def exercise_prefix_limit(self, finalize_while_waiting):
        w = self.worker
        w._prefix_limit = 1
        w._active_pulls = set()
        w.reqs_need_send, w.finished_sending_reqs = {}, set()
        w.sender_loop = asyncio.get_running_loop()
        w._sender_executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(w._sender_executor.shutdown)
        w.kv_caches_base_addr, w.block_len = [1000], 16
        w.kv_flag_src_ptrs = list(range(32))
        w.engine = MagicMock()
        w.engine.batch_transfer_sync_write.return_value = 0
        started, release, merged = asyncio.Event(), asyncio.Event(), asyncio.Event()
        ranges = []

        async def prefix(rid, state, meta, start, end):
            ranges.append((rid, start, end))
            if rid == "A":
                started.set()
                await release.wait()
            else:
                merged.set()
            return 0

        w._write_sealed_pages = prefix
        tasks = []
        async def start(rid, dest):
            update = self.rx.MooncakeConnectorMetadata()
            update.reqs_to_send[rid] = (rid, [])
            update.partial_sends[rid] = ([1], 3)
            await w.record_send_reqs(update)
            meta = self.rx.MooncakeXferMetadata("D", 42, 1, 0, {rid: (rid, dest)},
                                                [2000], list(range(32)))
            task = asyncio.create_task(w.send_kv_to_decode(b"id", AsyncMock(), meta))
            tasks.append(task)
            return task

        async def finish(rid):
            update = self.rx.MooncakeConnectorMetadata()
            update.reqs_to_send[rid] = (rid, [1, 2, 3])
            await w.record_send_reqs(update)

        async def wait_for_limit():
            while not w._prefix_waiters:
                await asyncio.sleep(0)

        try:
            a = await start("A", [6, 7, 8])
            await asyncio.wait_for(started.wait(), 2)
            b = await start("B", [9, 10, 11])
            await asyncio.wait_for(wait_for_limit(), 2)
            update = self.rx.MooncakeConnectorMetadata()
            update.partial_sends["B"] = ([1, 2], 3)
            await w.record_send_reqs(update)
            if finalize_while_waiting:
                await finish("B")
                await asyncio.wait_for(b, 2)
                self.assertEqual(w.finished_sending_reqs, {"B"})
                self.assertEqual(w._prefix_inflight, 1)
                self.assertEqual(ranges, [("A", 0, 1)])
            else:
                release.set()
                await asyncio.wait_for(merged.wait(), 2)
                self.assertEqual(ranges, [("A", 0, 1), ("B", 0, 2)])
                await finish("B")
                await asyncio.wait_for(b, 2)
            release.set()
            await finish("A")
            await asyncio.wait_for(a, 2)
            self.assertFalse(w._prefix_waiters)
            self.assertEqual(w._prefix_inflight, 0)
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_final_bypasses_saturated_worker_prefix_limit(self):
        await self.exercise_prefix_limit(finalize_while_waiting=True)

    async def test_pending_prefix_coalesces_when_capacity_returns(self):
        await self.exercise_prefix_limit(finalize_while_waiting=False)

    async def test_short_is_released_after_flags_before_finish(self):
        worker = self.worker
        checking = asyncio.Event()
        next_reply = asyncio.Event()
        wait_flags = worker._wait_for_kv_flags_ready

        async def observe_flags(block_ids):
            checking.set()
            await wait_flags(block_ids)

        async def recv():
            if checking.is_set():
                next_reply.set()
            return await self.replies.get()

        worker._wait_for_kv_flags_ready = observe_flags
        self.socket.recv = recv
        self.reply(self.status.CONTINUE, "short", self.result.SUCCESS)
        task = asyncio.create_task(
            worker.receive_kv_from_single_worker("P", "", self.pulls)
        )
        try:
            await asyncio.wait_for(checking.wait(), 2)
            self.assertEqual(worker.finished_recving_reqs, set())
            worker.kv_flag[0] = 1
            await asyncio.wait_for(next_reply.wait(), 2)
            self.assertEqual(await worker.fetch_finished_recving_reqs(), ({"short"}, set()))
            self.assertEqual(worker.kv_flag[0], 0)
            worker.kv_flag[1] = 1
            self.reply(self.status.FINISH, "long", self.result.SUCCESS)
            await asyncio.wait_for(task, 2)
            self.assertEqual(await worker.fetch_finished_recving_reqs(), ({"long"}, set()))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_failure_and_timeout_remain_distinct(self):
        self.reply(self.status.CONTINUE, "short", self.result.XFER_FAIL)
        self.reply(self.status.FINISH, "long", self.result.TIMEOUT)
        await self.worker.receive_kv_from_single_worker("P", "", self.pulls)
        self.assertEqual(
            await self.worker.fetch_finished_recving_reqs(), ({"short"}, {"short"})
        )
        self.assertEqual(
            self.worker.timeout_reqs_to_recv, {"P": {"long": self.pulls["long"]}}
        )

    async def test_duplicate_reply_does_not_reprocess_released_blocks(self):
        self.worker.kv_flag[0] = 1
        self.reply(self.status.CONTINUE, "short", self.result.SUCCESS)
        self.reply(self.status.FINISH, "short", self.result.SUCCESS)
        with self.assertLogs(self.rx.logger, "ERROR") as logs:
            await self.worker.receive_kv_from_single_worker("P", "", self.pulls)
        self.assertIn("repeated Mooncake request IDs", logs.output[-1])
        self.assertEqual(self.pulls["short"].pull_tasks_count, 0)
        self.assertEqual(self.worker.finished_recving_reqs, {"short"})

    async def test_incomplete_finish_does_not_publish(self):
        self.reply(self.status.FINISH, "short", self.result.SUCCESS)
        with self.assertLogs(self.rx.logger, "ERROR") as logs:
            await self.worker.receive_kv_from_single_worker("P", "", self.pulls)
        self.assertIn("FINISH missing requests", logs.output[-1])
        self.assertFalse(self.worker.finished_recving_reqs)
