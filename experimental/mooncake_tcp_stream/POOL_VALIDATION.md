# Resource-pool validation — 2026-11-10

Implemented a per-device, mutex-protected idle cache of exclusively leased
pinned-buffer/private-stream pairs. `MC_TCP_CUDA_POOL_SLOTS=0` is the default;
32 enables the experiment. Idle retention is bounded by slots and 1 MiB per
buffer. Active transfers retain their original allocation fallback and are not
blocked by an empty pool. Session destruction returns only healthy resources,
after Asio shared owners and synchronous CUDA completion have finished use.

The prior experimental engine is backed up at
`/workspace/mooncake-tcp-stream-experiment/before-pool-binary/`.
No installed wheel or InfiniLM native library was replaced.

Validation completed:

- Native concurrent test: reuse, distinct simultaneous leases, one-slot
  overflow cleanup, two GPUs and 64 threaded D2H/H2D round trips passed.
- Actual TCP READ/WRITE data and tail checks passed with pool 32 and pool 0.
- Two independent eight-request PD runs (pool 32) completed. All 16 generated
  responses and usage values matched the earlier unpooled run. Recorded CUDA
  statuses were successful.
- P-side closed-session summaries in **each** PD run: 4 host allocations,
  4 stream creations, 1532 pool hits, 1536 returns; no host frees/stream destroys
  during those recorded session lifetimes. Earlier unpooled runs allocated
  1280/1536 times. Pinned resources retained in the idle pool are reclaimed at
  process exit, not freed per request.

Limitations: these are small diagnostic runs without benchmark warmup. Initial
allocation/stream creation can still block. One D-side cudaMemcpyAsync call in
run 2 took 442 ms, so pooling does not prove the entire transfer bottleneck is
resolved. D-side lifetime summaries are incomplete due to persistent sessions
at shutdown. No injected network-cancellation/fatal-CUDA-error testing was done.

Logs: `/workspace/mooncake-tcp-stream-experiment/pd-pool-r1/` and `pd-pool-r2/`,
including `cuda-timing.json`, P/D logs and request outputs.

Next full benchmark, explicitly run by the user inside the container:

```bash
cd /workspace/InfiniLM
bash experimental/mooncake_tcp_stream/run_pool_pair.sh
```

The script keeps private streams enabled, compares pool slots 0 and 32 using
the same binary, disables detailed CUDA logging, and runs two repeats per
configuration. It does not compare against the original synchronous-copy path;
that remains a separate baseline before choosing a deployment default.
