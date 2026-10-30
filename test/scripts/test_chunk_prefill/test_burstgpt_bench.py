#!/usr/bin/env python3
"""Compare baseline, chunk prefill, and chunk prefill + graph on identical loads."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_SERVER = PROJECT_ROOT / "python/infinilm/server/inference_server.py"
BENCH_SCRIPT = SCRIPT_DIR / "pure_bench_serve.py"
MODES = ("baseline", "chunk", "chunk_graph")


def parse_rate(value):
    rate = float(value)
    if math.isnan(rate) or rate <= 0:
        raise argparse.ArgumentTypeError("must be positive or inf")
    return rate


def parse_args(argv=None, default_workload="burstgpt"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--served-model-name", help="Defaults to the model directory name.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=["cpp"], default="cpp")
    p.add_argument(
        "--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--client-host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2333)
    p.add_argument("--max-batch-size", "--batch-size", type=int, default=16)
    p.add_argument("--server-max-new-tokens", "--max-new-tokens", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=512)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-cache-len", type=int, default=32768)
    p.add_argument(
        "--enable-paged-attn", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--enable-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode graph setting, held constant across all modes.",
    )
    p.add_argument(
        "--enable-chunk-prefill-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the separate chunk_graph comparison mode.",
    )
    p.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefill-chunk-size", "--chunk-size", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument(
        "--modes",
        choices=["all", *MODES, "both", "off", "on"],
        default="all",
        help="all: three modes; off/on/both retained as baseline/chunk-graph aliases.",
    )
    p.add_argument(
        "--workload", choices=["uniform", "mixed", "burstgpt"], default=default_workload
    )
    p.add_argument(
        "--dataset-path",
        default=str(SCRIPT_DIR / "datasets/BurstGPT/BurstGPT_1000.csv"),
    )
    p.add_argument("--use-full-dataset", action="store_true")
    p.add_argument("--max-request-tokens", type=int, default=8192)
    p.add_argument("--max-response-tokens", type=int, default=256)
    p.add_argument("--long-input-tokens", type=int, default=4096)
    p.add_argument("--short-input-tokens", type=int, default=128)
    p.add_argument("--long-ratio", type=float, default=0.5)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--request-rate", type=parse_rate, default=float("inf"))
    p.add_argument("--burstiness", type=parse_rate, default=1.0)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-requests", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--server-timeout", type=float, default=900)
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--result-dir", default=str(SCRIPT_DIR / "bench_results"))
    p.add_argument("--save-detailed", action="store_true")
    p.add_argument(
        "--save-result-files",
        action="store_true",
        help="Compatibility: results are always saved.",
    )
    p.add_argument("--show-server-output", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting a server.",
    )
    args = p.parse_args(argv)
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.tokenizer = (
        str(Path(args.tokenizer).expanduser().resolve())
        if args.tokenizer
        else args.model_path
    )
    args.dataset_path = str(Path(args.dataset_path).expanduser().resolve())
    args.result_dir = str(Path(args.result_dir).expanduser().resolve())
    args.served_model_name = args.served_model_name or Path(args.model_path).name
    for name in (
        "max_batch_size",
        "server_max_new_tokens",
        "num_blocks",
        "block_size",
        "max_cache_len",
        "prefill_chunk_size",
        "max_num_batched_tokens",
        "long_input_tokens",
        "short_input_tokens",
        "output_tokens",
        "num_prompts",
        "max_concurrency",
        "repeats",
        "max_request_tokens",
        "max_response_tokens",
    ):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_requests < 0 or not 0 < args.long_ratio < 1:
        p.error("warmup must be nonnegative; long-ratio must be in (0, 1)")
    if not 0 < args.port < 65536:
        p.error("invalid port")
    for name in ("server_timeout", "request_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be positive/finite")
    if not args.enable_paged_attn:
        p.error("These chunk prefill comparisons require paged attention")
    if args.max_num_batched_tokens < args.prefill_chunk_size:
        p.error("batch token budget must be >= prefill chunk size")
    if "chunk_graph" in benchmark_modes(args) and args.prefill_chunk_size <= 1:
        p.error("chunk graph recording requires chunk size > 1")
    if args.workload != "burstgpt" and args.long_input_tokens < args.prefill_chunk_size:
        p.error("long input must contain a full chunk")
    if args.workload == "mixed" and (
        args.short_input_tokens >= args.prefill_chunk_size or args.num_prompts < 2
    ):
        p.error(
            "mixed workload needs short input < chunk size and at least two requests"
        )
    return args


def benchmark_modes(args):
    on = "chunk_graph" if args.enable_chunk_prefill_graph else "chunk"
    if args.modes in ("off", "baseline"):
        return ["baseline"]
    if args.modes == "on":
        return [on]
    if args.modes == "both":
        return ["baseline", on]
    if args.modes == "all":
        return list(MODES) if args.enable_chunk_prefill_graph else ["baseline", "chunk"]
    if args.modes == "chunk_graph" and not args.enable_chunk_prefill_graph:
        raise ValueError("chunk_graph conflicts with --no-enable-chunk-prefill-graph")
    return [args.modes]


def build_server_command(args, mode):
    cmd = [
        sys.executable,
        "-u",
        str(INFERENCE_SERVER),
        "--device",
        args.device,
        "--model",
        args.model_path,
        "--backend",
        args.backend,
        "--host",
        args.server_host,
        "--port",
        str(args.port),
        "--enable-paged-attn",
    ]
    for flag, value in (
        ("max-batch-size", args.max_batch_size),
        ("max-new-tokens", args.server_max_new_tokens),
        ("num-blocks", args.num_blocks),
        ("block-size", args.block_size),
        ("max-cache-len", args.max_cache_len),
        ("max-num-batched-tokens", args.max_num_batched_tokens),
    ):
        cmd.extend([f"--{flag}", str(value)])
    if args.enable_graph:
        cmd.append("--enable-graph")
    if not args.enable_prefix_caching:
        cmd.append("--disable-prefix-caching")
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if mode != "baseline":
        cmd.extend(
            [
                "--enable-chunked-prefill",
                "--prefill-chunk-size",
                str(args.prefill_chunk_size),
            ]
        )
    if mode == "chunk_graph":
        cmd.append("--enable-chunk-prefill-graph")
    return cmd


def build_bench_command(args, result_path):
    cmd = [sys.executable, "-u", str(BENCH_SCRIPT)]
    values = {
        "base-url": f"http://{args.client_host}:{args.port}",
        "model": args.served_model_name,
        "tokenizer": args.tokenizer,
        "dataset-name": args.workload,
        "dataset-path": args.dataset_path,
        "num-prompts": args.num_prompts,
        "max-concurrency": args.max_concurrency,
        "request-rate": args.request_rate,
        "burstiness": args.burstiness,
        "seed": args.seed,
        "long-input-tokens": args.long_input_tokens,
        "short-input-tokens": args.short_input_tokens,
        "long-ratio": args.long_ratio,
        "output-tokens": args.output_tokens,
        "prefill-chunk-size": args.prefill_chunk_size,
        "max-request-tokens": args.max_request_tokens,
        "max-response-tokens": args.max_response_tokens,
        "temperature": args.temperature,
        "warmup-requests": args.warmup_requests,
        "request-timeout": args.request_timeout,
        "result-dir": result_path.parent,
        "result-filename": result_path.name,
    }
    for flag, value in values.items():
        cmd.extend([f"--{flag}", str(value)])
    for flag in ("ignore-eos", "save-detailed", "use-full-dataset"):
        if getattr(args, flag.replace("-", "_")):
            cmd.append(f"--{flag}")
    return cmd


def process_options():
    return (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )


def terminate_process_group(proc):
    if proc is None:
        return
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    else:
        # The process group may outlive its leader (e.g. failed model startup).
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def port_busy(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_server(args, proc):
    deadline = time.monotonic() + args.server_timeout
    base = f"http://{args.client_host}:{args.port}"
    next_notice = 0.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                if response.status == 200:
                    with urllib.request.urlopen(
                        base + "/v1/models", timeout=2
                    ) as model_response:
                        ids = [m["id"] for m in json.load(model_response)["data"]]
                    if args.served_model_name not in ids:
                        raise ValueError(
                            f"model id {args.served_model_name!r} is not served: {ids}"
                        )
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if time.monotonic() >= next_notice:
            print("Waiting for model loading / graph recording...", flush=True)
            next_notice = time.monotonic() + 15
        time.sleep(1)
    raise TimeoutError("server startup timeout")


def run_mode(args, mode, result_path):
    if port_busy(args.client_host, args.port):
        raise RuntimeError("benchmark port is already in use; select an unused --port")
    log_path = result_path.with_suffix(".server.log")
    server_cmd = build_server_command(args, mode)
    bench_cmd = build_bench_command(args, result_path)
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print(f"\nMode={mode}\n{shlex.join(server_cmd)}", flush=True)
    server = client = None
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            server = subprocess.Popen(
                server_cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=None if args.show_server_output else log,
                stderr=subprocess.STDOUT,
                **process_options(),
            )
            wait_for_server(args, server)
            startup_s = time.perf_counter() - started
            client = subprocess.Popen(bench_cmd, cwd=PROJECT_ROOT, **process_options())
            if client.wait() != 0:
                raise RuntimeError(
                    f"benchmark failed; inspect {result_path} and {log_path}"
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result.update(
                mode=mode,
                startup_s=startup_s,
                server_command=server_cmd,
                benchmark_command=bench_cmd,
                server_log=str(log_path),
            )
            result_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if result["failed"] or result["completed"] != args.num_prompts:
                raise RuntimeError(
                    "incomplete workload; refusing a performance comparison"
                )
            return result
        except BaseException:
            if not args.show_server_output:
                log.flush()
                print(
                    "\nServer log tail:\n"
                    + "\n".join(
                        log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()[-40:]
                    )
                )
            raise
        finally:
            if client is not None and client.poll() is None:
                terminate_process_group(client)
            terminate_process_group(server)
            deadline = time.monotonic() + 10
            while (
                port_busy(args.client_host, args.port) and time.monotonic() < deadline
            ):
                time.sleep(0.2)
            if port_busy(args.client_host, args.port):
                raise RuntimeError("server port did not close after cleanup")


METRICS = (
    ("request_throughput", False),
    ("input_throughput", False),
    ("output_throughput", False),
    ("mean_ttft_ms", True),
    ("p95_ttft_ms", True),
    ("p99_ttft_ms", True),
    ("mean_tpot_ms", True),
    ("p99_tpot_ms", True),
    ("mean_e2el_ms", True),
    ("mean_offered_ttft_ms", True),
    ("mean_client_queue_ms", True),
)


def speedup_pct(new, old, lower):
    if new is None or old is None or old == 0:
        return None
    return (old - new if lower else new - old) / old * 100


def summarize(results):
    # Never compare different requests or silently shorter outputs as a speedup.
    if len({r["workload_sha256"] for r in results}) != 1:
        raise RuntimeError("workloads differ across runs")
    if len({tuple(r["output_lens"]) for r in results}) != 1:
        raise RuntimeError("output token counts differ across runs; comparison invalid")
    summary = {}
    for mode in MODES:
        runs = [r for r in results if r["mode"] == mode]
        if not runs:
            continue
        summary[mode] = {}
        for group in (None, "short", "long"):
            for key, _ in METRICS:
                values = [
                    (r if group is None else r["groups"][group]).get(key) for r in runs
                ]
                values = [v for v in values if v is not None]
                name = key if group is None else f"{group}.{key}"
                summary[mode][name] = {
                    "median": statistics.median(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
    comparisons = {}
    for old, new in (
        ("baseline", "chunk"),
        ("chunk", "chunk_graph"),
        ("baseline", "chunk_graph"),
    ):
        if old not in summary or new not in summary:
            continue
        label = f"{new} vs {old}"
        comparisons[label] = {}
        print(f"\n{label}: median across runs; positive = better, negative = worse")
        for group in (None, "short", "long"):
            for key, lower in METRICS:
                name = key if group is None else f"{group}.{key}"
                a, b = summary[old][name]["median"], summary[new][name]["median"]
                improvement = speedup_pct(b, a, lower)
                comparisons[label][name] = improvement
                if group is None or key in ("mean_ttft_ms", "p99_ttft_ms"):

                    def fmt(v):
                        return "N/A" if v is None else f"{v:.3f}"

                    pct = "N/A" if improvement is None else f"{improvement:+.2f}%"
                    print(f"  {name:28s} {fmt(a):>12s} -> {fmt(b):>12s}  {pct}")
    return {"metrics": summary, "improvement_percent": comparisons}


def main(argv=None, default_workload="burstgpt"):
    args = parse_args(argv, default_workload)
    modes = benchmark_modes(args)
    run_dir = Path(args.result_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.dry_run:
        for mode in modes:
            print(f"\n[{mode}]\n{shlex.join(build_server_command(args, mode))}")
            print(shlex.join(build_bench_command(args, run_dir / f"{mode}.json")))
        return
    run_dir.mkdir(parents=True)
    config = {
        k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
        for k, v in vars(args).items()
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"Results: {run_dir}\nDecode graph={args.enable_graph}; prefix caching={args.enable_prefix_caching}"
    )
    results = []
    for repeat in range(args.repeats):
        # Rotate startup order to reduce a fixed warm/cold ordering bias.
        offset = repeat % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            result = run_mode(args, mode, run_dir / f"r{repeat + 1}_{mode}.json")
            result["repeat"] = repeat + 1
            results.append(result)
    summary = summarize(results)
    summary["runs"] = results
    summary["graph_replay_verified"] = False
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        "\nGraph mode is an end-to-end flag comparison, not proof of device graph replay."
    )
    print(
        "Current prefill-first scheduling can delay decode under sustained prompt traffic."
    )
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
