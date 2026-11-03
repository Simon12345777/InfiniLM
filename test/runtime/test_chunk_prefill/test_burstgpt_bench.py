#!/usr/bin/env python3
"""Compare whole-prompt and chunked prefill on the same single-server or Mooncake PD workload."""

from __future__ import annotations

import argparse
from burstgpt_workload import add_burstgpt_args, sample_burstgpt
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
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_SERVER = PROJECT_ROOT / "python/infinilm/server/inference_server.py"
BENCH_SCRIPT = SCRIPT_DIR / "pure_bench_serve.py"
PROXY_SERVER = INFERENCE_SERVER.with_name("mooncake_proxy_server.py")
OVERLAP_MODES = ("pd_overlap", "pd_overlap_limited")
MODES = ("baseline", "chunk", "pd", "pd_chunk", *OVERLAP_MODES)


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
    p.add_argument("--device", default="nvidia")
    p.add_argument("--backend", choices=["cpp"], default="cpp")
    p.add_argument(
        "--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    p.add_argument("--prefill-gpu", help="One physical CUDA device for the P process.")
    p.add_argument("--decode-gpu", help="One physical CUDA device for the D process.")
    p.add_argument("--prefill-port", type=int, default=8100)
    p.add_argument("--decode-port", type=int, default=8200)
    p.add_argument("--bootstrap-port", type=int, default=8998)
    p.add_argument("--mooncake-protocol", choices=("tcp", "rdma"), default="tcp")
    p.add_argument("--max-inflight-prefix-writes", type=int, default=1,
                   help="Per-producer cap in pd_overlap_limited (default: 1).")
    p.add_argument("--attn", choices=("default", "paged-attn", "flash-attn"), default="default")
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
        default=False,
        help="Decode graph setting, held constant across all modes.",
    )
    p.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--chunked-prefill-size", "--prefill-chunk-size", "--chunk-size",
        dest="prefill_chunk_size", type=int, default=256,
    )
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument(
        "--modes",
        choices=["all", "pd_pair", "overlap_pair", "prefix_limit_pair", "local_pair", *MODES],
        default="pd_pair",
        help="overlap_pair compares pd_chunk vs pd_overlap; pd_pair compares pd vs pd_chunk.",
    )
    p.add_argument(
        "--workload", choices=["uniform", "mixed", "burstgpt"], default=default_workload
    )
    p.add_argument(
        "--dataset-path",
        default=str(SCRIPT_DIR / "datasets/BurstGPT/BurstGPT_1000.csv"),
    )
    p.add_argument("--use-full-dataset", action="store_true")
    add_burstgpt_args(p)
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
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--server-timeout", type=float, default=900)
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--result-dir", default=str(SCRIPT_DIR / "bench_results"))
    p.add_argument("--save-detailed", action="store_true")
    p.add_argument(
        "--save-result-files",
        action="store_true",
        help="Compatibility: results are always saved.",
    )
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
    modes = benchmark_modes(args)
    if args.max_inflight_prefix_writes < 1:
        p.error("--max-inflight-prefix-writes must be positive for the limited arm")
    if any(mode in OVERLAP_MODES for mode in modes) and args.enable_prefix_caching:
        p.error("pd_overlap requires prefix caching to be disabled")
    if any(mode.startswith("pd") for mode in modes):
        ports = [args.port, args.prefill_port, args.decode_port, args.bootstrap_port]
        visible = (args.cuda_visible_devices or "0,1").split(",")
        args.prefill_gpu = args.prefill_gpu or visible[0].strip()
        args.decode_gpu = args.decode_gpu or (visible[1].strip() if len(visible) > 1 else None)
        if not args.decode_gpu or args.decode_gpu == args.prefill_gpu or any(
            not gpu or "," in gpu for gpu in (args.prefill_gpu, args.decode_gpu)
        ):
            p.error("PD requires two distinct devices; set --prefill-gpu and --decode-gpu")
    else:
        ports = [args.port]
    if len(set(ports)) != len(ports) or any(not 0 < port < 65536 for port in ports):
        p.error("all service ports must be distinct and in 1..65535")
    for name in ("server_timeout", "request_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be positive/finite")
    if not args.enable_paged_attn:
        p.error("These chunk prefill comparisons require paged attention")
    if args.max_num_batched_tokens < args.prefill_chunk_size:
        p.error("batch token budget must be >= prefill chunk size")
    if args.max_num_batched_tokens < 1024 or args.prefill_chunk_size % args.block_size:
        p.error("batch budget must be >= 1024 and chunk size must be a multiple of block size")
    if args.workload != "burstgpt" and args.long_input_tokens <= args.prefill_chunk_size:
        p.error("long input must contain a full chunk")
    if args.workload == "mixed" and (
        args.short_input_tokens >= args.prefill_chunk_size or args.num_prompts < 2
    ):
        p.error(
            "mixed workload needs short input < chunk size and at least two requests"
        )
    if args.workload == 'burstgpt':
        # Fail before launching P/D if the dataset cannot satisfy the request
        # count or server output limit. Use the same selector as the client.
        try:
            lengths, report = sample_burstgpt(args)
            warmup_lengths, _ = sample_burstgpt(args, warmup=True)
        except (ValueError, OSError) as exc:
            p.error(str(exc))
        if max(b for _, b in lengths + warmup_lengths) > args.server_max_new_tokens:
            p.error('BurstGPT output budget exceeds --server-max-new-tokens; increase it or use cap policy')
        print(f"BurstGPT preflight: {report['eligible_rows']} eligible rows, "
              f"{report['selected_unique_rows']} unique selected, "
              f"{report['selected_long_rows']} long, "
              f"{report['selected_output_capped']} capped outputs")
        for warning in report['warnings']:
            print(f'WARNING: {warning}')
    return args


def benchmark_modes(args):
    if args.modes == "prefix_limit_pair":
        return list(OVERLAP_MODES)
    if args.modes == "overlap_pair":
        return ["pd_chunk", "pd_overlap"]
    if args.modes == "pd_pair":
        return ["pd", "pd_chunk"]
    if args.modes == "local_pair":
        return ["baseline", "chunk"]
    return ["baseline", "chunk", "pd", "pd_chunk"] if args.modes == "all" else [args.modes]


def build_server_command(args, mode, role):
    ports = {"local": args.port, "prefill": args.prefill_port, "decode": args.decode_port}
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
        str(ports[role]),
        "--dtype", "bfloat16", "--tp", "1", "--attn", args.attn,
        "--enable-paged-attn",
    ]
    for flag, value in (
        ("max-batch-size", args.max_batch_size),
        ("max-new-tokens", args.server_max_new_tokens),
        ("num-blocks", args.num_blocks),
        ("block-size", args.block_size),
        ("max-cache-len", args.max_cache_len),
    ):
        cmd.extend([f"--{flag}", str(value)])
    if args.enable_graph:
        cmd.append("--enable-graph")
    if not args.enable_prefix_caching:
        cmd.append("--disable-prefix-caching")
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if mode in ("chunk", "pd_chunk", *OVERLAP_MODES) and role != "decode":
        cmd.extend(["--chunked-prefill-size", str(args.prefill_chunk_size)])
    if role != "local":
        config = {
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "kv_connector_extra_config": {"mooncake_protocol": args.mooncake_protocol},
        }
        if mode in OVERLAP_MODES and role == "prefill":
            config["kv_connector_extra_config"]["chunk_transfer_overlap"] = True
            if mode == "pd_overlap_limited":
                config["kv_connector_extra_config"]["max_inflight_prefix_writes"] = args.max_inflight_prefix_writes
        cmd.extend(["--kv-transfer-config", json.dumps(config)])
    return cmd


def services(args, mode):
    budget = {"INFINILM_MAX_NUM_BATCHED_TOKENS": str(args.max_num_batched_tokens)}
    if not mode.startswith("pd"):
        gpu = args.prefill_gpu or (args.cuda_visible_devices or "").split(",")[0].strip()
        env = {**budget, "CUDA_VISIBLE_DEVICES": gpu} if gpu else budget
        yield "local", args.port, build_server_command(args, mode, "local"), env
        return
    for role, port, gpu in (("prefill", args.prefill_port, args.prefill_gpu),
                            ("decode", args.decode_port, args.decode_gpu)):
        env = {**budget, "CUDA_VISIBLE_DEVICES": gpu}
        if role == "prefill":
            env.update(INFINILM_MOONCAKE_BOOTSTRAP_HOST=args.client_host,
                       INFINILM_MOONCAKE_BOOTSTRAP_PORT=str(args.bootstrap_port))
        yield role, port, build_server_command(args, mode, role), env
    proxy = [sys.executable, "-u", str(PROXY_SERVER), "--host", args.server_host,
             "--port", str(args.port),
             "--prefill", f"http://{args.client_host}:{args.prefill_port}", str(args.bootstrap_port),
             "--decode", f"http://{args.client_host}:{args.decode_port}"]
    yield "proxy", args.port, proxy, {}


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
        "chunked-prefill-size": args.prefill_chunk_size,
        "max-request-tokens": args.max_request_tokens,
        "max-response-tokens": args.max_response_tokens,
        "burstgpt-model": getattr(args, 'burstgpt_model', 'GPT-4'),
        "burstgpt-output-policy": getattr(args, 'burstgpt_output_policy', 'cap'),
        "temperature": args.temperature,
        "warmup-requests": args.warmup_requests,
        "request-timeout": args.request_timeout,
        "result-dir": result_path.parent,
        "result-filename": result_path.name,
    }
    for flag, value in values.items():
        cmd.extend([f"--{flag}", str(value)])
    for flag in ("ignore-eos", "save-detailed", "use-full-dataset", "allow-repeat-samples"):
        if getattr(args, flag.replace("-", "_"), False):
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


def wait_ready(args, processes, role, port):
    deadline = time.monotonic() + args.server_timeout
    base = f"http://{args.client_host}:{port}"
    while time.monotonic() < deadline:
        for name, proc in processes.items():
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode}; see its log")
        try:
            if role == "proxy":
                # Before handshake the proxy returns 503. After it, invalid JSON
                # returns 400 before a request is dispatched to either GPU.
                probe = urllib.request.Request(base + "/v1/chat/completions", data=b"{")
                with urllib.request.urlopen(probe, timeout=2):
                    pass
            else:
                with urllib.request.urlopen(base + "/health", timeout=2):
                    pass
                with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
                    ids = [m["id"] for m in json.load(response)["data"]]
                if args.served_model_name not in ids:
                    raise ValueError(f"requested model {args.served_model_name!r}, available {ids}")
                if role == "prefill":
                    url = f"http://{args.client_host}:{args.bootstrap_port}/query"
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if not json.load(response).get("0", {}).get("engine_id"):
                            time.sleep(0.5)
                            continue
                return
        except urllib.error.HTTPError as exc:
            if role == "proxy" and exc.code == 400:
                return
            if exc.code != 503:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{role} did not become ready at {base}")


def verify_execution(args, mode, result, logs):
    chunk_log = logs.get("prefill", logs.get("local", ""))
    resumes = chunk_log.count("Scheduling final prefill chunk for request ")
    result["chunked_requests_including_warmup"] = resumes
    if mode in ("chunk", "pd_chunk", *OVERLAP_MODES):
        size = args.prefill_chunk_size
        warmup_long = sum(n > size for n in result["warmup_input_lens"])
        measured_long = sum(n > size for n in result["input_lens"])
        if resumes <= warmup_long or "chunked_prefill_size disabled" in chunk_log:
            raise RuntimeError("could not verify chunking beyond warmup; inspect the prefill log and prompt lengths")
        if not args.enable_prefix_caching and resumes != warmup_long + measured_long:
            raise RuntimeError("resumed prompt count disagrees with server-reported input lengths")
    elif resumes:
        raise RuntimeError("unexpected chunked prefill in the baseline")
    if "decode" in logs:
        received = logs["decode"].count("finished receiving KV, scheduling for decode")
        result["pd_handoffs_including_warmup"] = received
        expected = result["completed"] + args.warmup_requests
        if "failed receiving KV" in logs["decode"] or received != expected:
            raise RuntimeError(f"PD handoff count {received} or KV fallback invalidates this run")


def run_mode(args, mode, result_path):
    specs = list(services(args, mode))
    ports = [port for _, port, _, _ in specs]
    if mode.startswith("pd"):
        ports.append(args.bootstrap_port)
    for port in ports:
        if port_busy(args.client_host, port):
            raise RuntimeError(f"port {port} is already in use")
    paths = {}
    processes = {}
    bench_cmd = build_bench_command(args, result_path)
    started = time.perf_counter()
    with ExitStack() as stack:
        for role, port, cmd, env in specs:
            paths[role] = result_path.with_suffix(f".{role}.log")
            log = stack.enter_context(paths[role].open("w", encoding="utf-8"))
            print(f"[{mode}/{role}] {paths[role]}", flush=True)
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env={**os.environ, **env},
                                    stdout=log, stderr=subprocess.STDOUT, **process_options())
            stack.callback(terminate_process_group, proc)
            processes[role] = proc
            wait_ready(args, processes, role, port)
        startup_s = time.perf_counter() - started
        client = subprocess.Popen(bench_cmd, cwd=PROJECT_ROOT, **process_options())
        try:
            while client.poll() is None:
                for role, proc in processes.items():
                    if proc.poll() is not None:
                        raise RuntimeError(f"{role} exited during benchmark; see {paths[role]}")
                time.sleep(0.2)
        finally:
            if client.poll() is None:
                terminate_process_group(client)
        if client.returncode:
            raise RuntimeError(f"benchmark failed; inspect {result_path} and service logs")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["failed"] or result["completed"] != args.num_prompts:
        raise RuntimeError("incomplete workload; refusing a performance comparison")
    logs = {role: path.read_text(encoding="utf-8", errors="replace") for role, path in paths.items()}
    verify_execution(args, mode, result, logs)
    result.update(mode=mode, startup_s=startup_s, benchmark_command=bench_cmd,
                  services={role: {"command": cmd, "env": env, "log": str(paths[role])}
                            for role, _, cmd, env in specs})
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


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
    if len({tuple(r["input_lens"]) for r in results}) != 1:
        raise RuntimeError("server-reported prompt lengths differ across runs")
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
        ("pd", "pd_chunk"),
        ("pd_chunk", "pd_overlap"),
        OVERLAP_MODES,
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
            for role, _, cmd, env in services(args, mode):
                print(f"[{mode}/{role}] {shlex.join([f'{k}={v}' for k,v in env.items()] + cmd)}")
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
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
