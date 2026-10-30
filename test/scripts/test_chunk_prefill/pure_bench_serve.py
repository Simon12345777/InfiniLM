#!/usr/bin/env python3
"""Streaming benchmark for InfiniLM; event gaps are not kernel/token timings."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import http.client
import json
import math
import os
import random
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SampleRequest:
    prompt: str
    prompt_len: int
    expected_output_len: int


@dataclass
class RequestOutput:
    success: bool = False
    generated_text: str = ""
    output_tokens: int | None = None
    token_count_source: str = ""
    ttft: float | None = None
    latency: float = 0.0
    last_content_time: float | None = None
    itl: list[float] = field(default_factory=list)
    client_queue_s: float = 0.0
    finish_reason: str | None = None
    error: str = ""


def parse_rate(value):
    rate = float(value)
    if math.isnan(rate) or rate <= 0:
        raise argparse.ArgumentTypeError("rate must be positive or inf")
    return rate


def parse_args(argv=None, default_dataset="burstgpt"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--endpoint", default="/v1/chat/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer")
    p.add_argument(
        "--dataset-name",
        choices=["burstgpt", "uniform", "mixed"],
        default=default_dataset,
    )
    p.add_argument("--dataset-path")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--request-rate", type=parse_rate, default=float("inf"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--burstiness", type=parse_rate, default=1.0)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--long-input-tokens", type=int, default=4096)
    p.add_argument("--short-input-tokens", type=int, default=128)
    p.add_argument("--long-ratio", type=float, default=0.5)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=256)
    p.add_argument("--max-request-tokens", type=int, default=8192)
    p.add_argument("--max-response-tokens", type=int, default=256)
    p.add_argument("--use-full-dataset", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Require length finish; server must also use --ignore-eos.",
    )
    p.add_argument("--warmup-requests", type=int, default=8)
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--metric-percentiles", default="50,95,99")
    p.add_argument("--save-result", action="store_true")
    p.add_argument("--save-detailed", action="store_true")
    p.add_argument("--result-dir")
    p.add_argument("--result-filename")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--backend", choices=["openai-chat"], default="openai-chat")
    p.add_argument("--disable-tqdm", action="store_true")
    args = p.parse_args(argv)
    for name in (
        "num_prompts",
        "max_concurrency",
        "long_input_tokens",
        "short_input_tokens",
        "output_tokens",
        "prefill_chunk_size",
        "max_request_tokens",
        "max_response_tokens",
    ):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.warmup_requests < 0
        or not math.isfinite(args.request_timeout)
        or args.request_timeout <= 0
    ):
        p.error("warmup must be nonnegative and timeout positive/finite")
    if not 0 < args.long_ratio < 1:
        p.error("--long-ratio must be between 0 and 1")
    if args.dataset_name == "burstgpt" and not args.dataset_path:
        p.error("--dataset-path is required for BurstGPT")
    if args.dataset_name == "mixed" and args.num_prompts < 2:
        p.error("mixed workload needs at least two requests")
    if bool(args.result_dir) != bool(args.result_filename):
        p.error("--result-dir and --result-filename must be supplied together")
    return args


def chat_length(tokenizer, prompt):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer(rendered, add_special_tokens=False).input_ids)


def make_prompt(tokenizer, target, nonce):
    prefix = f"Request {nonce}. Continue this text:"
    repeats = max(1, target - chat_length(tokenizer, prefix))
    prompt = prefix + " hello" * repeats
    # Target is approximate; always report the locally tokenized chat length.
    for _ in range(4):
        adjusted = max(1, repeats + target - chat_length(tokenizer, prompt))
        if adjusted == repeats:
            break
        repeats = adjusted
        prompt = prefix + " hello" * repeats
    return prompt, chat_length(tokenizer, prompt)


def sample_requests(args, tokenizer, *, warmup=False):
    seed = args.seed + (1_000_003 if warmup else 0)
    rng = random.Random(seed)
    n = args.warmup_requests if warmup else args.num_prompts
    if not n:
        return []
    if args.dataset_name == "burstgpt":
        with open(args.dataset_path, newline="", encoding="utf-8") as f:
            rows = [
                (int(r["Request tokens"]), int(r["Response tokens"]))
                for r in csv.DictReader(f)
                if r["Model"] == "GPT-4"
                and int(r["Request tokens"]) > 0
                and int(r["Response tokens"]) > 0
            ]
        if not args.use_full_dataset:
            rows = [
                (a, b)
                for a, b in rows
                if a <= args.max_request_tokens and b <= args.max_response_tokens
            ]
        if not rows:
            raise ValueError("No valid BurstGPT rows after filtering")
        lengths = rng.sample(rows, n) if n <= len(rows) else rng.choices(rows, k=n)
    else:
        long_count = (
            n
            if args.dataset_name == "uniform"
            else max(1, min(n - 1, round(n * args.long_ratio)))
        )
        lengths = [(args.long_input_tokens, args.output_tokens)] * long_count
        lengths += [(args.short_input_tokens, args.output_tokens)] * (n - long_count)
        rng.shuffle(lengths)
    requests = []
    for i, (input_len, output_len) in enumerate(lengths):
        prompt, actual_len = make_prompt(tokenizer, input_len, f"{seed:08x}-{i:06x}")
        if args.dataset_name != "burstgpt":
            intended_long = input_len == args.long_input_tokens
            if (actual_len >= args.prefill_chunk_size) != intended_long:
                raise ValueError(
                    "Synthetic prompt is in wrong chunk class; adjust input lengths"
                )
        requests.append(SampleRequest(prompt, actual_len, output_len))
    return requests


def build_delays(n, request_rate, burstiness, seed=0):
    if math.isinf(request_rate):
        return [0.0] * n
    rng = np.random.default_rng(seed)
    gaps = (
        np.full(max(0, n - 1), 1 / request_rate)
        if math.isinf(burstiness)
        else rng.gamma(burstiness, 1 / (request_rate * burstiness), max(0, n - 1))
    )
    return [0.0, *np.cumsum(gaps).tolist()][:n]


def _do_request_blocking(
    base_url, endpoint, model, req, temperature, ignore_eos, timeout, scheduled_at=None
):
    out = RequestOutput()
    parsed = urllib.parse.urlparse(base_url)
    path = parsed.path.rstrip("/") + "/" + endpoint.lstrip("/")
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": temperature,
        "max_tokens": req.expected_output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    start = time.perf_counter()
    out.client_queue_s = (
        max(0.0, start - scheduled_at) if scheduled_at is not None else 0.0
    )
    previous_content = None
    done = False
    try:
        conn.request("POST", path, body=json.dumps(payload).encode(), headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {response.read(2048).decode(errors='replace')}"
            )
        for raw in response:
            if time.perf_counter() - start > timeout:
                raise TimeoutError("request deadline exceeded")
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                out.output_tokens = int(usage["completion_tokens"])
                out.token_count_source = "server_usage"
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            reason = choice.get("finish_reason")
            if reason is not None:
                out.finish_reason = reason
                if reason not in ("stop", "length"):
                    raise RuntimeError(f"unsuccessful finish_reason: {reason}")
            delta = choice.get("delta") or {}
            content = (delta.get("content") or "") + (
                delta.get("reasoning_content") or ""
            )
            if content:
                now = time.perf_counter()
                if out.ttft is None:
                    out.ttft = now - start
                if previous_content is not None:
                    out.itl.append(now - previous_content)
                previous_content = now
                out.last_content_time = now - start
                out.generated_text += content
        out.latency = time.perf_counter() - start
        if not done or out.finish_reason is None:
            raise RuntimeError("truncated stream: missing finish_reason or [DONE]")
        if ignore_eos and out.finish_reason != "length":
            raise RuntimeError("expected length finish; start server with --ignore-eos")
        if out.output_tokens is None and out.finish_reason == "length":
            out.output_tokens = req.expected_output_len
            out.token_count_source = "max_tokens_at_length_finish"
        if ignore_eos and out.output_tokens != req.expected_output_len:
            raise RuntimeError("output length differs from requested max_tokens")
        out.success = True
    except Exception as exc:
        out.latency = time.perf_counter() - start
        out.error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    return out


async def run_benchmark(args, requests, request_rate, *, progress=True):
    sem = asyncio.Semaphore(args.max_concurrency)
    delays = build_delays(len(requests), request_rate, args.burstiness, args.seed)
    outputs = [None] * len(requests)
    completed = 0
    start = time.perf_counter()

    async def one(i):
        nonlocal completed
        scheduled_at = start + delays[i]
        await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
        async with sem:
            outputs[i] = await asyncio.to_thread(
                _do_request_blocking,
                args.base_url,
                args.endpoint,
                args.model,
                requests[i],
                args.temperature,
                args.ignore_eos,
                args.request_timeout,
                scheduled_at,
            )
        completed += 1
        if progress:
            print(f"[pure-bench] PROGRESS {completed}/{len(requests)}", flush=True)

    await asyncio.gather(*(one(i) for i in range(len(requests))))
    return outputs, time.perf_counter() - start


def compute_result(args, requests, outputs, duration, tokenizer, percentiles):
    def metrics(indices):
        successful = [i for i in indices if outputs[i].success]
        result = {
            "completed": len(successful),
            "failed": len(indices) - len(successful),
        }
        series = {
            k: []
            for k in ("ttft", "tpot", "e2el", "itl", "client_queue", "offered_ttft")
        }
        for i in successful:
            out = outputs[i]
            if out.output_tokens is None:
                out.output_tokens = len(
                    tokenizer(out.generated_text, add_special_tokens=False).input_ids
                )
                out.token_count_source = "retokenized_text_estimate"
            series["e2el"].append(out.latency)
            series["client_queue"].append(out.client_queue_s)
            series["itl"].extend(out.itl)
            if out.ttft is not None:
                series["ttft"].append(out.ttft)
                series["offered_ttft"].append(out.ttft + out.client_queue_s)
                if out.output_tokens > 1:
                    series["tpot"].append(
                        (out.last_content_time - out.ttft) / (out.output_tokens - 1)
                    )
        for key, values in series.items():
            result[f"{key}_samples"] = len(values)
            result[f"mean_{key}_ms"] = float(np.mean(values) * 1000) if values else None
            for p in percentiles:
                result[f"p{p:g}_{key}_ms"] = (
                    float(np.percentile(values, p) * 1000) if values else None
                )
        result["total_input_tokens"] = sum(requests[i].prompt_len for i in successful)
        result["total_output_tokens"] = sum(
            outputs[i].output_tokens for i in successful
        )
        result["request_throughput"] = len(successful) / duration
        result["input_throughput"] = result["total_input_tokens"] / duration
        result["output_throughput"] = result["total_output_tokens"] / duration
        return result

    result = metrics(list(range(len(requests))))
    result["duration"] = duration
    result["groups"] = {
        name: metrics(
            [
                i
                for i, r in enumerate(requests)
                if (r.prompt_len >= args.prefill_chunk_size) == long
            ]
        )
        for name, long in (("short", False), ("long", True))
    }
    result["input_lens"] = [r.prompt_len for r in requests]
    result["requested_output_lens"] = [r.expected_output_len for r in requests]
    result["output_lens"] = [o.output_tokens for o in outputs]
    result["token_count_sources"] = sorted(
        {o.token_count_source for o in outputs if o.success}
    )
    result["errors"] = [o.error for o in outputs]
    result["workload_sha256"] = hashlib.sha256(
        json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
    ).hexdigest()
    result["potential_full_chunks"] = sum(
        r.prompt_len // args.prefill_chunk_size for r in requests
    )
    result["graph_replay_verified"] = False
    result["measurement_notes"] = [
        "Prompt counts use the local chat template with add_special_tokens=False, matching BasicLLMProcessor.",
        "TTFT is first non-empty text, not internal first-token compute time.",
        "ITL is content-event gaps; TPOT uses text arrival times and may include buffered tokens.",
        "Client queue time is separate; offered TTFT includes that time.",
        "Potential full chunks are shape estimates, not observed graph replay counts.",
    ]
    if args.save_detailed:
        result["requests"] = [asdict(o) for o in outputs]
    return result


async def execute(args, requests, warmups):
    # Avoid the default thread pool's hidden concurrency cap (often 32).
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=args.max_concurrency)
    )
    if warmups:
        outputs, _ = await run_benchmark(args, warmups, float("inf"), progress=False)
        if any(not o.success for o in outputs):
            raise RuntimeError(
                f"warmup failed: {[o.error for o in outputs if o.error]}"
            )
    return await run_benchmark(args, requests, args.request_rate)


def main(argv=None, default_dataset="burstgpt"):
    args = parse_args(argv, default_dataset)
    percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    if any(not 0 <= p <= 100 for p in percentiles):
        raise ValueError("percentiles must be in [0, 100]")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    requests = sample_requests(args, tokenizer)
    warmups = sample_requests(args, tokenizer, warmup=True)
    print(
        f"Workload={args.dataset_name}, input tokens={min(r.prompt_len for r in requests)}.."
        f"{max(r.prompt_len for r in requests)}, concurrency={args.max_concurrency}",
        flush=True,
    )
    outputs, duration = asyncio.run(execute(args, requests, warmups))
    result = compute_result(args, requests, outputs, duration, tokenizer, percentiles)
    result["config"] = {
        k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
        for k, v in vars(args).items()
    }
    keys = (
        "completed",
        "failed",
        "duration",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
    )
    print(json.dumps({k: result[k] for k in keys}, indent=2, ensure_ascii=False))
    if args.result_dir:
        path = Path(args.result_dir) / args.result_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Result: {path}")
    if args.verbose:
        for out in outputs:
            print(out)
    if result["failed"]:
        for error in result["errors"]:
            if error:
                print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
