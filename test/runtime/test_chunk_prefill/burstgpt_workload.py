"""Auditable length sampling; this does not replay original text or timestamps."""
from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter
from pathlib import Path


def add_burstgpt_args(parser):
    parser.add_argument('--burstgpt-model', choices=['GPT-4', 'ChatGPT', 'all'], default='GPT-4')
    parser.add_argument('--burstgpt-output-policy', choices=['cap', 'filter'], default='cap',
                        help='cap preserves eligible inputs but caps output budgets; filter preserves both lengths.')
    parser.add_argument('--allow-repeat-samples', action='store_true',
                        help='Explicitly allow sampling with replacement if the CSV is too small.')


def sample_burstgpt(args, *, warmup=False):
    n = args.warmup_requests if warmup else args.num_prompts
    seed = args.seed + (1_000_003 if warmup else 0)
    raw = Path(args.dataset_path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    required = {'Model', 'Request tokens', 'Response tokens'}
    if not required <= set(reader.fieldnames or []):
        raise ValueError(f'BurstGPT CSV requires columns {sorted(required)}')
    counts = Counter()
    rows = []
    for row_id, row in enumerate(reader, 2):
        counts['csv_rows'] += 1
        if args.burstgpt_model != 'all' and row.get('Model') != args.burstgpt_model:
            counts['excluded_model'] += 1
            continue
        try:
            input_len, output_len = int(row['Request tokens']), int(row['Response tokens'])
            if input_len <= 0 or output_len <= 0:
                raise ValueError('non-positive length')
        except (ValueError, TypeError):
            counts['invalid_lengths'] += 1
            continue
        if not args.use_full_dataset and input_len > args.max_request_tokens:
            counts['excluded_input_limit'] += 1
            continue
        original_output = output_len
        # Keep the input population when only the decode budget is constrained.
        # Record the original output too: capped runs change the joint distribution.
        if not args.use_full_dataset and output_len > args.max_response_tokens:
            if args.burstgpt_output_policy == 'filter':
                counts['excluded_output_limit'] += 1
                continue
            output_len = args.max_response_tokens
            counts['eligible_output_capped'] += 1
        rows.append((row_id, input_len, output_len, original_output))
    if not rows:
        raise ValueError(f'No eligible BurstGPT rows: {dict(counts)}')
    replacement = n > len(rows)
    if replacement and not args.allow_repeat_samples:
        raise ValueError(
            f'BurstGPT requested {n} samples but only {len(rows)} eligible rows '
            f'(model={args.burstgpt_model}, output_policy={args.burstgpt_output_policy}). '
            f'Reduce --num-prompts/--warmup-requests, supply a larger CSV, or explicitly '
            f'use --allow-repeat-samples. Counts: {dict(counts)}')
    rng = random.Random(seed)
    # Never silently turn a small CSV into repeated "independent" observations.
    selected = rng.choices(rows, k=n) if replacement else rng.sample(rows, n)
    long_count = sum(r[1] > args.prefill_chunk_size for r in selected)
    warnings = []
    if args.use_full_dataset:
        warnings.append('use-full-dataset bypasses both token limits; output policy is not applied.')
    if replacement:
        warnings.append('Sampling with replacement: requests are not independent unique CSV rows.')
    if long_count < 20:
        warnings.append(f'Only {long_count} selected long requests; long-request P99 is exploratory.')
    report = dict(
        dataset_sha256=hashlib.sha256(raw).hexdigest(), seed=seed, model=args.burstgpt_model,
        output_policy='uncapped' if args.use_full_dataset else args.burstgpt_output_policy,
        counts=dict(counts), eligible_rows=len(rows),
        eligible_long_rows=sum(r[1] > args.prefill_chunk_size for r in rows),
        selected_rows=n, selected_unique_rows=len({r[0] for r in selected}),
        sampling_with_replacement=replacement, selected_long_rows=long_count,
        selected_output_capped=sum(r[2] != r[3] for r in selected),
        source_row_ids=[r[0] for r in selected],
        source_input_lens=[r[1] for r in selected],
        source_output_lens=[r[3] for r in selected],
        requested_output_lens=[r[2] for r in selected], warnings=warnings,
        notes=['Lengths are sampled from CSV; prompts are synthetic and arrivals are generated.',
               'Long rows above refer to CSV lengths; final grouping uses actual server token counts.'])
    return [(r[1], r[2]) for r in selected], report
