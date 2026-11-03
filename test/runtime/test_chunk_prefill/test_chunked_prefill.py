#!/usr/bin/env python3
"""Compare Mooncake PD with and without chunking on mixed long/short prompts."""

from test_burstgpt_bench import main

if __name__ == "__main__":
    main(default_workload="mixed")
