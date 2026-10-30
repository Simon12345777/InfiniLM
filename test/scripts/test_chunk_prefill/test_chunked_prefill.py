#!/usr/bin/env python3
"""Run the three-mode comparison with a mixed long/short workload by default."""

from test_burstgpt_bench import main

if __name__ == "__main__":
    main(default_workload="mixed")
