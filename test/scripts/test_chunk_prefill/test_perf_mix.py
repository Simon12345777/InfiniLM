#!/usr/bin/env python3
"""Benchmark an already running server with the shared mixed-workload client."""

from pure_bench_serve import main

if __name__ == "__main__":
    main(default_dataset="mixed")
