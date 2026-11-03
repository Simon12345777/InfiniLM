# Server validation, 2026-11-10 (Asia/Shanghai)

Source and builds: `/workspace/mooncake-tcp-stream-experiment`.
Installed Mooncake wheel was not replaced. InfiniLM native libraries were not rebuilt.
The rebuilt engine is selected only through `experiment-python` in PYTHONPATH.

- `engine` CMake target: built successfully with CUDA 12.9, g++ 11.4.
- Helper: 12 D2H/H2D round trips including resizing, partial slices and switching
  between GPUs 0 and 1, passed in both switch-parsing configurations. The helper
  test itself always exercises the private-copy class.
- Real TCP: OFF and ON each passed 10 WRITE/READ cases; byte contents and
  untouched destination tails verified, sizes 1 to 2 MiB + 17 bytes.
- Real PD: OFF and ON each completed 8 mixed requests, concurrency 4, 8 output
  tokens. All eight generated responses and usage values were identical.
- Enabled P and D logs both contain the private-stream activation message.
- Both GPUs were empty after owned test processes exited.

Logs and timing reports are under `pd-off/` and `pd-on/` in the source directory.
These tests have no warmup and too few requests to establish a performance gain.
Short-request mean final-payload time was 415.3 -> 449.0 ms; final-flag time
377.7 -> 274.0 ms; P-last-step to D-ready time 803.6 -> 732.3 ms.
Long-request P-last-step to D-ready time was 491.5 -> 308.7 ms.
Substantial delay remains: this does not demonstrate that the end-to-end
bottleneck has been eliminated. A full repeated benchmark is still needed.

No cancellation/fault-injection or multi-process tensor-parallel validation was
performed. Keep this an opt-in single-GPU-per-role experiment.

`run_pair.sh` is the next user-run full benchmark, OFF first then ON, same rebuilt
engine and identical parameters. Keep both printed Summary paths. For rigorous
performance analysis also reverse order and repeat to assess run-to-run variation.

Build prerequisites were isolated: yaml-cpp 0.8.0, jsoncpp 1.9.6, and local linker
symlinks to the container's existing libibverbs.so.1/libmlx5.so.1. No system
packages were installed. `build_server.sh` reuses these prepared dependencies.
