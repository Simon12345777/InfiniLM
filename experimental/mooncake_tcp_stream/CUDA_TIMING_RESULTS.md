# CUDA stage diagnosis — 2026-11-10

Implemented `MC_TCP_CUDA_TIMING=1` (default off) in the isolated Mooncake
private-stream helper and added `analyze_cuda_timing.py`. No installed wheel or
InfiniLM native library was replaced. The previous experimental binaries are
preserved in `/workspace/mooncake-tcp-stream-experiment/before-timing-binary/`.

Built successfully. Real TCP WRITE/READ data tests passed with timing enabled;
two runs of eight mixed PD requests completed with outputs and token usage
identical to the earlier uninstrumented private-stream test.

P-side CUDA host-call totals for completed session lifetimes:

| Operation | Run 1 | Run 2 |
|---|---:|---:|
| Host allocation count | 1280 | 1536 |
| cudaHostAlloc | 668.3 ms | 732.2 ms |
| cudaFreeHost | 1503.4 ms | 1050.2 ms |
| Stream creation | 163.7 ms | 182.4 ms |
| Stream destruction | 28.2 ms | 228.3 ms |
| cudaMemcpyAsync (9216 calls) | 28.3 ms | 30.5 ms |
| cudaStreamSynchronize (9216 calls) | 85.3 ms | 84.4 ms |

The largest allocation calls were 444.3 and 363.9 ms; both fully overlapped
model-runner intervals. In run 1, the 444.3 ms allocation interval also fully
overlapped a final payload write interval. These calls run on the TCP thread.
The per-session allocation/stream lifecycle is a substantial remaining blocking
source in this implementation. API wall time includes any implicit waits;
it must not be described as pure allocation CPU overhead or GPU execution time.

D-side persistent sessions did not emit lifetime summaries before process
shutdown. Their slow events were captured, but zero summary counts do not mean
zero D-side cost. This report is diagnostic evidence, not a statistically sound
measurement of the full benchmark's regression or of a future optimization.

Artifacts under `/workspace/mooncake-tcp-stream-experiment/`:

- `pd-stage-timing/{prefill.log,decode.log,cuda-timing.json,requests.json}`
- `pd-stage-timing-r2/{prefill.log,decode.log,cuda-timing.json,requests.json}`

Next implementation: a bounded pool of reusable pinned buffers and private
streams, separated by device and exclusive ownership. Return resources only
after CUDA and socket users have finished. Keep actual copy completion before
publishing KV readiness; do not share a buffer between active sessions. Pooling
has not been implemented in this change.
