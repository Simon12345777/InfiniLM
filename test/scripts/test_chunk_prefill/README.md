# Chunk prefill 性能对照

需要在已编译、安装当前 InfiniLM 的 GPU 环境运行，客户端还需要 `numpy` 和
`transformers`。这里修改的是性能脚本，不会自动下载模型或启动服务。

默认运行三组，每组独立启动服务、预热，重复 3 次并轮换组别顺序：

| 模式 | 分块 prefill | chunk graph |
|---|---|---|
| `baseline` | 关 | 关 |
| `chunk` | 开 | 关 |
| `chunk_graph` | 开 | 开 |

三组的 decode graph 配置始终一致，默认开启；可用 `--no-enable-graph` 全部关闭。
`chunk_graph vs chunk` 用于隔离 chunk graph 开关的影响；`chunk vs baseline`
包含分块及其调度策略的影响；`chunk_graph vs baseline` 是整体效果。
不会断言一定加速，负数表示退化。

以下命令从 InfiniLM 仓库根目录执行，替换模型路径和设备。旧环境使用天数卡时保留
`--device iluvatar`，CUDA 环境改为 `--device cuda`。不再硬编码 GPU 编号，
使用现有 `CUDA_VISIBLE_DEVICES`，也可显式传 `--cuda-visible-devices 0`。

## 长提示词，突出 prefill 和 chunk graph 的影响

```bash
python test/scripts/test_chunk_prefill/test_chunked_prefill.py \
  --model-path /path/to/model --device iluvatar \
  --workload uniform --long-input-tokens 4096 --output-tokens 1 \
  --prefill-chunk-size 256 --max-num-batched-tokens 4096 \
  --max-batch-size 16 --max-concurrency 16 --num-prompts 128 \
  --warmup-requests 16 --repeats 3
```

输入长度是含 chat template 的目标长度，脚本按本地 tokenizer 实际计算的长度记录；
部分 tokenizer 可能产生略有差异的长度。调度器仍将长输入切成固定大小的完整 chunk，
尾段走普通 prefill。`--output-tokens 1` 时没有 TPOT 样本，显示 `N/A`。
这仍是 HTTP 端到端测试，TTFT 包含排队、分词和传输时间。

## 长短混合，观察普通 prefill 优先和让行策略

```bash
python test/scripts/test_chunk_prefill/test_chunked_prefill.py \
  --model-path /path/to/model --device iluvatar \
  --workload mixed --long-input-tokens 4096 --short-input-tokens 128 \
  --long-ratio 0.5 --output-tokens 32 \
  --prefill-chunk-size 256 --max-num-batched-tokens 4096 \
  --max-batch-size 16 --max-concurrency 16 --num-prompts 128
```

默认 `--request-rate inf`，用并发上限维持压力。可改成 `--request-rate 2`
测试较低到达率；有限到达率使用固定 seed 的 gamma 到达间隔，
`--burstiness 1` 对应 Poisson 到达。低到达率下吞吐常由请求速率决定，
应重点看延迟。当前调度是提示词优先，持续有提示词时 decode 可能等待，
应一起观察 TPOT 和 E2E，不能仅凭短请求 TTFT 宣称整体性能提高。

## BurstGPT 长度分布

```bash
python test/scripts/test_chunk_prefill/test_burstgpt_bench.py \
  --model-path /path/to/model --device iluvatar \
  --prefill-chunk-size 256 --max-num-batched-tokens 4096 \
  --max-request-tokens 8192 --max-response-tokens 256 \
  --num-prompts 128 --max-concurrency 16 --request-rate inf
```

读取 CSV 字段名筛选 GPT-4 且输入、输出长度大于 0 的行，只使用长度分布，
合成提示词，不使用数据集时间戳或声称重放原始文本。过滤在内存中完成，原 CSV
保持不变。`--use-full-dataset` 取消长度上限过滤；请确保模型上下文和 KV 池容量
能容纳输入、输出，可用 `--num-blocks` / `--block-size` 调整相同的池配置。

## 输出与测量口径

- 每组使用相同 seed、请求内容、到达计划、并发和 batch token 预算。
  默认关闭前缀复用，避免预热或重复请求把 prefill 变成缓存命中。
- 服务就绪及预热后才计时；启动和图录制时间另记为 `startup_s`。
- 默认通过服务端 `--ignore-eos` 保持请求输出长度；发送当前服务支持的
  `max_tokens`。遇到失败、截断流或非预期提前结束则退出，不用失败请求计算收益。
- TTFT 从实际发起请求到首个非空文本；role、usage、finish 事件不计入首 token。
  本地并发等待单独统计为 `client_queue`，`offered_ttft` 包含该等待。
- 当前服务端流式响应无 usage。正常 `length` 结束时按请求上限计数；若使用
  `--no-ignore-eos` 且提前停止，则标记为文本重新分词估算。
  不把 SSE 事件数当作 token 数；跨组输出 token 数不一致时拒绝生成收益对比。
- TPOT 为首尾文本到达时间差除以输出 token 数减一；ITL 是文本事件间隔。
  空 token、缓冲输出会使它们与引擎内部逐 token 延迟不同。
- 汇总请求/输入/输出吞吐、TTFT/TPOT/E2E、P95/P99、短/长请求分组，
  记录多次运行的中位数、最小值、最大值。
- `bench_results/<时间>/` 保存配置、每组 JSON、服务日志、`summary.json`。
  `--save-detailed` 额外保存逐请求文本和时序。请求指纹用于校验各组负载相同。
- 服务 API 暂不暴露 graph replay 计数，因此 `graph_replay_verified=false`；
  `potential_full_chunks` 只是输入形状估算。图开关对比可测量端到端影响，
  要证明实际 device graph replay 或 kernel launch 减少，仍需在目标设备上使用 profiler。

加 `--dry-run` 可仅检查服务端和客户端命令。单跑组别使用
`--modes baseline`、`--modes chunk` 或 `--modes chunk_graph`。
旧 `--chunk-size` 仅保留为测试脚本参数别名，实际传给服务端的是
`--prefill-chunk-size` 和 `--enable-chunked-prefill`，不再用 0 表示关闭。

`test_perf_mix.py` 是对已运行服务的独立客户端，需显式指定
`--base-url http://127.0.0.1:2333 --model 模型ID --tokenizer /path/to/model`；
需要固定输出长度时同时传 `--ignore-eos` 并确保服务以该开关启动。
