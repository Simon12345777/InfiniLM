# 在 InfiniLM 上对照测试 chunked prefill

在安装了当前 checkout 的 Linux GPU 环境，从仓库根目录运行。单机 Mooncake PD 使用两张支持 BF16 的 GPU，每侧各加载完整模型，TP=1。先确认单卡能容纳模型和 KV 池，当前 Python 能导入 InfiniLM、Mooncake Transfer Engine 和服务依赖。客户端使用 numpy、transformers 和本地模型的 tokenizer。

## 两卡冒烟测试

```bash
python test/runtime/test_chunk_prefill/test_chunked_prefill.py \
  --model-path /models/Qwen3-0.6B --prefill-gpu 0 --decode-gpu 1 \
  --modes pd_chunk --workload uniform --long-input-tokens 1536 \
  --block-size 256 --chunked-prefill-size 512 --num-blocks 64 \
  --num-prompts 2 --max-concurrency 1 --warmup-requests 0 --output-tokens 8
```

脚本会启动 P、D 和 proxy，等待两侧模型就绪、P 注册 bootstrap、proxy 完成握手，再将请求发给 proxy。默认使用 TCP，配置好 RDMA 后可传 `--mooncake-protocol rdma`。默认端口为 proxy 2333、P 8100、D 8200、bootstrap 8998，分别通过 `--port`、`--prefill-port`、`--decode-port`、`--bootstrap-port` 修改。每组结束或报错后都会清理该组启动的进程。加 `--dry-run` 可以先看各进程的设备、环境变量和完整命令。

## 同一对 GPU 上对照性能

```bash
python test/runtime/test_chunk_prefill/test_chunked_prefill.py \
  --model-path /models/Qwen3-0.6B --prefill-gpu 0 --decode-gpu 1 \
  --modes pd_pair --workload mixed --long-input-tokens 4096 --short-input-tokens 128 \
  --chunked-prefill-size 512 --max-num-batched-tokens 4096 --output-tokens 32 \
  --num-prompts 64 --warmup-requests 8 --max-concurrency 8 --repeats 3
```

| 模式 | 服务拓扑 | Prefill |
|---|---|---|
| `pd` | P + D + proxy | P 整段计算 |
| `pd_chunk` | 相同的 P + D + proxy | P 分块计算，完成后统一交接 KV |
| `baseline` | 单实例 | 整段计算 |
| `chunk` | 单实例 | 分块计算 |

默认 `pd_pair` 依次对比 `pd` 和 `pd_chunk`。可选 `local_pair` 或 `all`。重复多次会轮换运行顺序。各组使用相同的 seed、请求、到达计划、batch 配置和输出长度。默认关闭前缀复用和计算图。batch token 预算通过服务端环境变量 `INFINILM_MAX_NUM_BATCHED_TOKENS` 设置，须在 1024 和模型上下文上限之间。chunk 大小须为 block 大小的正整数倍且不超过预算。

## BurstGPT 和已有服务

```bash
python test/runtime/test_chunk_prefill/test_burstgpt_bench.py \
  --model-path /models/Qwen3-0.6B --prefill-gpu 0 --decode-gpu 1 \
  --chunked-prefill-size 512 --num-prompts 64 --max-concurrency 8
```

BurstGPT 使用随目录提供的 CSV，默认选择 GPT-4 的长度记录并合成文本；不回放原始文本或原始到达时间。三组对比应使用相同 CSV、seed 和采样参数。

- 默认 `--burstgpt-output-policy cap`：保留输入长度符合 `--max-request-tokens` 的有效记录，将输出预算限制到 `--max-response-tokens`。这样不会仅因输出较长就删掉对应的长输入，但输出长度分布经过截断，不能称为原始输入/输出联合分布回放。
- `--burstgpt-output-policy filter`：沿用原来的双长度过滤，输入或输出超过上限的记录都会排除，保留记录的两个长度不变。
- 默认无放回采样。样本不足时在启动服务前报错；确实需要重复抽样时显式传 `--allow-repeat-samples`。预热独立抽样，可能与正式样本重合；正式结果不计预热。
- `--burstgpt-model GPT-4|ChatGPT|all` 选择模型记录。`--use-full-dataset` 沿用旧语义：跳过两个长度上限，不代表自动选择所有行；此时需确保服务端输出和上下文容量足够。
- 每轮结果的 `workload_sampling` 保存 CSV SHA256、行号、原始长度、实际输出预算、筛选计数和重复采样情况。长请求数量少时会提示尾延迟统计不稳定，不会人为提高长请求比例。

当前附带 CSV 在 GPT-4、输入上限 8192、输出上限 256、chunk=1024 时：`filter` 有 116 条合格记录（6 条长输入），`cap` 有 184 条（19 条长输入）。可用 `--num-prompts 184 --burstgpt-output-policy cap --seed 0` 覆盖全部合格记录并随机排列；原来的 `--num-prompts 256` 需要更大的 CSV 或显式允许重复采样。即使全选，19 条长输入仍不足以稳定估计 P99。

若样本都短于或等于 chunk，分块验收会失败。调整 chunk 大小、采样数或改用上述 uniform 冒烟测试。更换筛选策略后应重跑全部对照组，不与旧策略的结果直接比较。

对已有 proxy，可单独运行客户端。此方式只验证 HTTP 响应并测量性能，服务端分块和 KV 交接需要另查服务日志。

```bash
python test/runtime/test_chunk_prefill/test_perf_mix.py \
  --base-url http://127.0.0.1:2333 --model Qwen3-0.6B \
  --tokenizer /models/Qwen3-0.6B --prefill-chunk-size 512
```

## 如何判读结果

- 每轮 JSON、各进程日志和最终 `summary.json` 在 `bench_results/<时间>/`。客户端验证 SSE 的 finish、usage、[DONE]，吞吐按服务端返回的实际 token 数计算；TPOT 需要至少两个输出 token。TTFT 是到首个非空文本的端到端延迟。
- 自动启动的分块组要求日志能证明正式请求至少有一个经历过分块续算；默认关闭前缀复用时，最终 chunk 的调度次数还须与超过阈值的请求数一致（含预热）。测量结果分别记录 `chunked_requests_including_warmup` 与 `pd_handoffs_including_warmup`。
- PD 组要求 D 成功接收 KV 的次数等于预热和正式请求总数。发现本地 prefill 回退、交接缺失、请求失败或截断流会导致测试失败。不同负载或不同输出长度不会生成性能对比。
## 单机 PD 提前传输对照

测试前缀写并发上限时，把下方命令的模式改成
`--modes prefix_limit_pair --max-inflight-prefix-writes 1`。两组均开启 overlap：
`pd_overlap` 沿用发送线程池的原有并发，`pd_overlap_limited` 在每个 P worker
提交前缀写之前限制在途数。两组最终写都直接提交。直接启动服务可在 P 的
`kv_connector_extra_config` 中设置 `max_inflight_prefix_writes`，省略或 0 表示不限。

要单独比较计算与传输重叠，使用 `--modes overlap_pair`。两组的 P 都按
相同大小分块；`pd_chunk` 在结束时一次发送，`pd_overlap` 在 P 端配置
`kv_connector_extra_config.chunk_transfer_overlap=true`。该选项默认关闭。
首版用于普通文本、每侧一张卡、TP=PP=1、关闭前缀缓存的 Mooncake 部署。
D 先公布目标页。P 在每次中间 forward 完成后发布已封闭的整页，发送线程
按顺序写入尚未确认的范围；D 晚到时可合并多个 chunk，最后一次写入补齐尾页
并设置整条请求的就绪标志。D 在完整 KV 就绪后才能 decode。

从仓库根目录运行：

```bash
python test/runtime/test_chunk_prefill/test_chunked_prefill.py \
  --model-path "$MODEL" --prefill-gpu 0 --decode-gpu 1 --modes overlap_pair \
  --workload mixed --long-input-tokens 4096 --short-input-tokens 128 \
  --block-size 256 --chunked-prefill-size 1024 --max-num-batched-tokens 4096 \
  --output-tokens 32 --num-prompts 64 --warmup-requests 8 --max-concurrency 8 \
  --repeats 2
```
