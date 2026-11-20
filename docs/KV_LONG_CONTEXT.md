# KV 缓存与长上下文显存优化

目标：让同一个模型在同样 8GB 级显卡上，跑更长的上下文、更大的 batch，或至少把 KV
占用的显存省下来给模型本身。本文记录 2026-11-06 落地的三项改动、实测数据与取舍。

## 1. KV cache int8（`[model] kv_cache_dtype`）

- 取值：`auto`（默认，CUDA 上 bf16、CPU 上 f32）/ `f32` / `bf16` / `int8`。
- int8 存法：每个 token、每个 head 取 K/V 的 absmax 作为 scale（`scale = amax/127`），
  int8 存数据，fp32 存 scale。读取时 `k.to(dtype) * scale` 还原。
- 这是 KIVI（arXiv:2402.02750）思路的简化版：KIVI 按 channel 量化 K、按 token 量化 V
  才能压到 2-bit；我们只做 per-token per-head 的 int8，实现简单、对生成质量影响小，
  换来 47% 的 KV 显存下降。
- 实测（8GB 显存级显卡，medium 47.8M，bf16 权重）：

  | KV 类型 | 生成速度（64 token） | KV 显存（相对 bf16） |
  |---|---:|---:|
  | bf16 | 155 tok/s | 100% |
  | int8 | 106 tok/s | **53%** |

  长序列的 KV 占用（单序列）：

  | 上下文 | bf16 | int8 |
  |---:|---:|---:|
  | 512 | 2.5 MB | 1.3 MB |
  | 2,048 | 10.0 MB | 5.3 MB |
  | 8,192 | 40.0 MB | 21.2 MB |
  | 32,768 | 160.0 MB | 85.0 MB |

- 取舍：int8 生成慢约 30%（每步多一次量化和反量化）。**显存吃紧时用 int8，追求吞吐用
  bf16**。KV 本身在 8GB 上只是中等头，激活和优化器状态才是大头；int8 的主要价值是
  「长上下文 + 大 batch」时把预算让出来。

## 2. 分块交叉熵（`[train] loss_chunk`，默认 8192）

- 训练时 logits 形状是 `[tokens, vocab]`，softmax 的中间量约为 `tokens × vocab × 2 字节`
  （bf16）。block=2048、batch=8、vocab=8192 时单是这块就约 268 MB。
- `loss_chunk=4096` 把交叉熵按 token 行分块累加（`reduction="sum"` 后除以有效 token 数），
  数学上与原实现等价，峰值只取决于分块大小，而不是整批 token 数。
- 实测（medium 47.8M，block 1024、batch 8、3 步）：峰值显存 4.11 GB → **3.99 GB**
  （−120 MB，约 −3%）；两步 loss 完全一致。模型和 batch 越大、词表越大，省得越多
  （收益 ≈ `(总token − chunk) × vocab × 2B`）。

## 3. 论文取舍

| 论文 | 结论 | 我们的处理 |
|---|---|---|
| Chen 2016（arXiv:1604.06174）、Korthikanti 2022（arXiv:2205.05198） | 激活重计算省显存，代价是算力 | 已实现 `train.grad_checkpoint` |
| Dettmers 2021（arXiv:2110.02871） | 8-bit 优化器状态 | 已实现 `train.optim_bits=8` |
| KIVI（arXiv:2402.02750） | KV 2-bit 可用，需 per-channel/per-token 量化 | 先落 int8 简化版 |
| FP8 训练（arXiv:2209.05433 等） | 大规模训练精度可持平 | 单卡小模型收益有限，列入路线图 |
| StreamingLLM（arXiv:2309.17453） | 4 个 attention sink + 滑窗可跑 4M token | 推理侧优化，下一步 |
| Mistral SWA（arXiv:2310.06825） | 窗口 4096、32 层感受野 131K | 需要重训，暂不做 |
| MLA（arXiv:2405.04434） | KV −93% | 需要重训 + 改架构，暂不做 |
| GQA（arXiv:2305.13245） | 分组查询省 KV | 引擎已支持（`n_kv_head`） |
| PagedAttention（arXiv:2309.06180）、chunked prefill（arXiv:2308.16369） | 多请求 / 长 prompt 吞吐 | 单用户场景收益小，暂不做 |

## 4. 怎么用

```powershell
# 显存吃紧：KV 量化 + 梯度检查点 + 8-bit 优化器
lingyu train --config configs/chat_medium.ini `
  --set model.kv_cache_dtype=int8 `
  --set train.grad_checkpoint=true `
  --set train.optim_bits=8

# 长上下文训练：分块交叉熵（默认 8192，可调小）
lingyu train --config configs/chat_medium.ini --set train.loss_chunk=4096
```

注意：`kv_cache_dtype` 只影响推理时的 KV 存储；训练时注意力不缓存（与 v0.3 相同）。
