# Round 3：加速与数据管线优化（2026-11-20）

目标：在**不改模型定义、不改断点格式、不改变训练随机序列**的前提下，
把「算得快」「读得快」「等得少」三件事各推进一档，并且每一项都有可复现的实测。

> 约定：下面的数字都是本仓库自己的验收基准（`scripts/` 里没有专门的 bench 脚本，
> 命令写在每节末尾，可直接重跑）。机器为单卡 8 GB 显存 + 6 核桌面 CPU，
> 你换硬件后绝对值会变，**只应比较同一台机器上的相对值**。

## 1. 训练加速总览（47.8M medium，batch 16 × 512 = 8,192 token/步，bf16 autocast）

| 配置 | 吞吐（token/秒） | 相对 |
|---|---:|---:|
| AdamW 默认实现 | 42,840 | 100% |
| `fused=True`（本次默认启用） | 44,214 | +3.2% |
| `fused=True` + TF32 | 44,449 | +3.8% |

结论：**fused AdamW 是稳赚的（+3%）**；TF32 在 bf16 训练里只照顾残留 fp32 算子，
收益 <1%，但零风险，所以一并打开。round 2 的评估批量并行保持（见 2）。

## 2. 评估批量并行（round 2 已做，本次补测纠正）

`data.eval_loss(..., eval_batch=N)` 一次前向评估 N 块，块间因果、短尾用 -1 补齐、
按有效 token 加权。真实 47.8M 模型上重测（此前 3.5× 的说法来自一个 d_ff 配错、
实际只有 15M 参数的误建模型，**已纠正**）：

| eval_batch | 20,480 token 评估耗时 | 吞吐 |
|---:|---:|---:|
| 1 | 0.324 s | 63,231 token/s |
| 8 | 0.122 s | 168,189 token/s |

即 **2.66×**，且 loss 完全一致（`tests/test_data.py::test_eval_loss_batched_equals_single_chunk`）。

## 3. 训练 batch 预取（新增）

`data.BatchPrefetcher`：后台线程用 `queue` 双缓冲预取 `(x, y)`，与 GPU 计算重叠。
实测（100 步，data1）：

| `LINGYU_PREFETCH` | 平均速度 |
|---|---:|
| 0（同步） | 32,674 token/s |
| 1（预取，默认） | 32,925 token/s |

**+0.8%**，原因是数据读取只占单步约 5~8%（小模型、NVMe、CPU 不弱）。
真正收益场景：batch 更大、CPU 更慢、或数据在慢盘/网络盘上。默认开启、无副作用；
`LINGYU_PREFETCH=0` 可关。等价性用测试钉住：同 seed 下预取序列与同步序列逐项相等。

## 4. 语料准备并行读文件（新增）

`prepare.run_prepare` 的「读取 + 解码/解压 + JSON 解析」改成线程池，顺序由
`pool.map` 保证，清洗/拼接仍单线程，产物与串行**逐字符一致**。实测（5.27 GB 语料里
取 4 个 480 MB 大文件，1.92 GB）：

| 线程数 | 耗时 | 相对 |
|---:|---:|---:|
| 1 | 4.97 s | 1.00× |
| 2 | 4.40 s | 1.13× |
| 4 | 4.43 s | 1.12× |

**只有 1.1×**：大文件场景瓶颈在磁盘顺序读与解码 CPU，不在等待。
预期收益主要在**小文件很多**的目录（上千个 md，每文件一次的 seek/解码开销可重叠）；
这一项尚无真实小文件语料基准，标为「待测」，不写进收益承诺。
`LINGYU_PREPARE_WORKERS=N` 可调，`=1` 强制串行（排查用）。

## 5. 加速开关统一（新增 `src/lingyu/accelerators.py`）

- CPU 线程：默认 `min(8, cpu_count)`（`LINGYU_CPU_THREADS` 可改）；
- `torch.set_float32_matmul_precision("high")`；
- CUDA 上统一开 TF32 + `cudnn.benchmark`（train / sft 共用同一入口）；
- `LINGYU_NO_ACCEL=1` 一键全关，便于排查「换设置后结果不同」。
- **`LINGYU_COMPILE=1`（默认关）**：探测到 Triton 才启用 `torch.compile`；
  Windows 官方轮子没有 Triton，会打印提示并跳过（实测强行 compile 会让进程
  以退出码 1 直接死掉，属于不可接受的默认行为，故必须探测后再用）。

## 6. 生成稳态速度（参考）

47.8M 模型、64 new token、`use_cache=True`：第 2 次及以后 **127 token/s**；
第 1 次 9.85 s（CUDA 算子懒加载/分配器预热），属于一次性开销，不代表稳态。

## 7. 验证

- 全套测试：**616 条全绿**（`pytest tests -q`；本次 +8 条）。
- 新增测试：`test_accelerators.py` 4 条、`test_data.py` 预取 2 条、
  `test_prepare.py` 并行读 2 条。
- 兼容性：分词器、断点格式、配置文件、CLI 参数均未变；老断点可直接读。

## 8. 没做 / 下一步

- 预取在 CPU 训练、大 batch、慢盘上的收益未测（预期更明显）；
- 小文件海量目录的并行读基准待补；
- MLA / FP8 / 序列并行 / FSDP2 真循环仍只停留在论文与路线图（单卡 8 GB 场景
  收益/风险不划算，且需要重训或改并行层）；
- 增量断点（只存变化的张量）仍是 2.2 的待办。

## 复现命令

```powershell
# 训练吞吐（改 --set train.max_steps=100，比较 LINGYU_PREFETCH=1/0）
lingyu train --config configs/chat_medium.ini --force --set data.data_dir=data1 `
  --set train.max_steps=100 --set train.eval_interval=0 --set train.ckpt_interval=0

# 评估并行（Python 里调 eval_loss('eval_batch'=1 vs 8)）
# 准备阶段并行（LINGYU_PREPARE_WORKERS=1 vs 8 后对比耗时/产物哈希）
```
