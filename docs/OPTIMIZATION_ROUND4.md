# Round 4：细项优化记录（2026-11-20，1 小时时间盒）

## 背景
Round 3 已落地预取/并行读/加速器（见 OPTIMIZATION_ROUND3.md）。本轮限定 1 小时，
目标是「训练 + 分词器 + 其它」再挖一轮，允许失败，但结论必须真实。

## 做了什么、测到什么
### 1. 训练取批向量化（已保留）
`TokenDataset.get_batch` 从「逐条切片 + np.stack」改为一次 fancy indexing 取 `(B, block+1)` 窗口。
- 微基准（data/train.bin，block=512）：每批准备 **1.1 ms → 0.02 ms（约 50×）**；
- 单步训练里取批占比很小（medium/batch16 单步约 0.18 s，取批约 1 ms），
  所以**端到端提升在噪声内**，本项价值主要是「大 batch / 慢盘 / 小 block 时不再成为尾巴」；
- 正确性：新增 `tests/test_data.py::test_get_batch_vectorized_matches_windows`，
  并且既有「预取线程 == 同步取批逐项相等」测试继续覆盖 seed 一致性。

### 2. 分词器单字符快路径（试了，**回滚**）
在 v2 编码里对 `len(piece)==1` 的片段直接查词表、跳过堆合并。
在自训 v2 分词器（4k 词表，4 MB 语料）上实测 **慢约 5%**（1,668,692 → 1,587,845 字符/s）：
旧路径对单字符会命中 `_bpe_piece` 的 piece 缓存，而新路径改成生成器 + 逐次查词表，反而更慢。
→ **结论：已有缓存的路径不要做想当然的「快路径」；未达收益不合并。**

### 3. 生成用 inference_mode（已保留）
`LingYu.generate` 装饰器由 `torch.no_grad()` 换成 `torch.inference_mode()`。
语义更严格（禁止把生成张量带进自动求导图），生成结果不变；本轮未单独出速度数字，
属于正确性/防呆改进而非性能项。

### 4. SFT 验证关梯度（已保留，真 bug 级修复）
`eval_sft_loss`（sft.py:158）此前**没有包 `torch.no_grad()`**：每次验证都在建一整张
计算图（显存白涨、速度白慢）。本轮补上 `with torch.no_grad():`，语义不变，
并新增 `tests/test_sft.py::test_eval_sft_loss_runs_without_grad` 钉住「验证时梯度必须关闭」。

实测（medium 47.8M，GPU，验证 batch 8 × 512 token，3 个 batch）：

| | 耗时 | 峰值显存 |
|---|---:|---:|
| 旧（无 no_grad） | 2.875 s | 4,482 MiB |
| 新（no_grad） | 0.327 s | 582 MiB |
| 变化 | **8.78× 更快** | **−3,899 MiB** |

CPU 小模型微基准（4 层 128 宽）两者无差异（0.061 s vs 0.064 s，噪声内）——
图构建开销在大模型/GPU 上才显著，印证了「验证不建图」主要是显存收益 + 大模型速度收益。

### 5. SFT 训练也走 batch 预取（已保留）
`BatchPrefetcher` 增加可选 `fetch` 回调（SFT 的取数方法叫 `batch()`、返回三元组，
不叫 `dataset.get_batch`）；`sft.train` 循环改为预取流，`finally` 里关闭线程。
同类预取在预训练实测 +0.8%（Round 3，batch=16/block=512）；SFT 单步约 75 ms，
批次构造占比相近，属于「顺手把一致性补齐」，未单独出端到端数字。
新增 `tests/test_data.py::test_batch_prefetcher_supports_custom_fetch` 钉住三元组场景。

## 未做 / 下一轮候选
- SFT 验证批量并行（`eval_sft_loss` 仍逐批串行，可复用 Round 2 的 eval batching）；
- `prepare` 分词并行（进程池，需处理 Windows spawn 与 4 GB 级内存回传）；
- `generate` 每步 Python 开销（采样在 CPU 侧，可尝试批量/向量化采样）；
- attention 后端抽象在 KV int8 与本地窗口插件间的组合测试。

## 复现
```powershell
python -m pytest tests/test_data.py -q
python -m pytest tests -q    # 619 条
```
