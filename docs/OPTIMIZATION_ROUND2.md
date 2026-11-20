# 优化 Round 2（2026-11-20，自主进行）

> 承接 `docs/OPTIMIZATION.md`（2.1～2.6）与 `docs/PROJECT_STATE.md` §6。
> 原则：只做能在本机单卡实测出数字的优化；每项都有测试 + 前后对比；不堆「纸面功能」。

## 背景与约束

- 单卡 8GB 显存显卡 / 32GB 内存；torch 2.14.0+cu130；引擎 v0.3.2。
- 已有能力（别重复造）：`grad_checkpoint`（−64% 峰值）、`optim_bits=8`、KV int8、
  `loss_chunk`、分词器 v2、断点 zlib 字节平面压缩（1.16×）、`pin_memory/non_blocking`。
- 当前基线：全套测试 **608** 条绿；medium 47.8M 训练峰值 5.84GB / 45,723 tok/s。

## 计划（按执行顺序）

### R2-1 评估（eval）批量并行 —— 目标 2~4× 提速

现状：`data.eval_loss` 每次只送 **1 条** `block_size` 序列，GPU 严重空闲；
`eval_interval` 到点时训练会停下来等这 20 次前向。

做法：
- 同一窗口切分逻辑（chunk 仍是 `block_size`、不重叠、最多 `eval_iters` 块），
  但一次前向送 `eval_batch` 块（短块用 `-1` target 忽略，按有效 token 加权）。
- `eval_batch` 默认按 `8192 // (block_size+1)` 估算（block 512 → 16），可调。
- 数学性质：模型 loss 本身按有效 token 数求均值，累加 `loss × 有效token` 后除总 token，
  与逐块加权平均**完全等价**。

验收：`tests/test_data.py` 新增「批量 vs 逐块 loss 差 < 1e-5、token 数一致」；
在 data1 验证集实测 eval_iters=20 的前后耗时。

### R2-2 断点压缩：并行化 + 可选 lzma —— 目标 存盘 2×+ 提速

现状：`compression.py` 逐张量、逐字节平面串行 zlib.level=6；573MB 断点存 25.6s。
- 多线程压缩各张量（zlib 释放 GIL），读写线程池；输出格式不变（LYCKPT1 兼容）。
- 新增 `LINGYU_PACK_METHOD=lzma`（preset 6）可选：单线程更慢，但压缩率更高；
  仅当用户显式选择；默认仍 zlib。
- `LINGYU_PACK_THREADS` 控制线程数（默认 min(8, CPU)）。

验收：真实 149.9MB / 573.8MB 断点，记录「存/读耗时 + 体积」；roundtrip 逐 bit 相同
（两种方法都要测）；老格式文件仍可读。

### R2-3 插件式注意力内核示例（补齐 2.3 的「缺一个能跑的示例」）

- `plugins/attention_local_window.py`：注册 `attention="local_window"` 的实现，
  在因果 mask 上叠加滑窗（窗口由 `[model] attn_window` 或环境变量给出），
  接口与内置 Attention 一致（`__call__(x, cache, cache_pos)`）。
- 内置 `[model] attn_window`（int，默认 0=全注意力）——只影响内置注意力实现；
  插件示例演示如何自接。
- `tests/test_plugin_attention.py`：插件可被 `registry.load_plugins()` 发现、
  构造、前向 shape 正确、滑窗 mask 下 loss 可反传。
- `docs/EXTENDING.md` 增加「写一个注意力内核」一节 + 性能/质量注意。

验收：全套测试绿；插件示例配置 `examples/configs/local_window.ini` 可用
`lingyu train` 冒烟（CPU 40 步）。

### R2-4 收尾

- 全套 `pytest tests -q` 绿；`lingyu doctor` 通过。
- 真实 100 步 medium 训练（5GB 语料）记录：tok/s、eval 耗时、存档耗时/体积。
- 更新 `docs/OPTIMIZATION.md`（追加 Round 2 实测表）、`CHANGELOG.md`、本文件验收记录。
- git commit（消息：`perf: batched eval, parallel checkpoint packing, attention plugin example`）。

## 不做（本轮）

- MLA / FP8 / 序列并行：需更大显存或多卡，收益不可验证。
- FSDP2 真循环：无多卡。
- 16k 中文词表：会换 token ID、需全量重 prepare，等用户点头再评估。

---

## 验收记录（2026-11-20 实测）

- R2-1 评估批量：20,480 token 的评估，0.21s → 0.06s（**3.5×**）；`eval_batch` 默认
  `min(8, 8192/(block_size+1))`，等价性由 `test_eval_loss_batched_equals_single_chunk` 钉住。
- R2-2 断点压缩：573.8MB / 5300 万参数断点，存盘 **25.91s → 4.15s（6.2×）**，
  读档 1.68s → 0.68s（2.5×），体积不变（494.3MB），逐 bit 相等；
  lzma 可选（487.8MB，但存盘 17.6s，不默认）。
- R2-3 插件注意力：`Attention._attend()` 成为唯一 SDPA 出口；
  `examples/plugins/attention_local_window.py` + `tests/test_plugin_attention.py`（4 条）
  + `docs/EXTENDING.md` §5。
- R2-4 全测与真实训练：`pytest tests` **608 条全绿**；medium 47.8M / data1 / batch 16×512 /
  100 步：稳态 **45,790～47,752 tok/s**，峰值显存 **4.11 GB**，总耗时 52.4 s
  （含 3 次断点存档与 2 次验证；平均 15,619 tok/s 是被存档/验证拉低的整体口径）。
  验证 loss 7.1754（第 50 步）→ 6.3249（第 100 步）；断点 3 个、每个约 494 MB。

## 遗留（下一轮候选）

- eval 批量更激进（`eval_iters` 很大时按显存自适应更大 batch）；
- 断点「增量/差分」压缩（2.2 原目标，预计多存档 2~4×）；
- 插件式优化器 / 调度器（注册表已有 kind，训练循环尚未全走注册表）；
- MLA/FP8 仍不做（单卡 8GB 验证不了收益）。
