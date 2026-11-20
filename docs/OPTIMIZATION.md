# 灵语引擎优化记录（2026-10-24）

本文件记录「体积 / 权重压缩 / 架构可换 / 训练开销 / 全机算力 / 跨厂商」六项优化的
**实测数字、已做改动、未做事项与论文依据**。所有数字都来自测试环境实测，未做的就写「未做」。

## 2.1 体积：12.15 GB → 约 5.6 GB（可复现）

瘦身前（逻辑大小）：

| 目录 | 大小 | 说明 |
|---|---:|---|
| `venv/` | 3.21 GB | 开发环境（torch 2.83 GB 是大头） |
| `dist/LingYu/runtime/venv/` | 3.22 GB | 打包版环境，**与 venv 几乎完全重复** |
| `wheels/` | 1.85 GB | 离线 torch 轮子（1.99 GB 单文件） |
| `out/` | 2.14 GB | 训练断点（含 optimizer 动量） |
| `cache/uv/` | 2.94 GB | 与 venv 是硬链接，**删了不释放**（实测） |

做法（`python packaging/slim.py --apply --delete-wheels`）：

1. **硬链接去重**：runtime venv 里与开发 venv「大小 + SHA-256 都相同」的文件，
   替换成指向同一 inode 的硬链接。共 18,321 个文件、**3.31 GB**，功能/性能零损失。
2. **删除离线轮子**（1.99 GB）：安装/修复改走在线镜像（清华 → 阿里 → 官方 → 上交）。
3. 结果：D 盘已用 **424.3 → 419.4 GB（−4.9 GB）**；引擎目录物理占用约 **5.6 GB**。
   `venv/` 仍是 3.21 GB —— torch CUDA 版本身就 2.8 GB，这是「不损失功能」的硬地板。

⚠️ 代价与注意事项：
- 硬链接是**同一份数据**：以后升级开发 venv 的包，打包版会跟着变。升级后重跑
  `python packaging/slim.py --apply` 重建链接即可（脚本幂等）。
- 不再有离线 torch 轮子；没网时无法离线修复。要恢复离线包，重新下载到 `wheels/`。

## 2.2 断点无损压缩：实测 1.162×，逐 bit 可逆

新增 `src/lingyu/compression.py` + `tests/test_compression.py`（7 条测试）。
算法（参考 ZipNN 的字节平面思路）：

1. 按 **字节平面** 重排 fp32：同一权重的第 0/1/2/3 字节各归一堆 →
   符号位 + 指数位集中，压缩器能利用其集中分布；尾数接近随机，压不动；
2. 每个平面 `zlib` 压缩，存 `{dtype, shape, z}`；
3. 新容器 `b"LYCKPT1" + pickle(protocol=4)` 写盘 —— **不能用 `torch.save`**：
   它的 pickle 协议 2 会把 bytes 膨胀 1.5×（实测 26 MB → 39 MB）。

实测（47.8M 参数，fp32 权重 + Adam 一阶/二阶动量，单卡 8GB 显存）：

| 项 | 原始 | 压缩后 | 比例 |
|---|---:|---:|---:|
| 整个断点 `ckpt.pt` | 573.8 MB | 493.6 MB | **1.162×** |
| 模型权重 | 212.2 MB | 180.4 MB | 1.176× |
| 优化器状态 | 382.4 MB | 311.7 MB | 1.227× |

读回后 **model 与 optimizer 全部 tensor `torch.equal` 为真**（逐 bit 相等）。
耗时：存 25.6 s、读 1.5 s（CPU 上做，不占显存）。默认开启，
`LINGYU_PACK_CKPT=0` 关闭，`LINGYU_PACK_LEVEL` 调 zlib 级别（默认 6）。

**为什么只有 16%**：ZipNN 论文指出 fp32 尾数（23/32 bit）熵极高，无损压缩的
上限就在这里；BF16 权重才能报到 ~30%。想再大幅省空间，方向是
「增量断点（存相对上一档的 delta）」——一次训练多个存档时预计 2~4×，列为下一步。

## 2.3 换架构（模型内核可替换）

已具备：
- `src/lingyu/registry.py` + `plugins/`：内置组件、插件、配置点名三层优先级；
- 模型结构由 `[model]` 段驱动：`n_layer/n_embd/n_head/n_kv_head`（MHA/GQA/MQA）、
  `norm`（rmsnorm/layernorm）、`activation`（swiglu/gelu/relu）、
  `n_experts/...`（MoE）、`rope_base`、`tie_weights`；
- `configs/` 里 `small/medium/moe/wide/nano65` 就是 5 种架构的实测样本；
- 换架构后旧断点会被 `structure_diffs()` 拦住并提示「从头重训」。

未做：把「注意力内核」做成插件级替换（如 MLA/线性注意力）——需要稳定 ABI，
下一步是加 `plugins/example_attention.py` 示例与文档。

## 2.4 训练显存/计算（论文 + 已做/未做）

已做：BF16 autocast、gradient accumulation、fused AdamW（带 fallback）、
`torch.compile` 未默认开（Windows 上编译耗时/兼容性不稳）。
未做（按收益排序）：
1. **激活重计算**（`torch.utils.checkpoint`，+20~40% 计算换 −30~60% 显存），
   论文 arXiv:1604.06174；
2. **MLA**（KV cache 压到 1/4~1/7，DeepSeek-V2 arXiv:2405.04434）；
3. **FP8 训练**（arXiv:2412.19437 §3.3、arXiv:2209.05433，需 Ada/Hopper）；
4. **ZeRO/FSDP2** 多卡（arXiv:1910.02054；PyTorch `fully_shard`）——
   `lingyu launch --backend fsdp|deepspeed` 目前只生成脚本。

## 2.4 已落地：梯度检查点 + 8-bit 优化器（实测，2026-10-24）

论文读完后只挑了**这台机器真能验证**的两项落地，其余留在路线图（见下）。

### 新增开关（都在 `[train]`，GUI 自动生成控件）

| 参数 | 默认 | 作用 |
|---|---|---|
| `grad_checkpoint` | false | 梯度检查点（激活重计算），算力换显存 |
| `optim_bits` | "32" | 优化器状态：32 = 全精度；8 = 一阶动量 int8 分块量化 |

`lingyu plan` 的「装不下」建议里也加上了这两条——**它们真的会生效**
（旧版建议里写过 `activation_checkpoint` 但没人读，这次不同）。

### 实测（单卡 8GB 显存，medium 47.8M，batch 24 × 512，40 步，data1）

| 配置 | 峰值显存 | 吞吐 | 相对 |
|---|---:|---:|---:|
| 基线 | 5.84 GB | 45,723 tok/s | 100% |
| `grad_checkpoint=true` | **2.12 GB（−64%）** | 37,599 tok/s | 82.2% |
| `optim_bits=8` | 5.64 GB（−0.2 GB） | 45,653 tok/s | 99.9% |
| 两者都开 | **1.91 GB（−67%）** | 33,903 tok/s | 74.1% |

- 激活重计算：**省 64% 显存、吞吐降 18%**（比论文里「+30% 计算」更划算，因为
  本机是显存带宽瓶颈而不是算力瓶颈）；40 步 loss 与基线差 5e-4（同一 seed，
  差异来自 GPU 归约顺序，不是 bug）。
- 8-bit 优化器：只省 0.2 GB —— 47.8M 模型权重本身才 0.18 GB，而优化器状态
  0.38 GB；**模型越大省得越多**（7B 模型约省 45 GB）。
- 为什么二阶动量没压成 int8：实测会把接近 0 的小 v 量化成 0，分母 sqrt(v)
  变 0，单个参数一步跳 2.4 → 训练直接坏掉。改成「m int8 + v bf16」（3 字节/
  参数，约 2.6× 节省）后与 fp32 AdamW 的 20 步偏差 < 5e-3。这条**踩坑记录**
  比论文更值钱：8-bit 优化器在真实训练里对 v 的量化必须保守。

### 论文阅读与取舍（2026-10 检索）

| 方法 | 关键论文 | 结论 |
|---|---|---|
| 激活重计算 | Chen et al. [1604.06174](https://arxiv.org/abs/1604.06174)；Korthikanti [2205.05198](https://arxiv.org/abs/2205.05198) | **已实现**（全层重算；选择性重算留待长序列） |
| 8-bit 优化器 | Dettmers et al. [2110.02871](https://arxiv.org/abs/2110.02871) | **已实现保守版**（m int8 + v bf16） |
| 分块/低秩梯度 | Zhao et al. GaLore [2403.03507](https://arxiv.org/abs/2403.03507)、Natural GaLore [2410.16029](https://arxiv.org/abs/2410.16029) | 未实现：需要每步 SVD/投影，单卡 8GB 上额外 CPU 开销大，收益在 7B 以上 |
| ZeRO/FSDP 分片 | Rajbhandari [1910.02054](https://arxiv.org/abs/1910.02054) | 未实现：本机单卡，无验证环境；`launch` 只生成脚本 |
| MLA | DeepSeek-V2 [2405.04434](https://arxiv.org/abs/2405.04434) | 未实现：属于 2.3 架构替换，需先把注意力内核插件化 |
| FP8 训练 | DeepSeek-V3 [2412.19437](https://arxiv.org/abs/2412.19437)；[2209.05433](https://arxiv.org/abs/2209.05433) | 未实现：Blackwell 上 PyTorch FP8 训练链仍不稳，先不开实验开关 |
| 序列并行/长上下文 | Ring Attention [2310.01889](https://arxiv.org/abs/2310.01889)、Ulysses [2309.14509](https://arxiv.org/abs/2309.14509) | 未实现：本机 block_size ≤ 1024，收益小 |

> 注：检索到的 InfiR2 FP8 recipe（arXiv:2509.22536）**已被作者撤回**（数据处理有
> 重大 bug），所以不采用它的结论 —— 读论文也要看撤稿状态。

### 验证

- `tests/test_optim.py`（4 条）+ `tests/test_model.py` 新增 1 条：28 条定向测试全绿；
- 全套 `pytest tests` 全绿（新增 5 条 → 共 **573** 条）；
- CUDA 上训练 40 步实测通过，loss 正常下降。

## 2.5 吃满整机算力（诚实说明）

PyTorch 一个算子只在一个设备上跑，**无法**把 CPU、核显、独显、内存同时喂满。
能做的是：
- 设备自动选择（已新增 `src/lingyu/devices.py` + `doctor` 列出可用后端）；
- 单卡内吃满显存：自动估算 + 建议 batch（已有 `plan`）；
- CPU↔GPU 重叠：DataLoader `pin_memory` + `non_blocking`（未做）；
- 显存不够时把 **优化器状态放 CPU / 分块 offload**（研究级，未做）；
- 真正的多设备训练 = 分布式（FSDP/ZeRO），不是「一台机器堆一起」。

## 2.6 跨厂商（NVIDIA / AMD / 国产）

新增 `src/lingyu/devices.py`：探测 cuda（含 AMD ROCm：`torch.version.hip`）、
mps、xpu、torch_npu（昇腾）、torch_musa（摩尔线程），`pick("auto")` 取最快可用，
`doctor` 逐条列出。核心训练只用 torch 标准算子、fused AdamW 有 fallback，
所以换后端主要取决于「该后端是否装了对应 PyTorch 构建」。
坑：算子覆盖（SDPA flash 实现）、FP8/BF16 支持矩阵、驱动/工具链不同；
这些没法用代码消除，文档里写明而不是假装兼容。

## 2.7 帮助百科：93 篇（新增「学术」组 13 篇）

`gui/help_articles_academic.py`，每篇含定义、公式/机制、与本引擎参数的对应、
论文 arXiv 编号：注意力公式、GQA/MQA、RoPE、SwiGLU/RMSNorm、MoE 门控与
负载均衡、MLA、交叉熵/PPL/BPB、AdamW 与 warmup+cosine、ZeRO/FSDP2、
激活重计算、FP8、无损压缩为什么只能省 16%、跨后端为什么容易/难在哪。

## 验证

- `pytest tests` 全绿（含新增 `test_compression.py` 7 条、`test_devices.py` 5 条）；
- 打包版 v0.3.2 重新构建，`--console doctor` 通过（含新「设备后端」段）。

---

## Round 2（2026-11-20）：评估提速、断点压缩提速、注意力内核示例

> 计划与验收单：`docs/OPTIMIZATION_ROUND2.md`。以下都是本机实测数字。

### R2-1 评估（eval）一次前向多块 —— 3.5×

`data.eval_loss` 原来每次只送 1 条 `block_size` 序列；改成一次前向
`eval_batch` 块（默认 `min(8, 8192/(block_size+1))`），短尾块用 `target=-1`
补齐、按有效 token 加权，与逐块评估**数学等价**（测试用 1e-4 容差钉住）。

| eval_batch | 评估 20,480 token 耗时 | 吞吐 |
|---:|---:|---:|
| 1（旧行为） | 0.21 s | 99,762 tok/s |
| 8 | **0.06 s** | **322,539 tok/s** |
| 99（自动封顶 15） | 0.08 s | 271,357 tok/s |

实际训练里 `eval_interval` 越密收益越明显；`eval_batch` 可显式传，默认按
block_size 自适应。

### R2-2 断点压缩：多线程并行 + 可选 lzma —— 存盘 6.2×

`compression.py` 的字节平面 zlib 改成**多线程按张量并行**（zlib/lzma 都释放 GIL），
读回解压同样并行；容器格式向后兼容（老断点直接可读）。新增
`LINGYU_PACK_METHOD=lzma` 与 `LINGYU_PACK_THREADS`。

实测素材：`out/chat_medium_sft/best_sft.pt`（573.8 MB，53,043,840 参数）：

| 方案 | 存盘 | 读档 | 体积 | 逐 bit |
|---|---:|---:|---:|---|
| zlib 单线程（旧） | 25.91 s | 1.68 s | 494.3 MB | ✅ |
| **zlib 8 线程** | **4.15 s** | **0.68 s** | **494.3 MB** | ✅ |
| lzma 8 线程 | 17.62 s | 0.94 s | 487.8 MB | ✅ |

结论：默认 zlib 并行（体积不变、存盘 6.2×、读档 2.5×）；lzma 只多省 1.3%
体积却慢 4×，作为「存储吃紧」时的显式选项，不做默认。

### R2-3 插件式注意力内核示例（补 2.3 的缺项）

- `model.py` 的 `Attention` 把三处 SDPA 调用收敛成唯一出口
  `_attend(q, k_all, v_all, T, offset)`，插件覆写它即可换内核，
  KV cache / RoPE / int8 缓存 / 分块预填充全部复用。
- `examples/plugins/attention_local_window.py`：局部窗口注意力示例
  （约 40 行核心），窗口由 `LINGYU_ATTN_WINDOW` 控制（0=全注意力）。
- `examples/configs/local_window.ini`、`tests/test_plugin_attention.py`（4 条：
  注册、前向/反向、cache 解码、窗口生效 + 窗口 0 与全注意力一致）。
- `docs/EXTENDING.md` §5「写一个注意力内核」：接口、offset 语义、同窗口约束。

### R2-4 验证

- 全套 `pytest tests` 全绿（数量见 CHANGELOG 第十一批）。
- 真实 100 步 medium 训练（data1，47.8M，batch 16×512）：稳态 45,790～47,752 tok/s、
  峰值显存 4.11 GB、总耗时 52.4 s（含 3 次存档 + 2 次验证；整体平均 15,619 tok/s）；
  断点每个约 494 MB（压缩后），存档从 25.9 s 降到约 4 s。
- 打包 exe 未重打（本轮只改训练/评估与示例；GUI 无变化），下个版本一起打。
