# -*- coding: utf-8 -*-
"""学术向百科（第二批）：每个条目给出定义、公式/机制、与论文的对应关系。

写作原则：
  - 公式用纯文本 Unicode（GUI 不渲染 LaTeX），符号先解释再用；
  - 只写能落到本引擎实现/参数上的内容，不空谈；
  - 论文给 arXiv 编号，方便读者按号检索。
"""

ACADEMIC = [
    {
        "id": "acad-attention",
        "title": "学术：注意力机制的数学定义",
        "group": "学术",
        "keywords": ["attention", "softmax", "即 qkv", "数学", "公式"],
        "body": """**定义**：给定查询矩阵 Q（n×d_k）、键 K（m×d_k）、值 V（m×d_v），
缩放点积注意力为

  Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V

其中 √d_k 的缩放项用来抵消点积随维度增大而增大的方差，
使 softmax 不进入饱和区、梯度不至于消失。

**直觉**：Q 是「我要找什么」，K 是「我有什么」，两者点积越高说明越匹配；
softmax 把匹配分数变成概率分布；再用它加权求和 V，得到「检索到的信息」。

**因果掩码（causal mask）**：自回归训练时，位置 i 只能看 ≤i 的位置，
实现上把上三角的分数设为 -∞（本引擎用 PyTorch SDPA 的 is_causal=True）。

**论文**：Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762；
缩放点积注意力的复杂度与训练细节见该文 §3.2.1。""",
    },
    {
        "id": "acad-gqa",
        "title": "学术：GQA / MQA —— 用更少 KV 头换更小缓存",
        "group": "学术",
        "keywords": ["gqa", "mqa", "kv", "多头", "分组查询"],
        "body": """**问题**：标准多头注意力（MHA）里每个头都有独立的 K/V，
KV cache 大小随头数线性增长；推理时这是显存大头。

**MQA**（Multi-Query Attention）：所有查询头共享**一份** K/V。
KV cache 缩小到 1/n_head，但质量下降较明显。

**GQA**（Grouped-Query Attention）：折中——把 n_head 个查询头分成 n_kv_head 组，
每组共享一组 K/V。n_kv_head = n_head 时等价于 MHA；
n_kv_head = 1 时等价于 MQA。

本引擎参数：`[model] n_head`（查询头数）、`n_kv_head`（KV 头数，必须整除 n_head）。
KV cache / token 的字节数 ≈ 2 × n_layer × n_kv_head × head_dim × 2（fp16）。

**论文**：Ainslie et al., *GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints*, arXiv:2305.13245；MQA 见 Shazeer,
*Fast Transformer Decoding*, arXiv:1911.02150。""",
    },
    {
        "id": "acad-rope",
        "title": "学术：RoPE 旋转位置编码为什么能外推",
        "group": "学术",
        "keywords": ["rope", "位置编码", "旋转", "外推", "相对位置"],
        "body": """**做法**：把每个位置的 q/k 向量两两分组，按位置 m 旋转角度 m·θ_i：

  [x_{2i}, x_{2i+1}] → [x_{2i}·cos(mθ_i) − x_{2i+1}·sin(mθ_i),
                        x_{2i}·sin(mθ_i) + x_{2i+1}·cos(mθ_i)]

**关键性质**：两个位置的旋转点积只依赖它们的**相对距离** m−n，
所以注意力天然编码相对位置，不需要学习位置 embedding。

**频率**：θ_i = base^(−2i/d)，本引擎 `[model] rope_base`（默认 10000）。
base 越大，低频分量越慢，能覆盖更长的上下文（长度外推的常见做法）。

**注意**：RoPE 本身不保证外推；超出训练长度后效果取决于频率分布与
训练数据，常见做法是继续训练或调整 base（例如 1e6）。

**论文**：Su et al., *RoFormer: Enhanced Transformer with Rotary Position
Embedding*, arXiv:2104.09864。""",
    },
    {
        "id": "acad-swiglu-rmsnorm",
        "title": "学术：SwiGLU 前馈层与 RMSNorm",
        "group": "学术",
        "keywords": ["swiglu", "rmsnorm", "前馈", "归一化", "激活"],
        "body": """**SwiGLU**（Shazeer 2020）：前馈层从两层 MLP 变成三矩阵门控结构

  FFN(x) = (Swish(x·W1) ⊙ (x·W3)) · W2，  Swish(z) = z·sigmoid(βz)

「门控」让网络按输入内容选择性地放行特征；同等参数量下比 ReLU/GELU MLP
效果更好，代价是多一个矩阵乘。本引擎 `[model] activation = swiglu|gelu|relu`。

**RMSNorm**（Zhang & Sennrich 2019）：只按均方根归一化、不减均值

  RMSNorm(x) = x / sqrt(mean(xᵢ²) + ε) · γ

比 LayerNorm 少一次均值计算与一组偏置，训练更稳、略快；本引擎
`[model] norm = rmsnorm|layernorm`，`norm_eps` 即公式里的 ε。

**论文**：Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202；
Zhang & Sennrich, *Root Mean Square Layer Normalization*, arXiv:1910.07467。""",
    },
    {
        "id": "acad-moe",
        "title": "学术：MoE 门控、负载均衡与激活参数",
        "group": "学术",
        "keywords": ["moe", "门控", "top-k", "负载均衡", "专家", "aux loss"],
        "body": """**结构**：每层有 E 个专家（前馈网络）+ 一个门控 g(x)=softmax(x·W_g)。
只取 top-k 个专家：

  y = Σ_{i∈TopK(g)} g_i(x)·Expert_i(x)

本引擎默认 1 个共享专家 + E 个路由专家（DeepSeekMoE 思路）：共享专家
永远参与，路由专家按 token 选择，从而把「通用知识」与「专精知识」分开。

**负载均衡**：如果门控总选同几个专家，其余专家训不动。DeepSeek-V3 用
**无辅助损失的偏置调节**：给每个专家一个偏置 b_i，按最近一个 batch 的
被选频率微调，不往主损失里加 aux loss，避免干扰语言建模目标。
本引擎 `[model] moe_aux_loss` / `moe_z_loss` 提供传统辅助损失（可调系数）。

**为什么训练快、生成慢**：训练时每个 token 只算 top-k 个专家 → 计算量
≈ k/E 的稠密 FFN；但生成时 expert 权重分散、batch 小，访存不连续，
所以实测生成更慢。本引擎的 `moe` 配置实测 46,657 tok/s 训练 vs 114 tok/s 生成。

**论文**：DeepSeek-AI, *DeepSeekMoE: Towards Ultimate Expert Specialization*,
arXiv:2401.06066；*DeepSeek-V3 Technical Report*, arXiv:2412.19437 §2.1。""",
    },
    {
        "id": "acad-mla",
        "title": "学术：MLA（多头潜在注意力）与 KV 压缩",
        "group": "学术",
        "keywords": ["mla", "潜在注意力", "kv cache", "压缩", "deepseek"],
        "body": """**动机**：长上下文推理的瓶颈是 KV cache。GQA 只能按头数缩，
MLA 进一步把每个 token 的 K/V 压成一个低维**潜在向量** c（维度 d_c ≪ n_head·d_head）。

**做法**（DeepSeek-V2）：先把输入投影到 c，再从 c 上投影出 K、V：
  c_t = W_DKV · h_t
  k_t = W_UK · c_t,  v_t = W_UV · c_t
推理时只需缓存 c_t，K/V 现算。代价是每次生成多两次小矩阵乘。

DeepSeek-V3 在此基础上用解耦的 RoPE 分支（Q 与 K 的 RoPE 部分单独处理）
并配合 FP8 训练，官方给出的 KV cache 相对 MHA 缩小到约 1/4 ~ 1/7。

**与本引擎的关系**：目前实现的是 MHA / GQA / MQA；MLA 在路线图中
（需要新的注意力内核 + KV 缓存布局，见 docs/OPTIMIZATION.md）。

**论文**：DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model*, arXiv:2405.04434 §2.1.1。""",
    },
    {
        "id": "acad-ce-ppl",
        "title": "学术：交叉熵、困惑度与 bits-per-byte",
        "group": "学术",
        "keywords": ["交叉熵", "perplexity", "ppl", "bpb", "loss"],
        "body": """**交叉熵**（逐 token）：L = −(1/N)·Σ log p(x_t | x_<t)

**困惑度**：PPL = exp(L) —— 可以读成「模型在每一步平均要在多少个词里猜」。
PPL 越低越好；跨数据集比较时注意分词器不同会改变 L 的基准。

**bits-per-byte**：BPB = L / (每 token 平均字节数) × log₂e
它不是按 token 算而是按**原始字节**算，所以**换分词器也能横比**，
是评测语言模型压缩能力的更公平指标。本引擎的 `eval` 同时输出
`val_loss` 与 `bits_per_byte`（见 `data.py` 的 bits_per_byte）。

**陷阱**：SFT 的 loss 只在「回答」token 上算（loss mask），与预训练
「全 token」的 loss 不可横向比较。

**论文**：Shannon 1948 的信息论定义；语言模型评测中的 BPB 用法见
*The Pile*（arXiv:2101.00027）等数据集论文。""",
    },
    {
        "id": "acad-adamw",
        "title": "学术：AdamW、权重衰减与 warmup+cosine",
        "group": "学术",
        "keywords": ["adamw", "优化器", "权重衰减", "warmup", "cosine", "梯度"],
        "body": """**Adam**：维护一阶动量 m_t 与二阶动量 v_t，
  m_t = β₁m_{t−1} + (1−β₁)g_t
  v_t = β₂v_{t−1} + (1−β₂)g_t²
  θ_t = θ_{t−1} − lr·m̂_t / (√v̂_t + ε)
（m̂、v̂ 是偏差修正后的值。）

**AdamW**（Loshchilov & Hutter 2019）：把 L2 正则从梯度里拿出来，
改成参数上的解耦衰减 θ ← θ − lr·λ·θ。它比 Adam+L2 泛化更好，
是当前 LLM 训练默认优化器。本引擎 `[train] weight_decay` 即 λ。

**warmup + cosine**：先线性升到峰值 lr（避免早期大梯度把参数打坏），
再余弦退火到 min_lr。本引擎参数：`warmup_iters`、`lr`、`min_lr`、
`lr_decay_iters`（默认等于 max_steps）。

**论文**：Kingma & Ba, *Adam*, arXiv:1412.6980；
Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, arXiv:1711.05101；
余弦退火见 Loshchilov & Hutter, *SGDR*, arXiv:1608.03983。""",
    },
    {
        "id": "acad-distributed",
        "title": "学术：数据并行、ZeRO 与 FSDP2",
        "group": "学术",
        "keywords": ["fsdp", "zero", "数据并行", "分片", "分布式", "显存"],
        "body": """**数据并行（DDP）**：每卡一份完整模型，各算各的 batch，
梯度 all-reduce 后一起更新。显存 = 模型 + 梯度 + 优化器状态（各一份）。

**ZeRO**（Rajbhandari et al. 2020）：把这三份状态按 stage 分片到各卡：
  Stage-1 优化器状态分片；Stage-2 再加梯度；Stage-3 再加参数。
每卡显存降到约 1/N（N=卡数），代价是通信量增加。

**FSDP2**（PyTorch 2.4+ 的 `fully_shard`）：参数按层分片（类似 ZeRO-3），
前向时按需 all-gather、反向时 reduce-scatter，是 PyTorch 官方推荐的
「单机多卡 / 多机」方案，可与 TP/PP/EP 组合。

**本引擎现状**：单卡训练循环已实现；`lingyu launch --backend fsdp|deepspeed`
只**生成启动脚本**（经实测能跑的 torchrun/DeepSpeed 配置），真正的
多卡训练循环还没接进引擎 —— 文档与 CSV 里都标了「实验性」，不假装能跑万亿。

**论文**：Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training
Trillion Parameter Models*, arXiv:1910.02054；FSDP 见 PyTorch 文档
`torch.distributed.fsdp.fully_shard`。""",
    },
    {
        "id": "acad-activation-recompute",
        "title": "学术：激活重计算（gradient checkpointing）省的是什么",
        "group": "学术",
        "keywords": ["激活重计算", "checkpointing", "省显存", "反向", "计算换显存"],
        "body": """**问题**：训练时反向传播要保留每层前向的中间激活，
层数×宽度×batch 一大，激活就吃掉大量显存（长上下文时尤其明显）。

**做法**：前向时只保留每层输入的「检查点」，丢弃中间激活；
反向到该层时用检查点**重新算一遍前向**，再算梯度。

**代价**：训练计算量约 +20%~40%（多一次前向），换显存约 −30%~60%。
这是「计算换显存」的标准手段，长上下文/大 batch 时非常划算。

**与本引擎**：目前没有在训练循环里开启；路线图里作为 `[train] grad_checkpoint`
选项（每层包 `torch.utils.checkpoint`）。本机单卡 8GB 跑 47.8M 模型不需要；
但换更大模型/更长上下文时会先加它。

**论文**：Chen et al., *Training Deep Nets with Sublinear Memory Cost*,
arXiv:1604.06174；PyTorch 文档 `torch.utils.checkpoint`。""",
    },
    {
        "id": "acad-fp8",
        "title": "学术：FP8 混合精度训练的收益与坑",
        "group": "学术",
        "keywords": ["fp8", "混合精度", "e4m3", "量化", "训练"],
        "body": """**格式**：FP8 有两种 IEEE 风格格式 —— E4M3（4 位指数、3 位尾数，
数值范围 ±448，精度高）与 E5M2（范围更大、精度低）。权重/激活常用 E4M3，
梯度常用 E5M2。

**用法**（DeepSeek-V3 的做法）：前向矩阵乘用 FP8，主权重与优化器状态
仍保持 BF16/FP32；每 2^k 个 token 做一次块级缩放（tile-wise scaling），
减少离群值带来的精度损失。官方报告训练吞吐约 2×、显存下降约 1/3。

**坑**：FP8 需要硬件支持（Hopper/Ada 及以上）；数值范围窄，缩放因子
选不好会溢出/下溢；收敛对超参更敏感，不能直接照搬 BF16 配置。

**与本引擎**：目前用 BF16 autocast（`[train] dtype = bf16`），不支持 FP8。
较新的 NVIDIA 显卡在硬件上支持 FP8，但 PyTorch 训练路径的成熟度有限，
因此列为路线图，而不是现在假装支持。

**论文**：DeepSeek-V3 技术报告 arXiv:2412.19437 §3.3；
Micikevicius et al., *FP8 Formats for Deep Learning*, arXiv:2209.05433。""",
    },
    {
        "id": "acad-lossless-compress",
        "title": "学术：权重/断点的无损压缩为什么只能省 ~16%",
        "group": "学术",
        "keywords": ["zipnn", "无损压缩", "权重", "断点", "熵"],
        "body": """**为什么难**：训练好的 fp32 权重尾数（23 bit）在统计上接近随机，
熵很高；能压缩的只有符号位与指数位——不同权重的指数分布集中，
所以「把字节按平面拆开再压」能省一些。

**ZipNN**（arXiv:2411.05239）的思路：利用 BF16/FP16 的「指数位集中 +
尾数随机」特性，只对指数位做熵编码，尾数原样存；对 BF16 权重官方
报告可省约 30%，对 FP32 收益明显更低（尾数占比 23/32，压不动）。

**本引擎实测**（47.8M 参数、含 Adam 一阶/二阶动量、fp32）：
断点 573.8 MB → 493.6 MB，**1.162×**，逐 bit 可逆；
其中模型权重 212.2 MB → 180.4 MB（1.18×），优化器状态 382.4 MB →
311.7 MB（1.23×）。优化器动量因为数值平滑，比模型权重更好压。

**真正大的收益不在「算法」，而在少存**：
  - 增量断点（存相对上一断点的 delta），一次训练多存档时可省 2~4×；
  - 只保留 best + 最新，定期清理中间档；
  - BF16 训练时按 BF16 存（这属于「训练精度选择」，不是无损压缩）。

**论文**：ZipNN: *Lossless Compression for AI Models*, arXiv:2411.05239；
Hershey et al., *Lossless Compression of Neural Network Components*, arXiv:2508.19263。""",
    },
    {
        "id": "acad-backend",
        "title": "学术：为什么「换显卡」比想象中容易（以及难在哪）",
        "group": "学术",
        "keywords": ["rocm", "amd", "昇腾", "npu", "musa", "跨平台", "后端"],
        "body": """**容易的部分**：PyTorch 把设备抽象成 `torch.device("cuda")` /
`torch.device("mps")` / …，模型与训练循环只要用 torch 算子写，换后端
基本不用改代码。AMD 的 ROCm 版 PyTorch 里设备**仍叫 "cuda"**
（`torch.version.hip` 有值），所以同一份代码可以直接跑 AMD 卡。

**难的部分**：
  1. **算子覆盖**：某个后端可能缺 fused kernel（如 F.scaled_dot_product_attention
     的 flash 实现），会回退到慢速路径，甚至报错；
  2. **编译**：`torch.compile`、自定义 CUDA extension 需要该后端的工具链；
  3. **驱动/安装**：国产卡（昇腾 CANN、摩尔线程 MUSA）需要各自的
     PyTorch 插件（`torch_npu`、`torch_musa`）与系统驱动；
  4. **精度差异**：BF16/FP8 支持矩阵不同，同一配置可能需要改 dtype。

**本引擎**：核心只用 torch 标准算子（SDPA、RMSNorm 手写、AdamW 用
torch.optim 的 fused 实现并带 fallback），新增 `src/lingyu/devices.py`
做后端探测与 auto 选择；`lingyu doctor` 会列出当前可用的后端，
不会假装支持没装的后端。

**参考**：PyTorch PrivateUse1 后端接入教程（官方 docs）；
ROCm 文档；华为昇腾 `torch_npu`；摩尔线程 `torch_musa`。""",
    },
]
