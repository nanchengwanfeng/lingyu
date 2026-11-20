# -*- coding: utf-8 -*-
"""帮助页的「百科」正文：概念、参数、训练技巧、排查、术语。

和 `help_content.py` 里的 14 篇教程分开维护 —— 那边是「手把手」，
这边是「查字典」。每篇 = {id, title, group, keywords, body}。

写作口径：面向零基础，但数字和参数必须和本引擎一致；不写废话，
不提「作为一个 AI」。引擎没实现的东西（DPO/LoRA/量化/多卡）一律标
「计划中/未实现」，别让用户按文档跑空。
"""

EXTRA = [
    # ================================================================ 入门
    dict(id="x_time", title="训练一个模型要多久", group="入门",
         keywords=["时间", "多久", "耗时", "速度", "时长"],
         body="""**看两个数**：每步耗时 × 总步数。

本机实测（8GB 显存显卡，词表 8192）：

| 配置 | 参数量 | 训练速度 | 3000 步 |
|---|---|---|---|
| small | 13.08M | 约 27 分钟 | loss 8 左右降到 3.5 |
| medium | 47.8M | 约 9 分钟（batch 16） | — |

**别拿「步数」当进度条**：一次「步」处理 `batch_size × block_size` 个
token。`small` 默认 8 × 512 = 4096 token/步；`medium` 是 16 × 512 = 8192。

想快点：先调小 `block_size`（512→256），再调小 batch；调层数/宽度是最后手段，
那会直接改变模型能力。"""),

    dict(id="x_vram_oom", title="显存不够怎么办", group="入门",
         keywords=["显存", "OOM", "爆显存", "装不下", "估算"],
         body="""按**性价比**依次试：

1. **batch_size 减半**（最有效，几乎线性省激活显存）；
2. `block_size` 512 → 256（序列越短，激活越小）；
3. `grad_accum` 调大，补偿 batch 变小（不省显存，只省「等效 batch」）；
4. 关掉 `sample_while_train`（采样会额外占几百 MB）；
5. 最后才减 `n_layer` / `n_embd` —— 改结构等于换模型，旧断点作废。

引擎的估算器在训练页右侧：「合计」是权重 + 优化器 + 梯度 + 激活的粗估，
误差可达数倍，仅供参考。红色「装不下」时别硬开，先按上面五条改。"""),

    dict(id="x_data_from", title="语料从哪来", group="入门",
         keywords=["数据", "语料", "数据集", "下载", "小说", "alpaca"],
         body="""三条路：

- **自己的文本**：txt / md / json / jsonl 放一个文件夹即可。本机 `D:\\ai\\语料准备`
  下有 10 本书（三体、红楼梦…）和转换好的对话数据。
- **对话数据**：`chat/fetch_sft_data.py` 会从 ModelScope 拉
  `AI-ModelScope/alpaca-gpt4-data-zh`（5 万条中文指令，CC BY-NC 4.0，
  只限非商用），过滤后转成 JSONL。
- **网上公开语料**：注意许可证。CC BY-NC 的不能商用；GPL 数据混进训练集
  不会「传染」模型，但数据本身不能再分发。

**别一上来就堆数据**：先把 1 万条跑通、确认 loss 会降，再加量。"""),

    dict(id="x_token", title="什么是 token", group="入门",
         keywords=["token", "词元", "分词", "压缩比", "词表"],
         body="""模型不读汉字，读 **token**：把文本切成小片段后编号。

- 中文一个字通常 1 个 token；英文一个常见词 1 个 token；
- 词表大小（`vocab_size`）就是「最多有多少种片段」，本引擎默认 8192；
- 「压缩比」= 字符数 ÷ token 数，中文语料一般 1.2～1.8，越高越省算力；
- 8192 词表大约能覆盖常用汉字 + 常见词，生僻字会被拆成 2 个 token。

训练前会打印语料的 token 总数；GUI 语料页也能看到。"""),

    dict(id="x_judge", title="怎么判断训练成功", group="入门",
         keywords=["loss", "判断", "成功", "看什么", "验证"],
         body="""看三个层次，按顺序：

1. **loss 在降**：预训练从 8+ 降到 3 左右算正常；验证 loss 不升太多
   （升 = 过拟合）。
2. **续写像中文**：训练 1000 步后「续写」页应该能接出通顺片段，
   而不是乱码。乱码先查分词器指纹。
3. **SFT 后能答**：问「你是谁」，13M 模型答得空泛是正常的 —— 它学的是
   回答格式，不是知识。要看知识得上 medium/wide + 大量语料。

**看 loss 别看单个点**，看最近 20 步均值（`history.csv` 里
`recent_mean` 那列）。"""),

    # ================================================================ 概念
    dict(id="x_transformer", title="Transformer 是什么", group="概念",
         keywords=["transformer", "架构", "decoder", "注意力"],
         body="""灵语用的是 **decoder-only Transformer**：给定前文，预测下一个 token。

一层里只有三个东西：

1. **注意力**：每个 token 去「看」前面所有 token，决定该关注谁；
2. **前馈网络 FFN**：对每个位置单独做一次升维→激活→降维；
3. **残差 + 归一化**：让深层网络能训得动。

GPT 系列、Llama、Qwen 都是这个结构，差别在位置编码、激活函数、
归一化位置、注意力变体这些细节。"""),

    dict(id="x_attention", title="注意力机制（Q/K/V）", group="概念",
         keywords=["attention", "注意力", "qkv", "query", "key", "value"],
         body="""把每个 token 变成三个向量：

- **Query**：我在找什么；
- **Key**：我能被谁找到；
- **Value**：找到我之后给你什么。

分数 = Q·Kᵀ/√d，softmax 后加权求和 V。实现上就是几次矩阵乘，
`softmax(QK^T/√d) V`；引擎用 PyTorch 的 `scaled_dot_product_attention`，
会自动走 FlashAttention 或内存高效实现。

训练时「能看到」的范围由 `block_size` 决定：它只看前 block_size 个 token。"""),

    dict(id="x_mha", title="多头注意力与 GQA/MQA", group="概念",
         keywords=["mha", "gqa", "mqa", "多头", "n_head", "n_kv_head"],
         body="""**多头**：把隐藏维切成几份，每份独立算一次注意力，再拼回来。
`n_head` 是头数；`n_head × head_dim = n_embd`，所以 `n_embd` 必须能被
`n_head` 整除。

**GQA**（分组查询）：Q 有 n_head 个，K/V 只留 `n_kv_head` 组共享。
好处是推理时 KV cache 小、生成快；`n_kv_head = n_head` 退化为 MHA，
`n_kv_head = 1` 退化为 MQA。

本引擎默认 `n_kv_head=2`，这也是 Llama 2/3 的常用做法。"""),

    dict(id="x_rope", title="位置编码 RoPE", group="概念",
         keywords=["rope", "位置编码", "位置", "旋转"],
         body="""Transformer 本身不知道顺序，得把位置信息加进去。

**RoPE**（旋转位置编码）把位置编码成「旋转角度」乘到 Q/K 上：
两个 token 的相对位置决定它们的点积相位。优点是能外推到比训练更长的
上下文（效果有限），也不需要额外的位置向量表。

本引擎在注意力里对 Q/K 做旋转；`rope_base`（默认 10000）控制旋转频率，
改大能让模型更适应长文本，但得重新训练才有效。"""),

    dict(id="x_ffn", title="前馈网络与 SwiGLU", group="概念",
         keywords=["ffn", "前馈", "swiglu", "激活", "d_ff"],
         body="""FFN 是每层参数最多的部分（约占 2/3）：

`x → 升维 → 激活 → 降维 → x`

传统用 ReLU + 4 倍宽度；SwiGLU 用三组权重（门控），激活是
`swish(xW1) ⊙ xW3`。本引擎默认 `d_ff = 0` 表示自动取
`8/3 × n_embd`（对齐 8 的倍数），这是 SwiGLU 的常用比例。

想省参数：调小 `d_ff`；想加容量：调大它比加层数更划算（同等参数下更宽
通常更好训）。"""),

    dict(id="x_residual", title="残差连接", group="概念",
         keywords=["residual", "残差", "skip connection"],
         body="""`y = x + f(x)`：每个子层输出都加回输入。

作用：梯度可以直接沿 `x` 回流，几十层也能训；没有残差，深层网络
梯度会消失或爆炸。

本引擎每个注意力、每个 FFN 外层都有残差；归一化放在子层**之前**
（Pre-Norm），这是现代 LLM 的标准做法，比 Post-Norm 更稳。"""),

    dict(id="x_norm", title="层归一化（RMSNorm / LayerNorm）", group="概念",
         keywords=["norm", "归一化", "rmsnorm", "layernorm"],
         body="""把每个 token 的向量缩放到稳定范围，防止数值越训越大。

- **LayerNorm**：减均值、除标准差，再乘 γ、加 β；
- **RMSNorm**：不减均值，只除均方根，少一次减法，更快。Llama 用这个。

本引擎可选 `norm_type`（rmsnorm / layernorm），默认 RMSNorm。
归一化在残差分支内部、子层之前（Pre-Norm）。"""),

    dict(id="x_embedding", title="词嵌入 Embedding", group="概念",
         keywords=["embedding", "词嵌入", "tie_weights", "权重共享"],
         body="""词表里每个 token 对应一个 `n_embd` 维向量，就是 embedding 表。

`tie_weights = true` 时输出层直接复用 embedding 权重（转置相乘），
省 `vocab_size × n_embd` 参数，小模型常用；`false` 则单独一套输出矩阵。

词表 8192、n_embd 256 时，embedding 约 2.1M 参数 —— 13M 的 small 里
它占了一大块，`tie_weights` 能明显省参数。"""),

    dict(id="x_context", title="上下文窗口（block_size）", group="概念",
         keywords=["block_size", "上下文", "窗口", "长文本"],
         body="""`block_size` = 模型一次能看多少个 token。

- 训练时每条样本截断/填充到这个长度；
- 推理时超过窗口的部分会被滚出（聊天页会按 token 裁剪历史）；
- 越长越吃显存（注意力是 O(n²)，KV cache 线性增长）。

本机 8GB 建议：13M 用 512～1024；47.8M 用 512。
改 block_size **不需要**重训全部，但模型没见过更长上下文，效果会打折。"""),

    dict(id="x_bpe", title="BPE 分词器怎么工作", group="概念",
         keywords=["bpe", "分词器", "tokenizer", "合并", "训练分词器"],
         body="""BPE（字节对编码）从单个字节开始，反复把「最常见的相邻对」合并成
新 token，直到词表满。

本引擎 `prepare` 阶段会：统计语料 → 学 BPE 合并规则 → 生成
`tokenizer.txt`（词表 + 合并规则 + 特殊 token）→ 把语料编码成
`train.bin` / `val.bin`。

**分词器和断点必须配套**：引擎用 `fingerprint()`（词表 + 合并规则的哈希）
核对，对不上直接拒绝开跑 —— 这种不匹配 loss 照降、输出全乱码，
是全项目最隐蔽的故障。"""),

    dict(id="x_vocab", title="词表大小怎么选", group="概念",
         keywords=["vocab", "词表", "vocab_size"],
         body="""- **小了**：常见词被拆成多个 token，压缩比低、序列变长、训练变慢；
- **大了**：embedding 参数变多，小模型里占比过高，尾部 token 训练不足。

本引擎默认 8192，适合中文小说 + 对话；语料特别杂（多语言/代码）可以
调到 16384；13M 这种小模型别超过 8192。

改词表 = 重跑 `prepare` = 所有 token ID 变化 = 旧断点全部作废。"""),

    # ================================================================ 训练
    dict(id="x_pretrain", title="预训练到底在学什么", group="概念",
         keywords=["预训练", "pretrain", "学什么", "下一 token"],
         body="""目标只有一个：**给定前文，预测下一个 token**。

模型为了把这个任务做好，会自己学出词法、句法、事实关联、风格……
所以数据越多越杂，它「像」的语言就越广。

代价：它只学会「接着写」，不会「回答问题」。要会答，必须再做 SFT
（监督微调），用「问-答」对教它对话格式。"""),

    dict(id="x_loss", title="交叉熵损失怎么读", group="概念",
         keywords=["loss", "交叉熵", "ce", "损失"],
         body="""`loss = -log(模型给正确 token 的概率)`，单位是 nat。

- loss 8.3 ≈ 词表 4096 的随机水平（log 4096 ≈ 8.3）；
- loss 3.0 ≈ 平均正确 token 概率 5%；
- loss 2.0 ≈ 13.5%；
- loss 1.0 ≈ 37%。

每降 0.1 都很难，越到后面越慢。**看趋势不看绝对值**：不同词表、不同
语料的 loss 不能横向比。"""),

    dict(id="x_ppl", title="困惑度 PPL 是什么", group="概念",
         keywords=["ppl", "困惑度", "perplexity"],
         body="""`PPL = exp(loss)`，可以理解成「模型在每一步平均犹豫多少个 token」。

- loss 4.39 → PPL 80.6（small 训练早期）
- loss 3.0 → PPL 20
- loss 2.0 → PPL 7.4

PPL 只在**同一词表、同一验证集**之间可比；换语料就没意义了。"""),

    dict(id="x_lr", title="学习率、预热与余弦衰减", group="参数",
         keywords=["learning_rate", "lr", "学习率", "warmup", "预热", "衰减"],
         body="""学习率决定每步走多远：太大发散（loss 变 nan 或剧烈震荡），
太小训得慢甚至卡住。

本引擎用 **warmup + 余弦衰减**：

- 前 `warmup_iters` 步从 0 线性升到 `lr`，避免一开始就把权重带崩；
- 之后按余弦曲线降到 `min_lr`。

经验值：13M 预训练 `lr=1e-3`，SFT `lr=3e-4`；47.8M 预训练 `lr=6e-4`～`8e-4`。
改结构后学习率要重调，别照抄。"""),

    dict(id="x_clip", title="梯度裁剪 grad_clip", group="参数",
         keywords=["grad_clip", "梯度裁剪", "梯度爆炸"],
         body="""每一步算完梯度后，如果梯度范数超过 `grad_clip`，就整体按比例缩小。

作用：防单个坏 batch 把参数带飞（loss 突然飙升/变 nan）。
常用值 1.0；训练稳定时可以关掉，遇到偶发尖峰再打开。

日志里的 `gnorm` 就是裁剪前的梯度范数；长期远大于 1 说明学习率偏大。"""),

    dict(id="x_accum", title="梯度累积 grad_accum", group="参数",
         keywords=["grad_accum", "梯度累积", "大 batch"],
         body="""显存太小放不下大 batch 时，用「攒 K 个小 batch 再更新一次」模拟
大 batch：`等效 batch = batch_size × grad_accum`。

代价：同样 token 数下更新次数变少，训练可能变慢；而且 BatchNorm
（本引擎没用）那类统计会受影响 —— Transformer + LayerNorm 基本没问题。

示例：显存只够 batch 4，想要等效 32 → `batch_size=4, grad_accum=8`。"""),

    dict(id="x_bf16", title="混合精度（bf16）", group="概念",
         keywords=["bf16", "fp16", "amp", "混合精度", "autocast"],
         body="""前向/反向用 16 位浮点，权重和优化器状态保持 fp32：

- 省显存（激活减半）、提速（Tensor Core）；
- **bf16** 指数范围大，不需要 loss scaling，几乎不溢出；
- 老卡（图灵之前）不支持 bf16，退回 fp16 需要 loss scaling。

本引擎默认 `dtype=bf16`；CUDA 不可用或卡不支持时会自动走 fp32。
日志里会打印实际 dtype。"""),

    dict(id="x_vram_parts", title="训练显存都被谁吃了", group="概念",
         keywords=["显存", "组成", "优化器", "激活", "权重"],
         body="""四块，按大致比例：

1. **权重**：参数量 × 2 字节（bf16）；
2. **梯度**：参数量 × 2 字节；
3. **优化器状态**（AdamW）：参数量 × 8 字节（fp32 动量 + 方差）；
4. **激活**：随 batch × block_size × 层数增长，训练时最不可控。

所以 13M 模型训练峰值能到 2.7GB，而权重本身只有 26MB。
省显存优先动激活（batch / block_size），优化器状态基本动不了。"""),

    dict(id="x_resume", title="断点续训怎么做", group="训练",
         keywords=["resume", "续训", "断点", "checkpoint", "ckpt"],
         body="""命令行：`lingyu train --config configs/small.ini --resume`，
或 GUI 训练页点「续训」。

续训会恢复：权重、优化器状态、step、best_val、RNG 状态（保证可复现）。
**结构必须一致**：改了 n_layer / n_embd / block_size 就会报
「模型结构和断点对不上」—— 这是保护，不是 bug。

每个断点旁边有 `<文件名>.json` 小档案（步数、结构、tokenizer 指纹），
断点页直接读它，不用整个加载 `.pt`。"""),

    dict(id="x_overfit", title="过拟合怎么判断", group="训练",
         keywords=["过拟合", "overfit", "验证", "正则", "dropout"],
         body="""训练 loss 一直降，但**验证 loss 先降后升** = 过拟合。

对策（按性价比）：
1. 加数据（最有效）；
2. 调小 `dropout` —— 不对，是**调大** dropout（0.05 → 0.1）；
3. 早点停（取验证 loss 最低的断点，引擎自动存 `best.pt`）；
4. 减小模型（最后手段，因为等于换模型）。

语料只有几 MB 时，13M 模型几千步就可能过拟合。"""),

    dict(id="x_val", title="验证集是怎么来的", group="训练",
         keywords=["验证集", "val", "val.bin", "划分"],
         body="""`prepare` 时按 `val_ratio`（默认 2%）从语料末尾切一段做验证集，
其余做训练集。验证集**不参与训练**，只用来测「模型没见过的话能不能预测好」。

验证 loss 每 `eval_interval` 步算一次（默认 500）。eval 会占一点显存和时间，
训练稳了可以把间隔调大。

注意：SFT 的验证 loss 只统计回答位置，和预训练 loss 口径不同，不能直接比。"""),

    dict(id="x_sft_what", title="SFT 和预训练的区别", group="后训练",
         keywords=["sft", "微调", "区别", "问答"],
         body="""| | 预训练 | SFT |
|---|---|---|
| 数据 | 连续文本 | 一问一答 |
| 目标 | 预测下一个 token | 只学「回答」部分 |
| 学到的 | 语言/知识 | 对话格式/风格 |
| loss 口径 | 全部 token | 只统计回答位 |

SFT 不改模型结构，只是换数据 + 用模板把「问题」mask 掉。
所以 SFT 断点可以继续当预训练断点用（权重兼容），反之亦然。"""),

    dict(id="x_loss_mask", title="loss mask 是什么", group="后训练",
         keywords=["mask", "loss mask", "忽略", "ignore_index"],
         body="""SFT 时把「不该学的位置」的 target 设成 -1，交叉熵自动跳过。

本引擎的做法：
- 渲染成 `用户：…\\n助手：…`；
- 「用户：…」和模板标记位的 target = -1；
- 只对「助手回答」的 token 算 loss。

这样模型学的是「怎么答」，不会把问题的措辞也背下来。
`plain` 模板用中文字段标记，不需要特殊 token；chatml/qwen 需要，
用之前先确认分词器里有那些特殊 token。"""),

    dict(id="x_template", title="对话模板 plain / chatml / qwen", group="后训练",
         keywords=["template", "模板", "plain", "chatml", "qwen"],
         body="""三种：

- **plain（默认）**：`用户：…\\n助手：…`，纯文字，任何断点都能用；
- **chatml**：`<|im_start|>user ... <|im_end|>`，需要词表里有特殊 token；
- **qwen**：Qwen 系列的 `<|im_start|>` + 默认 system prompt。

新建的分词器自带 lo 特殊 token；老分词器用 chatml 会**报错**并给出
两条出路（换 plain 或重跑 prepare），不会静默乱码。

`system_prompt` 只在 qwen 模板下自动注入，plain/chatml 需要自己写进数据。"""),

    dict(id="x_chatdata", title="对话数据长什么样", group="后训练",
         keywords=["数据格式", "jsonl", "alpaca", "messages", "对话数据"],
         body="""JSONL，一行一条，引擎认四种格式：

1. `{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}`
2. `{"conversations":[{"from":"human","value":"…"},{"from":"gpt","value":"…"}]}`
3. `{"instruction":"…","input":"…","output":"…"}`（Alpaca）
4. `{"prompt":"…","response":"…"}`

读取时会校验：必须有问答对；空回答/空问题会跳过并计 warning。
`chat/fetch_sft_data.py` 转出来的就是第 3 种（过滤到 ≤700 字符）。"""),

    dict(id="x_dpo", title="DPO 是什么（计划中）", group="后训练",
         keywords=["dpo", "偏好", "rlhf", "计划"],
         body="""DPO（直接偏好优化）：给模型看「好回答/坏回答」成对数据，
让它更倾向好回答。比 RLHF 简单，不用奖励模型。

**状态：本引擎尚未实现**。配置里 `[post] method` 目前只支持 `sft`；
`train/dpo.py` 不存在，等 P7 阶段。文档里的 DPO 参数属于预留，
现在改了也不会生效。"""),

    dict(id="x_lora", title="LoRA 是什么（计划中）", group="后训练",
         keywords=["lora", "低秩", "微调", "计划"],
         body="""LoRA：冻结原权重，在旁边加一对低秩矩阵 `A(m×r)·B(r×n)`，
只训这两个小矩阵。r 通常 8～64，可训练参数能降到 1% 以下。

适合「一张卡微调大模型」，但小模型上收益不明显。

**状态：本引擎的 LoRA 尚未实现**（GUI 的 SFT 页有入口，
`lora_r` 等参数已预留，但后端会走全量 SFT）。等 P7。"""),

    dict(id="x_moe", title="MoE 是什么", group="概念",
         keywords=["moe", "专家", "混合专家", "路由"],
         body="""MoE = 多个「专家 FFN」+ 一个路由器：每个 token 只走其中
top-k 个专家（本引擎 k=2），所以**总参数大、每 token 计算量小**。

代价：
- 显存要装下**所有**专家（总参数）；
- 生成慢（路由 + 多个专家矩阵乘）；
- 训练要加负载均衡损失，否则专家会「塌缩」（大家只用那两三个）。

本引擎 `configs/moe.ini` 是 53.5M 总参数 / 34.6M 激活的示例：
训练比同参数稠密模型快，生成慢一倍。"""),

    dict(id="x_moe_route", title="专家路由与负载均衡", group="概念",
         keywords=["路由", "router", "负载均衡", "aux loss", "z-loss"],
         body="""路由器对每个 token 输出「去哪个专家」的分数，取 top-k。

问题：如果少数专家被大量使用，其它专家训不到 → 容量浪费。
所以训练时加两个辅助项：

- **aux loss**（负载均衡）：鼓励各专家被用的概率均匀（系数 `aux_loss_coef`）；
- **z-loss**（路由稳定）：惩罚路由 logits 过大，防数值发散（`router_z_loss_coef`）。

这两个系数调大能让专家更均衡，但会稍微影响主任务 loss。"""),

    dict(id="x_moe_why", title="为什么 MoE 训练快、生成慢", group="概念",
         keywords=["moe", "快", "慢", "激活参数"],
         body="""训练时每 token 只算 2 个专家 → FLOPs 低 → 快；
但生成时要一个个 token 算，路由 + 多专家矩阵乘的**延迟**摊不开 →
tok/s 反而低（实测 moe 114 tok/s vs small 351 tok/s）。

另外训练要额外算负载均衡，也有点开销。

结论：想「同等算力下更强」用 MoE；想「生成快」用稠密小模型。"""),

    # ================================================================ 参数
    dict(id="x_nlayer", title="n_layer（层数）", group="参数",
         keywords=["n_layer", "层数", "深度"],
         body="""堆几层 Transformer。层数越多，模型能学的关系越复杂，但：

- 参数线性增长（每层 attn + FFN）；
- 训练变慢，显存增加；
- 太深的小宽度模型反而不如「浅而宽」。

经验：13M 用 12～16 层，47.8M 用 10 层（`medium.ini`），
再大就 24/32 层。改层数 = 旧断点不能用。"""),

    dict(id="x_nembd", title="n_embd（宽度）", group="参数",
         keywords=["n_embd", "宽度", "hidden", "隐藏维"],
         body="""每层内部向量的维度。对参数量影响最大：FFN 参数约
`n_embd² × 8/3 × 2`（SwiGLU 三组权重、d_ff≈8/3·n_embd）。

所以「加宽」比「加深」更烧参数，通常也更有效（同等参数下更宽好训）。
`n_embd` 必须能被 `n_head` 整除；必须是 8 的倍数（实现里按 8 对齐）。

13M：256；47.8M：640；79M：768。"""),

    dict(id="x_nhead", title="n_head 怎么定", group="参数",
         keywords=["n_head", "头数", "head_dim"],
         body="""`head_dim = n_embd / n_head`，常见 64～128。

- 头太多 → 每个头维度太小，注意力表达力下降；
- 头太少 → 一个头管太多关系，学不好。

经验：`n_embd=256` → n_head=8（dim 32，偏小但省参数）；
`n_embd=640` → n_head=10（dim 64，标准）。
改 n_head 不改变总参数量（只改 KV 形状），但仍需重训。"""),

    dict(id="x_nkv", title="n_kv_head（GQA 比例）", group="参数",
         keywords=["n_kv_head", "kv头", "gqa", "kv cache"],
         body="""K/V 的头数，必须能整除 `n_head`。
`n_kv_head = n_head` → MHA；`= 1` → MQA；中间值 → GQA。

好处：推理 KV cache 小 `n_head/n_kv_head` 倍。13M 默认 2/8 = 1/4 的 KV 显存。

本机聊天时 KV cache 影响不大（上下文短），主要是省显存 + 生成快一点点。
改这个不改参数量，但改了权重形状 → 旧断点不兼容。"""),

    dict(id="x_block", title="block_size 怎么选", group="参数",
         keywords=["block_size", "上下文", "序列长度"],
         body="""一次训练的序列长度。

- 512：最省显存，适合 8GB 卡和小模型；
- 1024：能学稍长依赖，显存约 ×1.5～2；
- 2048+：8GB 基本别想（注意力 O(n²) + 激活）。

注意 `prepare` 不按 block_size 切数据，是训练时按样本截断的；
所以同一个 data 目录可以配不同 block_size 训练，但断点不通用。"""),

    dict(id="x_dropout", title="dropout 调多少", group="参数",
         keywords=["dropout", "正则", "防过拟合"],
         body="""训练时随机丢掉一部分激活，防过拟合。

- 13M 小模型 + 少语料：0.05～0.1；
- 大数据/大模型：0～0.05（数据够多就不太需要）；
- 推理时自动关闭。

调大 dropout → 训练 loss 会更高但验证/生成更稳。调这个要重训才看得出来。"""),

    dict(id="x_batch", title="batch_size 与显存/速度", group="参数",
         keywords=["batch_size", "批大小"],
         body="""一次前向的样本数。越大：

- 梯度越稳（噪声小）；
- 显存线性增长（激活部分）；
- 同样步数下处理 token 更多（速度看着快）。

但太大会「一步跨过太多细节」，小模型+少数据时反而不如小 batch + 梯度累积。

13M 预训练：8～16；47.8M：8～16；SFT 常用 8。
显存不够先减它。"""),

    dict(id="x_maxsteps", title="max_steps 怎么设", group="参数",
         keywords=["max_steps", "步数", "训练多久"],
         body="""总训练步数。经验值（本机）：

- small 预训练：3000 步（约 1.1 epoch，27 分钟）；
- medium 预训练：3000 步（约 0.7 epoch，9 分钟）；
- SFT：3000～6000 步（数据 4.8 万条，约 1～2 epoch）。

**别只看步数**：`步数 × batch × block_size` 才是见过的 token 数。
一般 1～3 个 epoch 比较合适；太多会过拟合。"""),

    dict(id="x_out_dir", title="out_dir 与断点文件", group="参数",
         keywords=["out_dir", "输出", "ckpt.pt", "best.pt", "断点"],
         body="""训练产物目录，默认 `out/<配置名>`。

里面会有：

- `best.pt`：验证 loss 最低的断点（推荐用它）；
- `ckpt.pt`：最近一次断点（续训用）；
- `best.pt.json` / `ckpt.pt.json`：小档案（步数/结构/指纹，断点页用）；
- `history.csv`：每步/每次验证的记录（监控页、分析页用）；
- `train.log`：原始日志。

**别手动改 `.pt`**；要换目录直接改配置里的 `out_dir`。"""),

    # ================================================================ 进阶
    dict(id="x_fsdp", title="多卡训练：FSDP2", group="进阶",
         keywords=["fsdp", "多卡", "分布式", "torchrun", "并行"],
         body="""FSDP（Fully Sharded Data Parallel）把参数/梯度/优化器状态切到各卡上，
每层计算前 all-gather、算完释放。

**本引擎状态**：`lingyu launch --config x.ini` 能生成 `launch_fsdp.sh`
（torchrun 命令），但训练循环目前**还是单卡实现** —— 脚本能跑，但不会
真正分片。诚实标注：多卡要等后续版本；单卡 8GB 先用 small/medium。

需要多少卡？`lingyu plan` 会算（54GB 估算 → 约 10 张卡）。"""),

    dict(id="x_deepspeed", title="DeepSpeed 与 Megatron 适配", group="进阶",
         keywords=["deepspeed", "megatron", "适配", "启动脚本"],
         body="""引擎不自己实现这些框架，只做**参数翻译 + 启动脚本生成**：

- `launch --backend deepspeed` → `launch_deepspeed.sh` + `ds_config.json`；
- `launch --backend megatron` → 参考模板（引擎不实现 Megatron 训练）。

理由：万亿参数不是本机能训的；假装能跑没有意义。真要上集群，
把生成的脚本交给对应框架，参数（层数/宽度/并行度）已经按配置翻译好。"""),

    dict(id="x_parallel", title="数据并行/张量并行/流水线并行", group="进阶",
         keywords=["数据并行", "张量并行", "流水线", "tp", "pp", "dp"],
         body="""- **DP**：每卡一份完整模型，切数据。显存不省，只提速。
- **TP**：把一层矩阵横竖切开，多卡算一层。通信量大，适合单机 NVLink。
- **PP**：把层按 stage 分到不同卡，前向像流水线。适合超大模型。
- **FSDP/ZeRO**：切参数/梯度/优化器状态，介于 DP 和 TP 之间。

本引擎 `[parallel]` 段预留了这些参数（tp/pp/dp 等），但只有 `local`
后端真正实现训练；其余只生成脚本。"""),

    dict(id="x_estimate", title="显存估算器怎么算", group="进阶",
         keywords=["估算", "显存", "plan", "公式"],
         body="""引擎里唯一的估算是 `src/lingyu/backends/plan.py`：

- 权重：参数量 × 2（bf16）；
- 优化器：参数量 × 8（AdamW fp32 动量+方差）；
- 梯度：参数量 × 2；
- 激活/其他：按 batch × block_size × 层数 × 宽度 估，再乘 1.2 余量；
- 可用显存先扣 15%（桌面/浏览器占用）。

误差可达数倍（实测峰值常比估算高 23%～45%），所以结论分三档：
「够」「不够」「不知道（没探测到显存）」。"""),

    dict(id="x_quant", title="量化（INT8/INT4）与 GGUF", group="进阶",
         keywords=["量化", "int8", "int4", "gguf", "llama.cpp"],
         body="""量化 = 用更低精度存权重（int8 省 2 倍，int4 省 4 倍），
推理时反量化回浮点算。

**状态：本引擎未实现量化导出**，也没有 GGUF / llama.cpp 适配。
`export --format gguf` 之类的命令不存在（计划 P4/P7 做 HF 导出）。

想在本地跑别家量化模型，请用 LM Studio / llama.cpp，别指望灵语。"""),

    dict(id="x_hf_export", title="导出成 HF 格式（计划）", group="进阶",
         keywords=["导出", "huggingface", "safetensors", "model card"],
         body="""计划 P4：把灵语断点转成 `config.json` + `model.safetensors` +
`tokenizer.json`，让 Transformers 能直接加载。

**当前状态：未实现**。现在只有灵语自己的 `.pt` 断点。
想换框架用，只能自己按 `model.py` 的结构写转换脚本（模型是标准
decoder-only，不难，但没现成工具）。"""),

    # ================================================================ 排查
    dict(id="x_cuda", title="CUDA 不可用怎么办", group="排查",
         keywords=["cuda", "显存", "驱动", "torch", "cpu"],
         body="""`lingyu doctor` 会区分两种：

1. **装了 CPU 版 torch**：`torch.__version__` 里没有 `+cuXXX`，
   重装 GPU 版即可（启动器会自动重试镜像）。
2. **驱动太老 / 显卡不支持**：更新驱动；8GB 显存显卡 需要 cu128 以上的 wheel。

PyTorch 需要 2.14+cu130 这类匹配版本。GUI 概览页也能看 PyTorch 版本；
状态栏「PyTorch 未就绪」= 引擎环境里没装好。"""),

    dict(id="x_fingerprint", title="分词器指纹不匹配", group="排查",
         keywords=["指纹", "fingerprint", "分词器", "断点对不上", "乱码"],
         body="""报错形如「分词器与断点不匹配」。

原因：断点训练时用的 tokenizer 和现在 data 目录里的不是同一份
（例如重跑过 prepare，或者把 data1 和 runtime\\data 混用）。

**为什么危险**：不匹配时不报错、loss 照降、输出全是乱码 —— 所以引擎
用词表+合并规则的哈希强制拦截。

解决：用「与断点配套」的 data 目录重跑；或者换一个匹配的断点。
没有捷径，token ID 体系一变就全变了。"""),

    dict(id="x_loss_up", title="loss 不降 / 变成 nan", group="排查",
         keywords=["loss", "nan", "不降", "发散", "学习率"],
         body="""按顺序查：

1. **学习率太大**：先减半再试；nan 基本是它。
2. **数据有问题**：空文件、二进制混进语料、token 全是 0 → `prepare` 会统计，
   看语料 token 总数是否异常小。
3. **分词器/断点不匹配**：见指纹那条。
4. **loss mask 没生效**（SFT）：验证 loss 只统计回答位，若全是 -1 会是 nan。
5. **bf16 在极老卡上溢出**：换 fp32 试。

先看 `train.log` 前 20 行有没有 warning。"""),

    dict(id="x_garbage", title="训练后输出乱码", group="排查",
         keywords=["乱码", "输出", "乱", "方框"],
         body="""99% 是**分词器与断点不配套**。

其它可能：
- 模型还没训够（100 步以内基本是乱码，正常）；
- 聊天页没选 SFT 断点，用了纯预训练断点（它只会续写，不会答）；
- 聊天页给的 prompt 带特殊 token，而模板不匹配。

`lingyu chat --ckpt <路径>` 命令行先试；命令行也乱码就与 GUI 无关。"""),

    dict(id="x_slow", title="训练很慢怎么办", group="排查",
         keywords=["慢", "速度", "tok/s", "GPU 利用率"],
         body="""先确认瓶颈：

- GPU 利用率低（<50%）→ 数据加载/CPU 瓶颈。加大 `num_workers`（若支持）、
  或把语料放在 SSD；
- 显存打满且速度低 → batch/block 太大，减小；
- 速度正常但「感觉慢」→ 看 tokens/步：8×512=4096 token/步，3000 步 = 1229 万 token。

本机 baseline：small 25k tok/s、medium 25k、moe 46k、wide 22k。
比这慢 3 倍以上再查环境。"""),

    dict(id="x_json_extract", title="JSON 语料抽不出正文", group="排查",
         keywords=["json", "提取", "体检", "正文", "语料"],
         body="""`prepare` 默认**保留 JSON 原文**（结构不动）；要抽正文得勾选
「抽 JSON 正文」或加 `--json-extract`。

抽取规则：优先 `text/content/instruction/output/response/answer/question/
prompt/completion/messages/value` 等字段；跳过 `id/url/role/from/timestamp` 等。
如果 JSON 里全是元数据，会抽空 → 引擎输出「⚠ 跳过：没从 JSON 里抽出正文」
并回退成原文，不会静默丢文件。

语料页「体检」会按文件列出：`utf-8 / JSON 提取 123 字符（2 条记录）`。"""),

    dict(id="x_config_missing", title="找不到配置文件", group="排查",
         keywords=["配置", "config", "找不到", "路径"],
         body="""报「找不到配置」时，先看它到底在找哪：

- 相对路径一律相对**项目根**（`paths.home()`），不是当前目录；
- 打包版根目录是 `dist\\LingYu\\runtime\\`，配置在 `runtime\\configs\\`；
- 用 `lingyu doctor` 会打印项目根和配置目录；
- GUI 训练页下拉框列的是 `configs/` 下所有 `.ini`，选错配置就换个。

**别用「先试 cwd」的思路排**：打包版的工作目录是 `runtime/app`，
那里恰好也有一份 `configs/small.ini`，容易指错。"""),

    dict(id="x_empty_corpus", title="语料准备结果是 0", group="排查",
         keywords=["语料为0", "空", "0 token", "prepare"],
         body="""可能原因：

1. 目录里没有可识别文件（只认 .txt/.md/.json/.jsonl/.ndjson，含子目录）；
2. 文件编码不是 UTF-8（引擎会尝试 GBK，但仍可能失败）→ 用记事本另存为 UTF-8；
3. JSON 抽取后为空（见上条）→ 关掉「抽 JSON 正文」；
4. 文件是二进制/空文件。

体检页会逐个文件列出「字符数」和跳过原因，先看那里。
准备好后检查 `data/train.bin` 是否非空。"""),

    # ================================================================ 引擎
    dict(id="x_cli", title="命令行怎么用", group="引擎",
         keywords=["cli", "命令行", "命令", "lingyu"],
         body="""主命令 `lingyu <子命令>`：

```
lingyu doctor                       # 环境自检
lingyu prepare --raw <目录> --data data
lingyu scan --raw <目录>            # 只体检不写文件
lingyu train  --config configs/small.ini
lingyu sft    --config configs/sft.ini
lingyu chat   --config configs/sft.ini
lingyu plan   --config configs/small.ini   # 显存/耗时估算
lingyu launch --config x.ini --backend fsdp # 生成启动脚本
lingyu gui
```

`--json` 会让输出变成结构化事件（给 GUI 用），适合脚本集成。
`python -m lingyu` 必须在项目根目录下跑。"""),

    dict(id="x_config", title="配置文件怎么写", group="引擎",
         keywords=["ini", "配置", "config", "参数"],
         body="""INI 分段：`[model] [train] [data] [post] [parallel] [runtime] [sample]`。

最常用的就 `[model]`（层数/宽度/长度）和 `[train]`（batch/lr/步数）。
改单个参数不必动文件：`lingyu train --config x.ini --set train.lr=0.0005`。

优先级：内置默认 < 配置文件 < `--set`。
参数含义看「参数」分组，或直接看 GUI 参数后的「?」。"""),

    dict(id="x_doctor", title="doctor 自检做什么", group="引擎",
         keywords=["doctor", "自检", "环境检测"],
         body="""检查：Python 版本、项目根、依赖（torch/PySide6）、CUDA 与驱动、
显存、配置目录、语料目录、断点目录。

结论用人话输出：CPU 版 torch 会提示「重装 GPU 版」，驱动太老会说更新驱动，
而不是一句「CUDA 不可用」。

打包版：`LingYu.exe --console doctor`。GUI 概览页的「重新检测环境」
做的是同一件事（只重探显存/设备，不重装依赖）。"""),

    dict(id="x_gui", title="GUI 各页干什么", group="引擎",
         keywords=["gui", "界面", "页面", "帮助"],
         body="""- **概览**：这台机器能训多大、当前语料/配置状态；
- **语料准备**：体检 + 准备 tokenizer/train.bin；
- **训练**：选配置 → 看显存估算 → 开始/续训/停止；
- **监控**：实时 loss/速度（预训练与 SFT 分开）；
- **分析**：回看历史训练的 loss/lr/速度曲线；
- **断点**：列出、续训、拿去做 SFT/聊天、删除；
- **续写**：预训练断点直接续写文本；
- **微调 SFT**：选数据 + 模板 + 基础断点；
- **聊天**：加载 SFT 断点提问；
- **设置 / 帮助**：路径、主题、语言、百科。

`Ctrl+1..9` 切页，`Ctrl+0` 帮助。"""),

    dict(id="x_help_search", title="帮助页搜索怎么用", group="引擎",
         keywords=["帮助", "搜索", "关键词", "精确", "模糊"],
         body="""三种模式：

- **智能**（默认）：空格分词，每个词都要命中（标题/关键词权重高）；
- **精确**：整串必须原样出现，适合搜报错原文；
- **模糊**：标题/关键词做相似度匹配，容错错别字（「显存不够」也能找到）。

输入框左边选模式；结果按组排序，点标题看全文。
搜不到就换个词（比如「OOM」「续训」「gqa」）。"""),

    dict(id="x_plugin", title="插件怎么写", group="引擎",
         keywords=["插件", "plugin", "扩展", "components"],
         body="""把 `.py` 丢进 `plugins/` 目录即可，模块里定义构造函数并注册：

```python
from lingyu.registry import register

@register("model", "my_model")
def build_my_model(cfg): ...
```

注册表三层优先级：内置 < 插件 < 配置点名。
插件不需要改引擎代码；但接口要跟 `components.py` 的约定一致
（`build_model(cfg)` / `build_tokenizer(cfg)` 等）。
文档在 `docs/DESIGN.md` 的注册表一节。"""),

    # ================================================================ 术语
    dict(id="x_terms", title="术语速查表", group="术语",
         keywords=["术语", "名词", "缩写", "glossary"],
         body="""按字母排：

- **AdamW**：带权重衰减的 Adam 优化器，LLM 默认；
- **AMP**：自动混合精度；
- **BPE**：字节对编码分词；
- **CE**：交叉熵损失；
- **ckpt**：checkpoint，断点；
- **DPO**：直接偏好优化（未实现）；
- **FFN**：前馈网络；
- **GQA**：分组查询注意力；
- **LoRA**：低秩适配（未实现）；
- **LR**：学习率；
- **MoE**：混合专家；
- **MQA**：多查询注意力；
- **OOM**：显存不足；
- **PPL**：困惑度 = exp(loss)；
- **RoPE**：旋转位置编码；
- **SFT**：监督微调；
- **SwiGLU**：门控激活 FFN；
- **token**：词元；
- **VRAM**：显存。"""),
]
