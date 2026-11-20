# -*- coding: utf-8 -*-
"""参数规格表 —— 整个项目的**单一事实源**。

这一份表同时驱动五样东西，所以加参数只改一处：

    1. 配置校验（类型/范围/依赖关系）
    2. GUI 表单生成（按 tier 折叠成基础/进阶/专家）
    3. CLI 的 --help 文案
    4. 配置文件的注释模板
    5. 文档里的参数表

以前参数散落在三处（KNOWN_KEYS、各 train() 里的 get() 调用、build_config 签名），
加一个参数要改三个地方，漏一处就出现"文档说能改但代码不认"的 bug。
"""

from ..version import CKPT_FORMAT, CONFIG_FORMAT  # noqa: F401  (对外转出)

# tier 决定 GUI 里放在哪一层折叠面板
TIERS = ("basic", "advanced", "expert")

# 组件名 -> 可选实现（由 registry 提供实际实现，这里只用于生成下拉框和校验）
COMPONENT_CHOICES = {
    "attention": ["mha", "gqa", "mqa"],
    "ffn": ["swiglu", "geglu", "moe"],
    "norm": ["rmsnorm", "layernorm"],
    "pos": ["rope", "learned", "none"],
}


def P(type, default, help_zh, help_en, tier="advanced", **kw):
    """构造一个参数描述。**所有参数都必须有中英双语帮助**，开源要用。"""
    d = {"type": type, "default": default, "help_zh": help_zh, "help_en": help_en,
         "tier": tier}
    d.update(kw)
    return d


# ==========================================================================
SPEC = {
    # ---------------------------------------------------------------- 模型
    "model": {
        "type": P(str, "lingyu", "模型实现（注册表名，默认 lingyu）",
                  "Model implementation (registry name, default lingyu)",
                  tier="advanced"),
        "n_layer": P(int, 8, "层数：堆几层 Transformer", "Number of layers",
                     tier="basic", min=1, max=4096),
        "n_embd": P(int, 512, "宽度：每层内部多少通道",
                    "Embedding / hidden dimension", tier="basic",
                    min=8, max=131072),
        "n_head": P(int, 8, "注意力头数", "Number of attention heads",
                    tier="basic", min=1, max=1024),
        "block_size": P(int, 512, "上下文长度（token 数）",
                        "Context length in tokens", tier="basic",
                        min=8, max=1048576),
        "n_kv_head": P(int, 2, "KV 头数（GQA）。越小 KV cache 越省",
                       "Number of KV heads (GQA)", min=1, max=1024),
        "kv_cache_dtype": P(str, "auto",
                            "KV cache 精度：auto = 跟随激活；f32 / bf16 显式指定；"
                            "int8 = 每 token 一组 scale 的量化（约省一半显存）",
                            "KV cache dtype: auto / f32 / bf16 / int8 (int8 halves memory)",
                            choices=["auto", "f32", "bf16", "int8"]),
        "d_ff": P(int, 0, "前馈中间维度。0 = 自动（8/3 × n_embd，对齐 8）",
                  "FFN intermediate size. 0 = auto", min=0, max=1048576),
        "dropout": P(float, 0.0, "丢弃率，防过拟合",
                     "Dropout rate", min=0.0, max=0.9),
        "bias": P(bool, False, "线性层是否带偏置（现代做法一般不带）",
                  "Use bias in linear layers"),
        "tie_weights": P(bool, True, "输入输出共用词嵌入矩阵（省一半 embedding 参数）",
                         "Tie input/output embeddings"),
        "norm_eps": P(float, 1e-6, "归一化的 epsilon",
                      "Normalization epsilon", min=1e-12, max=1e-2),
        "init_std": P(float, 0.02, "权重初始化标准差",
                      "Weight init std", min=1e-5, max=1.0),
        "residual_std": P(float, 0.02, "残差投影初始化标准差",
                          "Residual projection init std", min=1e-5, max=1.0),
        # 组件选择（注册表）。
        #
        # ⚠️ 这四项的默认值**必须留空字符串**，不能填具体实现名。
        # 空 = 「引擎自己决定」，这是配置语言里统一的约定：
        #     ffn 空 + use_moe=true   -> 用 moe
        #     ffn 空 + use_moe=false  -> 用 swiglu
        # 如果这里默认写死 "swiglu"，那么 config.resolve() 会把
        # moe.ini（只写了 use_moe=true、没写 ffn）补成 ffn="swiglu"，
        # 于是 use_moe=true 被静默推翻、MoE 档变成稠密档 —— 实测踩过。
        "attention": P(str, "", "注意力实现（注册表名）。空 = 自动（按 n_kv_head 推断 MHA/GQA/MQA）",
                       "Attention implementation (registry name). Empty = auto",
                       choices=COMPONENT_CHOICES["attention"]),
        "ffn": P(str, "", "前馈实现（注册表名）。空 = 自动（按 use_moe 决定 swiglu / moe）",
                 "FFN implementation (registry name). Empty = auto",
                 choices=COMPONENT_CHOICES["ffn"]),
        "norm": P(str, "", "归一化实现（注册表名）。空 = rmsnorm",
                  "Normalization implementation. Empty = rmsnorm",
                  choices=COMPONENT_CHOICES["norm"]),
        "pos": P(str, "", "位置编码实现（注册表名）。空 = rope",
                 "Positional encoding. Empty = rope",
                 choices=COMPONENT_CHOICES["pos"]),
        "rope_base": P(float, 10000.0, "RoPE 底数（调大可以外推更长上下文）",
                       "RoPE base", min=1.0, max=1e9),
        # MoE
        "use_moe": P(bool, False, "是否启用稀疏专家（MoE）",
                     "Enable Mixture-of-Experts"),
        "n_expert": P(int, 4, "专家总数", "Total number of experts",
                      min=1, max=1024),
        "n_expert_per_tok": P(int, 2, "每个 token 走几个专家（top-k）",
                              "Experts per token (top-k)", min=1, max=64),
        "n_shared_expert": P(int, 1, "永远激活的共享专家数",
                             "Number of shared experts", min=0, max=64),
        "expert_d_ff": P(int, 0, "每个专家的中间维度。0 = 自动（d_ff 的一半）",
                         "Per-expert intermediate size. 0 = auto", min=0),
        "moe_every": P(int, 1, "每隔几层放一个 MoE 层",
                       "Place a MoE layer every N layers", min=1),
        "moe_router_jitter": P(float, 0.0, "路由器噪声（训练时抖动，帮助负载均衡）",
                               "Router jitter during training", min=0.0, max=1.0),
        "shared_expert_gate": P(bool, False, "共享专家是否加门控",
                                "Gate the shared expert"),
        "vocab_size": P(int, 0, "词表大小。0 = 自动跟随分词器",
                        "Vocabulary size. 0 = follow tokenizer", min=0),
    },

    # ---------------------------------------------------------- 训练（预训练）
    "train": {
        "out_dir": P(str, "out", "训练产物输出目录",
                     "Output directory", tier="basic"),
        "batch_size": P(int, 32, "批大小：一步同时看多少段文本",
                        "Batch size", tier="basic", min=1, max=65536),
        "max_steps": P(int, 5000, "训练总步数", "Total training steps",
                       tier="basic", min=1),
        "learning_rate": P(float, 1e-3, "最大学习率", "Peak learning rate",
                           tier="basic", min=0.0),
        "grad_accum": P(int, 1, "梯度累积：等效批大小 = batch_size × grad_accum",
                        "Gradient accumulation steps", min=1),
        "min_lr": P(float, 1e-4, "余弦衰减的终点学习率",
                    "Minimum learning rate", min=0.0),
        "warmup_iters": P(int, 200, "学习率预热步数",
                          "LR warmup steps", min=0),
        "lr_decay_iters": P(int, 0, "衰减到 min_lr 的步数。0 = 跟随 max_steps",
                            "Steps to decay over. 0 = follow max_steps", min=0),
        "weight_decay": P(float, 0.1, "权重衰减（只作用于矩阵权重）",
                          "Weight decay", min=0.0, max=1.0),
        "beta1": P(float, 0.9, "AdamW 一阶动量", "AdamW beta1",
                   min=0.0, max=1.0),
        "beta2": P(float, 0.95, "AdamW 二阶动量", "AdamW beta2",
                   min=0.0, max=1.0),
        "grad_clip": P(float, 1.0, "梯度裁剪阈值。0 = 不裁剪",
                       "Gradient clipping. 0 = off", min=0.0),
        "eval_interval": P(int, 250, "每多少步做一次验证集评估。0 = 不评估",
                           "Validation interval. 0 = off", min=0),
        "eval_iters": P(int, 50, "每次评估跑多少批",
                        "Batches per evaluation", min=1),
        "log_interval": P(int, 10, "每多少步打印一次进度",
                          "Log every N steps", min=1),
        "ckpt_interval": P(int, 500, "每多少步存一次断点。0 = 不存中间断点",
                           "Checkpoint interval. 0 = only final", min=0),
        "sample_interval": P(int, 1000, "每多少步试写一段。0 = 不试写",
                             "Sample interval. 0 = off", min=0),
        "sample_while_train": P(bool, True, "训练中是否试写",
                                "Sample during training"),
        "aux_loss_coef": P(float, 0.01, "MoE 负载均衡损失系数",
                           "MoE auxiliary loss coefficient", min=0.0),
        "z_loss_coef": P(float, 0.001, "MoE 路由 z-loss 系数",
                         "MoE router z-loss coefficient", min=0.0),
        "seed": P(int, 1234, "随机种子（固定后结果可复现）",
                  "Random seed", min=0),
        "device": P(str, "auto", "训练设备：auto / cuda / cpu",
                    "Device", choices=["auto", "cuda", "cpu"]),
        "amp_dtype": P(str, "bf16", "混合精度类型",
                       "Autocast dtype", choices=["bf16", "fp16", "off"]),
        "compile": P(bool, False, "是否用 torch.compile 加速（首次编译较慢）",
                     "Use torch.compile"),
        "grad_checkpoint": P(bool, False,
                             "梯度检查点：用约 1.3 倍算力换 30~60% 激活显存",
                             "Gradient checkpointing (compute for memory)"),
        "optim_bits": P(str, "32",
                        "优化器状态位宽：32 = 全精度；8 = 分块量化，约省 4 倍",
                        "Optimizer state bits: 32 or 8", choices=["32", "8"]),
        "loss_chunk": P(int, 8192,
                        "交叉熵分块大小（token 数）。0 = 不分块；分块可显著降低长上下文训练的峰值显存",
                        "Cross-entropy chunk size in tokens. 0 = off", min=0),
        "num_workers": P(int, 0, "数据加载进程数（0 = 主进程加载）",
                         "DataLoader workers", min=0, max=64),
    },

    # ------------------------------------------------------------ 数据
    "data": {
        "data_dir": P(str, "data", "数据目录（prepare 的输出）",
                      "Data directory", tier="basic"),
        "tokenizer_path": P(str, "", "分词器文件路径。空 = 用 data_dir/tokenizer.txt",
                            "Tokenizer path. Empty = data_dir/tokenizer.txt"),
        "val_ratio": P(float, 0.02, "验证集比例",
                       "Validation split ratio", min=0.0, max=0.5),
        "streaming": P(bool, False, "流式加载（超大数据集用，省内存）",
                       "Streaming data loading"),
    },

    # ------------------------------------------------------------ 采样
    "sample": {
        "prompt": P(str, "他", "默认开头文字", "Default prompt", tier="basic"),
        "max_new_tokens": P(int, 200, "最多生成多少 token",
                            "Max new tokens", tier="basic", min=1),
        "temperature": P(float, 0.9, "温度：低=保守，高=放飞",
                         "Temperature", tier="basic", min=0.01, max=5.0),
        "top_k": P(int, 40, "只在前 k 个候选里抽。0 = 关闭",
                   "Top-k sampling. 0 = off", min=0),
        "top_p": P(float, 0.95, "核采样阈值。1.0 = 关闭",
                   "Nucleus sampling. 1.0 = off", min=0.0, max=1.0),
        "repetition_penalty": P(float, 1.05, "重复惩罚（1.0 = 关闭）",
                                "Repetition penalty. 1.0 = off", min=1.0, max=3.0),
        "seed": P(int, 0, "随机种子。0 = 每次不同",
                  "Sampling seed. 0 = random", min=0),
    },

    # -------------------------------------------------- 后训练（SFT / 对话）
    "post": {
        # 默认空字符串 = 不做后训练。这样纯预训练的配置不会被"缺 data_path"报错。
        # 用户选了 sft/dpo/lora 才会去校验 data_path。
        "method": P(str, "", "后训练方法。空 = 不做后训练",
                    "Post-training method. Empty = skip", tier="basic",
                    choices=["", "sft", "dpo", "lora"]),
        "data_path": P(str, "", "对话数据（jsonl）",
                       "Chat data (jsonl)", tier="basic"),
        "template": P(str, "plain", "对话模板。plain = 中文标记「用户：/助手：」"
                                   "（不需要特殊 token，任何断点都能用）；"
                                   "chatml/qwen = <|im_start|> 格式（需要重新 prepare）",
                      "Chat template", tier="basic",
                      choices=["plain", "chatml", "qwen", "llama3"]),
        "system_prompt": P(str, "", "系统提示词（给模型定人设，可留空）",
                           "System prompt. Empty = none", tier="basic"),
        "init_from": P(str, "", "从哪个预训练断点开始（空 = 自动找 [train] out_dir）",
                       "Initial checkpoint. Empty = auto-detect", tier="basic"),
        "out_dir": P(str, "out_sft", "输出目录", "Output directory"),
        "batch_size": P(int, 16, "批大小", "Batch size", min=1),
        "max_steps": P(int, 2000, "总步数", "Total steps", min=1),
        "learning_rate": P(float, 5e-4, "学习率。SFT 要比预训练小得多，"
                                        "建议 1e-5 ~ 5e-4；太大模型会「忘光」预训练知识",
                           "Learning rate. Much smaller than pretraining", min=0.0),
        "loss_on_prompt": P(bool, False,
                            "是否对「用户问题」部分也算 loss。"
                            "默认关（开着会让模型学着自己生成问题）",
                            "Compute loss on prompt tokens too"),
        "max_seq_len": P(int, 512, "对话最长 token 数（超出的样本会被丢掉）",
                         "Max sequence length for chats", min=16),
        "log_interval": P(int, 10, "每多少步打印一次进度",
                          "Log every N steps", min=1),
        "eval_interval": P(int, 100, "每多少步做一次验证",
                           "Evaluate every N steps", min=1),
        "ask_interval": P(int, 200, "每多少步真的问模型一句，肉眼看看会不会答",
                          "Ask the model every N steps", min=1),
        "ckpt_interval": P(int, 200, "每多少步存一次断点",
                           "Checkpoint every N steps", min=1),
        "lora_r": P(int, 8, "LoRA 秩", "LoRA rank", min=1, max=256),
        "lora_alpha": P(int, 16, "LoRA alpha", "LoRA alpha", min=1),
        "lora_dropout": P(float, 0.05, "LoRA dropout", "LoRA dropout",
                          min=0.0, max=0.9),
        "lora_targets": P(str, "q_proj,k_proj,v_proj,o_proj",
                          "LoRA 作用在哪些模块（逗号分隔）",
                          "LoRA target modules (comma separated)"),
    },

    # ------------------------------------------------------ 分布式（大模型）
    "parallel": {
        "backend": P(str, "local", "执行后端：local=单卡，fsdp=单机多卡，"
                                   "deepspeed/megatron=生成集群脚本",
                     "Execution backend", tier="basic",
                     choices=["local", "fsdp", "deepspeed", "megatron"]),
        "tensor_parallel": P(int, 1, "张量并行度（切分单层内的矩阵）",
                             "Tensor parallel size", min=1),
        "pipeline_parallel": P(int, 1, "流水线并行度（按层切分）",
                               "Pipeline parallel size", min=1),
        "expert_parallel": P(int, 1, "专家并行度（MoE 专用）",
                             "Expert parallel size", min=1),
        "zero_stage": P(int, 2, "DeepSpeed ZeRO 阶段（1/2/3）",
                        "ZeRO stage", choices=[0, 1, 2, 3]),
        "cpu_offload": P(bool, False, "把优化器状态卸载到内存（省显存，变慢）",
                         "Offload optimizer state to CPU"),
        "activation_checkpoint": P(bool, False, "梯度检查点（用时间换显存）",
                                   "Activation checkpointing"),
        "gradient_accum": P(int, 1, "分布式梯度累积步数",
                            "Distributed grad accumulation", min=1),
        "nccl_timeout_min": P(int, 30, "NCCL 超时（分钟）",
                              "NCCL timeout in minutes", min=1),
    },

    # ------------------------------------------------------------ 运行时
    "runtime": {
        "lang": P(str, "zh_CN", "界面语言", "UI language",
                  choices=["zh_CN", "en_US"]),
        "log_level": P(str, "info", "日志级别", "Log level",
                       choices=["debug", "info", "warn", "error"]),
        "auto_exit_on_oom": P(bool, False, "显存不足时自动按建议调小并重试",
                              "Auto-retry with smaller batch on OOM"),
        "monitor": P(str, "console", "指标输出方式：console / tensorboard / wandb",
                     "Metrics sink", choices=["console", "tensorboard", "wandb"]),
    },
}

SECTIONS = tuple(SPEC.keys())


def all_keys(section):
    """某个配置节里所有合法的键。"""
    return tuple(SPEC.get(section, {}).keys())


def get_param(section, key):
    """取一个参数描述，不存在返回 None。"""
    return SPEC.get(section, {}).get(key)


def iter_params(section=None, tier=None):
    """遍历参数。GUI 用它生成表单。"""
    for sec, params in SPEC.items():
        if section is not None and sec != section:
            continue
        for key, p in params.items():
            if tier is not None and p["tier"] != tier:
                continue
            yield sec, key, p


def make_ini_template(section, values=None):
    """按 spec 生成带中文注释的配置文件片段（给 GUI 和 CLI 用）。"""
    lines = []
    maxlen = max((len(k) for k in SPEC.get(section, {})), default=0)
    for key, p in SPEC.get(section, {}).items():
        v = p["default"]
        if values and key in values:
            v = values[key]
        if isinstance(v, bool):
            v = "true" if v else "false"
        if isinstance(v, str):
            v = v if v else '""'
        line = "%-*s = %s" % (maxlen, key, v)
        lines.append("%-46s # [%s] %s" % (line, p["tier"], p["help_zh"]))
    return "\n".join(lines)
