# -*- coding: utf-8 -*-
r"""启动脚本生成：**不假装能本地跑万亿参数，但给出下一步该走的路**。

## 这个文件的态度

引擎自己实现单机多卡（FSDP），但**不去重新实现 Megatron 那样的
多机超大模型训练栈** —— 那是一个几十人年的工程量，硬做只会做出一个
又慢又容易崩的半成品，害了用它的人。

所以对超出单机的规模，引擎做的是：

1. 老老实实算出**要多少张卡**（`plan.py`）
2. 生成一份**能直接执行的启动脚本**，把配置翻译成对应框架的参数
3. 在脚本开头用注释写明：这是给谁用的、前置条件是什么、
   哪些地方你必须自己改

**绝不生成一份看起来能跑、实际上跑不起来的东西。**
所以每份脚本里都有 `TODO` 标记明确指出需要人工确认的地方，
而不是假装「一键启动万亿模型」。

## 生成哪几种

| 后端 | 产出 | 真实性 |
|---|---|---|
| `fsdp` | `torchrun` 启动脚本 | **实验性**：引擎只生成脚本，分布式训练**尚未实现**——脚本会把 N 个进程各自当单卡训练跑 |
| `deepspeed` | `deepspeed` 启动脚本 + `ds_config.json` | 生成脚本 + 配置（用官方 DeepSpeed；入口脚本需你按 TODO 自己写） |
| `megatron` | 参考脚本 + 详细的前置说明 | 引擎**不实现**，只把参数翻译好给你 |

Megatron 那一份刻意写成「参考模板」而不是「一键脚本」——
因为它要求你先装 Megatron-LM、准备好它格式的数据集，
这些不是一份脚本能替你搞定的。
"""

import json
import os

SCRIPT_HEADER = """\
# ==========================================================================
#  灵语（LingYu）自动生成的分布式训练启动脚本
#
#  生成自配置 : {config_path}
#  目标后端   : {backend}
#  模型规模   : {params_h} 参数（{scale_note}）
#  并行配置   : 数据并行 {dp} × 张量并行 {tp} × 流水线并行 {pp}{ep}
#  预计显存   : 每卡约 {per_card}，共需约 {cards} 张卡
#
#  ⚠️ 生成这份脚本时，引擎**没有**在你机器上验证过这个规模能跑 ——
#     它是按你给的配置算出来的。第一次跑请盯着显存，并准备好调小配置。
#
#  这份脚本里所有标了 [TODO] 的地方都需要你自己确认。
# ==========================================================================
"""


def _scale_note(params):
    p = float(params)
    if p < 1e6:
        return "不到 1M"
    if p < 1e9:
        return "%.1fM" % (p / 1e6)
    if p < 1e12:
        return "%.1fB" % (p / 1e9)
    return "%.2fT" % (p / 1e12)


def _fmt_gb(n):
    return "%.1f GB" % (float(n) / 1024 ** 3)


def backend_info(name):
    """后端的真实支持程度。给 doctor / GUI 用，也用来防止文案吹牛。"""
    return {
        "local": {
            "label": "单卡",
            "implemented": True,
            "summary": "引擎自己实现，开箱即用",
        },
        "fsdp": {
            "label": "FSDP2 单机多卡（实验性）",
            "implemented": "config-only",
            "summary": "实验性：仅生成脚本，引擎尚未实现分布式训练 ——"
                       "脚本里的多个进程会各自做单卡训练，别直接用于生产",
        },
        "deepspeed": {
            "label": "DeepSpeed 多机",
            "implemented": "config-only",
            "summary": "引擎**不重写** DeepSpeed，只生成启动脚本和 ds_config.json，"
                       "用官方实现来跑",
        },
        "megatron": {
            "label": "Megatron-LM 超大模型",
            "implemented": False,
            "summary": "引擎**不实现**。只把配置翻译成参考脚本和参数清单，"
                       "你需要自己装 Megatron-LM 并准备它格式的数据",
        },
    }[name]


def gen_torchrun(cfg, est, params, active=None):
    """FSDP / torchrun 启动脚本（**实验性**）。

    ⚠️ 引擎自己的训练循环里还没有分布式代码（没有 init_process_group /
    fully_shard / all_reduce）。这份脚本目前只是把并行参数翻译成 torchrun
    命令行；直接跑会是 N 个独立单卡训练。等训练代码实现了 DDP/FSDP2 再升级。
    """
    tp = int(cfg.get("parallel", {}).get("tensor_parallel", 1) or 1)
    pp = int(cfg.get("parallel", {}).get("pipeline_parallel", 1) or 1)
    ep = int(cfg.get("parallel", {}).get("expert_parallel", 1) or 1)
    dp = max(1, int(est.get("cards_needed", 1)) // max(1, tp * pp))
    zero = int(cfg.get("parallel", {}).get("zero_stage", 2) or 0)
    ac = bool(cfg.get("parallel", {}).get("activation_checkpoint", False))
    offload = bool(cfg.get("parallel", {}).get("cpu_offload", False))

    head = SCRIPT_HEADER.format(
        config_path=cfg.get("_path", "<配置>"), backend="fsdp (torchrun)",
        params_h="{:,}".format(int(params)), scale_note=_scale_note(params),
        dp=dp, tp=tp, pp=pp, ep=(" × 专家并行 %d" % ep) if ep > 1 else "",
        per_card=_fmt_gb(est.get("total_bytes", 0) / max(1, dp)),
        cards=est.get("cards_needed", 1))

    extra = []
    if zero:
        extra.append("parallel.zero_stage=%d" % zero)
    if ac:
        extra.append("parallel.activation_checkpoint=true")
    if offload:
        extra.append("parallel.cpu_offload=true")
    extra_flags = "".join(" \\\n  --set %s" % e for e in extra)

    return head + """
# ---- 前置条件 --------------------------------------------------------------
#   1. 每台机器都装好同版本的 PyTorch + CUDA，且能互相免密 SSH
#   2. 所有机器上这个项目在**同样的路径**下（数据、词表、配置都一致）
#   3. 数据已经 prepare 好（train.bin / val.bin / tokenizer.txt 都在）

# [TODO] 改成你实际的机器列表和每台机器的卡数
NPROC_PER_NODE={nproc}
NNODES={nnodes}
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# 单机多卡：直接用这一段就够了
torchrun \\
  --standalone \\
  --nproc_per_node=$NPROC_PER_NODE \\
  -m lingyu train \\
  --config {config_path} \\
  --set parallel.backend=fsdp{extra}

# ---- 多机：改用下面这段（每台机器都要执行，RANK 各自不同）------------------
# torchrun \\
#   --nnodes=$NNODES \\
#   --node_rank=$RANK \\            # 每台机器不同：0, 1, 2, ...
#   --nproc_per_node=$NPROC_PER_NODE \\
#   --master_addr=$MASTER_ADDR \\
#   --master_port=$MASTER_PORT \\
#   -m lingyu train \\
#   --config {config_path} \\
#   --set parallel.backend=fsdp{extra}

# ---- 跑起来之后怎么确认是正常的 --------------------------------------------
#   - 每个进程的日志都会写到自己那份 train.log
#   - 第 10 步左右的 loss 应该和单卡跑同配置时**接近**（差太多说明有问题）
#   - 存断点只有 rank 0 会写，别被其他 rank 没写吓到
""".format(nproc=dp, nnodes=1, config_path=cfg.get("_path", "<配置>"),
           extra=extra_flags)


def gen_deepspeed(cfg, est, params):
    """DeepSpeed：引擎不重写它，只生成官方启动脚本 + 配置文件。"""
    zero = int(cfg.get("parallel", {}).get("zero_stage", 2) or 0)
    offload = bool(cfg.get("parallel", {}).get("cpu_offload", False))
    ac = bool(cfg.get("parallel", {}).get("activation_checkpoint", False))
    dp = max(1, int(est.get("cards_needed", 1)))

    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": zero,
            "offload_optimizer": ({"device": "cpu", "pin_memory": True}
                                  if offload else {"device": "none"}),
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": float(cfg.get("train", {}).get("grad_clip", 1.0) or 1.0),
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }
    if ac:
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "contiguous_memory_optimization": False,
        }

    head = SCRIPT_HEADER.format(
        config_path=cfg.get("_path", "<配置>"), backend="deepspeed",
        params_h="{:,}".format(int(params)), scale_note=_scale_note(params),
        dp=dp, tp=1, pp=1, ep="",
        per_card=_fmt_gb(est.get("total_bytes", 0) / dp),
        cards=est.get("cards_needed", 1))

    return head + """
# ⚠️ 重要：这个文件的 ds_config.json 是**引擎生成的**，但真正的训练
#    要由 DeepSpeed 来跑。引擎没有实现 DeepSpeed 后端 —— 与其自己写一个
#    又慢又不全的，不如老老实实用官方的。
#
#    这也意味着：这份脚本**不能**用 `lingyu train` 跑。
#    你需要自己写一个入口脚本调用 deepspeed.initialize()，
#    把 lingyu.model 建出来的模型交给它。
#
# [TODO] 写入口脚本（约 30 行）：
#
#     import torch, deepspeed
#     from lingyu.config import load_config, resolve
#     from lingyu.model import config_from_section, LingYu
#     from lingyu.tokenizer import load_tokenizer
#
#     cfg = resolve(load_config("{config_path}"))
#     tok = load_tokenizer(cfg["data"]["tokenizer_path"])
#     model = LingYu(config_from_section(cfg["model"], tok.vocab_size))
#     engine, optimizer, _, _ = deepspeed.initialize(
#         model=model, config="ds_config.json")
#     # 之后就是普通的训练循环，用 engine.backward(loss) / engine.step()

# ---- 前置条件 --------------------------------------------------------------
#   pip install deepspeed
#   每台机器同版本 CUDA / PyTorch

# [TODO] 改成你的机器
deepspeed --num_gpus={dp} --num_nodes=1 your_entry.py

# ---- zero_stage 怎么选 ------------------------------------------------------
#   {zero_tip}
""".format(config_path=cfg.get("_path", "<配置>"), dp=dp,
           zero_tip=("你现在配的是 stage %d。经验：优化器状态放不下就用 2；"
                     "权重也放不下才上 3（3 的通信开销很大，能不用就不用）" % zero)), ds_config


def gen_megatron(cfg, est, params, active=None):
    """Megatron：只给参考，明确说引擎不实现。"""
    mc = cfg.get("model", {})
    L = int(mc.get("n_layer", 8) or 8)
    d = int(mc.get("n_embd", 512) or 512)
    h = int(mc.get("n_head", 8) or 8)
    V = int(mc.get("vocab_size", 8192) or 8192)
    blk = int(mc.get("block_size", 512) or 512)
    moe = bool(mc.get("use_moe", False))
    n_expert = int(mc.get("n_expert", 1) or 1)
    tp = int(cfg.get("parallel", {}).get("tensor_parallel", 1) or 1)
    pp = int(cfg.get("parallel", {}).get("pipeline_parallel", 1) or 1)
    ep = int(cfg.get("parallel", {}).get("expert_parallel", 1) or 1)
    total_cards = int(est.get("cards_needed", 1))

    head = SCRIPT_HEADER.format(
        config_path=cfg.get("_path", "<配置>"), backend="megatron（参考脚本）",
        params_h="{:,}".format(int(params)), scale_note=_scale_note(params),
        dp=max(1, total_cards // max(1, tp * pp)), tp=tp, pp=pp,
        ep=(" × 专家并行 %d" % ep) if ep > 1 else "",
        per_card=_fmt_gb(est.get("total_bytes", 0) / max(1, total_cards)),
        cards=total_cards)

    return head + """
# ==========================================================================
#  ⚠️ 引擎**没有实现** Megatron 后端，这是刻意的。
#
#  理由：Megatron-LM 是一个庞大的独立训练栈（张量并行、流水线并行、
#  序列并行、分布式优化器、自定义 CUDA kernel……）。要在灵语里重写一遍，
#  结果是做出一个又慢又不全、还很难维护的复制品。**那对用户是有害的。**
#
#  所以灵语做的是：把你的配置**翻译**成 Megatron 需要的参数，
#  让你不用自己对着文档换算。真正的训练由 NVIDIA 的 Megatron-LM 执行。
# ==========================================================================

# ---- 第 0 步：把灵语的数据集转成 Megatron 格式 -----------------------------
#   Megatron 要求数据是它自己的二进制格式（.bin + .idx），和灵语的
#   train.bin 不一样。用它的 preprocess_data.py 转，或者写个脚本按
#   "文档边界 + EOD token" 重新打包。
# [TODO] 这一步必须你自己做，灵语不替你决定文档怎么切。

# ---- 第 1 步：装 Megatron-LM -------------------------------------------------
#   git clone https://github.com/NVIDIA/Megatron-LM
#   cd Megatron-LM && pip install -e .

# ---- 第 2 步：按下面这组参数启动 ---------------------------------------------
#   下面每个值都是从你的灵语配置**换算**过来的，不是随便填的：

GPUS_PER_NODE={total_cards}
NNODES=1
TP={tp}          # 张量并行：切单层里的矩阵
PP={pp}          # 流水线并行：按层切
EP={ep}          # 专家并行{ep_note}
DP={dp}          # 数据并行 = 总卡数 / (TP × PP)

# [TODO] 确认 TP × PP × DP 是否等于你的卡数，不相等会直接报错

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES \\
  pretrain_gpt.py \\
  --tensor-model-parallel-size $TP \\
  --pipeline-model-parallel-size $PP \\
  --num-layers {L} \\
  --hidden-size {d} \\
  --num-attention-heads {h} \\
  --seq-length {blk} \\
  --max-position-embeddings {blk} \\
  --vocab-size {V} \\
  --micro-batch-size 1 \\
  --global-batch-size {gbs} \\
  --bf16 \\
  --train-iters {iters}

# ---- 灵语配置里对不上的地方（必须人工决定）----------------------------------
#   [TODO] 1. 归一化：灵语用 RMSNorm。Megatron 要加 --normalization RMSNorm
#   [TODO] 2. 激活函数：灵语用 SwiGLU。Megatron 要加 --swiglu
#             并令 --ffn-hidden-size 等于灵语的 d_ff
#   [TODO] 3. 位置编码：灵语用 RoPE。Megatron 要加 --position-embedding-type rope
#   [TODO] 4. 权重共享：灵语 tie_weights={tie}。Megatron 加 --untie-embeddings-and-output-weights
#             是**反过来**的语义，注意别搞错
#   [TODO] 5. GQA：灵语 n_kv_head={kv}。Megatron 用 --group-query-attention
#             --num-query-groups {kv}
{moe_note}
# ---- 为什么这些不能自动生成 ------------------------------------------------
#   上面每一条都涉及「灵语的语义 vs Megatron 的语义」的对应关系。
#   换算错了不会报错，只会训出一个和别人不一样的东西 ——
#   而你会以为自己在复现某个结果。所以引擎把选择权留给你。
""".format(total_cards=total_cards, tp=tp, pp=pp, ep=ep, dp=max(1, total_cards // max(1, tp * pp)),
           ep_note="（MoE 专用，非 MoE 请保持 1）" if moe else "（你的配置不是 MoE，保持 1）",
           L=L, d=d, h=h, blk=blk, V=V,
           gbs=int(cfg.get("train", {}).get("batch_size", 8) or 8),
           iters=int(cfg.get("train", {}).get("max_steps", 1000) or 1000),
           tie=bool(mc.get("tie_weights", True)),
           kv=int(mc.get("n_kv_head", 1) or 1),
           moe_note=("\n#   [TODO] 6. MoE：灵语 n_expert=%d，Megatron 要 --num-experts %d\n"
                     "#             并配好 --moe-grouped-gemm 等一整套参数\n"
                     % (n_expert, n_expert)) if moe else "")


GENERATORS = {
    "fsdp": gen_torchrun,
    "local": gen_torchrun,
    "deepspeed": gen_deepspeed,
    "megatron": gen_megatron,
}


def generate(backend, cfg, est, params, active=None):
    """按后端生成脚本。

    返回 `{"script": 脚本文本, "ds_config": dict 或 None}`。

    为什么带 ds_config：DeepSpeed 要两个文件才能跑（启动脚本 + 配置），
    分开返回比让调用方再调一次函数清楚。
    """
    if backend not in GENERATORS:
        raise ValueError("不认识的后端 %r，可选：%s"
                         % (backend, "、".join(sorted(GENERATORS))))
    if backend == "deepspeed":
        text, ds_config = gen_deepspeed(cfg, est, params)
        return {"script": text, "ds_config": ds_config}
    if backend == "megatron":
        return {"script": gen_megatron(cfg, est, params, active), "ds_config": None}
    return {"script": gen_torchrun(cfg, est, params, active), "ds_config": None}


def write_outputs(out_dir, backend, result):
    """把生成结果写到磁盘。返回写出的文件列表。"""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    # 用 .sh（Linux 集群上最常见）；内容本身是 shell，Windows 用户
    # 用 Git Bash / WSL 也能跑。
    sh = os.path.join(out_dir, "launch_%s.sh" % backend)
    with open(sh, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(result["script"])
    # 生成的脚本是给用户直接 `./launch_x.sh` 的：非 Windows 上要补执行位，
    # 否则用户会撞上 `Permission denied`（Linux 集群上尤其常见）。
    # Windows 没有 POSIX 执行位，跳过。
    if os.name != "nt":
        os.chmod(sh, 0o755)
    written.append(sh)
    if result.get("ds_config") is not None:
        p = os.path.join(out_dir, "ds_config.json")
        with open(p, "w", encoding="utf-8") as fp:
            json.dump(result["ds_config"], fp, indent=2, ensure_ascii=False)
        written.append(p)
    return written
