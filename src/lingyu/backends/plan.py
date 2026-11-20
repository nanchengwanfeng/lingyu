# -*- coding: utf-8 -*-
r"""资源估算：**一个引擎从 13M 到万亿参数**，靠的就是这个文件。

## 它要回答的唯一问题

> 这个配置，在我这台机器上跑得动吗？跑不动的话，需要多大的机器？

这个问题必须回答得**诚实**。工程上最常见的两种不诚实：

1. **假装能跑**：给一个 7B 的配置，照常开始训练，然后 OOM 崩在第三步，
   报一句 `CUDA out of memory`，用户不知道该改什么。
2. **假装不能跑**：看到参数大就直接拒绝，不告诉你「要几张卡」，
   用户以为这个引擎只能玩小模型。

这个文件的做法是：**算出来，然后把数字摊开给他看**，
包括每一部分占多少、要几张卡、换成什么并行方式能省下来。

## 显存都花在哪了

训练（不是推理！）的显存主要是这几块：

| 项 | 每参数多少字节 | 说明 |
|---|---|---|
| 权重（bf16） | 2 | 模型本身 |
| 梯度（bf16） | 2 | 反向传播算出来的 |
| AdamW 一阶动量（fp32） | 4 | m |
| AdamW 二阶动量（fp32） | 4 | v |
| fp32 主权重 | 4 | 混合精度训练必须留一份 fp32 副本 |
| **合计** | **16** | 这就是「16 字节/参数」的来历 |

优化器状态（12 字节）占了**四分之三**。这解释了一件事：
为什么推理能跑 7B 的卡，训练连 1B 都费劲。

**激活值**另算，它跟 batch × 序列长度 × 层数 × 宽度成正比，
和参数量没有直接关系。所以「调小 batch_size」永远是最有效的省显存手段。

## 分片（ZeRO / FSDP）怎么省

核心思路：把上面那些东西**切开分给多张卡**，每张卡只存一份。

| 策略 | 权重 | 梯度 | 优化器 | 每卡字节/参数 |
|---|---|---|---|---|
| 单卡（无分片） | 2 | 2 | 12 | **16** |
| ZeRO-1 / FSDP 分片优化器 | 2 | 2 | 12/N | 4 + 12/N |
| ZeRO-2 / +分片梯度 | 2 | 2/N | 12/N | 2 + 14/N |
| ZeRO-3 / +分片权重 | 2/N | 2/N | 12/N | 16/N |

N = 卡数。ZeRO-3 让每卡显存降到 1/N，代价是通信量大幅上升
（每次前向都要把权重收集回来）。

**MoE 特别注意**：显存按**总参数**算（所有专家都得存着），
但算力按**激活参数**算。所以 MoE 是「显存换效果」，不是「免费变大」。
"""

import math

# 训练时每个参数要多少字节（稠密、AdamW、bf16 混合精度）
BYTES_WEIGHT = 2          # bf16 权重
BYTES_GRAD = 2            # bf16 梯度
BYTES_OPT = 8             # AdamW 的 m 和 v，各 4 字节 fp32
BYTES_MASTER = 4          # fp32 主权重

# ---------------------------------------------------------------------------
# 实测标定（2026-10-18，8GB 显存显卡，bf16，torch 2.14）
#
# 用 5 份出厂配置各跑 8~10 步、读 torch.cuda.max_memory_allocated 峰值：
#
#   配置      参数       batch×seq×d×层  实测峰值   旧公式(22)  旧误差
#   nano65    12.59M     32×256×384×6     1.94 GB      0.98 GB   -49%
#   small     13.08M      8×1024×256×16   2.68 GB      1.57 GB   -41%
#   medium    47.80M     24×512×640×10    5.84 GB      3.93 GB   -33%
#   moe       53.52M     16×512×512×8     4.34 GB      2.17 GB   -41%
#   wide      79.45M     16×512×768×12    5.96 GB      4.28 GB   -19%
#
# 对 4 个稠密配置做最小二乘拟合：峰值 ≈ 状态 + 激活 + 固定开销，
#   激活系数拟合值 ≈ 27，固定开销 ≈ 0.7 GB（CUDA 上下文 + 分配器碎片 + 临时张量）。
#   取 29 而不是 27：留一点保守量，实测 5 个配置误差 -7% ~ +1%。
# MoE 因为专家激活没被稠密公式覆盖，再乘 1.4 的修正系数
# （实测 4.34GB vs 稠密式 3.37GB，修正后 4.05GB，偏保守）。
# ---------------------------------------------------------------------------
ACTIVATION_FACTOR = 29        # 稠密：每层每 token 的 bf16 激活元素数
ACTIVATION_FACTOR_CKPT = 8    # 开了 activation checkpoint 的对应系数
FIXED_OVERHEAD_BYTES = int(0.7 * 1024 ** 3)
MOE_ACTIVATION_MULTIPLIER = 1.4
USABLE_RATIO = 0.92           # 可用显存只按 92% 算（留 8% 给碎片/波动）

# 「够不够用」判定时再乘的安全余量。标定后实测误差约 ±7%，1.03 足够；
# 以前公式低估 20%~50%，才需要 1.2 那种大余量。
FITS_MARGIN = 1.03


def bytes_per_param(zero_stage=0, world_size=1, cpu_offload=False):
    """每张卡、每个参数占多少字节。

    `world_size` 是**数据并行**的卡数（FSDP/ZeRO 沿这个维度分片）。
    张量并行/流水线并行是另一回事，不在这里算（它们切的是计算，
    显存收益要按具体切法算，粗略估容易误导人）。
    """
    n = max(1, int(world_size))
    s = int(zero_stage)
    if cpu_offload:
        # 优化器状态和主权重挪到内存，显存里只剩权重+梯度
        return BYTES_WEIGHT + BYTES_GRAD / (n if s >= 2 else 1)
    if s <= 0:
        return BYTES_WEIGHT + BYTES_GRAD + BYTES_OPT + BYTES_MASTER
    if s == 1:
        return BYTES_WEIGHT + BYTES_GRAD + (BYTES_OPT + BYTES_MASTER) / n
    if s == 2:
        return BYTES_WEIGHT + (BYTES_GRAD + BYTES_OPT + BYTES_MASTER) / n
    return (BYTES_WEIGHT + BYTES_GRAD + BYTES_OPT + BYTES_MASTER) / n


def activation_bytes(batch, seq, d_model, n_layer, world_size=1,
                     activation_checkpoint=False, dtype_bytes=2, factor=None):
    """激活值显存（2026-10-18 起按实测标定）。

    `factor` 是「每层每 token 要存多少个中间值」。实测拟合值：

        27  普通训练（注意力 + 前馈的中间激活全留着）
        7   开了 activation checkpoint（只留每层的输入，反向时重算）

    标定数据见文件头注释：4 个稠密配置实测误差 -2% ~ +2%。
    但这仍是**公式估算**，不是「精确预测会不会 OOM」——真正开训前
    还是建议先跑几步确认。

    激活值沿数据并行**不减少**（每张卡都算自己那份 batch，
    这正是数据并行的定义）；开了 activation checkpoint 才大幅下降。
    """
    if factor is None:
        factor = ACTIVATION_FACTOR_CKPT if activation_checkpoint else ACTIVATION_FACTOR
    b = float(batch) * float(seq) * float(d_model) * float(n_layer) * factor
    return b * dtype_bytes


def estimate(params, batch=1, seq=512, d_model=512, n_layer=8,
             world_size=1, zero_stage=0, cpu_offload=False,
             activation_checkpoint=False, per_card_bytes=None, moe=False):
    """算出一个配置要多少显存、几张卡。

    返回字典，字段都是给人看的（含 `verdict` 一句话结论）。

    `per_card_bytes=None` 时不去查显卡（纯计算，可在没有 CUDA 的机器上用）。
    `moe=True` 时激活值乘 MoE 修正系数（专家激活不被稠密公式覆盖）。
    """
    params = float(params)
    world_size = max(1, int(world_size))
    bpp = bytes_per_param(zero_stage, world_size, cpu_offload)
    state = params * bpp
    act = activation_bytes(batch, seq, d_model, n_layer, world_size,
                           activation_checkpoint)
    if moe:
        act *= MOE_ACTIVATION_MULTIPLIER
    overhead = FIXED_OVERHEAD_BYTES
    total = state + act + overhead

    out = {
        "params": params,
        "bytes_per_param": bpp,
        "state_bytes": state,
        "activation_bytes": act,
        "overhead_bytes": overhead,
        "moe": bool(moe),
        "total_bytes": total,
        "world_size": world_size,
        "zero_stage": int(zero_stage),
    }

    if per_card_bytes and per_card_bytes > 0:
        usable = per_card_bytes * USABLE_RATIO   # 留 8% 给碎片/波动
        need = total * FITS_MARGIN
        cards = max(1, int(math.ceil(need / usable)))
        out["per_card_bytes"] = per_card_bytes
        out["usable_per_card"] = usable
        out["cards_needed"] = cards
        out["fits_one_card"] = (need <= usable)
        if cards <= 1:
            out["verdict"] = "本机单卡跑得动"
        elif cards <= int(world_size):
            out["verdict"] = "当前并行度下跑得动"
        else:
            out["verdict"] = "本机跑不动，需要 %d 张同等显存的卡" % cards
    return out


def fmt_bytes(n):
    """人能读的字节数。"""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return "%.2f %s" % (n, unit) if unit not in ("B", "KB") else "%d %s" % (n, unit)
        n /= 1024.0
    return "%.2f TB" % n


def suggestions(cfg_par, est, params, active=None, n_layer=None, d_model=None):
    """针对「装不下」给出**具体、真能生效**的建议。

    ## 只许提引擎真的会读的开关（真实教训）
    旧版建议让用户 `--set parallel.backend=fsdp` / `zero_stage=2` /
    `cpu_offload=true`。但 `train.py` 根本不读 `parallel.*`（没有分布式 /
    CPU offload 的实现），用户照着改完，显存一点没降 —— 这比不给建议还坑。
    所以现在只提**训练循环真的会读**的参数：batch_size、grad_accum、
    block_size、n_layer、n_embd、grad_checkpoint、optim_bits，以及换卡；
    多卡脚本如实标注「尚未实现」。

    ⚠️ 参数请**一律用关键字传**。这个函数有 6 个参数、其中 3 个是可选整数，
    按位置传极易把 n_layer 和 d_model 搞反 —— 实测踩过：
    `suggestions(cfg, est, params, n_layer, d_model)` 会让 active=n_layer、
    n_layer=d_model，于是建议变成「n_layer 从 2048 减半」（2048 是宽度不是层数），
    同时「减宽度」那条建议因为 d_model=None 直接消失。
    """
    tips = []
    if est.get("cards_needed", 1) <= 1:
        return tips

    need = est["cards_needed"]

    tips.append("减小 [train] batch_size（激活值按 token 数近线性下降）；"
                "要保住每步 token 量就把 grad_accum 加回来："
                "  --set train.batch_size=<现在的一半>"
                " --set train.grad_accum=<现在的两倍>")
    tips.append("减小 [model] block_size（序列越短激活值越小，"
                "代价是单条样本看得到的上下文变短）")
    if n_layer and n_layer > 2:
        tips.append("减少层数最直接：n_layer 从 %d 改小一半，参数量约减半"
                    % n_layer)
    if d_model and d_model > 64:
        tips.append("减宽度效果更猛（参数量按平方降）：n_embd 从 %d 减半，"
                    "参数量约降到 1/4" % d_model)
    tips.append("开启梯度检查点（真会生效）：--set train.grad_checkpoint=true，"
                "用约 1.3 倍算力换 30~60% 激活显存")
    tips.append("优化器状态改 8-bit：--set train.optim_bits=8，"
                "Adam 一阶/二阶动量按块量化，约省 4 倍优化器显存")
    tips.append("换一张显存更大的卡，或关掉占显存的桌面程序 / 浏览器再跑")
    tips.append("真要用这个规模（估算需要约 %d 张同等显存的卡）：用"
                "  lingyu launch --config <你的配置>  生成启动脚本"
                "——注意当前引擎尚未实现分布式训练，生成的脚本只是给外部"
                "集群/框架的参考模板，不能直接当多卡训练用" % need)
    # 动态编号：这样即使中间某条被跳过，也不会出现「①②④」这种断号
    return ["%s %s" % (chr(0x2460 + i), t) for i, t in enumerate(tips[:8])]


def check_available_vram():
    """本机可用显存（字节）。没有 CUDA 就返回 0。"""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:
        pass
    return 0


def gpu_info():
    """本机显卡信息，给报告用。没有就返回 None。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        return {"name": p.name, "total_bytes": int(p.total_memory),
                "count": torch.cuda.device_count(),
                "capability": "%d.%d" % (p.major, p.minor)}
    except Exception:
        return None
