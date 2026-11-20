# -*- coding: utf-8 -*-
'''训练循环：从随机初始化到能写中文。

一次训练在做什么（按执行顺序）：

    1. 读出分词器、配置、数据
    2. 建模型、建优化器（AdamW）、算出学习率计划（预热 + 余弦衰减）
    3. 循环 max_steps 次：
         a. 取一批数据（随机片段）
         b. 前向算 loss（预测下一个 token 有多准）
         c. 反向算梯度
         d. 梯度裁剪（防止偶发的超大梯度把模型带飞）
         e. 更新参数
         f. 每隔一段：评估验证集、存档、打印进度、试写一小段
    4. 存最终模型

关键概念（都是踩过的坑总结）：

    梯度累积（grad_accum）
        显存不够放不下大批量时，可以分几次算梯度再一起更新，
        等效于大批量但显存只按小批量算。

    混合精度（bf16）
        前向反向用 16 位算（快、省显存），参数更新仍用 32 位（准）。
        bf16 不会像 fp16 那样溢出，所以不需要梯度缩放器。

    梯度裁剪（grad_clip）
        把梯度整体缩放，使它的范数不超过阈值。偶尔一个坏批次可能产生巨大梯度，
        不裁剪的话一步就能把练了几小时的模型毁掉。默认 1.0。

    验证集 loss
        训练集 loss 一直降不代表模型变好（可能只是背下来了）。
        必须同时看验证集 loss，它开始上升就是过拟合。
'''

import math
import os
import random
import time

import torch

from . import accelerators, compression, utils
from .data import (BatchPrefetcher, TokenDataset, eval_loss, perplexity,
                   bits_per_byte)
from .model import build_model, config_from_section
from .tokenizer import BOS, EOS, load_tokenizer
from .version import CKPT_FORMAT, __version__ as LINGYU_VERSION


def lr_at(step, cfg):
    '''算第 step 步该用多大学习率。

    形状：先线性预热（warmup），再余弦衰减到 min_lr。
    预热是因为一开始参数是随机的，梯度方向很乱，直接上大学习率容易崩。
    '''
    lr = float(cfg["learning_rate"])
    min_lr = float(cfg["min_lr"])
    warmup = max(1, int(cfg["warmup_iters"]))
    # lr_decay_iters 的 0（或缺失）表示「跟随 max_steps」，不是「不衰减」。
    decay = int(cfg.get("lr_decay_iters") or 0)
    if decay <= 0:
        decay = int(cfg.get("max_steps") or 0)
    decay = max(warmup + 1, decay if decay > 0 else warmup + 1)
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > decay:
        return min_lr
    ratio = (step - warmup) / float(decay - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer(model, cfg):
    '''AdamW。权重衰减只加在矩阵上，归一化和偏置不加。

    为什么分开：对归一化的缩放向量做衰减会把它们推向 0，反而有害；
    对权重矩阵做衰减能抑制过拟合。
    '''
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": float(cfg["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    betas = (float(cfg["beta1"]), float(cfg["beta2"]))
    lr = float(cfg["learning_rate"])
    bits = int(cfg.get("optim_bits", 32) or 32)
    if bits not in (8, 32):
        raise ValueError("[train] optim_bits 只支持 32 或 8（收到 %r）" % bits)
    if bits == 8:
        # 8-bit 优化器：动量按 2048 个一组做 absmax 分块量化
        # （Dettmers et al., arXiv:2110.02871）。
        from .optim import AdamW8bit
        return AdamW8bit(groups, lr=lr, betas=betas)
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


#: 当前这次训练/微调用的配置文件路径。由 `set_config_path()` 设置，
#: 写进断点的旁车 JSON —— GUI 断点页据此知道「这个断点是用哪份配置训的」。
_CONFIG_PATH = None


def set_config_path(path):
    global _CONFIG_PATH
    _CONFIG_PATH = path


def ckpt_extra(tok, tok_path, template=None, config_path=None):
    """断点里要记的额外信息。

    ★ `tokenizer_fingerprint` 是关键：光记路径没用 —— 同一个路径下
    `prepare` 重跑一次，词表大小可能一点没变（都是 8192），
    可每个 token 的 ID 全变了，续训出来的模型直接报废还不报错。
    记指纹之后就能当场比出来。
    """
    extra = {"tokenizer_path": tok_path,
             "tokenizer_fingerprint": tok.fingerprint()}
    if template:
        extra["chat_template"] = template
    cfg_path = config_path or _CONFIG_PATH
    if cfg_path:
        extra["config_path"] = cfg_path
    return extra


def _rng_state():
    """收集可序列化的随机数状态（python / torch / cuda / numpy）。

    续训时恢复它，dropout 与数据采样顺序才能接着原来的流，而不是从头重放。
    """
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def _restore_rng(state):
    """恢复 _rng_state() 存下的状态。返回是否成功（失败只 warn，不中断续训）。

    ★ 注意：`load_checkpoint(..., map_location=device)` 会把断点里**所有**张量
    都搬到目标设备，包括 RNG 的 ByteTensor；而 `torch.set_rng_state()` 只接受
    **CPU uint8 张量**。实测踩过：CUDA 续训时报
    `RNG state must be a torch.ByteTensor`，RNG 静默没恢复。这里统一搬回 CPU。
    """
    if not state:
        return False
    try:
        def _cpu_bytes(v):
            if torch.is_tensor(v):
                return v.detach().to("cpu", torch.uint8)
            return torch.tensor(v, dtype=torch.uint8)

        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(_cpu_bytes(state["torch"]))
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_cpu_bytes(s) for s in state["cuda"]])
        if "numpy" in state:
            import numpy as np
            np.random.set_state(state["numpy"])
        return True
    except Exception as e:      # RNG 状态坏了不该让几小时的续训直接起不来
        utils.warn("随机数状态没能恢复（%s），续训后 dropout/采样顺序会重放" % e)
        return False


# 续训结构一致性检查表：(配置键, 人话标签)
_STRUCT_KEYS = (
    ("n_layer", "层数 n_layer"),
    ("d_model", "宽度 n_embd"),
    ("n_head", "头数 n_head"),
    ("n_kv_head", "KV 头数 n_kv_head"),
    ("d_ff", "前馈 d_ff"),
    ("block_size", "上下文 block_size"),
    ("vocab_size", "词表大小"),
    ("tie_weights", "权重共享 tie_weights"),
    ("bias", "线性层 bias"),
    ("use_moe", "是否用 MoE"),
)


def structure_diffs(old_cfg, new_cfg):
    """对比断点配置与当前模型配置。

    返回 (diffs, unchecked)：
      diffs     = [(标签, 断点值, 当前值)]，非空就必须拒绝续训
      unchecked = 旧断点里没有的字段标签（老格式无法核对，只能 warn）
    """
    old_cfg = old_cfg or {}
    diffs, unchecked = [], []
    for key, label in _STRUCT_KEYS:
        a, b = old_cfg.get(key), (new_cfg or {}).get(key)
        if a is None:
            unchecked.append(label)
            continue
        if b is not None and a != b:
            diffs.append((label, a, b))
    return diffs, unchecked


def _write_ckpt_meta(path, payload):
    """在断点旁边写一份小 JSON（旁车文件）。

    断点本身（含 optimizer 动量）动辄 300MB+，`torch.load` 一次几百毫秒，
    断点页列 20 个就要卡十来秒。旁车文件只有几百字节，读它是瞬时的；
    断点页 / 续训选择器先读它，读不到再退回 torch.load。
    """
    import json
    meta = {
        "ckpt_format": payload.get("ckpt_format"),
        "lingyu_version": payload.get("lingyu_version"),
        "step": payload.get("step"),
        "best_val": payload.get("best_val"),
        "model_cfg": payload.get("config") or {},
        "tokenizer_path": payload.get("tokenizer_path"),
        "tokenizer_fingerprint": payload.get("tokenizer_fingerprint"),
        "chat_template": payload.get("chat_template"),
        "config_path": payload.get("config_path"),
        "size": os.path.getsize(path) if os.path.exists(path) else 0,
        "mtime": time.time(),
    }
    tmp = path + ".json.tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    os.replace(tmp, path + ".json")


def read_checkpoint_meta(path):
    """读旁车元数据（不 import torch）。老断点没有，返回 {}。"""
    import json
    try:
        with open(path + ".json", "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_checkpoint(path, model, optimizer, step, best_val, cfg_snapshot, extra=None):
    '''存断点。存了优化器状态，续训才能接得上（否则动量丢失、loss 会抖一下）。

    用「先写 .tmp 再原子替换」的写法：万一写入过程被打断（Ctrl+C、
    进程被杀、磁盘满），旧的断点仍然完好，不会留下一个半截的坏文件。

    断点契约（load_checkpoint 会校验）：
      ckpt_format / lingyu_version / step / config(model_cfg) / rng /
      tokenizer_fingerprint（由 extra 带入）
    '''
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    raw = getattr(model, "_orig_mod", model)
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "best_val": float(best_val),
        "config": cfg_snapshot,
        "ckpt_format": CKPT_FORMAT,
        "lingyu_version": LINGYU_VERSION,
        "rng": _rng_state(),
    }
    if extra:
        payload.update(extra)
    meta_src = payload
    tmp = path + ".tmp"
    # 上次保存被打断可能留下一个 0 字节的 .tmp（原子替换没跑到），
    # 这里先清掉，免得你看目录时被它吓到以为存档坏了。
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    try:
        if compression.pack_enabled():
            # 无损压缩：逐字节平面 + zlib，自有容器（避开 torch.save 对 bytes 的 1.5x 膨胀）
            compression.save_checkpoint_file(path, meta_src)
        else:
            torch.save(meta_src, tmp)
            os.replace(tmp, path)
        # 旁车文件失败绝不能让「断点已经存好」变成异常（只是 GUI 读得慢点）
        try:
            _write_ckpt_meta(path, meta_src)
        except OSError:
            pass
    except BaseException:
        # 写失败就清掉半截文件，避免留下一个 0 字节的 .tmp 让人误会
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def load_checkpoint(path, map_location="cpu"):
    """读断点，并做格式契约校验（老格式兼容读取，但会明确 warn）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到断点文件: " + path)
    ck = compression.load_checkpoint_file(path, map_location=map_location)
    if not isinstance(ck, dict) or "model" not in ck:
        raise ValueError("这不是灵语断点（缺少 model 字段）：%s" % path)
    fmt = ck.get("ckpt_format")
    if fmt is None:
        utils.warn("这是旧格式断点（没有 ckpt_format 字段），无法核对格式版本；"
                   "仍按兼容模式读取：%s" % os.path.abspath(path))
    else:
        try:
            fmt = int(fmt)
        except (TypeError, ValueError):
            raise ValueError("断点里的 ckpt_format 字段不是整数：%r（文件 %s）"
                             % (fmt, os.path.abspath(path)))
        if fmt > CKPT_FORMAT:
            raise ValueError(
                "断点格式版本 %d 比当前引擎支持的 %d 新，请先升级灵语再续训：%s"
                % (fmt, CKPT_FORMAT, os.path.abspath(path)))
    return ck


@torch.no_grad()
def quick_sample(model, tok, scfg, device, n=80, prompt=None):
    '''训练途中试写一小段，直观感受进度。'''
    was_training = model.training
    model.eval()
    if not prompt:
        prompt = scfg.get("prompt") or "他"
    ids = tok.encode(prompt) or [BOS]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=int(n),
        temperature=float(scfg.get("temperature", 0.9)),
        top_k=int(scfg.get("top_k", 40)) or None,
        top_p=float(scfg.get("top_p", 0.95)) or None,
        # EOS 从分词器取（老分词器没有 eos_id 属性时回退到模块常量 EOS=2）
        eos_id=int(getattr(tok, "eos_id", EOS)), use_cache=True)
    if was_training:
        model.train()
    return tok.decode(out[0].tolist())


def train(cfg, device=None, resume=None, force=False,
          init_from=None, config_path=None):
    '''主训练函数。cfg 是 {model:.., train:.., data:.., sample:..}。

    force=False（默认）时，如果 out_dir 里已有存档而又没指定 --resume，
    会停下并提示 —— 避免手滑把练了很久的模型静默覆盖掉。
    '''
    from .config import get
    t_start = time.time()
    set_config_path(config_path)

    # -------------------------------------------------- 覆盖保护（先于一切重活）
    out_dir = get(cfg, "train", "out_dir", "out")
    old_ckpt = os.path.join(out_dir, "ckpt.pt")
    if not resume and not force and os.path.isfile(old_ckpt):
        try:
            import torch as _t
            _ck = _t.load(old_ckpt, map_location="cpu", weights_only=False)
            _step = int(_ck.get("step", 0))
            _val = float(_ck.get("best_val", float("inf")))
            _has_best = os.path.isfile(os.path.join(out_dir, "best.pt"))
        except Exception:
            _step, _val, _has_best = -1, float("inf"), False
        # 提示信息一律走 stdout —— 走 stderr 的话在 PowerShell 里会被包成
        # NativeCommandError，后半段提示就看不见了（实测踩过）。
        utils.warn("输出目录里已经有存档了：%s" % old_ckpt)
        if _step >= 0:
            utils.warn("  它在第 %s 步" % "{:,}".format(_step))
            if _val == float("inf") or _val != _val:
                # 还没跑过验证就中断了，best_val 仍是初始的无穷大
                utils.info("  （还没做过验证集评估，所以没有「最好 loss」）")
                if not _has_best:
                    utils.info("  （也没有 best.pt —— 它要到第一次验证时才会生成）")
            else:
                utils.info("  最好验证 loss：%.4f" % _val)
        print()
        utils.info("你想干什么？三选一：")
        utils.info("  ① 接着往下训     -> 加 --resume")
        utils.info("  ② 删档从头重训   -> 加 --force（会覆盖），或先删 out 目录")
        utils.info("  ③ 换个目录不覆盖 -> --set train.out_dir=%s"
                   % os.path.join(os.path.dirname(os.path.abspath(out_dir)),
                                  "试验2"))
        print()
        utils.info("删档重训的完整做法：")
        utils.info("  Remove-Item -Recurse -Force \"%s\"" % os.path.abspath(out_dir))
        utils.info("  然后重新运行 train（不加 --resume）")
        raise FileExistsError("已有存档，已停下以免覆盖")

    # ---------------------------------------------------------------- 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = str(device).startswith("cuda")
    accel = accelerators.apply(device)
    utils.title("设备")
    utils.info("使用: %s" % device)
    utils.info("加速: %s" % accelerators.describe(accel))
    if use_cuda:
        pr = torch.cuda.get_device_properties(0)
        utils.info("显卡: %s  显存 %.2f GB" % (pr.name, pr.total_memory / 1024 ** 3))

    # ---------------------------------------------------------------- 分词器
    tok_path = get(cfg, "data", "tokenizer_path", None)
    data_dir = get(cfg, "data", "data_dir", None)
    if not tok_path and data_dir:
        tok_path = os.path.join(data_dir, "tokenizer.txt")
    if not tok_path or not os.path.isfile(tok_path):
        raise FileNotFoundError(
            "找不到分词器文件: %s\n"
            "先运行数据准备:  python -m lingyu prepare --raw <语料目录> --data <输出目录>"
            % tok_path)
    tok = load_tokenizer(tok_path)
    utils.info("分词器: %s，词表 %s，文件 %s"
               % ("字符级" if tok.kind == "char" else "BPE",
                  "{:,}".format(tok.vocab_size), tok_path))

    # ---------------------------------------------------------------- 模型配置
    model_cfg = config_from_section(cfg.get("model", {}), tok.vocab_size)
    block_size = int(model_cfg["block_size"])

    # ---------------------------------------------------------------- 数据
    if not data_dir:
        raise FileNotFoundError("配置里缺少 [data] data_dir")
    # 语料 bin 和分词器必须来自同一次 prepare。不配套时**形状对得上、
    # 训练也不报错，但解码出来全是乱码** —— 直接拦下来，别浪费电费。
    from .data import check_data_tokenizer
    _mismatch = check_data_tokenizer(data_dir, tok)
    if _mismatch:
        utils.warn(_mismatch)
    ds_train = TokenDataset(os.path.join(data_dir, "train.bin"), block_size,
                            device=device, seed=int(get(cfg, "train", "seed", 1234)))
    # 验证集是**可选**的，不能因为它而训不了。
    #
    # 踩过的坑：这里原来只 catch FileNotFoundError，可小语料 + val_ratio
    # 切出来的 val.bin 常常比 block_size 还短，TokenDataset 会抛 ValueError
    # —— 于是「验证集太小」把整个训练拦死了。用户看到的是一句
    # 「数据太短了：只有 48 个 token，但 block_size=96」，
    # 完全看不出问题出在验证集上（训练集明明有 2310 个 token）。
    #
    # 正确做法：验证集有问题就**降级**（不验证 + 说清楚原因），
    # 并且告诉用户怎么才能有验证集。
    try:
        ds_val = TokenDataset(os.path.join(data_dir, "val.bin"), block_size,
                              device=device, seed=999)
    except FileNotFoundError:
        ds_val = None
        utils.warn("没有 val.bin，将不做验证集评估（强烈建议做验证集）")
    except ValueError as e:
        ds_val = None
        utils.warn("验证集用不了，已跳过验证：%s" % e)
        utils.info("  原因通常是语料太小，切出来的 val.bin 比 block_size 还短。")
        utils.info("  想要有验证集，三选一：")
        utils.info("    ① 多加点语料（最实在）")
        utils.info("    ② 把 [model] block_size 调小")
        utils.info("    ③ 重新 prepare 时调大 --val-ratio（默认 0.02）")

    # ---------------------------------------------------------------- 模型
    utils.set_seed(int(get(cfg, "train", "seed", 1234)))
    model = build_model(model_cfg, device)
    model = accelerators.maybe_compile(model, device)
    _tcfg = cfg.get("train", {})
    model.grad_checkpoint = bool(_tcfg.get("grad_checkpoint", False))
    model.loss_chunk = int(_tcfg.get("loss_chunk", 0) or 0)
    if model.grad_checkpoint:
        utils.info("梯度检查点    : 开（用约 1.3 倍算力换 30~60% 激活显存）")
    if int(_tcfg.get("optim_bits", 32) or 32) == 8:
        utils.info("优化器        : AdamW 8-bit（动量分块量化，约省 4 倍）")
    total = sum(p.numel() for p in model.parameters())
    active = (model.active_param_count() if hasattr(model, "active_param_count")
              else total)

    utils.title("模型")
    utils.info("总参数        : %s (%.2f M)" % ("{:,}".format(total), total / 1e6))
    if model_cfg["use_moe"]:
        utils.info("激活参数      : %s (%.2f M)  <- 每个 token 实际只算这么多"
                   % ("{:,}".format(active), active / 1e6))
    utils.info("层数/宽度/头数: n_layer=%d  n_embd=%d  n_head=%d  head_dim=%d"
               % (model_cfg["n_layer"], model_cfg["d_model"], model_cfg["n_head"],
                  model_cfg["head_dim"]))
    utils.info("前馈维度 d_ff : %d%s" % (model_cfg["d_ff"],
                                     "（自动算的）" if not get(cfg, "model", "d_ff", 0) else ""))
    utils.info("KV 头数       : %d（Q 头 %d，KV cache 省 %.0f%%）"
               % (model_cfg["n_kv_head"], model_cfg["n_head"],
                  100.0 * (1 - model_cfg["n_kv_head"] / float(model_cfg["n_head"]))))
    per_tok = 2 * model_cfg["n_layer"] * model_cfg["n_kv_head"] * model_cfg["head_dim"] * 2
    utils.info("KV cache      : %s / token  →  %d 上下文共 %s"
               % (utils.fmt_bytes(per_tok), block_size,
                  utils.fmt_bytes(per_tok * block_size)))
    if model_cfg["use_moe"]:
        e = model_cfg["n_expert"]
        k = model_cfg["n_expert_per_tok"]
        s = model_cfg["n_shared_expert"]
        utils.info("MoE           : %d 专家，每 token 走 %d 个 + %d 个共享专家"
                   % (e, k, s))
    utils.info("混合精度      : %s" % ("bfloat16" if use_cuda else "关闭（CPU 训练）"))
    utils.info("训练数据      : %s token" % "{:,}".format(len(ds_train)))
    if ds_val is not None:
        utils.info("验证数据      : %s token" % "{:,}".format(len(ds_val)))

    # ---------------------------------------------------------------- 优化器
    opt = configure_optimizer(model, cfg.get("train", {}))
    start_step = 0
    best_val = float("inf")
    out_dir = get(cfg, "train", "out_dir", "out")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if init_from:
        if resume:
            raise ValueError("--init-from 和 --resume 只能二选一")
        ck = load_checkpoint(init_from, map_location=device)
        old_cfg = ck.get("config") or {}
        diffs, unchecked = structure_diffs(old_cfg, model_cfg)
        if unchecked:
            utils.warn("这是旧格式断点，以下结构项无法核对：%s"
                       % "、".join(unchecked))
        if diffs:
            utils.err("模型结构和要继承的断点对不上，不能当作起点：")
            for label, a, b in diffs:
                utils.info("  %-16s 断点里是 %-8s，现在配置是 %s" % (label, a, b))
            raise ValueError("模型结构与起点断点不一致")
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError as e:
            utils.err("加载起点断点失败：%s" % e)
            raise ValueError("起点断点加载失败")
        utils.ok("已用 %s 的权重初始化（原第 %s 步；步数从 0 开始，优化器状态不复用）"
                 % (init_from, "{:,}".format(int(ck.get("step", 0)))))
    if resume:
        ck = load_checkpoint(resume, map_location=device)
        # ---- 结构一致性检查（放在 load 之前，报错才看得懂）----
        # 直接 load_state_dict 失败时，PyTorch 会吐一屏 "size mismatch"，
        # 完全看不出该改哪个参数。这里先比对配置，给出人话。
        old_cfg = ck.get("config") or {}
        diffs, unchecked = structure_diffs(old_cfg, model_cfg)
        if unchecked:
            utils.warn("这是旧格式断点，以下结构项无法核对：%s"
                       % "、".join(unchecked))
        if diffs:
            utils.err("模型结构和断点对不上，没法续训：")
            for label, a, b in diffs:
                utils.info("  %-16s 断点里是 %-8s，现在配置是 %s" % (label, a, b))
            print()
            utils.info("为什么会这样：配置里的模型结构和当初训练时不一样了")
            utils.info("续训必须用和「当初训练时同样的结构」。三选一：")
            utils.info("  ① 把配置改回去，再 --resume")
            for label, a, _b in diffs[:2]:
                key = {"层数 n_layer": "n_layer", "宽度 n_embd": "n_embd",
                       "头数 n_head": "n_head", "前馈 d_ff": "d_ff",
                       "KV 头数 n_kv_head": "n_kv_head",
                       "上下文 block_size": "block_size",
                       "权重共享 tie_weights": "tie_weights",
                       "线性层 bias": "bias",
                       "词表大小": None, "是否用 MoE": "use_moe"}.get(label)
                if key == "d_ff" and a == 0:
                    continue
                if key:
                    utils.info("       在 [model] 里写 %s = %s" % (key, a))
            utils.info("  ② 不改配置，用 --set 临时对上，例如：")
            utils.info("       --resume --set model.n_layer=%s --set model.n_embd=%s"
                       % (old_cfg.get("n_layer", "?"), old_cfg.get("d_model", "?")))
            utils.info("  ③ 就想要新结构 -> 删档重训：加 --force（或删掉 %s）"
                       % os.path.abspath(out_dir))
            raise ValueError("模型结构与断点不一致，已停下")
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError as e:
            utils.err("加载断点失败：%s" % e)
            utils.info("通常是模型结构和断点不一致。断点结构：")
            for k in ("n_layer", "d_model", "n_head", "d_ff", "vocab_size"):
                if old_cfg.get(k) is not None:
                    utils.info("  %-12s = %s" % (k, old_cfg[k]))
            raise ValueError("断点加载失败，已停下（建议删档重训 --force）")
        if ck.get("optimizer") is not None:
            try:
                opt.load_state_dict(ck["optimizer"])
            except Exception as e:
                utils.warn("优化器状态没接上（%s），动量会重新开始" % e)
        start_step = int(ck.get("step", 0))
        best_val = float(ck.get("best_val", float("inf")))
        # 恢复 RNG：dropout 与数据采样顺序接着原来的流，避免续训后重放
        rng = ck.get("rng")
        if rng:
            if _restore_rng(rng):
                utils.info("随机数状态已从断点恢复")
        else:
            utils.warn("旧格式断点没有 RNG 状态，续训后 dropout/数据采样顺序会从头重放")
        # ---- 词表一致性检查 ----
        # 断点里记了训练时用的分词器。如果数据目录被新的 prepare 覆盖过，
        # 词表「大小」可能碰巧一样（比如都是 8192），但每个 token 的含义
        # 完全不同 —— 那样续训出来的模型会直接报废，而且不会报任何错。
        #
        # 优先比**指纹**（能发现「同路径、同大小、ID 全变」的情况）；
        # 老断点没有指纹时退回比路径。
        old_fp = ck.get("tokenizer_fingerprint")
        new_fp = tok.fingerprint()
        if old_fp:
            if old_fp != new_fp:
                utils.err("断点用的分词器和现在这个**不是同一份词表**：")
                utils.info("  断点记录指纹: %s" % old_fp)
                utils.info("  当前词表指纹: %s" % new_fp)
                utils.info("  断点记录路径: %s" % (ck.get("tokenizer_path") or "(未记录)"))
                utils.info("  当前分词器  : %s" % tok_path)
                print()
                utils.info("为什么必须停下：续训等于让模型接着学一套**不同的字母表**。")
                utils.info("token id 只是词表下标，两边 ID 对不上时，同一个数字")
                utils.info("在旧模型眼里是「剑」、在新语料里是「的」—— 模型会报废，")
                utils.info("而且训练过程中**一点错都不会报**。")
                utils.info("两个选择：")
                utils.info("  ① 用当初那份分词器（把旧 data 目录找回来）")
                utils.info("  ② 就拿新数据从头练：加 --force")
                raise ValueError("分词器与断点不配套，已停下")
        elif ck.get("tokenizer_path"):
            same_path = os.path.normcase(os.path.abspath(ck["tokenizer_path"])) == \
                        os.path.normcase(os.path.abspath(tok_path))
            if not same_path:
                utils.warn("断点训练时用的分词器不是现在这个：")
                utils.info("  断点记录: %s" % ck["tokenizer_path"])
                utils.info("  现在使用: %s" % tok_path)
                utils.warn("如果两个文件内容不同，续训会让模型报废（且不会报错）。")
                utils.info("建议：确认无误再继续；或加 --force 从头重训。")
        utils.ok("已从 %s 续训，从第 %d 步继续" % (resume, start_step))

    amp_dtype = torch.bfloat16 if use_cuda else None
    use_amp = amp_dtype is not None

    batch_size = int(get(cfg, "train", "batch_size", 32, required=True))
    grad_accum = max(1, int(get(cfg, "train", "grad_accum", 1)))
    max_steps = int(get(cfg, "train", "max_steps", 5000, required=True))
    grad_clip = float(get(cfg, "train", "grad_clip", 1.0))
    eval_interval = int(get(cfg, "train", "eval_interval", 250))
    eval_iters = int(get(cfg, "train", "eval_iters", 50))
    log_interval = max(1, int(get(cfg, "train", "log_interval", 10)))
    ckpt_interval = int(get(cfg, "train", "ckpt_interval", 500))
    aux_coef = float(get(cfg, "train", "aux_loss_coef", 0.01))
    z_coef = float(get(cfg, "train", "z_loss_coef", 0.001))
    do_sample = bool(get(cfg, "train", "sample_while_train", True))
    sample_interval = int(get(cfg, "train", "sample_interval", 0))

    tokens_per_step = batch_size * grad_accum * block_size
    utils.title("训练计划")
    utils.info("总步数        : %s" % "{:,}".format(max_steps))
    utils.info("批大小        : %d × %d 累积 = %d 条/步（%s token/步）"
               % (batch_size, grad_accum, batch_size * grad_accum,
                  "{:,}".format(tokens_per_step)))
    utils.info("学习率        : %.2e（预热 %d 步 → 余弦降到 %.2e）"
               % (float(get(cfg, "train", "learning_rate", 1e-3)),
                  int(get(cfg, "train", "warmup_iters", 200)),
                  float(get(cfg, "train", "min_lr", 1e-4))))
    utils.info("总吞吐        : %s token（相当于把数据集过 %.2f 遍）"
               % ("{:,}".format(tokens_per_step * max_steps),
                  tokens_per_step * max_steps / max(1, len(ds_train))))
    utils.info("输出目录      : %s" % os.path.abspath(out_dir))
    if model_cfg["use_moe"]:
        utils.info("MoE 辅助损失  : aux=%.4f  z=%.4f（平衡专家 + 稳定路由）"
                   % (aux_coef, z_coef))

    # ---------------------------------------------------------------- 记录
    os.makedirs(out_dir, exist_ok=True)
    # GUI 监控页据此把「预训练 / SFT」两套数据分开，并把 tok/s 算对（审计 1.2）
    utils.write_run_meta(
        out_dir, "pretrain",
        max_steps=int(max_steps), start_step=int(start_step),
        batch_size=int(batch_size), grad_accum=int(grad_accum),
        block_size=int(block_size), tokens_per_step=int(tokens_per_step),
        n_params=int(sum(p.numel() for p in model.parameters())),
        device=str(device), resume=start_step > 0,
        seed=int(get(cfg, "train", "seed", 1234)),
        dataset_tokens=int(len(ds_train)),
    )
    utils.emit_event(
        "start", task="train", config=config_path,
        max_steps=int(max_steps), start_step=int(start_step),
        batch_size=int(batch_size), grad_accum=int(grad_accum),
        block_size=int(block_size), tokens_per_step=int(tokens_per_step),
        device=str(device), resume=start_step > 0)
    hist_path = os.path.join(out_dir, "history.csv")
    if start_step == 0 or not os.path.isfile(hist_path):
        with open(hist_path, "w", encoding="utf-8") as fp:
            fp.write("step,train_loss,val_loss,lr,grad_norm,elapsed_sec\n")
    log_path = os.path.join(out_dir, "train.log")

    def log_line(s):
        print(s, flush=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(s + "\n")

    # ---------------------------------------------------------------- 主循环
    model.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    t_last = time.time()
    loss_hist = []
    step = start_step
    interrupted = False
    stop_seen = False
    scfg = cfg.get("sample", {})

    utils.title("开始训练（按 Ctrl+C 可安全中断，会先存一次断点）")
    # ★ 后台预取训练 batch：与 GPU 计算重叠，批越大/CPU 越慢收益越明显。
    #   LINGYU_PREFETCH=0 可退回原来的同步取数（排查用）。
    _prefetch_on = os.environ.get("LINGYU_PREFETCH", "1").strip().lower() \
        not in ("0", "false", "no", "off")
    batch_stream = (BatchPrefetcher(ds_train, batch_size)
                    if _prefetch_on else None)
    try:
        while step < max_steps:
            if utils.stop_requested():
                # 外部优雅停止协议：LINGYU_STOP_FILE 指向的文件出现 → 存断点后正常退出
                stop_seen = True
                log_line("")
                log_line("  [停止] 收到停止请求（%s），保存断点后正常退出……"
                         % utils.stop_file())
                break
            lr = lr_at(step, cfg.get("train", {}))
            for g in opt.param_groups:
                g["lr"] = lr

            micro_loss = None        # GPU 上累加，避免每个 micro-step 一次 .item() 同步
            for _ in range(grad_accum):
                if batch_stream is not None:
                    x, y = next(batch_stream)
                else:
                    x, y = ds_train.get_batch(batch_size)
                with torch.autocast(device_type="cuda" if use_amp else "cpu",
                                    dtype=amp_dtype, enabled=use_amp):
                    _, loss, aux = model(x, y)
                    tot = loss
                    if aux is not None:
                        tot = tot + aux_coef * aux["aux_loss"] + z_coef * aux["z_loss"]
                    tot = tot / grad_accum
                tot.backward()
                loss_d = loss.detach()
                micro_loss = loss_d if micro_loss is None else micro_loss + loss_d

            gnorm = 0.0
            if grad_clip > 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            step_loss = micro_loss / grad_accum
            loss_hist.append(step_loss)
            if len(loss_hist) > 10000:      # 长训练别让历史无限涨
                loss_hist = loss_hist[-1000:]

            # --- 进度 ---
            if step % log_interval == 0 or step == start_step + 1:
                now = time.time()
                if step % log_interval == 0:
                    dt = max(1e-6, now - t_last)
                    tps = tokens_per_step * log_interval / dt
                else:
                    tps = 0.0
                t_last = now
                recent = loss_hist[-log_interval:]
                avg = float(torch.stack(recent).mean().item())
                extra = ("gnorm %.2f" % gnorm) if gnorm else ""
                utils.emit_event("step", task="train", step=int(step),
                                 total=int(max_steps), loss=avg, lr=float(lr),
                                 grad_norm=float(gnorm), tok_s=float(tps),
                                 elapsed=float(now - t0),
                                 tokens_per_step=int(tokens_per_step),
                                 dataset_tokens=int(len(ds_train)))
                log_line(utils.human_step_line(step, max_steps, avg, lr, 0.0, tps, extra))

            # --- 验证 ---
            if eval_interval > 0 and (step % eval_interval == 0 or step >= max_steps):
                v = float("nan")
                val_tokens = 0
                if ds_val is not None:
                    v, val_tokens = eval_loss(
                        model, ds_val, batch_size, eval_iters, device, amp_dtype,
                        block_size=model_cfg["block_size"])
                    msg = "  [验证] 第 %d 步  验证 loss %.4f  (困惑度 %.2f" % (
                        step, v, perplexity(v))
                    nb = ds_val.meta.get("n_bytes")
                    if nb and val_tokens:
                        msg += "  每字节 %.3f bit" % bits_per_byte(v, val_tokens, nb)
                    msg += ")"
                    log_line(msg)
                    utils.emit_event("eval", task="train", step=int(step),
                                     val_loss=float(v), best=bool(v < best_val))
                    if v < best_val:
                        best_val = v
                        save_checkpoint(os.path.join(out_dir, "best.pt"), model, opt,
                                        step, best_val, model_cfg,
                                        extra=ckpt_extra(tok, tok_path))
                        utils.emit_event("ckpt", task="train", step=int(step),
                                         path=os.path.abspath(
                                             os.path.join(out_dir, "best.pt")),
                                         best=True)
                        log_line("           ^ 目前最好，已存 best.pt")
                recent = loss_hist[-log_interval:] if loss_hist else []
                recent_mean = (float(torch.stack(recent).mean().item())
                               if recent else float("nan"))
                with open(hist_path, "a", encoding="utf-8") as fp:
                    fp.write("%d,%.4f,%.4f,%.6f,%.4f,%.1f\n"
                             % (step, recent_mean, v, lr, gnorm,
                                time.time() - t_start))

            # --- 试写 ---
            if do_sample and sample_interval > 0 and step % sample_interval == 0:
                txt = quick_sample(model, tok, scfg, device)
                log_line("  [试写] %s" % txt.replace("\n", " ")[:150])
                utils.emit_event("sample", task="train", step=int(step),
                                 text=txt.replace("\n", " ")[:400])

            # --- 存档 ---
            if ckpt_interval > 0 and step % ckpt_interval == 0:
                save_checkpoint(ckpt_path, model, opt, step, best_val, model_cfg,
                                extra=ckpt_extra(tok, tok_path))
                log_line("  [存档] 第 %d 步 → %s" % (step, ckpt_path))
                utils.emit_event("ckpt", task="train", step=int(step),
                                 path=os.path.abspath(ckpt_path), best=False)

    except KeyboardInterrupt:
        interrupted = True
        log_line("")
        log_line("  收到 Ctrl+C，正在安全收尾……")
    finally:
        if batch_stream is not None:
            batch_stream.close()

    # ---------------------------------------------------------------- 收尾
    save_checkpoint(ckpt_path, model, opt, step, best_val, model_cfg,
                    extra=ckpt_extra(tok, tok_path))
    if stop_seen:
        utils.ok("[停止] 收到停止请求，已保存断点 %s" % os.path.abspath(ckpt_path))
    total_time = time.time() - t_start
    utils.title("训练结束")
    utils.info("总步数        : %s / %s" % ("{:,}".format(step), "{:,}".format(max_steps)))
    utils.info("总耗时        : %s" % utils.fmt_duration(total_time))
    if step > start_step:
        utils.info("平均速度      : %s token/秒"
                   % "{:,}".format(int(tokens_per_step * (step - start_step)
                                       / max(1e-6, total_time))))
    if loss_hist:
        head = loss_hist[:min(20, len(loss_hist))]
        tail = loss_hist[-min(50, len(loss_hist)):]
        utils.info("loss          : %.4f → %.4f"
                   % (float(torch.stack(head).mean().item()),
                      float(torch.stack(tail).mean().item())))
    if best_val != float("inf"):
        utils.info("最好验证 loss : %.4f" % best_val)
    utils.info("断点          : %s" % ckpt_path)
    utils.info("日志          : %s" % log_path)
    utils.info("曲线数据      : %s" % hist_path)
    if use_cuda:
        utils.info("峰值显存      : %s" % utils.fmt_bytes(torch.cuda.max_memory_allocated(0)))
    if interrupted:
        utils.warn("训练被中断。加 --resume 就能接着跑。")
    utils.emit_event(
        "done", task="train", steps=int(step), max_steps=int(max_steps),
        best_val=None if best_val == float("inf") else float(best_val),
        out_dir=os.path.abspath(out_dir), stopped=bool(stop_seen),
        interrupted=bool(interrupted), elapsed=float(total_time),
        peak_vram=(int(torch.cuda.max_memory_allocated(0)) if use_cuda else None))
    print()
    utils.ok('看效果:  python -m lingyu sample --config <你的配置> --prompt "他抬起头"')
    return model, tok
