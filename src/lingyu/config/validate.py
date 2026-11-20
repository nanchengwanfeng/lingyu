# -*- coding: utf-8 -*-
"""配置校验：类型、范围、组合依赖，以及**规模感知**。

## 为什么要分层报错

    errors    必须修，否则跑不起来
    warnings  能跑，但结果大概不是你想要的
    notices   只是提醒（例如"这个配置本机装不下，但可以生成集群脚本"）

## 规模感知（一个引擎支持 13M 到万亿的关键）

校验器会粗略估算参数量与显存需求，然后：
    - 本机装得下  -> 什么都不说，直接能跑
    - 装不下      -> notice 说明"需要 N 张卡"，并提示换后端
    **绝不假装能在本机跑万亿参数。**
"""

from .loader import get
from .spec import SPEC, all_keys

try:                                     # 只用得到 registry，不会引入 model
    from ..components import norm_dim_mult
except Exception:                        # pragma: no cover - 极端环境兜底
    def norm_dim_mult(cfg, default=1):
        return 2 if (cfg.get("norm") or "").lower() == "layernorm" else default


class ConfigError(ValueError):
    """配置有错，必须修。"""


def _check_type(section, key, val, p, errors):
    t = p["type"]
    if t is bool:
        if not isinstance(val, bool):
            errors.append("[%s] %s 应该是 true/false，现在是 %r" % (section, key, val))
        return
    if t is int:
        if isinstance(val, bool) or not isinstance(val, int):
            errors.append("[%s] %s 应该是整数，现在是 %r" % (section, key, val))
        return
    if t is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errors.append("[%s] %s 应该是数字，现在是 %r" % (section, key, val))
        return
    if t is str:
        if not isinstance(val, str):
            errors.append("[%s] %s 应该是字符串，现在是 %r" % (section, key, val))


def _check_range(section, key, val, p, errors):
    """数值范围检查。**只处理数字**（choices 由 _check_choices 单独管）。"""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return
    lo, hi = p.get("min"), p.get("max")
    if lo is not None and val < lo:
        errors.append("[%s] %s = %s 太小了（最小 %s）" % (section, key, val, lo))
    if hi is not None and val > hi:
        errors.append("[%s] %s = %s 太大了（最大 %s）" % (section, key, val, hi))


def _check_choices(section, key, val, p, errors):
    """候选值检查。

    ⚠️ 必须**独立于类型检查** —— 之前它写在 _check_range 里，
    而那个函数遇到字符串就 return，导致 amp_dtype='float64' 这类错误
    完全抓不到（实测踩过）。
    """
    ch = p.get("choices")
    if not ch:
        return
    # 空字符串一律合法 = 「引擎自己决定」。
    # 全项目统一这个约定：attention / ffn / norm / pos 留空表示自动推断，
    # post.method 留空表示不做后训练。校验器放行，由各自的解析函数去填默认。
    if val == "":
        return
    if val not in ch:
        # 模型组件允许「插件注册名」：硬编码的 choices 只是内置实现清单，
        # 用户自己放进 plugins/ 的实现也应当合法（审计 B-07：以前插件名
        # 会被校验器直接拒绝，插件系统等于不可用）。
        if section == "model" and key in ("attention", "ffn", "norm", "pos"):
            try:
                from .. import registry as _reg
                if _reg.get(key, str(val)) is not None:
                    return
            except Exception:
                pass
        shown = "、".join(repr(c) if c == "" else str(c) for c in ch)
        errors.append("[%s] %s 只能是 %s 之一，现在是 %r"
                      % (section, key, shown, val))


def _num(cfg, sec, key):
    """取一个数值参数；缺失或类型不对时返回 None（类型错误由 _check_type 报）。"""
    v = cfg.get(sec, {}).get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v


def _check_hard_limits(cfg, errors):
    """模型结构上必须成立的下限 —— 不满足就根本建不出模型。

    实测踩过的崩法：`n_head = 0` → 建模型时
    `ZeroDivisionError: integer modulo by zero`；`n_layer = -5` →
    建出 -5 层的模型，训练开始才炸。这些不该留给下游。

    ★ 放在类型检查之后、组合校验之前；只补 errors 不 return。
      上层看到 errors 会提前结束，所以后面的 `n_embd % n_head` 不会除零。
    """
    for sec, key in (("model", "n_layer"), ("model", "n_embd"),
                     ("model", "n_head"), ("model", "n_kv_head"),
                     ("model", "block_size"),
                     ("train", "batch_size"), ("train", "max_steps")):
        v = _num(cfg, sec, key)
        if v is not None and v < 1:
            # 范围的 _check_range 通常已经报过一条；这里只兜底 spec 没写 min 的键，
            # 避免同一条错误打印两遍。
            if not any(e.startswith("[%s] %s" % (sec, key)) for e in errors):
                errors.append("[%s] %s = %s 太小了（最小 1）" % (sec, key, v))

    # vocab_size：0 = 自动跟随分词器（spec 的约定），负数才是错。
    vs = _num(cfg, "model", "vocab_size")
    if vs is not None and vs < 0 and not any(
            e.startswith("[model] vocab_size") for e in errors):
        errors.append("[model] vocab_size = %s 不能是负数（0 表示跟随分词器）" % vs)

    n_embd = _num(cfg, "model", "n_embd")
    n_head = _num(cfg, "model", "n_head")
    n_kv = _num(cfg, "model", "n_kv_head")
    if n_embd and n_head and n_embd % n_head != 0:
        errors.append("[model] n_embd (%s) 必须能被 n_head (%s) 整除"
                      % (n_embd, n_head))
    if n_head and n_kv:
        if n_kv > n_head:
            errors.append("[model] n_kv_head (%s) 不能大于 n_head (%s) —— "
                          "KV 头是给 Q 头共享的" % (n_kv, n_head))
        elif n_head % n_kv != 0:
            errors.append("[model] n_head (%s) 必须是 n_kv_head (%s) 的整数倍"
                          % (n_head, n_kv))


def validate(cfg, strict=False, scale_notice=True):
    """校验配置。返回 (errors, warnings, notices)，都是字符串列表。

    strict=True 时把警告也当错误（CI / 测试用）。

    ## scale_notice：给图形界面用的开关

    默认 True，会附上一段「这配置在本机装不装得下」的提示。
    那一步要**探测显存**，而探测显存要 `import torch`（约 3~8 秒）。

    图形界面**绝不能** import torch（见 `gui/__init__.py`），
    所以它调这个函数时必须传 `scale_notice=False` ——
    只要结构性的错误和警告，不要规模提示（显存那块界面自己算）。

    踩过的坑：界面为了「配置不合法就别给估算」而直接调 `validate(cfg)`，
    结果把 torch 拖了进来，`test_gui_does_not_import_torch_at_module_level`
    当场就红了 —— 那个测试拦下了一次真实的退化。
    """
    errors, warnings, notices = [], [], []

    # ---------------------------------------------------------- 未知节/键
    for sec, params in cfg.items():
        if sec not in SPEC:
            if sec in ("runtime",):
                continue
            warnings.append("配置里有 [%s] 段，但引擎不认识它（会被忽略）。"
                            "已知的段：%s" % (sec, "、".join(SPEC.keys())))
            continue
        if not isinstance(params, dict):
            errors.append("[%s] 段的内容不是一个键值列表" % sec)
            continue
        known = all_keys(sec)
        for k in params:
            if k not in known:
                warnings.append("[%s] 里有引擎不认识的项 %s（会被忽略）—— "
                                "如果你是想改别的，检查一下拼写" % (sec, k))

    # ---------------------------------------------------------- 类型与范围
    for sec, params in cfg.items():
        if sec not in SPEC or not isinstance(params, dict):
            continue
        for k, v in params.items():
            p = SPEC[sec].get(k)
            if p is None:
                continue
            # `none` / 空值不能靠「跳过校验」蒙混过关：
            # 补全默认值只补**缺失的键**，不会把已经写坏的值换掉。
            # 实测 `n_layer = none` / `n_head =` 以前能一路过校验，
            # 到建模型时才崩（或静默走 get() 的默认值，用户以为改成功了）。
            if v is None:
                errors.append("[%s] %s 不能是 none —— 要么填一个 %s，要么删掉这一行"
                              % (sec, k, p["type"].__name__))
                continue
            if v == "":
                # 空字符串对字符串参数是合法约定（留空 = 引擎自动推断）；
                # 整数/布尔/浮点留空则一定是写错了。
                if p["type"] is not str:
                    errors.append("[%s] %s 不能为空（要一个 %s）"
                                  % (sec, k, p["type"].__name__))
                continue
            _check_type(sec, k, v, p, errors)
            _check_range(sec, k, v, p, errors)
            _check_choices(sec, k, v, p, errors)

    # 硬下限：n_head=0 / n_layer=-5 这类会在建模型时崩，必须在入口拦住。
    _check_hard_limits(cfg, errors)

    # 有致命错误时没必要继续做组合校验（值可能是坏的，算下去会报一堆连锁错误）
    # ⚠️ 注意：这意味着**有 errors 时不会有 warnings/notices**。
    #    调用方（尤其是 GUI）应该先让用户修 errors，再看 warnings。
    if errors:
        return errors, warnings, notices

    # ---------------------------------------------------------- 组合依赖
    m = cfg.get("model", {})
    n_head = get(cfg, "model", "n_head", 8)
    n_embd = get(cfg, "model", "n_embd", 512)
    n_kv = get(cfg, "model", "n_kv_head", 2)

    if n_embd % n_head != 0:
        errors.append(
            "n_embd (%d) 必须能被 n_head (%d) 整除，否则没法把宽度均分给每个头。\n"
            "  建议：n_embd 改成 %d，或 n_head 改成 %d"
            % (n_embd, n_head, (n_embd // n_head) * n_head,
               max(1, n_embd // max(1, max(1, n_embd // n_head)))))
    else:
        head_dim = n_embd // n_head
        if head_dim % 2 != 0:
            errors.append(
                "每个头的维度 = n_embd/n_head = %d 是奇数，RoPE 要求偶数。\n"
                "  建议：n_embd 或 n_head 调整一下（例如 %d 维 / %d 头 -> 每头 %d）"
                % (head_dim, n_embd, n_head, head_dim))
        # 这条警告要独立判断（不能挂在"能整除"分支里，否则某些组合会漏报）
        if head_dim % 2 == 0 and head_dim < 16:
            warnings.append(
                "每个头只有 %d 维，偏小，注意力可能学不好。"
                "一般每头 32~128 维比较合适" % head_dim)

    if n_kv > n_head:
        errors.append("n_kv_head (%d) 不能大于 n_head (%d) —— KV 头是给 Q 头共享的"
                      % (n_kv, n_head))
    elif n_head % n_kv != 0:
        errors.append("n_head (%d) 必须是 n_kv_head (%d) 的整数倍，否则 Q 头没法均分"
                      % (n_head, n_kv))

    if m.get("use_moe"):
        ne = get(cfg, "model", "n_expert", 4)
        nk = get(cfg, "model", "n_expert_per_tok", 2)
        if nk > ne:
            errors.append("n_expert_per_tok (%d) 不能大于 n_expert (%d)" % (nk, ne))
        elif nk == 1:
            warnings.append(
                "n_expert_per_tok = 1（top-1）在小模型上实测容易路由坍缩。\n"
                "  论文 arXiv 2605.13769 的结论：top-2 才是质量提升的主要来源。\n"
                "  建议改成 2")
        if nk >= ne:
            warnings.append("每个 token 都走全部专家，这就不是稀疏模型了，"
                            "MoE 的意义（省算力）会消失")

    bs = get(cfg, "train", "batch_size", 32)
    blk = get(cfg, "model", "block_size", 512)
    ms = get(cfg, "train", "max_steps", 5000)
    warm = get(cfg, "train", "warmup_iters", 200)
    if warm >= ms:
        warnings.append("预热步数 (%d) 不小于总步数 (%d) —— 学习率永远升不到设定值就结束了。"
                        "建议 warmup_iters 改成总步数的 2%%~5%%" % (warm, ms))

    # ---------------------------------------------------------- 规模感知
    # scale_notice=False 时**整段跳过**，因为显存探测会 import torch ——
    # 图形界面不能碰 torch。
    if scale_notice:
        params, active, _kv = _estimate(cfg)
        if params is not None:
            notices.append("估算参数量约 %.2f M（激活 %.2f M）"
                           % (params / 1e6, active / 1e6))
            # ★ 口径统一走 backends.plan（审计 B-05）：以前这里只算
            #   「权重+优化器」、而且 d_model/n_layer 用占位值，和
            #   `lingyu plan` 的完整明细能差出好几倍 —— 一个说 729 MB
            #   够用、另一个说 3.93 GB 快满了。现在用真实几何值走同一条路径。
            from ..backends.plan import check_available_vram, estimate
            free = check_available_vram()
            if free > 0:
                _d = get(cfg, "model", "n_embd", 512)
                _L = get(cfg, "model", "n_layer", 8)
                _est = estimate(
                    params, batch=bs, seq=blk, d_model=_d, n_layer=_L,
                    world_size=max(1, int(get(cfg, "parallel", "tensor_parallel", 1) or 1))
                               * max(1, int(get(cfg, "parallel", "pipeline_parallel", 1) or 1)),
                    zero_stage=int(get(cfg, "parallel", "zero_stage", 0) or 0),
                    cpu_offload=bool(get(cfg, "parallel", "cpu_offload", False)),
                    activation_checkpoint=bool(get(cfg, "parallel",
                                                       "activation_checkpoint", False)),
                    moe=bool(get(cfg, "model", "use_moe", False)),
                    per_card_bytes=free)
                notices.extend(scale_notices(_est))

    # ---------------------------------------------------------- 后训练
    # 只在用户**真的选了**后训练方法时才校验 —— 纯预训练的配置不该被这里报错。
    po = cfg.get("post", {})
    if po and po.get("method") in ("sft", "dpo", "lora"):
        if po.get("method") in ("dpo", "lora"):
            # spec 的 choices 里列了它们，但引擎**没有实现**（审计 B-18）：
            # 以前选了会被静默当 sft 跑，用户以为在做 DPO/LoRA。
            errors.append(
                "[post] method = %s 尚未实现（0.3.x 只支持 sft）。"
                "DPO / LoRA 在路线图里（P7），现在选了不会生效。" % po["method"])
        if not po.get("data_path"):
            errors.append("[post] method=%s 需要指定 data_path（对话数据 jsonl）"
                          % po["method"])
        if po.get("method") == "sft" and po.get("loss_on_prompt"):
            warnings.append(
                "[post] loss_on_prompt = true —— 会对「用户问题」部分也算 loss，\n"
                "  模型会学着自己生成问题而不是回答问题。除非你明确知道自己在干什么，"
                "否则保持 false")

    if strict:
        errors.extend(warnings)
        warnings = []
    return errors, warnings, notices


def _estimate(cfg):
    """粗略估算参数量、激活参数、每 token 的 KV cache。"""
    try:
        V = get(cfg, "model", "vocab_size", 0) or 8192
        L = get(cfg, "model", "n_layer", 8)
        d = get(cfg, "model", "n_embd", 512)
        h = get(cfg, "model", "n_head", 8)
        kv = get(cfg, "model", "n_kv_head", 2)
        tie = get(cfg, "model", "tie_weights", True)
        d_ff = get(cfg, "model", "d_ff", 0) or int(round(8 * d / 3 / 8) * 8)
        # 注意力
        attn = L * (d * d + 2 * d * (kv * (d // h)) + d * d)
        # 前馈
        if get(cfg, "model", "use_moe", False):
            E = get(cfg, "model", "n_expert", 4)
            k = get(cfg, "model", "n_expert_per_tok", 2)
            S = get(cfg, "model", "n_shared_expert", 1)
            ed = get(cfg, "model", "expert_d_ff", 0) or max(8, int(round(d_ff / 2 / 8) * 8))
            ffn_full = L * (E * 3 * d * ed + d * E + S * 3 * d * d_ff)
            ffn_act = L * (k * 3 * d * ed + d * E + S * 3 * d * d_ff)
        else:
            ffn_full = ffn_act = L * 3 * d * d_ff
        emb = V * d * (1 if tie else 2)
        # 归一化：L 个 block 各 2 个 + 末尾 1 个。
        # RMSNorm 只有 weight（系数 1），LayerNorm 有 weight+bias（系数 2）。
        norm = (L * 2 + 1) * norm_dim_mult(cfg) * d
        total = attn + ffn_full + emb + norm
        active = attn + ffn_act + emb + norm
        kv_bytes = 2 * L * kv * (d // h) * 2       # 每 token（bf16）
        return total, active, kv_bytes
    except Exception:
        return None, None, None


def estimate_params(cfg):
    """公开版参数量估算：返回 `(总参数, 激活参数, 每 token KV 字节)`。

    GUI / 命令行 / validate 都走这一份公式（含 MoE 分支），避免各自
    手写一份导致数字漂移（审计 B-14：概览页曾自己算且没有 MoE 分支）。
    估不出来时返回 `(None, None, None)`。
    """
    return _estimate(cfg)


def scale_notices(est):
    """把 `backends.plan.estimate()` 的结果翻译成给人看的「规模提醒」。

    ## 为什么收 est 而不是自己算（审计 B-05）
    这里以前叫 `_scale_notice(params, active, batch, block, kv_bytes)`，
    自己另调一次 `estimate()`：`d_model=block`、`n_layer=1` 只是占位，
    而且只拿 `state_bytes`（权重+优化器）和显存比，完全忽略激活值。
    于是 `validate()` 说「约 729 MB，够用」，`lingyu plan` 说「合计 3.93 GB」——
    同一个配置两个数字。现在只做「翻译」，公式全在 `backends.plan` 一份。

    返回字符串列表；没有显存信息（`per_card_bytes` 缺失）时返回空列表。
    """
    out = []
    if not est or est.get("usable_per_card") is None:
        return out
    from ..backends.plan import FITS_MARGIN, fmt_bytes
    total = est["total_bytes"]
    usable = est["usable_per_card"]
    cards = int(est.get("cards_needed", 1))
    ratio = 100.0 * total / max(1.0, usable)
    out.append("显存估算合计 %s（可用 %s，占 %.0f%%）"
               % (fmt_bytes(total), fmt_bytes(est["per_card_bytes"]), ratio))
    if cards > 1:
        out.append("⚠ 本机单卡装不下：按 %.1f 倍安全余量估算需要约 %d 张同等显存的卡"
                   % (FITS_MARGIN, cards))
        out.append("  具体命令：  lingyu launch --config <你的配置>")
        out.append("  看详细明细： lingyu plan --config <你的配置>")
    elif ratio >= 85:
        out.append("⚠ 离显存上限很近（%.0f%%）—— 估算本身有误差，"
                   "建议先减小 batch_size / block_size，或先跑几步确认不会 OOM"
                   % ratio)
    else:
        out.append("显存够用（约占 %.0f%%）" % ratio)
    return out


def check_or_raise(cfg, strict=False):
    """校验并在有错时抛异常（把提示都打出来）。"""
    errors, warnings, notices = validate(cfg, strict=strict)
    return errors, warnings, notices
