# -*- coding: utf-8 -*-
"""模型配置与参数量测试。

历史注记：这个文件早期是个独立脚本，用**假的 torch 桩**来在没有 torch 的
机器上验证配置逻辑。那个桩会写进 `sys.modules`，**污染同进程里后面的测试**
（实测踩过：组件测试拿到假 torch，报 `unsupported operand ... '_T'`）。

现在真 torch 已经装好，所以改成正经的 pytest 测试，直接用真 torch ——
桩没了，污染问题也就不存在了。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

torch = pytest.importorskip("torch")

from lingyu.model import (LingYu, build_config, config_from_section,
                          estimate_param_count, kv_cache_bytes,
                          kv_cache_per_token)


# ============================================================== 参数校验

def test_n_embd_must_divide_n_head():
    with pytest.raises(ValueError, match="整除"):
        build_config(vocab_size=1000, n_embd=384, n_head=7)


def test_head_dim_must_be_even():
    with pytest.raises(ValueError, match="偶数"):
        build_config(vocab_size=1000, n_embd=390, n_head=6)


def test_even_head_dim_is_accepted():
    """300/6 = 50，偶数，不该报错。"""
    cfg = build_config(vocab_size=1000, n_embd=300, n_head=6)
    assert cfg["head_dim"] == 50


def test_n_kv_head_cannot_exceed_n_head():
    with pytest.raises(ValueError, match="不能大于"):
        build_config(vocab_size=1000, n_embd=512, n_head=8, n_kv_head=16)


def test_n_head_must_be_multiple_of_n_kv_head():
    with pytest.raises(ValueError, match="整数倍"):
        build_config(vocab_size=1000, n_embd=512, n_head=8, n_kv_head=3)


def test_n_kv_head_at_least_one():
    with pytest.raises(ValueError, match="至少"):
        build_config(vocab_size=1000, n_embd=512, n_head=8, n_kv_head=0)


def test_d_ff_auto_aligns_to_8():
    cfg = build_config(vocab_size=1000, n_embd=512, n_head=8)
    assert cfg["d_ff"] % 8 == 0
    assert cfg["d_ff"] == int(round(8 * 512 / 3 / 8) * 8)


def test_expert_d_ff_auto_is_half_of_d_ff():
    cfg = build_config(vocab_size=1000, n_embd=512, n_head=8, use_moe=True)
    assert cfg["expert_d_ff"] == int(round(cfg["d_ff"] / 2 / 8) * 8)


# ============================================================== 从配置段构造

def test_config_from_section_basic():
    cfg = config_from_section({"n_layer": 6, "n_embd": 384, "n_head": 6,
                               "n_kv_head": 2, "block_size": 256, "d_ff": 0}, 7000)
    assert cfg["n_layer"] == 6
    assert cfg["d_model"] == 384
    assert cfg["head_dim"] == 64
    assert cfg["vocab_size"] == 7000


def test_config_from_section_ignores_unknown_keys():
    """配置里写注释性字段不该报错。"""
    cfg = config_from_section({"n_layer": 6, "n_embd": 384, "n_head": 6,
                               "这是个注释": "不该炸"}, 100)
    assert cfg["n_layer"] == 6


def test_config_from_section_requires_core_keys():
    with pytest.raises(KeyError, match="必填项"):
        config_from_section({"n_embd": 384, "n_head": 6}, 100)


def test_component_keys_are_honored():
    """★ 防「配置被静默忽略」的核心测试。

    实测踩过：组件键不在白名单里，配置写了 norm=layernorm 完全没效果，
    模型照样用 RMSNorm 且不报错。
    """
    cfg = config_from_section({"n_layer": 2, "n_embd": 64, "n_head": 4,
                               "norm": "layernorm", "attention": "mha",
                               "ffn": "swiglu"}, 100)
    assert cfg["norm"] == "layernorm"
    assert cfg["attention"] == "mha"
    assert cfg["ffn"] == "swiglu"


def test_component_name_is_lowercased():
    cfg = config_from_section({"n_layer": 2, "n_embd": 64, "n_head": 4,
                               "norm": "  LayerNorm  "}, 100)
    assert cfg["norm"] == "layernorm"


# ============================================================== 参数量实测

def _params(cfg):
    return sum(p.numel() for p in LingYu(cfg).parameters())


def test_estimate_param_count_matches_real_dense():
    """`lingyu plan` 用的解析式估算必须和真实模型一致（不能差一个 bias）。"""
    cfg = build_config(vocab_size=7000, n_layer=4, n_embd=128, n_head=4,
                       n_kv_head=2, block_size=128, bias=True,
                       norm="layernorm")
    assert estimate_param_count(cfg) == _params(cfg)

    cfg2 = build_config(vocab_size=512, n_layer=2, n_embd=64, n_head=4,
                        n_kv_head=2, block_size=32, use_learned_pos=True,
                        use_rope=False)
    assert estimate_param_count(cfg2) == _params(cfg2)


def test_estimate_param_count_matches_real_moe():
    cfg = build_config(vocab_size=1000, n_layer=3, n_embd=64, n_head=4,
                       n_kv_head=2, block_size=32, use_moe=True,
                       n_expert=4, moe_every=2, n_shared_expert=1)
    assert estimate_param_count(cfg) == _params(cfg)


def test_estimate_param_count_none_for_custom_type():
    cfg = build_config(vocab_size=1000, n_layer=1, n_embd=64, n_head=4,
                       n_kv_head=2, block_size=32)
    cfg = dict(cfg, type="my_own_model")
    assert estimate_param_count(cfg) is None


def test_param_count_matches_hand_calculation():
    """公式与实测必须一致（差一点就说明公式错了）。

    参数构成（逐项实测核对过）：
        tok_emb   V × d
        ln_f      d               ← **RMSNorm 只有一个 weight，是 d 不是 2d**
        每层      ln1 + ln2 (2×d) + attn + mlp
        attn      d·h·hd + 2·d·kvh·hd + d·h·hd
        mlp       3 · d · d_ff

    踩坑记录（教训很值）：
        我一开始把归一化写成 `2*d`（以为像 LayerNorm 那样 weight+bias）。
        RMSNorm 只有一个 weight，所以是 `d`。两处都多算了一倍，
        手算比实测多 384 = 2×192。
        更蠢的是我连续 6 次把断言里的 `expect - got` 看成 `got - expect`，
        方向搞反，白白多查了好几轮 —— **看数值前先确认谁减谁**。
    """
    V, L, d, h, kvh = 7000, 6, 384, 6, 2
    cfg = build_config(vocab_size=V, n_layer=L, n_embd=d, n_head=h,
                       n_kv_head=kvh, block_size=256)
    hd = cfg["head_dim"]
    d_ff = cfg["d_ff"]
    emb = V * d                                   # 词嵌入（lm_head 与它共享，不另算）
    ln_f = d                                      # 最终归一化：RMSNorm 只有 weight
    attn = L * (d * h * hd + 2 * d * kvh * hd + d * h * hd)
    ffn = L * 3 * d * d_ff
    norm = L * 2 * d                              # 每层两个归一化
    expect = emb + ln_f + attn + ffn + norm
    got = _params(cfg)
    assert got == expect, "手算 %s，实测 %s（手算 - 实测 = %s）" % (
        "{:,}".format(expect), "{:,}".format(got), "{:,}".format(expect - got))


def test_layernorm_costs_one_extra_dim_per_norm():
    """LayerNorm 比 RMSNorm 多一组 bias，每处多 d 个参数。

    这条测试把两个坑一起钉死：
      1. 归一化参数量是 `d` 不是 `2d`（RMSNorm 只有一个 weight）；
      2. 末尾那个 `ln_f` 必须跟着 `norm` 配置一起换实现 —— 曾经它写死
         成 RMSNorm，于是 model.norm=layernorm 只有一半生效。
    归一化总处数 = 每层 2 个 × L + 末尾 1 个。
    """
    L, d = 2, 64
    base = dict(vocab_size=1000, n_layer=L, n_embd=d, n_head=4, n_kv_head=2,
                block_size=32, d_ff=128)
    n_rms = _params(build_config(**base))
    n_ln = _params(build_config(**{**base, "norm": "layernorm"}))
    n_norm = 2 * L + 1                          # 5 处
    assert n_ln - n_rms == n_norm * d, (
        "LayerNorm 应该比 RMSNorm 多 %d×d = %d 个参数，实测多 %d"
        % (n_norm, n_norm * d, n_ln - n_rms))


def test_final_norm_follows_config():
    """末尾归一化层必须跟 block 里的实现一致（配置说什么就是什么）。"""
    from lingyu.model import LayerNorm, LingYu, RMSNorm
    for name, cls in (("rmsnorm", RMSNorm), ("layernorm", LayerNorm)):
        m = LingYu(build_config(vocab_size=256, n_layer=1, n_embd=32,
                                n_head=2, n_kv_head=1, block_size=16,
                                norm=name))
        assert isinstance(m.ln_f, cls), "norm=%s 时 ln_f 是 %s" % (
            name, type(m.ln_f).__name__)
        assert isinstance(m.blocks[0].ln1, cls)


def test_param_counts_agree_with_reality():
    """估算公式（param_groups / active_param_count）必须和真实参数量对上。

    这套数字被 `lingyu params`、`lingyu plan` 和校验器的显存估算使用，
    算错了就会给出错误的「你这卡装不装得下」结论。
    """
    from lingyu.model import LingYu
    for size in (dict(n_embd=64, n_head=4, n_kv_head=2, n_layer=2),
                 dict(n_embd=256, n_head=8, n_kv_head=2, n_layer=4)):
        for norm in ("rmsnorm", "layernorm"):
            cfg = build_config(vocab_size=1000, block_size=32, d_ff=128,
                               norm=norm, **size)
            m = LingYu(cfg)
            real = sum(p.numel() for p in m.parameters())
            # tie_weights 时 lm_head 与 tok_emb 共享，named_parameters 去重，
            # 所以真实值就是 sum；active_param_count 也按共享算。
            active = m.active_param_count()
            assert active == real, "norm=%s %s: 估算 %d，真实 %d（差 %d）" % (
                norm, size, active, real, active - real)


def test_moe_has_more_total_but_fewer_active_params():
    """MoE 的本质：总参数更大，但每 token 只算一部分。"""
    dense = build_config(vocab_size=4096, n_layer=4, n_embd=256, n_head=8,
                         n_kv_head=2, block_size=128)
    moe = build_config(vocab_size=4096, n_layer=4, n_embd=256, n_head=8,
                       n_kv_head=2, block_size=128, use_moe=True,
                       n_expert=8, n_expert_per_tok=2, expert_d_ff=128)
    m_dense, m_moe = LingYu(dense), LingYu(moe)
    total_dense = sum(p.numel() for p in m_dense.parameters())
    total_moe = sum(p.numel() for p in m_moe.parameters())
    assert total_moe > total_dense, "MoE 的总参数应该更多"
    assert m_moe.active_param_count() < total_moe, "MoE 的激活参数应该少于总参数"


def test_moe_sparsity_ratio_reasonable():
    cfg = build_config(vocab_size=4096, n_layer=4, n_embd=256, n_head=8,
                       n_kv_head=2, block_size=128, use_moe=True,
                       n_expert=8, n_expert_per_tok=2, expert_d_ff=128)
    m = LingYu(cfg)
    total = sum(p.numel() for p in m.parameters())
    active = m.active_param_count()
    ratio = total / active
    assert 1.2 < ratio < 4.0, "稀疏比 %.2fx 不太合理" % ratio


# ============================================================== KV cache

def test_kv_cache_formula():
    cfg = build_config(vocab_size=7000, n_layer=6, n_embd=384, n_head=6,
                       n_kv_head=2, block_size=512)
    per_token = kv_cache_per_token(cfg)
    assert per_token == 2 * 6 * 2 * 64 * 2       # 2(KV) × 层 × KV头 × 每头维 × 2字节
    assert kv_cache_bytes(cfg, 1024) == per_token * 1024


def test_gqa_reduces_kv_cache_proportionally():
    mha = build_config(vocab_size=1000, n_layer=6, n_embd=384, n_head=6,
                       n_kv_head=6, block_size=512)
    gqa = build_config(vocab_size=1000, n_layer=6, n_embd=384, n_head=6,
                       n_kv_head=2, block_size=512)
    assert kv_cache_per_token(mha) == kv_cache_per_token(gqa) * 3


def test_mqa_has_smallest_cache():
    mqa = build_config(vocab_size=1000, n_layer=6, n_embd=384, n_head=6,
                       n_kv_head=1, block_size=512)
    gqa = build_config(vocab_size=1000, n_layer=6, n_embd=384, n_head=6,
                       n_kv_head=2, block_size=512)
    assert kv_cache_per_token(mqa) * 2 == kv_cache_per_token(gqa)


# ============================================================== 前向传播

def test_forward_runs_and_shapes_are_right():
    cfg = build_config(vocab_size=100, n_layer=2, n_embd=64, n_head=4,
                       n_kv_head=2, block_size=16, d_ff=128)
    model = LingYu(cfg).eval()
    x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        logits, loss, aux = model(x, x)
    assert logits.shape == (2, 16, 100)
    assert loss is not None and loss.item() > 0


def test_generate_produces_tokens():
    cfg = build_config(vocab_size=100, n_layer=2, n_embd=64, n_head=4,
                       n_kv_head=2, block_size=32, d_ff=128)
    model = LingYu(cfg).eval()
    idx = torch.randint(0, 100, (1, 4))
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=8)
    assert out.shape[1] == 12, "应该生成 8 个新 token"


# ============================================================== 位置编码 / 词表（审计 B-08 / B-09）

def test_pos_learned_is_the_real_switch():
    """pos = learned 必须真的切到学习式位置嵌入（旧代码里 pos 是装饰品）。"""
    cfg = build_config(vocab_size=1000, n_layer=2, n_embd=64, n_head=4,
                       n_kv_head=2, block_size=32, d_ff=128, pos="learned")
    assert cfg["pos"] == "learned"
    assert cfg["use_learned_pos"] is True
    assert cfg["use_rope"] is False
    model = LingYu(cfg)
    assert model.pos_emb is not None
    assert model.rope is None


def test_pos_none_disables_both():
    cfg = build_config(vocab_size=1000, n_layer=2, n_embd=64, n_head=4,
                       n_kv_head=2, block_size=32, d_ff=128, pos="none")
    assert cfg["use_rope"] is False and cfg["use_learned_pos"] is False
    model = LingYu(cfg)
    assert model.pos_emb is None and model.rope is None


def test_pos_unknown_is_rejected():
    with pytest.raises(ValueError, match="pos"):
        build_config(vocab_size=1000, n_layer=2, n_embd=64, n_head=4,
                     n_kv_head=2, block_size=32, pos="alibi")


def test_rope_and_learned_pos_cannot_both_be_true():
    with pytest.raises(ValueError, match="不能同时"):
        build_config(vocab_size=1000, n_layer=2, n_embd=64, n_head=4,
                     n_kv_head=2, use_rope=True, use_learned_pos=True)


def test_vocab_size_mismatch_is_rejected():
    """配置里写死 vocab_size 且和分词器不一致 —— 必须报错，不能静默覆盖。"""
    sec = {"n_layer": 2, "n_embd": 64, "n_head": 4, "vocab_size": 32000}
    with pytest.raises(ValueError, match="vocab_size"):
        config_from_section(sec, vocab_size=8192)


def test_vocab_size_zero_follows_tokenizer():
    sec = {"n_layer": 2, "n_embd": 64, "n_head": 4, "vocab_size": 0}
    cfg = config_from_section(sec, vocab_size=8192)
    assert cfg["vocab_size"] == 8192

    # 不写 vocab_size 也一样（spec 默认 0 = 跟随分词器）
    sec2 = {"n_layer": 2, "n_embd": 64, "n_head": 4}
    cfg2 = config_from_section(sec2, vocab_size=7000)
    assert cfg2["vocab_size"] == 7000
