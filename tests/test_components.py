# -*- coding: utf-8 -*-
"""组件可插拔性测试：换实现真的生效了吗。

这组测试防的是最阴的一类 bug：**配置写了但被静默忽略**。
实测踩过一次 —— `config_from_section` 用白名单过滤，新加的组件键
不在白名单里，于是配置里写 `norm = layernorm` 完全没效果，
模型照样用 RMSNorm，而且不报任何错。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

torch = pytest.importorskip("torch")   # torch 没装时整组跳过，不算失败

from lingyu import registry as R
from lingyu.components import (BUILTIN_ATTENTION, BUILTIN_FFN, BUILTIN_NORM,
                               BUILTIN_POS, get_attention, get_ffn, get_norm, get_pos)
from lingyu.model import (LayerNorm, LingYu, MoE, RMSNorm, SwiGLU,
                          config_from_section, kv_cache_per_token)

BASE = {"n_layer": 2, "n_embd": 64, "n_head": 4, "n_kv_head": 2,
        "block_size": 32, "d_ff": 128}


def build(**over):
    sec = dict(BASE)
    sec.update(over)
    cfg = config_from_section(sec, 100)
    model = LingYu(cfg)
    model.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        logits, loss, aux = model(x, x)
    return cfg, model, logits


# ============================================================== 注册表内容

def test_all_builtins_registered():
    for name in BUILTIN_ATTENTION:
        assert R.get("attention", name) is not None, "attention.%s 没注册" % name
    for name in BUILTIN_FFN:
        assert R.get("ffn", name) is not None, "ffn.%s 没注册" % name
    for name in BUILTIN_NORM:
        assert R.get("norm", name) is not None, "norm.%s 没注册" % name
    for name in BUILTIN_POS:
        assert R.get("pos", name) is not None, "pos.%s 没注册" % name


def test_attention_aliases_share_one_class():
    """mha / gqa / mqa 必须是同一个类 —— 它们是同一份代码靠 n_kv_head 区分。

    如果哪天有人把它们拆成三个类，这条会挂，提醒他别这么干。
    """
    a = R.get("attention", "mha")
    b = R.get("attention", "gqa")
    c = R.get("attention", "mqa")
    assert a is b is c, "mha/gqa/mqa 应该是同一个类（靠 n_kv_head 区分）"


def test_norm_implementations_are_distinct():
    assert R.get("norm", "rmsnorm") is not R.get("norm", "layernorm")


# ============================================================== 配置真的生效

def test_default_uses_gqa_swiglu_rmsnorm():
    _, model, logits = build()
    b = model.blocks[0]
    assert b.norm_name == "rmsnorm"
    assert b.attn_name == "gqa"
    assert b.ffn_name == "swiglu"
    assert logits.shape == (2, 8, 100)


def test_norm_switch_takes_effect():
    """换 layernorm 必须真的换掉 —— 这是防"配置被静默忽略"的核心测试。"""
    _, model, _ = build(norm="layernorm")
    assert isinstance(model.blocks[0].ln1, LayerNorm)
    assert not isinstance(model.blocks[0].ln1, RMSNorm)


def test_layernorm_changes_param_count():
    """LayerNorm 多一组 bias，参数量必须变 —— 变了才说明真的换了。"""
    _, m1, _ = build()
    _, m2, _ = build(norm="layernorm")
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    assert n2 > n1, "换成 layernorm 后参数量应该变多（多了 bias）"
    # 归一化处数 = 每层 2 个 × L + 末尾 1 个；每处多 dim 个 bias。
    # （末尾那个 ln_f 也必须跟着换，否则这里会少 d 个 —— 曾经的 bug）
    assert n2 - n1 == (2 * BASE["n_layer"] + 1) * BASE["n_embd"], (
        "多了 %d 个参数，应该是 %d 个" % (n2 - n1, (2 * BASE["n_layer"] + 1) * BASE["n_embd"]))


def test_attention_name_reflected_in_block():
    _, model, _ = build(attention="mha", n_kv_head=4)
    assert model.blocks[0].attn_name == "mha"
    _, model, _ = build(attention="mqa", n_kv_head=1)
    assert model.blocks[0].attn_name == "mqa"


def test_kv_cache_scales_with_kv_heads():
    """MHA 的 KV cache 应该比 MQA 大 n_head 倍。"""
    cfg_mha, _, _ = build(attention="mha", n_kv_head=4)
    cfg_mqa, _, _ = build(attention="mqa", n_kv_head=1)
    assert kv_cache_per_token(cfg_mha) == kv_cache_per_token(cfg_mqa) * 4


def test_moe_switch_takes_effect():
    _, model, _ = build(ffn="moe", use_moe=True, n_expert=4,
                        n_expert_per_tok=2, expert_d_ff=64)
    assert isinstance(model.blocks[0].mlp, MoE)
    assert model.blocks[0].is_moe


def test_swiglu_is_default_ffn():
    _, model, _ = build()
    assert isinstance(model.blocks[0].mlp, SwiGLU)
    assert not isinstance(model.blocks[0].mlp, MoE)


def test_ffn_moe_implies_use_moe():
    """配置里点名 ffn=moe 但忘了开 use_moe —— 引擎要帮他开上。"""
    cfg, model, _ = build(ffn="moe", n_expert=4, n_expert_per_tok=2, expert_d_ff=64)
    assert cfg["use_moe"] is True, "点名了 moe 就应该自动打开 use_moe"
    assert isinstance(model.blocks[0].mlp, MoE)


def test_ffn_swiglu_turns_off_moe():
    """反过来：点名 swiglu 就该关掉 use_moe，否则配置自相矛盾。"""
    cfg, _, _ = build(ffn="swiglu", use_moe=True)
    assert cfg["use_moe"] is False


def test_moe_every_alternates_layers():
    """moe_every=2 时，只有偶数层是 MoE，奇数层退回全宽前馈。"""
    _, model, _ = build(ffn="moe", use_moe=True, n_expert=4, n_expert_per_tok=2,
                        expert_d_ff=64, moe_every=2, n_layer=4)
    flags = [model.blocks[i].is_moe for i in range(4)]
    assert flags == [True, False, True, False], flags


# ============================================================== 报错质量

def test_unknown_component_raises_with_list():
    with pytest.raises(KeyError) as ei:
        build(norm="batchnorm")
    msg = str(ei.value)
    assert "batchnorm" in msg
    assert "rmsnorm" in msg and "layernorm" in msg, "报错要列出可用项"


def test_get_helpers_resolve():
    assert get_attention({"attention": "gqa"}) is R.get("attention", "gqa")
    assert get_norm({"norm": "layernorm"}) is LayerNorm
    assert get_pos({"pos": "rope"}) is R.get("pos", "rope")
    assert get_ffn({}) is SwiGLU
    assert get_ffn({"use_moe": True}) is MoE


# ============================================================== 插件覆盖

def test_plugin_can_replace_builtin(tmp_path, monkeypatch):
    """插件覆盖内置实现 —— 这是「别人不改我的代码就能换架构」的兑现。"""
    plug = tmp_path / "plug.py"
    plug.write_text(
        "import torch.nn as nn\n"
        "from lingyu.registry import register\n"
        "\n"
        "@register('norm', 'my_norm', override=True)\n"
        "class MyNorm(nn.Module):\n"
        "    def __init__(self, dim, eps=1e-6):\n"
        "        super().__init__()\n"
        "        self.dim = dim\n"
        "    def forward(self, x):\n"
        "        return x\n",
        encoding="utf-8")
    monkeypatch.setenv("LINGYU_PLUGINS", str(tmp_path))
    errs = R.load_plugins(force=True)
    assert not errs, errs
    try:
        # 真的用配置点名它，并且模型能建起来、能前向
        cfg, model, logits = build(norm="my_norm")
        assert model.blocks[0].ln1.__class__.__name__ == "MyNorm"
        assert logits.shape == (2, 8, 100)
        # 参数量应该少掉 norm 的 weight/bias
        _, m_base, _ = build()
        assert sum(p.numel() for p in model.parameters()) < \
               sum(p.numel() for p in m_base.parameters())
    finally:
        R.clear_plugins()
        R.load_plugins(force=True)   # 把内置的恢复回来


def test_repeated_registration_is_safe():
    """重复注册必须安全 —— 否则 reload 模型模块就会崩。"""
    import importlib

    import lingyu.model as M
    importlib.reload(M)
    importlib.reload(M)
    assert R.get("norm", "rmsnorm") is not None
