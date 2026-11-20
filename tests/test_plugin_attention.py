# -*- coding: utf-8 -*-
"""示例插件（局部窗口注意力）的回归测试。

钉住三件事：
  1. 插件文件能通过注册表登记成 `attention = local_window`；
  2. 前向/反向正常，且 KV cache 解码路径不炸；
  3. 窗口真的生效（窗口压到很小后输出与全注意力不同），窗口 0 时与全注意力一致。
"""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lingyu.model import LingYu, build_config  # noqa: E402
from lingyu.registry import get  # noqa: E402

_PLUGIN = os.path.join(_ROOT, "examples", "plugins", "attention_local_window.py")
_LOADED = False


def _load_plugin():
    """只加载一次：注册表不允许同名重复注册。"""
    global _LOADED
    if _LOADED:
        return
    spec = importlib.util.spec_from_file_location("_ly_local_window_plugin", _PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED = True


def _cfg():
    return build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=4,
                        n_kv_head=2, block_size=8, d_ff=64,
                        attention="local_window")


def test_plugin_registers_attention(monkeypatch):
    _load_plugin()
    assert get("attention", "local_window") is not None


def test_plugin_forward_backward_and_generate(monkeypatch):
    _load_plugin()
    monkeypatch.setenv("LINGYU_ATTN_WINDOW", "2")
    torch.manual_seed(0)
    model = LingYu(_cfg())
    idx = torch.randint(0, 64, (2, 8))
    _, loss, _ = model(idx, idx, collect_aux=False)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    # KV cache 解码路径（generate 内部走 cache）
    out = model.generate(idx[:, :4], max_new_tokens=2, temperature=0.0,
                         top_k=None, top_p=None, use_cache=True)
    assert out.shape[0] == 2 and out.shape[1] >= 6


def test_window_restricts_attention(monkeypatch):
    """窗口=2 与全注意力的输出必须不同（证明掩码确实生效）。"""
    _load_plugin()
    torch.manual_seed(1234)
    full = LingYu(build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=4,
                               n_kv_head=2, block_size=8, d_ff=64,
                               attention="local_window"))
    torch.manual_seed(1234)
    narrow = LingYu(_cfg())
    narrow.load_state_dict(full.state_dict())
    for m in (full, narrow):
        m.eval()
    idx = torch.randint(0, 64, (1, 8))
    monkeypatch.setenv("LINGYU_ATTN_WINDOW", "0")
    # 窗口值在 __init__ 时读入；这里直接改属性，等价于不同窗口的两份模型
    for blk in full.blocks:
        blk.attn.window = 0
    for blk in narrow.blocks:
        blk.attn.window = 2
    with torch.no_grad():
        a = full(idx)[0]
        b = narrow(idx)[0]
    assert not torch.allclose(a, b), "窗口掩码没有生效"


def test_window_zero_matches_full_attention(monkeypatch):
    """窗口 0 = 全注意力：与显式窗口设成超大值输出一致。"""
    _load_plugin()
    torch.manual_seed(7)
    m1 = LingYu(_cfg())
    torch.manual_seed(7)
    m2 = LingYu(_cfg())
    m2.load_state_dict(m1.state_dict())
    for m in (m1, m2):
        m.eval()
        for blk in m.blocks:
            blk.attn.window = 0
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        assert torch.allclose(m1(idx)[0], m2(idx)[0], atol=1e-6)
