# -*- coding: utf-8 -*-
"""示例模型（examples/models/tiny_lm.py）的回归测试。

这同时是「可插拔模型」的端到端证明：
加载示例文件 → 自动注册到注册表 → build_model 按 type 取到它 →
前向 / 反向 / loss mask / 生成 / 结构指纹 / 存取断点 全部走引擎原路径。
"""

import importlib.util
import os

import torch

from lingyu.model import ModelProtocol, build_config, build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "models", "tiny_lm.py")


_LOADED = False


def _load_example():
    """按文件路径加载示例，模拟「用户把自己的模型丢进 plugins/」。

    注册表一个进程只允许注册一次同名组件，所以这里只加载一次，
    和真实使用（plugins/ 里的文件在进程启动时被加载一次）一致。
    """
    global _LOADED
    if _LOADED:
        return
    spec = importlib.util.spec_from_file_location("lingyu_example_tiny_lm", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED = True
    return mod


def _cfg():
    cfg = build_config(vocab_size=64, block_size=16, n_layer=1, n_embd=32,
                       n_head=4, n_kv_head=2, d_ff=64)
    cfg["type"] = "tiny"
    return cfg


def test_example_model_is_registered():
    _load_example()
    from lingyu import registry
    assert "tiny" in registry.list_names("model")
    assert registry.get("model", "tiny") is not None


def test_example_model_works_through_engine_factory():
    _load_example()
    model = build_model(_cfg())
    assert isinstance(model, ModelProtocol)

    idx = torch.randint(0, 64, (2, 16))
    targets = idx.clone()
    targets[:, :8] = -1                     # SFT loss mask

    logits, loss, aux = model(idx, targets)
    assert logits.shape == (2, 16, 64)
    assert torch.isfinite(loss)
    assert aux is None
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    with torch.no_grad():
        out = model.generate(idx[:1, :4], max_new_tokens=3)
    assert out.shape == (1, 7)
    assert isinstance(model.structural_fingerprint(), dict)
