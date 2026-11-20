# -*- coding: utf-8 -*-
"""模型接口契约测试（可插拔模型 / P0）。

任何注册进注册表 ``model`` 种类的实现，都必须满足引擎的最小要求：

  - 能被 ``build_model()`` 按 ``[model] type`` 实例化
  - ``forward(idx, targets=...)`` 返回 ``(logits, loss, aux)``
  - ``targets`` 里 -1 表示「这个位置不算损失」（SFT loss mask）
  - ``generate()`` 能自回归续写
  - ``structural_fingerprint()`` 返回可比较的结构字段（续训一致性检查）
  - ``state_dict`` 能存能读

参数化跑注册表里的所有模型，所以以后新增一个模型实现，
只要在 ``lingyu/model.py``（或 plugins/、examples/）里注册，
就会自动被这批测试覆盖 —— 不用再改测试。
"""

import ast
import os

import pytest
import torch

from lingyu import registry
from lingyu.model import (
    ModelProtocol,
    build_config,
    build_model,
    config_from_section,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = registry.list_names("model")


def _tiny_cfg(**over):
    """小到能在 CPU 上秒跑的配置：2 层 × 32 维，词表 64。"""
    base = dict(
        vocab_size=64,
        block_size=32,
        n_layer=2,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        d_ff=64,
        dropout=0.0,
    )
    base.update(over)
    return build_config(**base)


# ---------------------------------------------------------------- 注册表本身
def test_builtin_model_is_registered():
    assert "lingyu" in registry.list_names("model")


def test_unknown_model_type_lists_available_ones():
    with pytest.raises(KeyError) as ei:
        build_model({"type": "no_such_model", **_tiny_cfg()})
    msg = str(ei.value)
    assert "no_such_model" in msg
    # 报错必须告诉用户「有哪些是可用的」
    assert "lingyu" in msg


# ---------------------------------------------------------------- 配置层
def test_config_from_section_passes_type_through():
    sec = {
        "type": "LingYu",          # 大小写不敏感
        "n_layer": 1,
        "n_embd": 32,
        "n_head": 4,
        "n_kv_head": 2,
        "block_size": 32,
    }
    cfg = config_from_section(sec, 64)
    assert cfg["type"] == "lingyu"
    assert build_model(cfg).cfg["type"] == "lingyu"


def test_config_from_section_defaults_to_lingyu():
    cfg = config_from_section(
        {"n_layer": 1, "n_embd": 32, "n_head": 4, "n_kv_head": 2,
         "block_size": 32}, 64)
    assert cfg["type"] == "lingyu"


# ---------------------------------------------------------------- 协议覆盖
@pytest.mark.parametrize("name", MODELS)
def test_model_satisfies_protocol(name):
    model = build_model(dict(_tiny_cfg(), type=name))
    assert isinstance(model, ModelProtocol)


@pytest.mark.parametrize("name", MODELS)
def test_forward_returns_logits_loss_aux(name):
    torch.manual_seed(0)
    cfg = dict(_tiny_cfg(), type=name)
    model = build_model(cfg)
    model.train()
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    logits, loss, aux = model(idx, idx)
    assert logits.shape == (2, cfg["block_size"], cfg["vocab_size"])
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", MODELS)
def test_forward_without_targets_returns_logits_only(name):
    cfg = dict(_tiny_cfg(), type=name)
    model = build_model(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, 8))
    with torch.no_grad():
        logits, loss, _aux = model(idx)
    assert logits.dim() == 3
    assert logits.shape[0] == 1
    assert logits.shape[-1] == cfg["vocab_size"]
    assert loss is None


@pytest.mark.parametrize("name", MODELS)
def test_forward_honours_loss_mask(name):
    """SFT 只学「回答」位置：targets 里被设成 -1 的位置不能贡献 loss。"""
    cfg = dict(_tiny_cfg(), type=name)
    model = build_model(cfg)
    model.train()
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    tgt = idx.clone()
    tgt[:, : cfg["block_size"] // 2] = -1
    _, loss, _ = model(idx, tgt, collect_aux=False)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("name", MODELS)
def test_generate_extends_sequence(name):
    cfg = dict(_tiny_cfg(), type=name)
    model = build_model(cfg)
    model.eval()
    prompt = torch.zeros((1, 4), dtype=torch.long)
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=5)
    assert out.shape[0] == 1
    assert out.shape[1] >= prompt.shape[1] + 1


@pytest.mark.parametrize("name", MODELS)
def test_structural_fingerprint_and_state_dict_roundtrip(name, tmp_path):
    cfg = dict(_tiny_cfg(), type=name)
    model = build_model(cfg)
    fp = model.structural_fingerprint()
    assert isinstance(fp, dict)
    assert fp.get("vocab_size") == cfg["vocab_size"]
    assert fp.get("n_layer") == cfg["n_layer"]

    path = os.path.join(str(tmp_path), "tiny.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model2 = build_model(dict(ckpt["config"]))
    model2.load_state_dict(ckpt["model"])

    idx = torch.randint(0, cfg["vocab_size"], (1, 8))
    model.eval()
    model2.eval()
    with torch.no_grad():
        a = model(idx)[0]
        b = model2(idx)[0]
    assert torch.allclose(a, b)


# ---------------------------------------------------------------- 去耦守卫
def _calls_named(path, func_name):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == func_name:
                hits.append(getattr(node, "lineno", 0))
            elif isinstance(fn, ast.Attribute) and fn.attr == func_name:
                hits.append(getattr(node, "lineno", 0))
    return hits


def test_no_direct_model_constructor_outside_model_module():
    """训练/微调/采样/CLI 必须走 build_model，不能直接 new LingYu。

    否则「换模型实现」只对一部分代码生效，其它入口仍然绑死内置模型。
    """
    targets = ["train.py", "sft.py", "sample.py", "cli.py", "chat.py"]
    offenders = {}
    for name in targets:
        path = os.path.join(ROOT, "src", "lingyu", name)
        hits = _calls_named(path, "LingYu")
        if hits:
            offenders[name] = hits
    assert not offenders, "这些文件仍直接构造 LingYu：%r" % offenders
