# -*- coding: utf-8 -*-
"""KV cache int8 量化的回归测试（KIVI 思路的简化实现）。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

torch = pytest.importorskip("torch")

from lingyu.model import LingYu, build_config, kv_cache_bytes, kv_cache_per_token


def _model(**kw):
    base = dict(vocab_size=100, n_layer=2, n_embd=32, n_head=4,
                n_kv_head=2, block_size=32, d_ff=64)
    base.update(kw)
    return LingYu(build_config(**base)).eval()


def _make_cache(model, batch, cap, int8):
    cfg = model.cfg
    out = []
    for _ in range(cfg["n_layer"]):
        item = {
            "k": torch.empty(batch, cfg["n_kv_head"], cap, cfg["head_dim"],
                             dtype=torch.int8 if int8 else torch.float32),
            "v": torch.empty(batch, cfg["n_kv_head"], cap, cfg["head_dim"],
                             dtype=torch.int8 if int8 else torch.float32),
            "pos": 0,
            "int8": int8,
        }
        if int8:
            item["k_scale"] = torch.empty(batch, cfg["n_kv_head"], cap, dtype=torch.float32)
            item["v_scale"] = torch.empty(batch, cfg["n_kv_head"], cap, dtype=torch.float32)
        out.append(item)
    return out


def test_int8_cache_dtypes_and_scales():
    model = _model()
    idx = torch.randint(0, 100, (2, 6))
    cache = _make_cache(model, 2, 32, int8=True)
    model(idx, cache=cache, cache_pos=0)
    for item in cache:
        assert item["k"].dtype == torch.int8
        assert item["v"].dtype == torch.int8
        assert item["k_scale"].dtype == torch.float32
        assert item["pos"] == 6
        assert torch.isfinite(item["k_scale"]).all()
        assert (item["k_scale"][:, :, :6].abs() > 0).all()


def test_int8_cache_logits_close_to_fp32_cache():
    model = _model()
    torch.manual_seed(7)
    idx = torch.randint(0, 100, (1, 7))
    with torch.no_grad():
        _, _, _ = model(idx, cache=_make_cache(model, 1, 32, False), cache_pos=0)
        cache_f = _make_cache(model, 1, 32, False)
        logits_f, _, _ = model(idx, cache=cache_f, cache_pos=0)
        cache_q = _make_cache(model, 1, 32, True)
        logits_q, _, _ = model(idx, cache=cache_q, cache_pos=0)
    ce = torch.nn.functional.cross_entropy
    assert torch.isfinite(logits_q).all()
    # 量化误差应很小：余弦相似度 > 0.999
    cos = torch.nn.functional.cosine_similarity(
        logits_f.reshape(-1), logits_q.reshape(-1), dim=0)
    assert float(cos) > 0.999
    assert float((logits_f - logits_q).abs().max()) < 0.2


def test_int8_chunked_prefill_matches_single_shot():
    model = _model()
    torch.manual_seed(11)
    idx = torch.randint(0, 100, (1, 9))
    with torch.no_grad():
        one = _make_cache(model, 1, 32, True)
        logits_all, _, _ = model(idx, cache=one, cache_pos=0)
        two = _make_cache(model, 1, 32, True)
        _, _, _ = model(idx[:, :4], cache=two, cache_pos=0)
        logits_tail, _, _ = model(idx[:, 4:], cache=two, cache_pos=4)
    cos = torch.nn.functional.cosine_similarity(
        logits_all[:, -1, :].reshape(-1), logits_tail[:, -1, :].reshape(-1), dim=0)
    assert float(cos) > 0.999


def test_int8_generate_runs_and_returns_expected_length():
    model = _model(block_size=16)
    model.cfg["kv_cache_dtype"] = "int8"
    idx = torch.randint(0, 100, (1, 5))
    out = model.generate(idx, max_new_tokens=4, temperature=0.8)
    assert out.shape == (1, 9)


def test_kv_cache_bytes_accounts_for_int8():
    cfg = build_config(vocab_size=100, n_layer=2, n_embd=256, n_head=4,
                       n_kv_head=2, block_size=32, d_ff=64,
                       kv_cache_dtype="int8")
    q8 = kv_cache_bytes(cfg, 1024)
    cfg2 = dict(cfg)
    cfg2["kv_cache_dtype"] = "bf16"
    bf16 = kv_cache_bytes(cfg2, 1024)
    assert q8 < bf16 * 0.6, (q8, bf16)
    assert q8 == kv_cache_per_token(cfg) * 1024
