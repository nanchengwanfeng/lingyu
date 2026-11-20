# -*- coding: utf-8 -*-
"""分块交叉熵（长上下文训练的显存优化）的回归测试。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

torch = pytest.importorskip("torch")

from lingyu.model import LingYu, build_config


def _model():
    return LingYu(build_config(vocab_size=120, n_layer=2, n_embd=32, n_head=4,
                               n_kv_head=2, block_size=16, d_ff=64)).train()


def test_chunked_loss_equals_plain_loss():
    model = _model()
    torch.manual_seed(3)
    idx = torch.randint(0, 120, (2, 9))
    tgt = torch.randint(0, 120, (2, 9))
    tgt[0, 0] = -1
    tgt[1, 3] = -1
    model.loss_chunk = 0
    _, loss_plain, _ = model(idx, targets=tgt)
    model.loss_chunk = 4                     # 强制切成 3 块
    _, loss_chunk, _ = model(idx, targets=tgt)
    assert torch.allclose(loss_plain, loss_chunk, atol=1e-5), (loss_plain, loss_chunk)


def test_chunked_loss_ignores_masked_positions():
    model = _model()
    model.loss_chunk = 5
    torch.manual_seed(5)
    idx = torch.randint(0, 120, (1, 8))
    tgt = torch.randint(0, 120, (1, 8))
    tgt[:, :6] = -1                          # 只有 2 个位置算 loss
    _, loss, _ = model(idx, targets=tgt)
    assert torch.isfinite(loss)
    # 手工算这 2 个位置的交叉熵，应与分块结果一致
    model.loss_chunk = 0
    with torch.no_grad():
        logits, _, _ = model(idx, targets=tgt)
        manual = torch.nn.functional.cross_entropy(
            logits[0, 6:], tgt[0, 6:], ignore_index=-1)
    assert torch.allclose(loss, manual, atol=1e-5)


def test_chunked_loss_backward_gives_grads():
    model = _model()
    model.loss_chunk = 4
    torch.manual_seed(9)
    idx = torch.randint(0, 120, (2, 8))
    tgt = torch.randint(0, 120, (2, 8))
    _, loss, _ = model(idx, targets=tgt)
    loss.backward()
    grad = model.tok_emb.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_loss_chunk_zero_means_off():
    model = _model()
    assert model.loss_chunk == 0
    idx = torch.randint(0, 120, (1, 5))
    tgt = torch.randint(0, 120, (1, 5))
    _, loss, _ = model(idx, targets=tgt)
    assert torch.isfinite(loss)
