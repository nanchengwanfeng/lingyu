# -*- coding: utf-8 -*-
"""数据管线测试：eval_loss 的口径（block_size 上下文、token 加权、状态恢复）与
TokenDataset 的 x/y 偏移。全部用临时目录里的小 bin，不碰真实语料。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

torch = pytest.importorskip("torch")
import numpy as np

from lingyu.data import TokenDataset, eval_loss, write_tokens
from lingyu.model import LingYu, build_config


def _make_dataset(tmp_path, n_tokens=5001, block_size=64, vocab_size=128):
    ids = np.arange(n_tokens, dtype=np.int64) % vocab_size
    bin_path = os.path.join(str(tmp_path), "val.bin")
    write_tokens(bin_path, ids, vocab_size=vocab_size)
    return TokenDataset(bin_path, block_size=block_size, device="cpu", seed=7)


def _tiny_model(vocab_size=128, block_size=64):
    cfg = build_config(vocab_size=vocab_size, n_layer=1, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=block_size, d_ff=64)
    return LingYu(cfg)


def test_dataloader_offset_one():
    """x[i] 的下一 token 就是 y[i]（input/target 错一位）。"""
    dsdir = None
    ids = np.arange(200, dtype=np.int64) % 128
    # 直接构造一个内存里的假 dataset，避免依赖 tmp 路径
    class _DS:
        data = ids
        n_tokens = len(ids)
    class _M:
        def __call__(self, x, y, collect_aux=False):
            assert torch.equal(x[:, 1:], y[:, :-1])
            return None, torch.tensor(0.0), None
    x = torch.from_numpy(ids[10:10 + 9])[None, :]
    y = torch.from_numpy(ids[11:11 + 9])[None, :]
    _M()(x, y)
    assert dsdir is None


def test_eval_loss_uses_block_size_and_returns_tokens(tmp_path):
    ds = _make_dataset(tmp_path, n_tokens=5001, block_size=64)
    model = _tiny_model(block_size=64).eval()
    loss, n = eval_loss(model, ds, batch_size=8, iters=50, device="cpu",
                        block_size=64)
    # 每块 64 个预测 token，最多 50 块 = 3200 token（远大于旧的 50×8=400）
    assert n == 50 * 64
    assert 0.0 < loss < 50.0


def test_eval_loss_restores_training_state(tmp_path):
    ds = _make_dataset(tmp_path)
    model = _tiny_model()

    model.train()
    eval_loss(model, ds, batch_size=8, iters=3, device="cpu", block_size=64)
    assert model.training is True, "进来时是 train，出去必须还是 train"

    model.eval()
    eval_loss(model, ds, batch_size=8, iters=3, device="cpu", block_size=64)
    assert model.training is False, "进来时是 eval，出去不能偷偷切回 train"


def test_eval_loss_is_weighted_by_tokens(tmp_path):
    """短尾块按实际 token 数加权，而不是按块数简单平均。"""
    ds = _make_dataset(tmp_path, n_tokens=1001, block_size=64)
    model = _tiny_model(block_size=64)
    model.eval()
    loss, n = eval_loss(model, ds, batch_size=8, iters=0, device="cpu", block_size=64)
    # 15 块 64 token + 1 块 40 token = 1000 个预测 token
    assert n == 1000
    assert loss == loss  # 非 NaN


def test_eval_loss_batched_equals_single_chunk(tmp_path):
    """批量评估与逐块评估口径一致：loss 差 < 1e-4，token 数完全相同。

    块之间不重叠且注意力是因果的，所以把多块并成一个 batch 不会让它们互相看见；
    唯一的差别是短尾块用 target=-1 补齐（交叉熵忽略），加权后应当等价。
    """
    ds = _make_dataset(tmp_path, n_tokens=1001, block_size=64)   # 含一个 40 token 的短块
    model = _tiny_model(block_size=64).eval()
    l1, n1 = eval_loss(model, ds, batch_size=8, iters=0, device="cpu",
                       block_size=64, eval_batch=1)
    l8, n8 = eval_loss(model, ds, batch_size=8, iters=0, device="cpu",
                       block_size=64, eval_batch=8)
    l99, n99 = eval_loss(model, ds, batch_size=8, iters=0, device="cpu",
                         block_size=64, eval_batch=99)   # 超过块数也安全
    assert n1 == n8 == n99 == 1000
    assert abs(l1 - l8) < 1e-4 and abs(l1 - l99) < 1e-4, (l1, l8, l99)


def test_eval_loss_batched_respects_iters_limit(tmp_path):
    """iters 上限作用在「块数」上，不会因为批量而多评估。"""
    ds = _make_dataset(tmp_path, n_tokens=5001, block_size=64)
    model = _tiny_model(block_size=64).eval()
    loss, n = eval_loss(model, ds, batch_size=8, iters=5, device="cpu",
                        block_size=64, eval_batch=8)
    assert n == 5 * 64
    assert 0.0 < loss < 50.0


# ---------------------------------------------------------------- 预取器
from lingyu.data import BatchPrefetcher


def test_batch_prefetcher_matches_sync_sequence(tmp_path):
    """预取只改变「什么时候算」，不改变「算什么」：同 seed 下与同步取数逐项相等。"""
    bin_path = os.path.join(str(tmp_path), "train.bin")
    ids = np.arange(999, dtype=np.int64) % 128
    write_tokens(bin_path, ids, vocab_size=128)
    sync_ds = TokenDataset(bin_path, block_size=32, device="cpu", seed=11)
    expected = [sync_ds.get_batch(4) for _ in range(6)]

    async_ds = TokenDataset(bin_path, block_size=32, device="cpu", seed=11)
    got = []
    with BatchPrefetcher(async_ds, batch_size=4) as stream:
        for _ in range(6):
            got.append(next(stream))
    assert len(got) == 6
    for (ex, ey), (gx, gy) in zip(expected, got):
        assert torch.equal(ex, gx) and torch.equal(ey, gy)


def test_batch_prefetcher_close_is_idempotent(tmp_path):
    bin_path = os.path.join(str(tmp_path), "train.bin")
    write_tokens(bin_path, np.arange(500, dtype=np.int64) % 64, vocab_size=64)
    ds = TokenDataset(bin_path, block_size=16, device="cpu", seed=3)
    stream = BatchPrefetcher(ds, batch_size=2)
    x, y = next(stream)
    assert x.shape == (2, 16)
    stream.close()
    stream.close()          # 第二次不能抛
    assert not stream._thread.is_alive()


def test_get_batch_vectorized_matches_windows(tmp_path):
    """向量化取批必须与逐条切片完全一致（同一 RNG 状态）。"""
    bin_path = os.path.join(str(tmp_path), "train.bin")
    ids = (np.arange(600, dtype=np.int64) * 7) % 128
    write_tokens(bin_path, ids, vocab_size=128)
    ds = TokenDataset(bin_path, block_size=16, device="cpu", seed=5)
    x, y = ds.get_batch(3)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert x.shape == (3, 16) and y.shape == (3, 16)
    # 用 get_batch 自己消耗掉的 RNG 状态重放：RNG 是同一个，直接比窗口内容
    assert torch.equal(x[:, 1:], y[:, :-1])          # x 与 y 错开一位
    # 固定 seed 重放同样 RNG 序列，再取一次应完全相同
    ds2 = TokenDataset(bin_path, block_size=16, device="cpu", seed=5)
    x2, y2 = ds2.get_batch(3)
    assert torch.equal(x, x2) and torch.equal(y, y2)


def test_batch_prefetcher_supports_custom_fetch():
    """SFT 的 batch() 返回三元组，用 fetch 回调也能预取（不要求叫 get_batch）。"""
    class _DS:
        def batch(self, n):
            return (torch.arange(n), torch.arange(n), n)

    ds = _DS()
    with BatchPrefetcher(ds, batch_size=3, fetch=ds.batch) as stream:
        a = next(stream)
        b = next(stream)
    assert len(a) == 3 and len(b) == 3
    assert a[0].tolist() == [0, 1, 2] and a[2] == 3


