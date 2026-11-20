# -*- coding: utf-8 -*-
"""模型行为回归测试：rope_base 生效、长 prompt 不再 0 token、use_cache=False 历史不丢。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

torch = pytest.importorskip("torch")

from lingyu.model import LingYu, build_config


def test_rope_base_actually_changes_rope_tables():
    def make(base):
        cfg = build_config(vocab_size=100, n_layer=1, n_embd=32, n_head=4,
                           n_kv_head=2, block_size=16, rope_base=base)
        return LingYu(cfg).eval()

    m1 = make(10000.0)
    m2 = make(500000.0)
    assert not torch.allclose(m1.rope.inv_freq, m2.rope.inv_freq)
    cos1, _ = m1.rope.forward(16, torch.device("cpu"), torch.float32)
    cos2, _ = m2.rope.forward(16, torch.device("cpu"), torch.float32)
    assert not torch.allclose(cos1, cos2), "rope_base 必须真正影响 cos 表"


def test_generate_with_prompt_equal_to_block_size_not_zero_tokens():
    cfg = build_config(vocab_size=100, n_layer=1, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=8, d_ff=64)
    model = LingYu(cfg).eval()
    idx = torch.randint(0, 100, (1, 8))          # prompt 长度 == block_size
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape[1] == 8 + 5, "prompt 打满上下文也必须照常生成"


def test_generate_with_prompt_longer_than_block_size():
    cfg = build_config(vocab_size=100, n_layer=1, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=8, d_ff=64)
    model = LingYu(cfg).eval()
    idx = torch.randint(0, 100, (1, 15))         # 超过上下文，滚动窗口
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape[1] == 15 + 3


def test_generate_without_cache_keeps_history():
    """use_cache=False 时每步喂完整窗口，不能只喂最后 1 个 token。"""
    cfg = build_config(vocab_size=100, n_layer=1, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=8, d_ff=64)
    model = LingYu(cfg).eval()
    idx = torch.randint(0, 100, (1, 6))
    out = model.generate(idx, max_new_tokens=4, use_cache=False)
    assert out.shape[1] == 6 + 4


def test_chunked_prefill_logits_match_full_forward():
    """cache 已有 5 个 token 再喂 3 个：logits 必须等于 8 个 token 一次性前向。"""
    cfg = build_config(vocab_size=100, n_layer=2, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=16, d_ff=64)
    model = LingYu(cfg).eval()
    torch.manual_seed(0)
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        full = model(x, targets=torch.zeros_like(x), collect_aux=False)[0]

        cache = [{"k": torch.empty(1, 2, 0, 8), "v": torch.empty(1, 2, 0, 8)}
                 for _ in range(2)]
        model(x[:, :5], cache=cache, cache_pos=0, collect_aux=False)
        part = model(x[:, 5:], targets=torch.zeros(1, 3, dtype=torch.long),
                     cache=cache, cache_pos=5, collect_aux=False)[0]

        # 预分配 buffer 版 cache（generate 用的新接口）也必须一致
        cache2 = [{"k": torch.empty(1, 2, 16, 8),
                   "v": torch.empty(1, 2, 16, 8), "pos": 0}
                  for _ in range(2)]
        model(x[:, :5], cache=cache2, cache_pos=0, collect_aux=False)
        part2 = model(x[:, 5:], targets=torch.zeros(1, 3, dtype=torch.long),
                      cache=cache2, cache_pos=5, collect_aux=False)[0]
    assert torch.allclose(part, full[:, 5:], atol=1e-5), (
        "cache_pos>0 且 T>1 时掩码必须按真实偏移构造，不能直接用 is_causal")
    assert torch.allclose(part2, full[:, 5:], atol=1e-5), (
        "预分配 cache 与一次性前向的 logits 必须一致")


def test_generate_preallocated_cache_matches_no_cache():
    """预分配 KV buffer 不改变生成结果：cache 版 == 每步全窗口版（不触发滚动）。"""
    cfg = build_config(vocab_size=100, n_layer=2, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=16, d_ff=64, dropout=0.0)
    model = LingYu(cfg).eval()
    torch.manual_seed(0)
    idx = torch.randint(0, 100, (1, 6))
    # 6 + 5 = 11 < block_size，不会触发滚动窗口，两条路径必须逐 token 相同
    out_cache = model.generate(idx, max_new_tokens=5, temperature=1e-5, use_cache=True)
    out_full = model.generate(idx, max_new_tokens=5, temperature=1e-5, use_cache=False)
    assert torch.equal(out_cache, out_full), "预分配 buffer 改变了生成结果"


def test_repetition_penalty_applies_per_row():
    """batch>1 时每一行各看自己的历史（旧实现只惩罚第 0 行）。"""
    from lingyu.model import apply_repetition_penalty
    logits = torch.tensor([[1.0, -1.0, 2.0, 0.5],
                           [1.0, -1.0, 2.0, 0.5]])
    idx = torch.tensor([[0], [2]])          # 第 0 行见过 token0；第 1 行见过 token2
    out = apply_repetition_penalty(logits, idx, 2.0)
    assert out[0, 0].item() == pytest.approx(0.5)      # 第 0 行 token0 被罚
    assert out[1, 0].item() == pytest.approx(1.0)      # 第 1 行 token0 不动
    assert out[1, 2].item() == pytest.approx(1.0)      # 第 1 行 token2 被罚
    assert out[0, 2].item() == pytest.approx(2.0)      # 第 0 行 token2 不动
    # 负 logits：除以 penalty 是错的，必须乘（更负）
    out2 = apply_repetition_penalty(torch.tensor([[-1.0, 1.0]]),
                                    torch.tensor([[0]]), 2.0)
    assert out2[0, 0].item() == pytest.approx(-2.0)


def test_generate_eos_is_per_row_and_stops_output_growth():
    """2 行 batch：第 0 行立刻 EOS，第 1 行两次后 EOS；EOS 之后不再出现内容 token。"""
    cfg = build_config(vocab_size=20, n_layer=1, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=16, d_ff=64)
    model = LingYu(cfg).eval()
    calls = []

    def fake_forward(idx, targets=None, cache=None, cache_pos=0, collect_aux=True):
        B, T = idx.shape
        logits = torch.full((B, T, 20), -10.0, device=idx.device)
        n = len(calls)
        calls.append(1)
        logits[0, -1, 2] = 10.0                     # 第 0 行：马上 EOS
        logits[1, -1, 2 if n >= 2 else 7] = 10.0    # 第 1 行：7,7,然后 EOS
        return logits, None, None

    model.forward = fake_forward
    prompt = torch.tensor([[1], [1]])
    out = model.generate(prompt, max_new_tokens=8, eos_id=2)
    gen = out[:, 1:].tolist()
    assert len(calls) == 3, "两行都 EOS 后应立刻停，不该空转到 max_new_tokens"
    assert gen[0] == [2, 2, 2], "第 0 行 EOS 之后只能是 EOS 填充"
    assert gen[1] == [7, 7, 2]


def test_preallocated_cache_matches_legacy_cat_cache():
    """A8 对拍：预分配 buffer 与旧「每步 torch.cat」逐 token 前向，logits 必须一致。"""
    cfg = build_config(vocab_size=100, n_layer=2, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=16, d_ff=64, dropout=0.0)
    model = LingYu(cfg).eval()
    torch.manual_seed(0)
    idx = torch.randint(0, 100, (1, 6))
    n_kv, hd = cfg["n_kv_head"], cfg["head_dim"]

    # 旧接口：0 长度历史 + 每层每步 torch.cat
    legacy = [{"k": torch.zeros(1, n_kv, 0, hd),
               "v": torch.zeros(1, n_kv, 0, hd)} for _ in range(cfg["n_layer"])]
    # 新接口：预分配 cap 的 buffer + 写入位置 pos
    prealloc = [{"k": torch.zeros(1, n_kv, 16, hd),
                 "v": torch.zeros(1, n_kv, 16, hd), "pos": 0}
                for _ in range(cfg["n_layer"])]

    with torch.no_grad():
        for t in range(idx.shape[1]):
            l_old, _, _ = model(idx[:, t:t+1], cache=legacy, cache_pos=t,
                                collect_aux=False)
            l_new, _, _ = model(idx[:, t:t+1], cache=prealloc, cache_pos=t,
                                collect_aux=False)
            assert torch.allclose(l_old, l_new, atol=1e-5), (
                "预分配 cache 与旧 cat cache 数值不一致（第 %d 个 token）" % t)


def test_grad_checkpoint_forward_matches_plain():
    torch.manual_seed(0)
    cfg = build_config(vocab_size=100, n_layer=2, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=16, dropout=0.1)
    m = LingYu(cfg)
    m.train()
    idx = torch.randint(0, 100, (2, 16))
    tgt = torch.randint(0, 100, (2, 16))
    torch.manual_seed(7)
    _, loss_plain, _ = m(idx, tgt)
    m.grad_checkpoint = True
    torch.manual_seed(7)
    _, loss_gc, _ = m(idx, tgt)
    # 前向数值必须一致；反向能穿过 checkpoint 正常回传梯度
    assert torch.allclose(loss_plain, loss_gc, atol=1e-6)
    loss_gc.backward()
    assert m.tok_emb.weight.grad is not None
