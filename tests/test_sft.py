# -*- coding: utf-8 -*-
"""SFT 相关测试。

SFT 的 bug 特别难发现，因为它们**不会让 loss 变成 NaN**，只会让模型
学错东西：mask 划错 -> 模型学会复述问题；padding 参与了 loss ->
模型学会输出一堆 PAD；扩词表没有重新绑定 tie -> 参数量悄悄翻倍。

所以这里逐条钉住这些不变式。
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

torch = pytest.importorskip("torch")

from lingyu.model import LingYu, build_config
from lingyu.sft import (SFTDataset, _clean_reply, _load_base_weights,
                        find_base_checkpoint, sft)
from lingyu.tokenizer import PAD, train_tokenizer


@pytest.fixture(scope="module")
def tok():
    text = ("用户：助手：系统：你好光合作用是植物把光能变成化学能的过程"
            "李白唐代诗人一加一等于二再见谢谢") * 30
    return train_tokenizer(text, 200, kind="char", verbose=False)


def _sample(n_ids, n_sup, fill=8):
    """造一条样本：前 (n_ids-n_sup) 个不学，后 n_sup 个学。"""
    ids = [fill] * n_ids
    mask = [0] * (n_ids - n_sup) + [1] * n_sup
    return {"ids": ids, "mask": mask, "n_supervised": n_sup}


# ==================================================== 批次构造

def test_batch_pads_and_marks_targets_negative():
    """★ 不该学的目标位置必须是 -1，padding 位置也必须是 -1。

    这是 loss mask 落地的地方。如果这里错了：
      - padding 参与 loss -> 模型学会吐一串 PAD
      - 问题参与 loss     -> 模型学会复述问题
    两种都不会报错，只会让模型变傻。
    """
    samples = [_sample(10, 4), _sample(6, 2)]
    ds = SFTDataset(samples, block_size=32, device="cpu", seed=1)
    # 显式指定下标，避免随机抽样让这条测试时灵时不灵
    idx, tgt, n_sup = ds.batch(2, indices=[0, 1])

    assert idx.shape == tgt.shape == (2, 10), "应该 padding 到本批最长"
    # padding 用 PAD 填
    assert int(idx[1, 6]) == PAD and int(idx[1, 9]) == PAD
    # padding 位置的目标是 -1（不学）
    assert int(tgt[1, 6]) == -1 and int(tgt[1, 9]) == -1
    # 第 0 条：mask 标记回答是 ids[6..9]，要学的是「下一个 token」，
    # 所以监督落在位置 5..8（target=ids[6..9]），位置 9 没有下一个 token。
    assert list(tgt[0, :5].tolist()) == [-1] * 5
    assert list(tgt[0, 5:9].tolist()) == [8, 8, 8, 8]
    assert int(tgt[0, 9]) == -1
    # 第 1 条：mask 标记回答是 ids[4..5]，监督落在位置 3..4
    assert list(tgt[1, :3].tolist()) == [-1] * 3
    assert int(tgt[1, 3]) == 8 and int(tgt[1, 4]) == 8
    assert int(tgt[1, 5]) == -1
    assert n_sup == 6, "监督 token 总数应该是 4+2"


def test_batch_targets_are_shifted_by_one():
    """★ 回归测试：target 必须是「下一个 token」，不能是当前 token。

    曾经这里写成 tgt[i]=ids[i]，模型学的是「复制当前 token」，
    loss 看起来在降，但 SFT 出来的模型不会对话（实测同位置口径 loss
    7.75 vs next-token 口径 5.37）。用各不相同的 id 才抓得住这个 bug。
    """
    ids = [11, 12, 13, 14, 15, 16]
    mask = [0, 0, 0, 1, 1, 1]          # 后 3 个是回答
    sample = {"ids": ids, "mask": mask, "n_supervised": 3}
    ds = SFTDataset([sample], block_size=16, device="cpu", seed=0)
    _, tgt, n_sup = ds.batch(0, indices=[0])

    # 位置 2 预测 token 13？—— 不对：回答从 ids[3]=14 开始，
    # 所以位置 2 的 target 是 14，位置 3 是 15，位置 4 是 16；
    # 位置 5 没有下一个 token，必须是 -1。
    assert list(tgt[0, :6].tolist()) == [-1, -1, 14, 15, 16, -1]
    assert n_sup == 3


def test_batch_supervised_count_matches_samples():
    samples = [_sample(20, 7) for _ in range(20)]
    ds = SFTDataset(samples, block_size=64, device="cpu", seed=3)
    idx, tgt, n_sup = ds.batch(4)
    assert n_sup == 4 * 7
    # 直接数一遍 -1 之外的位置，交叉验证
    assert int((tgt != -1).sum()) == n_sup


def test_batch_indices_are_deterministic():
    """给了 indices 就必须完全确定 —— 评估要能复现。"""
    samples = [_sample(10, 3, fill=i + 4) for i in range(10)]
    ds = SFTDataset(samples, block_size=32, device="cpu", seed=42)
    a = ds.batch(0, indices=[3, 1, 7])[0]
    b = ds.batch(0, indices=[3, 1, 7])[0]
    assert torch.equal(a, b)
    assert a.shape[0] == 3


def test_batch_is_deterministic_with_same_seed():
    """同一个 seed 必须抽出同样的批次 —— 否则没法复现一次训练。"""
    samples = [_sample(10, 3, fill=i) for i in range(30)]
    a = SFTDataset(samples, block_size=32, device="cpu", seed=42).batch(4)[0]
    b = SFTDataset(samples, block_size=32, device="cpu", seed=42).batch(4)[0]
    assert torch.equal(a, b)


def test_too_long_sample_raises_with_actionable_message():
    """超长样本必须**报错**，不能默默截断。

    截断会砍掉回答的后半截，等于教模型说话说一半 —— 比报错糟得多。
    """
    samples = [_sample(10, 3), _sample(100, 50)]
    with pytest.raises(ValueError) as ei:
        SFTDataset(samples, block_size=32, device="cpu", seed=1)
    msg = str(ei.value)
    assert "block_size" in msg and "max_seq_len" in msg
    assert "100" in msg and "32" in msg


def test_empty_dataset_is_refused():
    with pytest.raises(ValueError):
        SFTDataset([], block_size=32, device="cpu")


def test_supervised_token_count_recorded():
    ds = SFTDataset([_sample(10, 4), _sample(10, 6)], block_size=32,
                    device="cpu", seed=1)
    assert ds.n_supervised == 10
    assert len(ds) == 2


# ==================================================== 数值上真的生效

def test_masked_loss_equals_manual_masked_loss():
    """★ 端到端验证：模型算出来的 loss 必须等于「只对 mask 位置算」的 loss。

    这条测试是整条链路的关键 —— 它证明「把不该学的位置设成 -1」
    这个做法真的达到了「只在回答上算 loss」的效果，而不是我们一厢情愿。
    """
    cfg = build_config(vocab_size=64, n_layer=2, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=16, tie_weights=False)
    model = LingYu(cfg)
    model.eval()

    ids = torch.randint(4, 64, (2, 8))
    tgt = ids.clone()
    tgt[:, :5] = -1                       # 前 5 个位置不学

    with torch.no_grad():
        logits, loss, _ = model(ids, targets=tgt)

    # 手算：把 logits 和 tgt 摊平，跳过 -1 的位置
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_tgt = tgt.reshape(-1)
    keep = flat_tgt != -1
    manual = torch.nn.functional.cross_entropy(flat_logits[keep], flat_tgt[keep])
    assert torch.allclose(loss, manual, atol=1e-6), (
        "模型 loss %.6f 与手算的 masked loss %.6f 不一致"
        % (loss.item(), manual.item()))
    assert keep.sum() == 2 * 3, "应该只有 6 个位置参与"


def test_ignoring_more_positions_changes_loss():
    """对照组：把 mask 改小，loss 必须跟着变（证明 mask 真的起作用）。

    如果 mask 被忽略，两次 loss 会完全一样 —— 这条断言就会失败。
    """
    cfg = build_config(vocab_size=64, n_layer=2, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=16, tie_weights=False)
    torch.manual_seed(0)
    model = LingYu(cfg)
    model.eval()
    ids = torch.randint(4, 64, (1, 10))

    out = []
    for cut in (3, 7):
        tgt = ids.clone()
        tgt[:, :cut] = -1
        with torch.no_grad():
            _, loss, _ = model(ids, targets=tgt)
        out.append(float(loss))
    assert out[0] != out[1], "mask 范围变了但 loss 没变 —— mask 没生效"


# ==================================================== 扩词表

def test_resize_vocab_keeps_tie_weights():
    """★ 扩词表后 tok_emb 和 lm_head 必须**仍然是同一个张量**。

    踩过的坑：只换了 tok_emb，忘了重新绑定 lm_head，于是模型突然多出
    一整个独立的输出矩阵 —— 参数量悄悄翻倍，加载断点形状对不上，
    而且报错信息完全指不到这里。
    """
    cfg = build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=8, tie_weights=True)
    m = LingYu(cfg)
    n_before = sum(p.numel() for p in m.parameters())
    m.resize_vocab(80)
    assert m.cfg["vocab_size"] == 80
    assert m.tok_emb.weight.shape[0] == 80
    assert m.lm_head.weight.shape[0] == 80
    assert m.tok_emb.weight is m.lm_head.weight, "tie 被破坏了"
    # 新增 16 行 × 32 维 = 512 个参数（只算一次，因为共享）
    assert sum(p.numel() for p in m.parameters()) - n_before == 16 * 32


def test_resize_vocab_preserves_existing_rows():
    """老 token 的 embedding 必须原样保留，只把新行随机初始化。"""
    cfg = build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=8)
    m = LingYu(cfg)
    old = m.tok_emb.weight.detach().clone()
    m.resize_vocab(80)
    assert torch.equal(m.tok_emb.weight[:64], old), "老 token 的向量被改动了"


def test_resize_vocab_noop_when_same_size():
    cfg = build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=8)
    m = LingYu(cfg)
    w = m.tok_emb.weight
    m.resize_vocab(64)
    assert m.tok_emb.weight is w, "大小没变就不该重建张量"


def test_resize_vocab_untied_keeps_lm_head_independent():
    cfg = build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=8, tie_weights=False)
    m = LingYu(cfg)
    m.resize_vocab(80)
    assert m.tok_emb.weight is not m.lm_head.weight
    assert m.tok_emb.weight.shape[0] == m.lm_head.weight.shape[0] == 80


def test_model_still_runs_after_resize():
    cfg = build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=8)
    m = LingYu(cfg)
    m.resize_vocab(80)
    idx = torch.randint(0, 80, (1, 6))
    logits, loss, _ = m(idx, targets=idx)
    assert logits.shape[-1] == 80, "输出维度应该跟着词表走"


# ==================================================== 回答清理

def test_clean_reply_stops_at_next_role():
    """小模型爱自问自答，回答里出现下一个角色标记就该截断。"""
    assert _clean_reply("唐代。\n用户：那杜甫呢？") == "唐代。"
    assert _clean_reply("你好！助手：你也好") == "你好！"


def test_clean_reply_strips_chatml_markers():
    assert _clean_reply("唐代。<|im_end|>") == "唐代。"
    assert _clean_reply("唐代。<|im_start|>user") == "唐代。"


def test_clean_reply_keeps_normal_text():
    s = "光合作用是植物把光能变成化学能的过程。"
    assert _clean_reply(s) == s
    assert _clean_reply("  有多余空白  ") == "有多余空白"


# ==================================================== 找基础断点

def test_find_base_checkpoint_prefers_best_pt(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "ckpt.pt").write_bytes(b"x")
    (out / "best.pt").write_bytes(b"x")
    cfg = {"train": {"out_dir": str(out)}}
    path, tried = find_base_checkpoint(cfg)
    assert path.endswith("best.pt"), "应该优先用 best.pt"


def test_find_base_checkpoint_honours_init_from(tmp_path):
    p = tmp_path / "my.pt"
    p.write_bytes(b"x")
    cfg = {"post": {"init_from": str(p)}, "train": {"out_dir": "nope"}}
    path, _ = find_base_checkpoint(cfg)
    assert path == str(p)


def test_find_base_checkpoint_reports_where_it_looked(tmp_path):
    """找不到时必须列出**找过哪些地方** —— 只说「找不到」是最气人的报错。"""
    cfg = {"train": {"out_dir": str(tmp_path / "nothing")}}
    path, tried = find_base_checkpoint(cfg)
    assert path is None
    assert len(tried) >= 2, "应该记录找过的位置"
    assert all(len(t) == 2 for t in tried)


# ==================================================== 断点/分词器指纹（C02）

class _FakeTok:
    def __init__(self, fp):
        self._fp = fp

    def fingerprint(self):
        return self._fp


def _save_tiny_base(tmp_path, fingerprint="fp"):
    """存一个最小预训练断点，结构要和 build_config 一致才能 strict 加载。"""
    cfg = build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=16, tie_weights=False)
    ck = {"config": cfg, "model": LingYu(cfg).state_dict(), "step": 7}
    if fingerprint is not None:
        ck["tokenizer_fingerprint"] = fingerprint
    p = tmp_path / "base.pt"
    torch.save(ck, str(p))
    return str(p)


def _tiny_model():
    cfg = build_config(vocab_size=64, n_layer=1, n_embd=32, n_head=2,
                       n_kv_head=1, block_size=16, tie_weights=False)
    return LingYu(cfg)


def test_load_base_weights_rejects_tokenizer_mismatch(tmp_path):
    """★ 指纹不符必须拦下：同词表大小不同 id 是最危险的错配。"""
    p = _save_tiny_base(tmp_path, fingerprint="deadbeefdeadbeef")
    with pytest.raises(ValueError) as ei:
        _load_base_weights(_tiny_model(), p, "cpu", _FakeTok("cafebabecafebabe"))
    msg = str(ei.value)
    assert "不一致" in msg
    assert "data.tokenizer_path" in msg, "报错要给出换分词器的出路"


def test_load_base_weights_warns_when_fingerprint_missing(tmp_path, capsys):
    """老断点没指纹时不能静默通过，要明确警告「无法核对」。"""
    p = _save_tiny_base(tmp_path, fingerprint=None)
    ck = _load_base_weights(_tiny_model(), p, "cpu", _FakeTok("whatever"))
    assert ck is not None
    assert "无法核对" in capsys.readouterr().out


def test_load_base_weights_accepts_matching_fingerprint(tmp_path):
    p = _save_tiny_base(tmp_path, fingerprint="cafebabecafebabe")
    ck = _load_base_weights(_tiny_model(), p, "cpu",
                            _FakeTok("cafebabecafebabe"))
    assert ck is not None, "指纹一致时应该正常返回断点内容"


# ==================================================== post.* 传参（C06/C07）

def test_sft_passes_system_prompt_and_loss_on_prompt_to_renderer(
        monkeypatch, tmp_path, tok):
    """sft() 必须把 [post] 的 system_prompt / loss_on_prompt 传给渲染器。

    这里只验证「参数有没有传对」：在渲染处截住，不真的训练。
    """
    import lingyu.sft as sft_mod

    tok_path = tmp_path / "tokenizer.txt"
    tok.save(str(tok_path))
    data_path = tmp_path / "chat.jsonl"
    data_path.write_text(
        '{"messages": [{"role": "user", "content": "你好"},'
        ' {"role": "assistant", "content": "你好呀"}]}\n',
        encoding="utf-8")

    captured = {}

    def fake_render(rows, tok_, **kw):
        captured.update(kw)
        raise RuntimeError("_stop_here")

    monkeypatch.setattr(sft_mod, "render_dataset", fake_render)
    cfg = {"model": {"n_layer": 1, "n_embd": 16, "n_head": 2,
                     "n_kv_head": 1, "block_size": 256},
           "data": {"tokenizer_path": str(tok_path)},
           "post": {"data_path": str(data_path), "template": "plain",
                    "system_prompt": "你是助手", "loss_on_prompt": True,
                    "max_seq_len": 128},
           "train": {"out_dir": str(tmp_path / "out")}}
    with pytest.raises(RuntimeError, match="_stop_here"):
        sft(cfg, device="cpu")
    assert captured["system"] == "你是助手"
    assert captured["loss_on_prompt"] is True
    assert captured["max_seq_len"] == 128


def test_sft_cli_accepts_resume_flag():
    """`lingyu sft --resume` 必须存在（审计 C05：曾让用户加 --resume 但 CLI 没这个参数）。"""
    from lingyu.cli import build_parser
    p = build_parser()

    a = p.parse_args(["sft", "--config", "x.ini"])
    assert a.resume is None

    b = p.parse_args(["sft", "--config", "x.ini", "--resume"])
    assert b.resume == "auto", "只写 --resume 时应续 post.out_dir/ckpt_sft.pt"

    c = p.parse_args(["sft", "--config", "x.ini", "--resume", "out/ckpt_sft.pt"])
    assert c.resume == "out/ckpt_sft.pt"


# ==================================================== 监控数据契约（审计 1.2）

def test_sft_writes_history_csv_and_run_json(tmp_path, tok):
    """★ SFT 必须自己写 history.csv / run.json。

    以前 SFT 什么都不写，GUI 监控页只能拿预训练的口径套：token/步按
    `[train] batch × grad_accum × block` 算、总步数按预训练配置算，
    于是微调页显示出 9 万 tok/s 这种数 —— 用户实测撞到过。

    这里跑一次真实的最小 SFT（1 层 32 维、2 步），钉住三件事：
    1. run.json 里 `kind=sft`，token/步 = batch_size × max_seq_len；
    2. history.csv 有表头和数据行，列数固定 6（GUI 按列解析）；
    3. 最后一行是 max_steps 那一步（监控页读到的步数要对得上）。
    """
    import json

    tok_path = tmp_path / "tokenizer.txt"
    tok.save(str(tok_path))

    # 基础断点必须和 SFT 用的模型结构一致，且指纹对得上才会被接受
    base_cfg = build_config(vocab_size=tok.vocab_size, n_layer=1, n_embd=32,
                            n_head=2, n_kv_head=1, block_size=32,
                            tie_weights=False)
    base_path = tmp_path / "base.pt"
    torch.save({"config": base_cfg, "model": LingYu(base_cfg).state_dict(),
                "step": 3, "tokenizer_fingerprint": tok.fingerprint()},
               str(base_path))

    ds = tmp_path / "chat.jsonl"
    row = {"messages": [{"role": "user", "content": "你好"},
                        {"role": "assistant", "content": "你好呀"}]}
    ds.write_text("\n".join(json.dumps(row, ensure_ascii=False) for _ in range(4)),
                  encoding="utf-8")

    out_dir = tmp_path / "out"
    cfg = {
        "model": {"n_layer": 1, "n_embd": 32, "n_head": 2,
                  "n_kv_head": 1, "block_size": 32},
        "data": {"tokenizer_path": str(tok_path)},
        "post": {"data_path": str(ds), "template": "plain", "max_seq_len": 32,
                 "batch_size": 2, "max_steps": 2, "eval_interval": 1,
                 "log_interval": 1, "ask_interval": 999, "ckpt_interval": 999,
                 "learning_rate": 1e-4, "init_from": str(base_path),
                 "out_dir": str(out_dir)},
        "train": {"out_dir": str(tmp_path / "pretrain_out")},
    }
    sft(cfg, device="cpu")

    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "sft"
    assert meta["tokens_per_step"] == 2 * 32, "SFT 的 token/步 = batch × max_seq_len"
    assert meta["max_steps"] == 2
    assert meta["resume"] is False
    assert meta["template"] == "plain"

    lines = (out_dir / "history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "step,train_loss,val_loss,lr,grad_norm,elapsed_sec"
    assert len(lines) >= 2, "至少要有一步的数据"
    last = lines[-1].split(",")
    assert last[0] == "2", "最后一行应是 max_steps 那一步"
    assert len(last) == 6, "列数固定 6，GUI 按列解析"


def test_eval_sft_loss_runs_without_grad():
    """验证 loss 必须关梯度：以前忘了 no_grad，每次验证都白建一整张计算图。"""
    import torch
    from lingyu.sft import eval_sft_loss

    class _Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.grad_flags = []

        def forward(self, idx, targets=None, collect_aux=False):
            self.grad_flags.append(torch.is_grad_enabled())
            loss = self.p * 0 + 1.25          # grad 开启时 requires_grad=True
            return None, loss, {}

    class _DS:
        def batch(self, n):
            return (torch.zeros((n, 4), dtype=torch.long),
                    torch.zeros((n, 4), dtype=torch.long), [])

    m = _Spy()
    m.train()
    v = eval_sft_loss(m, _DS(), batch_size=2, iters=3, device="cpu", amp_dtype=None)
    assert abs(v - 1.25) < 1e-6
    assert m.grad_flags and not any(m.grad_flags), "验证时不能有梯度追踪"
    assert m.training, "验证完要把模型恢复回训练模式"
