# -*- coding: utf-8 -*-
"""训练循环的纯函数测试（不启动训练）。

重点钉住：lr_decay_iters=0 / 缺失 = 跟随 max_steps（spec.py 的文档语义）。
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from lingyu.train import (lr_at, save_checkpoint, load_checkpoint,
                          structure_diffs, _rng_state, _restore_rng)
from lingyu.model import LingYu, build_config
from lingyu.version import CKPT_FORMAT, __version__ as LINGYU_VERSION
from lingyu import utils


def test_stop_requested_helper(tmp_path, monkeypatch):
    """LINGYU_STOP_FILE 协议：未设置 / 文件不存在 → False；文件出现 → True。"""
    monkeypatch.delenv("LINGYU_STOP_FILE", raising=False)
    assert utils.stop_file() == ""
    assert utils.stop_requested() is False

    stop = tmp_path / "STOP"
    monkeypatch.setenv("LINGYU_STOP_FILE", str(stop))
    assert utils.stop_file() == str(stop)
    assert utils.stop_requested() is False          # 文件还没出现

    stop.write_text("stop", encoding="utf-8")
    assert utils.stop_requested() is True           # 文件出现 = 请求停止

    stop.unlink()
    assert utils.stop_requested() is False

    monkeypatch.setenv("LINGYU_STOP_FILE", "   ")
    assert utils.stop_requested() is False          # 空/空白路径不算


BASE = {"learning_rate": 1e-3, "min_lr": 1e-4,
        "warmup_iters": 10, "lr_decay_iters": 0, "max_steps": 100}


def test_lr_decay_zero_follows_max_steps():
    # step == warmup：刚预热完，应仍是峰值学习率
    assert lr_at(10, BASE) == pytest.approx(1e-3)
    # 若 0 被当成 decay=warmup+1（旧 bug），这里会直接掉到 min_lr
    assert lr_at(11, BASE) > 9e-4
    # 中途在 (min_lr, lr) 之间且在下降
    mid = lr_at(50, BASE)
    assert 1e-4 < mid < 1e-3
    # 到 max_steps 正好衰减到 min_lr，之后保持
    assert lr_at(100, BASE) == pytest.approx(1e-4, rel=1e-9)
    assert lr_at(150, BASE) == pytest.approx(1e-4, rel=1e-9)


def test_lr_decay_missing_key_falls_back_to_max_steps():
    cfg = {k: v for k, v in BASE.items() if k != "lr_decay_iters"}
    assert lr_at(100, cfg) == pytest.approx(1e-4, rel=1e-9)
    assert lr_at(12, cfg) > 9e-4


def test_lr_explicit_decay_still_works():
    cfg = dict(BASE, lr_decay_iters=1000)
    assert lr_at(1000, cfg) == pytest.approx(1e-4, rel=1e-9)
    assert lr_at(500, cfg) > 1e-4


def test_lr_warmup_linear():
    cfg = dict(BASE)
    assert lr_at(0, cfg) == pytest.approx(1e-3 / 10)
    assert lr_at(4, cfg) == pytest.approx(1e-3 * 5 / 10)


# ---------------------------------------------------------------------------
# A10：断点契约（格式版本 / 结构键 / RNG 状态）
# ---------------------------------------------------------------------------

def _tiny_model(vocab=100):
    cfg = build_config(vocab_size=vocab, n_layer=1, n_embd=32, n_head=4,
                       n_kv_head=2, block_size=16, d_ff=64)
    return LingYu(cfg), cfg


def test_checkpoint_contract_roundtrip(tmp_path):
    torch.manual_seed(0)
    model, mcfg = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, opt, 7, 1.25, mcfg,
                    extra={"tokenizer_fingerprint": "abc123"})

    ck = load_checkpoint(str(path))
    assert ck["ckpt_format"] == CKPT_FORMAT
    assert ck["lingyu_version"] == LINGYU_VERSION
    assert ck["step"] == 7
    assert ck["best_val"] == pytest.approx(1.25)
    assert ck["config"]["n_layer"] == 1
    assert ck["tokenizer_fingerprint"] == "abc123"
    for key in ("python", "torch", "numpy"):
        assert key in ck["rng"], "断点里缺 RNG 状态: " + key


def test_load_checkpoint_rejects_newer_format(tmp_path):
    path = tmp_path / "future.pt"
    torch.save({"model": {}, "ckpt_format": CKPT_FORMAT + 1}, str(path))
    with pytest.raises(ValueError, match="格式版本"):
        load_checkpoint(str(path))


def test_load_checkpoint_old_format_warns_but_loads(capsys, tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"model": {}, "step": 3}, str(path))     # 老格式：没有 ckpt_format
    ck = load_checkpoint(str(path))
    assert ck["step"] == 3
    out = capsys.readouterr().out
    assert "旧格式断点" in out
    assert "ckpt_format" in out


def test_load_checkpoint_rejects_non_lingyu_dict(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"state_dict": {}}, str(path))
    with pytest.raises(ValueError, match="不是灵语断点"):
        load_checkpoint(str(path))


def test_structure_diffs_detects_mismatch_and_old_fields():
    old = {"n_layer": 2, "d_model": 32, "n_head": 4, "d_ff": 64,
           "vocab_size": 100, "use_moe": False}    # 老断点没有 n_kv_head/block_size/tie/bias
    new = {"n_layer": 2, "d_model": 32, "n_head": 4, "d_ff": 64,
           "vocab_size": 200, "use_moe": False, "n_kv_head": 2,
           "block_size": 16, "tie_weights": True, "bias": False}
    diffs, unchecked = structure_diffs(old, new)
    assert [(d[0], d[1], d[2]) for d in diffs] == [("词表大小", 100, 200)]
    for label in ("KV 头数 n_kv_head", "上下文 block_size",
                  "权重共享 tie_weights", "线性层 bias"):
        assert label in unchecked, "旧断点无法核对的项必须报出来: " + label


def test_structure_diffs_all_match():
    cfg = {"n_layer": 2, "d_model": 32, "n_head": 4, "n_kv_head": 2, "d_ff": 64,
           "block_size": 16, "vocab_size": 100, "tie_weights": True,
           "bias": False, "use_moe": False}
    diffs, unchecked = structure_diffs(dict(cfg), dict(cfg))
    assert diffs == [] and unchecked == []


def test_rng_state_roundtrip_replays_same_stream():
    random.seed(1234)
    torch.manual_seed(1234)
    state = _rng_state()
    a_t = torch.rand(3)
    a_p = random.random()
    assert _restore_rng(state) is True
    assert torch.equal(torch.rand(3), a_t), "torch RNG 恢复后应重放同一串随机数"
    assert random.random() == a_p, "python RNG 恢复后应重放同一串随机数"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_restore_rng_accepts_cuda_tensors():
    """续训时 load_checkpoint(map_location=cuda) 会把 RNG 字节张量搬到 GPU，
    _restore_rng 必须能搬回 CPU —— 否则静默不恢复（实测踩过）。"""
    state = _rng_state()
    state["torch"] = state["torch"].cuda()
    if "cuda" in state:
        state["cuda"] = [s.cuda() for s in state["cuda"]]
    assert _restore_rng(state) is True


# ==================================================== run.json（审计 1.2 数据契约）

def test_write_run_meta_roundtrip(tmp_path):
    """训练结束时要在输出目录写 run.json —— 监控页靠它区分预训练 / SFT。"""
    meta = utils.write_run_meta(str(tmp_path), "pretrain", max_steps=3000,
                                tokens_per_step=4096, batch_size=16)
    assert meta["kind"] == "pretrain"
    assert meta["format"] == 1, "格式版本是数据契约，改结构要升它"
    on_disk = utils.load_json(os.path.join(str(tmp_path), "run.json"))
    assert on_disk["max_steps"] == 3000
    assert on_disk["tokens_per_step"] == 4096
    assert on_disk["started"], "要带开始时间，事后回看才知道是哪次跑的"


def test_write_run_meta_failure_never_breaks_training(tmp_path, capsys, monkeypatch):
    """写不进去只告警 —— 训练不能因为一个展示用的元数据文件挂掉。

    注意 `save_json` 会先 makedirs，普通的「目录不存在」根本不会失败，
    这里直接让写文件抛 OSError（模拟磁盘满 / 权限不足）。
    """
    def _boom(path, obj):
        raise OSError("磁盘满（测试模拟）")

    monkeypatch.setattr(utils, "save_json", _boom)
    utils.write_run_meta(str(tmp_path), "sft", max_steps=1)
    cap = capsys.readouterr()
    assert "run.json 没写成" in (cap.out + cap.err)


def test_train_checkpoint_dir_contract_uses_run_meta():
    """train.py 必须真的调用 write_run_meta（契约的另一端）。"""
    import inspect

    from lingyu import train as train_mod

    src = inspect.getsource(train_mod)
    assert "write_run_meta(" in src
    assert '"pretrain"' in src

# ==================================================== 2026-10-24 断点旁车 JSON


def test_read_checkpoint_meta_roundtrip(tmp_path):
    """断点旁车 JSON：读写一致；不 import torch；老断点返回 {}。"""
    import json
    from lingyu.train import read_checkpoint_meta
    ck = tmp_path / "ckpt.pt"
    ck.write_bytes(b"x")
    meta = {"ckpt_format": 1, "step": 42, "best_val": 2.5,
            "model_cfg": {"n_layer": 8}, "tokenizer_path": "tok.txt",
            "tokenizer_fingerprint": "abc", "chat_template": "plain",
            "config_path": "configs/a.ini", "size": 1, "mtime": 0.0}
    (tmp_path / "ckpt.pt.json").write_text(json.dumps(meta), encoding="utf-8")
    got = read_checkpoint_meta(str(ck))
    assert got["step"] == 42
    assert got["model_cfg"]["n_layer"] == 8
    assert got["config_path"] == "configs/a.ini"
    assert read_checkpoint_meta(str(tmp_path / "nope.pt")) == {}


def test_train_parser_accepts_init_from_and_config_path():
    """B-03：train 要能用 --init-from 从别的 run 的断点接着预训练。"""
    from lingyu.cli import build_parser
    parser = build_parser()
    ns = parser.parse_args(["train", "--config", "configs/small.ini",
                            "--init-from", "out/a/best.pt", "--device", "cpu"])
    assert ns.init_from == "out/a/best.pt"
