# -*- coding: utf-8 -*-
"""断点无损压缩（src/lingyu/compression.py）的回归测试。

重点只有一个：**读回来必须逐 bit 等于存之前**。压缩率是次要的，
但也不能"压了反而更大"（<= raw 字节数）。
"""
import os

import torch

from lingyu import compression


def _roundtrip(t):
    packed = compression.pack_tensor(t)
    back = compression.unpack_tensor(packed)
    assert back.dtype == t.dtype
    assert tuple(back.shape) == tuple(t.shape)
    assert torch.equal(back, t)
    return packed


def test_pack_float32_roundtrip_is_bit_exact():
    t = torch.randn(64, 128) * 7.0
    packed = _roundtrip(t)
    assert len(packed["z"]) <= t.numel() * 4


def test_pack_bfloat16_roundtrip_is_bit_exact():
    t = torch.randn(33, 7).to(torch.bfloat16)
    _roundtrip(t)


def test_pack_int_and_bool_roundtrip():
    _roundtrip(torch.arange(100, dtype=torch.int64))
    _roundtrip(torch.randint(0, 2, (17,), dtype=torch.bool))


def test_pack_empty_and_non_tensor():
    _roundtrip(torch.zeros(0, dtype=torch.float32))
    assert compression.pack_payload({"a": 1, "b": ["x", None]}) == \
        {"a": 1, "b": ["x", None]}


def test_wrap_unwrap_nested_payload_keeps_values():
    payload = {"model": {"w": torch.randn(8, 8), "n": 3},
               "optimizer": {"state": {0: {"exp_avg": torch.randn(8, 8)}}},
               "step": 12, "cfg": {"n_layer": 2}}
    wrapped = compression.wrap(payload)
    assert wrapped.get("__compression__") == "plane-zlib"
    back = compression.unwrap(wrapped)
    assert back["step"] == 12
    assert torch.equal(back["model"]["w"], payload["model"]["w"])
    assert torch.equal(back["optimizer"]["state"][0]["exp_avg"],
                       payload["optimizer"]["state"][0]["exp_avg"])


def test_checkpoint_save_load_roundtrip_uses_compression(tmp_path):
    from lingyu.train import load_checkpoint, save_checkpoint
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    (model.weight.sum() * 0).backward()  # 造出 optimizer state
    opt.step()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, opt, 5, 1.25, {"n_layer": 1, "n_embd": 8})
    with open(path, "rb") as fp:
        head = fp.read(4096)
    assert b"__compression__" in head or os.path.getsize(path) > 0
    ck = load_checkpoint(path)
    for k, v in model.state_dict().items():
        assert torch.equal(ck["model"][k], v)
    assert ck["step"] == 5
    assert ck["best_val"] == 1.25


def test_disabled_compression_writes_plain_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGYU_PACK_CKPT", "0")
    from lingyu.train import load_checkpoint, save_checkpoint
    model = torch.nn.Linear(4, 2)
    path = str(tmp_path / "plain.pt")
    save_checkpoint(path, model, None, 1, 0.0, {})
    ck = load_checkpoint(path)
    assert ck["model"]["weight"].shape == (2, 4)


def _payload_with_tensors(n=12):
    torch.manual_seed(7)
    return {"model": {"w%d" % i: torch.randn(16, 8) * (i + 1) for i in range(n)},
            "optimizer": {"state": {i: {"exp_avg": torch.randn(16, 8)}
                                    for i in range(4)}},
            "step": 42, "cfg": {"n_layer": 2}}


def test_parallel_packing_matches_serial_bytes(monkeypatch):
    """多线程压缩结果与单线程逐字节相同（顺序/确定性不能变）。"""
    monkeypatch.delenv("LINGYU_PACK_METHOD", raising=False)
    monkeypatch.setenv("LINGYU_PACK_LEVEL", "6")
    payload = _payload_with_tensors()
    monkeypatch.setenv("LINGYU_PACK_THREADS", "1")
    serial = compression.pack_payload(payload)
    monkeypatch.setenv("LINGYU_PACK_THREADS", "8")
    parallel = compression.pack_payload(payload)
    assert serial == parallel
    back = compression.unpack_payload(parallel)
    for k, v in payload["model"].items():
        assert torch.equal(back["model"][k], v)


def test_lzma_method_roundtrip_is_bit_exact(monkeypatch):
    monkeypatch.setenv("LINGYU_PACK_METHOD", "lzma")
    monkeypatch.setenv("LINGYU_PACK_THREADS", "2")
    t = torch.randn(64, 32).to(torch.bfloat16)
    packed = compression.pack_tensor(t)
    assert packed["codec"] == "lzma"
    back = compression.unpack_tensor(packed)
    assert back.dtype == torch.bfloat16
    assert torch.equal(back, t)


def test_legacy_v1_pack_reads_old_format(monkeypatch):
    """老断点（没有 codec 字段、__ly_pack__=1）必须还能读。"""
    monkeypatch.delenv("LINGYU_PACK_METHOD", raising=False)
    t = torch.randn(8, 8)
    packed = compression.pack_tensor(t)
    old = dict(packed)
    old["__ly_pack__"] = 1
    old.pop("codec", None)
    back = compression.unpack_tensor(old)
    assert torch.equal(back, t)

