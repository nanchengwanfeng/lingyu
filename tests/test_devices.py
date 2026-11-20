# -*- coding: utf-8 -*-
from lingyu import devices


def test_available_always_has_cpu_fallback():
    rows = devices.available()
    assert rows, "至少要有 cpu"
    assert rows[-1][0] == "cpu"


def test_pick_auto_returns_available_device():
    dev = devices.pick("auto")
    assert dev in [d for d, _v, _n in devices.available()]


def test_pick_explicit_cpu():
    assert devices.pick("cpu") == "cpu"


def test_pick_unknown_device_raises():
    try:
        devices.pick("definitely-not-a-device")
    except ValueError as exc:
        assert "不可用" in str(exc)
    else:
        raise AssertionError("应当抛 ValueError")


def test_describe_is_text_lines():
    rows = devices.describe()
    assert rows and all(isinstance(x, str) for x in rows)
