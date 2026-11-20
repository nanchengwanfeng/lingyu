# -*- coding: utf-8 -*-
"""加速开关测试（accelerators.py）。

原则：这些开关只影响「怎么算」，不能改变模型定义或训练结果格式；
测试只断言「能设置/能被关掉/不抛异常」，不依赖具体硬件是否可用。
"""

import os

import pytest

torch = pytest.importorskip("torch")

from lingyu import accelerators


def test_apply_returns_dict_and_sets_threads(monkeypatch):
    monkeypatch.delenv("LINGYU_NO_ACCEL", raising=False)
    monkeypatch.setenv("LINGYU_CPU_THREADS", "3")
    applied = accelerators.apply("cpu")
    assert isinstance(applied, dict)
    assert applied.get("cpu_threads") == 3
    assert torch.get_num_threads() >= 1


def test_apply_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LINGYU_NO_ACCEL", "1")
    assert accelerators.apply("cpu") == {}
    assert "关闭" in accelerators.describe({})


def test_describe_mentions_enabled_items(monkeypatch):
    monkeypatch.delenv("LINGYU_NO_ACCEL", raising=False)
    monkeypatch.setenv("LINGYU_CPU_THREADS", "2")
    text = accelerators.describe(accelerators.apply("cpu"))
    assert "CPU 线程 2" in text


def test_maybe_compile_off_by_default(monkeypatch):
    monkeypatch.delenv("LINGYU_COMPILE", raising=False)
    m = torch.nn.Linear(2, 2)
    assert accelerators.maybe_compile(m, "cpu") is m
