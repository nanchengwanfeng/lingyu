# -*- coding: utf-8 -*-
"""组件注册表测试：注册、覆盖、插件加载、错误信息质量。

这个模块是「别人能改我的架构」的核心，所以测试要比别处更严。
"""

import os
import sys
import tempfile
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lingyu import registry as R


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后都清干净，避免互相污染。"""
    saved = dict(R._REGISTRY)
    R.clear_plugins()
    yield
    R._REGISTRY.clear()
    R._REGISTRY.update(saved)
    R._PLUGIN_ERRORS.clear()


# ============================================================== 基本注册

def test_register_and_get():
    @R.register("attention", "demo_attn")
    class Demo:
        pass
    assert R.get("attention", "demo_attn") is Demo


def test_register_uses_classname_when_no_name():
    @R.register("ffn")
    class MyFFN:
        pass
    assert R.get("ffn", "myffn") is MyFFN


def test_name_is_case_insensitive():
    @R.register("norm", "MixedCase")
    class N:
        pass
    assert R.get("norm", "mixedcase") is N
    assert R.get("norm", "MIXEDCASE") is N


def test_get_returns_default_when_missing():
    assert R.get("attention", "nope") is None
    assert R.get("attention", "nope", "fallback") == "fallback"


def test_bad_kind_raises_with_list():
    with pytest.raises(ValueError) as ei:
        @R.register("nosuchkind", "x")
        class X:
            pass
    msg = str(ei.value)
    assert "不认识组件种类" in msg
    for k in R.KINDS:
        assert k in msg


def test_duplicate_registration_raises():
    """重名默认报错 —— 静默覆盖是 bug 的温床。"""
    @R.register("attention", "dup")
    class A:
        pass
    with pytest.raises(ValueError, match="已经被注册过"):
        @R.register("attention", "dup")
        class B:
            pass


def test_override_true_allows_replace():
    @R.register("attention", "ovr")
    class A:
        pass

    @R.register("attention", "ovr", override=True)
    class B:
        pass
    assert R.get("attention", "ovr") is B


# ============================================================== 错误信息质量

def test_get_or_die_lists_available():
    @R.register("attention", "aaa")
    class A:
        pass

    @R.register("attention", "bbb")
    class B:
        pass

    with pytest.raises(KeyError) as ei:
        R.get_or_die("attention", "ccc")
    msg = str(ei.value)
    assert "aaa" in msg and "bbb" in msg, "报错必须列出可用名字"
    assert "插件目录" in msg, "报错要告诉用户插件该放哪"


def test_get_or_die_mentions_plugin_errors():
    """插件加载失败时，报错要提示 —— 否则用户以为插件根本没被读到。"""
    R._PLUGIN_ERRORS.append(("/path/plugins/broken.py", "SyntaxError: bad"))
    with pytest.raises(KeyError) as ei:
        R.get_or_die("attention", "missing")
    msg = str(ei.value)
    assert "broken.py" in msg
    assert "加载失败" in msg


def test_list_and_describe():
    @R.register("ffn", "f1")
    class F:
        pass
    # 指定 kind 时只给名字；不指定时给 "kind.name" 全限定名（否则重名分不清）
    assert "f1" in R.list_names("ffn")
    assert "ffn.f1" in R.list_names()
    rows = R.describe("ffn")
    assert any(r["name"] == "f1" and r["kind"] == "ffn" for r in rows)


def test_describe_records_origin_file():
    """describe 要能说出实现来自哪个文件（排查"到底加载了谁"用）。"""
    @R.register("pos", "p1")
    class P:
        pass
    row = [r for r in R.describe("pos") if r["name"] == "p1"][0]
    assert row["origin"] and row["origin"] != "?"


# ============================================================== 插件加载

def _write_plugin(dirpath, name, body):
    p = os.path.join(dirpath, name)
    with open(p, "w", encoding="utf-8") as fp:
        fp.write(textwrap.dedent(body))
    return p


def test_plugin_is_loaded(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _write_plugin(d, "myplug.py", '''
            from lingyu.registry import register

            @register("attention", "from_plugin", override=True)
            class PlugAttn:
                def __init__(self, cfg, layer_idx):
                    pass
                def forward(self, x, cache=None, cache_pos=0):
                    return x
        ''')
        monkeypatch.setenv("LINGYU_PLUGINS", d)
        errs = R.load_plugins(force=True)
        assert not errs, errs
        got = R.get("attention", "from_plugin")
        assert got is not None and got.__name__ == "PlugAttn"
        rows = R.describe("attention")
        row = [r for r in rows if r["name"] == "from_plugin"][0]
        assert row["source"] == "plugin", "来源要标成 plugin"


def test_broken_plugin_does_not_crash_engine(monkeypatch):
    """**关键行为**：一个坏插件不该让整个引擎起不来，但要留下错误记录。"""
    with tempfile.TemporaryDirectory() as d:
        _write_plugin(d, "broken.py", "this is not valid python !!!")
        _write_plugin(d, "good.py", '''
            from lingyu.registry import register

            @register("norm", "good_norm", override=True)
            class G:
                pass
        ''')
        monkeypatch.setenv("LINGYU_PLUGINS", d)
        errs = R.load_plugins(force=True)
        assert len(errs) == 1, "坏插件要被记录"
        assert "broken.py" in errs[0][0]
        assert R.get("norm", "good_norm") is not None, "好插件仍要加载成功"


def test_plugin_package_dir_is_loaded(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "mypkg")
        os.makedirs(pkg)
        _write_plugin(pkg, "__init__.py", '''
            from lingyu.registry import register

            @register("ffn", "pkg_ffn", override=True)
            class P:
                pass
        ''')
        monkeypatch.setenv("LINGYU_PLUGINS", d)
        errs = R.load_plugins(force=True)
        assert not errs, errs
        assert R.get("ffn", "pkg_ffn") is not None


def test_files_starting_with_underscore_are_skipped(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _write_plugin(d, "_helper.py", "raise RuntimeError('不该被加载')")
        _write_plugin(d, "__init__.py", "raise RuntimeError('不该被加载')")
        monkeypatch.setenv("LINGYU_PLUGINS", d)
        errs = R.load_plugins(force=True)
        assert not errs, "下划线开头的文件应该被跳过"


def test_missing_plugins_dir_is_fine(monkeypatch):
    monkeypatch.setenv("LINGYU_PLUGINS", os.path.join(tempfile.gettempdir(), "no_such_dir_xyz"))
    errs = R.load_plugins(force=True)
    assert errs == []


def test_clear_plugins_keeps_builtin(monkeypatch):
    @R.register("attention", "builtin_one")
    class B:
        pass
    with tempfile.TemporaryDirectory() as d:
        _write_plugin(d, "p.py", '''
            from lingyu.registry import register

            @register("attention", "plug_one", override=True)
            class P:
                pass
        ''')
        monkeypatch.setenv("LINGYU_PLUGINS", d)
        R.load_plugins(force=True)
    assert R.get("attention", "plug_one") is not None
    R.clear_plugins()
    assert R.get("attention", "plug_one") is None, "插件项要清掉"
    assert R.get("attention", "builtin_one") is B, "内置项必须留着"


def test_plugins_dir_default_points_to_project():
    d = R.plugins_dir()
    assert os.path.isabs(d)
    assert d.endswith("plugins")
