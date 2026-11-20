# -*- coding: utf-8 -*-
"""组件注册表 —— 让第三方不改核心代码就能替换模型组件。

## 三层优先级

同名时**后面覆盖前面的**：

    1. 内置实现      @register("attention", "gqa")            ← lingyu/model/*
    2. 插件目录      plugins/*.py 里的 @register              ← 用户丢个文件进去就行
    3. 配置指定      model.attention = "my_attn"              ← 配置文件里点名

## 插件怎么写

在 `plugins/` 目录下放一个 .py 文件（或一个包），里面写：

    from lingyu.registry import register

    @register("attention", "my_attention")
    class MyAttention:
        def __init__(self, cfg, layer_idx):
            ...
        def forward(self, x, cache=None, cache_pos=0):
            return x      # 形状必须保持 (B, T, d_model)

引擎启动时会自动扫描 `plugins/`（可用环境变量 `LINGYU_PLUGINS` 指定别的目录）。

## 怎么在配置里用

    [model]
    attention = my_attention

注册名不存在时，报错会列出所有可用的名字（包括插件注册的），不会只报一句
"KeyError" 让人摸不着头脑。
"""

import importlib
import importlib.util
import os
import sys
import traceback

# {(kind, name): {"obj": 对象, "source": "builtin"/"plugin", "origin": 文件路径}}
_REGISTRY = {}

# 合法的组件种类
KINDS = ("model", "attention", "ffn", "norm", "pos", "optimizer",
         "scheduler", "backend")

_PLUGINS_LOADED = False
_PLUGIN_ERRORS = []


def register(kind, name=None, override=False):
    """装饰器：把一个实现注册进注册表。

    用法：
        @register("attention", "gqa")
        class GQA: ...

        @register("ffn")            # 不写名字就用类名的小写
        class SwiGLU: ...

    override=True 时允许覆盖已注册的同名实现（插件覆盖内置要用它）。
    默认情况下重名会报错 —— 静默覆盖是 bug 的温床。
    """
    def deco(obj):
        key_name = (name or obj.__name__).lower()
        if kind not in KINDS:
            raise ValueError("不认识组件种类 %r。合法的有：%s"
                             % (kind, "、".join(KINDS)))
        key = (kind, key_name)
        old = _REGISTRY.get(key)
        if old is not None and not override:
            raise ValueError(
                "组件 %s.%s 已经被注册过了（来自 %s）。\n"
                "要覆盖它请加 override=True，或换个名字。"
                % (kind, key_name, old.get("origin") or old.get("source")))
        _REGISTRY[key] = {
            "obj": obj,
            "source": "plugin" if _loading_plugin else "builtin",
            "origin": _current_file(),
        }
        return obj
    return deco


# 加载插件时置 True，用来标记来源
_loading_plugin = False


def _current_file():
    """猜一下当前注册动作发生在哪个文件（用于报错时指明来源）。"""
    try:
        for fr in reversed(traceback.extract_stack()[:-2]):
            if fr.filename and not fr.filename.endswith("registry.py"):
                return os.path.basename(fr.filename)
    except Exception:
        pass
    return "?"


def get(kind, name, default=None):
    """按名字取实现。找不到返回 default。"""
    _ensure_plugins_loaded()
    item = _REGISTRY.get((kind, (name or "").lower()))
    return item["obj"] if item else default


def get_or_die(kind, name):
    """按名字取实现，找不到就报一个**有用的**错。

    报错里会列出所有可用名字，并提示插件是否加载失败 ——
    这比一句 KeyError 强太多。
    """
    _ensure_plugins_loaded()
    item = _REGISTRY.get((kind, (name or "").lower()))
    if item is not None:
        return item["obj"]
    avail = list_names(kind)
    msg = ["没有注册名为 %r 的 %s 实现。" % (name, kind)]
    if avail:
        msg.append("可用的有：%s" % "、".join(avail))
    else:
        msg.append("当前一个都没注册 —— 可能是内置模块没被导入。")
    if _PLUGIN_ERRORS:
        msg.append("另外，以下插件加载失败（可能就是你要找的实现）：")
        for path, err in _PLUGIN_ERRORS:
            msg.append("  %s -> %s" % (path, err))
    msg.append("插件目录：%s" % plugins_dir())
    raise KeyError("\n".join(msg))


def list_names(kind=None):
    """列出已注册的名字。kind=None 时列出全部。"""
    _ensure_plugins_loaded()
    out = []
    for (k, n) in sorted(_REGISTRY):
        if kind is None or k == kind:
            out.append("%s.%s" % (k, n) if kind is None else n)
    return out


def describe(kind=None):
    """给 GUI / doctor 用：列出所有组件及其来源。"""
    _ensure_plugins_loaded()
    rows = []
    for (k, n), info in sorted(_REGISTRY.items()):
        if kind is not None and k != kind:
            continue
        rows.append({"kind": k, "name": n,
                     "source": info["source"], "origin": info["origin"]})
    return rows


def unregister(kind, name):
    """移除一个注册项（测试用）。"""
    _REGISTRY.pop((kind, name.lower()), None)


def clear_plugins():
    """清掉所有来自插件的注册项（测试用，避免插件互相污染）。"""
    for key in [k for k, v in _REGISTRY.items() if v["source"] == "plugin"]:
        _REGISTRY.pop(key, None)


# ==========================================================================
# 插件加载
# ==========================================================================

def plugins_dir():
    """插件目录。优先环境变量 LINGYU_PLUGINS，默认 <项目根>/plugins。"""
    env = os.environ.get("LINGYU_PLUGINS")
    if env:
        return os.path.abspath(env)
    # src/lingyu/registry.py -> 上溯三级 = 项目根
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return os.path.join(root, "plugins")


def load_plugins(force=False, verbose=False):
    """扫描并加载插件目录。**失败不影响主流程**，但会记录错误。

    为什么要容错：一个坏插件不该让整个引擎起不来。
    但错误必须留痕，`doctor` 会把它们打出来。
    """
    global _PLUGINS_LOADED, _loading_plugin
    if _PLUGINS_LOADED and not force:
        return _PLUGIN_ERRORS
    _PLUGINS_LOADED = True
    _PLUGIN_ERRORS.clear()

    d = plugins_dir()
    if not os.path.isdir(d):
        return _PLUGIN_ERRORS

    entries = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("_") or fn.startswith("."):
            continue
        full = os.path.join(d, fn)
        if fn.endswith(".py"):
            entries.append((fn[:-3], full, False))
        elif os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.py")):
            entries.append((fn, os.path.join(full, "__init__.py"), True))

    if d not in sys.path:
        sys.path.insert(0, d)

    _loading_plugin = True
    try:
        for name, path, is_pkg in entries:
            try:
                if is_pkg:
                    spec = importlib.util.spec_from_file_location(
                        "lingyu_plugin_" + name, path,
                        submodule_search_locations=[os.path.dirname(path)])
                else:
                    spec = importlib.util.spec_from_file_location(
                        "lingyu_plugin_" + name, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                if verbose:
                    print("  [插件] 已加载 %s" % name)
            except Exception as e:
                _PLUGIN_ERRORS.append((path, "%s: %s" % (type(e).__name__, e)))
    finally:
        _loading_plugin = False
    return _PLUGIN_ERRORS


def plugin_errors():
    return list(_PLUGIN_ERRORS)


def _ensure_plugins_loaded():
    """惰性加载 plugins/ 目录（首次访问注册表时）。

    ## 为什么需要它（审计 B-07）
    文档把「插件」写成架构核心（三层优先级：内置 < 插件 < 配置点名），
    但 `load_plugins()` 在全项目**没有任何调用点** —— `plugins/` 目录形同虚设，
    用户丢进去的组件永远不会被注册。这里做成惰性加载：任何组件查找/列举
    之前先扫一次目录；加载失败不影响主流程（`doctor` 会打印 plugin_errors）。
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    try:
        load_plugins()
    except Exception as e:      # 插件目录不可读之类的极端情况，不能拖垮引擎
        _PLUGIN_ERRORS.append(("plugins/", "%s: %s" % (type(e).__name__, e)))
    finally:
        _PLUGINS_LOADED = True
