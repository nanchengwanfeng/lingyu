# -*- coding: utf-8 -*-
"""路径解析的回归测试。

## 这一组要守住什么

打包版曾经出现「界面刚说语料准备完成，训练页就说还没准备语料」。
两个目录都合法、都真实存在、代码里没有任何异常 —— 只是**不是同一个**：

    界面的语料准备页写到了   <runtime>/data
    界面读的配置却指着       <runtime>/app/data1

根因是**同一个问题在三处各算了一遍「根目录在哪」**，而其中两处算错了：

    gui/context.py   按 __file__ 往上数 -> runtime/app   （错）
    gui/runner.py    按 __file__ 往上数 -> runtime/app   （错）
    cli.py           LINGYU_HOME       -> runtime        （对）
    paths.find_config 还把「当前工作目录」当第一候选，而打包版的工作目录
                      恰好就是 runtime/app —— 于是稳稳命中那份源码副本

所以这里的测试都用 `LINGYU_HOME` 模拟打包版布局，重点验两件事：
**根目录只有一种算法**、**相对路径一律相对它**。
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lingyu import paths  # noqa: E402


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    """造一个「打包版」布局：<rt>/app/src/lingyu + <rt>/configs + <rt>/data。

    开发版和打包版只差 `LINGYU_HOME` 一个环境变量，所以这个夹具
    能用最小的代价把打包版的路径行为搬进测试里。
    """
    rt = tmp_path / "runtime"
    app = rt / "app"
    pkg = app / "src" / "lingyu"
    pkg.mkdir(parents=True)
    (rt / "configs").mkdir()
    (rt / "configs" / "small.ini").write_text(
        "[data]\ndata_dir = data\n", encoding="utf-8")
    (app / "configs").mkdir()
    # 源码自带的那份：同样是相对路径，但基准完全不同
    (app / "configs" / "small.ini").write_text(
        "[data]\ndata_dir = data1\n", encoding="utf-8")
    monkeypatch.setenv("LINGYU_HOME", str(rt))
    monkeypatch.setattr(paths, "__file__", str(pkg / "paths.py"))
    return rt, app


def test_home_honours_lingyu_home(packaged):
    rt, app = packaged
    assert paths.home() == os.path.normpath(str(rt))
    # 源码的家和用户数据的家是**两个**目录，别混
    assert paths.package_root() == os.path.normpath(str(app))


def test_home_falls_back_to_package_root(monkeypatch, tmp_path):
    monkeypatch.delenv("LINGYU_HOME", raising=False)
    pkg = tmp_path / "src" / "lingyu"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(paths, "__file__", str(pkg / "paths.py"))
    assert paths.home() == os.path.normpath(str(tmp_path))


def test_configs_dir_prefers_runtime(packaged):
    rt, app = packaged
    assert paths.configs_dir() == os.path.normpath(str(rt / "configs"))


def test_abspath_is_relative_to_home_not_cwd(packaged, monkeypatch):
    """★ 核心回归。

    `data_dir = data` 必须解析到 `<runtime>/data`，
    **不能**因为「当前工作目录是 runtime/app」就变成 `<runtime>/app/data`。
    """
    rt, app = packaged
    monkeypatch.chdir(app)                    # 打包版真实的工作目录
    assert paths.abspath("data") == os.path.normpath(str(rt / "data"))
    assert paths.abspath("out/small") == os.path.normpath(str(rt / "out" / "small"))
    # 绝对路径原样返回（顺便把两种分隔符统一）
    assert paths.abspath(str(rt / "data")) == os.path.normpath(str(rt / "data"))
    assert paths.abspath("") == ""


def test_find_config_picks_user_config_not_source_copy(packaged, monkeypatch):
    """★ 核心回归：**工作目录里的那份源码副本不许赢**。

    打包版的工作目录就是 `runtime/app`，而 `runtime/app/configs/small.ini`
    真实存在、格式合法、能正常读出来 —— 它只是不是用户那份。
    以前候选列表把 cwd 排在第一位，于是稳定命中它，
    连带 `data_dir = data1` 一起生效，语料就「消失」了。
    """
    rt, app = packaged
    monkeypatch.chdir(app)
    got = paths.find_config("configs/small.ini")
    assert got == os.path.normpath(str(rt / "configs" / "small.ini"))
    assert got != os.path.normpath(str(app / "configs" / "small.ini"))


def test_find_config_accepts_bare_name(packaged):
    rt, app = packaged
    assert paths.find_config("small.ini") == \
        os.path.normpath(str(rt / "configs" / "small.ini"))


def test_find_config_keeps_absolute_path(packaged):
    rt, app = packaged
    p = str(rt / "configs" / "small.ini")
    assert paths.find_config(p) == os.path.normpath(p)


def test_find_config_missing_returns_writable_default(packaged):
    """找不到时返回的路径要落在**用户配置目录**，而不是 cwd 或源码目录。"""
    rt, app = packaged
    got = paths.find_config("nonexistent.ini")
    assert got == os.path.normpath(str(rt / "configs" / "nonexistent.ini"))


def test_dev_layout_unchanged(monkeypatch):
    """开发版没设 LINGYU_HOME，根就是项目根，老行为不能变。"""
    monkeypatch.delenv("LINGYU_HOME", raising=False)
    monkeypatch.chdir(ROOT)
    assert paths.home() == os.path.normpath(ROOT)
    assert paths.find_config("configs/small.ini") == \
        os.path.normpath(os.path.join(ROOT, "configs", "small.ini"))


# ==================================================== 三处入口必须同源

def test_gui_and_cli_share_one_root():
    """GUI、子进程、命令行必须算出**同一个**根。

    这条不是形式主义：它们各算各的正是上面那起事故的成因。
    """
    pytest.importorskip("PySide6", reason="GUI 是可选依赖")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.context import Context
    from gui.runner import Runner
    from lingyu import cli

    QApplication.instance() or QApplication([])
    assert Context().project_root == paths.home()
    assert Runner(None)._project_root() == paths.home()
    assert cli._project_root() == paths.home()


def test_root_helpers_do_not_walk_up_from_dunder_file():
    """不许再出现「按 __file__ 往上数三层」这种自己算根目录的写法。

    源码在打包版里位于 `runtime/app/...`，往上数三层得到的是**源码的家**，
    而不是用户数据的家。这个错误没有任何症状，只是东西不在你以为的地方 ——
    所以用静态检查钉住，别指望下次 review 能看出来。
    """
    import ast

    bad = []
    for rel in ("gui/context.py", "gui/runner.py"):
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            # 找形如 dirname(dirname(dirname(abspath(__file__)))) 的调用链
            depth = 0
            cur = node
            while (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name)
                   and cur.func.id in ("dirname", "abspath")):
                if cur.func.id == "dirname":
                    depth += 1
                if not cur.args:
                    break
                cur = cur.args[0]
            if depth >= 3:
                bad.append("%s:%d" % (rel, getattr(node, "lineno", 0)))
    assert not bad, "这些地方在自己算根目录，应该用 lingyu.paths.home()：%s" % bad
