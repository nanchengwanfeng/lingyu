# -*- coding: utf-8 -*-
r"""主窗口：侧边栏 + 页面堆栈 + 状态栏。

## 界面结构

    ┌──────────┬─────────────────────────────────┐
    │ ◆ 灵语    │                                 │
    │          │                                 │
    │ 训练      │                                 │
    │  概览     │         页面内容                 │
    │ 数据      │                                 │
    │  语料准备 │                                 │
    │  训练     │                                 │
    │  监控     │                                 │
    │  断点     │                                 │
    │ 对话      │                                 │
    │  续写     │                                 │
    │  微调     │                                 │
    │  聊天     │                                 │
    │          │                                 │
    │ ┌──────┐ │                                 │
    │ │状态灯 │ │                                 │
    │ └──────┘ │                                 │
    ├──────────┴─────────────────────────────────┤
    │ 设备 8GB 显存显卡   显存 6.9 GB   空闲          │
    └────────────────────────────────────────────┘

## 启动速度是硬指标

用户要求「小破显卡也得能打开这个引擎」，所以：

- 模块级**不 import torch**（只有概览页的 `_env_info()` 在真正显示时才 import）
- 页面**懒构建**：切到哪页才建哪页。九页全建要一秒钟以上，第一屏却是概览。
- 启动后立刻显示窗口，后台再刷新数据（用 QTimer.singleShot(0, …) 排到事件循环）

实测冷启动（含 Python 解释器 + PySide6 导入）应该在 1 秒出头。
"""

import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QStackedWidget,
                               QStatusBar, QVBoxLayout, QWidget)

try:                                  # 侧栏版本号只是装饰，取不到就不显示
    from lingyu.version import __version__
except Exception:                     # pragma: no cover - 引擎元数据缺失时
    __version__ = None

from .context import ctx
from .i18n import set_lang, t
from .pages import all_pages
from .theme import FONT_SIZE, SPACE, stylesheet, th
from .widgets import StatusDot, TopBar


class Sidebar(QFrame):
    """左侧导航。用 QPushButton(checkable) + QButtonGroup 做成互斥选中。

    没用 QListWidget：列表项的「选中态」样式在深色主题下很难调干净，
    而且不好在右边加分组小标题。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(208)
        self.buttons = {}
        self.groups = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(SPACE["md"], SPACE["lg"], SPACE["md"], SPACE["md"])
        lay.setSpacing(2)

        # ---- 品牌 ----
        # 以前是「◆  灵语」一个字符标签：不同机器上的字体回退会让那个菱形
        # 忽大忽小、位置跳动。改成「小色块 + 文字」的横向组合，稳定得多。
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(SPACE["sm"])
        mark = QFrame()
        mark.setObjectName("BrandMark")
        brand_row.addWidget(mark, 0, Qt.AlignVCenter)
        brand = QLabel(t("app.name"))
        brand.setObjectName("Brand")
        brand_row.addWidget(brand, 1)
        lay.addLayout(brand_row)
        sub = QLabel(t("app.sub"))
        sub.setObjectName("BrandSub")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        try:
            ver = QLabel("v%s" % __version__)
            ver.setObjectName("BrandVersion")
            lay.addWidget(ver)
        except NameError:
            pass
        lay.addSpacing(SPACE["lg"])

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.nav_keys = []

        last_group = None
        for cls, nav_key, group_key, icon in all_pages():
            if group_key != last_group:
                lbl = QLabel(t(group_key).upper())
                lbl.setObjectName("NavSection")
                lay.addWidget(lbl)
                last_group = group_key
            b = QPushButton("%s   %s" % (icon, t(nav_key)))
            b.setObjectName("NavItem")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setProperty("nav", nav_key)
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[nav_key] = b
            self.nav_keys.append(nav_key)

        lay.addStretch(1)

        # ---- 底部状态 ----
        box = QFrame()
        box.setObjectName("SidebarStatus")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(SPACE["md"], SPACE["sm"], SPACE["md"], SPACE["sm"])
        bl.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(SPACE["sm"])
        self.dot = StatusDot("idle")
        row.addWidget(self.dot)
        self.state_label = QLabel(t("sb.idle"))
        self.state_label.setWordWrap(True)
        row.addWidget(self.state_label, 1)
        bl.addLayout(row)
        self.task_label = QLabel("")
        self.task_label.setWordWrap(True)
        self.task_label.setStyleSheet("color:%s; font-size:9pt;" % th.TEXT_MUTED)
        bl.addWidget(self.task_label)
        lay.addWidget(box)

    def set_state(self, state, text="", task=""):
        self.dot.set_state(state)
        self.state_label.setText(text or t("sb.idle"))
        self.task_label.setText(task)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pages = {}
        self.setWindowTitle("%s — %s" % (t("app.name"), t("app.sub")))
        self.resize(int(ctx.settings.get("window_w", 1280)),
                    int(ctx.settings.get("window_h", 820)))
        self.setMinimumSize(1024, 660)

        # ---------------- 布局 ----------------
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        self.topbar = TopBar()
        self.topbar.theme_toggled.connect(self._toggle_theme)
        self.topbar.lang_toggled.connect(self._toggle_lang)

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        body_lay.addWidget(self.topbar)

        self.stack = QStackedWidget()
        body_lay.addWidget(self.stack, 1)
        root.addWidget(body, 1)
        self.setCentralWidget(central)

        # ---------------- 页面（懒构建）----------------
        self.page_specs = all_pages()
        for i, (cls, nav_key, group_key, icon) in enumerate(self.page_specs):
            holder = QWidget()
            v = QVBoxLayout(holder)
            v.setContentsMargins(0, 0, 0, 0)
            self.stack.addWidget(holder)
            self.pages[nav_key] = None            # 占位，切到时才建
        self._classes = {k: c for c, k, _g, _i in self.page_specs}
        self._holders = {k: self.stack.widget(i)
                         for i, (_c, k, _g, _i) in enumerate(self.page_specs)}

        self._nav_group = {k: g for _c, k, g, _i in self.page_specs}
        self.sidebar.group.buttonClicked.connect(
            lambda b: self.goto(b.property("nav")))
        self.sidebar.buttons[self.page_specs[0][1]].setChecked(True)

        # ---------------- 状态栏 ----------------
        sb = QStatusBar()
        sb.setObjectName("StatusBar")
        self.setStatusBar(sb)
        self.status_device = QLabel("")
        self.status_vram = QLabel("")
        self.status_hint = QLabel("")
        sb.addWidget(self.status_device)
        sb.addPermanentWidget(self.status_hint)
        sb.addPermanentWidget(self.status_vram)

        # ---------------- 信号 ----------------
        ctx.theme_changed.connect(self._apply_theme)
        ctx.lang_changed.connect(self._rebuild_for_lang)
        ctx.task_started.connect(lambda n: self._on_task(n, None))
        ctx.task_finished.connect(self._on_task)
        ctx.corpus_changed.connect(lambda: self._refresh_chrome())
        ctx.checkpoints_changed.connect(lambda: self._refresh_chrome())
        ctx.config_changed.connect(lambda: self._refresh_chrome())
        # 环境探测（后台线程）回来后刷新状态栏和当前的页面
        ctx.env_ready.connect(self._on_env_ready)

        # 快捷键：Ctrl+1..9 切前九页，第 10 页（帮助）用 Ctrl+0
        for i, (_c, k, _g, _i) in enumerate(self.page_specs[:10], 1):
            key = "Ctrl+%d" % (i if i <= 9 else 0)
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda kk=k: self.goto(kk))

        # 第一屏先画出来，数据刷新排到事件循环里 ——
        # 这样用户立刻看到窗口，而不是等几秒白屏
        self.goto(self.page_specs[0][1])
        QTimer.singleShot(0, self._refresh_chrome)
        # 窗口已经画出来了，现在才去后台 import torch 探测显卡。
        # 放在窗口构造里而不是 run() 里，是为了让**任何**入口路径
        # （lingyu gui / python -m gui / 测试里直接 new）都能拿到环境信息。
        # 测试里不自动探：探针线程 import torch 要 3~8 秒，而 pytest 进程
        # 往往在这之前就退出 —— QThreadPool 的线程还活着，解释器收尾时
        # Qt 会直接 0xC0000005 崩掉（实测 test_gui.py 全绿但退出码
        # -1073741819）。探针逻辑本身有 G-15 测试直接驱动，不靠窗口起。
        if not os.environ.get("LINGYU_GUI_NO_PROBE"):
            QTimer.singleShot(50, ctx.start_env_probe)

    # ------------------------------------------------------------ 导航
    def goto(self, nav_key):
        if nav_key not in self._holders:
            return
        # 懒构建
        if self.pages.get(nav_key) is None:
            holder = self._holders[nav_key]
            page = self._classes[nav_key]()
            lay = holder.layout()
            lay.addWidget(page)
            self.pages[nav_key] = page
        page = self.pages[nav_key]
        self.stack.setCurrentWidget(self._holders[nav_key])
        try:
            self.topbar.set_page(t(self._nav_group.get(nav_key, "")),
                                 t(nav_key))
        except Exception:
            pass
        self._fade_in(self._holders[nav_key])
        b = self.sidebar.buttons.get(nav_key)
        if b and not b.isChecked():
            b.setChecked(True)
        try:
            page.ensure_built()
            page.on_show()
        except Exception as e:
            # 单页出错不该让整个界面挂掉 —— 用户至少能切到别的页
            import traceback
            traceback.print_exc()
            self.status_hint.setText("页面出错：%s" % e)

    # ------------------------------------------------------------ 动效
    def _fade_in(self, widget):
        """切页时 140ms 淡入（审计 1.3：要动效，但不花哨）。

        - 只做透明度，不做位移/缩放 —— 动效是过渡，不是表演；
          九页里有表单和表格，位移动画会让文字在移动中发虚。
        - `LINGYU_GUI_NO_ANIM=1` 时完全跳过：测试和 audit 截图需要
          立刻拿到最终画面（否则 grab() 可能拍到半透明的一帧）。
        """
        if os.environ.get("LINGYU_GUI_NO_ANIM"):
            return
        try:
            from PySide6.QtCore import QEasingCurve, QPropertyAnimation
            from PySide6.QtWidgets import QGraphicsOpacityEffect
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(140)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)

            def _done():
                try:
                    widget.setGraphicsEffect(None)
                except RuntimeError:      # 窗口已销毁
                    pass
                self._fade_anim = None

            anim.finished.connect(_done)
            self._fade_anim = anim
            anim.start()
        except Exception:
            # 动效失败绝不能影响切页本身
            try:
                widget.setGraphicsEffect(None)
            except Exception:
                pass

    def _on_env_ready(self, info):
        self._refresh_chrome()
        # 概览页和训练页的估算依赖可用显存，探测到之后要重算一遍
        for key in ("nav.overview", "nav.train"):
            p = self.pages.get(key)
            if p is not None:
                try:
                    p.on_show()
                except Exception:
                    pass

    # ------------------------------------------------------------ 外观
    def _apply_theme(self, name=None):
        app = QApplication.instance()
        app.setStyleSheet(stylesheet())
        self.update()
        for p in self.pages.values():
            if p is not None:
                p.update()

    def _rebuild_for_lang(self, code=None):
        """换语言：重建整个窗口（页面文案是创建时定的）。

        简单粗暴但可靠 —— 逐个控件 retranslate 要写几百行，还容易漏。

        ⚠️ 重建**不能连带杀掉后台任务**（审计 G-02）：`close()` 会走
        `closeEvent`，那里原本会 stop/kill 所有页面的 runner。
        这里先打 `_rebuilding` 标记，closeEvent 看到就跳过任务清理。
        任务还在跑时旧窗口必须保留（否则 QProcess 析构会杀子进程），
        空闲时直接 deleteLater，避免旧窗口越积越多。
        """
        if getattr(self, "_rebuilding", False):
            return
        # ★ 只重建窗口不切 i18n 的话，新窗口里的 t() 还是旧语言（审计 G-16）。
        set_lang(code or ctx.settings.get("lang", "zh_CN"))
        self._rebuilding = True
        try:
            ctx.settings["window_w"] = self.width()
            ctx.settings["window_h"] = self.height()
            ctx.save()
        except Exception:
            pass
        self.close()          # closeEvent 因 _rebuilding 跳过 runner 清理
        win = MainWindow()
        win.show()
        if ctx.is_busy():
            _KEEP.append(self)    # 任务还在跑：旧窗口必须活着，任务结束再回收
        else:
            self.deleteLater()

    # ------------------------------------------------------------ 主题/语言
    def _toggle_theme(self):
        name = "light" if th.is_dark else "dark"
        self._apply_theme(name)

    def _toggle_lang(self):
        code = ("en_US" if str(ctx.settings.get("lang", "zh_CN")).startswith("zh")
                else "zh_CN")
        ctx.settings["lang"] = code
        ctx.save()
        ctx.lang_changed.emit(code)

    # ------------------------------------------------------------ 状态
    def _on_task(self, name, code):
        if code is None:
            self.sidebar.set_state("run", t("common.running"),
                                   "%s: %s" % (t("sb.task"), name))
            self.topbar.set_state("run", "%s: %s" % (t("sb.task"), name))
            self.status_hint.setText("%s %s…" % (name, t("common.running")))
        else:
            # ★ 用户主动停止 ≠ 失败（审计 G-04/G-01）。Runner 退出前会把
            #   stopped 标记写进 ctx；这里优先按"已停止"显示，不再报红。
            stopped = bool(getattr(ctx, "last_finished_stopped", False))
            if stopped:
                self.sidebar.set_state("ok", t("common.done"),
                                       "%s: %s" % (t("sb.task"), name))
            else:
                self.sidebar.set_state("ok" if code == 0 else "error",
                                       t("common.done") if code == 0
                                       else t("common.failed"))
            self.status_hint.setText("")
            self._refresh_chrome()
        for p in self.pages.values():
            if p is not None:
                try:
                    p.on_task_event(name, code)
                except Exception:
                    pass
        # 因换语言而保留的旧窗口：任务全结束后回收，避免 _KEEP 无限增长
        if getattr(self, "_rebuilding", False) and not ctx.is_busy():
            self.deleteLater()

    def _refresh_chrome(self):
        """状态栏信息 —— 只读 `ctx.gpu` 缓存，**绝不在这里 import torch**。

        torch 由后台线程探测一次，探测完发 `env_ready` 信号回来刷新。
        这样窗口能立刻出现并保持可交互，而不是冻住几秒等 import。
        """
        g = ctx.gpu
        if g.get("probing"):
            self.status_device.setText("● 正在检测环境…")
            self.status_vram.setText("")
            return
        if not g.get("torch") or g.get("torch") == "未安装":
            self.status_device.setText("● PyTorch 未就绪")
            self.status_vram.setText("")
            return
        if g.get("vram_total"):
            name = g.get("device_short", "")
            self.status_device.setText("● %s" % name)
            self.status_vram.setText("显存 %s" % g.get("vram_text", ""))
        else:
            self.status_device.setText("● " + t("sb.cpu"))
            self.status_vram.setText("")

    def closeEvent(self, e):
        try:
            ctx.settings["window_w"] = self.width()
            ctx.settings["window_h"] = self.height()
            ctx.save()
        except Exception:
            pass
        if getattr(self, "_rebuilding", False):
            # 换语言只是重建窗口：后台训练/推理必须继续跑
            return super().closeEvent(e)
        # 关窗口时把还在跑的页面任务停掉，别留孤儿进程
        for p in self.pages.values():
            r = getattr(p, "runner", None)
            if r is not None and r.is_running():
                r.stop(grace_ms=800)
            proc = getattr(p, "proc", None)
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        # 任务可能来自被重建前的旧窗口（当前窗口页面上的 runner 是新的）：
        # 全局总线上挂的那个才是真正在跑的（审计 1.7 / G-02）
        r = ctx.task_runner
        try:
            if r is not None and r.is_running():
                r.stop(grace_ms=800)
        except Exception:
            pass
        super().closeEvent(e)


# 换语言时用来托住旧窗口的列表（Qt 里 Python 对象一旦没人引用就会被回收，
# 那样新窗口可能连带被销毁）
_KEEP = []


def run(argv=None):
    """启动界面。返回退出码。"""
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("LingYu")
    app.setApplicationDisplayName(t("app.name"))
    # 退出前等后台探针线程收尾（否则解释器析构时 QThreadPool 回调会硬崩）
    app.aboutToQuit.connect(ctx.shutdown)

    th.set(ctx.settings.get("theme", "dark"))
    app.setStyleSheet(stylesheet())

    # 中文字体：显式设一个，避免在英文系统上落回点阵字体
    f = QFont()
    f.setPointSize(FONT_SIZE["body"])
    app.setFont(f)

    win = MainWindow()
    win.show()
    _KEEP.append(win)
    return app.exec()
