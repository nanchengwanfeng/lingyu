# -*- coding: utf-8 -*-
"""页面基类与注册表。

每个页面继承 `BasePage`，实现 `build()`；返回的界面自己管。
`on_show()` 在切到该页时调用，用来刷新数据（而不是一上来就把
所有页面的数据都算一遍 —— 那会让启动变慢）。
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from ..context import ctx
from ..theme import SPACE
from ..widgets import PageHeader


class BasePage(QWidget):
    #: 侧边栏里显示的名字（i18n key）
    nav_key = ""
    #: 分组（i18n key）
    group_key = "nav.group.misc"
    #: 侧边栏图标（用 Unicode 符号，不引外部资源文件 —— 少一个打包麻烦）
    icon = "•"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._built = False
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(SPACE["xl"], SPACE["xl"],
                                      SPACE["xl"], SPACE["xl"])
        self._root.setSpacing(SPACE["lg"])

    # ------------------------------------------------------------ 子类实现
    def header(self):
        """返回 (标题, 说明)。子类覆盖。"""
        return ("", "")

    def build(self):
        """构建界面。在这里往 `self.add()` 里加东西。子类必须覆盖。"""
        raise NotImplementedError

    # ------------------------------------------------------------ 生命周期
    def ensure_built(self):
        if self._built:
            return
        title, desc = self.header()
        if title:
            self._root.addWidget(PageHeader(title, desc))
        self.build()
        self._built = True

    def on_show(self):
        """切到本页时调用。默认什么都不做。"""

    def on_task_event(self, name, code=None):
        """有任务开始/结束时调用，用来禁用按钮之类的。"""

    # ------------------------------------------------------------ 滚动
    def use_scroll_area(self):
        """把页面内容放进滚动区。

        **什么时候该用**：页面全是「固定高度」的内容（卡片、数字、文字），
        窗口变矮时没有哪个元素会主动让位 —— 这时 Qt 会把所有控件压到
        最小尺寸以下，表现为文字互相重叠、卡片挤成一团。

        **什么时候不该用**：页面里有「应该跟着窗口长高」的元素
        （日志框、表格、曲线）。滚动区会让子控件只用 sizeHint 的高度，
        那些元素就再也长不高了 —— 反而更糟。

        所以这个方法得**由页面自己选**，不能一刀切。
        踩过的坑：语料页、训练页左右栏都吃过这个亏（截图里看到的
        标签重叠、键值行叠字），而概览页一直没加 —— 窗口一小就会中招。
        """
        if getattr(self, "_scrolled", False):
            return
        inner = QWidget()
        body = QVBoxLayout(inner)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(self._root.spacing())
        # 把已经加进去的东西搬到 inner 里
        while self._root.count():
            item = self._root.takeAt(0)
            w = item.widget()
            if w is not None:
                body.addWidget(w, item.sizeHint().height() and 0)
            else:
                lay = item.layout()
                if lay is not None:
                    body.addLayout(lay)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(inner)
        self._root.addWidget(sa)
        self._body = body
        self._scrolled = True

    def add(self, w, stretch=0):
        target = getattr(self, "_body", None) or self._root
        target.addWidget(w, stretch)
        return w

    def add_stretch(self):
        target = getattr(self, "_body", None) or self._root
        target.addStretch(1)

    def scrollable(self, inner_widget):
        """把一个很高的内容包进滚动区，避免小窗口下内容被挤没。

        ★ 左右分栏的**每一栏都该包一层**。踩过的坑：语料页左边的表单栏
          没包，内容一超过窗口高度，Qt 就把所有控件压到最小尺寸以下 ——
          表现是标签和输入框互相重叠、挤成一条条，整个页面像坏了。
          QScrollArea 有很小的 minimumHeight，能把这些压力吸收掉。
        """
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(inner_widget)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setFrameShape(QScrollArea.NoFrame)
        return sa

    def column(self, widget, stretch=1):
        """把一个分栏包成可滚动的一栏，返回滚动区。"""
        return self.scrollable(widget)


# ==========================================================================
# 页面注册表：侧边栏按这个顺序生成，不写死在 app.py 里
# ==========================================================================
def all_pages():
    """返回 [(类, nav_key, group_key, icon), …]，顺序即侧边栏顺序。"""
    from .overview import OverviewPage
    from .corpus import CorpusPage
    from .train import TrainPage
    from .monitor import MonitorPage
    from .analysis import AnalysisPage
    from .checkpoints import CheckpointsPage
    from .sample import SamplePage
    from .sft import SftPage
    from .chat import ChatPage
    from .settings import SettingsPage
    from .help import HelpPage

    pages = [
        (OverviewPage,    "nav.overview",    "nav.group.run",  "◎"),
        (CorpusPage,      "nav.corpus",      "nav.group.data", "▤"),
        (TrainPage,       "nav.train",       "nav.group.data", "▶"),
        (MonitorPage,     "nav.monitor",     "nav.group.data", "◔"),
        (AnalysisPage,    "nav.analysis",    "nav.group.data", "∿"),
        (CheckpointsPage, "nav.checkpoints", "nav.group.data", "❖"),
        (SamplePage,      "nav.sample",      "nav.group.post", "✎"),
        (SftPage,         "nav.sft",         "nav.group.post", "✦"),
        (ChatPage,        "nav.chat",        "nav.group.post", "◍"),
        (SettingsPage,    "nav.settings",    "nav.group.misc", "⚙"),
        (HelpPage,        "nav.help",        "nav.group.misc", "?"),
    ]
    return pages
