# -*- coding: utf-8 -*-
"""帮助页：搜索 + 文章阅读。

不联网：所有文章内置在 `gui/help_content.py`（模块级、纯文本，GUI 进程
没有 torch，也没有网络请求）。

搜索三档：
- 智能：空格分词，每个词都要命中，标题/关键词加权；
- 精确：整串原样出现；
- 模糊：difflib 相似度，容错错别字（比如「mde」也能搜到 moe）。
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QSplitter,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import help_content
from ..i18n import t
from ..theme import SPACE
from ..widgets import chip_button, ghost_button, muted
from . import BasePage

_MODES = [("smart", "hp.mode.smart"), ("exact", "hp.mode.exact"),
          ("fuzzy", "hp.mode.fuzzy")]


class HelpPage(BasePage):
    nav_key = "nav.help"
    group_key = "nav.group.misc"
    icon = "?"

    def header(self):
        return (t("hp.title"), t("hp.desc"))

    def build(self):
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # ---------------- 左：搜索 + 文章列表 ----------------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(SPACE["sm"])

        search_row = QHBoxLayout()
        search_row.setSpacing(SPACE["sm"])
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(t("hp.search"))
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._refresh)
        search_row.addWidget(self.search_box, 1)

        self.mode = QComboBox()
        for value, key in _MODES:
            self.mode.addItem(t(key), value)
        self.mode.setToolTip(t("help.mode"))
        self.mode.currentIndexChanged.connect(self._refresh)
        search_row.addWidget(self.mode)
        lv.addLayout(search_row)

        # 热门词：帮助页第一屏要能「点着用」，而不是先教用户怎么搜
        tags = QHBoxLayout()
        tags.setSpacing(SPACE["xs"])
        self._tag_buttons = []
        for key in ("pretrain", "sft", "vram", "moe", "ckpt", "lr"):
            b = chip_button(t("hp.tag." + key), checkable=False)
            b.clicked.connect(lambda _=False, k=key: self._tag(k))
            tags.addWidget(b)
            self._tag_buttons.append(b)
        tags.addStretch(1)
        lv.addLayout(tags)

        # 分组筛选：全部 + 每个组一个 chip
        filters = QHBoxLayout()
        filters.setSpacing(SPACE["xs"])
        self._group = "全部"
        self._filter_buttons = {}
        b = chip_button(t("hp.filter.all"), checkable=True)
        b.setChecked(True)
        b.clicked.connect(lambda _=False: self._set_group("全部"))
        filters.addWidget(b)
        self._filter_buttons["全部"] = b
        for g in help_content._GROUP_ORDER:
            b = chip_button(g, checkable=True)
            b.clicked.connect(lambda _=False, gg=g: self._set_group(gg))
            filters.addWidget(b)
            self._filter_buttons[g] = b
        filters.addStretch(1)
        lv.addLayout(filters)

        self.list_title = muted(t("hp.all"))
        lv.addWidget(self.list_title)

        self.list = QListWidget()
        self.list.setObjectName("HelpList")
        self.list.currentRowChanged.connect(self._show_row)
        lv.addWidget(self.list, 1)

        self.back = ghost_button(t("hp.back"))
        self.back.clicked.connect(self._go_home)
        lv.addWidget(self.back, 0, Qt.AlignLeft)

        split.addWidget(left)

        # ---------------- 右：文章 ----------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(SPACE["sm"])
        self.view = QTextBrowser()
        self.view.setObjectName("HelpView")
        self.view.setOpenExternalLinks(False)
        rv.addWidget(self.view, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 900])
        self.add(split, 1)

        self._articles = []
        self._refresh()

    # ------------------------------------------------------------ 列表
    def _tag(self, key):
        self.search_box.setText(t("hp.tag." + key))

    def _set_group(self, group):
        self._group = group
        for g, b in self._filter_buttons.items():
            b.setChecked(g == group)
        self._refresh()

    def _refresh(self, *_):
        query = self.search_box.text() if hasattr(self, "search_box") else ""
        mode = self.mode.currentData() if hasattr(self, "mode") else "smart"
        if query.strip():
            articles = help_content.search(query, mode or "smart")
            self.list_title.setText(t("hp.results"))
        else:
            articles = help_content.by_group()
            if getattr(self, "_group", "全部") != "全部":
                articles = [a for a in articles if a["group"] == self._group]
            self.list_title.setText(t("hp.all"))
        self._articles = articles
        self.list.blockSignals(True)
        self.list.clear()
        for art in articles:
            item = QListWidgetItem(art["title"])
            item.setData(Qt.UserRole, art["id"])
            self.list.addItem(item)
        self.list.blockSignals(False)
        if not articles:
            self.view.setPlainText(t("hp.empty"))
            self.back.setVisible(False)
            return
        self.back.setVisible(bool(query.strip()))
        self.list.setCurrentRow(0)

    def _show_row(self, row):
        if row < 0 or row >= len(self._articles):
            return
        art = self._articles[row]
        self.view.setMarkdown(art["body"])

    def _go_home(self):
        self.search_box.clear()
        self.list.setCurrentRow(0)

    # ------------------------------------------------------------ 生命周期
    def on_show(self):
        if hasattr(self, "search_box"):
            self.search_box.setFocus()
