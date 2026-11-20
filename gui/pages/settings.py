# -*- coding: utf-8 -*-
"""设置页：语言、主题、默认路径。

语言和主题都是**立刻生效**的（不用重启），因为这两个是用户最想立刻看到
效果的东西 —— 让他改完点保存再重启才能看到，体验很差。

主题切换的做法是重新生成整份 QSS 并 setStyleSheet 到主窗口；
已经创建的自绘控件（曲线图、状态点）要收到通知后 update() 重画。
"""

import os

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QVBoxLayout,
                               QWidget)

from ..i18n import t
from ..theme import SPACE, th
from ..widgets import (Card, Field, PathPicker, make_combo,
                       muted, primary_button)
from . import BasePage


class SettingsPage(BasePage):
    nav_key = "nav.settings"
    icon = "⚙"

    def header(self):
        return (t("st.title"), t("st.desc"))

    def build(self):
        cols = QHBoxLayout()
        cols.setSpacing(SPACE["lg"])

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(SPACE["lg"])

        c1 = Card(t("st.appearance"))
        self.lang = make_combo([("简体中文", "zh_CN"), ("English", "en_US")],
                               self.ctx.settings.get("lang", "zh_CN"))
        self.lang.currentIndexChanged.connect(self._lang_changed)
        c1.add(Field(t("st.lang"), self.lang))
        self.theme = make_combo([(t("st.theme.dark"), "dark"),
                                 (t("st.theme.light"), "light")],
                                self.ctx.settings.get("theme", "dark"))
        self.theme.currentIndexChanged.connect(self._theme_changed)
        c1.add(Field(t("st.theme"), self.theme))
        lv.addWidget(c1)

        c2 = Card(t("st.paths"))
        self.raw = PathPicker(self.ctx.settings.get("raw_dir", ""), is_dir=True)
        c2.add(Field(t("st.paths.raw"), self.raw))
        self.data = PathPicker(self.ctx.settings.get("data_dir", ""), is_dir=True)
        c2.add(Field(t("st.paths.data"), self.data))
        self.out = PathPicker(self.ctx.settings.get("out_dir", ""), is_dir=True)
        c2.add(Field(t("st.paths.out"), self.out))
        save = primary_button(t("common.save"))
        save.clicked.connect(self._save)
        c2.add(save)
        self.saved_label = muted("")
        c2.add(self.saved_label)
        lv.addWidget(c2)
        lv.addStretch(1)
        cols.addWidget(left, 3)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(SPACE["lg"])
        c3 = Card(t("st.about"))
        about = QLabel(t("st.about.text"))
        about.setWordWrap(True)
        c3.add(about)
        c3.add(muted(t("st.gui_json", path=os.path.join(
            os.path.expanduser("~"), ".lingyu", "gui.json")), wrap=True))
        rv.addWidget(c3)
        rv.addStretch(1)
        cols.addWidget(right, 2)

        self.add(self._wrap(cols))
        self.add_stretch()

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    # ------------------------------------------------------------ 动作
    def _lang_changed(self):
        code = self.lang.currentData()
        self.ctx.settings["lang"] = code
        self.ctx.save()
        # 换语言要重建界面（文案分散在各页），所以这里通知主窗口
        self.ctx.lang_changed.emit(code)

    def _theme_changed(self):
        name = self.theme.currentData()
        th.set(name)
        self.ctx.settings["theme"] = name
        self.ctx.save()
        self.ctx.theme_changed.emit(name)

    def _save(self):
        self.ctx.settings.update({
            "raw_dir": self.raw.value(),
            "data_dir": self.data.value(),
            "out_dir": self.out.value(),
        })
        ok = self.ctx.save()
        self.saved_label.setText(t("st.saved") if ok
                                 else "保存失败（检查用户目录权限）")
        self.saved_label.setStyleSheet(
            "color:%s;" % (th.SUCCESS if ok else th.DANGER))
        # 顺手把路径写进当前配置，用户在别处就能直接用
        try:
            cfg = self.ctx.config
            if self.data.value():
                cfg.setdefault("data", {})["data_dir"] = self.data.value()
                if not cfg["data"].get("tokenizer_path"):
                    cfg["data"]["tokenizer_path"] = os.path.join(
                        self.data.value(), "tokenizer.txt")
            if self.out.value():
                cfg.setdefault("train", {})["out_dir"] = self.out.value()
            self.ctx.save_config()
        except Exception:
            pass
        self.ctx.corpus_changed.emit()
