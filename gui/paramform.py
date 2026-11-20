# -*- coding: utf-8 -*-
r"""参数表单 —— **由 spec.py 自动生成**，所以 89 个参数一个都不会漏。

## 为什么必须自动生成

「GUI 要能控制几乎所有参数」这个需求，如果靠手写表单，只有一个结局：
spec 里加了新参数，界面忘了加，于是「文档说能改、配置文件里能改、界面里找不到」。
改三次以后就没人信这个界面了。

所以这里读 `config/spec.py`，按类型自动选控件：

    int/float -> 数字框（带 min/max 范围，从 spec 里取）
    bool      -> 勾选框
    str + choices -> 下拉框
    str       -> 输入框
    路径类    -> 输入框 + 「浏览…」

参数的**说明文字**也来自 spec 的 `help_zh` / `help_en`，界面不抄第二份。

## 三个分页：常用 / 进阶 / 专家

spec 里每个参数都有 `tier`。初学者只该看到「常用」那几条（层数、宽度、
批大小、学习率……），不然一打开界面就是 89 个框，直接劝退。
进阶和专家放在另外两个标签页里，想调的人自己点进去。

## 一个容易忽略的细节：dirty 追踪

改了哪些参数要能告诉用户（否则他不知道「我刚才改的生效了吗」）。
这里给每个被改动过的控件加一圈高亮边框，并且提供 `dirty_keys()`。
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QScrollArea,
                               QTabWidget, QVBoxLayout, QWidget)

from .i18n import param_help, t
from .theme import SPACE, th
from .widgets import (Field, PathPicker, ghost_button, hline, make_check,
                      make_combo, make_float, make_int, muted, section_label,
                      set_combo_value, unify_labels)

# 哪些参数是「路径」，要配浏览按钮
_PATH_HINTS = ("dir", "path", "out_dir", "init_from", "from")

# 这些参数的候选项来自注册表（插件可以加），所以选项要在运行时问 registry，
# 而不是照抄 spec 里那份静态的 choices（那份只是给校验器和文档用的）
_REGISTRY_CHOICES = {
    "attention": "attention",
    "ffn": "ffn",
    "norm": "norm",
    "pos": "pos",
}


def _live_choices(sec, key, spec_choices):
    """候选值：组件类参数从注册表现取（能看到插件注册的名字）。"""
    kind = _REGISTRY_CHOICES.get(key)
    if sec == "model" and kind:
        try:
            from lingyu import registry
            names = registry.list_names(kind)
            if names:
                # 空字符串 = 自动，永远排在最前面
                return [""] + [n for n in names if n]
        except Exception:
            pass
    return list(spec_choices or [])


def _decimals_for(val, lo, hi):
    """按数值量级决定小数位数。

    ## 这个函数是拿来修一个严重 bug 的

    原来这里写的是「learning_rate 和 min_lr 用 2 位小数，其余 4 位」。
    而学习率的典型值是 5e-4 —— **2 位小数会把它显示成 0.00，
    取回来就是 0.0**。也就是说：打开训练页、什么都不改、点开始训练，
    存进配置的学习率就是 0，模型一步都学不动。

    同样的问题也发生在 norm_eps（1e-6，4 位小数变成 0.0）。

    所以小数位必须**按量级算**，不能按参数名硬编码：
    宁可多给几位（显示长一点），也不能把值截成 0。
    """
    vals = [abs(float(v)) for v in (val, lo, hi)
            if v is not None and float(v) != 0.0]
    m = min(vals) if vals else 1.0
    if m >= 1.0:
        return 2
    if m >= 0.01:
        return 4
    if m >= 1e-4:
        return 6
    if m >= 1e-6:
        return 9
    return 12


class ParamForm(QWidget):
    """一整段（section）的参数表单，按 tier 分成三个标签页。"""

    changed = Signal()

    def __init__(self, section, values=None, only_keys=None, parent=None):
        super().__init__(parent)
        self.section = section
        self.widgets = {}         # key -> 控件
        self.fields = {}          # key -> Field
        self._initial = {}        # key -> 初始值（用来判断有没有改过）
        self._dirty = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACE["md"])

        from lingyu.config import SPEC
        params = SPEC.get(section, {})

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.tabs = tabs
        outer.addWidget(tabs)

        values = values or {}
        by_tier = {"basic": [], "advanced": [], "expert": []}
        for key, p in params.items():
            if only_keys is not None and key not in only_keys:
                continue
            tier = p.get("tier", "advanced")
            by_tier.setdefault(tier, []).append((key, p))

        for tier, label in (("basic", t("tr.tier.basic")),
                            ("advanced", t("tr.tier.advanced")),
                            ("expert", t("tr.tier.expert"))):
            page = self._build_page(by_tier.get(tier) or [], values)
            tabs.addTab(page, "%s (%d)" % (label, len(by_tier.get(tier) or [])))

        # 常用那页如果没有参数，就直接停在第一个非空页
        for i in range(tabs.count()):
            if tabs.widget(i).findChild(QWidget, "FormPageInner") is not None:
                tabs.setCurrentIndex(i)
                break

        # ★ 跨页再统一一次：上面每个 tier 页各自 unify，不同页的标签列宽度
        #   可能不同（Linux 字体下实测 99 vs 109），切页时左边缘会轻微跳动。
        #   全表单取同一个宽度，三页的输入框左边缘才完全对齐。
        unify_labels(list(self.fields.values()))

    # ------------------------------------------------------------ 构建
    def _build_page(self, items, values):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setObjectName("FormPageInner")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(2, SPACE["md"], 2, SPACE["md"])
        lay.setSpacing(SPACE["md"])

        if not items:
            lay.addWidget(muted(t("common.empty")))
            lay.addStretch(1)
            scroll.setWidget(inner)
            return scroll

        # 按 spec 里的声明顺序排列（spec 本身就是按「重要 -> 次要」写的）
        fields = []
        for key, p in items:
            cur = values.get(key, p["default"])
            w = self._make_widget(key, p, cur)
            self.widgets[key] = w
            self._initial[key] = cur
            # 参数说明**不常驻**在输入框下面（1.5）：参数多的页会变成一堵
            # 灰色小字墙。改成标签后的「?」徽标，悬停才出完整说明。
            f = Field(key, w, help_text=param_help(p))
            self.fields[key] = f
            fields.append(f)

        # ★ 整组标签等宽。不统一的话控件那一列的左边缘会参差不齐 ——
        #   单看每个都对，整片看就是"没对齐"，这是界面显廉价最直接的原因。
        unify_labels(fields)
        for f in fields:
            lay.addWidget(f)

        lay.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _make_widget(self, key, p, cur):
        typ = p["type"]
        if typ is bool:
            w = make_check("", bool(cur))
        elif typ is int:
            w = make_int(p.get("min", -10 ** 9), p.get("max", 10 ** 9),
                         int(cur or 0), p.get("step", 1))
        elif typ is float:
            lo = p.get("min", -1e12)
            hi = p.get("max", 1e12)
            w = make_float(lo, hi, float(cur or 0.0),
                           p.get("step", 0.1),
                           _decimals_for(cur, lo, hi))
        elif typ is str and (_looks_like_path(key) or p.get("path")):
            is_dir = key.endswith("dir") or key.endswith("dirs")
            w = PathPicker(str(cur or ""), is_dir=is_dir)
        elif typ is str and p.get("choices"):
            choices = _live_choices(self.section, key, p["choices"])
            items = [(_choice_label(choices, c), c) for c in choices]
            w = make_combo(items or [("", "")], cur)
        else:
            from PySide6.QtWidgets import QLineEdit
            w = QLineEdit(str(cur if cur is not None else ""))

        self._connect(w, key)
        return w

    def _connect(self, w, key):
        """统一接上「值变了」的信号，并标脏。"""
        def on_change(*_a):
            self._mark(key)
            self.changed.emit()

        if isinstance(w, PathPicker):
            w.changed.connect(lambda *_: on_change())
        elif hasattr(w, "textChanged") and not isinstance(w, QComboBox):
            w.textChanged.connect(on_change)
        if hasattr(w, "valueChanged"):
            try:
                w.valueChanged.connect(on_change)
            except Exception:
                pass
        if hasattr(w, "stateChanged"):
            try:
                w.stateChanged.connect(on_change)
            except Exception:
                pass
        if isinstance(w, QComboBox):
            w.currentIndexChanged.connect(on_change)

    def _mark(self, key):
        w = self.widgets.get(key)
        if w is None:
            return
        cur = self.value(key)
        if cur == self._initial.get(key):
            self._dirty.discard(key)
            w.setStyleSheet("")
        else:
            self._dirty.add(key)
            # 改过的参数加一圈淡蓝描边 —— 用户一眼能看出自己动了哪几项
            w.setStyleSheet("border: 1px solid %s; border-radius: 6px;" % th.ACCENT)

    # ------------------------------------------------------------ 读写
    def value(self, key):
        w = self.widgets.get(key)
        if w is None:
            return None
        if isinstance(w, PathPicker):
            return w.value()
        if hasattr(w, "isChecked") and w.__class__.__name__ in ("QCheckBox",):
            return bool(w.isChecked())
        if isinstance(w, QComboBox):
            return w.currentData()
        if hasattr(w, "value"):
            return w.value()
        if hasattr(w, "text"):
            return w.text()
        return None

    def set_value(self, key, val):
        w = self.widgets.get(key)
        if w is None:
            return
        w.blockSignals(True)
        if isinstance(w, PathPicker):
            w.set_value(val)
        elif w.__class__.__name__ == "QCheckBox":
            w.setChecked(bool(val))
        elif isinstance(w, QComboBox):
            set_combo_value(w, val)
        elif hasattr(w, "setValue"):
            w.setValue(val)
        elif hasattr(w, "setText"):
            w.setText(str(val))
        w.blockSignals(False)

    def values(self):
        return {k: self.value(k) for k in self.widgets}

    def dirty_keys(self):
        return sorted(self._dirty)

    def mark_clean(self):
        self._dirty.clear()
        for w in self.widgets.values():
            w.setStyleSheet("")
        self._initial = {k: self.value(k) for k in self.widgets}

    def filter_keys(self, pred):
        """只显示满足条件的参数（其余隐藏）。用于「只改这一个」的场景。"""
        for key, f in self.fields.items():
            f.setVisible(bool(pred(key)))


def _looks_like_path(key):
    low = key.lower()
    return any(h in low for h in ("dir", "path", "from", "file")) and \
        not low.endswith("_files")


def _choice_label(choices, c):
    """下拉框里那一行显示什么。

    空字符串必须显示成「自动」而不是一个空白项 —— 空白项看起来像
    「这项坏了」，而它其实是「引擎自己决定」这个有意义的选项。
    """
    if c == "":
        return "（%s）" % t("common.auto")
    return str(c)


class SectionForm(QWidget):
    """带标题和「重置」按钮的一段表单。页面直接用这个包一层。"""

    changed = Signal()

    def __init__(self, section, title, values=None, only_keys=None, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACE["sm"])

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(section_label(title))
        head.addStretch(1)
        self.reset_btn = ghost_button(t("common.reset"))
        self.reset_btn.clicked.connect(self.reset)
        head.addWidget(self.reset_btn)
        lay.addLayout(head)
        lay.addWidget(hline())

        self.form = ParamForm(section, values, only_keys)
        self.form.changed.connect(self.changed.emit)
        lay.addWidget(self.form, 1)
        self._original = dict(values or {})

    def reset(self):
        for k, v in self._original.items():
            self.form.set_value(k, v)
        self.form.mark_clean()
        self.changed.emit()

    def values(self):
        return self.form.values()

    def set_values(self, vals):
        for k, v in (vals or {}).items():
            self.form.set_value(k, v)
        self.form.mark_clean()
