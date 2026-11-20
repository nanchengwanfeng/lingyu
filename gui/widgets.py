# -*- coding: utf-8 -*-
r"""可复用的界面零件。

## 为什么要有这一层

八页界面如果各写各的卡片、标题、状态点，最后一定会「每页长得都不太一样」——
这是廉价感的主要来源，比配色难看更致命。

所以这里把「一个区块长什么样」定死：`Card` 管边框和留白，`PageHeader` 管
标题和说明，`StatTile` 管数字，`Field` 管一行输入。页面只负责**往里塞内容**，
不负责长什么样。
"""

import html as _html
import os

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QWheelEvent)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QCheckBox,
                               QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QSizePolicy, QSpinBox, QVBoxLayout,
                               QWidget)

from .i18n import t
from .theme import SPACE, FONT_SIZE, th


def _measure_font():
    """用来量文字宽度的字体 —— 必须和 QSS 里给标签设的那个一致。

    见 `Field._raw_width` 的注释：在控件构造期间 `self.fontMetrics()`
    拿到的是默认字体（QSS 还没生效），会低估宽度导致标签被截断。
    """
    f = QFont()
    # 只设字号就够准了：中西文宽度差异由具体的 family 决定，
    # 而这里量的是中文标签，各家中文字体的字宽基本都是 1em。
    f.setPointSize(FONT_SIZE["body"])
    return f


# ==========================================================================
# 基础容器
# ==========================================================================
class Card(QFrame):
    """一个带边框圆角的区块。

    参数：
        title   标题（可空）
        hint    标题下面的灰色说明（可空）
        flat    True = 更淡的边框，用于卡片里再套一层

    内部布局是 `self.body`（一个 QVBoxLayout），直接往里 addWidget 就行。
    """

    def __init__(self, title=None, hint=None, flat=False, parent=None):
        super().__init__(parent)
        self.setObjectName("CardFlat" if flat else "Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE["lg"], SPACE["lg"],
                                 SPACE["lg"], SPACE["lg"])
        outer.setSpacing(SPACE["md"])

        if title:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("CardTitle")
            outer.addWidget(self.title_label)
        else:
            self.title_label = None

        if hint:
            self.hint_label = QLabel(hint)
            self.hint_label.setObjectName("CardHint")
            self.hint_label.setWordWrap(True)
            outer.addWidget(self.hint_label)
        else:
            self.hint_label = None

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(SPACE["md"])
        outer.addLayout(self.body)

    def add(self, w, stretch=0):
        self.body.addWidget(w, stretch)
        return w

    def add_layout(self, lay, stretch=0):
        self.body.addLayout(lay, stretch)
        return lay


class PageHeader(QWidget):
    """页面顶部的标题 + 一句话说明。

    每页都用同一个，这样切页面时标题的位置和字号不会跳。
    """

    def __init__(self, title, desc="", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, SPACE["sm"])
        lay.setSpacing(SPACE["xs"])
        self.title = QLabel(title)
        self.title.setObjectName("PageTitle")
        lay.addWidget(self.title)
        self.desc = QLabel(desc)
        self.desc.setObjectName("PageDesc")
        self.desc.setWordWrap(True)
        self.desc.setVisible(bool(desc))
        lay.addWidget(self.desc)

    def set_text(self, title, desc=""):
        self.title.setText(title)
        self.desc.setText(desc)
        self.desc.setVisible(bool(desc))


class Divider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFixedHeight(1)


def section_label(text):
    """小的分区标签（全大写、灰、加粗）。"""
    lbl = QLabel(text)
    lbl.setObjectName("SectionLabel")
    return lbl


def muted(text, wrap=False):
    lbl = QLabel(text)
    lbl.setObjectName("Muted")
    lbl.setWordWrap(wrap)
    return lbl


def hline():
    return Divider()


# ==========================================================================
# 数字卡片
# ==========================================================================
class StatTile(QFrame):
    """一个「大数字 + 小标题」的方块。概览页和估算面板用。

    ## 字号按**实际宽度**算，不按字数

    一开始按字符数分档（<=12 大字、<=18 中字……）。但「360.51 M」只有
    8 个字符，在概览页那种宽格子里很合适，到了训练页右栏的窄格子里
    就被挤成两行（「360.51」换行「M」）—— 同一段文字在不同宽度下
    表现完全不同，按字数判根本判不准。

    现在量真实像素宽度，挑一个放得下的字号。`resizeEvent` 里重算，
    所以窗口缩放时也会自动跟着变。
    """

    SIZES = (20, 17, 14, 12, 11)     # 从大到小依次试

    def __init__(self, label, value="—", sub="", accent=None, parent=None,
                 compact=False):
        super().__init__(parent)
        self.setObjectName("CardFlat")
        # compact=True：训练页右栏只有 400 多像素宽，四块并排时
        # 20pt 的数字会把格子撑高、把估算卡挤扁。紧凑档只缩小
        # 字号上限与留白，不动结构。
        self._sizes = (14, 12, 11) if compact else self.SIZES
        pad = SPACE["sm"] if compact else SPACE["md"]
        self._pad = pad
        lay = QVBoxLayout(self)
        lay.setContentsMargins(pad, pad, pad, pad)
        lay.setSpacing(1 if compact else 2)
        # 紧凑档要保证「数值 + 单位 + 标签」三层放得下（14pt/12pt/12pt），
        # 否则右栏被压时 Qt 会把 tile 高度压到 50px 以下，数字被切成半截。
        if compact:
            # 60 = 数值 14pt(≈25px) + 单位/标签 12pt(≈16px) + 上下留白 16 + 间距 1
            # 再高就会把「估算卡 + 看板 + 运行卡」在 1440×900 下挤出页面，
            # 结果 Qt 反过来把所有格子压到 45px、文字被切成半截。
            self.setMinimumHeight(60)

        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("StatValue")
        self.value_label.setWordWrap(True)
        lay.addWidget(self.value_label)

        # 顺序：数字 → 单位 → 标题。单位贴着数字读最顺
        # （「46955 / tok/s / 速度」），标题放在最后一行。
        self.sub_label = QLabel(sub)
        self.sub_label.setObjectName("Muted")
        self.sub_label.setWordWrap(True)
        self.sub_label.setVisible(bool(sub))
        lay.addWidget(self.sub_label)

        self.label_label = QLabel(label)
        self.label_label.setObjectName("StatLabel")
        lay.addWidget(self.label_label)

        self._accent = accent
        self._fit_font(str(value))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_font(self.value_label.text())

    def _fit_font(self, text):
        """挑一个在当前宽度下放得下的最大字号。"""
        text = text or ""
        pad = getattr(self, "_pad", SPACE["md"])
        avail = max(30, self.width() - 2 * pad)
        sizes = getattr(self, "_sizes", self.SIZES)
        size = sizes[-1]
        for s in sizes:
            f = QFont(self.value_label.font())
            f.setPointSize(s)
            if QFontMetrics(f).horizontalAdvance(text) <= avail:
                size = s
                break
        self.value_label.setStyleSheet(
            "font-size:%dpt; font-weight:600;" % size
            + ("color:%s;" % self._accent if self._accent else ""))

    def set_value_color(self, color):
        self._accent = color
        self._fit_font(self.value_label.text())

    def set(self, value, sub=None, accent=None):
        self.value_label.setText(str(value))
        if sub is not None:
            self.sub_label.setText(sub)
            self.sub_label.setVisible(bool(sub))
        if accent:
            self._accent = accent
        self._fit_font(str(value))

    def set_label(self, label):
        """改标题（预训练 / SFT 的验证 loss 口径不同，监控页要切换）。"""
        self.label_label.setText(str(label))


# ==========================================================================
# 状态点
# ==========================================================================
class StatusDot(QWidget):
    """一个小圆点，用来表示「就绪 / 运行中 / 出错」。

    比一整行文字省地方，而且在侧边栏、状态栏、日志行里都能复用同一个视觉语言。
    """

    COLORS = {"idle": "text_muted", "ok": "success",
              "run": "accent", "warn": "warning", "error": "danger"}

    def __init__(self, state="idle", parent=None):
        super().__init__(parent)
        self._state = state
        self._pulse = 0.0
        self.setFixedSize(10, 10)

    def set_state(self, state):
        if state != self._state:
            self._state = state
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor(th[self.COLORS.get(self._state, "text_muted")])
        if self._state == "run":
            # 运行中的点外圈淡一点，像在「呼吸」，比静止的圆点更容易注意到
            halo = QColor(color)
            halo.setAlpha(70)
            p.setBrush(halo)
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, 10, 10)
            p.setBrush(color)
            p.drawEllipse(2, 2, 6, 6)
        else:
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawEllipse(1, 1, 8, 8)


# ==========================================================================
# 日志输出
# ==========================================================================
class LogView(QPlainTextEdit):
    """只读的日志框：等宽字体、深色底、自动滚到底。

    带一层**简单的着色**：错误红、警告黄、成功绿、标题加粗。
    日志如果全是一个颜色，找关键信息要靠肉眼一行行扫，很难用。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LogView")
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)      # 防止长时间训练把内存吃光
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._autoscroll = True
        f = QFont()
        f.setStyleHint(QFont.Monospace)
        self.setFont(f)

    def set_autoscroll(self, on):
        self._autoscroll = bool(on)

    def append_line(self, text, kind=None):
        """追加一行。kind 决定颜色：error / warn / ok / title / None。"""
        color = {
            "error": th.DANGER, "warn": th.WARNING, "ok": th.SUCCESS,
            "info": th.INFO, "title": th.TEXT, "dim": th.TEXT_MUTED,
        }.get(kind)
        safe = (text.replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;").replace(" ", "&nbsp;"))
        if color:
            weight = "600" if kind == "title" else "400"
            safe = ('<span style="color:%s;font-weight:%s">%s</span>'
                    % (color, weight, safe))
        self.appendHtml(safe)
        if self._autoscroll:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum())


# ==========================================================================
# 文件/目录选择行
# ==========================================================================
class PathPicker(QWidget):
    """一个输入框 + 一个「浏览…」按钮。目录和文件共用。

    空的时候要给一句占位提示 —— 否则一个空的输入框在深色卡片上
    几乎看不见边框，会让人以为「这里没有输入框」。
    """

    changed = Signal(str)

    def __init__(self, value="", is_dir=True, placeholder="", parent=None):
        super().__init__(parent)
        self.is_dir = is_dir
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACE["sm"])
        self.edit = QLineEdit(value)
        self.edit.setPlaceholderText(
            placeholder or ("选择文件夹…" if is_dir else "选择文件…"))
        self.edit.textChanged.connect(self.changed.emit)
        lay.addWidget(self.edit, 1)
        self.btn = QPushButton(t("common.browse"))
        self.btn.setFixedWidth(80)
        self.btn.clicked.connect(self._browse)
        lay.addWidget(self.btn)

    def _browse(self):
        start = self.edit.text() or os.path.expanduser("~")
        if self.is_dir:
            path = QFileDialog.getExistingDirectory(self, t("common.open"), start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, t("common.open"), start)
        if path:
            self.edit.setText(os.path.normpath(path))

    def value(self):
        return self.edit.text().strip()

    def set_value(self, v):
        self.edit.setText(v or "")


# ==========================================================================
# 表单行
# ==========================================================================
class HelpBadge(QLabel):
    """参数名后面的灰色半透明「?」徽标（审计 1.5）。

    UI 文字要短：能写「n_layer」就不写「Transformer 堆叠层数」。
    完整解释放 tooltip —— 需要时悬停一下，不需要时它只占 15px。

    用 `QLabel` 而不是 `QToolButton`：不需要点击，也没有按下态，
    样式可以完全交给 QSS（`#HelpBadge`）。tooltip 用富文本装进 320px
    宽的 div —— Qt 对纯文本 tooltip 不主动换行，长说明会变成一条
    横穿屏幕的细线。
    """

    def __init__(self, text, parent=None):
        super().__init__("?", parent)
        self.setObjectName("HelpBadge")
        self.setFixedSize(16, 16)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.WhatsThisCursor)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setToolTip(self._tooltip_html(text))

    @staticmethod
    def _tooltip_html(text):
        body = _html.escape(str(text or "")).replace("\n", "<br>")
        return ('<div style="max-width:320px; white-space:normal;">'
                '%s</div>' % body)

    def show_help(self):
        """显式弹出说明。

        ★ 用户实测「鼠标悬浮问号不出说明」：Qt 自带的 hover tooltip 在
        窗口刚激活/控件无焦点/被其他弹窗打断后可能不出现。这里自己接管：
        鼠标进入直接 QToolTip.showText，离开就收掉，不再赌 Qt 的计时器。
        """
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QToolTip
        QToolTip.showText(self.mapToGlobal(QPoint(18, 18)),
                          self.toolTip(), self)

    def enterEvent(self, ev):
        self.show_help()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        from PySide6.QtWidgets import QToolTip
        QToolTip.hideText()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        self.show_help()
        super().mousePressEvent(ev)


class Field(QWidget):
    """一行「标签 + 控件 + 说明」。

    两种排法：

        top=False（默认，内联）
            [   标签 ] [ 控件                    ]
                       说明文字…

        top=True（上下）
            [ 标签 ]
            [ 控件                    ]
            说明文字…

    ## 标签宽度必须**按文字量算**，不能写死

    踩过的坑：标签宽度写死 132px，于是「一个 token 最多几个字」被从左边
    截成「token 最多几个字」，而「字对最少出现几次才合并」直接截掉一半 ——
    用户看到的是一堆残缺的标签，很像界面坏了。

    现在用 QFontMetrics 量出实际需要的宽度，取本组里最宽的那个，
    这样同一组表单的标签列宽度一致（整齐），又不会被截断。

    ## 什么时候用上下排法

    中文标签普遍比英文长（「字对最少出现几次才合并」12 个字 ≈ 160px）。
    在窄列里内联排会挤压输入框，这时就该用上下排法。
    判断标准很简单：**标签超过 7 个字就用上下排**。
    """

    #: 内联排法时，标签列的宽度上下限
    LABEL_MIN = 96
    LABEL_MAX = 168

    def __init__(self, label, widget, hint="", parent=None, top=None,
                 label_w=None, help_text=""):
        super().__init__(parent)
        self._label_text = label
        self.widget = widget
        self._hint_text = hint
        # help_text：放进「?」徽标的 tooltip（审计 1.5）。
        # 与 hint 的区别：hint 是**常驻**在输入框下面的一行字，
        # help_text 默认收起、悬停才出现 —— 参数多的表单靠它保持简洁。
        self._help_text = help_text or ""
        self._raw = self._raw_width(label)

        # 排法：显式给的优先；否则按**需要的像素宽度**判断 ——
        # 不要按字数判断。`block_size` 有 10 个字符但很窄，按字数判会把它
        # 单独排成上下式，同一张表单里就它一个不一样高，看着像没对齐。
        if top is None:
            self._top = self._raw > self.LABEL_MAX
        else:
            self._top = bool(top)
        self._w = max(self.LABEL_MIN, min(self.LABEL_MAX, self._raw))

        self._build()

    # ------------------------------------------------------------ 构建
    def _build(self):
        """（重新）搭出这一行。

        允许重建是因为「统一标签宽度」可能要把整组从内联切成上下式，
        而这两种排法的控件树结构不同，只改宽度是不够的。
        """
        old = self.layout()
        if old is not None:
            while old.count():
                item = old.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
        outer = old or QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)

        self.label = QLabel(self._label_text)
        self.label.setStyleSheet("color: %s;" % th.TEXT_DIM)
        # 「?」徽标：只有给了 help_text 才出现（1.5）。
        # 放在标签后面而不是控件后面 —— 读者先看名字，再看解释。
        self.badge = (HelpBadge(self._help_text) if self._help_text else None)

        if self._top:
            self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.setSpacing(SPACE["xs"] if "xs" in SPACE else 4)
            head.addWidget(self.label)
            if self.badge is not None:
                head.addWidget(self.badge, 0, Qt.AlignTop)
            head.addStretch(1)
            outer.addLayout(head)
            outer.addWidget(self.widget)
            indent = 0
        else:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACE["md"])
            self.label.setFixedWidth(int(self._w))
            self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(self.label)
            if self.badge is not None:
                row.addWidget(self.badge, 0, Qt.AlignVCenter)
            row.addWidget(self.widget, 1)
            outer.addLayout(row)
            indent = int(self._w) + SPACE["md"]

        self.hint = QLabel(self._hint_text)
        self.hint.setObjectName("Muted")
        self.hint.setWordWrap(True)
        self.hint.setContentsMargins(indent, 0, 0, 0)
        self.hint.setVisible(bool(self._hint_text))
        outer.addWidget(self.hint)

    # ------------------------------------------------------------ 度量
    def _raw_width(self, text):
        """标签文字的实际像素宽度（**不设上限**，用来判断该用哪种排法）。

        ★ 这里必须**自己造一个字体来量**，不能用 `self.fontMetrics()`。
          原因：QSS 里的 `font-size: 12pt` 是控件被 polish（第一次显示）时
          才生效的，而 Field 是在构造期间量的 —— 那时候量到的是默认字体，
          比实际字体小。结果就是标签宽度算少了，文字被从左边截掉：
          「从哪个预训练断点开始」显示成「个预训练断点开始」。

          所以按 theme 里的字号建一个 QFont 来量，再乘一个安全系数
          （不同字体/DPI 下 horizontalAdvance 会略有出入）。
        """
        if not text:
            return 0
        fm = QFontMetrics(_measure_font())
        return int(fm.horizontalAdvance(text) * 1.12) + 14

    def measure_label(self):
        """本控件标签需要的宽度（给「整组统一宽度」用）。"""
        return self._raw

    def apply_label_width(self, width, top=None):
        """统一宽度。必要时重建布局。"""
        new_top = (width > self.LABEL_MAX) if top is None else bool(top)
        new_w = max(self.LABEL_MIN, min(self.LABEL_MAX, width))
        if abs(new_w - self._w) < 1 and new_top == self._top:
            return
        self._w, self._top = new_w, new_top
        self._build()


def unify_labels(fields):
    """让一组 Field 的标签列**等宽**。

    为什么必须统一：每个 Field 各按自己的文字量算宽度，会导致控件那一列的
    左边缘参差不齐 —— 单看每个都对，整片看就是「没对齐」，
    这是界面显得廉价最直接的原因。

    取这一组里最宽的那个当统一宽度；如果连最宽的都超过上限，
    说明这组标签太长（多半是中文说明式标签），就整组改成上下排法。
    """
    if not fields:
        return None
    widest = max(f.measure_label() for f in fields)
    for f in fields:
        f.apply_label_width(widest)
    return widest


def labeled(label, widget, hint="", top=None):
    return Field(label, widget, hint, top=top)


class _WheelGuard(QObject):
    """滚轮守卫：输入框没被点中时，滚轮应该滚页面，而不是改数值。

    ## 为什么需要

    Qt 的默认行为是「滚轮落在谁身上谁处理」。训练页参数有几十个输入框，
    用户想往下翻页，鼠标恰好停在某个数字框上 —— 滚轮直接把学习率/层数/步数
    改了，而且看不出是什么时候改的。这是真实的误操作来源，不是理论问题。

    ## 规则

    - 控件**有焦点**（用户点进去过）→ 交给控件自己处理，滚轮改数值是预期行为；
    - 控件**没焦点** → 吞掉事件，把它转投给最近的外层 `QAbstractScrollArea`，
      于是「滚轮滚页面」始终成立。

    这样既保住了「点进输入框后可以滚轮微调」，又不会让路过的滚轮改参数。
    """

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Wheel:
            return False
        try:
            if obj.hasFocus():
                return False
        except Exception:
            return False
        self._to_scroll_area(obj, event)
        return True

    @staticmethod
    def _to_scroll_area(widget, event):
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                forward = QWheelEvent(
                    event.position(), event.globalPosition(),
                    event.pixelDelta(), event.angleDelta(),
                    event.buttons(), event.modifiers(),
                    event.phase(), event.inverted())
                QApplication.sendEvent(parent.viewport(), forward)
                return
            parent = parent.parentWidget()


def guard_wheel(widget):
    """给输入控件装上滚轮守卫（返回原控件，方便链式返回）。"""
    guard = _WheelGuard(widget)
    widget.installEventFilter(guard)
    widget._wheel_guard = guard
    # WheelFocus 会在滚轮经过时抢焦点 —— 那样守卫永远等不到「没焦点」的状态。
    widget.setFocusPolicy(Qt.StrongFocus)
    return widget


def make_int(minimum=-10 ** 9, maximum=10 ** 9, value=0, step=1):
    w = QSpinBox()
    w.setRange(int(minimum), int(maximum))
    w.setValue(int(value))
    w.setSingleStep(int(step))
    w.setButtonSymbols(QSpinBox.UpDownArrows)
    w.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return guard_wheel(w)


def make_float(minimum=-1e12, maximum=1e12, value=0.0, step=0.1, decimals=4):
    """小数输入框。

    ⚠️ **顺序很重要：必须先 setDecimals 再 setValue。**

    QDoubleSpinBox 的 decimals 默认是 2。如果先 setValue(0.0005)，
    它会立刻把值四舍五入成 0.00，再被 minimum 顶到下限 ——
    于是学习率在界面上显示成 0.000001（也就是最小值），
    而用户以为自己看到的是配置里那个数。

    实测就是靠这个发现 SFT 页的学习率显示成了 0.000001。
    """
    w = QDoubleSpinBox()
    w.setDecimals(int(decimals))          # ★ 必须在 setValue 之前
    w.setRange(float(minimum), float(maximum))
    w.setValue(float(value))
    w.setSingleStep(step)
    w.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return guard_wheel(w)


def make_combo(items, value=None):
    """items 是 [(显示文字, 实际值), …]。"""
    w = QComboBox()
    for label, val in items:
        w.addItem(label, val)
    if value is not None:
        set_combo_value(w, value)
    return guard_wheel(w)


def set_combo_value(combo, value):
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)
    else:
        # 配置里的值不在候选项里（比如插件注册的组件名）——
        # 临时加一项，否则用户会看到下拉框莫名其妙跳到第一项
        combo.addItem(str(value), value)
        combo.setCurrentIndex(combo.count() - 1)


def make_check(text, checked=False):
    w = QCheckBox(text)
    w.setChecked(bool(checked))
    return w


def primary_button(text, icon=None):
    b = QPushButton(text)
    b.setObjectName("Primary")
    b.setMinimumHeight(32)
    b.setCursor(Qt.PointingHandCursor)
    return b


def ghost_button(text):
    b = QPushButton(text)
    b.setObjectName("Ghost")
    b.setCursor(Qt.PointingHandCursor)
    return b


def danger_button(text):
    b = QPushButton(text)
    b.setObjectName("Danger")
    b.setCursor(Qt.PointingHandCursor)
    return b


def chip_button(text, checkable=True):
    b = QPushButton(text)
    b.setObjectName("Chip")
    b.setCheckable(checkable)
    b.setCursor(Qt.PointingHandCursor)
    return b


# ==========================================================================
# 键值展示
# ==========================================================================
class KeyValue(QWidget):
    """竖排的「键：值」列表，键灰色、值正常色。适合展示一堆只读信息。

    键那一列固定宽度是为了让所有值的左边缘对齐（不对齐看着就乱）。
    宽度定 132 是量出来的：最长的键是「KV cache / token」，
    再窄就会把它截断成「KV cache / toke」——实测踩过。
    """

    KEY_W = 132

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(SPACE["xs"] + 2)
        self._rows = {}

    def add(self, key, value="—"):
        row = QWidget()
        # 每个键值行给一个最小高度，防止被上层布局压扁。
        # 踩过的坑：训练页右栏的估算卡片在窗口不够高时被压，
        # 四行文字挤到一起互相重叠，看着像渲染坏了。
        row.setMinimumHeight(21)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(SPACE["sm"])
        k = QLabel(key)
        k.setObjectName("Muted")
        k.setFixedWidth(self.KEY_W)
        v = QLabel(str(value))
        v.setWordWrap(True)
        v.setTextInteractionFlags(Qt.TextSelectableByMouse)
        h.addWidget(k)
        h.addWidget(v, 1)
        self._lay.addWidget(row)
        self._rows[key] = v
        return v

    def set(self, key, value, color=None):
        v = self._rows.get(key)
        if v is None:
            v = self.add(key)
        v.setText(str(value))
        v.setStyleSheet("color: %s;" % color if color else "")

    def clear(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._rows.clear()


def spacer():
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return w


class TopBar(QFrame):
    """窗口顶部的一条「窄导航」：面包屑 + 当前任务 + 主题/语言。

    为什么单独做一条：侧边栏只放「去哪」，这条放「我在哪、正在发生什么」。
    高度固定 44px，不抢内容区空间；窄屏下也不会换行。
    """

    theme_toggled = Signal()
    lang_toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(44)

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACE["xl"], 0, SPACE["lg"], 0)
        row.setSpacing(SPACE["md"])

        self.crumb = QLabel("")
        self.crumb.setObjectName("TopCrumb")
        row.addWidget(self.crumb)
        row.addStretch(1)

        self.dot = StatusDot("idle")
        row.addWidget(self.dot)
        self.task = QLabel(t("sb.idle"))
        self.task.setObjectName("TopTask")
        row.addWidget(self.task)

        sep = QFrame()
        sep.setObjectName("Divider")
        sep.setFixedWidth(1)
        sep.setFixedHeight(18)
        row.addWidget(sep)

        self.theme_btn = QPushButton("\u25d1")
        self.theme_btn.setObjectName("TopIcon")
        self.theme_btn.setToolTip(t("st.theme"))
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.clicked.connect(self.theme_toggled)
        row.addWidget(self.theme_btn)

        self.lang_btn = QPushButton("\u4e2d")
        self.lang_btn.setObjectName("TopIcon")
        self.lang_btn.setToolTip(t("st.lang"))
        self.lang_btn.setCursor(Qt.PointingHandCursor)
        self.lang_btn.clicked.connect(self.lang_toggled)
        row.addWidget(self.lang_btn)

    # ------------------------------------------------------------------
    def set_page(self, section, title):
        if not section or section == title:
            self.crumb.setText(title or "")
        else:
            self.crumb.setText("%s  /  %s" % (section, title))

    def set_state(self, state, text=""):
        self.dot.set_state(state)
        self.task.setText(text or t("sb.idle"))

