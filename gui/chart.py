# -*- coding: utf-8 -*-
r"""loss 曲线图 —— 用 QPainter 手画，不用 matplotlib。

## 为什么不用 matplotlib

1. **配色对不上**。matplotlib 有自己的主题系统，想让它和界面配色完全一致
   得折腾半天，还经常在深色主题下露出白底。
2. **慢**。把 FigureCanvas 塞进 Qt 里，每次更新都要重绘整张图，
   训练时每几秒更新一次会明显卡。
3. **依赖重**。GUI 本来能在不装 matplotlib 的机器上跑起来，
   为了一张曲线图多背一个几十 MB 的依赖不划算。

手画一个这样的折线图只要一百多行，而且能做到：
渐变填充、双曲线、自适应坐标、高 DPI 清晰、跟随主题换色。

## 画了什么

- 训练 loss（主色实线）+ 验证 loss（成功色，只在实际有验证点时画点）
- 自适应 Y 轴（按数据范围取整，不是死板的 0~max）
- 淡淡的横向网格 + Y 轴刻度
- X 轴按步数显示，只在首尾和中间标几个刻度，避免挤成一团
- 右下角显示最新值，眼睛不用追着线跑
"""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPolygonF)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .theme import FONT_MONO, FONT_SIZE, th


def _nice_ticks(lo, hi, n=4):
    """把 [lo, hi] 切成 n 段「好看」的刻度（1/2/5 × 10^k）。

    直接等分会出现 3.7142 这种刻度值，很丑也不好读。
    """
    if not (hi > lo):
        hi = lo + 1.0
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(v)
        v += step
    return ticks, step


class LossChart(QWidget):
    """训练曲线。`set_data(steps, losses, val_steps, val_losses)` 更新。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps, self.losses = [], []
        self.vsteps, self.vlosses = [], []
        self.title = ""
        self.empty_text = ""
        # 坐标/角标格式可配置：分析页用它画学习率、速度（同一个控件，三种单位）
        self.y_fmt = "%.3f"        # Y 轴刻度
        self.badge_name = "loss"   # 右下角角标前缀；置空则只显示数值
        self.badge_fmt = "%.4f"
        self.val_badge_fmt = "%.4f"
        self.show_fill = True      # 是否画渐变填充
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ------------------------------------------------------------ 数据
    def set_data(self, steps, losses, vsteps=None, vlosses=None):
        self.steps = list(steps)
        self.losses = list(losses)
        self.vsteps = list(vsteps or [])
        self.vlosses = list(vlosses or [])
        self.update()

    def append(self, step, loss, val_loss=None):
        self.steps.append(step)
        self.losses.append(loss)
        if val_loss is not None and val_loss == val_loss:      # 排除 NaN
            self.vsteps.append(step)
            self.vlosses.append(val_loss)
        self.update()

    def clear(self):
        self.steps, self.losses = [], []
        self.vsteps, self.vlosses = [], []
        self.update()

    def has_data(self):
        return len(self.losses) >= 1

    # ------------------------------------------------------------ 绘制
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        W, H = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 52, 16, 16, 28
        plot = QRectF(pad_l, pad_t, max(1, W - pad_l - pad_r),
                      max(1, H - pad_t - pad_b))

        # 背景 + 圆角
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(th.SURFACE))
        p.drawRoundedRect(QRectF(0, 0, W, H), 8, 8)

        if not self.losses:
            p.setPen(QColor(th.TEXT_MUTED))
            f = QFont(FONT_MONO.split(",")[0].strip('"'))
            f.setPointSize(FONT_SIZE["small"])
            p.setFont(f)
            p.drawText(QRectF(0, 0, W, H), Qt.AlignCenter,
                       self.empty_text or "—")
            return

        xs = self.steps or list(range(len(self.losses)))
        ys = list(self.losses)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 == x0:
            x1 = x0 + 1
        # Y 轴上下各留一点余量，曲线不贴边。
        # 0.15 是量出来的：留太小时第一个点会正好落在顶部刻度线上被切掉半个。
        span = max(1e-6, y1 - y0)
        y0 -= span * 0.15
        y1 += span * 0.15

        ticks, _ = _nice_ticks(y0, y1, 4)
        if ticks:
            y0, y1 = min(y0, ticks[0]), max(y1, ticks[-1])

        def px(x):
            return plot.left() + (x - x0) / (x1 - x0) * plot.width()

        def py(y):
            return plot.bottom() - (y - y0) / max(1e-9, y1 - y0) * plot.height()

        # ---- 网格 + Y 刻度 ----
        f = QFont(FONT_MONO.split(",")[0].strip('"'))
        f.setPointSize(FONT_SIZE["tiny"])
        p.setFont(f)
        p.setPen(QPen(QColor(th.CHART_GRID), 1))
        for tv in ticks:
            yy = py(tv)
            if yy < plot.top() - 1 or yy > plot.bottom() + 1:
                continue
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(QColor(th.TEXT_MUTED))
            p.drawText(QRectF(2, yy - 9, pad_l - 8, 18),
                       Qt.AlignRight | Qt.AlignVCenter, self.y_fmt % tv)
            p.setPen(QPen(QColor(th.CHART_GRID), 1))

        # ---- 坐标轴 ----
        p.setPen(QPen(QColor(th.CHART_AXIS), 1))
        p.drawLine(QPointF(plot.left(), plot.top()),
                   QPointF(plot.left(), plot.bottom()))
        p.drawLine(QPointF(plot.left(), plot.bottom()),
                   QPointF(plot.right(), plot.bottom()))

        # ---- X 刻度（最多 5 个，避免挤成一团）----
        n_tick = min(5, max(2, int(plot.width() // 90)))
        p.setPen(QColor(th.TEXT_MUTED))
        for i in range(n_tick + 1):
            vx = x0 + (x1 - x0) * i / n_tick
            xx = px(vx)
            if i and i != n_tick:
                p.setPen(QPen(QColor(th.CHART_GRID), 1))
                p.drawLine(QPointF(xx, plot.top()), QPointF(xx, plot.bottom()))
            p.setPen(QColor(th.TEXT_MUTED))
            label = "%d" % vx if vx >= 100 else "%.0f" % vx
            p.drawText(QRectF(xx - 30, plot.bottom() + 4, 60, 16),
                       Qt.AlignCenter, label)

        # ---- loss 曲线（带渐变填充）----
        poly = QPolygonF([QPointF(px(x), py(y)) for x, y in zip(xs, ys)])
        fill = QPainterPath()
        fill.moveTo(QPointF(px(xs[0]), plot.bottom()))
        for pt in poly:
            fill.lineTo(pt)
        fill.lineTo(QPointF(px(xs[-1]), plot.bottom()))
        fill.closeSubpath()

        if self.show_fill:
            grad = QLinearGradient(0, plot.top(), 0, plot.bottom())
            a = QColor(th.CHART_LINE)
            a.setAlpha(90 if th.is_dark else 70)
            b = QColor(th.CHART_LINE)
            b.setAlpha(0)
            grad.setColorAt(0.0, a)
            grad.setColorAt(1.0, b)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawPath(fill)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(th.CHART_LINE), 2))
        p.drawPolyline(poly)

        # ---- 验证 loss：只画点，不连线 ----
        # 验证是每隔几百步才做一次，连成线会让人误以为中间也有数据
        if self.vlosses:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(th.CHART_VAL))
            for x, y in zip(self.vsteps, self.vlosses):
                if x < x0 or x > x1:
                    continue
                p.drawEllipse(QPointF(px(x), py(y)), 3.2, 3.2)

        # ---- 最新值角标 ----
        last = ys[-1]
        p.setFont(f)
        label = ((self.badge_name + " ") if self.badge_name else "") \
            + (self.badge_fmt % last)
        if self.vlosses:
            label += "   val " + (self.val_badge_fmt % self.vlosses[-1])
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + 16
        box = QRectF(plot.right() - tw, plot.top() + 4, tw, 20)
        p.setPen(Qt.NoPen)
        bg = QColor(th.SURFACE_2)
        bg.setAlpha(235)
        p.setBrush(bg)
        p.drawRoundedRect(box, 5, 5)
        p.setPen(QColor(th.TEXT))
        p.drawText(box, Qt.AlignCenter, label)


class Sparkline(QWidget):
    """迷你曲线，塞在卡片里当装饰/趋势提示。没有坐标轴。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_values(self, vs):
        self.values = list(vs)
        self.update()

    def paintEvent(self, _e):
        if len(self.values) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        lo, hi = min(self.values), max(self.values)
        if hi == lo:
            hi = lo + 1
        path = QPolygonF([
            QPointF(i / (len(self.values) - 1) * W,
                    H - 3 - (v - lo) / (hi - lo) * (H - 6))
            for i, v in enumerate(self.values)])
        p.setPen(QPen(QColor(th.CHART_LINE), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(path)
