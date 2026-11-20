# -*- coding: utf-8 -*-
"""分析页：回看跑过的训练。

监控页回答「现在跑到哪了」，分析页回答「上一次/那几次跑得怎么样」。

数据全部来自输出目录，不需要引擎额外配合：

    out/<run>/history.csv   每 eval_interval 一行：step,train_loss,val_loss,lr,grad_norm,elapsed_sec
    out/<run>/run.json      训练开始时写下的元信息（kind / tokens_per_step / n_params …）

`history.csv` 是数据契约，监控页和这里共用。没有 CSV 的目录（例如手动放进来的
断点目录）不会出现在列表里 —— 分析页不读断点文件，只读曲线。
"""

import csv

from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QScrollArea,
                               QVBoxLayout, QWidget)

from ..chart import LossChart
from ..i18n import t
from ..theme import SPACE, th
from ..widgets import Card, StatTile, ghost_button, make_combo, muted
from . import BasePage


# ---------------------------------------------------------------- 纯函数（可单测）
def load_history(path):
    """读 history.csv，返回 [{"step","loss","val","lr","gnorm","elapsed"}, …]。

    坏行直接跳过：CSV 是训练中途写的，正在写的最后一行可能不完整。
    """
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as fp:
            for raw in csv.DictReader(fp):
                try:
                    step = int(float(raw["step"]))
                    loss = float(raw["train_loss"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append({
                    "step": step,
                    "loss": loss,
                    "val": _f(raw.get("val_loss")),
                    "lr": _f(raw.get("lr")),
                    "gnorm": _f(raw.get("grad_norm")),
                    "elapsed": _f(raw.get("elapsed_sec")),
                })
    except (OSError, UnicodeDecodeError):
        return []
    return rows


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def speed_series(rows, tokens_per_step):
    """由相邻两行的 elapsed 差算瞬时速度（token/s）。

    训练是「隔 eval_interval 步写一行」，所以速度 = 行间 token 差 / 时间差。
    第一行没有前一行可减，从第二行开始。
    """
    out, prev = [], None
    tps = max(1, int(tokens_per_step or 0))
    for r in rows:
        e = r.get("elapsed")
        if e is None:
            prev = None
            continue
        if prev is not None and e > prev[1] + 1e-9:
            dt = e - prev[1]
            dstep = max(1, r["step"] - prev[0])
            out.append((r["step"], dstep * tps / dt))
        prev = (r["step"], e)
    return out


def compute_kpis(rows, meta=None):
    """顶部 KPI：总步数 / 最好验证 loss（及步数）/ 末段 loss / 平均速度 / 耗时 / 总 token。"""
    meta = meta or {}
    if not rows:
        return {}
    steps = rows[-1]["step"]
    best, best_step = None, None
    for r in rows:
        if r["val"] is not None and (best is None or r["val"] < best):
            best, best_step = r["val"], r["step"]
    tail = [r["loss"] for r in rows[-20:]]
    elapsed = rows[-1].get("elapsed")
    tps = int(meta.get("tokens_per_step") or 0)
    total_tokens = tps * steps
    speed = (total_tokens / elapsed) if (elapsed and elapsed > 0 and tps) else None
    return {
        "steps": steps,
        "best": best, "best_step": best_step,
        "last": (sum(tail) / len(tail)) if tail else None,
        "speed": speed,
        "elapsed": elapsed,
        "tokens": total_tokens if tps else None,
        "kind": meta.get("kind") or "",
    }


def _fmt_dur(sec):
    try:
        s = int(round(float(sec)))
    except (TypeError, ValueError):
        return "—"
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


class AnalysisPage(BasePage):
    nav_key = "nav.analysis"
    group_key = "nav.group.data"
    icon = "∿"

    def header(self):
        return (t("an.title"), t("an.desc"))

    def build(self):
        self._runs = []
        self._rows = []
        self._meta = {}

        # ---------------- 选一次训练 ----------------
        self.run_card = Card(t("an.run"))
        bar = QHBoxLayout()
        bar.setSpacing(SPACE["md"])
        self.run_combo = make_combo([], None)
        self.run_combo.currentIndexChanged.connect(
            lambda _i: self._select(self.run_combo.currentData()))
        bar.addWidget(self.run_combo, 1)
        self.refresh_btn = ghost_button(t("an.refresh"))
        self.refresh_btn.clicked.connect(self._refresh_runs)
        bar.addWidget(self.refresh_btn)
        self.export_btn = ghost_button(t("an.export"))
        self.export_btn.setToolTip(t("an.export.hint"))
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.export_btn)
        self.run_card.add_layout(bar)
        self.meta_label = muted("")
        self.run_card.add(self.meta_label)
        self.add(self.run_card)

        # ---------------- KPI ----------------
        row = QHBoxLayout()
        row.setSpacing(SPACE["md"])
        self.k_steps = StatTile(t("an.kpi.steps"), "—", accent=th.ACCENT)
        self.k_best = StatTile(t("an.kpi.best"), "—", accent=th.SUCCESS)
        self.k_last = StatTile(t("an.kpi.last"), "—")
        self.k_speed = StatTile(t("an.kpi.speed"), "—")
        self.k_time = StatTile(t("an.kpi.time"), "—")
        self.k_tokens = StatTile(t("an.kpi.tokens"), "—")
        for w in (self.k_steps, self.k_best, self.k_last,
                  self.k_speed, self.k_time, self.k_tokens):
            row.addWidget(w, 1)
        self.add(self._wrap(row))

        # ---------------- 三张图 ----------------
        charts = QWidget()
        clay = QVBoxLayout(charts)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(SPACE["lg"])

        self.chart_loss = LossChart()
        self.chart_loss.setMinimumHeight(170)
        self.chart_loss.title = t("an.chart.loss")
        clay.addWidget(self._in_card(self.chart_loss, t("an.chart.loss")))

        self.chart_lr = LossChart()
        self.chart_lr.setMinimumHeight(170)
        self.chart_lr.title = t("an.chart.lr")
        self.chart_lr.badge_name = "lr"
        self.chart_lr.badge_fmt = "%.2e"
        self.chart_lr.y_fmt = "%.0e"
        self.chart_lr.show_fill = False
        clay.addWidget(self._in_card(self.chart_lr, t("an.chart.lr")))

        self.chart_speed = LossChart()
        self.chart_speed.setMinimumHeight(170)
        self.chart_speed.title = t("an.chart.speed")
        self.chart_speed.badge_name = ""
        self.chart_speed.badge_fmt = "%.0f"
        self.chart_speed.y_fmt = "%.0f"
        self.chart_speed.show_fill = False
        clay.addWidget(self._in_card(self.chart_speed, t("an.chart.speed")))

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setWidget(charts)
        self.add(sa, 1)
        self._empty = muted(t("an.empty"))
        self.add(self._empty)

    def _in_card(self, widget, title):
        card = Card(title)
        card.add(widget, 1)
        return card

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    # ------------------------------------------------------------ 数据
    def on_show(self):
        self._refresh_runs()

    def _refresh_runs(self):
        try:
            runs = self.ctx.runs()
        except Exception:
            runs = []
        self._runs = runs
        current = self.run_combo.currentData()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for i, run in enumerate(runs):
            kind = t("an.kind.sft") if run["kind"] == "sft" else t("an.kind.pretrain")
            self.run_combo.addItem("%s · %s" % (run["name"], kind), i)
        self.run_combo.blockSignals(False)
        self._empty.setVisible(not runs)
        self.export_btn.setEnabled(bool(runs))
        if not runs:
            self.meta_label.setText("")
            self._clear()
            return
        idx = current if isinstance(current, int) and 0 <= current < len(runs) else 0
        self.run_combo.setCurrentIndex(idx)
        self._select(idx)

    def _select(self, idx):
        if not isinstance(idx, int) or not (0 <= idx < len(self._runs)):
            return
        run = self._runs[idx]
        self._meta = run.get("meta") or {}
        self._rows = load_history(run["history"])
        kpi = compute_kpis(self._rows, self._meta)
        if not self._rows:
            self.meta_label.setText(t("an.empty"))
            self._clear()
            return
        meta_bits = [run["name"]]
        if self._meta.get("started"):
            meta_bits.append(str(self._meta["started"]))
        if self._meta.get("device"):
            meta_bits.append(str(self._meta["device"]))
        self.meta_label.setText(" · ".join(meta_bits))

        self.k_steps.set(_fmt_int(kpi.get("steps")))
        self.k_best.set(_fmt_loss(kpi.get("best")),
                        sub=(t("an.best_at").format(step=_fmt_int(kpi.get("best_step")))
                             if kpi.get("best_step") else ""))
        self.k_last.set(_fmt_loss(kpi.get("last")))
        self.k_speed.set(_fmt_int(kpi.get("speed")),
                         sub=t("an.unit.tok_s") if kpi.get("speed") else "")
        self.k_time.set(_fmt_dur(kpi.get("elapsed")))
        self.k_tokens.set(_fmt_int(kpi.get("tokens")))

        steps = [r["step"] for r in self._rows]
        self.chart_loss.set_data(
            steps, [r["loss"] for r in self._rows],
            [r["step"] for r in self._rows if r["val"] is not None],
            [r["val"] for r in self._rows if r["val"] is not None])
        self.chart_lr.set_data(
            [r["step"] for r in self._rows if r["lr"] is not None],
            [r["lr"] for r in self._rows if r["lr"] is not None])
        speed = speed_series(self._rows, self._meta.get("tokens_per_step"))
        self.chart_speed.set_data([s for s, _v in speed], [v for _s, v in speed])
        for chart in (self.chart_loss, self.chart_lr, self.chart_speed):
            chart.empty_text = t("an.empty")

    def _clear(self):
        for chart in (self.chart_loss, self.chart_lr, self.chart_speed):
            chart.clear()
            chart.empty_text = t("an.empty")
        for tile in (self.k_steps, self.k_best, self.k_last,
                     self.k_speed, self.k_time, self.k_tokens):
            tile.set("—")

    # ------------------------------------------------------------ 导出
    def _export(self):
        if not self._rows:
            return
        path, _filt = QFileDialog.getSaveFileName(
            self, t("an.export"), "", "PNG (*.png)")
        if not path:
            return
        # 三张图纵向拼起来：先各自 grab，再画到一张 QPixmap 上
        shots = [chart.grab() for chart in
                 (self.chart_loss, self.chart_lr, self.chart_speed)]
        if not shots:
            return
        w = max(s.width() for s in shots)
        gap = 12
        h = sum(s.height() for s in shots) + gap * (len(shots) - 1)
        from PySide6.QtGui import QColor, QPainter, QPixmap
        out = QPixmap(w, h)
        out.fill(QColor("#14161b"))
        p = QPainter(out)
        y = 0
        for s in shots:
            p.drawPixmap(0, y, s)
            y += s.height() + gap
        p.end()
        out.save(path)
        self.meta_label.setText(t("an.export.hint"))


def _fmt_int(v):
    if v is None:
        return "—"
    try:
        return "{:,}".format(int(round(float(v))))
    except (TypeError, ValueError):
        return "—"


def _fmt_loss(v):
    if v is None:
        return "—"
    return "%.4f" % float(v)
