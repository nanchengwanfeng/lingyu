# -*- coding: utf-8 -*-
"""监控页：训练时盯这里。

## 两条数据线（审计 1.2）

预训练和 SFT 是**两套不同的数**：

    pretrain  token/步 = batch × grad_accum × block    loss 对全部位置平均
    SFT       token/步 = batch × max_seq_len           loss 只算回答位置

以前监控页只有一条线：它拿当前配置的 `[train]` 参数套到**所有**任务上。
SFT 跑起来时，界面显示的是预训练的 tok/s、总步数，曲线和数字混在一起
（用户实测看到过「微调页 93k tok/s」这种明显不对的数）。
现在训练脚本会往输出目录写 `run.json`（`kind` = pretrain / sft），
监控页据此切换数据源，两套数字互不污染。

## 曲线数据从哪来

不解析 stdout，而是轮询 `out_dir/history.csv`：

- `train.py` 和 `sft.py` 都按同一列写：`step,train_loss,val_loss,lr,grad_norm,elapsed_sec`
- 日志格式以后可能改，但 CSV 是**数据契约**
- 轮询文件还能「事后回看」：训练跑完关掉界面，下次打开照样能看到曲线

## 一个容易被忽略的点

训练还没开始时，曲线区显示一句「等待第一步……」而不是空白的坐标系。
空坐标系看起来像坏了。
"""

import json
import os

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..chart import LossChart
from ..i18n import t
from ..runner import Poller
from ..theme import SPACE, th
from ..widgets import (Card, LogView, StatTile, ghost_button, make_check,
                       make_combo, muted)
from . import BasePage

#: 任务名 → 数据源类型。只有这两个任务会写 history.csv。
TASK_KIND = {"train": "pretrain", "sft": "sft"}


class MonitorPage(BasePage):
    nav_key = "nav.monitor"
    icon = "◔"

    def header(self):
        return (t("mo.title"), t("mo.desc"))

    def build(self):
        # 不自带 Runner：任务输出统一走 ctx 总线（审计 1.7），
        # 监控页只负责「展示」和「轮询文件」。
        self.poller = Poller(1200, self)
        self.poller.changed.connect(self._on_history)

        self._cur_kind = getattr(self.ctx, "monitor_kind", "pretrain")
        self._meta = {}

        # 任务输出（只收当前数据源那个任务的，避免两套日志混在一起）
        self.ctx.task_line.connect(self._bus_line)
        self.ctx.task_progress.connect(self._bus_progress)
        self.ctx.task_validation.connect(self._bus_val)

        # ---------------- 上半部分（数据源 + 数字 + 曲线）----------------
        # ★ 这三块总高约 600px 起步，窗口 1440×900 时整页放不下。
        #   以前直接 add 到根布局，Qt 会把曲线控件压到最小尺寸以下 ——
        #   截图里表现为曲线只画了左边一截、下面坐标轴被裁掉。
        #   所以上半部分放进滚动区，日志块留在外面占满剩余高度。
        body = QWidget()
        blay = QVBoxLayout(body)
        blay.setContentsMargins(0, 0, 0, 0)
        blay.setSpacing(SPACE["lg"])

        src_card = Card(t("mo.runsrc"))
        bar = QHBoxLayout()
        bar.setSpacing(SPACE["md"])
        self.src_combo = make_combo([], None)
        self.src_combo.currentIndexChanged.connect(
            lambda _i: self._select(self.src_combo.currentData()))
        bar.addWidget(self.src_combo, 1)
        self.sync_btn = ghost_button(t("common.refresh"))
        self.sync_btn.clicked.connect(lambda: self._refresh_runs())
        bar.addWidget(self.sync_btn)
        src_card.add_layout(bar)
        self.meta_label = muted("")
        src_card.add(self.meta_label)
        blay.addWidget(src_card)

        # ---------------- 状态数字 ----------------
        row = QHBoxLayout()
        row.setSpacing(SPACE["md"])
        self.tile_step = StatTile(t("mo.step"), "—", accent=th.ACCENT)
        self.tile_loss = StatTile(t("mo.loss"), "—")
        self.tile_val = StatTile(t("mo.valloss"), "—", accent=th.SUCCESS)
        self.tile_speed = StatTile(t("mo.speed"), "—")
        self.tile_elapsed = StatTile(t("mo.elapsed"), "—")
        for w in (self.tile_step, self.tile_loss, self.tile_val,
                  self.tile_speed, self.tile_elapsed):
            row.addWidget(w, 1)
        blay.addWidget(self._wrap(row))

        # ---------------- 曲线 ----------------
        chart_card = Card(t("mo.chart"))
        head = QHBoxLayout()
        self.state_label = muted(t("mo.norun"))
        head.addWidget(self.state_label, 1)
        chart_card.add_layout(head)
        self.chart = LossChart()
        self.chart.empty_text = t("mo.waiting")
        chart_card.add(self.chart, 1)

        # ---------------- 下半部分：曲线 + 日志并排 ----------------
        low = QHBoxLayout()
        low.setSpacing(SPACE["lg"])
        low.addWidget(chart_card, 2)

        log_card = Card(t("mo.log"))
        bar = QHBoxLayout()
        self.autoscroll = make_check(t("mo.autoscroll"), True)
        self.autoscroll.stateChanged.connect(
            lambda: self.log.set_autoscroll(self.autoscroll.isChecked()))
        bar.addWidget(self.autoscroll)
        bar.addStretch(1)
        clear = ghost_button(t("mo.clear"))
        clear.clicked.connect(lambda: self.log.clear())
        bar.addWidget(clear)
        log_card.add_layout(bar)
        self.log = LogView()
        log_card.add(self.log, 1)
        low.addWidget(log_card, 1)
        blay.addLayout(low, 1)

        # 注意：scrollable() 只**返回**滚动区，不会自己挂到页面上。
        self._root.addWidget(self.scrollable(body), 1)

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    def add(self, w, stretch=0):
        self._root.addWidget(w, stretch)
        return w

    # ------------------------------------------------------------ 数据源
    def _dir_of(self, kind):
        try:
            return self.ctx.sft_out_dir() if kind == "sft" else self.ctx.out_dir()
        except Exception:
            return ""

    def _history_path(self):
        d = self._dir_of(self._cur_kind)
        return os.path.join(d, "history.csv") if d else ""

    def _run_label(self, kind):
        return t("mo.run.sft") if kind == "sft" else t("mo.run.pretrain")

    def _read_meta(self, d):
        p = os.path.join(d, "run.json") if d else ""
        if not p or not os.path.isfile(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            if not isinstance(meta, dict):
                return {}
            return meta
        except (OSError, ValueError):
            return {}

    def _refresh_runs(self):
        """重建下拉框（配置换了 / 输出目录变了 / 任务切换过）。"""
        runs = [("pretrain", self.ctx.out_dir()), ("sft", self.ctx.sft_out_dir())]
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        for kind, d in runs:
            name = os.path.basename(os.path.normpath(d)) if d else "—"
            self.src_combo.addItem("%s · %s" % (self._run_label(kind), name), kind)
        idx = max(0, self.src_combo.findData(self._cur_kind))
        self.src_combo.setCurrentIndex(idx)
        self.src_combo.blockSignals(False)
        self._select(self.src_combo.currentData() or "pretrain")

    def _select(self, kind):
        if not kind:
            return
        self._cur_kind = kind
        self.ctx.monitor_kind = kind
        self.ctx.settings["monitor_kind"] = kind
        d = self._dir_of(kind)
        self._meta = self._read_meta(d)
        self.chart.clear()
        for tile in (self.tile_step, self.tile_loss, self.tile_val,
                     self.tile_speed, self.tile_elapsed):
            tile.set("—")
        self.tile_val.set_label(
            t("mo.valloss_sft") if kind == "sft" else t("mo.valloss"))
        self._sync_meta_label()

        path = os.path.join(d, "history.csv") if d else ""
        running = (TASK_KIND.get(getattr(self.ctx, "running_task", None) or "")
                   == self._cur_kind)
        if path and (os.path.isfile(path) or running):
            # 正在训练时 history.csv 可能还没出现（刚开始跑）——先挂上轮询，
            # Poller 会等文件生成；否则页面第一次打开会停在一个空图上。
            exists = os.path.isfile(path)
            self.poller.watch(path)
            if exists:
                self._load_history_once(path)
            elif running:
                self._set_state(t("common.running"), th.ACCENT)
        else:
            self.poller.stop()
            self.state_label.setText(t("mo.empty_dir"))
            self.state_label.setStyleSheet("color:%s;" % th.TEXT_MUTED)

    def _sync_meta_label(self):
        meta = self._meta
        if not meta:
            self.meta_label.setText("")
            return
        self.meta_label.setText("%s · %s token/步 · 共 %s 步" % (
            self._run_label(self._cur_kind),
            "{:,}".format(int(meta.get("tokens_per_step") or 0)),
            "{:,}".format(int(meta.get("max_steps") or 0))))

    def _set_state(self, text, color):
        self.state_label.setText(text)
        self.state_label.setStyleSheet("color:%s;" % color)

    # ------------------------------------------------------------ 曲线数据
    def _load_history_once(self, path):
        if not path or not os.path.isfile(path):
            return
        rows = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        rows.append(line.split(","))
        except OSError:
            return
        self._on_history(rows)

    def _on_history(self, rows):
        steps, losses, vsteps, vlosses = [], [], [], []
        parsed = []                      # (step, 已用秒数 or None)，算速度用
        for r in rows:
            if len(r) < 3:
                continue
            try:
                s = int(float(r[0]))
                tl = float(r[1])
                vl = float(r[2])
            except ValueError:
                continue
            el = None
            if len(r) >= 6:
                try:
                    el = float(r[5])
                except ValueError:
                    el = None
            steps.append(s)
            losses.append(tl)
            parsed.append((s, el))
            # val_loss 是 0 或 nan 表示这一步没做验证
            if vl == vl and vl > 0:
                vsteps.append(s)
                vlosses.append(vl)
        if not steps:
            return
        self.chart.set_data(steps, losses, vsteps, vlosses)
        self.tile_step.set("{:,}".format(steps[-1]))
        self.tile_loss.set("%.4f" % losses[-1], accent=th.ACCENT)
        if vlosses:
            self.tile_val.set("%.4f" % vlosses[-1], accent=th.SUCCESS)
        self._update_rate(parsed)

    def _update_rate(self, parsed):
        """速度 / 耗时 / 预计剩余。

        ★ 速度必须用**相邻两行的增量**算，不能用「总步数 / 累计耗时」：
        续训时 elapsed_sec 会从 0 重新计（history.csv 是追加写的），
        拿末行「195 秒 / 21000 步」会算出 137 万 tok/s 这种离谱值
        —— 2026-10-18 那批真实历史就踩到了（500 步只花 54 秒是正常的，
        但后面两次续训让累计时间不可比）。
        """
        els = [(s, el) for s, el in parsed if el is not None]
        if not els:
            return
        last_step, last_el = els[-1]
        tok = self._tokens_per_step()
        resumed = any(b_step > a_step and b_el + 1e-6 < a_el
                      for (a_step, a_el), (b_step, b_el) in zip(els, els[1:]))

        rate = None                     # tok/s
        for (a_step, a_el), (b_step, b_el) in zip(els, els[1:]):
            d_step = b_step - a_step
            d_el = b_el - a_el
            if d_step > 0 and d_el > 0:
                rate = tok * d_step / d_el
        if rate is None and not resumed and last_step > 0 and last_el > 0:
            rate = tok / (last_el / last_step)

        sub = "%s token/步" % "{:,}".format(tok)
        if not self._meta.get("tokens_per_step"):
            sub += t("mo.est")

        eta = ""
        if rate:
            self.tile_speed.set("%.0f tok/s" % rate, sub=sub)
            total = self._total_steps()
            if total and total > last_step:
                eta = t("mo.eta") + " " + _dur((total - last_step) * tok / rate)
        if eta:
            self.tile_elapsed.set(_dur(last_el), sub=eta)
        elif resumed:
            self.tile_elapsed.set(_dur(last_el), sub=t("mo.thisrun"))
        else:
            self.tile_elapsed.set(_dur(last_el))

    def _total_steps(self):
        if self._meta.get("max_steps"):
            return int(self._meta["max_steps"])
        try:
            return int(self.ctx.get("train", "max_steps", 0) or 0)
        except Exception:
            return 0

    def _tokens_per_step(self):
        """token/步。优先读 run.json（训练脚本自己写的，最准）。"""
        if self._meta.get("tokens_per_step"):
            return int(self._meta["tokens_per_step"])
        try:
            if self._cur_kind == "sft":
                bs = int(self.ctx.get("post", "batch_size", 8) or 8)
                mx = int(self.ctx.get("post", "max_seq_len", 512) or 512)
                return bs * mx
            bs = int(self.ctx.get("train", "batch_size", 8) or 8)
            ga = int(self.ctx.get("train", "grad_accum", 1) or 1)
            blk = int(self.ctx.get("model", "block_size", 512) or 512)
            return bs * ga * blk
        except Exception:
            return 0

    # ------------------------------------------------------------ 任务总线
    def _bus_line(self, name, text, level):
        if TASK_KIND.get(name) != self._cur_kind:
            return
        self.log.append_line(text, level)

    def _bus_progress(self, name, info):
        if TASK_KIND.get(name) != self._cur_kind:
            return
        if info.get("step"):
            self.tile_step.set("{:,}".format(info["step"]))
        if info.get("loss") is not None:
            self.tile_loss.set("%.4f" % info["loss"], accent=th.ACCENT)

    def _bus_val(self, name, step, loss):
        if TASK_KIND.get(name) != self._cur_kind:
            return
        self.tile_val.set("%.4f" % loss, accent=th.SUCCESS)

    # ------------------------------------------------------------ 生命周期
    def on_show(self):
        # 正在跑的任务优先：自动切到对应的数据源
        kind = TASK_KIND.get(self.ctx.running_task or "")
        if kind:
            self._cur_kind = kind
        self._refresh_runs()
        if kind:
            self._set_state(
                t("common.running") + "：" + self._run_label(kind), th.ACCENT)
        elif self._history_path() and os.path.isfile(self._history_path()):
            # 有曲线但没在训练 —— 显示的是上一次跑的结果。
            # 不说明的话用户会以为训练还在跑（数字看着像实时的）。
            self._set_state(t("mo.lastrun"), th.TEXT_MUTED)

    def on_task_event(self, name, code=None):
        kind = TASK_KIND.get(name)
        if not kind:
            return
        if code is None:
            self._cur_kind = kind
            self.ctx.monitor_kind = kind
            self._refresh_runs()
            self._set_state(
                t("common.running") + "：" + self._run_label(kind), th.ACCENT)
            self.poller.watch(os.path.join(self._dir_of(kind), "history.csv"))
            return
        stopped = bool(getattr(self.ctx, "last_finished_stopped", False))
        if code == 0:
            self._set_state(t("common.done"), th.SUCCESS)
        elif stopped:
            self._set_state(t("common.stopped"), th.TEXT_MUTED)
        else:
            self._set_state(t("common.failed"), th.DANGER)
        # 文件里可能还有最后一两行没读到
        self._load_history_once(os.path.join(self._dir_of(kind), "history.csv"))
        self.poller.stop()


def _dur(sec):
    sec = float(sec)
    if sec < 60:
        return "%.0f 秒" % sec
    if sec < 3600:
        return "%d 分 %02d 秒" % (sec // 60, sec % 60)
    return "%.1f 小时" % (sec / 3600)
