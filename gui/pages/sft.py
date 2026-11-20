# -*- coding: utf-8 -*-
"""SFT 页：教模型回答问题。

## 这一页要防住的三个翻车点

1. **学习率填成预训练那个量级** → 模型几十步内「忘光」。
   所以界面上实时检查：超过 2e-3 就在输入框下面红字警告。

2. **数据集路径不对/格式不对** → 跑到一半才发现。
   所以给了「检查数据」按钮，**开跑前**就把条数、平均长度、
   丢掉了多少条报出来。用户能立刻发现「哦我路径写错了，0 条样本」。

3. **不知道 SFT 到底有没有效果** → 光看 loss 看不出来。
   所以默认开启「训练中试问」，每隔几百步真的问它两句，
   把回答打在日志里。这是判断 SFT 成没成最直接的办法。
"""

import os

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget)

from ..i18n import t
from ..runner import Runner
from ..theme import SPACE, th
from ..widgets import (Card, Field, LogView, PathPicker, danger_button,
                       ghost_button, make_combo, make_float, make_int, muted,
                       primary_button, unify_labels)
from . import BasePage

SAMPLE_DATA = os.path.join("chat", "sample_chat.jsonl")


class SftPage(BasePage):
    nav_key = "nav.sft"
    icon = "✦"

    def header(self):
        return (t("sft.title"), t("sft.desc"))

    def build(self):
        self.runner = Runner(self)
        # 任务输出走 ctx 总线（页面重建后也收得到，审计 1.7）
        self.ctx.task_line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)
        self._forced_ckpt = None
        # 收集本页所有 Field，最后统一标签宽度（见 unify_labels 的说明）
        self._fields = []

        def F(label, w, hint="", help=""):
            f = Field(label, w, hint, help_text=help)
            self._fields.append(f)
            return f

        cols = QHBoxLayout()
        cols.setSpacing(SPACE["lg"])

        # ================= 左：表单 =================
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(SPACE["lg"])

        c1 = Card(t("sft.data"), t("sft.data.hint"))
        # 默认用仓库自带的示例数据 —— 让用户第一遍就能跑通，
        # 而不是先卡在「我去哪找数据集」
        default_data = self.ctx.get("post", "data_path", "") or \
            self.ctx.abspath(SAMPLE_DATA)
        self.data_path = PathPicker(str(default_data), is_dir=False)
        self.data_path.changed.connect(lambda *_: self._data_dirty())
        c1.add(F(t("sft.data"), self.data_path, help=t("help.sft.data")))
        row = QHBoxLayout()
        self.check_btn = ghost_button(t("sft.check"))
        self.check_btn.clicked.connect(self._check_data)
        row.addWidget(self.check_btn)
        self.sample_btn = ghost_button(t("sft.use_sample"))
        self.sample_btn.clicked.connect(self._use_sample)
        row.addWidget(self.sample_btn)
        row.addStretch(1)
        c1.add_layout(row)
        self.data_info = muted("")
        c1.add(self.data_info)
        lv.addWidget(c1)

        c2 = Card("模型与模板")
        self.ckpt_combo = make_combo([], None)
        self.ckpt_combo.currentIndexChanged.connect(self._on_ckpt_changed)
        c2.add(F(t("sft.init"), self.ckpt_combo, help=t("help.sft.init")))
        self.template = make_combo(
            [(t("sft.template.plain"), "plain"),
             (t("sft.template.chatml"), "chatml"),
             (t("sft.template.qwen"), "qwen")],
            self.ctx.get("post", "template", "plain") or "plain")
        c2.add(F(t("sft.template"), self.template, help=t("help.sft.template")))
        self.system = QLineEdit(self.ctx.get("post", "system_prompt", "") or "")
        self.system.setPlaceholderText(t("common.optional"))
        c2.add(F(t("sft.system"), self.system, help=t("help.sft.system")))
        lv.addWidget(c2)

        c3 = Card("训练参数")
        self.steps = make_int(1, 1000000, int(self.ctx.get("post", "max_steps", 2000) or 2000), 100)
        c3.add(F(t("sft.steps"), self.steps, help=t("help.sft.steps")))
        self.lr = make_float(1e-6, 1.0, float(self.ctx.get("post", "learning_rate", 5e-4) or 5e-4), 0.0001, 6)
        self.lr.valueChanged.connect(self._check_lr)
        self.lr_field = Field(t("sft.lr"), self.lr, t("sft.lr.hint"))
        c3.add(self.lr_field)
        self.lr_warn = QLabel("")
        self.lr_warn.setVisible(False)
        # 警告文字跟说明文字左对齐（内联排法时缩进到控件那一列）
        self.lr_warn.setContentsMargins(self.lr_field.label.width() + SPACE["md"],
                                        0, 0, 0)
        self.lr_warn.setWordWrap(True)
        c3.add(self.lr_warn)
        self.batch = make_int(1, 512, int(self.ctx.get("post", "batch_size", 16) or 16))
        c3.add(F(t("sft.batch"), self.batch, help=t("help.sft.batch")))
        self.seqlen = make_int(16, 8192, int(self.ctx.get("post", "max_seq_len", 512) or 512), 64)
        c3.add(F(t("sft.seqlen"), self.seqlen, help=t("help.sft.seqlen")))
        self.ask = make_int(1, 10000, int(self.ctx.get("post", "ask_interval", 200) or 200), 50)
        c3.add(F(t("sft.ask"), self.ask, t("sft.ask.hint"), help=t("help.sft.ask")))
        lv.addWidget(c3)
        lv.addStretch(1)
        unify_labels(self._fields)
        cols.addWidget(self.scrollable(left), 3)

        # ================= 右：日志 =================
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(SPACE["lg"])

        run_card = Card()
        self.status = muted("")
        run_card.add(self.status)
        btns = QHBoxLayout()
        self.start_btn = primary_button(t("sft.start"))
        self.start_btn.clicked.connect(self._start)
        btns.addWidget(self.start_btn, 2)
        self.stop_btn = danger_button(t("sft.stop"))
        # clicked(bool) 会污染 grace_ms（审计 G-05）；先问全局任务总线，
        # 页面重建过也能停到真正在跑的那个 Runner（审计 1.7）。
        self.stop_btn.clicked.connect(lambda: self._stop())
        self.stop_btn.setVisible(False)
        btns.addWidget(self.stop_btn, 1)
        run_card.add_layout(btns)
        rv.addWidget(run_card)

        log_card = Card(t("mo.log"))
        self.log = LogView()
        log_card.add(self.log)
        rv.addWidget(log_card, 1)
        cols.addWidget(right, 4)

        self.add(self._wrap(cols))
        self._reload_ckpts()
        self._check_lr()
        self._data_dirty()

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    # ------------------------------------------------------------ 状态
    def _reload_ckpts(self):
        cks = self.ctx.list_checkpoints()
        self.ckpt_combo.blockSignals(True)
        self.ckpt_combo.clear()
        # 优先 best.pt（验证 loss 最低的那个最适合当起点）
        cks.sort(key=lambda c: (0 if c["name"].startswith("best") else 1,
                                -c["mtime"]))
        for c in cks:
            self.ckpt_combo.addItem(c["name"], c["path"])
        if not cks:
            self.ckpt_combo.addItem(t("sft.need_ckpt"), "")
        self.ckpt_combo.blockSignals(False)
        self.start_btn.setEnabled(bool(cks) and not self.ctx.is_busy())
        self.status.setText("" if cks else "⚠ " + t("sft.need_ckpt"))
        if not cks:
            self.status.setStyleSheet("color:%s;" % th.WARNING)

    def set_checkpoint(self, path):
        self._forced_ckpt = path
        i = self.ckpt_combo.findData(path)
        if i >= 0:
            self.ckpt_combo.setCurrentIndex(i)

    def _on_ckpt_changed(self):
        p = self.ckpt_combo.currentData()
        if p:
            self.status.setText(os.path.basename(p))
            self.status.setStyleSheet("color:%s;" % th.TEXT_MUTED)

    def _check_lr(self):
        lr = self.lr.value()
        if lr > 2e-3:
            self.lr_warn.setText("⚠ " + t("sft.lr.warn"))
            self.lr_warn.setStyleSheet("color:%s;" % th.DANGER)
            self.lr_warn.setVisible(True)
        else:
            self.lr_warn.setVisible(False)

    def _data_dirty(self):
        self.data_info.setText("")

    def _use_sample(self):
        p = self.ctx.abspath(SAMPLE_DATA)
        if os.path.isfile(p):
            self.data_path.set_value(p)
            self._check_data()
        else:
            self.data_info.setText("示例数据不在：%s" % p)

    def _check_data(self):
        """开跑前先数一遍样本。**这一步能省掉大量「跑了一半才发现数据不对」。**"""
        p = self.data_path.value()
        if not p or not os.path.isfile(p):
            self.data_info.setText("⚠ 找不到文件：%s" % p)
            self.data_info.setStyleSheet("color:%s;" % th.DANGER)
            return
        try:
            from lingyu.chat_template import load_jsonl, render_dataset
            from lingyu.tokenizer import load_tokenizer
            rows = load_jsonl(p)
            tok_path = self.ctx.tokenizer_path()
            tok = load_tokenizer(tok_path) if os.path.isfile(tok_path) else None
            tpl = self.template.currentData()
            if tok is None:
                self.data_info.setText("✓ %d 条样本（分词器还没准备，无法预览渲染）"
                                       % len(rows))
                self.data_info.setStyleSheet("color:%s;" % th.SUCCESS)
                return
            out = render_dataset(rows, tok, template=tpl,
                                 max_seq_len=self.seqlen.value())
            st = out["stats"]
            msg = t("sft.data_count", n=len(rows), L="%.0f" % st["avg_len"])
            if out["skipped"]:
                msg += "，丢掉 %d 条" % out["skipped"]
                if out["skip_reasons"]:
                    msg += "（%s）" % out["skip_reasons"][0]
            if st["n_samples"] == 0:
                self.data_info.setText("⚠ " + msg)
                self.data_info.setStyleSheet("color:%s;" % th.DANGER)
            else:
                msg += "，回答占 %.0f%%" % (st["supervised_ratio"] * 100)
                self.data_info.setText("✓ " + msg)
                self.data_info.setStyleSheet("color:%s;" % th.SUCCESS)
        except Exception as e:
            self.data_info.setText("⚠ %s" % e)
            self.data_info.setStyleSheet("color:%s;" % th.DANGER)

    # ------------------------------------------------------------ 运行
    def _start(self):
        ck = self.ckpt_combo.currentData()
        if not ck:
            self.log.append_line(t("sft.need_ckpt"), "warn")
            return
        data = self.data_path.value()
        if not os.path.isfile(data):
            self.log.append_line("数据集不存在：%s" % data, "error")
            return
        # 把界面上的值写回配置，子进程才读得到
        cfg = self.ctx.config
        cfg.setdefault("post", {}).update({
            "method": "sft", "data_path": data,
            "template": self.template.currentData(),
            "init_from": ck,
            "system_prompt": self.system.text().strip(),
            "max_steps": self.steps.value(),
            "learning_rate": self.lr.value(),
            "batch_size": self.batch.value(),
            "max_seq_len": self.seqlen.value(),
            "ask_interval": self.ask.value(),
        })
        try:
            self.ctx.save_config()
        except Exception as e:
            self.log.append_line("保存配置失败：%s" % e, "error")
            return
        args = ["sft", "--config", self.ctx.config_path, "--json",
                "--data", data,
                "--template", self.template.currentData(),
                "--init-from", ck]
        self.log.clear()
        if self.runner.start(args):
            self.ctx.begin_task("sft")
            self.ctx.attach_runner("sft", self.runner)
            self._set_running(True)

    def _stop(self):
        # 页面可能被重建过：自己的 runner 不一定是正在跑的那个
        if not self.ctx.stop_current_task():
            self.runner.stop()

    def _set_running(self, running):
        self.start_btn.setVisible(not running)
        self.stop_btn.setVisible(running)

    def _on_line(self, name, text, level):
        if name != "sft":
            return
        self.log.append_line(text, level)

    def _on_finished(self, code, stopped):
        self.ctx.end_task("sft", code)
        self._set_running(False)
        if stopped:
            # 用户主动停止 ≠ 失败（审计 G-04）
            self.log.append_line("已停止（断点已保存）", "ok")
            self.ctx.settings["chat_dir"] = self.ctx.sft_out_dir()
            self.ctx.save()
            self.ctx.checkpoints_changed.emit()
        elif code == 0:
            self.log.append_line("SFT 完成。可以去「聊天」页试试了。", "ok")
            self.ctx.settings["chat_dir"] = self.ctx.sft_out_dir()
            self.ctx.save()
            self.ctx.checkpoints_changed.emit()
        else:
            self.log.append_line("SFT 失败，退出码 %d" % code, "error")

    def on_show(self):
        self._reload_ckpts()
        if self._forced_ckpt:
            self.set_checkpoint(self._forced_ckpt)
        # 任务在后台跑时切回本页：恢复「停止」按钮（审计 1.7）
        if self.ctx.running_task == "sft":
            self._set_running(True)

    def on_task_event(self, name, code=None):
        self._set_running(self.ctx.running_task == "sft")
        if code is not None:
            self._reload_ckpts()
