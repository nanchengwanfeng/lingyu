# -*- coding: utf-8 -*-
"""续写页（sample）：给个开头，让它往下写。

这是预训练模型的原始能力，也是**验证训练有没有效果**最直观的方式。
所以这一页的重点不是参数多，而是：

- 参数给**合理的默认值和一句话解释**，不要求用户懂 temperature 是什么
- 生成结果**流式显示**（逐字出来），这比等十秒突然蹦出一段更有反馈感
- 旁边放一句「训得越久差别越大」的提示 —— 新手最容易训 100 步就来看效果
"""


from PySide6.QtWidgets import (QHBoxLayout, QPlainTextEdit,
                               QVBoxLayout, QWidget)

from ..i18n import t
from ..runner import Runner
from ..theme import SPACE
from ..widgets import (Card, Field, LogView, danger_button, make_combo, make_float, make_int, muted, primary_button)
from . import BasePage


class SamplePage(BasePage):
    nav_key = "nav.sample"
    icon = "✎"

    def header(self):
        return (t("sa.title"), t("sa.desc"))

    def build(self):
        self.runner = Runner(self)
        # 任务输出走 ctx 总线（页面重建后也收得到，审计 1.7）
        self.ctx.task_line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)
        self._forced_ckpt = None

        cols = QHBoxLayout()
        cols.setSpacing(SPACE["lg"])

        # ================= 左：参数 =================
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(SPACE["lg"])

        c1 = Card()
        self.ckpt_combo = make_combo([], None)
        c1.add(Field(t("sa.ckpt"), self.ckpt_combo))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(t("sa.prompt.hint"))
        self.prompt.setPlainText(self.ctx.get("sample", "prompt", "他") or "他")
        self.prompt.setFixedHeight(72)
        c1.add(Field(t("sa.prompt"), self.prompt))
        lv.addWidget(c1)

        c2 = Card("采样参数")
        self.temp = make_float(0.01, 5.0, float(self.ctx.get("sample", "temperature", 0.9) or 0.9), 0.05, 2)
        c2.add(Field(t("sa.temp"), self.temp, t("sa.temp.hint")))
        self.topk = make_int(0, 1000, int(self.ctx.get("sample", "top_k", 40) or 40))
        c2.add(Field(t("sa.topk"), self.topk))
        self.topp = make_float(0.0, 1.0, float(self.ctx.get("sample", "top_p", 0.95) or 0.95), 0.05, 2)
        c2.add(Field(t("sa.topp"), self.topp))
        self.rep = make_float(1.0, 3.0, float(self.ctx.get("sample", "repetition_penalty", 1.05) or 1.05), 0.05, 2)
        c2.add(Field(t("sa.rep"), self.rep, t("sa.rep.hint")))
        self.maxnew = make_int(8, 4000, int(self.ctx.get("sample", "max_new_tokens", 200) or 200), 50)
        c2.add(Field(t("sa.maxnew"), self.maxnew))
        lv.addWidget(c2)

        btns = QHBoxLayout()
        self.run_btn = primary_button(t("sa.run"))
        self.run_btn.clicked.connect(self._run)
        btns.addWidget(self.run_btn, 2)
        self.stop_btn = danger_button(t("sa.stop"))
        # clicked(bool) 会把 checked 当 grace_ms 传进去 → 变成 0ms 强杀。
        # 必须包一层，让 stop() 用默认宽限（审计 G-05）；走全局总线，
        # 页面重建过也能停到真正在跑的 Runner（审计 1.7）。
        self.stop_btn.clicked.connect(lambda: self._stop())
        self.stop_btn.setVisible(False)
        btns.addWidget(self.stop_btn, 1)
        lv.addLayout(btns)
        lv.addStretch(1)
        cols.addWidget(self.scrollable(left), 3)

        # ================= 右：结果 =================
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(SPACE["lg"])

        out_card = Card(t("sa.output"))
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(220)
        self.out.setStyleSheet(
            "font-family:'Microsoft YaHei UI','Segoe UI',sans-serif; font-size:13pt;")
        out_card.add(self.out)
        hint = muted("训 100 步输出像乱码、1000 步开始像中文、5000 步以上才像样。"
                     "别训几步就急着看效果。", wrap=True)
        out_card.add(hint)
        rv.addWidget(out_card, 2)

        log_card = Card(t("mo.log"))
        self.log = LogView()
        log_card.add(self.log)
        rv.addWidget(log_card, 1)
        cols.addWidget(right, 4)

        self.add(self._wrap(cols))
        self._reload_ckpts()

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    # ------------------------------------------------------------ 断点
    def _reload_ckpts(self):
        cks = self.ctx.list_checkpoints()
        items = [(c["name"], c["path"]) for c in cks]
        self.ckpt_combo.blockSignals(True)
        self.ckpt_combo.clear()
        for label, val in items:
            self.ckpt_combo.addItem(label, val)
        if not items:
            self.ckpt_combo.addItem(t("sa.no_ckpt"), "")
        self.ckpt_combo.blockSignals(False)
        self.run_btn.setEnabled(bool(items))
        self.run_btn.setToolTip("" if items else t("sa.no_ckpt"))

    def set_checkpoint(self, path):
        self._forced_ckpt = path
        i = self.ckpt_combo.findData(path)
        if i >= 0:
            self.ckpt_combo.setCurrentIndex(i)

    def on_show(self):
        self._reload_ckpts()
        if self._forced_ckpt:
            self.set_checkpoint(self._forced_ckpt)
        # 任务在后台跑时切回本页：恢复「停止」按钮（审计 1.7）
        if self.ctx.running_task == "sample":
            self.run_btn.setVisible(False)
            self.stop_btn.setVisible(True)

    def on_task_event(self, name, code=None):
        running = self.ctx.running_task == "sample"
        self.run_btn.setVisible(not running)
        self.stop_btn.setVisible(running)

    def _stop(self):
        # 页面可能被重建过：自己的 runner 不一定是正在跑的那个
        if not self.ctx.stop_current_task():
            self.runner.stop()

    # ------------------------------------------------------------ 运行
    def _run(self):
        ck = self.ckpt_combo.currentData()
        if not ck:
            self.log.append_line(t("sa.no_ckpt"), "warn")
            return
        self.out.clear()
        self.log.clear()
        args = ["sample", "--config", self.ctx.config_path, "--ckpt", ck,
                "--json",
                "--prompt", self.prompt.toPlainText().strip() or "他",
                "--max-new-tokens", str(self.maxnew.value()),
                "--temperature", "%.2f" % self.temp.value(),
                "--top-k", str(self.topk.value()),
                "--top-p", "%.2f" % self.topp.value(),
                "--repetition-penalty", "%.2f" % self.rep.value()]
        if self.runner.start(args):
            self.ctx.begin_task("sample")
            self.ctx.attach_runner("sample", self.runner)

    def _on_line(self, name, text, level):
        if name != "sample":
            return
        # 生成结果是逐字流式打出来的，直接在输出框里追加；
        # 其余诊断信息进日志框
        if level == "info" and not text.startswith("[") and \
                "生成" not in text and "断点" not in text and \
                "分词器" not in text and "=" not in text[:3]:
            cur = self.out.toPlainText()
            self.out.setPlainText(cur + "\n" + text if cur else text)
            self.out.verticalScrollBar().setValue(
                self.out.verticalScrollBar().maximum())
        else:
            self.log.append_line(text, level)

    def _on_finished(self, code, stopped):
        self.ctx.end_task("sample", code)
        self.on_task_event("sample")
