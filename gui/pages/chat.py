# -*- coding: utf-8 -*-
"""聊天页。

## 为什么不用「一轮一个子进程」

一开始想的是每问一句就跑一次 `lingyu chat --prompt ...`。这样很省事，
但有两个问题：

1. **每次都要重新加载模型**（13M 还好，79M 就要一两秒，体验很碎）
2. **多轮上下文没法保持** —— 子进程退出就忘了

所以聊天走**常驻子进程**：`lingyu chat` 不带 `--prompt` 时是交互模式，
从 stdin 读问题、往 stdout 写回答。界面开一个 QProcess 挂着它，
每轮把问题写进 stdin 就行。模型只加载一次，历史由 chat.py 自己维护。

## 怎么知道「它说完了」

`chat.py` 每轮回答完会打印下一轮的提示符 `你 > `。界面等这个标记出现，
就认为这一轮结束了，把助手的气泡定稿。

这比「等进程退出」靠得住 —— 进程不会退出。
"""

import os

from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit,
                               QScrollArea, QVBoxLayout, QWidget)

from ..i18n import t
from ..runner import child_environment
from ..theme import SPACE, th
from ..widgets import (Card, Field, LogView, danger_button, ghost_button,
                       make_combo, make_float, make_int, muted, primary_button)
from . import BasePage

PROMPT_MARK = "你 > "
REPLY_MARK = "灵语 > "


class Bubble(QWidget):
    """一条聊天气泡。用户右对齐（主色底），助手左对齐（卡片底）。"""

    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setMaximumWidth(620)
        pad = "%dpx %dpx" % (SPACE["md"], SPACE["lg"])
        if is_user:
            self.label.setStyleSheet(
                "background:%s; color:%s; border-radius:12px; padding:%s;"
                % (th.ACCENT, th.TEXT_ON_ACCENT, pad))
            lay.addStretch(1)
            lay.addWidget(self.label)
        else:
            self.label.setStyleSheet(
                "background:%s; color:%s; border:1px solid %s;"
                "border-radius:12px; padding:%s;"
                % (th.SURFACE, th.TEXT, th.BORDER, pad))
            lay.addWidget(self.label)
            lay.addStretch(1)

    def set_text(self, text):
        self.label.setText(text)


class ChatInput(QPlainTextEdit):
    """回车发送，Shift+回车换行。"""

    submitted = Signal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if e.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(e)
            else:
                self.submitted.emit()
            return
        super().keyPressEvent(e)


class ChatPage(BasePage):
    nav_key = "nav.chat"
    icon = "◍"

    def header(self):
        return (t("ch.title"), t("ch.desc"))

    def build(self):
        self.proc = None
        self.bubbles = []
        self._pending = None
        self._buf = ""
        self._history = []          # [(role, text), …] 用于导出
        self._forced_ckpt = None

        cols = QHBoxLayout()
        cols.setSpacing(SPACE["lg"])

        # ================= 左：设置 =================
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(SPACE["lg"])

        c1 = Card()
        self.ckpt_combo = make_combo([], None)
        c1.add(Field(t("ch.ckpt"), self.ckpt_combo))
        self.temp = make_float(0.01, 2.0, 0.8, 0.05, 2)
        c1.add(Field(t("ch.temp"), self.temp))
        self.maxnew = make_int(16, 2000, 200, 32)
        c1.add(Field(t("ch.maxnew"), self.maxnew))
        lv.addWidget(c1)

        btns = QHBoxLayout()
        self.start_btn = primary_button("加载模型")
        self.start_btn.clicked.connect(self._load_model)
        btns.addWidget(self.start_btn, 2)
        self.stop_btn = danger_button(t("common.stop"))
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setVisible(False)
        btns.addWidget(self.stop_btn, 1)
        lv.addLayout(btns)

        self.state = muted("")
        lv.addWidget(self.state)

        self.export_btn = ghost_button(t("ch.export"))
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        lv.addWidget(self.export_btn)

        tip = muted(t("ch.tip"), wrap=True)
        lv.addWidget(tip)
        lv.addStretch(1)
        cols.addWidget(left, 2)

        # ================= 右：对话 =================
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(SPACE["md"])

        # 对话区放进滚动区，**外面套一层带背景的容器**
        self.chat_card = Card()
        self.scroll = QScrollArea()
        self.scroll.setObjectName("ChatArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.holder = QWidget()
        self.chat_lay = QVBoxLayout(self.holder)
        self.chat_lay.setContentsMargins(SPACE["md"], SPACE["md"],
                                         SPACE["md"], SPACE["md"])
        self.chat_lay.setSpacing(SPACE["sm"])
        self.chat_lay.addStretch(1)
        self.scroll.setWidget(self.holder)
        self.chat_card.add(self.scroll, 1)
        rv.addWidget(self.chat_card, 1)

        self.input = ChatInput()
        self.input.setObjectName("ChatInput")
        self.input.setPlaceholderText(t("ch.placeholder"))
        self.input.setFixedHeight(78)
        self.input.submitted.connect(self._send)
        rv.addWidget(self.input)

        bar = QHBoxLayout()
        self.send_btn = primary_button(t("ch.send"))
        self.send_btn.clicked.connect(self._send)
        bar.addWidget(self.send_btn, 2)
        self.clear_btn = ghost_button(t("ch.clear"))
        self.clear_btn.clicked.connect(self._clear)
        bar.addWidget(self.clear_btn, 1)
        rv.addLayout(bar)

        self.log = LogView()
        self.log.setMaximumHeight(120)
        rv.addWidget(self.log)
        cols.addWidget(right, 5)

        self.add(self._wrap(cols))
        self._add_bubble(t("ch.welcome"), is_user=False)
        self._reload_ckpts()

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    # ------------------------------------------------------------ 模型
    def _reload_ckpts(self):
        """优先列 SFT 出来的断点 —— 聊天要的是会答话的模型。"""
        items = []
        sft_dir = self.ctx.sft_out_dir()
        for c in self.ctx.list_checkpoints(sft_dir):
            items.append(("✦ " + c["name"], c["path"]))
        for c in self.ctx.list_checkpoints():
            items.append((c["name"], c["path"]))
        self.ckpt_combo.blockSignals(True)
        self.ckpt_combo.clear()
        for label, val in items:
            self.ckpt_combo.addItem(label, val)
        if not items:
            self.ckpt_combo.addItem(t("ch.no_ckpt"), "")
        self.ckpt_combo.blockSignals(False)
        self.start_btn.setEnabled(bool(items))

    def set_checkpoint(self, path):
        self._forced_ckpt = path
        i = self.ckpt_combo.findData(path)
        if i >= 0:
            self.ckpt_combo.setCurrentIndex(i)

    def on_show(self):
        self._reload_ckpts()
        if self._forced_ckpt:
            self.set_checkpoint(self._forced_ckpt)
        self._update_state()

    # ------------------------------------------------------------ 常驻进程
    def _load_model(self):
        ck = self.ckpt_combo.currentData()
        if not ck:
            self.log.append_line(t("ch.no_ckpt"), "warn")
            return
        if self.proc is not None:
            self._stop()

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(self.ctx.project_root)
        # 同 runner：必须继承当前进程环境，否则打包版丢 PYTHONPATH
        env = child_environment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_proc_finished)
        self.proc = proc
        self._buf = ""
        self._stderr_tail = []
        self._stopping = False

        args = ["-u", "-m", "lingyu", "chat",
                "--config", self.ctx.config_path, "--ckpt", ck,
                "--set", "sample.temperature=%.2f" % self.temp.value(),
                "--set", "sample.max_new_tokens=%d" % self.maxnew.value()]
        import sys
        proc.start(sys.executable, args)
        self.log.append_line("正在加载模型……（第一次要几秒到几十秒）", "info")
        self.state.setText(t("common.running"))
        self.state.setStyleSheet("color:%s;" % th.ACCENT)
        self.start_btn.setEnabled(False)

    def _stop(self):
        """主动停掉聊天进程。

        ★ 必须打标记。`kill()` 之后 `finished` 会带着**非零退出码**回来，
        不去区分的话就会把「用户自己点的停止」报成「聊天进程退出，码 1」——
        看起来像崩溃，用户会以为坏了。

        实测踩过：用户看模型加载慢，又点了一下「加载模型」，
        界面上就出现两行吓人的「聊天进程退出，码 1」。
        """
        if self.proc is not None:
            self._stopping = True
            self.proc.kill()
            self.proc.waitForFinished(2000)
            self.proc = None
        self._update_state()

    def _on_proc_finished(self, code, _st):
        was_stopping = getattr(self, "_stopping", False)
        self._stopping = False
        self.proc = None
        self._update_state()
        if was_stopping:
            self.log.append_line("（已停止）", "dim")
            return
        if code not in (0, None):
            self.log.append_line("聊天进程异常退出，码 %s" % code, "error")
            # ★ 把子进程的真实输出打出来。
            #   以前只挑「以 [ 开头」或含「错误」的行打，Python 的
            #   `ModuleNotFoundError: No module named 'lingyu'` 这类
            #   关键信息**全被过滤掉了** —— 用户只看到一句「码 1」，
            #   谁也没法据此排查。
            tail = [x for x in getattr(self, "_stderr_tail", []) if x.strip()]
            if tail:
                self.log.append_line("—— 它最后说的话 ——", "dim")
                for line in tail[-14:]:
                    self.log.append_line("  " + line, "dim")
            else:
                self.log.append_line("（它什么都没输出就退了，"
                                     "多半是环境或路径问题）", "dim")

    def _update_state(self):
        running = self.proc is not None and \
            self.proc.state() != QProcess.NotRunning
        self.start_btn.setVisible(not running)
        self.start_btn.setEnabled(not running and self.ckpt_combo.count() > 0
                                  and bool(self.ckpt_combo.currentData()))
        self.stop_btn.setVisible(running)
        self.send_btn.setEnabled(running)
        self.input.setEnabled(running)
        if not running:
            self.state.setText("模型未加载 —— 点上面的按钮加载")
            self.state.setStyleSheet("color:%s;" % th.TEXT_MUTED)

    # ------------------------------------------------------------ 收发
    def _send(self):
        if self.proc is None or self.proc.state() == QProcess.NotRunning:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._add_bubble(text, is_user=True)
        self._history.append(("user", text))
        self._pending = Bubble(t("ch.thinking"), is_user=False)
        self._append_widget(self._pending)
        self._buf = ""
        self.proc.write((text + "\n").encode("utf-8"))

    def _on_output(self):
        if self.proc is None:
            return
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += raw
        # 回答可能在任意位置被切开，所以要缓冲到看见下一个提示符
        if REPLY_MARK in self._buf:
            i = self._buf.index(REPLY_MARK) + len(REPLY_MARK)
            reply = self._buf[i:]
            # 去掉结尾的 "你 > "（下一轮提示符）
            for mark in (PROMPT_MARK, "你 >"):
                if mark in reply:
                    reply = reply[:reply.index(mark)]
            reply = reply.strip()
            if self._pending is not None:
                self._pending.set_text(reply or "（模型没说话）")
                self._history.append(("assistant", reply))
                self._pending = None
            self.export_btn.setEnabled(len(self._history) >= 2)
            self._buf = ""
        # 把非对话的诊断信息丢进小日志。
        # 同时**留一份尾巴**：进程异常退出时要把最后几行原样打出来，
        # 否则真正的报错会被下面的过滤条件吃掉（见 _on_proc_finished）。
        tail = getattr(self, "_stderr_tail", None)
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if tail is not None:
                tail.append(s)
                tail[:] = tail[-40:]
            # 对话正文和提示符不往小日志里塞（它们属于气泡区）
            if s.startswith(REPLY_MARK) or PROMPT_MARK in s:
                continue
            if s.startswith("=") or "灵语对话" in s:
                continue
            if s.startswith("[") or "错误" in s or "注意" in s:
                self.log.append_line(s, "warn" if "注意" in s else "info")

    # ------------------------------------------------------------ 界面
    def _add_bubble(self, text, is_user=False):
        b = Bubble(text, is_user)
        self._append_widget(b)
        return b

    def _append_widget(self, w):
        self.chat_lay.insertWidget(self.chat_lay.count() - 1, w)
        QTimer.singleShot(30, self._scroll_bottom)

    def _scroll_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear(self):
        for b in self.bubbles:
            b.deleteLater()
        self.bubbles = []
        while self.chat_lay.count() > 1:
            item = self.chat_lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._history = []
        self.export_btn.setEnabled(False)
        self._add_bubble(t("ch.welcome"), is_user=False)

    # ------------------------------------------------------------ 导出
    def _export(self):
        """把这段对话导成 SFT 数据 —— 闭环：聊出来的对话能反哺训练。"""
        import json
        pairs = []
        msgs = []
        for role, text in self._history:
            msgs.append({"role": role, "content": text})
            if role == "assistant" and len(msgs) >= 2:
                pairs.append({"messages": list(msgs)})
                msgs = msgs[:-2] + []      # 下一轮从头开始
                msgs = []
        if not pairs:
            return
        out_dir = self.ctx.abspath("chat")
        os.makedirs(out_dir, exist_ok=True)
        import time
        path = os.path.join(out_dir, "chat_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
        with open(path, "w", encoding="utf-8") as fp:
            for p in pairs:
                fp.write(json.dumps(p, ensure_ascii=False) + "\n")
        self.log.append_line(t("ch.exported", n=len(pairs), path=path), "ok")

    def on_task_event(self, name, code=None):
        pass
