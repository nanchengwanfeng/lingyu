# -*- coding: utf-8 -*-
"""语料准备页。

用户明确要求「至少要有个语料准备界面」，而这一步又恰好是新手最容易翻车的
地方（路径填错、词表大小选错、把说明文件当语料）。所以这一页的设计目标是：

1. **分四步走**，每步一个小卡片，顺序写在标题里（①②③④），不用猜先做什么。
2. **先体检再决定**。词表大小不该让用户瞎填 —— 点「体检」后自动填建议值。
3. **体检报告直接显示在界面上**（不是让他去翻日志），
   覆盖率不到 100% 时**高亮警告**。
4. **进度实时可见**。prepare 要跑几十秒到几分钟，没有进度条用户会以为卡死。
"""

import os
import re

from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QWidget)

from ..i18n import t
from ..runner import Runner
from ..theme import SPACE, th
from ..widgets import (Card, Field, KeyValue, LogView, PathPicker, ghost_button,
                       make_check, make_combo, make_float, make_int, muted,
                       primary_button, unify_labels)
from . import BasePage


class CorpusPage(BasePage):
    nav_key = "nav.corpus"
    icon = "▤"

    def header(self):
        return (t("cp.title"), t("cp.desc"))

    def build(self):
        self.runner = Runner(self)
        self.runner.line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)
        self._scanned = False

        # 这一页的标签都是中文说明式的（「字对最少出现几次才合并」），
        # 比英文长。先照常建，最后用 unify_labels 一次性对齐 ——
        # 同一页里所有控件的左边缘就落在同一条竖线上。
        fields = []

        def F(label, w, hint=""):
            f = Field(label, w, hint)
            fields.append(f)
            return f

        cols = QHBoxLayout()
        cols.setSpacing(SPACE["lg"])

        # ================= 左边：四步表单 =================
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(SPACE["lg"])

        # ---- ① 语料来源 ----
        c1 = Card(t("cp.src"), t("cp.src.dir.hint"))
        self.raw = PathPicker(self.ctx.settings.get("raw_dir", ""), is_dir=True)
        c1.add(F(t("cp.src.dir"), self.raw))
        row = QHBoxLayout()
        self.scan_btn = ghost_button(t("cp.scan"))
        self.scan_btn.clicked.connect(self._scan)
        row.addWidget(self.scan_btn)
        row.addStretch(1)
        c1.add_layout(row)
        self.scan_result = muted("", wrap=True)
        # 空的时候藏起来 —— 一个空的 QLabel 仍然占一行高度，
        # 会在按钮和下一个控件之间留出一块莫名其妙的空白。
        self.scan_result.setVisible(False)
        c1.add(self.scan_result)
        self.strip_ads = make_check(t("cp.clean.strip_ads"), True)
        c1.add(self.strip_ads)
        # 默认不抽正文：.json/.jsonl 原样进语料，{}、键名、缩进都保留。
        # 想只要正文（跳过 id/url 之类）就勾上。
        self.json_extract = make_check(t("cp.clean.json_extract"), False)
        c1.add(self.json_extract)
        lv.addWidget(c1)

        # ---- ② 分词器 ----
        c2 = Card(t("cp.tok"))
        self.tok_kind = make_combo(
            [(t("cp.tok.kind.bpe"), "bpe"), (t("cp.tok.kind.char"), "char")],
            "bpe")
        self.tok_kind.currentIndexChanged.connect(self._on_kind_changed)
        c2.add(F(t("cp.tok.kind"), self.tok_kind))
        self.vocab = make_int(256, 200000, 8192, 512)
        self.vocab_field = F(t("cp.tok.vocab"), self.vocab, t("cp.tok.vocab.hint"))
        c2.add(self.vocab_field)
        self.max_chars = make_int(1, 16, 4)
        c2.add(F(t("cp.tok.maxchars"), self.max_chars,
                 "「中华人民共和国」要是能合成一个 token，这里就填 7"))
        self.min_pair = make_int(1, 100, 2)
        c2.add(F(t("cp.tok.minpair"), self.min_pair))
        lv.addWidget(c2)

        # ---- ③ 验证集 ----
        c3 = Card(t("cp.val"))
        self.val_ratio = make_float(0.0, 0.5, 0.02, 0.01, 3)
        # 卡片标题已经写了「验证集比例」，这里不再重复一遍标签
        c3.add(F("", self.val_ratio, t("cp.val.hint")))
        lv.addWidget(c3)

        # ---- ④ 输出 ----
        c4 = Card(t("cp.out"))
        # ★ 默认值取自**配置里的 data_dir**，不是 gui.json。
        #   训练读的是配置，语料就该写在配置说的地方 —— 一个目录只能有一个主人。
        #   踩过的坑：这里原先取 gui.json，于是语料落在 runtime/data，
        #   而配置指着 runtime/app/data1，刚准备完就提示「还没准备语料」。
        self.data_dir = PathPicker(
            self.ctx.data_dir() or self.ctx.settings.get("data_dir", ""),
            is_dir=True)
        c4.add(F(t("cp.out.dir"), self.data_dir))
        self.start_btn = primary_button(t("cp.start"))
        self.start_btn.clicked.connect(self._start)
        self.start_btn.setEnabled(False)
        c4.add(self.start_btn)
        self.start_hint = muted(t("cp.scan.first"), wrap=True)
        c4.add(self.start_hint)
        self.sync_hint = muted("", wrap=True)
        self.sync_hint.setVisible(False)
        c4.add(self.sync_hint)
        lv.addWidget(c4)
        lv.addStretch(1)

        unify_labels(fields)

        cols.addWidget(self.scrollable(left), 3)

        # ================= 右边：日志与结果 =================
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(SPACE["lg"])

        self.result_card = Card(t("cp.result"))
        self.kv = KeyValue()
        self.result_card.add(self.kv)
        self.preview_btn = ghost_button(t("cp.preview"))
        self.preview_btn.clicked.connect(self._open_preview)
        self.preview_btn.setVisible(False)
        self.result_card.add(self.preview_btn)
        rv.addWidget(self.result_card)

        log_card = Card(t("mo.log"))
        self.log = LogView()
        self.log.setMinimumHeight(300)
        log_card.add(self.log)
        rv.addWidget(log_card, 1)

        cols.addWidget(right, 4)
        self.add(self._wrap(cols))

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    # ------------------------------------------------------------ 事件
    def _on_kind_changed(self):
        """字符级分词器的「最多几个字」「最少字对」没有意义，禁掉。

        留着能点但不生效，用户会以为设置了没起作用 —— 不如直接灰掉并说明。
        """
        char = self.tok_kind.currentData() == "char"
        for w in (self.max_chars, self.min_pair):
            w.setEnabled(not char)
        if char:
            self.vocab_field.hint.setText("字符级：词表大小 = 保留多少种字符")
            self.vocab_field.hint.setVisible(True)
        else:
            self.vocab_field.hint.setText(t("cp.tok.vocab.hint"))
            self.vocab_field.hint.setVisible(True)

    def _scan(self):
        raw = self.raw.value()
        if not raw or not os.path.isdir(raw):
            self._warn("先选一个放语料的目录（.txt / .md / .json / .jsonl）")
            return
        self.log.clear()
        self.scan_result.setText("正在体检……")
        self.scan_result.setVisible(True)
        scan_args = ["scan", "--raw", raw]
        if self.json_extract.isChecked():
            scan_args.append("--json-extract")
        ok = self.runner.start(scan_args)
        if ok:
            self.ctx.begin_task("scan")
            self.scan_btn.setEnabled(False)

    def _start(self):
        raw, data = self.raw.value(), self.data_dir.value()
        if not raw or not os.path.isdir(raw):
            self._warn("先选语料目录")
            return
        if not data:
            self._warn("先选数据输出目录")
            return
        args = ["prepare", "--raw", raw, "--data", data,
                "--tokenizer", self.tok_kind.currentData(),
                "--vocab-size", str(self.vocab.value()),
                "--val-ratio", "%.3f" % self.val_ratio.value(),
                "--min-pair-count", str(self.min_pair.value()),
                "--max-token-chars", str(self.max_chars.value())]
        if self.strip_ads.isChecked():
            args.append("--strip-ads")
        if self.json_extract.isChecked():
            args.append("--json-extract")
        self.log.clear()
        if self.runner.start(args):
            self.ctx.begin_task("prepare")
            self.start_btn.setEnabled(False)
            self._set_busy(True)

    def _set_busy(self, busy):
        self.scan_btn.setEnabled(not busy)
        self.start_btn.setEnabled(not busy and self._scanned)
        self.start_hint.setText(t("common.running") if busy
                                else ("" if self._scanned else t("cp.scan.first")))

    def _on_line(self, text, level):
        self.log.append_line(text, level)
        # 从体检输出里抓「建议词表大小」，自动填进输入框 ——
        # 这一步替用户做掉，他就不会瞎猜一个数字
        m = re.search(r"建议词表大小[：:]\s*([\d,]+)", text)
        if m:
            try:
                v = int(m.group(1).replace(",", ""))
                if 256 <= v <= 200000:
                    self.vocab.setValue(v)
                    self.vocab_field.hint.setText(
                        t("cp.tok.suggest", n="%d" % v))
                    self.vocab_field.hint.setVisible(True)
            except ValueError:
                pass

    def _on_finished(self, code, stopped):
        name = self.ctx.running_task
        self.ctx.end_task(name or "prepare", code)
        self._set_busy(False)
        if name == "scan":
            self._scanned = (code == 0)
            self.scan_result.setText(t("common.done") if code == 0
                                     else t("common.failed"))
            self.scan_result.setVisible(True)
            self.scan_result.setStyleSheet(
                "color:%s;" % (th.SUCCESS if code == 0 else th.DANGER))
            self.start_btn.setEnabled(code == 0)
            if code == 0:
                self.start_hint.setText("")
        elif name == "prepare" and code == 0:
            self._fill_result()
            self.preview_btn.setVisible(True)
            self.ctx.settings["raw_dir"] = self.raw.value()
            self.ctx.settings["data_dir"] = self.data_dir.value()
            self.ctx.save()
            self._sync_config()
            self.ctx.corpus_changed.emit()

    def _sync_config(self):
        """把刚准备好的目录写进**当前配置**。

        ## 为什么必须做这一步

        以前语料页只更新 `gui.json`，训练页只看配置文件 —— 同一个事实
        记在两个地方，谁也不知道对方改了。用户照着提示准备完语料，
        一转头训练页还是说「还没准备语料」，而两边都查不出毛病。

        准备完就顺手把配置指过来，两个地方从此永远是同一个目录。

        ## 用 patch_config 而不是 save_config

        `save_config()` 会按 spec 重新生成整份文件，把出厂配置里那些
        中文注释全抹掉 —— 用户只是准备了个语料，回头打开 small.ini
        发现自己看不懂了。`patch_config()` 只改这两行，其余原样不动。
        """
        import os as _os

        from lingyu.config import patch_config

        dd = self.data_dir.value()
        tp = _os.path.join(dd, "tokenizer.txt")
        data_sec = (self.ctx.config or {}).get("data") or {}

        def same(a, b):
            return _os.path.normcase(_os.path.normpath(a)) == \
                _os.path.normcase(_os.path.normpath(b))

        if (data_sec.get("data_dir") and same(data_sec["data_dir"], dd)
                and data_sec.get("tokenizer_path")
                and same(data_sec["tokenizer_path"], tp)):
            self.sync_hint.setVisible(False)
            return
        try:
            n = patch_config(self.ctx.config_path,
                             {("data", "data_dir"): dd.replace("\\", "/"),
                              ("data", "tokenizer_path"): tp.replace("\\", "/")})
        except Exception as e:                       # noqa: BLE001
            self.log.append_line("把语料路径写进配置失败：%s" % e, "warn")
            return
        if n:
            self.log.append_line(
                "已把语料路径写进配置：%s" % self.ctx.config_path, "ok")
            self.sync_hint.setText("语料路径已写进配置：%s"
                                   % self.ctx.config_path)
            self.sync_hint.setVisible(True)
            try:
                self.ctx.load_config()               # 让训练页立刻看到新路径
            except Exception:                        # noqa: BLE001
                pass

    def _fill_result(self):
        """从 train.meta.json 读结果填进「结果」卡片。

        读文件而不是解析日志 —— 日志格式以后可能改，meta 文件是稳定的数据契约。
        """
        import json
        dd = self.data_dir.value()
        self.kv.clear()
        for name, key in (("train.meta.json", "train"), ("val.meta.json", "val")):
            p = os.path.join(dd, name)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    m = json.load(fp)
            except (OSError, ValueError):
                continue
            self.kv.add("%s tokens" % key, "{:,}".format(m.get("n_tokens", 0)))
            if key == "train":
                self.kv.add(t("ov.data.vocab"),
                            "{:,}".format(m.get("vocab_size", 0)))
                comp = m.get("compression")
                if comp:
                    self.kv.add(t("ov.data.comp"), "%.3f 字符/token" % float(comp))
                self.kv.add("分词器", m.get("tokenizer_type", "—"))
                self.kv.add(t("ov.data.files"),
                            "%d 个" % len(m.get("corpus_files") or []))
                fp_ = m.get("tokenizer_fingerprint")
                if fp_:
                    self.kv.add("词表指纹", fp_)
        size = 0
        tb = os.path.join(dd, "train.bin")
        if os.path.isfile(tb):
            size = os.path.getsize(tb) / 1024 / 1024
            self.kv.add(t("ck.size"), "%.2f MB" % size)

    def _open_preview(self):
        p = os.path.join(self.data_dir.value(), "corpus_preview.txt")
        if os.path.isfile(p):
            os.startfile(p) if hasattr(os, "startfile") else None

    def _warn(self, msg):
        self.log.append_line(msg, "warn")

    def on_show(self):
        if self._scanned:
            self.start_btn.setEnabled(not self.ctx.is_busy())
