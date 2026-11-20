# -*- coding: utf-8 -*-
"""断点管理页。

一个训练项目跑几天会攒下几十个 `ckpt.pt` / `best.pt`，光看文件名分不清
哪个是好的。所以这一页要做的不是「列文件」，而是**回答三个问题**：

1. 哪个断点最好？（显示 best_val，并高亮最小的那个）
2. 每个断点是多大的模型、第几步？（读断点里的 config）
3. 我拿它下一步能干什么？（续训 / 续写 / 聊天 / 做 SFT，一键跳过去）

## 为什么读 config 要放在线程里

`torch.load` 一个断点要几百毫秒（optimizer 状态很大）。十个断点就是几秒，
界面会卡住。所以这里用 QThreadPool 后台读，读到就更新那一行。
"""

import os
import time

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,
                               QLabel, QMessageBox, QTableWidget,
                               QTableWidgetItem)

from ..i18n import t
from ..widgets import (Card, PathPicker, ghost_button, make_combo, muted)
from . import BasePage


class _Loader(QRunnable):
    """后台读一个断点的元信息（不加载 optimizer，只读几个键）。

    ⚠️ 调用方必须**留着这个对象的引用**。QThreadPool 在 runnable 跑完后
    会自动删掉它的 C++ 对象，信号载体 `sig` 跟着被回收，于是
    `sig.done.emit()` 抛 `RuntimeError: Signal source has been deleted`，
    界面上那一行就永远停在「…」。所以页面里用 `self._loaders` 兜着。
    """

    class Sig(QObject):
        done = Signal(str, dict)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.sig = _Loader.Sig()

    def run(self):
        info = {"path": self.path}
        try:
            import torch
            # 先用 weights_only=True（只允许张量/基本类型，安全）；
            # 老格式断点里含自定义对象才退回 False。
            try:
                ck = torch.load(self.path, map_location="cpu",
                                weights_only=True)
            except Exception:
                ck = torch.load(self.path, map_location="cpu",
                                weights_only=False)
            cfg = ck.get("config") or {}
            info.update({
                "step": ck.get("step"),
                "best_val": ck.get("best_val"),
                "n_layer": cfg.get("n_layer"),
                "d_model": cfg.get("d_model"),
                "vocab_size": cfg.get("vocab_size"),
                "use_moe": cfg.get("use_moe"),
                "tokenizer_path": ck.get("tokenizer_path"),
                "template": ck.get("chat_template"),
                "has_opt": ck.get("optimizer") is not None,
            })
        except Exception as e:
            info["error"] = str(e)
        try:
            self.sig.done.emit(self.path, info)
        except RuntimeError:
            # 窗口已经关了、信号载体被回收 —— 这时候没人关心结果了，
            # 静默收场即可。不处理的话 Qt 会往控制台吐一屏 traceback，
            # 用户关个窗口看到一堆红字会以为程序坏了。
            pass


class CheckpointsPage(BasePage):
    nav_key = "nav.checkpoints"
    icon = "❖"

    COLS = ("ck.file", "ck.step", "结构", "ck.bestval", "ck.size", "ck.time")

    def header(self):
        return (t("ck.title"), t("ck.desc"))

    def build(self):
        self.pool = QThreadPool.globalInstance()
        # path -> _Loader。必须是 dict：同一文件不重复加载，回调到了就移除
        # （审计 G-14：旧实现每次刷新都 append，列表随刷新次数无限增长）。
        self._loaders = {}

        bar = Card()
        srow = QHBoxLayout()
        srow.addWidget(QLabel(t("ck.src")))
        self.src_combo = make_combo(
            [(t("ck.src.all"), "all"), (t("ck.src.current"), "current")],
            "all")     # 默认「全部」：用户实测最痛的是「看不到 SFT 断点」
        self.src_combo.currentIndexChanged.connect(lambda *_: self.refresh())
        srow.addWidget(self.src_combo, 1)
        bar.add_layout(srow)
        row = QHBoxLayout()
        self.dir_picker = PathPicker(self.ctx.out_dir(), is_dir=True)
        self.dir_picker.changed.connect(lambda *_: self.refresh())
        row.addWidget(QLabel(t("ck.dir")))
        row.addWidget(self.dir_picker, 1)
        self.refresh_btn = ghost_button(t("common.refresh"))
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.refresh_btn)
        bar.add_layout(row)

        # ---- 行动按钮 ----
        act = QHBoxLayout()
        self.btn_resume = ghost_button(t("ck.action.resume"))
        self.btn_sample = ghost_button(t("ck.action.sample"))
        self.btn_chat = ghost_button(t("ck.action.chat"))
        self.btn_sft = ghost_button(t("ck.action.sft"))
        self.btn_del = ghost_button(t("common.delete"))
        for b in (self.btn_resume, self.btn_sample, self.btn_chat,
                  self.btn_sft, self.btn_del):
            act.addWidget(b)
        act.addStretch(1)
        self.hint = muted("")
        act.addWidget(self.hint)
        bar.add_layout(act)
        self.add(bar)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([t(k) if k.startswith("ck.")
                                              else k for k in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        h = self.table.horizontalHeader()
        # 表头默认是居中的，而单元格内容是左对齐 —— 两者不一致看着就别扭。
        # 统一成左对齐。
        h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(self.COLS)):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.add(self.table, 1)

        self.btn_resume.clicked.connect(lambda: self._use("resume"))
        self.btn_sample.clicked.connect(lambda: self._use("sample"))
        self.btn_chat.clicked.connect(lambda: self._use("chat"))
        self.btn_sft.clicked.connect(lambda: self._use("sft"))
        self.btn_del.clicked.connect(self._delete)

        self._metas = {}
        self._rows = []
        self._update_buttons()

    # ------------------------------------------------------------ 数据
    def on_show(self):
        self.refresh()

    def on_task_event(self, name, code=None):
        if code is not None:
            self.refresh()

    def refresh(self):
        files = self._collect()
        self._rows = files
        self.table.setRowCount(len(files))
        self._metas.clear()
        # 只保留「当前列表里还在的文件」的 loader：
        # ① 换目录/文件被删后，旧 loader 的引用立即释放（审计 G-14）；
        # ② 同一路径已有在跑的 loader 就复用，不再每次刷新新起一个；
        # ③ 回调到了再从 _loaders 里移除，任务结束不留引用。
        wanted = {f["path"] for f in files}
        for p in list(self._loaders):
            if p not in wanted:
                self._loaders.pop(p, None)

        for i, f in enumerate(files):
            label = f["name"]
            kind = f.get("kind")
            if kind == "sft":
                label = t("ck.kind.sft") + " · " + label
            elif kind == "pretrain":
                label = t("ck.kind.pretrain") + " · " + label
            name = QTableWidgetItem(label)
            name.setData(Qt.UserRole, f["path"])
            self.table.setItem(i, 0, name)
            self.table.setItem(i, 1, QTableWidgetItem("…"))
            self.table.setItem(i, 2, QTableWidgetItem("…"))
            self.table.setItem(i, 3, QTableWidgetItem("…"))
            self.table.setItem(i, 4, QTableWidgetItem(_size(f["size"])))
            self.table.setItem(i, 5, QTableWidgetItem(
                time.strftime("%m-%d %H:%M", time.localtime(f["mtime"]))))
            # 后台补元信息。★ 必须留引用，否则信号载体被回收，
            #   回调永远不来（QThreadPool 跑完会自动删 runnable）。
            # 新断点带旁车 JSON：不用开线程、不用 torch，瞬时就能填好
            sidecar = self.ctx.read_ckpt_meta(f["path"])
            if sidecar:
                info = {"path": f["path"], "step": sidecar.get("step"),
                        "best_val": sidecar.get("best_val"),
                        "model_cfg": sidecar.get("model_cfg") or {},
                        "tokenizer_path": sidecar.get("tokenizer_path")}
                self._metas[f["path"]] = info
                self._fill_row(i, sidecar)
                continue
            if f["path"] in self._loaders:
                continue             # 这个文件的 loader 还在跑，别重复加载
            ld = _Loader(f["path"])
            ld.sig.done.connect(self._on_meta)
            self._loaders[f["path"]] = ld
            self.pool.start(ld)

        if not files:
            self.hint.setText(t("ck.none"))
        else:
            self.hint.setText("")
        self._update_buttons()

    def _collect(self):
        """按「来源」下拉取断点：当前项目 / 全部（预训练 + SFT + 历史 run）。"""
        try:
            src = self.src_combo.currentData()
        except Exception:
            src = "current"
        if src == "all":
            return self.ctx.all_checkpoints()
        d = self.dir_picker.value()
        files = self.ctx.list_checkpoints(d)
        for f in files:
            f.setdefault("kind", "sft" if "sft" in f["name"].lower()
                         else "pretrain")
        return files

    def _fill_row(self, row, info):
        """用元信息填一行（旁车 JSON 与 _Loader 共用）。"""
        step = info.get("step")
        self.table.setItem(row, 1, QTableWidgetItem(
            "{:,}".format(step) if isinstance(step, int) else "—"))
        mc = info.get("model_cfg") or {}
        layer = info.get("n_layer") or mc.get("n_layer")
        dim = info.get("d_model") or mc.get("d_model")
        if dim is None:
            dim = mc.get("n_embd")
        moe = info.get("use_moe") if "use_moe" in info else mc.get("use_moe")
        struct = ("%s层 · %s宽%s" % (layer, dim, " · MoE" if moe else "")
                  if layer else "—")
        self.table.setItem(row, 2, QTableWidgetItem(struct))
        bv = info.get("best_val")
        self.table.setItem(row, 3, QTableWidgetItem(
            "%.4f" % bv if isinstance(bv, (int, float))
            and bv == bv and bv != float("inf") else "—"))

    def _on_meta(self, path, info):
        self._loaders.pop(path, None)   # 回调已到：释放引用（审计 G-14）
        row = None
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            if it and it.data(Qt.UserRole) == path:
                row = i
                break
        if row is None:
            return
        self._metas[path] = info
        if info.get("error"):
            self.table.setItem(row, 1, QTableWidgetItem("—"))
            self.table.setItem(row, 2, QTableWidgetItem("读不了"))
            self.table.setItem(row, 3, QTableWidgetItem("—"))
            return
        step = info.get("step")
        self.table.setItem(row, 1, QTableWidgetItem(
            "{:,}".format(step) if isinstance(step, int) else "—"))
        layer = info.get("n_layer")
        dim = info.get("d_model")
        struct = ("%s层 · %s宽%s" % (layer, dim,
                                     " · MoE" if info.get("use_moe") else "")
                  if layer else "—")
        self.table.setItem(row, 2, QTableWidgetItem(struct))
        bv = info.get("best_val")
        cell = QTableWidgetItem("%.4f" % bv if isinstance(bv, (int, float))
                                and bv == bv and bv != float("inf") else "—")
        self.table.setItem(row, 3, cell)

    def _selected(self):
        items = self.table.selectedItems()
        if not items:
            return None
        it = self.table.item(items[0].row(), 0)
        return it.data(Qt.UserRole) if it else None

    def _update_buttons(self):
        has = self._selected() is not None
        for b in (self.btn_resume, self.btn_sample, self.btn_chat,
                  self.btn_sft, self.btn_del):
            b.setEnabled(has)

    def _on_select(self):
        self._update_buttons()
        p = self._selected()
        if not p:
            return
        info = self._metas.get(p, {})
        bits = []
        if info.get("tokenizer_path"):
            bits.append("分词器 " + os.path.basename(info["tokenizer_path"]))
        if info.get("template"):
            bits.append("对话模板 %s" % info["template"])
        if info.get("has_opt") is False:
            bits.append("无优化器状态（不能续训）")
        self.hint.setText("   ".join(bits))

    # ------------------------------------------------------------ 动作
    def _use(self, action):
        p = self._selected()
        if not p:
            return
        win = self.window()
        if action == "resume":
            if hasattr(win, "goto"):
                win.goto("nav.train")
                if hasattr(win, "pages"):
                    # 传选中断点的绝对路径：CLI 的 --resume <path> 支持它
                    win.pages["nav.train"]._start(resume=p)
        elif action == "sample":
            if hasattr(win, "goto"):
                win.goto("nav.sample")
                win.pages["nav.sample"].set_checkpoint(p)
        elif action == "chat":
            if hasattr(win, "goto"):
                win.goto("nav.chat")
                win.pages["nav.chat"].set_checkpoint(p)
        elif action == "sft":
            if hasattr(win, "goto"):
                win.goto("nav.sft")
                win.pages["nav.sft"].set_checkpoint(p)

    def _delete(self):
        p = self._selected()
        if not p:
            return
        ok = QMessageBox.question(
            self, t("common.delete"),
            t("ck.delete.confirm", name=os.path.basename(p)),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            return
        try:
            os.remove(p)
            self.hint.setText("已删除 %s" % os.path.basename(p))
        except OSError as e:
            self.hint.setText("删除失败：%s" % e)
        self.refresh()
        self.ctx.checkpoints_changed.emit()


def _size(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024
    return "%.1f GB" % n
