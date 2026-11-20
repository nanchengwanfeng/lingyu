# -*- coding: utf-8 -*-
"""概览页：一屏回答「我这台机器现在能干什么」。

设计意图 —— 新手打开界面第一个看到的就是这一页，所以它必须做到：

1. **不说黑话**。不出现「n_layer」「d_model」，写「层数」「宽度」。
2. **给结论，不只给数字**。「本机跑得动」比「估算 1.57 GB / 可用 6.87 GB」
   更重要 —— 后者作为补充小字放在下面。
3. **下一步永远可见**。四步流程卡片，做完一步亮一步，用户不会迷路。
"""

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget)

from ..i18n import t
from ..theme import SPACE, th
from ..widgets import (Card, KeyValue, StatTile, ghost_button, muted)
from . import BasePage

# 第一屏的小贴士。每一条都是「不看就会踩」的实际经验，
# 不是凑数的鼓励话。
_TIPS = [
    "别训几步就来看效果 —— 100 步像乱码、1000 步开始像中文、5000 步以上才像样。",
    "预训练教它「会写中文」，SFT 才教它「会答话」。两步都要做。",
    "显存不够先调小批大小（batch_size），再考虑减层数；改宽度对参数量的影响最大。",
    "13M 的模型 SFT 之后答不对事实是正常的 —— 它学的是回答的格式，不是知识。",
    "续训前别改模型结构（层数/宽度），否则接不上；改了就用「从头重训」。",
]


class StepCard(QWidget):
    """快速开始里的一个步骤：序号圆点 + 标题 + 说明。"""

    def __init__(self, num, title, desc, done=False, parent=None):
        super().__init__(parent)
        self.setObjectName("CardFlat")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        lay.setSpacing(SPACE["md"])

        self.dot = QLabel(str(num))
        self.dot.setFixedSize(26, 26)
        self.dot.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.dot, 0, Qt.AlignTop)

        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = QLabel(title)
        self.title.setWordWrap(True)
        col.addWidget(self.title)
        self.desc = muted(desc, wrap=True)
        col.addWidget(self.desc)
        lay.addLayout(col, 1)
        self.set_done(done)

    def set_done(self, done):
        if done:
            self.dot.setText("✓")
            self.dot.setStyleSheet(
                "background:%s; color:%s; border-radius:13px; font-weight:700;"
                % (th.SUCCESS, th.BG))
            self.title.setStyleSheet("color:%s; font-weight:600;" % th.TEXT)
        else:
            self.dot.setText(self.dot.text() if self.dot.text() != "✓" else "•")
            self.dot.setStyleSheet(
                "background:%s; color:%s; border-radius:13px; font-weight:700;"
                % (th.SURFACE_2, th.TEXT_MUTED))
            self.title.setStyleSheet("color:%s;" % th.TEXT_DIM)


class OverviewPage(BasePage):
    nav_key = "nav.overview"
    icon = "◎"

    def header(self):
        return (t("ov.title"), t("ov.desc"))

    def _build_content(self):
        # 顶部工具行：环境检测时间 + 重新检测（显存会变，不能逼用户重开界面）
        bar = QHBoxLayout()
        bar.setSpacing(SPACE["sm"])
        self.env_ts_label = muted("")
        bar.addWidget(self.env_ts_label, 1)
        self.refresh_btn = ghost_button(t("ov.refresh"))
        self.refresh_btn.clicked.connect(self._refresh_env_now)
        bar.addWidget(self.refresh_btn)
        self.add(self._wrap(bar))
        self.ctx.env_ready.connect(lambda *_: self._refresh_env())

        # ---------------- 第一行：四个关键数字 ----------------
        row = QHBoxLayout()
        row.setSpacing(SPACE["md"])
        self.tile_device = StatTile(t("ov.device"))
        self.tile_vram = StatTile(t("ov.vram"))
        self.tile_tokens = StatTile(t("ov.data.tokens"))
        self.tile_params = StatTile(t("ov.model.params"))
        for w in (self.tile_device, self.tile_vram,
                  self.tile_tokens, self.tile_params):
            row.addWidget(w, 1)
        self.add(row if isinstance(row, QWidget) else self._wrap(row))

        # ---------------- 第二行：环境 / 语料 / 模型 ----------------
        cols = QHBoxLayout()
        cols.setSpacing(SPACE["lg"])

        self.env_card = Card(t("ov.env"))
        self.env_kv = KeyValue()
        self.env_card.add(self.env_kv)
        cols.addWidget(self.env_card, 1)

        self.data_card = Card(t("ov.data"))
        self.data_kv = KeyValue()
        self.data_card.add(self.data_kv)
        self.data_btn = ghost_button(t("ov.data.prepare"))
        self.data_btn.clicked.connect(self._goto_corpus)
        self.data_card.add(self.data_btn)
        cols.addWidget(self.data_card, 1)

        self.model_card = Card(t("ov.model"))
        self.model_kv = KeyValue()
        self.model_card.add(self.model_kv)
        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.model_card.add(self.model_note)
        self.launch_btn = ghost_button(t("ov.model.launch"))
        self.launch_btn.clicked.connect(self._launch)
        self.launch_btn.setVisible(False)
        self.model_card.add(self.launch_btn)
        cols.addWidget(self.model_card, 1)

        self.add(self._wrap(cols))

        # ---------------- 第三行：四步流程 ----------------
        flow = Card(t("ov.quick"))
        self.steps = [
            StepCard(1, t("ov.step1"), t("ov.step1b")),
            StepCard(2, t("ov.step2"), t("ov.step2b")),
            StepCard(3, t("ov.step3"), t("ov.step3b")),
            StepCard(4, t("ov.step4"), t("ov.step4b")),
        ]
        grid = QGridLayout()
        grid.setSpacing(SPACE["sm"])
        for i, s in enumerate(self.steps):
            grid.addWidget(s, 0, i)
            grid.setColumnStretch(i, 1)
        flow.add_layout(grid)
        self.add(flow)

        # ---------------- 第四行：小贴士 ----------------
        # 这一块不是为了填满空白，而是因为下面这几条恰好是新手最容易
        # 花几个小时才明白的事。放在第一屏，比藏在文档里有用得多。
        tips = Card("小贴士")
        for text in _TIPS:
            row = QLabel("·  " + text)
            row.setWordWrap(True)
            row.setStyleSheet("color:%s;" % th.TEXT_DIM)
            tips.add(row)
        self.add(tips)

        self.add_stretch()

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        lay.setContentsMargins(0, 0, 0, 0)
        return w

    def build(self):
        # 这一页全是「固定高度」的内容（数字、卡片、小贴士），
        # 没有哪个元素会主动让位。窗口一矮 Qt 就把它们压到一起 ——
        # 所以整体放进滚动区。放在 build 的**最开头**，
        # 这样后面 add 的东西都进滚动区里的 body。
        self.use_scroll_area()
        self._build_content()

    # ------------------------------------------------------------ 刷新
    def on_show(self):
        self.refresh()

    def on_task_event(self, name, code=None):
        self.refresh()

    def refresh(self):
        self._refresh_env()
        self._refresh_data()
        self._refresh_model()
        self._refresh_steps()

    def _refresh_env_now(self):
        self.refresh_btn.setEnabled(False)
        self.env_ts_label.setText(t("ov.detected", ts="…"))
        self.ctx.refresh_env()

    def _refresh_env(self):
        info = _env_info()
        try:
            self.refresh_btn.setEnabled(True)
            if getattr(self.ctx, "gpu_ts", 0):
                self.env_ts_label.setText(t("ov.detected", ts=time.strftime(
                    "%H:%M:%S", time.localtime(self.ctx.gpu_ts))))
        except Exception:
            pass
        self.tile_device.set(info["device_short"], accent=th.ACCENT)
        self.tile_vram.set(info["vram"], sub=info["vram_sub"])
        self.env_kv.clear()
        self.env_kv.add("PyTorch", info["torch"])
        self.env_kv.add("Python", info["python"])
        self.env_kv.add(t("ov.device"), info["device"])
        try:
            from lingyu.backends import list_backends
            impl = [b["label"] for b in list_backends() if b["implemented"]]
            self.env_kv.add("后端", "、".join(impl))
        except Exception:
            pass

    def _refresh_data(self):
        dd = self.ctx.data_dir()
        meta = _read_json(os.path.join(dd, "train.meta.json")) if dd else None
        if meta:
            n = meta.get("n_tokens", 0)
            self.tile_tokens.set("{:,}".format(n), accent=th.SUCCESS)
            self.data_kv.clear()
            self.data_kv.add(t("ov.data.tokens"), "{:,}".format(n))
            self.data_kv.add(t("ov.data.vocab"), "{:,}".format(
                meta.get("vocab_size", 0)))
            comp = meta.get("compression")
            if comp:
                self.data_kv.add(t("ov.data.comp"),
                                 "%.3f 字符/token" % float(comp))
            files = meta.get("corpus_files") or []
            self.data_kv.add(t("ov.data.files"), "%d 个" % len(files))
            self.data_btn.setVisible(False)
        else:
            self.tile_tokens.set("—", sub=t("ov.data.none"))
            self.data_kv.clear()
            self.data_kv.add(t("ov.data"), t("ov.data.none"))
            self.data_btn.setVisible(True)

    def _refresh_model(self):
        """模型估算。**这里绝不能 import torch**（会让界面启动慢几秒）。"""
        from ..context import ctx as _ctx
        try:
            est = estimate_model(self.ctx.config, _ctx.vram_free())
        except Exception as e:
            self.model_kv.clear()
            self.model_kv.add("错误", str(e))
            return
        if not est:
            return
        # 配置不合法：把错误摆出来，一个数字都不给
        if est.get("errors"):
            self.tile_params.set("—")
            self.model_kv.clear()
            self.model_kv.add("配置有问题", "先改好再看估算")
            self.model_note.setText(
                "⚠ " + "；".join(est["errors"])[:160])
            self.model_note.setStyleSheet("color:%s; font-weight:600;" % th.DANGER)
            self.launch_btn.setVisible(False)
            return
        self.tile_params.set(est["params_h"], accent=th.ACCENT)
        self.model_kv.clear()
        self.model_kv.add(t("ov.model.params"), est["params_h"])
        self.model_kv.add(t("ov.model.kv"), est["kv"])
        self.model_kv.add(t("ov.model.mem"), est["mem"])
        fits = est["fits"]
        if fits is None:
            # 显存还没探测到，别急着下结论
            self.model_note.setText("… 正在检测显存")
            self.model_note.setStyleSheet("color:%s;" % th.TEXT_MUTED)
            self.launch_btn.setVisible(False)
        elif fits:
            self.model_note.setText("✓ " + t("ov.model.fits"))
            self.model_note.setStyleSheet("color:%s; font-weight:600;" % th.SUCCESS)
            self.launch_btn.setVisible(False)
        else:
            need = est["cards_needed"] or "?"
            self.model_note.setText("⚠ " + t("ov.model.notfits", n=need))
            self.model_note.setStyleSheet("color:%s; font-weight:600;" % th.WARNING)
            self.launch_btn.setVisible(True)

    def _refresh_steps(self):
        has_data = self.ctx.has_corpus()
        cks = self.ctx.list_checkpoints()
        has_ckpt = bool(cks)
        sft_dir = self.ctx.sft_out_dir()
        has_sft = bool(os.path.isdir(sft_dir) and [
            f for f in os.listdir(sft_dir) if f.endswith(".pt")])
        self.steps[0].set_done(has_data)
        self.steps[1].set_done(has_ckpt)
        self.steps[2].set_done(has_sft)
        self.steps[3].set_done(has_sft)

    # ------------------------------------------------------------ 动作
    def _goto_corpus(self):
        win = self.window()
        if hasattr(win, "goto"):
            win.goto("nav.corpus")

    def _launch(self):
        win = self.window()
        if hasattr(win, "goto"):
            win.goto("nav.train")


# ==========================================================================
# 取值辅助（都不 import torch）
# ==========================================================================
def _env_info():
    """环境信息 —— **从 ctx.gpu 缓存里取，绝不在这里 import torch**。

    踩过的坑：原来这里直接 `import torch`，结果 MainWindow 构造时
    第一页的 on_show() 就把 torch 拉进来了，窗口要等 3~8 秒才出现。
    现在探测在后台线程做（`ctx.start_env_probe()`），这里只读结果。
    """
    from ..context import ctx
    g = ctx.gpu
    if g.get("probing"):
        return {"device": "检测中…", "device_short": "…", "vram": "检测中…",
                "vram_sub": "", "torch": "检测中…", "python": g.get("python", "")}
    vram = g.get("vram_text", "—")
    sub = ""
    if g.get("vram_total"):
        sub = "共 %.1f GB" % (g["vram_total"] / 1024 ** 3)
    return {
        "device": g.get("device", "—"),
        "device_short": g.get("device_short", "—"),
        "vram": "—" if g.get("probing") else vram,
        "vram_sub": sub,
        "torch": g.get("torch", "未安装"),
        "python": g.get("python", ""),
    }


def _read_json(path):
    import json
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return None


def estimate_model(cfg, free_bytes=0):
    """按配置估算模型大小。

    ## 参数

        cfg         配置字典
        free_bytes  本机可用显存（字节）。**必须由调用方传真实值** ——
                    传 0 表示「还不知道」，这时不给「装不装得下」的结论。

    ## 返回

    正常时是一个含 `params_h` / `mem` / `fits` / `cards_needed` 的字典；
    **配置本身不合法时返回 `{"errors": [...]}`**，调用方必须先看这个。

    ## ★ 不合法就不给估算（用户实测踩到的）

    用户把 `n_embd` 填成 230、`n_head` 填成 8（230 不能被 8 整除），
    界面照样给出一版「✓ 显存够用（约占 80%）」的漂亮估算 ——
    可这个配置**连模型都建不出来**，引擎会直接报
    「n_embd (230) 必须能被 n_head (8) 整除」。

    根因：这里用 `d // h` 整除算了个 head_dim，把不合法的宽度**静默取整**了，
    然后基于一个不存在的模型算了一堆数字。

    现在先跑引擎自己的校验器（`config/validate.py`，纯 Python 不碰 torch），
    有错就**把错误原样交出去，一个数字都不给**。
    """
    from lingyu.backends import estimate, fmt_bytes
    from lingyu.config import get, validate

    # 先校验，再估算。校验器是**同一份**（引擎和界面共用），
    # 所以界面上不会再出现「界面说没问题、命令行一跑就报错」。
    #
    # ⚠️ 必须 `scale_notice=False`：默认会算「本机装不装得下」，
    #    而那一步要探测显存 -> `import torch` -> 界面启动慢几秒。
    #    （曾经漏了这个参数，被 test_gui_does_not_import_torch 当场拦下。）
    try:
        errors, warnings, _notices = validate(cfg, scale_notice=False)
    except Exception as e:                       # 校验器自己炸了也不能白屏
        return {"errors": ["配置校验失败：%s" % e], "warnings": []}
    if errors:
        return {"errors": errors, "warnings": warnings}

    L = int(get(cfg, "model", "n_layer", 8) or 8)
    d = int(get(cfg, "model", "n_embd", 512) or 512)
    blk = int(get(cfg, "model", "block_size", 512) or 512)
    bs = int(get(cfg, "train", "batch_size", 8) or 8)

    # ★ 参数量 / KV cache 统一走 validate.estimate_params 那一份公式（含 MoE 分支）。
    #   旧代码在这里自己复制了一份，而且没有 MoE 分支 —— 概览页和 `lingyu plan`
    #   会对同一个配置给出两个数字（审计 B-14）。现在只保留一个事实源。
    from lingyu.config.validate import estimate_params
    params, _active, kv_b = estimate_params(cfg)
    if params is None:
        return {"errors": ["参数量估算失败：配置里的模型字段缺失或类型不对"],
                "warnings": warnings}

    est = estimate(params, batch=bs, seq=blk, d_model=d, n_layer=L,
                   world_size=max(1, int(get(cfg, "parallel",
                                              "tensor_parallel", 1) or 1)),
                   zero_stage=int(get(cfg, "parallel", "zero_stage", 0) or 0),
                   cpu_offload=bool(get(cfg, "parallel", "cpu_offload", False)),
                   activation_checkpoint=bool(get(cfg, "parallel",
                                                   "activation_checkpoint", False)),
                   moe=bool(get(cfg, "model", "use_moe", False)),
                   per_card_bytes=int(free_bytes) or None)
    total = float(est["total_bytes"])
    out = {
        "params": params,
        "params_h": "%.2f M" % (params / 1e6) if params < 1e9
                    else "%.2f B" % (params / 1e9),
        "kv": fmt_bytes(kv_b),
        "mem": fmt_bytes(total),
        # 原始字节数也带出来：调用方要拿它自己再判一次「装不装得下」
        "mem_bytes": total,
        "free_bytes": int(free_bytes),
        # 显存都不知道在哪儿，就别给结论。
        # `cards_needed = None` 表示「不知道」，和「1 张就够」是完全不同的意思。
        "cards_needed": est.get("cards_needed"),
        "errors": [],
        "warnings": warnings,
    }
    # 装不装得下：直接用估算器的结论（已含 FITS_MARGIN 与 USABLE_RATIO）。
    # 不再自己另算一份 —— 当年「238% 还说够用」就是两套判断漂移导致的。
    if not free_bytes or free_bytes <= 0:
        out["fits"] = None
    else:
        out["fits"] = est.get("fits_one_card")
    return out



