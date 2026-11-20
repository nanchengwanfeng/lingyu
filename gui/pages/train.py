# -*- coding: utf-8 -*-
"""训练页：改参数 → 看估算 → 开跑。

## 布局的考虑

左边是参数表单（按常用/进阶/专家分三页），右边上半是**显存估算**，
下半是**命令行预览 + 启动按钮**。

估算放在参数旁边而不是另开一页，是因为「改一个数字 → 立刻看到显存变了多少」
这个反馈循环非常重要。新手把层数从 8 调到 32，如果看不到显存从 1.5G 涨到 6G，
他就会直接点开始，然后 OOM。

## 关于「命令行预览」

下面那行灰色的命令不是装饰。用户在这个界面上点的一切，最终都会变成一条
`lingyu ...` 命令；把它显示出来有两个好处：
1. 用户能学会命令行的用法，慢慢脱离界面
2. 出问题时他能把这条命令复制出来求助，而不是说「界面报错了」
"""

import os
import subprocess
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QInputDialog, QLabel, QMessageBox,
                               QProgressBar, QTabWidget,
                               QVBoxLayout, QWidget)

from ..i18n import t
from ..paramform import SectionForm
from ..runner import Runner
from ..theme import SPACE, th
from ..widgets import (Card, KeyValue, LogView, StatTile, danger_button,
                       ghost_button, make_check, muted, primary_button)
from . import BasePage


class TrainPage(BasePage):
    nav_key = "nav.train"
    icon = "▶"

    def header(self):
        return (t("tr.title"), t("tr.desc"))

    def build(self):
        self.runner = Runner(self)
        # 任务输出走 ctx 总线：页面被重建后（换语言/换主题）也收得到（审计 1.7）
        self.ctx.task_line.connect(self._on_line)
        self.runner.finished.connect(self._on_finished)

        cols = QHBoxLayout()
        cols.setSpacing(SPACE["lg"])

        # ================= 左：参数 =================
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(SPACE["md"])

        self.form_card = Card()
        head = QHBoxLayout()
        self.cfg_label = muted("")
        head.addWidget(self.cfg_label, 1)
        # ★ 配置文件下拉框：用户要「自己训练」时，第一件事往往是换一份配置
        #   （chat.ini 13M / chat_medium.ini 47.8M / small / medium …）。
        #   以前界面只认 gui.json 里钉死的那一份，换配置得去手改 JSON —— 这里补上。
        self.config_combo = QComboBox()
        # 主题 QSS 给 QComboBox 写了 min-width: 70px（全局生效），这里要更宽
        # 才能显示完整配置名，所以对这一个控件用内联样式覆盖 min-width。
        self.config_combo.setStyleSheet("QComboBox { min-width: 168px; }")
        self.config_combo.setMinimumWidth(200)
        self.config_combo.setMinimumContentsLength(18)
        self.config_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.config_combo.setToolTip(t("tr.config.pick"))
        try:
            from lingyu import paths as _paths
            _cdir = _paths.configs_dir()
            _paths_list = []
            if os.path.isdir(_cdir):
                for _name in sorted(n for n in os.listdir(_cdir)
                                    if n.endswith(".ini")):
                    _paths_list.append((_name, os.path.abspath(
                        os.path.join(_cdir, _name))))
            _cur = (os.path.abspath(self.ctx.config_path)
                    if self.ctx.config_path else "")
            _cur_norm = os.path.normcase(_cur)
            if _cur and all(os.path.normcase(p) != _cur_norm
                            for _n, p in _paths_list):
                # 当前配置在出厂配置目录之外（例如打包版 runtime\configs）：
                # 放一个「当前」项，保证下拉框选中状态和实际载入的文件一致。
                self.config_combo.addItem(
                    "%s（当前）" % os.path.basename(_cur), _cur)
                self.config_combo.setItemData(
                    self.config_combo.count() - 1, _cur, Qt.ToolTipRole)
            for _name, _full in _paths_list:
                self.config_combo.addItem(_name, _full)
                self.config_combo.setItemData(
                    self.config_combo.count() - 1, _full, Qt.ToolTipRole)
            # 最后一项：不加载任何配置文件，回到 spec 默认值（用户要「自定义」）
            self.config_combo.addItem(t("tr.custom"), "")
        except OSError:
            pass
        self.config_combo.currentIndexChanged.connect(self._on_config_pick)
        head.addWidget(self.config_combo)
        self.reload_btn = ghost_button(t("tr.load"))
        self.reload_btn.clicked.connect(self._reload)
        head.addWidget(self.reload_btn)
        self.est_btn = ghost_button(t("tr.estimate"))
        self.est_btn.clicked.connect(self._estimate)
        head.addWidget(self.est_btn)
        self.saveas_btn = ghost_button(t("tr.saveas"))
        self.saveas_btn.clicked.connect(self._save_as)
        head.addWidget(self.saveas_btn)
        self.form_card.add_layout(head)

        self.model_form = SectionForm("model", "[model] 模型", self.ctx.config.get("model"),
                                      parent=self)
        self.model_form.changed.connect(self._schedule_estimate)
        self.train_form = SectionForm("train", "[train] 训练", self.ctx.config.get("train"),
                                      parent=self)
        self.train_form.changed.connect(self._schedule_estimate)
        self.parallel_form = SectionForm("parallel", "[parallel] 并行与显存",
                                         self.ctx.config.get("parallel"), parent=self)

        inner_tabs = QTabWidget()
        inner_tabs.setDocumentMode(True)
        for w, name in ((self.model_form, "模型结构"),
                        (self.train_form, "训练参数"),
                        (self.parallel_form, "并行/省显存")):
            page = QWidget()
            v = QVBoxLayout(page)
            v.setContentsMargins(0, SPACE["sm"], 0, 0)
            v.addWidget(w)
            inner_tabs.addTab(page, name)
        self.form_card.add(inner_tabs)
        # 常用页只有 4 项，下面会空出一大块。与其让它空着，
        # 不如告诉用户「还有更多参数在哪儿」—— 那正是他会问的问题。
        self.more_hint = muted(t("tr.tier.hint"), wrap=True)
        self.form_card.add(self.more_hint)
        lv.addWidget(self.form_card, 1)

        cols.addWidget(left, 3)

        # ================= 右：估算 + 启动 =================
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(SPACE["md"])

        # ★ 估算卡片的内容**高度不固定** —— 装不下时会多出一段「省显存的办法」，
        #   大配置下能有七八行。不给它滚动区的话，Qt 会把整张卡片压到最小尺寸
        #   以下：键值行互相重叠、建议文字被裁掉。
        #
        #   实测截图抓到过：360M 配置下「合计」和「可用显存」两行叠在一起，
        #   建议只露出第一条。
        #
        #   注意：滚动区只包**估算卡片**，不包「开始训练」那一块 ——
        #   按钮必须永远看得见，不能滚出去。
        self.est_card = Card(t("tr.est.title"))

        # 两块内容互斥显示：正常时是「数字 + 结论 + 明细」，
        # 配置不合法时**只显示错误**，不给任何数字。
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.error_label.setStyleSheet(
            "color:%s; font-weight:600; background:%s; border:1px solid %s;"
            "border-radius:8px; padding:10px;"
            % (th.DANGER, th.SURFACE_2, th.DANGER))
        self.error_label.setVisible(False)
        self.est_card.add(self.error_label)

        self.est_body = QWidget()
        eb = QVBoxLayout(self.est_body)
        eb.setContentsMargins(0, 0, 0, 0)
        eb.setSpacing(SPACE["sm"])

        # 2 列网格：1440×900（用户主窗口尺寸）下右栏只有约 250 逻辑像素，
        # 三块挤一行时「47.80 M」会被迫折行。改成 2+1 后每块有 120px 左右，
        # 14pt 的数字放得下；耗时块横跨整行，字号也不浪费。
        grid = QGridLayout()
        grid.setSpacing(SPACE["sm"])
        self.tile_params = StatTile(t("tr.est.params"), compact=True)
        self.tile_mem = StatTile(t("tr.est.total"), compact=True)
        # 耗时用一个「横条」而不是方块：1440×900 下右栏只有约 250 逻辑像素，
        # 三块方块放一行时数字必被折行。参数/显存两列并排，耗时横跨整行、
        # 数字与标题同一行，省下约 30px 垂直空间给下面的明细区。
        self.tile_time = QFrame()
        self.tile_time.setObjectName("CardFlat")
        hb = QHBoxLayout(self.tile_time)
        hb.setContentsMargins(SPACE["sm"], SPACE["xs"], SPACE["sm"], SPACE["xs"])
        hb.setSpacing(SPACE["sm"])
        self.time_value = QLabel("—")
        self.time_value.setStyleSheet("font-size:14pt; font-weight:600;")
        self.time_label = QLabel(t("tr.est.time"))
        self.time_label.setObjectName("StatLabel")
        hb.addWidget(self.time_value)
        hb.addStretch(1)
        hb.addWidget(self.time_label)
        # 估算是分档粗估，误差可达数倍 —— 别让用户当成精确值（审计 G-10）
        self.tile_time.setToolTip(t("tr.est.time") + "：粗估，仅参照（±数倍）")
        grid.addWidget(self.tile_params, 0, 0)
        grid.addWidget(self.tile_mem, 0, 1)
        grid.addWidget(self.tile_time, 1, 0, 1, 2)
        eb.addLayout(grid)

        # 结论**紧跟在数字下面、永远可见** —— 它是这张卡片存在的理由。
        # 一开始把整张卡都塞进滚动区，结果结论被滚到了看不见的地方，
        # 用户得往下滚才知道「装不装得下」。那比压扁还糟。
        self.est_note = QLabel("")
        self.est_note.setWordWrap(True)
        eb.addWidget(self.est_note)

        # 只有「明细 + 省显存建议」这一块是可变的，放进滚动区
        self.est_detail = QWidget()
        dv = QVBoxLayout(self.est_detail)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(SPACE["md"])
        self.est_kv = KeyValue()
        dv.addWidget(self.est_kv)
        self.suggest_label = muted("", wrap=True)
        self.suggest_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        dv.addWidget(self.suggest_label)
        dv.addStretch(1)
        detail_scroll = self.scrollable(self.est_detail)
        detail_scroll.setMinimumHeight(56)
        eb.addWidget(detail_scroll, 1)

        self.est_card.add(self.est_body, 1)
        # 估算卡吃掉右栏剩余高度：它的明细本来就可滚动，
        # 放不下时滚动比把其它卡挤扁更好。
        # ★ 必须有最小高度：窗口矮 / 运行卡变高时，Qt 会把这张卡压到
        #   两个 tile 的 60px 之下，结果数字互相叠住（截图抓到过）。
        #   宁可整页滚动，也不能让数字重叠。
        # 估算卡 + 看板卡一起放进「可滚动舞台」：
        # 窗口变矮 / 日志展开时，这两张卡按自然高度排列、整列滚动，
        # 绝不把 tile 压到互相叠住（截图抓到过 360M 配置下的叠字）。
        # 运行卡与按钮放在滚动区**外面**：按钮永远看得见。
        self.stage = QWidget()
        sv = QVBoxLayout(self.stage)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(SPACE["md"])
        sv.addWidget(self.est_card)
        sv.addStretch(1)

        # ---- 运行看板（审计 1.1/1.3）----
        # 训练时用户真正关心的是「到哪一步了、loss 多少、速度多少」，
        # 而不是一整屏滚动的终端文字。把关键数字提出来放大，
        # 原始日志默认收起 —— 界面第一眼不再是 PowerShell。
        self.dash_card = Card(t("tr.dash"))

        dash_top = QHBoxLayout()
        dash_top.setSpacing(SPACE["sm"])
        self.dash_state = QLabel(t("tr.dash.idle"))
        self.dash_state.setStyleSheet(
            "color:%s; font-weight:600; font-size:10pt;" % th.TEXT_DIM)
        dash_top.addWidget(self.dash_state)
        dash_top.addStretch(1)
        self.dash_meta = muted("")
        dash_top.addWidget(self.dash_meta)
        # 日志开关放在看板标题行：入口永远看得见，又不占运行卡片的高度
        self.log_toggle = ghost_button(t("tr.log.show"))
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(self._toggle_log)
        dash_top.addWidget(self.log_toggle, 0, Qt.AlignVCenter)
        self.dash_card.add_layout(dash_top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.dash_card.add(self.progress)

        grid = QGridLayout()
        grid.setSpacing(SPACE["sm"])
        self.dash_step = StatTile(t("tr.dash.step"), "0 / —",
                                  accent=th.TEXT_MUTED, compact=True)
        self.dash_loss = StatTile(t("tr.dash.loss"), "0.0000",
                                  accent=th.TEXT_MUTED, compact=True)
        self.dash_lr = StatTile(t("tr.dash.lr"), "0.00e+00",
                                accent=th.TEXT_MUTED, compact=True)
        self.dash_speed = StatTile(t("tr.dash.speed"), "0", sub="tok/s",
                                   accent=th.TEXT_MUTED, compact=True)
        for i, w in enumerate((self.dash_step, self.dash_loss,
                               self.dash_lr, self.dash_speed)):
            grid.addWidget(w, i // 2, i % 2)
        self.dash_card.add_layout(grid)
        sv.addWidget(self.dash_card)
        sv.addStretch(1)
        rv.addWidget(self.scrollable(self.stage), 1)

        self.run_card = Card()

        # 续训起点：用户必须能选「从哪个断点续」，不能永远赌 out_dir/ckpt.pt
        rrow = QHBoxLayout()
        rrow.setSpacing(SPACE["sm"])
        rrow.addWidget(QLabel(t("tr.resume.src")))
        self.resume_combo = QComboBox()
        self.resume_combo.setMinimumWidth(170)
        self.resume_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.resume_combo.currentIndexChanged.connect(self._on_resume_pick)
        rrow.addWidget(self.resume_combo, 1)
        self.run_card.add_layout(rrow)
        self.resume_hint = muted("", wrap=True)
        self.run_card.add(self.resume_hint)

        btns = QHBoxLayout()
        self.start_btn = primary_button(t("tr.start"))
        self.start_btn.clicked.connect(lambda: self._start(resume=False))
        btns.addWidget(self.start_btn, 2)
        self.resume_btn = ghost_button(t("tr.resume"))
        self.resume_btn.clicked.connect(lambda: self._start(resume=True))
        btns.addWidget(self.resume_btn, 1)
        self.stop_btn = danger_button(t("tr.stop"))
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setVisible(False)
        btns.addWidget(self.stop_btn, 1)
        self.run_card.add_layout(btns)

        # 「从头重训」这个勾选框是必须的 —— 下面那句提示里提到了它，
        # 而且没有它就没法在界面上覆盖已有存档（命令行是 --force）。
        # 曾经提示写着「或勾选从头重训」而界面上根本没有这个框。
        self.force_check = make_check(t("tr.force"), False)
        self.run_card.add(self.force_check)

        self.status_label = muted("", wrap=True)
        self.run_card.add(self.status_label)
        rv.addWidget(self.run_card)

        self.log_card = Card(t("mo.log"))
        # 「即将执行」的命令行也收进详细区：命令行不该是界面主角（1.4），
        # 但排障时它是第一条要看的信息 —— 所以不删、只藏。
        self.cmd_label = QLabel("")
        self.cmd_label.setWordWrap(True)
        self.cmd_label.setStyleSheet(
            "color:%s; font-family:Consolas,monospace; font-size:10pt;"
            % th.TEXT_MUTED)
        self.log_card.add(self.cmd_label)
        self.log = LogView()
        self.log.setMinimumHeight(120)
        self.log_card.add(self.log)
        self.log_card.setVisible(False)
        rv.addWidget(self.log_card, 2)

        right.setMinimumWidth(330)
        cols.addWidget(right, 2)
        self.add(self._wrap(cols))

        # 进度数字统一走全局总线：页面别重建了/切走了也照样更新
        self.ctx.task_progress.connect(self._on_progress)
        self._refresh_state()
        self._estimate()
        self._set_dash_idle()

    def _wrap(self, lay):
        w = QWidget()
        w.setLayout(lay)
        return w

    # ------------------------------------------------------------ 估算
    def _schedule_estimate(self):
        """参数一改就重算，但**不立刻**算 —— 拖数字框会触发几百次。

        延迟 250ms 合并成一次，用户感觉是「实时」的，但不卡。
        """
        from PySide6.QtCore import QTimer
        if not hasattr(self, "_est_timer"):
            self._est_timer = QTimer(self)
            self._est_timer.setSingleShot(True)
            self._est_timer.setInterval(250)
            self._est_timer.timeout.connect(self._estimate)
        self._est_timer.start()

    def _collect(self):
        """把表单里的值合并进配置字典。"""
        cfg = self.ctx.config
        cfg.setdefault("model", {}).update(
            {k: v for k, v in self.model_form.values().items()})
        cfg.setdefault("train", {}).update(
            {k: v for k, v in self.train_form.values().items()})
        cfg.setdefault("parallel", {}).update(
            {k: v for k, v in self.parallel_form.values().items()})
        return cfg

    def _estimate(self):
        from .overview import estimate_model
        try:
            cfg = self._collect()
            free = self.ctx.vram_free()
            est = estimate_model(cfg, free)
        except Exception as e:
            self._show_config_error(["估算失败：%s" % e])
            return

        # ★ 配置本身不合法 —— 一个数字都不给，直接摆错误。
        #   踩过的坑：n_embd=230 配 n_head=8（除不尽）时，
        #   界面照样显示「✓ 显存够用」，而引擎一跑就报错。
        if est.get("errors"):
            self._show_config_error(est["errors"])
            return
        self._show_config_error(None)

        self._est = est
        self._est_free = free

        self.tile_params.set(est["params_h"], accent=th.ACCENT)
        # 「装不下」用警告色、「够用」用成功色、「不知道」用中性色 ——
        # 三种状态必须能一眼分开
        fits = est["fits"]
        self.tile_mem.set(est["mem"],
                          accent=(th.SUCCESS if fits is True
                                  else (th.WARNING if fits is False
                                        else th.TEXT_DIM)))
        self.time_value.setText(self._time_hint())

        self.est_kv.clear()
        self.est_kv.add(t("tr.est.params"), est["params_h"])
        self.est_kv.add(t("ov.model.kv"), est["kv"])
        self.est_kv.add(t("tr.est.total"), est["mem"])
        if free:
            self.est_kv.add(t("tr.est.avail"), "%.2f GB" % (free / 1024 ** 3))

        if fits is None:
            # 显存还没探测到 —— 不能瞎给结论（这正是「238% 还说够用」的根源）
            self.est_note.setText("… 正在检测显存，稍等一下")
            self.est_note.setStyleSheet("color:%s;" % th.TEXT_MUTED)
            self.suggest_label.setText("")
            self.suggest_label.setVisible(False)
        elif fits:
            pct = 100.0 * est["mem_bytes"] / max(1, free)
            self.est_note.setText("✓ " + t("tr.est.ok", pct="%.0f" % pct))
            self.est_note.setStyleSheet("color:%s; font-weight:600;" % th.SUCCESS)
            self.suggest_label.setText("")
            self.suggest_label.setVisible(False)
        else:
            need = est["cards_needed"] or _cards_needed(est, free)
            pct = 100.0 * est["mem_bytes"] / max(1, free)
            self.est_note.setText(
                "⚠ %s（估算占可用显存的 %.0f%%）"
                % (t("tr.est.over", n=need), pct))
            self.est_note.setStyleSheet("color:%s; font-weight:600;" % th.WARNING)
            tips = _tips(self._collect(), need)
            self.suggest_label.setText(
                t("tr.est.suggest") + "：\n" + _format_tips(tips))
            self.suggest_label.setVisible(True)

    def _show_config_error(self, errors):
        """把「配置不合法」和「估算结果」两种状态切开显示。

        合法时传 None，恢复到正常的数字 + 结论。
        """
        bad = bool(errors)
        self.error_label.setVisible(bad)
        self.est_body.setVisible(not bad)
        if bad:
            self.error_label.setText(
                "⚠ 配置有问题，先改好再训练：\n"
                + "\n".join("  · " + e for e in errors))
        # 开始训练按钮也要禁用 —— 参数不合法时点下去只会得到一个报错
        if hasattr(self, "start_btn"):
            self._config_ok = not bad
            self.start_btn.setEnabled(self._config_ok
                                      and self.ctx.corpus_status()[0]
                                      and not self.ctx.is_busy())

    def _time_hint(self):
        cfg = self._collect()
        try:
            from lingyu.config import get
            bs = int(get(cfg, "train", "batch_size", 8) or 8)
            ga = int(get(cfg, "train", "grad_accum", 1) or 1)
            ms = int(get(cfg, "train", "max_steps", 1000) or 1000)
            blk = int(get(cfg, "model", "block_size", 512) or 512)
        except Exception:
            return "—"
        toks = bs * ga * blk * ms
        # 按参数量分档估 token/s（粗估：真实速度还取决于 GPU/精度/并行，
        # 误差可达数倍）。以前写死 50000，360M 和 13M 显示同一个时间（G-10）。
        params = int((getattr(self, "_est", {}) or {}).get("params") or 0)
        tps = _tokens_per_sec(params) if params else 50000.0
        sec = toks / tps
        if sec < 90:
            eta = "%.0f 秒" % sec
        elif sec < 5400:
            eta = "%.0f 分钟" % (sec / 60)
        else:
            eta = "%.1f 小时" % (sec / 3600)
        # 前缀「≈」明示这是粗略值，不是承诺
        return "≈" + eta

    # ------------------------------------------------------------ 启动
    def _build_args(self, resume=False):
        path = self.ctx.config_path or self.ctx.ensure_config_file()
        args = ["train", "--config", path, "--json"]
        if resume:
            # resume 可以是 True（=续 out_dir/ckpt.pt，CLI 的 auto）、
            # 具体断点文件路径（审计 G-03），或 ("init", path)：
            # 那个断点属于别的 run，优化器状态对不上，只能当初始化权重
            # （= 用户在 SFT 之后接着预训练，审计 B-03）。
            if isinstance(resume, tuple) and resume and resume[0] == "init":
                args += ["--init-from", resume[1]]
            elif isinstance(resume, str):
                args += ["--resume", resume]
            else:
                args.append("--resume")
        if getattr(self, "force_check", None) is not None and \
                self.force_check.isChecked():
            args.append("--force")
        return args

    def _cmd_preview(self, args=None):
        """完整命令行预览（Windows 引号规则，路径带空格也能直接复制）。"""
        args = list(self._build_args() if args is None else args)
        return subprocess.list2cmdline(["lingyu"] + args)

    def _refresh_state(self):
        self.cfg_label.setText("%s: %s" % (
            t("tr.config"), _short(self.ctx.config_path)))
        self.cfg_label.setToolTip(self.ctx.config_path or "")
        # 下拉框与当前配置保持一致（「载入…」/「另存为」之后也要跟上）
        try:
            cur = os.path.normcase(os.path.abspath(self.ctx.config_path or ""))
            for i in range(self.config_combo.count()):
                item = self.config_combo.itemData(i)
                if item and os.path.normcase(os.path.abspath(item)) == cur:
                    if self.config_combo.currentIndex() != i:
                        self.config_combo.blockSignals(True)
                        self.config_combo.setCurrentIndex(i)
                        self.config_combo.blockSignals(False)
                    break
        except Exception:
            pass
        args = self._build_args()
        preview = self._cmd_preview(args)
        self.cmd_label.setText(t("tr.willrun") + "  " + preview)
        self.cmd_label.setToolTip(preview)
        has_data, ddir, why = self.ctx.corpus_status()
        cks = self.ctx.list_checkpoints()
        # 配置不合法时，开始训练必须禁用（_show_config_error 会置 _config_ok）
        ok = getattr(self, "_config_ok", True)
        self.start_btn.setEnabled(has_data and ok and not self.ctx.is_busy())
        self.resume_btn.setEnabled(bool(cks) and ok and not self.ctx.is_busy())
        if not ok:
            self.status_label.setText("⚠ 配置有问题，见右边 —— 改好才能开始训练")
            self.status_label.setStyleSheet("color:%s;" % th.DANGER)
        elif not has_data:
            # ★ 一定要说清「查的是哪个目录」。只写「还没准备语料」的话，
            #   用户刚在语料页看到「准备完成」时会彻底懵 ——
            #   而真正的原因往往是两个目录都合法，只是**不是同一个**。
            self.status_label.setText(
                "⚠ %s\n     查的是：%s\n     （%s）"
                % (t("tr.no_data"), _short(ddir) or "—", why))
            self.status_label.setStyleSheet("color:%s;" % th.WARNING)
        elif cks:
            # 断点提示已经由「续训起点」那一行说清了，这里不再重复占一行高度。
            self.status_label.setText("")
            self.status_label.setStyleSheet("color:%s;" % th.TEXT_MUTED)
        else:
            self.status_label.setText("")

    def _start(self, resume=False):
        # 先把界面上的值写回配置文件 —— 否则子进程读到的还是旧配置，
        # 用户会以为「我在界面上改了但没生效」
        try:
            if not self.ctx.config_path:
                self.ctx.ensure_config_file()
            self._collect()
            if self.model_form.form.dirty_keys() or \
                    self.train_form.form.dirty_keys() or \
                    self.parallel_form.form.dirty_keys():
                self.ctx.save_config()
                self.status_label.setText("已保存配置到 %s" % _short(
                    self.ctx.config_path))
        except Exception as e:
            self.log.append_line("保存配置失败：%s" % e, "error")
            return

        if resume:
            # 续训前先对齐结构：用户在多份配置间切换是常态，结构对不上
            # 时直接让引擎报错太粗暴；先问一句，点「是」就按断点自动对齐。
            try:
                resume = self._prepare_resume(resume)
            except Exception as e:
                self.log.append_line("准备续训失败：%s" % e, "error")
                self.status_label.setText("⚠ %s" % e)
                self.status_label.setStyleSheet("color:%s;" % th.DANGER)
                return

        if self.runner.start(self._build_args(resume)):
            self.ctx.begin_task("train")
            self.ctx.attach_runner("train", self.runner)
            self._set_running(True)
            self._dash_begin()

    def _stop(self):
        # 先问全局总线：页面重建过之后，自己的 runner 不是正在跑的那个
        if not self.ctx.stop_current_task():
            self.runner.stop()

    def _set_running(self, running):
        self.start_btn.setVisible(not running)
        self.resume_btn.setVisible(not running)
        self.stop_btn.setVisible(running)
        self.est_btn.setEnabled(not running)
        self.reload_btn.setEnabled(not running)

    # ------------------------------------------------------------ 运行看板
    def _toggle_log(self, checked):
        self.log_card.setVisible(bool(checked))
        self.log_toggle.setText(t("tr.log.hide") if checked else t("tr.log.show"))

    def _tokens_per_step(self):
        """每一步训练吃多少 token —— 进度行里没有 token，只能从配置算。

        block_size 在 [model] 段，不在 [train] 段；一开始从 train 里读，
        `get(..., 1)` 静默拿到 1，于是每一步的 token 数被算成「batch×累积」，
        速度数字整体少了 block_size 倍（审计 1.2 同类坑）。
        """
        try:
            cfg = self._collect()
            tr = cfg.get("train") or {}
            md = cfg.get("model") or {}
            n = (int(tr.get("batch_size", 1)) *
                 int(tr.get("grad_accum", 1)) *
                 int(md.get("block_size", 1)))
            return max(1, n)
        except Exception:
            return 1

    def _set_dash_idle(self):
        """看板的静止状态：不清空上一次的数字，只把「状态」变灰。"""
        self.dash_state.setText(t("tr.dash.idle"))
        self.dash_state.setStyleSheet(
            "color:%s; font-weight:600; font-size:10pt;" % th.TEXT_DIM)
        self._dash_active = False

    def _dash_begin(self):
        self._dash_active = True
        self._dash_t0 = time.monotonic()
        self._dash_last = None            # (step, monotonic)
        self._dash_speed = 0.0
        try:
            max_steps = int((self._collect().get("train") or {})
                            .get("max_steps", 0) or 0)
        except Exception:
            max_steps = 0
        self._dash_total = max_steps
        self.progress.setRange(0, max_steps or 1000)
        self.progress.setValue(0)
        try:
            self.progress.setFormat("%p%")
        except Exception:
            pass
        self.dash_step.set("0 / %s" % (_num(max_steps) if max_steps else "—"),
                           accent=th.TEXT_MUTED)
        self.dash_loss.set("0.0000", accent=th.TEXT_MUTED)
        self.dash_lr.set("0.00e+00", accent=th.TEXT_MUTED)
        self.dash_speed.set("0", sub="tok/s", accent=th.TEXT_MUTED)
        self.dash_state.setText(t("tr.dash.running"))
        self.dash_state.setStyleSheet(
            "color:%s; font-weight:600; font-size:10pt;" % th.ACCENT)
        self.dash_meta.setText(t("tr.dash.elapsed", elapsed="00:00"))

    def _on_progress(self, name, data):
        """进度行 → 看板数字。

        速度用「相邻两次进度的时间差」现算，再做一个 0.3 的 EMA ——
        单次测量会被 eval/存盘那一步拖慢，直接显示会一跳一跳。
        """
        if name != "train" or not isinstance(data, dict):
            return
        if not getattr(self, "_dash_active", False):
            self._dash_begin()
        step = int(data.get("step") or 0)
        total = int(data.get("total") or 0) or self._dash_total
        tps = self._tokens_per_step()

        self.progress.setRange(0, total or 1000)
        self.progress.setValue(step)
        self.dash_step.set(
            "%s / %s" % (_num(step), _num(total) if total else "—"),
            sub=t("tr.dash.tokens_per_step", n=_num(tps)),
            accent=th.ACCENT)

        loss = data.get("loss")
        if loss is not None:
            self.dash_loss.set("%.4f" % float(loss), accent=th.TEXT)
        lr = data.get("lr")
        if lr is not None:
            self.dash_lr.set("%.2e" % float(lr), accent=th.TEXT_DIM)

        now = time.monotonic()
        last = getattr(self, "_dash_last", None)
        if last is not None and step > last[0] and now - last[1] > 0.05:
            inst = (step - last[0]) * tps / (now - last[1])
            self._dash_speed = (inst if self._dash_speed <= 0
                                else 0.3 * inst + 0.7 * self._dash_speed)
        self._dash_last = (step, now)
        if self._dash_speed <= 0:
            # 第一次进度还没有「上一次」可比，直接用引擎上报的 tok_s
            try:
                self._dash_speed = float(data.get("tok_s") or 0.0)
            except (TypeError, ValueError):
                self._dash_speed = 0.0

        if self._dash_speed > 0:
            self.dash_speed.set(_num(int(self._dash_speed)), sub="tok/s",
                                accent=th.SUCCESS)

        # 用时优先用引擎上报的真实 elapsed（含 eval/存盘时间），没有才用墙钟
        elapsed = data.get("elapsed")
        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            elapsed = now - getattr(self, "_dash_t0", now)
        parts = [t("tr.dash.elapsed", elapsed=_fmt_dur(elapsed))]
        if total and self._dash_speed > 0:
            remain = (total - step) * tps / self._dash_speed
            parts.append(t("tr.dash.eta", eta=_fmt_dur(remain)))
        if step > 0:
            parts.append(t("tr.dash.perstep",
                           sec="%.2fs" % (elapsed / step)))
        # epoch：用引擎给的 tokens/步 和数据集 token 数（老日志没有就跳过）
        tps_real = data.get("tokens_per_step") or tps
        ds_tokens = data.get("dataset_tokens")
        try:
            if ds_tokens and int(ds_tokens) > 0:
                cur = step * float(tps_real) / int(ds_tokens)
                whole = (float(total) * float(tps_real) / int(ds_tokens)
                         if total else cur)
                parts.append(t("tr.dash.epoch",
                               cur="%.2f" % cur, total="%.2f" % whole))
        except (TypeError, ValueError):
            pass
        self.dash_meta.setText("  ·  ".join(parts))

    def _dash_end(self, code, stopped):
        self._dash_active = False
        if stopped:
            text, color = t("tr.dash.stopped"), th.WARNING
        elif code == 0:
            text, color = t("tr.dash.done"), th.SUCCESS
        else:
            text, color = t("tr.dash.failed"), th.DANGER
        self.dash_state.setText(text)
        self.dash_state.setStyleSheet(
            "color:%s; font-weight:600; font-size:10pt;" % color)

    def _on_line(self, name, text, level):
        if name != "train":
            return
        self.log.append_line(text, level)

    def _on_finished(self, code, stopped):
        self.ctx.end_task("train", code)
        self._set_running(False)
        self._dash_end(code, stopped)
        self._refresh_state()
        if stopped:
            # 用户主动停止：引擎按停止协议保存断点后退出。
            # 不管退出码是什么，都不能报「训练失败」（审计 G-04）。
            self.log.append_line("已停止（断点已保存）", "ok")
            self.ctx.settings["out_dir"] = self.ctx.out_dir()
            self.ctx.save()
            self.ctx.checkpoints_changed.emit()
        elif code == 0:
            self.log.append_line("训练结束", "ok")
            self.ctx.settings["out_dir"] = self.ctx.out_dir()
            self.ctx.save()
            self.ctx.checkpoints_changed.emit()
        else:
            self.log.append_line("训练失败，退出码 %d" % code, "error")

    def _reload(self):
        try:
            cfg = self.ctx.load_config()
        except Exception as e:
            self.log.append_line("读配置失败：%s" % e, "error")
            return
        self.model_form.set_values(cfg.get("model"))
        self.train_form.set_values(cfg.get("train"))
        self.parallel_form.set_values(cfg.get("parallel"))
        self._refresh_state()
        self._refresh_resume_combo()
        self._estimate()

    def _on_config_pick(self, _index=None):
        """下拉框换配置：载入 → 记住（写进 gui.json）→ 刷新表单。"""
        path = self.config_combo.currentData()
        # path == "" 是「自定义」项：不载入任何配置文件，用 spec 默认值
        if path is None:
            return
        if getattr(self.ctx, "running_task", None):
            self.log.append_line("任务运行中，不能切换配置", "warn")
            return
        try:
            if path == "":
                self.ctx.load_custom_config()
            else:
                self.ctx.load_config(path)
        except Exception as e:
            self.log.append_line("读配置失败：%s" % e, "error")
            return
        if path == "":
            self.ctx.settings.pop("config", None)
        else:
            self.ctx.settings["config"] = self.ctx.config_path
        self.ctx.save()
        self._reload()

    # ------------------------------------------------------------ 配置/续训
    def _save_as(self):
        """把当前内存里的配置另存为 configs/<name>.ini。"""
        name, ok = QInputDialog.getText(
            self, t("tr.saveas.title"), t("tr.saveas.name"), text="my.ini")
        if not ok or not name.strip():
            return
        name = name.strip()
        if not name.endswith(".ini"):
            name += ".ini"
        try:
            from lingyu import paths
            self._collect()
            path = os.path.join(paths.configs_dir(), name)
            self.ctx.save_config(self.ctx.config, path)
            self.ctx.load_config(path)
            self.ctx.settings["config"] = path
            self.ctx.save()
            self._reload()
            self.status_label.setText(t("tr.saveas.ok", path=path))
            self.status_label.setStyleSheet("color:%s;" % th.SUCCESS)
        except Exception as e:
            self.status_label.setText(t("tr.saveas.err", e=e))
            self.status_label.setStyleSheet("color:%s;" % th.DANGER)

    def _refresh_resume_combo(self):
        """列出可续训的断点：自动项 + 本项目/out 根下各 run 的 ckpt/best。"""
        combo = getattr(self, "resume_combo", None)
        if combo is None:
            return
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(t("tr.resume.auto"), "auto")
        try:
            for f in self.ctx.all_checkpoints()[:40]:
                label = "%s · 第 %s 步" % (f["name"], _num(f.get("step")))
                combo.addItem(label, f["path"])
                combo.setItemData(combo.count() - 1, f["path"], Qt.ToolTipRole)
            if cur not in (None, "auto"):
                # 从断点页「续训」跳过来时，把选中的那个文件也放进列表
                if all(combo.itemData(i) != cur for i in range(combo.count())):
                    combo.addItem("%s · 选中" % os.path.basename(cur), cur)
                for i in range(combo.count()):
                    if combo.itemData(i) == cur:
                        combo.setCurrentIndex(i)
                        break
        except Exception:
            pass
        combo.blockSignals(False)
        self._on_resume_pick()

    def _on_resume_pick(self, _index=None):
        combo = getattr(self, "resume_combo", None)
        if combo is None or not hasattr(self, "resume_hint"):
            return
        data = combo.currentData()
        if data in (None, "auto"):
            self.resume_hint.setText(t("tr.resume.hintauto"))
        else:
            meta = self.ctx.read_ckpt_meta(data) or {}
            step = meta.get("step")
            self.resume_hint.setText(
                t("tr.resume.hint", step=_num(step) if step is not None else "?"))
            # 这里只显示提示，真正决定 --resume / --init-from 在 _prepare_resume

    def _model_diffs(self, snapshot):
        """断点快照 vs 当前配置：返回 [(配置键, 当前值, 断点值), ...]。"""
        cur = self._collect().get("model") or {}
        pairs = (("n_layer", "n_layer"), ("n_embd", "d_model"),
                 ("n_head", "n_head"), ("n_kv_head", "n_kv_head"),
                 ("block_size", "block_size"), ("d_ff", "d_ff"),
                 ("tie_weights", "tie_weights"), ("vocab_size", "vocab_size"),
                 ("use_moe", "use_moe"))
        out = []
        for cfg_key, snap_key in pairs:
            if snap_key not in snapshot or snapshot[snap_key] is None:
                continue
            want = snapshot[snap_key]
            have = cur.get(cfg_key)
            if have is None:
                continue
            try:
                if float(have) == float(want):
                    continue
            except (TypeError, ValueError):
                if have == want:
                    continue
            out.append((cfg_key, have, want))
        return out

    def _apply_alignment(self, diffs):
        cur = self.ctx.config.setdefault("model", {})
        for key, _have, want in diffs:
            cur[key] = want
        self.ctx.save_config()
        for form in (self.model_form,):
            try:
                form.set_values(self.ctx.config.get("model"))
            except Exception:
                pass
        self.status_label.setText(t("tr.aligned"))
        self.status_label.setStyleSheet("color:%s;" % th.SUCCESS)

    def _prepare_resume(self, resume):
        """决定续训方式：结构对不上就对齐；断点不在本项目 out_dir 就退化成 init。"""
        if resume is True and getattr(self, "resume_combo", None) is not None:
            data = self.resume_combo.currentData()
            resume = True if data in (None, "auto") else data
        if not isinstance(resume, str):
            return resume
        meta = {}
        try:
            meta = self.ctx.read_ckpt_meta(resume)
        except Exception:
            meta = {}
        snapshot = (meta or {}).get("model_cfg") or {}
        if snapshot:
            diffs = self._model_diffs(snapshot)
            if diffs:
                labels = "、".join("%s %s→%s" % (k, have, want)
                                  for k, have, want in diffs)
                answer = QMessageBox.question(
                    self, t("tr.resume.align.title"),
                    t("tr.resume.mismatch", labels=labels),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if answer != QMessageBox.Yes:
                    raise RuntimeError("已取消续训（请先手动把模型结构改成断点的）")
                self._apply_alignment(diffs)
        # 断点目录与本项目 out_dir 不同：优化器状态不能跨 run 复用，
        # 只能把它当初始化权重（= SFT 之后继续预训练的正确姿势）。
        if os.path.normcase(os.path.dirname(os.path.abspath(resume))) != \
                os.path.normcase(os.path.abspath(self.ctx.out_dir())):
            return ("init", resume)
        return resume

    # ------------------------------------------------------------ 生命周期
    def on_show(self):
        self._refresh_state()
        self._refresh_resume_combo()
        self._estimate()
        # ★ 任务在后台跑时切回本页：按钮状态必须恢复成「运行中」，
        #   否则「停止」按钮消失、训练却停不下来（审计 1.7）。
        if self.ctx.running_task == "train":
            self._set_running(True)
            if not getattr(self, "_dash_active", False):
                self._dash_begin()

    def on_task_event(self, name, code=None):
        self._set_running(self.ctx.running_task == "train")
        if name == "train" and code is not None and \
                getattr(self, "_dash_active", False):
            self._dash_end(
                code, getattr(self.ctx, "last_finished_stopped", False))
        if code is not None:
            self._refresh_state()


def _num(n):
    """12345 → 12,345（看板里的步数/token 数）。"""
    try:
        return format(int(n), ",")
    except (TypeError, ValueError):
        return str(n)


def _fmt_dur(sec):
    """秒 → mm:ss / h:mm:ss。看板上不显示小数，训练本来就以分钟计。"""
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return "%d:%02d:%02d" % (h, m, s)
    return "%02d:%02d" % (m, s)


def _format_tips(tips):
    """把建议排版成适合**窄栏**的样子。

    `backends.plan.suggestions()` 返回的是「说明： --set xxx --set yyy」这种
    一行式（命令行里显示正好）。但这一栏只有 300 多像素宽，
    一行式会折成两三行、`--set` 被切在中间，很难读。

    所以把命令拎出来单独缩进一行：

        ① 换数据并行分片（FSDP），把优化器状态切开：
             --set parallel.backend=fsdp --set parallel.zero_stage=2
    """
    import re
    out = []
    for tip in tips:
        m = re.search(r"\s{2,}(--set|LingYu|lingyu)", tip)
        if m:
            head, tail = tip[:m.start()].rstrip(), tip[m.start():].strip()
            out.append(head + "\n      " + tail)
        else:
            out.append(tip)
    return "\n".join(out)


def _cards_needed(est, free):
    """估算器没给卡数时，自己按「总量 ÷ 单卡可用」估一个。

    用来兜底：宁可给一个粗略的整数，也不能显示「够用」。
    """
    if not free:
        return "?"
    import math
    return max(2, int(math.ceil(est["mem_bytes"] / (free * 0.85))))


# 训练速度分档（token/s，仅用于「心里有个数」的粗估）
_TPS_TIERS = (
    (50e6, 50000.0),
    (200e6, 20000.0),
    (1e9, 6000.0),
    (float("inf"), 1500.0),
)


def _tokens_per_sec(params):
    """按参数量粗分档估 token/s。params=0 表示未知，调用方自己兜底。"""
    for limit, tps in _TPS_TIERS:
        if params < limit:
            return tps
    return _TPS_TIERS[-1][1]


def _short(path, n=52):
    if not path:
        return "—"
    return path if len(path) <= n else "…" + path[-(n - 1):]


def _tips(cfg, need):
    from lingyu.backends import suggestions
    from lingyu.config import get
    d = int(get(cfg, "model", "n_embd", 512) or 512)
    L = int(get(cfg, "model", "n_layer", 8) or 8)
    fake = {"cards_needed": need, "cpu_offload": bool(
        get(cfg, "parallel", "cpu_offload", False))}
    return suggestions(cfg.get("parallel", {}), fake, 0, n_layer=L, d_model=d)
