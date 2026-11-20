# -*- coding: utf-8 -*-
"""GUI 冒烟测试。

## 这一组测试要守住什么

1. **界面绝不 import torch**。这是启动速度的生命线（见 `gui/__init__.py`）。
   一旦有人图省事在模块级 import torch，8GB 的机器上打开界面就要等 5 秒。
   这条规矩只有靠测试才守得住。

2. **九页都能建出来**。某个页面的 `build()` 里写错一个名字，
   用户点进去才会看到报错 —— 而那是他最难截图反馈的时刻。
   这里把九页全建一遍。

3. **主题/语言切换不崩**。这两条路径平时很少走，最容易留 bug。

## 为什么不用 pytest-qt

不想为了几个测试再引一个依赖。直接用 `QT_QPA_PLATFORM=offscreen`
建 QApplication 就够了 —— 我们要验的是「建得出来、不抛异常」，
不是「点下去有没有反应」。
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("PySide6", reason="GUI 是可选依赖，没装就跳过")

# 窗口构造 50ms 后会起后台线程 import torch 探测显卡。pytest 进程往往在
# 它跑完前就退出，Qt 在解释器收尾阶段回调会直接 0xC0000005（实测
# `pytest tests/test_gui.py` 全绿但退出码 -1073741819）。测试里关掉
# 自动探测；探针逻辑本身由 G-15 测试直接驱动，不依赖窗口。
os.environ.setdefault("LINGYU_GUI_NO_PROBE", "1")
# 切页淡入动效（140ms）在测试里没有意义，还会让 grab()/断言拍到半透明
# 的一帧。关掉后 goto() 同步完成 —— 测试要的是最终状态。
os.environ.setdefault("LINGYU_GUI_NO_ANIM", "1")


@pytest.fixture(scope="module")
def qapp():
    # offscreen 平台：不弹窗口，CI 上也能跑
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qapp):
    from gui.app import MainWindow
    from gui.theme import stylesheet
    qapp.setStyleSheet(stylesheet())
    w = MainWindow()
    yield w
    w.close()


# ==================================================== 架构约束

def test_gui_does_not_import_torch_at_module_level():
    """★ 界面绝不能把 torch 拉进来 —— 这是「8GB 机器 2 秒打开」的前提。

    单开一个子进程测，避免别的测试已经把 torch 导进来了造成假通过。
    """
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import gui;"
        "from PySide6.QtWidgets import QApplication;"
        "app = QApplication([]);"
        "from gui.app import MainWindow;"
        "w = MainWindow();"
        "print('TORCH' if 'torch' in sys.modules else 'CLEAN')"
        % ROOT
    )
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "-c", code], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=180, cwd=ROOT)
    out = p.stdout.decode("utf-8", "replace")
    assert "CLEAN" in out, (
        "构造主窗口时把 torch 导进来了 —— 界面启动会慢好几秒。\n"
        "检查是不是哪个页面在模块级或 on_show() 里直接 import torch，"
        "正确做法是走 ctx.start_env_probe() 让后台线程去导。\n"
        "子进程输出：%s" % (out + p.stderr.decode("utf-8", "replace")))


# ==================================================== 页面构建

def test_all_pages_build(window):
    """九页全建一遍，任何一页抛异常都要立刻暴露。"""
    errors = []
    for _cls, nav_key, _g, _i in window.page_specs:
        try:
            window.goto(nav_key)
        except Exception as e:
            errors.append("%s: %s" % (nav_key, e))
    assert not errors, "这些页面建不出来：\n  " + "\n  ".join(errors)


def test_every_page_declares_nav_key(window):
    for _cls, nav_key, group_key, _i in window.page_specs:
        assert nav_key.startswith("nav.")
        assert group_key.startswith("nav.group.")


def test_sidebar_has_button_for_every_page(window):
    for _cls, nav_key, _g, _i in window.page_specs:
        assert nav_key in window.sidebar.buttons, nav_key


def test_goto_switches_stack(window):
    first = window.page_specs[0][1]
    last = window.page_specs[-1][1]
    window.goto(last)
    assert window.stack.currentWidget() is window._holders[last]
    window.goto(first)
    assert window.stack.currentWidget() is window._holders[first]


def test_goto_unknown_key_is_ignored(window):
    """传一个不存在的 key 不能崩 —— 将来加页面时容易传错。"""
    before = window.stack.currentWidget()
    window.goto("nav.does_not_exist")
    assert window.stack.currentWidget() is before


# ==================================================== 主题

def test_both_themes_produce_valid_stylesheet(qapp):
    from gui.theme import stylesheet, th
    for name in ("dark", "light"):
        th.set(name)
        qss = stylesheet()
        assert len(qss) > 3000, name
        assert "{" in qss and "}" in qss
        qapp.setStyleSheet(qss)
    th.set("dark")


def test_theme_has_all_semantic_colors():
    """两套主题的**色名必须完全一致**。

    少一个色名时，页面代码里 `th.XXX` 会 AttributeError ——
    而且只在切到那套主题时才炸，很难发现。
    """
    from gui.theme import DARK, LIGHT
    missing_dark = set(LIGHT) - set(DARK)
    missing_light = set(DARK) - set(LIGHT)
    assert not missing_dark, "深色主题缺这些色名：%s" % sorted(missing_dark)
    assert not missing_light, "浅色主题缺这些色名：%s" % sorted(missing_light)


def test_theme_values_are_hex_colors():
    from gui.theme import LIGHT
    for k, v in LIGHT.items():
        assert isinstance(v, str) and v.startswith("#"), "%s = %r" % (k, v)
        assert len(v) in (4, 7, 9), "%s = %r" % (k, v)


def test_unknown_theme_falls_back_to_dark():
    from gui.theme import Theme
    t = Theme("nonsense")
    assert t.name == "dark"


# ==================================================== 语言

def test_both_languages_cover_the_same_keys():
    """中英两份文案的 key 必须一一对应。

    英文表漏一个 key 时，界面在英文下会显示 `⟨key⟩` 这种占位符 ——
    用户看到的就是「界面没做完」。这条测试把漏翻挡在提交之前。
    """
    from gui.i18n import EN, ZH
    missing_en = set(ZH) - set(EN)
    missing_zh = set(EN) - set(ZH)
    assert not missing_en, "英文缺这些 key：%s" % sorted(missing_en)
    assert not missing_zh, "中文缺这些 key：%s" % sorted(missing_zh)


def test_missing_key_returns_placeholder_not_crash():
    from gui.i18n import t
    assert t("no.such.key.exists") == "⟨no.such.key.exists⟩"


def test_format_placeholders_match_between_languages():
    """带 {} 占位符的文案，中英两边的占位符名字必须一致。

    英文写成 `{count}` 而中文写 `{n}`，格式化时就会抛 KeyError
    （或者更糟：静默显示成 `{count}`）。
    """
    import re
    from gui.i18n import EN, ZH
    bad = []
    for k, zh in ZH.items():
        en = EN.get(k)
        if not en:
            continue
        pz = set(re.findall(r"\{(\w+)\}", zh))
        pe = set(re.findall(r"\{(\w+)\}", en))
        if pz != pe:
            bad.append("%s: 中文 %s vs 英文 %s" % (k, sorted(pz), sorted(pe)))
    assert not bad, "占位符不一致：\n  " + "\n  ".join(bad)


def test_set_lang_switches_and_ignores_garbage():
    from gui.i18n import get_lang, set_lang, t
    try:
        set_lang("en_US")
        assert get_lang() == "en_US"
        assert t("nav.chat") == "Chat"
        set_lang("zh_CN")
        assert t("nav.chat") == "聊天"
        set_lang("klingon")
        assert get_lang() == "zh_CN", "不认识的语言不该改掉当前语言"
    finally:
        set_lang("zh_CN")


def test_param_help_follows_language():
    from gui.i18n import param_help, set_lang
    from lingyu.config import SPEC
    p = SPEC["model"]["n_layer"]
    try:
        set_lang("zh_CN")
        assert param_help(p) == p["help_zh"]
        set_lang("en_US")
        assert param_help(p) == p["help_en"]
    finally:
        set_lang("zh_CN")


# ==================================================== 参数表单（自动生成）

def test_param_form_covers_every_spec_param():
    """★ spec 里的参数必须**一个不漏**地出现在表单里。

    这是「GUI 能控制几乎所有参数」这个需求的验收点。
    手写表单迟早会漏，所以这里直接拿 spec 对账。
    """
    from gui.paramform import ParamForm
    from lingyu.config import SPEC
    for sec, params in SPEC.items():
        form = ParamForm(sec)
        missing = [k for k in params if k not in form.widgets]
        assert not missing, "[%s] 这些参数没有出现在界面上：%s" % (sec, missing)


def test_param_form_round_trips_values():
    """★ 写进去再读出来必须**完全一样**。

    这条测试抓到了一个严重 bug：小数位数按参数名硬编码（learning_rate 用
    2 位），而 5e-4 在 2 位小数下变成 0.00 —— 取回来就是 0.0。
    后果是「打开训练页 → 什么都不改 → 点开始训练」会把学习率存成 0，
    模型一步都学不动，而且界面上看着一切正常。

    教训：**任何「显示层」都可能悄悄改掉数据**，所以每种类型都要能往返。
    """
    from gui.paramform import ParamForm
    from lingyu.config import SPEC
    for sec in SPEC:
        form = ParamForm(sec)
        for key, p in SPEC[sec].items():
            form.set_value(key, p["default"])
        vals = form.values()
        bad = []
        for key, p in SPEC[sec].items():
            got, exp = vals[key], p["default"]
            if isinstance(exp, bool):
                ok = bool(got) == exp
            elif isinstance(exp, int):
                ok = int(got) == exp
            elif isinstance(exp, float):
                # 浮点：只要量级对就行（显示位数有限，允许很小的相对误差）
                ok = abs(float(got) - exp) <= max(1e-12, abs(exp) * 1e-6)
            else:
                ok = str(got) == str(exp)
            if not ok:
                bad.append("[%s] %s: 期望 %r，取回 %r" % (sec, key, exp, got))
        assert not bad, "表单往返丢了值：\n  " + "\n  ".join(bad)


def test_small_float_params_are_not_rounded_to_zero():
    """★ 专钉「学习率被四舍五入成 0」这类 bug。

    这些参数的值都很小（1e-4 ~ 1e-6），是「固定小数位」写法的重灾区。
    单独列出来测，是为了让失败信息一眼能看懂是哪个参数。
    """
    from gui.paramform import ParamForm
    from lingyu.config import SPEC
    watch = [("train", "learning_rate"), ("train", "min_lr"),
             ("model", "norm_eps"), ("post", "learning_rate"),
             ("post", "lora_dropout"), ("model", "init_std")]
    bad = []
    for sec, key in watch:
        p = SPEC[sec][key]
        form = ParamForm(sec, only_keys=[key])
        form.set_value(key, p["default"])
        got = float(form.values()[key] or 0.0)
        exp = float(p["default"])
        if exp != 0 and got == 0:
            bad.append("%s.%s 默认 %r 被存成 0" % (sec, key, exp))
        elif exp != 0 and abs(got - exp) > abs(exp) * 1e-6:
            bad.append("%s.%s 期望 %r，实际 %r" % (sec, key, exp, got))
    assert not bad, "小数值参数被截断了：\n  " + "\n  ".join(bad)


def test_make_float_keeps_small_values():
    """★ `make_float` 必须先 setDecimals 再 setValue。

    QDoubleSpinBox 的 decimals 默认是 **2**。如果先 setValue(0.0005)，
    它会立刻把值四舍五入成 0.00、再被 minimum 顶到下限 ——
    界面上显示成 0.000001（也就是最小值），而用户以为看到的是配置值。

    SFT 页的学习率就是这么变成 0.000001 的（截图里一眼看出来的）。
    """
    from gui.widgets import make_float
    for val, lo, hi in ((0.0005, 1e-6, 1.0), (2e-4, 1e-6, 1.0),
                        (1e-6, 1e-6, 1.0), (0.9, 0.0, 5.0), (1.05, 1.0, 3.0)):
        w = make_float(lo, hi, val, 0.0001, 6)
        assert abs(w.value() - val) <= max(1e-12, abs(val) * 1e-9), (
            "make_float(%r) 取回来是 %r —— 值在设置过程中被四舍五入掉了"
            % (val, w.value()))


def test_param_form_shows_the_right_initial_value():
    """★ 表单**初始显示**的值必须就是配置里的值。

    这条和 round_trip 那条不一样：round_trip 是先 set_value 再读，
    而这里读的是**刚构造出来**的控件。踩过的坑是 make_float 里
    setDecimals 的顺序错了，导致初始值是错的，但 set_value 之后又对了 ——
    只测 round_trip 会漏掉这种 bug。
    """
    from gui.paramform import ParamForm
    from lingyu.config import SPEC
    bad = []
    for sec in SPEC:
        form = ParamForm(sec)
        for key, p in SPEC[sec].items():
            if not isinstance(p["default"], float):
                continue
            got = form.values()[key]
            exp = float(p["default"])
            if exp != 0 and abs(float(got) - exp) > abs(exp) * 1e-6:
                bad.append("%s.%s 初始显示 %r，应该是 %r" % (sec, key, got, exp))
    assert not bad, "表单初始值不对：\n  " + "\n  ".join(bad)


def test_labels_are_not_clipped():
    """★ 标签不能被截断。

    踩过的坑：标签宽度用 `self.fontMetrics()` 在构造期间量，而 QSS 的
    字号要等控件 polish 之后才生效 —— 量到的是更小的默认字体，
    于是宽度算少了，「从哪个预训练断点开始」显示成「个预训练断点开始」。

    这里量的是**实际渲染需要**的宽度，而不是我算出来的那个数。
    """
    from PySide6.QtGui import QFontMetrics
    from gui.paramform import ParamForm
    from lingyu.config import SPEC

    bad = []
    for sec in SPEC:
        form = ParamForm(sec)
        form.ensurePolished()
        for key, f in form.fields.items():
            if f.label.text() == "":
                continue
            need = QFontMetrics(form.font()).horizontalAdvance(f.label.text())
            if f.label.width() < need:
                bad.append("[%s] %s：标签需要 %dpx，只给了 %dpx"
                           % (sec, key, need, f.label.width()))
    assert not bad, "这些标签会被截断：\n  " + "\n  ".join(bad)


def test_unify_labels_makes_controls_align():
    """同一组 Field 的标签列必须等宽 —— 否则控件左边缘参差不齐。"""
    from gui.paramform import ParamForm
    form = ParamForm("model", only_keys=["n_layer", "n_embd", "n_head",
                                         "block_size", "tie_weights"])
    widths = {k: f.label.width() for k, f in form.fields.items()
              if not f._top and f.label.text()}
    assert len(set(widths.values())) <= 1, (
        "同一组里标签宽度不一致：%s" % widths)


def test_unify_labels_falls_back_to_stacked_for_long_labels():
    """标签太长时整组改上下排法，而不是硬塞进固定宽度里截断。"""
    from gui.widgets import Field, unify_labels
    from PySide6.QtWidgets import QLineEdit
    long_label = "这是一个特别特别长的中文标签用来测试会不会被截断掉"
    fields = [Field("短", QLineEdit()), Field(long_label, QLineEdit())]
    unify_labels(fields)
    assert all(f._top for f in fields), "整组都该切到上下排法"
    assert fields[1].label.width() >= 0
    """三个分数页加起来覆盖全部参数，不能有参数掉进夹缝。"""
    from gui.paramform import ParamForm
    from lingyu.config import SPEC
    form = ParamForm("model")
    assert form.tabs.count() == 3, "应该是 常用/进阶/专家 三页"
    # 每个参数都归属到某一页（tier 合法），且都在 widgets 里
    tiers = {p.get("tier") for p in SPEC["model"].values()}
    assert tiers <= {"basic", "advanced", "expert"}, tiers
    assert len(form.widgets) == len(SPEC["model"])


def test_live_component_choices_come_from_registry():
    """组件下拉框的选项要**从注册表现取**，这样插件注册的名字能出现。"""
    from gui.paramform import ParamForm
    from lingyu import registry
    form = ParamForm("model", only_keys=["norm"])
    combo = form.widgets["norm"]
    names = set(registry.list_names("norm"))
    shown = {combo.itemData(i) for i in range(combo.count())}
    assert names <= shown, "注册表里有 %s，界面上只显示了 %s" % (names, shown)
    assert "" in shown, "应该有一个「自动」的空选项"


# ==================================================== 运行器

def test_runner_parses_progress_line():
    """训练进度行的解析 —— 曲线和数字面板全靠它。"""
    from gui.runner import _RE_SFT, _RE_VAL
    line = "    第   100/10000 步  loss 4.1234  lr 6.00e-04  |g| 1.23  0:00:12"
    m = _RE_SFT.search(line)
    assert m, "解析不了这行：%s" % line
    assert m.group(1) == "100" and m.group(2) == "10000"
    assert m.group(3) == "4.1234"
    v = _RE_VAL.search("  [验证] 第 500 步  验证 loss 3.9876  (困惑度 53.94)")
    assert v and v.group(1) == "500" and v.group(2) == "3.9876"


def test_runner_parses_checkpoint_line():
    from gui.runner import _RE_CKPT
    m = _RE_CKPT.search("  [存档] 第 1,000 步 → C:/path/to/lingyu/out/small/ckpt.pt")
    assert m, "解析不了存档行"
    assert m.group(1) == "1,000"
    assert m.group(2).endswith("ckpt.pt")


def test_runner_is_not_running_initially():
    from gui.runner import Runner
    r = Runner()
    assert r.is_running() is False
    r.stop()          # 没在跑时调用 stop 不能崩


# ==================================================== 子进程停止的报告

def test_chat_page_distinguishes_stop_from_crash():
    """★ 「用户主动停止」不能被报成「进程退出，码 1」。

    踩过的坑（用户当场撞上）：模型加载慢，用户又点了一下「加载模型」，
    代码把上一个进程 kill 掉。Windows 上被 kill 的进程退出码是 1，
    于是界面上出现两行「聊天进程退出，码 1」—— 看起来像崩溃。

    正确做法：停止前打标记，回调里先看标记。
    """
    src = open(os.path.join(ROOT, "gui", "pages", "chat.py"),
               encoding="utf-8").read()
    assert "_stopping = True" in src, "停止时没有打标记"
    assert "was_stopping" in src, "回调里没有检查标记"
    # 停止分支必须**先于**报错分支返回
    stop_at = src.find("if was_stopping:")
    err_at = src.find("聊天进程异常退出")
    assert stop_at != -1 and err_at != -1 and stop_at < err_at, (
        "应该先判断「是不是用户停的」，再决定要不要报错")


def test_chat_page_surfaces_subprocess_output_on_failure():
    """★ 子进程失败时必须把它**说过的话**打出来。

    踩过的坑：原来的日志过滤器只保留「以 [ 开头」或含「错误」的行，
    而 Python 的 `ModuleNotFoundError: No module named 'lingyu'`
    两条都不满足 —— 关键信息被静默吃掉，用户只看到一句「码 1」，
    完全没法排查。

    所以现在会留一份输出尾巴，失败时原样打出来。
    """
    src = open(os.path.join(ROOT, "gui", "pages", "chat.py"),
               encoding="utf-8").read()
    assert "_stderr_tail" in src, "没有保留子进程输出"
    assert "它最后说的话" in src, "失败时没有把输出打给用户"


# ==================================================== ★ 显存结论不能自相矛盾

def _cfg(**kw):
    """造一个配置字典（只给估算用得到的那几项）。"""
    base = {"model": {"vocab_size": 8192, "n_layer": 6, "n_embd": 384,
                      "n_head": 6, "n_kv_head": 2, "block_size": 256,
                      "d_ff": 0, "tie_weights": True},
            "train": {"batch_size": 8},
            "parallel": {"tensor_parallel": 1, "zero_stage": 0,
                         "activation_checkpoint": False}}
    for sec, vals in kw.items():
        base.setdefault(sec, {}).update(vals)
    return base


def test_big_config_is_reported_as_NOT_fitting():
    """★★ 用户截图里的那个 bug：16.37 GB 估算 / 6.87 GB 可用 = 238%，
    却写着「✓ 显存够用」。

    根因是估算函数收了个没有 `vram_free()` 的鸭子类型对象，
    可用显存恒为 0 -> 估算器不算 `cards_needed` -> 调用方
    `.get("cards_needed", 1)` 拿到默认值 1 -> **永远「够用」**。

    这条测试直接钉住「超预算就必须说装不下」。
    """
    from gui.pages.overview import estimate_model
    free = int(6.87 * 1024 ** 3)
    big = _cfg(model={"n_layer": 32, "n_embd": 1024, "n_head": 8})
    est = estimate_model(big, free)
    assert est["mem_bytes"] > free, "这个配置本来就该超预算，测试数据不对"
    assert est["fits"] is False, (
        "估算 %s > 可用 6.87 GB，却判成「装得下」" % est["mem"])
    assert est["cards_needed"] and est["cards_needed"] > 1, (
        "超预算时必须给出「需要几张卡」，现在是 %r" % est["cards_needed"])


def test_small_config_is_reported_as_fitting():
    from gui.pages.overview import estimate_model
    free = int(6.87 * 1024 ** 3)
    est = estimate_model(_cfg(), free)
    assert est["fits"] is True, "%s 应该装得下" % est["mem"]
    assert est["cards_needed"] == 1


def test_unknown_vram_gives_no_verdict():
    """★ 显存还没探测到时，**不许**给出任何结论。

    这是那个 bug 的另一半：结论不能来自「有默认值的字段」。
    不知道就老实说不知道。
    """
    from gui.pages.overview import estimate_model
    est = estimate_model(_cfg(), 0)
    assert est["fits"] is None, "不知道显存却说「%r」" % est["fits"]
    assert est["cards_needed"] is None, (
        "不知道显存却给出了卡数 %r" % est["cards_needed"])


def test_fits_never_contradicts_the_numbers():
    """★ 不变式：说「够用」时，估算值必须真的 ≤ 可用显存。

    遍历一批规模，任何一个出现「超了还说够用」就失败 ——
    这条是防「同一个错觉换个配置再来一次」。
    """
    from gui.pages.overview import estimate_model
    free = int(6.87 * 1024 ** 3)
    bad = []
    for n_layer, n_embd, n_head in ((6, 384, 6), (16, 256, 8), (12, 768, 12),
                                    (32, 1024, 8), (40, 2048, 16),
                                    (8, 512, 8), (2, 64, 2)):
        est = estimate_model(_cfg(model={"n_layer": n_layer, "n_embd": n_embd,
                                        "n_head": n_head}), free)
        if est["fits"] and est["mem_bytes"] > free:
            bad.append("%d层%d宽：估算 %s > 可用 6.87 GB 却说够用"
                       % (n_layer, n_embd, est["mem"]))
        if est["fits"] is False and est["mem_bytes"] <= free * 0.9:
            bad.append("%d层%d宽：估算 %s 明明放得下却说装不下"
                       % (n_layer, n_embd, est["mem"]))
    assert not bad, "\n  ".join(bad)


def test_train_page_does_not_default_cards_needed():
    """★ 训练页不许再用 `.get("cards_needed", 1)` 这种「缺失就当 1」。

    就是这一行让「238% 还说够用」的。改成先看三态的 `fits`。
    """
    src = open(os.path.join(ROOT, "gui", "pages", "train.py"),
               encoding="utf-8").read()
    assert 'get("cards_needed", 1)' not in src, (
        "又出现了「cards_needed 缺失就当 1」的写法 —— 这正是那个 bug 的根源")
    assert 'est["fits"]' in src, "没有用三态的 fits"


def test_invalid_config_gives_no_estimate():
    """★★ 配置不合法时**一个数字都不许给**。

    用户实测（第二次）：`n_embd=230` 配 `n_head=8` —— 230 不能被 8 整除，
    引擎会直接报「n_embd (230) 必须能被 n_head (8) 整除」，
    **可界面照样显示「✓ 显存够用（约占 80%）」**。

    根因：界面的估算函数用 `d // h` 整除算了个 head_dim，
    把不合法的宽度静默取整了，然后基于一个**根本不存在的模型**算了一堆数字。

    界面必须跑**引擎自己那份**校验器（`config/validate.py`），
    这样就不会再出现「界面说没问题、命令行一跑就报错」。
    """
    from gui.pages.overview import estimate_model
    bad = _cfg(model={"n_layer": 64, "n_embd": 230, "n_head": 8})
    est = estimate_model(bad, int(6.87 * 1024 ** 3))
    assert est.get("errors"), (
        "n_embd=230 / n_head=8 是不合法的，界面却给了估算：%r" % est)
    assert "整除" in " ".join(est["errors"]), est["errors"]
    # 关键：不能同时给出数字 —— 那会让人以为「能用，只是有点问题」
    assert "params_h" not in est, "不合法还给参数量"
    assert "fits" not in est, "不合法还给「装不装得下」的结论"


def test_invalid_config_of_every_kind_is_caught():
    """各种不合法组合都要被拦下，不只是除不尽那一种。"""
    from gui.pages.overview import estimate_model
    free = int(6.87 * 1024 ** 3)
    cases = {
        "除不尽": {"n_embd": 230, "n_head": 8},
        "每头奇数维": {"n_embd": 384, "n_head": 12},   # 384/12 = 32 偶数，合法
        "kv头比q头多": {"n_kv_head": 16, "n_head": 8},
        "q头不是kv头整数倍": {"n_head": 8, "n_kv_head": 3},
    }
    for name, kw in cases.items():
        base = {"n_layer": 4, "n_embd": 256, "n_head": 8, "n_kv_head": 2}
        base.update(kw)
        est = estimate_model(_cfg(model=base), free)
        if name == "每头奇数维":
            # 384/12=32，合法 —— 用它当反例，确认校验器不是「一律拒绝」
            assert not est.get("errors"), "%s 本来是合法的却被拒了：%r" % (
                name, est.get("errors"))
            continue
        assert est.get("errors"), "%s 这种不合法配置没被拦下：%r" % (name, est)


def test_gui_and_cli_use_the_same_validator():
    """界面和命令行必须用**同一份**校验逻辑。

    各写一份的话迟早会出现「界面说没问题、命令行报错」——
    用户会不知道信谁。
    """
    src = open(os.path.join(ROOT, "gui", "pages", "overview.py"),
               encoding="utf-8").read()
    assert "validate(cfg" in src, "界面没有调用引擎的校验器"
    # ★ 而且必须关掉规模提示 —— 那段要探测显存，会 import torch
    assert "scale_notice=False" in src, (
        "界面调 validate() 时没有 scale_notice=False —— 那会把 torch 拖进来，"
        "破坏「界面 2 秒内打开」")
    # 校验器本身必须是纯 Python（不能 import torch），否则界面用不了它
    v = open(os.path.join(ROOT, "src", "lingyu", "config", "validate.py"),
             encoding="utf-8").read()
    assert "def validate(cfg, strict=False, scale_notice=True)" in v, (
        "validate() 没有 scale_notice 开关，界面就没法在不加载 torch 的情况下用它")


def test_tiles_size_font_by_width_not_char_count():
    """StatTile 的字号要按**实际宽度**算，不能按字符数。

    按字符数分档时，「360.51 M」（8 字符）在概览页的宽格子里正好，
    到训练页的窄格子里就被挤成两行（「360.51」/「M」）。
    """
    from PySide6.QtGui import QFontMetrics
    from gui.widgets import StatTile
    tile = StatTile("x", "360.51 M")
    tile.resize(110, 60)
    tile._fit_font("360.51 M")
    css = tile.value_label.styleSheet()
    size = int(css.split("font-size:")[1].split("pt")[0])
    f = tile.value_label.font()
    f.setPointSize(size)
    avail = 110 - 2 * 8
    need = QFontMetrics(f).horizontalAdvance("360.51 M")
    assert need <= avail or size == StatTile.SIZES[-1], (
        "窄格子里字号 %dpt 放不下「360.51 M」（需要 %dpx，只有 %dpx）"
        % (size, need, avail))


# ==================================================== 语料路径（★ 真实事故回归）

def test_corpus_status_names_the_directory_it_checked(tmp_path):
    """★ 「还没准备语料」必须说清**查的是哪个目录**。

    真实事故：打包版读的配置指着 `runtime/app/data1`，而语料准备页
    把数据写在 `runtime/data`。两个目录都合法、都真实存在、谁都没报错，
    只是一个空一个满。

    这时界面上只有一句「还没准备语料」，用户只会说
    **「我明明刚准备完啊」** —— 然后两边都查不出问题。
    把目录名和原因一起显示出来，这种问题一眼就能看见。
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.context import Context
    QApplication.instance() or QApplication([])
    c = Context()
    dd = tmp_path / "empty_data"
    dd.mkdir()
    c._config = {"data": {"data_dir": str(dd)}}

    ok, where, why = c.corpus_status()
    assert ok is False
    assert where == str(dd), "没说清查的是哪个目录"
    assert why, "没说清为什么不算数"
    assert c.has_corpus() is False

    # 有 train.bin 才算数
    (dd / "train.bin").write_bytes(b"\x00\x00")
    ok, where, why = c.corpus_status()
    assert ok is True and why == ""


def test_corpus_status_says_when_the_directory_is_absent(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.context import Context
    QApplication.instance() or QApplication([])
    c = Context()
    missing = tmp_path / "nope"
    c._config = {"data": {"data_dir": str(missing)}}
    ok, where, why = c.corpus_status()
    assert ok is False and where == str(missing) and "不存在" in why


def test_gui_stores_an_absolute_config_path(tmp_path, monkeypatch):
    """★ 界面加载配置后，存回设置的必须是**绝对路径**。

    存相对路径的话，下次启动又要靠「根据某个根去猜」——
    而猜错的那个根（runtime/app）真实存在，错误毫无症状。
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.context import Context
    QApplication.instance() or QApplication([])
    cfg = tmp_path / "cfg" / "mine.ini"
    cfg.parent.mkdir()
    cfg.write_text("[model]\nn_layer = 4\n", encoding="utf-8")

    c = Context()
    c.load_config(str(cfg))
    assert os.path.isabs(c.settings["config"])
    assert os.path.normcase(c.settings["config"]) == os.path.normcase(str(cfg))


def test_factory_configs_load_through_the_gui_root():
    """出厂配置在界面里必须解析到**同一个**语料目录。

    这条把「配置里的相对路径按 home() 解析」和
    「六个出厂配置的 data_dir 一致」连起来验一遍。
    """
    from lingyu import paths
    from lingyu.config import get, load_config, resolve

    p = os.path.join(ROOT, "configs", "small.ini")
    cfg = resolve(load_config(p))
    dd = paths.abspath(get(cfg, "data", "data_dir"))
    assert dd == os.path.normpath(os.path.join(paths.home(), "data")), (
        "small.ini 的 data_dir 解析到了 %s —— 打包版下这必须落回 runtime/data" % dd)


# ==================================================== ★ 紧急修复回归（审计 D）
# 2026-10-17：停止语义 / 语言切换不杀任务 / 断点续训传路径

def test_runner_stop_cancels_fallback_kill_timer():
    """★ 用户点停止后必须：报告 stopped=True，且兜底强杀定时器被撤销。

    旧实现用 `QTimer.singleShot(4000, ...)`，定时器不可撤销 ——
    用户停止后 4 秒内重开任务，上一轮的定时器会把新进程误杀（审计 G-01）。
    """
    import sys
    from PySide6.QtCore import QProcess
    from gui.runner import Runner

    r = Runner()
    proc = QProcess()
    proc.start(sys.executable, ["-c", "import time; time.sleep(30)"])
    assert proc.waitForStarted(5000), "测试子进程没起来"

    r.proc = proc
    finished = []
    proc.finished.connect(r._on_finished)
    r.finished.connect(lambda code, stopped: finished.append((code, stopped)))

    r.stop(grace_ms=5000)
    assert r._stopping is True, "停止前没有打 _stopping 标记"
    assert r._kill_timer.isActive(), "停止后应该挂起兜底强杀定时器"
    # 模拟进程收到停止后正常退出（不跑事件循环，直接结束进程）
    proc.kill()
    assert proc.waitForFinished(5000), "子进程没有被终止"

    assert finished, "停止后没有发出 finished 回调"
    assert finished[-1][1] is True, "停止后的回调必须带 stopped=True，不能当崩溃"
    assert r._kill_timer.isActive() is False, (
        "进程已正常退出，兜底强杀定时器还活着 —— 它会误杀下一个任务")
    assert r._stopping_proc is None, "退出后没有清掉目标进程标记"


def test_stop_message_no_longer_promises_checkpoint():
    """★ runner 的停止提示必须如实：Windows 上 terminate 无法让子进程存断点。

    旧文案「会先存一次断点再退出」与行为相反，是用户可见的假承诺（审计 G-01）。
    """
    src = open(os.path.join(ROOT, "gui", "runner.py"), encoding="utf-8").read()
    assert ('self.line.emit("正在停止…… 会先存一次断点再退出"'
            not in src), "停止提示还在承诺存断点"
    assert "正在保存断点并退出" in src, "没有实现停止标记文件协议"
    assert "LINGYU_STOP_FILE" in src, "没有把停止标记路径传给子进程"
    assert "_kill_timer" in src, "兜底强杀没有做成可撤销的成员定时器"
    assert "grace_ms = int(grace_ms or 20000)" in src, (
        "grace_ms 默认不是 20 秒，或 clicked 的 False 没被兜住")


def test_app_reports_user_stopped_task_as_not_failed(window):
    """★ 用户主动停止后，侧边栏不能再显示「失败」（审计 G-04）。

    Runner 退出前会把 stopped 标记写进 ctx；app 优先按「已停止」显示。
    """
    from gui.context import ctx
    from gui.i18n import t

    old = getattr(ctx, "last_finished_stopped", False)
    try:
        ctx.last_finished_stopped = True
        window._on_task("train", 1)
        assert window.sidebar.state_label.text() != t("common.failed"), (
            "用户主动停止仍被显示成「失败」")
    finally:
        ctx.last_finished_stopped = old


def test_rebuild_flag_prevents_killing_running_tasks(qapp):
    """★ 换语言重建窗口时，closeEvent 不能停掉正在跑的任务（审计 G-02）。"""

    class FakeRunner:
        def __init__(self):
            self.stopped = 0

        def is_running(self):
            return True

        def stop(self, *a, **k):
            self.stopped += 1

    from gui.app import MainWindow

    # 1) 带 _rebuilding 标记关闭：任务绝不能被停
    w = MainWindow()
    w.goto("nav.train")
    fake = FakeRunner()
    w.pages["nav.train"].runner = fake
    w._rebuilding = True
    w.close()
    assert fake.stopped == 0, "语言切换重建窗口时把后台任务停掉了"
    w.deleteLater()

    # 2) 正常关闭：仍然要停任务（别把清理逻辑一起弄丢）
    w2 = MainWindow()
    w2.goto("nav.train")
    fake2 = FakeRunner()
    w2.pages["nav.train"].runner = fake2
    w2.close()
    assert fake2.stopped >= 1, "正常关窗没有停掉后台任务"
    w2.deleteLater()


def test_train_build_args_passes_selected_checkpoint_path(window):
    """★ 断点页「续训」必须把选中文件作为 --resume 的值传给 CLI（审计 G-03）。"""
    window.goto("nav.train")
    tp = window.pages["nav.train"]
    path = r"C:/path/to/lingyu/out/small/ckpt_step2000.pt"
    args = tp._build_args(path)
    assert "--resume" in args, "续训参数里没有 --resume"
    i = args.index("--resume")
    assert i + 1 < len(args) and args[i + 1] == path, (
        "没有把选中断点路径传给 --resume：%r" % (args,))
    # 旧的「只传 flag」写法在新代码路径里不应再出现
    src = open(os.path.join(ROOT, "gui", "pages", "train.py"),
               encoding="utf-8").read()
    assert "isinstance(resume, str)" in src, "train 页不支持传显式断点路径"


def test_checkpoints_page_uses_selected_path_for_resume():
    from gui.pages import checkpoints as ck
    src = open(ck.__file__, encoding="utf-8").read()
    assert "_start(resume=p)" in src, "断点页续训没有传选中路径"
    assert "_start(resume=True)" not in src, "断点页仍在用「只传 auto」的旧写法"


# ==================================================== ★ 第二批：优雅停止协议
# 2026-10-17：LINGYU_STOP_FILE 停止标记 + 20s 宽限 + train/sft 停止不报失败

class _FakeSignal:
    def connect(self, *a, **k):
        pass


class _FakeProc:
    """假装是 QProcess：只记录 env/start/kill，供协议测试用。"""

    NotRunning = 0
    MergedChannels = 0
    FailedToStart = 1

    def __init__(self, parent=None):
        self.env = None
        self.started = None
        self.killed = 0
        self.terminated = 0
        self.readyReadStandardOutput = _FakeSignal()
        self.finished = _FakeSignal()
        self.errorOccurred = _FakeSignal()

    def setProcessChannelMode(self, mode):
        pass

    def setWorkingDirectory(self, d):
        pass

    def setProcessEnvironment(self, env):
        self.env = env

    def start(self, prog, args):
        self.started = (prog, list(args))

    def state(self):
        return 1                       # != NotRunning

    def kill(self):
        self.killed += 1

    def terminate(self):
        self.terminated += 1

    def deleteLater(self):
        pass


def _start_runner_with_fake_proc(monkeypatch):
    import gui.runner as runner_mod
    monkeypatch.setattr(runner_mod, "QProcess", _FakeProc)
    r = runner_mod.Runner()
    assert r.start(["--dry"]) is True
    assert isinstance(r.proc, _FakeProc)
    return r


def test_stop_flag_file_protocol(monkeypatch):
    """★ 停止协议：启动时生成唯一路径并传给子进程；点停止才写入该文件。"""
    r = _start_runner_with_fake_proc(monkeypatch)
    proc = r.proc
    flag = r._stop_file
    assert flag and os.path.isabs(flag), "没有生成停止标记的绝对路径"
    assert os.path.basename(flag).startswith("lingyu_stop_"), flag
    assert not os.path.exists(flag), "启动时就存在停止标记 —— 引擎会立刻退出"
    assert proc.env.value("LINGYU_STOP_FILE") == flag, (
        "没有通过环境变量把停止标记路径传给子进程")

    lines, finished = [], []
    r.line.connect(lambda t, l: lines.append(t))
    r.finished.connect(lambda code, stopped: finished.append((code, stopped)))
    r.stop()
    assert os.path.isfile(flag), "点停止后没有写入停止标记文件"
    assert any("正在保存断点并退出" in t for t in lines), lines
    assert r._kill_timer.isActive() and r._kill_timer.interval() == 20000, (
        "没有起 20 秒宽限定时器：active=%r interval=%r"
        % (r._kill_timer.isActive(), r._kill_timer.interval()))

    r._on_finished(0, None)            # 模拟引擎保存断点后正常退出
    assert finished and finished[-1][1] is True, "正常退出没有上报 stopped=True"
    assert not r._kill_timer.isActive(), "正常退出后宽限定时器没撤销"
    assert not os.path.exists(flag), "任务结束后没有删除停止标记文件"


def test_stop_timeout_force_kills_and_says_so(monkeypatch):
    """★ 20 秒没退出：强杀 + 如实提示可能丢进度 + 清理标记文件。"""
    r = _start_runner_with_fake_proc(monkeypatch)
    lines = []
    r.line.connect(lambda t, l: lines.append(t))
    r.stop()
    flag = r._stop_file
    assert os.path.isfile(flag)
    r._force_kill()                    # 模拟宽限定时器到点
    assert r.proc.killed == 1, "超时后没有强杀进程"
    assert any("强制结束" in t for t in lines), lines
    assert not os.path.exists(flag), "强杀后没有清理停止标记文件"


def test_train_page_reports_stop_not_failure(window):
    """★ train 页在用户主动停止时打印「已停止」，不能打印「训练失败」。"""
    window.goto("nav.train")
    tp = window.pages["nav.train"]
    tp.log.clear()
    tp._on_finished(1, True)           # 退出码 1 + stopped=True（Windows 实况）
    text = tp.log.toPlainText()
    assert "训练失败" not in text, text
    assert "已停止" in text, text


def test_sft_page_reports_stop_not_failure(window):
    """★ SFT 页同理：主动停止 ≠ 失败。"""
    window.goto("nav.sft")
    sp = window.pages["nav.sft"]
    sp.log.clear()
    sp._on_finished(1, True)
    text = sp.log.toPlainText()
    assert "SFT 失败" not in text, text
    assert "已停止" in text, text


def test_graceful_stop_protocol_is_wired():
    """协议关键字必须同时出现在 runner 与两个页面的停止分支里。"""
    rsrc = open(os.path.join(ROOT, "gui", "runner.py"), encoding="utf-8").read()
    assert "tempfile.mkstemp" in rsrc, "没有创建临时停止标记"
    assert "LINGYU_STOP_FILE" in rsrc, "没有给子进程传停止标记路径"
    assert "正在保存断点并退出" in rsrc, "停止提示不是协议文案"
    for rel in ("train.py", "sft.py"):
        src = open(os.path.join(ROOT, "gui", "pages", rel),
                   encoding="utf-8").read()
        assert "if stopped:" in src, "%s 没先判断 stopped" % rel
        assert "已停止（断点已保存）" in src, "%s 没有停止成功文案" % rel


# ==================================================== 第三批修复回归（审计 D）

def test_stop_button_click_ignores_checked_bool(window):
    """★ G-05：QPushButton.clicked(bool) 不能把 checked 当成 grace_ms 传进去。

    旧实现 `stop_btn.clicked.connect(self.runner.stop)`：点一下会调
    `stop(False)` —— grace_ms=False 被当 0ms，立刻强杀，断点根本来不及存。

    1.7 起停止改走 `ctx.stop_current_task()`（页面重建后也能停到真正在跑的
    Runner），这里同时把全局总线接到页面自己的 Runner 上。
    """
    from gui.context import ctx

    for nav, attr in (("nav.sample", "sample"), ("nav.sft", "sft")):
        window.goto(nav)
        page = window.pages[nav]

        class _FakeRunner(object):
            def __init__(self):
                self.calls = []

            def stop(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return True

            def is_running(self):
                return True

        page.runner = _FakeRunner()
        old = ctx.task_runner
        ctx.task_runner = page.runner
        try:
            page.stop_btn.click()
        finally:
            ctx.task_runner = old
        assert page.runner.calls == [((), {})], (
            "%s 的停止按钮把 clicked(bool) 传给了 stop()：%r"
            % (attr, page.runner.calls))

    for rel in ("sample.py", "sft.py"):
        src = open(os.path.join(ROOT, "gui", "pages", rel),
                   encoding="utf-8").read()
        assert "clicked.connect(self.runner.stop)" not in src, (
            "%s 仍然直接连 runner.stop（会收到 checked=False）" % rel)


# ==================================================== 1.7 / 滚轮误改回归

def test_wheel_over_unfocused_input_does_not_change_value(qapp):
    """★ 1.1：鼠标路过输入框滚动页面，不能顺手把参数改掉。"""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QWidget

    from gui.widgets import make_int

    spin = make_int(0, 100, 10)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.addWidget(spin)
    for _ in range(40):
        from PySide6.QtWidgets import QLabel
        lay.addWidget(QLabel("填充行"))
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.resize(300, 200)
    area.show()
    qapp.processEvents()

    spin.clearFocus()
    bar = area.verticalScrollBar()
    bar.setValue(0)

    ev = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0),
                     QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    QApplication.sendEvent(spin, ev)
    assert spin.value() == 10, "滚轮路过输入框改掉了数值（1.1 的 bug）"
    assert bar.value() > 0, "滚轮事件没有转给外层滚动区，页面滚不动"
    area.close()
    area.deleteLater()


def test_ctx_task_bus_forwards_output_and_stop(qapp):
    """★ 1.7 的根因：任务状态必须挂在全局总线上，而不是页面实例里。"""
    from PySide6.QtCore import QObject, Signal

    from gui.context import ctx

    class _FakeRunner(QObject):
        line = Signal(str, str)
        progress = Signal(dict)
        validation = Signal(int, float)

        def __init__(self):
            super().__init__()
            self.stopped = 0
            self.running = True

        def is_running(self):
            return self.running

        def stop(self, *a, **k):
            self.stopped += 1
            self.running = False

    got = []
    handler = lambda name, text, level: got.append((name, text, level))
    ctx.task_line.connect(handler)
    old_runner, old_task = ctx.task_runner, ctx.running_task
    try:
        ctx.begin_task("train")
        fake = _FakeRunner()
        ctx.attach_runner("train", fake)
        fake.line.emit("hello", "info")
        assert got == [("train", "hello", "info")], (
            "ctx 没把任务输出广播出来：%r" % (got,))
        assert ctx.stop_current_task() is True
        assert fake.stopped == 1, "ctx.stop_current_task 没有停掉 Runner"
    finally:
        ctx.task_line.disconnect(handler)
        ctx.task_runner = old_runner
        ctx.running_task = old_task


def test_running_task_restores_stop_button_on_show(window):
    """★ 1.7：训练在后台跑时切回训练页，停止按钮必须还在。"""
    from gui.context import ctx

    class _FakeRunner(object):
        def is_running(self):
            return True

        def stop(self, *a, **k):
            return True

    window.goto("nav.train")
    page = window.pages["nav.train"]
    old_runner, old_task = ctx.task_runner, ctx.running_task
    try:
        ctx.begin_task("train")
        ctx.task_runner = _FakeRunner()
        page.on_show()
        assert not page.stop_btn.isHidden(), "切回训练页后停止按钮消失了（1.7）"
        assert page.start_btn.isHidden(), "任务跑着却又能点「开始训练」"
    finally:
        ctx.task_runner = old_runner
        ctx.running_task = old_task
        page._set_running(False)


def test_time_estimate_scales_with_model_size(window):
    """★ G-10：预计耗时不能写死 50000 token/s（13M 与 360M 会显示同一个时间）。"""
    from gui.pages import train as tm

    assert tm._tokens_per_sec(13e6) > tm._tokens_per_sec(360e6) > \
        tm._tokens_per_sec(2e9), "token/s 没有随参数量下降"

    window.goto("nav.train")
    tp = window.pages["nav.train"]

    def _minutes(params):
        tp._est = {"params": params}
        text = tp._time_hint()          # 形如 "≈27 分钟" / "≈3.8 小时"
        assert text.startswith("≈"), text
        import re
        m = re.search(r"([\d.]+)", text)
        assert m, text
        return float(m.group(1)) * (3600 if "小时" in text else 1)

    small = _minutes(13e6)
    big = _minutes(360e6)
    assert big > small * 2, "大模型预计耗时没有明显变长：%r vs %r" % (big, small)

    src = open(os.path.join(ROOT, "gui", "pages", "train.py"),
               encoding="utf-8").read()
    assert "toks / 50000.0" not in src, "还在写死 50000 token/s"


def test_cmd_preview_has_subcommand_and_quotes_paths(window):
    """★ G-11：命令预览必须能直接复制运行（含 train 子命令、路径带引号）。"""
    window.goto("nav.train")
    tp = window.pages["nav.train"]
    preview = tp._cmd_preview(tp._build_args())
    assert preview.startswith("lingyu train --config "), preview
    quoted = tp._cmd_preview(["train", "--config", r"C:\Program Files\x.ini"])
    assert r'"C:\Program Files\x.ini"' in quoted, quoted

    src = open(os.path.join(ROOT, "gui", "pages", "train.py"),
               encoding="utf-8").read()
    assert '" ".join(args[1:])' not in src, "预览还在手拼参数，丢了子命令"


def test_env_probe_can_retry_after_failure(monkeypatch):
    """★ G-15：探测失败/装好 torch 后必须能重新探测（_probing 要复位）。"""
    import types
    from gui import context as ctxmod

    class _FakePool(object):
        def __init__(self):
            self.started = []

        @classmethod
        def globalInstance(cls):
            return _POOL

        def start(self, runnable):
            self.started.append(runnable)

    _POOL = _FakePool()

    class _FakeProbe(object):
        def __init__(self):
            self.sig = types.SimpleNamespace(
                done=types.SimpleNamespace(connect=lambda *a, **k: None))

        def run(self):
            pass

    monkeypatch.setattr(ctxmod, "QThreadPool", _FakePool)
    monkeypatch.setattr(ctxmod, "_EnvProbe", _FakeProbe)

    c = ctxmod.Context()
    c.start_env_probe()
    assert len(_POOL.started) == 1
    c.start_env_probe()                 # 正在探测中：不重复启动
    assert len(_POOL.started) == 1

    c._on_env({"probing": True, "torch": "2.14.0"})
    assert c._probing is False, "探测结束后 _probing 没复位"
    assert c._probe is None, "探测结束后还留着 QRunnable 引用"

    c.start_env_probe()                 # 失败/换 torch 之后可以重试
    assert len(_POOL.started) == 2, "装好 torch 后无法重新探测（G-15）"


def test_checkpoints_loaders_are_deduped_and_recycled(tmp_path, monkeypatch,
                                                      window):
    """★ G-14：断点 loader 必须按路径去重，并且换目录后释放旧引用。

    旧实现每次 refresh 都 `self._loaders.append(...)`，列表随刷新次数
    无限增长；换输出目录后旧文件的 loader 永远不释放。
    """
    import types
    from gui.pages import checkpoints as cpm

    class _Sig(object):
        def __init__(self):
            self.done = types.SimpleNamespace(connect=lambda *a, **k: None)

    class _FakeLoader(object):
        def __init__(self, path):
            self.path = path
            self.sig = _Sig()

    class _FakePool(object):
        def start(self, loader):
            pass

    monkeypatch.setattr(cpm, "_Loader", _FakeLoader)

    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"\x00")
    b.write_bytes(b"\x00")

    window.goto("nav.checkpoints")
    page = window.pages["nav.checkpoints"]
    page._loaders.clear()
    page.pool = _FakePool()
    # 来源默认改为「全部」（用户报看不到 SFT 断点）；这里测的是「当前目录」模式
    page.src_combo.setCurrentIndex(page.src_combo.findData("current"))

    page.dir_picker.set_value(str(tmp_path))
    page.refresh()
    assert isinstance(page._loaders, dict), "loader 容器还是列表，会无限增长"
    assert len(page._loaders) == 2, page._loaders

    page.refresh()                      # 重复刷新不应重复加载
    assert len(page._loaders) == 2, "同一次刷新重复给同一文件起了 loader"

    os.remove(str(b))
    page.refresh()                      # 文件没了 → 旧 loader 引用要释放
    assert len(page._loaders) == 1, page._loaders
    page._loaders.clear()


def test_train_page_config_selector_switches_config(window, monkeypatch):
    """★ 训练页的配置文件下拉框。

    用户要「自己训练」时第一件事往往是换配置（chat.ini 13M / chat_medium.ini
    47.8M / small / medium…）。以前界面只认 gui.json 里钉死的那一份，
    换配置得手改 JSON —— 这里守住「下拉框能列出出厂配置、选中即切换」。
    """
    from gui.context import ctx

    window.goto("nav.train")
    page = window.pages["nav.train"]
    names = [page.config_combo.itemText(i)
             for i in range(page.config_combo.count())]
    for need in ("small.ini", "chat.ini", "chat_medium.ini"):
        assert any(n == need or n.startswith(need) or n.endswith(need)
                   for n in names), (need, names)

    monkeypatch.setattr(ctx, "save", lambda: None)   # 别写真实的 gui.json
    old = ctx.config_path
    try:
        page.config_combo.setCurrentIndex(names.index("chat.ini"))
        assert os.path.basename(ctx.config_path) == "chat.ini", ctx.config_path
        assert ctx.config.get("post", {}).get("method") == "sft"
    finally:
        ctx.load_config(old)
        ctx.settings["config"] = ctx.config_path


def test_child_environment_inherits_launcher_pythonpath(qapp, monkeypatch):
    """★ 打包版 GUI 的子进程必须继承启动器设的 PYTHONPATH。

    GUI 进程本身是启动器用 `PYTHONPATH=<runtime>/app/src` 起的；
    如果 runner / 聊天页改用 systemEnvironment() 重建环境，这个变量
    就会丢，子进程报 `No module named lingyu` —— 用户实测截图就是这个。
    """
    from gui.runner import child_environment

    monkeypatch.setenv("PYTHONPATH", "X:/inst/runtime/app/src")
    monkeypatch.setenv("LINGYU_HOME", "X:/inst/runtime")
    env = child_environment()
    assert env.value("PYTHONPATH") == "X:/inst/runtime/app/src"
    assert env.value("LINGYU_HOME") == "X:/inst/runtime"


def test_no_gui_module_rebuilds_env_from_system_environment():
    """守住「不许再从 systemEnvironment() 起家」——那会悄悄丢掉 PYTHONPATH。"""
    gui_dir = os.path.join(ROOT, "gui")
    hits = []
    for dirpath, _dirnames, filenames in os.walk(gui_dir):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as f:
                n = f.read().count("systemEnvironment()")
            if n:
                hits.append(os.path.relpath(path, gui_dir))
    assert hits and all(h == "runner.py" for h in hits), hits


# ==================================================== 监控页数据分离（审计 1.2）

def _monitor_fixture(tmp_path):
    """造两个数据源目录：预训练 + SFT，各带 run.json / history.csv。"""
    import json

    pre = tmp_path / "chat_pretrain"
    sft_dir = tmp_path / "chat_sft"
    pre.mkdir()
    sft_dir.mkdir()
    (pre / "run.json").write_text(json.dumps(
        {"kind": "pretrain", "format": 1, "tokens_per_step": 4096,
         "max_steps": 1000}, ensure_ascii=False), encoding="utf-8")
    (pre / "history.csv").write_text(
        "step,train_loss,val_loss,lr,grad_norm,elapsed_sec\n"
        "10,5.5000,6.0000,0.001000,0.9000,10.0\n"
        "20,5.0000,5.8000,0.000900,0.8000,20.0\n", encoding="utf-8")
    (sft_dir / "run.json").write_text(json.dumps(
        {"kind": "sft", "format": 1, "tokens_per_step": 2048,
         "max_steps": 140}, ensure_ascii=False), encoding="utf-8")
    (sft_dir / "history.csv").write_text(
        "step,train_loss,val_loss,lr,grad_norm,elapsed_sec\n"
        "100,2.5000,nan,0.000200,0.1000,100.0\n"
        "140,2.4000,2.4500,0.000200,0.0900,140.0\n", encoding="utf-8")
    return pre, sft_dir


def _monitor_page(monkeypatch, pre, sft_dir, qapp):
    from gui.pages.monitor import MonitorPage

    page = MonitorPage()
    page.ensure_built()
    monkeypatch.setattr(page.ctx, "out_dir", lambda: str(pre))
    monkeypatch.setattr(page.ctx, "sft_out_dir", lambda: str(sft_dir))
    page._refresh_runs()
    return page


def test_monitor_keeps_pretrain_and_sft_numbers_separate(
        qapp, tmp_path, monkeypatch):
    """★ 1.2：预训练 / SFT 的 tok/s、loss、总步数不能混用同一套。

    以前监控页只按当前配置的 `[train]` 参数算 token/步，套到 SFT 上会得到
    预训练量级的 9 万 tok/s —— 用户实测见过「微调页显示 93k tok/s」。
    """
    from gui.i18n import t

    pre, sft_dir = _monitor_fixture(tmp_path)
    page = _monitor_page(monkeypatch, pre, sft_dir, qapp)
    assert page.src_combo.count() == 2, "下拉框要有「预训练 / 微调 SFT」两个数据源"

    page._select("pretrain")
    assert page.tile_step.value_label.text() == "20"
    assert page.tile_loss.value_label.text() == "5.0000"
    assert page.tile_speed.sub_label.text() == "4,096 token/步"
    assert page.tile_val.label_label.text() == t("mo.valloss")
    assert page._tokens_per_step() == 4096

    page._select("sft")
    assert page.tile_step.value_label.text() == "140"
    assert page.tile_loss.value_label.text() == "2.4000"
    assert page.tile_val.value_label.text() == "2.4500"
    assert page.tile_speed.sub_label.text() == "2,048 token/步"
    assert page.tile_val.label_label.text() == t("mo.valloss_sft"), \
        "SFT 的验证 loss 只算回答部分，标题必须区分"
    assert page._tokens_per_step() == 2048
    assert page._total_steps() == 140


def test_monitor_progress_comes_from_current_source_only(
        qapp, tmp_path, monkeypatch):
    """训练输出也按数据源分流：SFT 的日志不该混进预训练监控里。"""
    pre, sft_dir = _monitor_fixture(tmp_path)
    page = _monitor_page(monkeypatch, pre, sft_dir, qapp)

    page._select("pretrain")
    page.log.clear()
    page._bus_line("sft", "SFT-ONLY", "info")
    page._bus_line("train", "PRETRAIN-ONLY", "info")
    text = page.log.toPlainText()
    assert "PRETRAIN-ONLY" in text
    assert "SFT-ONLY" not in text


def test_monitor_speed_uses_row_deltas_not_cumulative_time(
        qapp, tmp_path, monkeypatch):
    """★ 续训后 history.csv 的 elapsed_sec 会从头计，速度必须按相邻行增量算。

    真实历史（runtime/out/medium）：21000 步那行只写了 195 秒。按
    「步数 / 累计耗时」会算出 137 万 tok/s（审计截图里真的出现过）；
    按相邻两行增量（500 步 / 136.3 秒）才是 4.7 万 tok/s。
    """
    import json

    from gui.i18n import t
    from gui.pages.monitor import MonitorPage

    d = tmp_path / "medium"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(
        {"kind": "pretrain", "format": 1, "tokens_per_step": 12800,
         "max_steps": 30000}), encoding="utf-8")
    (d / "history.csv").write_text(
        "step,train_loss,val_loss,lr,grad_norm,elapsed_sec\n"
        "20000,2.0625,3.9454,0.000080,0.9652,2108.0\n"
        "20500,1.7420,3.9240,0.000081,0.5873,59.1\n"
        "21000,2.2355,3.8923,0.000080,0.6776,195.4\n", encoding="utf-8")

    page = MonitorPage()
    page.ensure_built()
    monkeypatch.setattr(page.ctx, "out_dir", lambda: str(d))
    monkeypatch.setattr(page.ctx, "sft_out_dir", lambda: str(tmp_path / "none"))
    page._refresh_runs()
    page._select("pretrain")

    assert page.tile_speed.value_label.text() == "46955 tok/s", \
        "12800 tok/步 × 500 步 ÷ 136.3 秒 = 46955 tok/s（不是 137 万）"
    assert "预计剩余" in page.tile_elapsed.sub_label.text(), \
        "续训场景也要给出合理的 ETA"


# ==================================================== 审计 1.5 / 1.1（2026-10-24）
# 这一轮把「参数说明」从常驻小字改成 ? 徽标（1.5），并把训练页改成
# 「运行看板 + 可收起日志」（1.1）。下面的测试钉住这两件事的行为，
# 不只是「代码里有没有写」。


def test_field_help_badge_keeps_explanation_in_tooltip(qapp):
    """1.5：标签后面是 ? 徽标；完整说明在 tooltip，不常驻占版面。"""
    from PySide6.QtWidgets import QLineEdit

    from gui.widgets import Field, HelpBadge

    f = Field("n_layer", QLineEdit(),
              help_text="堆叠多少层 Transformer（越多越强，也越吃显存）")
    badge = f.findChild(HelpBadge)
    assert badge is not None and badge.text() == "?"
    assert "Transformer" in badge.toolTip()
    assert badge.toolTip().startswith("<div"), (
        "长说明必须用带宽度限制的富文本 —— Qt 的纯文本 tooltip 不换行")
    # 常驻 hint 行不再重复这段说明（否则就是「一墙灰色小字」）
    assert f.hint.text() == ""


def test_paramform_uses_help_badges_not_hint_paragraphs(qapp):
    """参数表单的每个参数都要有 ? 徽标，而不是十几行灰色说明。"""
    from gui.paramform import ParamForm
    from gui.widgets import HelpBadge

    form = ParamForm("model")
    badges = form.findChildren(HelpBadge)
    assert len(badges) >= 5, "model 段参数不少，至少该有 5 个 ? 徽标"
    for b in badges:
        assert b.toolTip(), "每个 ? 徽标都必须带说明"


def test_train_run_dashboard_updates_from_task_bus(qapp, window):
    """1.1：训练中给的是「进度条 + 步数/loss/lr/速度」看板。"""
    from gui.i18n import t

    window.goto("nav.train")
    page = window.pages["nav.train"]
    page._dash_begin()
    page.ctx.task_progress.emit(
        "train", {"step": 1200, "total": 3000, "loss": 3.21,
                  "lr": 6e-4, "gnorm": 1.0})
    assert page.progress.value() == 1200
    assert page.progress.maximum() == 3000
    assert page.dash_step.value_label.text() == "1,200 / 3,000"
    assert page.dash_loss.value_label.text() == "3.2100"
    assert page.dash_lr.value_label.text() == "6.00e-04"

    # SFT 的进度不能混进预训练看板
    page.ctx.task_progress.emit("sft", {"step": 5, "total": 10})
    assert page.progress.value() == 1200

    page._dash_end(0, False)
    assert page.dash_state.text() == t("tr.dash.done")


def test_train_raw_log_is_collapsed_until_toggled(qapp, window):
    """1.1：原始日志默认收起 —— 第一眼不应是 PowerShell。"""
    window.goto("nav.train")
    page = window.pages["nav.train"]
    assert page.log_card.isHidden() is True
    page.log_toggle.setChecked(True)
    assert page.log_card.isHidden() is False
    page.log_toggle.setChecked(False)
    assert page.log_card.isHidden() is True


def test_sft_fields_have_help_badges(qapp, window):
    """1.5：SFT 页的每个可调参数也必须有 ? 徽标（不只是按 spec 生成的训练页）。"""
    from gui.widgets import HelpBadge

    window.goto("nav.sft")
    page = window.pages["nav.sft"]
    badges = page.findChildren(HelpBadge)
    assert len(badges) >= 8, (
        "SFT 页有数据/断点/模板/系统提示词/步数/批大小/长度/试问 8 个字段")
    for b in badges:
        assert b.toolTip(), "每个 ? 徽标都要有说明，否则等于没写"


def test_train_tokens_per_step_includes_block_size(qapp, window):
    """1.1 看板的「token/步」必须含 block_size（它在 [model] 段，不在 [train]）。"""
    window.goto("nav.train")
    page = window.pages["nav.train"]
    cfg = page._collect()
    expected = (int(cfg["train"]["batch_size"]) *
                int(cfg["train"].get("grad_accum", 1)) *
                int(cfg["model"]["block_size"]))
    assert page._tokens_per_step() == expected


# ==================================================== 2026-10-24 用户实测修复
# 用户发过 9 条实测 bug（B-01…B-11），这里各钉一条。


def test_help_badge_shows_tooltip_explicitly(qapp, monkeypatch):
    """B-01：悬停「?」不出帮助 —— 改成显式 QToolTip.showText。"""
    from PySide6.QtWidgets import QToolTip
    from gui.widgets import HelpBadge
    calls = []

    def _fake_show(*args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(QToolTip, "showText", _fake_show)
    badge = HelpBadge("这是参数说明")
    badge.show_help()
    assert calls, "显式调用 show_help() 没有弹出 tooltip"
    assert "这是参数说明" in calls[0][1]


def test_lang_switch_rebuilds_with_new_strings(qapp, monkeypatch):
    """B-06：切语言只重建窗口不切 i18n —— 重建时必须先 set_lang。"""
    from gui import app as appmod
    from gui import i18n

    made = []

    class _Recorder:
        def __init__(self):
            made.append(self)

        def show(self):
            pass

    real_win = appmod.MainWindow
    w = real_win()
    monkeypatch.setattr(appmod, "MainWindow", _Recorder)
    try:
        w._rebuilding = False
        w._rebuild_for_lang("en_US")
        assert i18n._lang == "en_US", "重建窗口后 i18n 语言没切换"
        assert made, "没有重建窗口"
    finally:
        i18n.set_lang("zh_CN")
        try:
            w.close()
            w.deleteLater()
        except Exception:
            pass


def test_refresh_env_restarts_probe_and_stamps_time(window, monkeypatch):
    """B-04/B-05：显存探测必须能手动重跑，并记下检测时间。"""
    from gui.context import ctx
    called = []
    monkeypatch.setattr(ctx, "start_env_probe", lambda: called.append(1))
    ctx.refresh_env()
    assert called, "refresh_env() 没有重新触发探测"
    monkeypatch.setattr(ctx, "gpu_ts", 0.0)
    ctx._on_env({"vram_free": 1, "total": 2, "device": "test",
                 "backend": "local", "torch": "2.0", "cuda": True,
                 "probing": False})
    assert ctx.gpu_ts > 0, "探测结束后没有记录 gpu_ts"


def test_build_args_init_from_for_foreign_checkpoint(window):
    """B-02/B-03：别的 run 的断点当初始化权重（--init-from），不当续训。"""
    window.goto("nav.train")
    tp = window.pages["nav.train"]
    path = r"C:/path/to/lingyu/out/chat_sft/ckpt_sft.pt"
    args = tp._build_args(("init", path))
    assert "--init-from" in args and path in args
    assert "--resume" not in args


def test_prepare_resume_alignment_asks_and_applies(window, monkeypatch):
    """B-02：结构对不上时先问，点「是」按断点对齐，返回原路径继续续训。"""
    from PySide6.QtWidgets import QMessageBox
    import os
    window.goto("nav.train")
    tp = window.pages["nav.train"]
    snapshot = dict(tp._collect()["model"])
    snapshot["n_layer"] = 99
    monkeypatch.setattr(tp.ctx, "read_ckpt_meta",
                        lambda p: {"model_cfg": snapshot})
    monkeypatch.setattr(tp.ctx, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    path = os.path.join(tp.ctx.out_dir(), "ckpt.pt")
    result = tp._prepare_resume(path)
    assert result == path
    assert tp._collect()["model"]["n_layer"] == 99, "没有按断点对齐结构"


def test_checkpoints_page_fills_from_sidecar_without_loader(window, tmp_path):
    """B-02：新断点优先读旁车 JSON，不 import torch、不开线程。"""
    import json
    window.goto("nav.checkpoints")
    page = window.pages["nav.checkpoints"]
    ck = tmp_path / "ckpt_step1200.pt"
    ck.write_bytes(b"\x00")
    meta = {"ckpt_format": 1, "step": 1200, "best_val": 1.23,
            "model_cfg": {"n_layer": 8, "d_model": 512, "use_moe": False},
            "tokenizer_path": "tok.txt", "config_path": "configs/a.ini"}
    (tmp_path / "ckpt_step1200.pt.json").write_text(
        json.dumps(meta), encoding="utf-8")
    page.src_combo.setCurrentIndex(1)          # 当前项目
    page.dir_picker.set_value(str(tmp_path))
    page.refresh()
    assert page.table.item(0, 1).text() == "1,200"
    assert "8层" in page.table.item(0, 2).text()
    assert page.table.item(0, 3).text() == "1.2300"
    assert not page._loaders, "有旁车还去起了 torch 线程"

# ---------------------------------------------------------------- 1.4 / 1.6


def test_runner_parses_structured_json_events(qapp):
    """引擎 `--json` 的事件行要能翻译成 progress/validation 信号（审计 1.4）。"""
    from gui.runner import Runner

    r = Runner()
    progress, validations, lines = [], [], []
    r.progress.connect(progress.append)
    r.validation.connect(lambda s, v: validations.append((s, v)))
    r.line.connect(lambda text, level: lines.append((text, level)))

    ok = r._handle_event(
        '{"event": "step", "task": "train", "step": 10, "total": 100,'
        ' "loss": 1.5, "lr": 0.001}')
    assert ok is True
    assert progress and progress[0]["step"] == 10
    assert lines and lines[0][1] == "info"

    ok = r._handle_event(
        '{"event": "eval", "task": "train", "step": 100, "val_loss": 2.5}')
    assert ok is True
    assert validations == [(100, 2.5)]

    assert r._handle_event('{"not": "an event"}') is False


def test_help_page_search_filters_and_renders(qapp):
    """帮助页（审计 1.6）：默认列出全部文章，搜索能过滤，右侧能渲染正文。"""
    from gui.pages.help import HelpPage

    page = HelpPage()
    page.ensure_built()
    assert page.list.count() >= 10

    page.search_box.setText("moe")
    assert 1 <= page.list.count() < 40
    assert page.view.toPlainText().strip()

    page.search_box.setText("学习率")
    assert page.list.count() >= 1


def test_help_page_is_registered_in_nav(qapp):
    """帮助页必须出现在侧边栏注册表里，否则用户根本点不到。"""
    from gui.pages import all_pages

    keys = [entry[1] for entry in all_pages()]
    assert "nav.help" in keys


def test_gui_asks_engine_for_json_events(qapp):
    """1.4 的 GUI 侧：train/sft/sample 都要带 --json，别再靠正则猜。"""
    for name in ("train.py", "sft.py", "sample.py"):
        src = open(os.path.join(ROOT, "gui", "pages", name),
                   encoding="utf-8").read()
        assert '"--json"' in src, name

def test_train_dashboard_shows_perstep_and_epoch(qapp, window):
    """看板必须显示「每步耗时」和「epoch」——这两个是用户点名要的真实统计。"""
    from gui.pages.train import TrainPage
    page = TrainPage()
    page.resize(1400, 900)
    page.ensure_built()
    page._dash_begin()
    page._on_progress("train", {
        "step": 1200, "total": 21000, "loss": 3.21, "lr": 4.7e-4,
        "tok_s": 43000.0, "elapsed": 240.0,
        "tokens_per_step": 12800, "dataset_tokens": 34805989,
    })
    qapp.processEvents()
    text = page.dash_meta.text()
    assert "每步" in text or "/step" in text
    assert "epoch" in text
    page.deleteLater()

