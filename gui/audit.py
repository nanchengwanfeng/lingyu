# -*- coding: utf-8 -*-
"""界面巡检：把每个页面在**关键状态下**截图，然后挨个看。

## 为什么要专门写这个

用户说得对：**光看代码发现不了「238% 还说够用」这种问题**。
那个 bug 是截图里一眼看出来的 —— 数字自相矛盾。

所以巡检脚本不只截「页面长什么样」，还要截**容易出错的状态**：

    - 训练页：小配置（该说够用）/ 大配置（该说装不下）/ 还没探测到显存
    - 概览页：有语料 / 没语料
    - 各页的空状态（没断点时、没数据时）

`--only` 用来只截某几张，排查时快。

产物在 `_shots/`（不进 git）。

跑法：
    python gui/audit.py               # 全部
    python gui/audit.py --only train_big
    python gui/audit.py --theme light
"""
import argparse
import os
import sys
import time

# 控制台默认是 GBK，打印 ⚠ / ✓ 这类字符会直接抛 UnicodeEncodeError，
# 把巡检中断在半路。先把它摁住。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 巡检截图要的是**最终稳定画面**：切页淡入动效会让 grab() 拍到半透明
# 的一帧，看起来像界面坏了（而且是随机的）。巡检里关掉动效。
os.environ.setdefault("LINGYU_GUI_NO_ANIM", "1")

from PySide6.QtCore import QTimer            # noqa: E402
from PySide6.QtWidgets import QApplication   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="dark")
    ap.add_argument("--lang", default="zh_CN")
    ap.add_argument("--out", default="_shots")
    ap.add_argument("--only", default=None, help="只跑名字含这个字的")
    ap.add_argument("--settle", type=float, default=6.0)
    a = ap.parse_args()

    app = QApplication([])
    from gui.context import ctx
    from gui.i18n import set_lang
    from gui.theme import stylesheet, th
    ctx.settings["theme"] = a.theme
    ctx.settings["lang"] = a.lang
    set_lang(a.lang)
    th.set(a.theme)
    app.setStyleSheet(stylesheet())

    from gui.app import MainWindow
    win = MainWindow()
    win.resize(1440, 900)
    win.move(-4000, -4000)      # 挪到屏幕外，不打扰用户
    win.show()

    os.makedirs(a.out, exist_ok=True)
    done = []

    def snap(name):
        win.repaint()
        app.processEvents()
        time.sleep(0.15)
        app.processEvents()
        p = os.path.join(a.out, "%s_%s.png" % (a.theme, name))
        win.grab().save(p)
        done.append(p)
        print("  截图:", p)

    def want(name):
        return (not a.only) or (a.only in name)

    # ---------------- 巡检步骤 ----------------
    def step_pages(i=0):
        """九页挨个截（默认状态）。"""
        keys = [k for _c, k, _g, _i in win.page_specs]
        if i >= len(keys):
            QTimer.singleShot(150, step_train_big)
            return
        k = keys[i]
        name = "page_" + k.replace("nav.", "")
        if want(name):
            win.goto(k)
            snap(name)
        QTimer.singleShot(220, lambda: step_pages(i + 1))

    def step_train_big():
        """★ 关键状态：大配置 —— 必须说「装不下」，绝不能出现 238% 够用。"""
        if not want("train_big"):
            return QTimer.singleShot(100, step_train_tiny)
        win.goto("nav.train")
        p = win.pages["nav.train"]
        p.model_form.form.set_value("n_layer", 32)
        p.model_form.form.set_value("n_embd", 1024)
        p.model_form.form.set_value("n_head", 8)
        p._estimate()
        snap("train_big")
        # 文字也打出来核对（截图看不清时用这个）
        print("     估算结论:", p.est_note.text())
        print("     参数/显存:", p.est_kv._rows)
        QTimer.singleShot(150, step_train_tiny)

    def step_train_tiny():
        """小配置 —— 应该说「够用」。"""
        if not want("train_tiny"):
            return QTimer.singleShot(100, step_train_novram)
        p = win.pages["nav.train"]
        p.model_form.form.set_value("n_layer", 6)
        p.model_form.form.set_value("n_embd", 384)
        p.model_form.form.set_value("n_head", 6)
        p._estimate()
        snap("train_tiny")
        print("     估算结论:", p.est_note.text())
        QTimer.singleShot(150, step_train_novram)

    def step_train_novram():
        """★ 关键状态：显存**还没探测到**时，不许给出「够用」这种结论。"""
        if not want("train_novram"):
            return QTimer.singleShot(100, step_parallel)
        saved = dict(ctx.gpu)
        ctx.gpu = {"ok": False, "vram_free": 0, "probing": True,
                   "torch": "检测中…", "device_short": "…"}
        p = win.pages["nav.train"]
        p._estimate()
        snap("train_novram")
        print("     无显存时结论:", p.est_note.text())
        ctx.gpu = saved
        QTimer.singleShot(150, step_parallel)

    def step_parallel():
        """大配置 + 显存建议 —— 关键是「省显存的办法」那段要出现。"""
        if not want("train_advice"):
            return QTimer.singleShot(100, step_done)
        win.goto("nav.train")
        p = win.pages["nav.train"]
        p.model_form.form.set_value("n_layer", 40)
        p.model_form.form.set_value("n_embd", 2048)
        p._estimate()
        snap("train_advice")
        print("     建议:", p.suggest_label.text().replace("\n", " | ")[:240])
        QTimer.singleShot(150, step_invalid)

    def step_invalid():
        """★ 关键状态：**不合法的配置**不许给估算。

        用户实测踩到的：`n_embd=230` 配 `n_head=8`（230 不能被 8 整除），
        界面照样显示「✓ 显存够用（约占 80%）」—— 可这配置连模型都建不出来。
        引擎会报「n_embd (230) 必须能被 n_head (8) 整除」。

        现在界面必须先跑引擎的校验器，有错就**只显示错误**。
        """
        if not want("train_invalid"):
            return QTimer.singleShot(100, step_done)
        win.goto("nav.train")
        p = win.pages["nav.train"]
        p.model_form.form.set_value("n_layer", 64)
        p.model_form.form.set_value("n_embd", 230)
        p.model_form.form.set_value("n_head", 8)
        p._estimate()
        snap("train_invalid")
        print("     非法配置下：错误框可见 =", p.error_label.isVisible(),
              "| 数值区可见 =", p.est_body.isVisible())
        print("     错误内容:", p.error_label.text().replace("\n", " | ")[:160])
        print("     开始训练按钮可用 =", p.start_btn.isEnabled())
        QTimer.singleShot(150, step_done)

    def step_done():
        app.quit()

    print("开始巡检……")
    QTimer.singleShot(int(a.settle * 1000), lambda: step_pages(0))
    app.exec()
    print("\n共 %d 张，都在 %s" % (len(done), a.out))


if __name__ == "__main__":
    main()
