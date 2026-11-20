# -*- coding: utf-8 -*-
"""给界面截图（开发时用来看长得对不对）。

跑法：

    python gui/shot.py                  # 深色，全部页面
    python gui/shot.py --theme light    # 浅色
    python gui/shot.py --page nav.chat  # 只截一页

产物在 `_shots/` 下。**这个目录不进 git**（截图是开发过程的中间产物）。

为什么要专门写这个脚本：界面的问题（间距不齐、颜色太脏、文字被截断）
只有**真的看一眼**才能发现，靠读代码是看不出来的。

## 关于 offscreen 模式下的「豆腐块」

一开始用 `QT_QPA_PLATFORM=offscreen` 截图，所有中文都渲染成方框。
原因不是代码问题，而是 Qt 在 offscreen 模式下加载不了 `.ttc`
字体集合（微软雅黑就是 msyh.ttc），报
`FT_New_Face failed with index 0`，于是回退到一个没有中文字形的字体。

所以要**用原生平台**截图。为了不打扰用户，窗口会被挪到屏幕外
（负坐标），`grab()` 仍然能正常渲染。
"""
import argparse
import os
import sys
import time

# 注意：**不设** QT_QPA_PLATFORM，走原生平台才能渲染中文
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer            # noqa: E402
from PySide6.QtWidgets import QApplication   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="dark")
    ap.add_argument("--lang", default="zh_CN")
    ap.add_argument("--page", default=None)
    ap.add_argument("--out", default="_shots")
    ap.add_argument("--settle", type=float, default=6.0,
                    help="等环境探测完成的时间（秒）")
    a = ap.parse_args()

    app = QApplication([])

    from gui.context import ctx
    from gui.theme import stylesheet, th
    ctx.settings["theme"] = a.theme
    ctx.settings["lang"] = a.lang
    from gui.i18n import set_lang
    set_lang(a.lang)
    th.set(a.theme)
    app.setStyleSheet(stylesheet())

    from gui.app import MainWindow
    win = MainWindow()
    win.resize(1360, 880)
    # 挪到屏幕外：grab() 是直接渲染控件，不需要窗口真的可见，
    # 但窗口如果不"显示"过，某些控件不会完成布局。
    win.move(-4000, -4000)
    win.show()

    os.makedirs(a.out, exist_ok=True)
    keys = [a.page] if a.page else [k for _c, k, _g, _i in win.page_specs]
    saved = []

    def shoot(i=0):
        if i >= len(keys):
            app.quit()
            return
        k = keys[i]
        win.goto(k)
        win.repaint()
        app.processEvents()
        time.sleep(0.12)
        app.processEvents()
        name = "%s_%s.png" % (a.theme, k.replace("nav.", ""))
        p = os.path.join(a.out, name)
        win.grab().save(p)
        saved.append(p)
        QTimer.singleShot(220, lambda: shoot(i + 1))

    # 先等环境探测完（状态栏、概览数字要它）
    QTimer.singleShot(int(a.settle * 1000), lambda: shoot(0))
    app.exec()
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
