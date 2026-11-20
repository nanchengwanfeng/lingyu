# -*- coding: utf-8 -*-
r"""灵语图形界面（PySide6）。

## 一条最重要的架构约束

**这个包里任何模块级代码都不许 import torch。**

理由：`import torch` 在这台机器上要 3~8 秒，而且一建 CUDA 上下文就常驻
几百 MB 显存。界面进程只需要画窗口，不需要张量。所有要 torch 的活
（训练/微调/采样/聊天）都交给 `lingyu` 命令在**子进程**里跑。

这条规矩让「8GB 的机器也能秒开界面」成为可能。启动要 2 秒以上，
用户就会觉得这个软件很笨重。

验证方式（改完界面请跑一下）：

    python -c "import sys,time; t=time.time(); import gui; \
               print('import 用时 %.2fs' % (time.time()-t))"

只要不是在 1 秒以内，就说明有模块级 import torch 混进来了。
"""

__all__ = ["run"]


def run(argv=None):
    """启动 GUI。等价于命令 `lingyu gui`。"""
    from .app import run as _run
    return _run(argv)
