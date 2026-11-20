# -*- coding: utf-8 -*-
"""`python -m gui` —— 直接启动界面（和 `lingyu gui` 等价）。"""

import sys

from .app import run

if __name__ == "__main__":
    sys.exit(run())
