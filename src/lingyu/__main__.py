# -*- coding: utf-8 -*-
'''让 `python -m lingyu` 能直接当命令行用。'''

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
