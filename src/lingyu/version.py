# -*- coding: utf-8 -*-
"""LingYu 版本与格式常量。

这里的常量有两个用途：
    1. 让人知道自己在用哪个版本（报 bug 时要提供）
    2. **让断点能被安全校验** —— 不兼容时明确报错，绝不静默加载出错的结果
"""

# 引擎版本。改动了会影响用户的功能就 +1。
__version__ = "0.4.0"

# 断点格式版本。**只有 ckpt.pt 的结构变化时才 +1**，
# 普通功能更新不要动它，否则用户的旧存档会被判为不兼容。
CKPT_FORMAT = 1

# 配置文件格式版本
CONFIG_FORMAT = 1

# 分词器文件格式版本
TOKENIZER_FORMAT = 1

# 项目信息（打包与文档用）
PROJECT_NAME = "lingyu-lm"
PROJECT_URL = "https://github.com/nanchengwanfeng/lingyu"
