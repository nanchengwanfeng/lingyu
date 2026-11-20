# -*- coding: utf-8 -*-
"""配置系统。

一份 `spec.py` 描述所有参数，加载/校验/覆盖都从它推导。

    from lingyu.config import load_config, apply_overrides, get, validate

配置分 7 个段：
    [model]     模型结构
    [train]     预训练超参
    [data]      数据路径与切分
    [sample]    采样生成
    [post]      后训练（SFT / DPO / LoRA）
    [parallel]  分布式（大模型）
    [runtime]   界面语言、日志、监控
"""

from .loader import (load_config, save_config, patch_config, apply_overrides,
                     get, as_dict, resolve, missing_required, _KEY_ALIASES)
from .spec import (SPEC, SECTIONS, TIERS, COMPONENT_CHOICES, all_keys,
                   get_param, iter_params, make_ini_template)
from .validate import validate, ConfigError

__all__ = [
    "load_config", "save_config", "patch_config", "apply_overrides", "get",
    "as_dict", "resolve", "missing_required",
    "SPEC", "SECTIONS", "TIERS", "COMPONENT_CHOICES", "all_keys",
    "get_param", "iter_params", "make_ini_template",
    "validate", "ConfigError",
]
