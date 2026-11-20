# -*- coding: utf-8 -*-
"""执行后端：**同一个配置，不同的跑法**。

这是「一个引擎从 13M 到万亿参数」这句话的落地点。

```
                    ┌──────────────────────────────┐
   同一份配置  ──▶  │  模型定义 / 数据 / 训练循环   │  ← 完全不变
                    └──────────────┬───────────────┘
                                   │ 只换这一层
                    ┌──────────────▼───────────────┐
                    │  backends/  执行后端          │
                    │    local       单卡（已实现）  │
                    │    fsdp        单机多卡（实验性：│
                    │                仅生成脚本）     │
                    │    deepspeed   多机（生成脚本） │
                    │    megatron    超大（只翻译参数）│
                    └──────────────────────────────┘
```

## 三条不能破的规矩

1. **诚实**。`backend_info()` 里明确标出每个后端是「已实现」、
   「只生成配置」还是「不实现」。文案不许吹牛 —— 用户看到
   「支持万亿参数」却发现跑不起来，比明说「不支持」伤害大得多。

2. **算得出来**。超出本机能力时，必须给出「需要几张卡」这个**具体数字**，
   而不是一句「太大了」。数字来自 `plan.py`，全项目只有这一份估算器。

3. **不重写别人的东西**。DeepSpeed 和 Megatron 是成熟的大型项目。
   引擎只做「配置翻译 + 脚本生成」，把参数换算这件烦人的事做掉，
   真正的训练交给它们。自己造一个半成品只会害了用它的人。
"""

from .plan import (bytes_per_param, estimate, fmt_bytes, gpu_info,
                   check_available_vram, suggestions)
from .launch import (backend_info, generate, write_outputs, GENERATORS)

# 后端名字 -> 是否由本引擎真正实现。
# ⚠️ fsdp 目前**只是脚本生成**（实验性）：训练循环里还没有分布式代码，
#    生成的 torchrun 脚本会让 N 个进程各自做单卡训练。别把它列进来。
IMPLEMENTED = ("local",)

__all__ = [
    "bytes_per_param", "estimate", "fmt_bytes", "gpu_info",
    "check_available_vram", "suggestions",
    "backend_info", "generate", "write_outputs", "GENERATORS",
    "IMPLEMENTED", "backend_is_implemented", "list_backends",
]


def backend_is_implemented(name):
    """这个后端是引擎自己实现的吗？（决定能不能直接 `lingyu train`）"""
    return name in IMPLEMENTED


def list_backends():
    """所有后端及真实支持程度。给 doctor 和 GUI 下拉框用。"""
    out = []
    for name in ("local", "fsdp", "deepspeed", "megatron"):
        info = backend_info(name)
        out.append({"name": name, "label": info["label"],
                    "implemented": backend_is_implemented(name),
                    "summary": info["summary"]})
    return out
