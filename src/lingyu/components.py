# -*- coding: utf-8 -*-
r"""组件分发：按配置里的名字查出该用哪个实现。

## 这个文件**刻意不 import model**

组件注册是循环依赖的高发区：

    model.py       定义 Attention / SwiGLU / RMSNorm …
    components.py  按名字查实现
    注册动作       把 model 里的类登记进 registry

如果 components.py 去 `from .model import Attention`，而 model.py 又要用
components 里的查表函数，就成了循环导入（`partially initialized module`）。

解法是**反转依赖**：

    components.py   只知道「怎么查」，不知道「有什么」   ← 无 model 依赖
    model.py        顶部 import components 拿到查表函数
                    末尾把类登记进 registry（此时类都定义完了）

代价是「引擎支持哪些组件」这个问题的答案分散在两处：
这里的 `BUILTIN_*` 常量列出**名字**，model.py 末尾列出**实现**。
"""

from . import registry as R

# ---------------------------------------------------------------------------
# 内置组件的名字（**只列名字，不列实现** —— 实现见 model.py 末尾的注册）
#
# 为什么 mha / gqa / mqa 都指向同一份实现：
#     三者靠 n_kv_head 的取值区分，本来就是同一段代码
#         n_kv_head == n_head  ->  MHA
#         n_kv_head <  n_head  ->  GQA
#         n_kv_head == 1       ->  MQA
#     注册成一个类是刻意的，免得有人以为要分三处改代码。
# ---------------------------------------------------------------------------
BUILTIN_ATTENTION = ("gqa", "mha", "mqa")
BUILTIN_FFN = ("swiglu", "moe")
BUILTIN_NORM = ("rmsnorm", "layernorm")
BUILTIN_POS = ("rope",)

# 等价别名：配置里写这些名字统一按后面的处理
_ATTENTION_ALIASES = {"mha": "gqa", "mqa": "gqa", "auto": "gqa", "": "gqa"}


def get_attention(cfg):
    """按配置取注意力的实现类。

    `model.attention` 只决定**用哪个实现**；是 MHA 还是 GQA 由
    `n_kv_head` 决定（同一份代码）。名字不认识时报一个列出所有可用名字的错
    （含插件注册的），而不是一句 KeyError。
    """
    name = (cfg.get("attention") or "gqa").lower()
    name = _ATTENTION_ALIASES.get(name, name)
    return R.get_or_die("attention", name)


def _resolved_ffn_name(cfg):
    """算出这次到底该用哪个前馈。

    没配 `ffn` 时按 `use_moe` 推。配置里点名 `moe` 但忘了开开关 —— 帮它开上，
    免得用户困惑「为什么写了 moe 却没生效」。
    """
    name = (cfg.get("ffn") or "").lower()
    if not name:
        name = "moe" if cfg.get("use_moe") else "swiglu"
    if name == "moe" and not cfg.get("use_moe"):
        cfg["use_moe"] = True
    if name != "moe":
        cfg["use_moe"] = False
    return name


def get_ffn(cfg):
    return R.get_or_die("ffn", _resolved_ffn_name(cfg))


def get_norm(cfg):
    return R.get_or_die("norm", (cfg.get("norm") or "rmsnorm").lower())


def get_pos(cfg):
    return R.get_or_die("pos", (cfg.get("pos") or "rope").lower())


def describe_all():
    """给 doctor / GUI 用：列出所有组件及来源（builtin / plugin）。"""
    return R.describe()


# ---------------------------------------------------------------------------
# 参数估算：组件「每个特征维度花多少参数」
#
# 估算模型大小时需要知道一个组件的参数量公式，但引擎不该为每个组件写死公式
# （那就没法支持插件了）。约定：组件类上写一个类属性
#
#     PARAM_DIM_MULT = n      # 该组件参数量 ≈ n * 特征维度
#
# 内置例子：RMSNorm 只有 weight -> 1；LayerNorm 有 weight + bias -> 2。
# 没写的组件按 1 算（保守偏小，只在 build 之前的粗估里用到）。
#
# 注意：**这只是一个粗略系数**，不是精确公式 —— 注意力/前馈的参数跟
# d_ff、head 数都有关，各自在 model.py 里有专门的公式。这里只解决
# 「归一化层到底占 d 还是 2d」这一类一维组件。
# ---------------------------------------------------------------------------
def param_dim_mult(kind, name, default=1):
    """查某个已注册组件的一维参数系数；查不到就返回 default。"""
    try:
        cls = R.get(kind, name)
    except Exception:
        return default
    return int(getattr(cls, "PARAM_DIM_MULT", default) or default)


def norm_dim_mult(cfg, default=1):
    """归一化层的一维参数系数：RMSNorm=1，LayerNorm=2，插件看它自己写没写。"""
    return param_dim_mult("norm", (cfg.get("norm") or "rmsnorm").lower(), default)
