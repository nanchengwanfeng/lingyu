"""全局加速开关：只改「怎么算」，不改「算什么」。

这里集中处理三件与硬件相关、但对结果影响可控的设置：

1. **CUDA 后端**：TF32 + cuDNN benchmark。训练用的是 bf16 autocast，矩阵乘
   本来就走 tensor core；打开 TF32 主要照顾模型里残留的 fp32 算子（LayerNorm、
   部分归约），cuDNN benchmark 则在输入尺寸固定的训练里自动挑最快卷积/注意力实现。
2. **CPU 线程数**：默认取 `min(8, cpu_count)`。不设的话 PyTorch 在部分机器上
   会开满核，和训练进程抢线程，反而更慢。
3. **fp32 矩阵精度**：CPU 上设 `high`（等价 bf16 拆分加速），GPU 上由 TF32 接管。

环境变量：
    LINGYU_NO_ACCEL=1       完全关掉（排查「换设置后结果不同」时用）
    LINGYU_CPU_THREADS=N    手动指定 CPU 线程数

这些设置都是**进程级、幂等**的，重复调用不会叠加副作用；返回实际生效的项目，
调用方可以打印出来，便于用户确认「到底开了什么」。
"""

import os

import torch


def _env_off():
    return os.environ.get("LINGYU_NO_ACCEL", "").strip().lower() in (
        "1", "true", "yes", "on")


def _cpu_threads():
    raw = os.environ.get("LINGYU_CPU_THREADS", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n > 0:
            return max(1, min(256, n))
    return max(1, min(8, os.cpu_count() or 1))


def apply(device=None):
    """按设备打开加速开关。返回 {名字: 生效值}；被环境变量关掉时返回 {}。"""
    if _env_off():
        return {}
    applied = {}

    # CPU 线程：无论跑 CPU 还是 GPU 都要设（数据加载/分词也可能吃 CPU）
    n = _cpu_threads()
    try:
        torch.set_num_threads(n)
        applied["cpu_threads"] = n
    except Exception:
        pass

    # fp32 矩阵精度：high 在 Ampere+ 上走 TF32，CPU 上是 bf16 拆分
    try:
        torch.set_float32_matmul_precision("high")
        applied["matmul_precision"] = "high"
    except Exception:
        pass

    dev = str(device) if device is not None else ""
    if (dev.startswith("cuda") and torch.cuda.is_available()):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            applied["tf32"] = True
            applied["cudnn_benchmark"] = True
        except Exception:
            pass
    return applied


def describe(applied):
    """把 apply() 的返回值变成一行中文，供 CLI/日志使用。"""
    if not applied:
        return "已关闭（LINGYU_NO_ACCEL）"
    bits = []
    if "cpu_threads" in applied:
        bits.append("CPU 线程 %d" % applied["cpu_threads"])
    if applied.get("tf32"):
        bits.append("TF32")
    if applied.get("cudnn_benchmark"):
        bits.append("cuDNN benchmark")
    if applied.get("matmul_precision"):
        bits.append("fp32 精度 %s" % applied["matmul_precision"])
    if applied.get("compiled"):
        bits.append("torch.compile")
    return "、".join(bits) if bits else "无"


def _triton_available():
    """torch.compile 的默认后端 inductor 依赖 Triton；Windows 官方轮子没有。"""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def maybe_compile(model, device=None):
    """可选 torch.compile：LINGYU_COMPILE=1 时启用，失败自动退回 eager。

    默认关闭：Windows 上 Triton 支持不稳定，且首次编译要 20~60 秒，小模型
    训练大多跑不到摊回编译成本；但在长训练里它能把逐层 Python 开销抹掉。
    所以做成显式开关，让用户按训练时长决定；缺 Triton 时直接跳过并提示，
    绝不冒着「进程原生崩溃」的风险硬上。
    """
    if os.environ.get("LINGYU_COMPILE", "").strip().lower() not in (
            "1", "true", "yes", "on"):
        return model
    if not _triton_available():
        import sys
        print("[信息] LINGYU_COMPILE=1，但当前环境没有 Triton（Windows 常见），"
              "已跳过 torch.compile", file=sys.stderr)
        return model
    try:
        return torch.compile(model)
    except Exception:
        return model

