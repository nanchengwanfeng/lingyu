# -*- coding: utf-8 -*-
"""跨厂商设备探测（不绑定 NVIDIA）。

PyTorch 的“cuda”设备在 AMD ROCm 版里也叫 "cuda"（torch.version.hip 有值），
所以代码层面**不用改训练循环**就能跑 AMD；Apple 用 "mps"，Intel 用 "xpu"；
昇腾/摩尔线程等由官方插件提供 "npu"/"musa"/"privateuseone" 设备。

本模块只做探测与选择，不 import 可选插件（import 失败就算没有）。
诚实原则：不假装能"把 CPU、核显、独显同时吃满"——PyTorch 一个算子
只在一个设备上跑；能自动选到最快的那个，并把可用的列出来。
"""


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def available():
    """返回按优先级排好的 [(device, vendor, note)]。"""
    torch = _torch()
    out = []
    if torch is None:
        return out
    # 1) NVIDIA CUDA / AMD ROCm（PyTorch 里都叫 cuda）
    try:
        if getattr(torch.version, "cuda", None):
            if torch.cuda.is_available():
                hip = getattr(torch.version, "hip", None)
                if hip:
                    out.append(("cuda", "AMD", "ROCm %s" % hip))
                else:
                    try:
                        name = torch.cuda.get_device_name(0)
                    except Exception:
                        name = "CUDA GPU"
                    out.append(("cuda", "NVIDIA", name))
            else:
                out.append(("cuda", "NVIDIA", "CUDA 编译版但当前不可用（驱动/显卡）"))
    except Exception:
        pass
    # 2) Apple Silicon
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            out.append(("mps", "Apple", "Metal Performance Shaders"))
    except Exception:
        pass
    # 3) Intel XPU
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            out.append(("xpu", "Intel", "oneAPI / XPU"))
    except Exception:
        pass
    # 4) 昇腾 / 摩尔线程 / 其他 PrivateUse1 插件（装了才知道）
    for mod, dev, vendor in (("torch_npu", "npu", "Ascend"),
                             ("torch_musa", "musa", "Moore Threads")):
        try:
            __import__(mod)
            out.append((dev, vendor, "%s 插件已安装" % mod))
        except Exception:
            continue
    # 5) CPU 永远是兜底
    out.append(("cpu", "CPU", "fallback"))
    return out


def pick(prefer="auto"):
    """选择训练/推理设备：auto 时取列表第一个，否则验证用户指定值。"""
    devices = available()
    names = [d for d, _v, _n in devices]
    if prefer and prefer != "auto":
        if prefer in names:
            return prefer
        raise ValueError(
            "设备 %r 不可用；当前可用：%s" % (prefer, ", ".join(names)))
    return names[0] if names else "cpu"


def describe():
    """给 doctor 用的文字行。"""
    rows = []
    for dev, vendor, note in available():
        rows.append("  %-5s %-8s %s" % (dev, vendor, note))
    return rows
