# -*- coding: utf-8 -*-
r"""灵语启动器 —— 那个「双击就用」的 .exe。

## 它要解决的核心问题

用户想要「一个文件夹，一个 .exe，双击就能用，运行时/配置/数据都在子目录里」。
但这里有个绕不过去的物理事实：

    **PyTorch + CUDA 有 2.5 GB 以上，而且没法塞进 PyInstaller 的单文件包里。**

塞进去会怎样：打出来的 exe 有 2.6 GB，而且**每次启动都要把整包解压到临时目录**
（单文件模式的机制），开一次等半分钟。这不是「开箱即用」，是「开箱等死」。

所以做法是分成两层：

    LingYu.exe        约 30 MB，只包含启动器和一份引擎源码
    runtime/          首次运行时创建，Python 环境和依赖都装在这里
      python/         独立 Python（本机没有合适的才下载）
      venv/           装了 torch / numpy / PySide6 的虚拟环境
      app/            引擎源码（从 exe 里解出来）
      configs/        配置文件（用户可以直接改）
      data/  out/     语料和训练输出
      launcher.log    出问题看这个

第一次双击会走一个向导：找 Python → 建环境 → 装依赖（显示进度）→ 解包引擎。
之后每次双击就直接启动界面，不再重复任何步骤。

## 三条设计原则

1. **每一步都要有进度和日志**。装 2.5 GB 的 torch 要几分钟，
   界面卡住不动用户就会以为死了然后强杀 —— 那会留下半个坏环境。
2. **失败要说人话**。网络断了、磁盘满了、Python 版本不对，
   都要给出「发生了什么 + 怎么办」，而不是一个 traceback。
3. **不重复劳动**。已经装好的步骤要跳过；用版本号文件记录装的是哪一版。

## 只用标准库 + tkinter

启动器本身不能依赖 PySide6 —— 因为「装 PySide6」正是它要做的事之一，
这就成了先有鸡还是先有蛋。tkinter 是 CPython 自带的，
PyInstaller 打包进去也只有几 MB。
"""

import argparse
import ctypes
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile

APP_NAME = "LingYu"
APP_ZH = "灵语"
# 启动器自己的版本。发版时和引擎版本一起改（`--version` 会同时打印两者）。
LAUNCHER_VERSION = "0.4.0"
# 依赖装好后写这个文件；里面的版本号变了就重装依赖
STATE_FILE = ".installed.json"
STATE_VERSION = 1

# 引擎需要的 Python 版本范围（torch 的 wheel 只覆盖这几个版本）
PY_MIN = (3, 10)
PY_MAX = (3, 12)

# PyPI 镜像。第一个是清华源（国内快），第二个是官方源（兜底）。
# 顺序很重要：国内直连 pypi.org 装 2.5GB 的 torch 可能要几小时。
INDEXES = [
    ("清华 TUNA", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
    ("官方 PyPI", "https://pypi.org/simple"),
]

# 要装的包。**torch 单独处理**，见 TORCH_INDEXES。
#
# ⚠️ 这里曾经把 torch 和 numpy 放在一起，用普通 PyPI 镜像装 ——
#    结果在 Windows 上装到了 **CPU 版**（PyPI 上 Windows 的 torch
#    wheel 就是 CPU-only 的，CUDA 版在 PyTorch 自己的索引里）。
#    后果是这个「GPU 训练引擎」装出来根本没有 GPU：
#    界面上显示 `PyTorch 2.14.0+cpu`、`CPU（没有可用显卡）`，
#    训练和聊天全崩。实测被用户当场撞上。
REQUIREMENTS = [
    ("numpy", "基础数值库"),
    ("PySide6-Essentials", "图形界面（LGPL）"),
]

# torch 的安装源，按顺序试。
#
# 每一项是 (说明, 地址, 用法)：
#   用法 "index" = PEP 503 索引，传给 `-i`
#   用法 "links" = 一个平铺的 wheel 目录，传给 `--find-links`
#
# ==========================================================================
# ★★ 顺序：**国内镜像优先，官方最后**。这里踩过一个很贵的坑，别改回去。
# ==========================================================================
#
# 原来写的是「先官方（CDN 最快也最全），再国内镜像兜底」。
# 那个判断是**想当然**的。本机（国内家宽）实测同一个文件
# `torch-2.14.0+cu130-cp312-cp312-win_amd64.whl`：
#
#     官方 download.pytorch.org/whl/cu130     0.22 MB/s   ← 最慢
#     官方 download.pytorch.org/whl/cu128     0.27 MB/s
#     阿里 mirrors.aliyun.com/pytorch-wheels  0.57 MB/s
#     上交 mirror.sjtu.edu.cn/pytorch-wheels 10.05 MB/s   ← 快 45 倍
#
# 1990 MB 的 torch wheel：官方 **2.5 小时**，上交 **3.3 分钟**。
#
# 更糟的是「兜底」根本不会触发 —— 镜像只在**失败**时才会被试到，
# 而慢不是失败：pip 会一直以 0.22 MB/s 爬下去，用户看到的是
# "下载中…" 卡几十分钟不动，然后以为程序死了。
# 实测就是这么卡住的（3 分钟写了 0 字节）。
#
# 所以：**慢 = 不可用**，必须排到前面去试。安全性由装完的
# `torch_cuda_ok()` 验证兜着 —— 哪个源给了坏轮子/没有轮子，
# 校验不过就自动试下一个，所以「镜像优先」不会引入风险。
#
# --------------------------------------------------------------------------
# ★ 而且 `"index"` 还是 `"links"` **必须看那个地址的真实形态**，猜不得。
# --------------------------------------------------------------------------
# 踩过的坑：上交那条一开始写成了 `"links"`（想当然：镜像站不都是平铺目录吗）。
# 结果 pip 用 `--find-links` 去扫，**一条 .whl 都没扫到**，
# 于是只能退回去用默认 PyPI 索引里那个 CPU 版 torch
# （`torch-2.14.0-cp312-cp312-win_amd64.whl`，124 MB，没有 +cu130）。
#
# 实测两个镜像的形态**完全不同**：
#
#   上交 /pytorch-wheels/cu130        PEP 503 索引（`pypi:repository-version`），
#                                    子目录 118 个，0 条直接 .whl  → "index"
#                                    torch 依赖齐全（filelock / sympy /
#                                    networkx / jinja2 / fsspec 全 200）
#   阿里 /pytorch-wheels/cu130        平铺 wheel 目录，993 条直接 .whl → "links"
#
# 判断方法：
#   curl -s <地址> | head       看有没有 `pypi:repository-version`
#
# 用错的后果不是报错，而是**悄悄装成 CPU 版** —— 又回到那个
# 「GPU 引擎装出来没有 GPU」的老问题上，只是这次从另一个门进来。
TORCH_INDEXES = [
    ("上海交大镜像 cu130", "https://mirror.sjtu.edu.cn/pytorch-wheels/cu130",
     "index"),
    ("PyTorch 官方 cu130", "https://download.pytorch.org/whl/cu130", "index"),
    ("PyTorch 官方 cu128", "https://download.pytorch.org/whl/cu128", "index"),
    ("阿里云镜像 cu130", "https://mirrors.aliyun.com/pytorch-wheels/cu130",
     "links"),
]

# CUDA 版装不上时退回 CPU 版的提示。**必须说得很清楚** ——
# 用户拿到一个「跑不了 GPU 的 GPU 引擎」，不说清楚他会以为引擎坏了。
CPU_FALLBACK_NOTE = """\
没能装上 CUDA 版 PyTorch，已退回 **CPU 版**。

这意味着：只能用 CPU 训练和推理，慢几十倍，界面里也会显示
「CPU（没有可用显卡）」。

想换成 CUDA 版，等网络好的时候执行（把 <runtime> 换成安装目录）：

  <runtime>\\venv\\Scripts\\python.exe -m pip install torch ^
      --index-url https://download.pytorch.org/whl/cu130 --force-reinstall

装完检查一下：

  <runtime>\\venv\\Scripts\\python.exe -c "import torch;print(torch.cuda.is_available())"

打印 True 就成了。"""

# 找系统 Python 时看这些地方。
#
# ⚠️ 这里必须尽量周全，因为「本机已经有 Python 却找不到」的代价是
#    白白下载 30 MB —— 更糟的是用户会觉得「它连我的 Python 都认不出来」。
#
# 实测踩过：C:\path\to\lingyu\python312\cpython-3.12.14-...\python.exe
# （uv 装的 Python 会长这样，多一层版本号目录）漏掉了，
# 于是本机明明有个好好的 3.12，启动器却只找到 3.14（torch 不支持）→
# 判定「没有可用的 Python」→ 去下载。所以下面既有 uv 的固定目录，
# 也有 `*/*/cpython-3*` 这种两级通配。
COMMON_PY_DIRS = [
    r"C:\Python3*",
    r"C:\Program Files\Python3*",
    r"%LOCALAPPDATA%\Programs\Python\Python3*",
    r"%USERPROFILE%\AppData\Local\Programs\Python\Python3*",
    r"D:\Python3*",
    # uv 管理的独立 Python（uv python install 的默认位置）
    r"%LOCALAPPDATA%\uv\python\cpython-3*",
    r"%USERPROFILE%\AppData\Local\uv\python\cpython-3*",
    r"%APPDATA%\uv\python\cpython-3*",
    # uv 装在自定义目录的情况：<任意>/<任意>/cpython-3*（两级通配）
    r"D:\*\*\cpython-3*",
    r"C:\*\*\cpython-3*",
    r"D:\*\cpython-3*",
    # 其他常见位置
    r"C:\ProgramData\miniconda3",
    r"C:\ProgramData\anaconda3",
    r"%USERPROFILE%\miniconda3",
    r"%USERPROFILE%\anaconda3",
    r"%USERPROFILE%\scoop\apps\python\current",
]


# ==========================================================================
# 日志
# ==========================================================================
class Log:
    def __init__(self, path=None):
        self.lines = []
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, msg, level="info"):
        stamp = time.strftime("%H:%M:%S")
        line = "[%s] %s" % (stamp, msg)
        with self.lock:
            self.lines.append(line)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as fp:
                        fp.write(line + "\n")
                except OSError:
                    pass
        # 打印**绝不能**因为编码问题抛异常 —— 安装都跑完了，
        # 结果崩在最后一行进度输出上，用户会以为安装失败了。
        try:
            print(line, file=sys.stderr if level == "error" else sys.stdout)
        except (UnicodeEncodeError, OSError, ValueError):
            try:
                print(line.encode("ascii", "replace").decode("ascii"))
            except Exception:
                pass


log = Log()


# ==========================================================================
# 路径
# ==========================================================================
def exe_dir():
    """exe（或脚本）所在的目录 —— 安装根就定在这里。

    注意不能用 `sys.argv[0]`：双击时它是相对路径，从别处调用时还可能
    带一堆参数。用 `sys.executable` 在打包后可靠，开发时退回脚本目录。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def bundle_dir():
    """PyInstaller 解包出来的临时目录（开发时就是项目根）。"""
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))


def runtime_dir(root):
    return os.path.join(root, "runtime")


def find_bundled_source():
    """找到打包进 exe 的引擎源码包。

    PyInstaller 会把 `--add-data` 指定的东西放到 `sys._MEIPASS` 下。
    开发时没有这个东西，就直接用项目目录。
    """
    bd = bundle_dir()
    for name in ("lingyu_src.zip", "app_src.zip"):
        p = os.path.join(bd, name)
        if os.path.isfile(p):
            return p
    # 开发模式：项目根就是源码
    if os.path.isdir(os.path.join(bd, "src", "lingyu")):
        return bd
    return None


# --------------------------------------------------------------------------
# 版本号
# --------------------------------------------------------------------------
def _read_engine_version_from_text(text):
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text or "")
    return m.group(1) if m else None


def _engine_version():
    """读引擎版本（`lingyu.__version__`），读不到返回 None。

    找的顺序：
      1. `runtime/app/src/lingyu/version.py`（已经解包过的安装）
      2. exe 里内嵌的源码包（首次运行前根本没有 runtime）
      3. 开发模式：项目里的 `src/lingyu/version.py`
    """
    cand = os.path.join(runtime_dir(exe_dir()), "app", "src", "lingyu",
                        "version.py")
    try:
        with open(cand, "r", encoding="utf-8") as fp:
            v = _read_engine_version_from_text(fp.read())
        if v:
            return v
    except OSError:
        pass

    src = find_bundled_source()
    if src:
        if src.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(src) as z:
                    for name in z.namelist():
                        if name.replace("\\", "/").endswith(
                                "src/lingyu/version.py"):
                            v = _read_engine_version_from_text(
                                z.read(name).decode("utf-8", "replace"))
                            if v:
                                return v
            except (OSError, zipfile.BadZipFile):
                pass
        else:
            try:
                with open(os.path.join(src, "src", "lingyu", "version.py"),
                          "r", encoding="utf-8") as fp:
                    return _read_engine_version_from_text(fp.read())
            except OSError:
                pass
    return None


def version_text():
    """`--version` 的一行输出：启动器版本 + 引擎版本。"""
    engine = _engine_version()
    return "灵语 LingYu 启动器 v%s（引擎 v%s）" % (
        LAUNCHER_VERSION, engine or "未知（首次安装后才能读）")


# ==========================================================================
# Python 探测
# ==========================================================================
def python_version_of(exe):
    """跑一下这个 python，返回 (major, minor) 或 None。"""
    try:
        out = subprocess.run(
            [exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20,
            creationflags=_no_window())
        txt = out.stdout.decode("utf-8", "replace").strip()
        m = re.match(r"^(\d+)\.(\d+)$", txt)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _no_window():
    """Windows 上跑子进程时不要弹黑框。"""
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return 0


def is_usable(exe):
    v = python_version_of(exe)
    if v is None:
        return False, None
    if PY_MIN <= v <= PY_MAX:
        # 还要确认是 64 位 —— 32 位 Python 装不了 torch
        try:
            out = subprocess.run(
                [exe, "-c", "import struct;print(struct.calcsize('P')*8)"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20,
                creationflags=_no_window())
            bits = int(out.stdout.decode().strip())
            if bits != 64:
                return False, "是 %d 位 Python，装不了 PyTorch" % bits
        except Exception:
            pass
        return True, None
    return False, "版本是 %d.%d，需要 %d.%d ~ %d.%d" % (
        v[0], v[1], PY_MIN[0], PY_MIN[1], PY_MAX[0], PY_MAX[1])


def candidate_pythons(root):
    """按优先级列出候选 Python。

    顺序：已经装在 runtime 里的 → 环境变量 → PATH → 常见安装位置。
    **先找本机已有的**是最省事的：不用下载几百 MB，几秒钟就能建好环境。
    """
    out = []
    rt = runtime_dir(root)

    # 1) runtime 里已经有的
    for pat in ("python/python.exe", "python/*/python.exe",
                "python/install/python.exe"):
        import glob
        for p in glob.glob(os.path.join(rt, pat.replace("/", os.sep))):
            out.append(("runtime 内置", p))

    # 2) 用户显式指定
    env = os.environ.get("LINGYU_PYTHON")
    if env and os.path.isfile(env):
        out.append(("环境变量 LINGYU_PYTHON", env))

    # 3) PATH 上的 python
    for name in ("python3.12", "python3.11", "python3.10",
                 "python", "python3"):
        p = shutil.which(name)
        if p:
            out.append(("PATH", p))

    # 4) 问 uv：它自己装的 Python 在哪
    #
    #    ★ 这一条比下面那堆通配目录靠谱得多。uv 装的 Python 放在
    #      `<某处>/<某处>/<某处>/cpython-3.12.14-windows-x86_64-none/`，
    #      层级不定、目录名还带完整版本号，靠通配去猜总会漏。
    #      `uv python find` 直接问它就行。
    #      实测：本机明明有个 3.12，通配没找到，uv 一问就出来了。
    uv = find_uv(root)
    if uv:
        for want in ("3.12", "3.11", "3.10"):
            try:
                r = subprocess.run([uv, "python", "find", want],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, timeout=30,
                                   creationflags=_no_window())
                if r.returncode == 0:
                    p = r.stdout.decode("utf-8", "replace").strip()
                    if p and os.path.isfile(p):
                        out.append(("uv 登记", p))
            except (OSError, subprocess.SubprocessError):
                break

    # 5) 常见安装目录（兜底）
    import glob
    for pat in COMMON_PY_DIRS:
        pat = os.path.expandvars(pat)
        for d in glob.glob(pat):
            for rel in ("python.exe", os.path.join("bin", "python.exe"),
                        os.path.join("bin", "python")):
                p = os.path.join(d, rel)
                if os.path.isfile(p):
                    out.append(("常见位置", p))

    # 去重（保留先出现的）
    seen, uniq = set(), []
    for label, p in out:
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((label, p))
    return uniq


def pick_python(root, progress):
    """挑一个能用的 Python。返回 (exe, 来源说明) 或 (None, 原因列表)。"""
    reasons = []
    for label, p in candidate_pythons(root):
        ok, why = is_usable(p)
        if ok:
            progress("找到可用的 Python：%s（%s）" % (p, label))
            return p, label
        reasons.append("%s（%s）：%s" % (p, label, why))
    return None, reasons


# ==========================================================================
# 用 uv 装 Python
# ==========================================================================
def find_uv(root):
    """找 uv：先在 exe 同目录找（发布时一起带），再找 PATH。"""
    bd = bundle_dir()
    for cand in (os.path.join(bd, "uv.exe"),
                 os.path.join(exe_dir(), "uv.exe"),
                 os.path.join(runtime_dir(root), "uv.exe")):
        if os.path.isfile(cand):
            return cand
    p = shutil.which("uv")
    return p


def install_python_with_uv(root, progress):
    """用 uv 下载一个独立 Python 到 runtime/python。

    为什么用 uv 而不是官方的 embeddable zip：embeddable 包里**没有 pip**，
    还得再去下 get-pip.py 并手工改 `._pth` 才能用，链路上任何一步
    出问题都很难诊断。uv 一条命令搞定，而且它自己就是单个 exe。
    """
    uv = find_uv(root)
    if not uv:
        return None, "没有找到 uv（发布包里应该带一个 uv.exe）"
    dest = os.path.join(runtime_dir(root), "python")
    os.makedirs(dest, exist_ok=True)
    progress("正在用 uv 下载 Python 3.12（约 30 MB）……")
    # 国内镜像：uv 支持指定 python-build-standalone 的镜像地址
    env = dict(os.environ)
    mirror = os.environ.get("LINGYU_UV_PYTHON_MIRROR")
    if mirror:
        env["UV_PYTHON_INSTALL_MIRROR"] = mirror
    try:
        r = subprocess.run([uv, "python", "install", "3.12",
                            "--install-dir", dest, "--no-bin"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=1800, env=env, creationflags=_no_window())
    except subprocess.TimeoutExpired:
        return None, "下载 Python 超时（30 分钟）"
    out = r.stdout.decode("utf-8", "replace")
    for line in out.strip().splitlines()[-6:]:
        progress("  " + line)
    if r.returncode != 0:
        return None, "uv 下载 Python 失败（退出码 %d）" % r.returncode

    import glob
    for p in glob.glob(os.path.join(dest, "**", "python.exe"), recursive=True):
        ok, _ = is_usable(p)
        if ok:
            return p, "uv 下载"
    return None, "uv 说装好了，但没找到可用的 python.exe"


# ==========================================================================
# 依赖安装
# ==========================================================================
def make_venv(base_python, root, progress):
    """建虚拟环境。用 venv 而不是直接往 Python 里装 —— 免得污染系统环境。

    ⚠️ 这里**不能**加 `--seed`。`--seed` 是 **uv** 的参数
    （`uv venv --seed`），标准库的 `venv` 不认它，会直接报
    `unrecognized arguments: --seed` 然后整个安装失败。

    标准库 venv 默认就会用 ensurepip 把 pip 装上，不需要额外要求。

    ## `LINGYU_VENV`：复用已有的环境

    设了这个环境变量就直接用它，不新建、不装依赖。用途：

    - **你自己已经有一个装好 torch 的环境**（比如从源码装过），
      不想为了试用打包版再下 2.5 GB
    - 公司内网有统一的环境，不想每台机器各装一份

    这算进阶用法，所以只在环境变量里支持，不放进图形向导 ——
    新手不需要看见它。
    """
    reuse = os.environ.get("LINGYU_VENV")
    if reuse:
        py = reuse
        if os.path.isdir(reuse):
            for rel in (("Scripts", "python.exe"), ("bin", "python")):
                cand = os.path.join(reuse, *rel)
                if os.path.isfile(cand):
                    py = cand
                    break
        if not os.path.isfile(py):
            raise RuntimeError(
                "LINGYU_VENV 指向的路径里没有 python：%s\n"
                "应该指到虚拟环境目录（里面有 Scripts\\python.exe），"
                "或者直接指到 python.exe。" % reuse)
        progress("使用 LINGYU_VENV 指定的环境：%s" % py)
        return py

    venv = os.path.join(runtime_dir(root), "venv")
    py = os.path.join(venv, "Scripts", "python.exe")
    if os.path.isfile(py):
        progress("虚拟环境已存在，跳过创建")
        return py
    progress("正在创建虚拟环境……")
    os.makedirs(os.path.dirname(venv), exist_ok=True)
    r = subprocess.run([base_python, "-m", "venv", venv],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=600, creationflags=_no_window())
    if r.returncode != 0 or not os.path.isfile(py):
        raise RuntimeError("创建虚拟环境失败：\n" +
                           r.stdout.decode("utf-8", "replace")[-800:])
    progress("虚拟环境建好了")
    return py


PIP_IDLE_TIMEOUT = 300      # 秒；pip 超过这么久没有任何输出 = 卡住，换源


def pip_install(venv_py, packages, index_url, progress, use_find_links=False):
    """装包，顺便把输出实时转成进度。

    `use_find_links=True` 时把地址当 `--find-links` 用（平铺的 wheel 目录），
    并且**同时加 `--no-index`**：不加的话 pip 还会去默认 PyPI 里找 torch，
    而 PyPI 上 Windows 的 torch 恰好是 CPU-only —— 那就是「静默装成 CPU 版」
    的另一扇门（2026-09-25 实装事故的根因之一）。平铺目录只放 torch，
    torch 的依赖由 `install_torch()` 事先从普通镜像装好，见 TORCH_DEPS。

    ## ★ 为什么要「读线程 + 队列」而不是直接 readline

    历史事故：镜像以 0.22 MB/s 爬（1990 MB 要两个多小时），或者干脆 0 字节
    卡住 —— 而 pip 不报错、进度条也不动。只把镜像顺序排快治不了「慢/卡」，
    「兜底」也只在失败时触发。这里给输出加**空闲超时**：300 秒没有新输出
    就 kill 掉、抛 RuntimeError，让上层换下一个源。
    """
    cmd = [venv_py, "-m", "pip", "install", "--disable-pip-version-check",
           "--no-input", "--upgrade"]
    if index_url:
        if use_find_links:
            # 平铺 wheel 目录没有依赖索引，只让 pip 在这里找 torch 本体；
            # 依赖已由 install_torch() 预先从普通 PyPI 镜像装好。
            cmd += ["--find-links", index_url, "--no-index"]
        else:
            cmd += ["-i", index_url]
    cmd += packages
    progress("运行：pip install %s%s" % (" ".join(packages),
                                       "" if not index_url else
                                       "（源：%s）" % index_url))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            creationflags=_no_window())
    tail = []
    q = queue.Queue()

    def _reader():
        try:
            for raw in iter(proc.stdout.readline, b""):
                q.put(raw)
        finally:
            q.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    while True:
        try:
            raw = q.get(timeout=PIP_IDLE_TIMEOUT)
        except queue.Empty:
            proc.kill()
            raise RuntimeError(
                "pip 装 %s 失败：%d 秒没有任何输出（网络卡住？），"
                "已中止并换下一个源。最后输出：\n%s"
                % (", ".join(packages), PIP_IDLE_TIMEOUT,
                   "\n".join(tail[-12:])))
        if raw is None:
            break
        line = raw.decode("utf-8", "replace").rstrip()
        if not line:
            continue
        tail.append(line)
        tail[:] = tail[-40:]
        # pip 的下载进度行很长且会不停刷新，挑有意义的显示
        if line.startswith(("Downloading", "Collecting", "Installing",
                            "Successfully", "ERROR", "Building")):
            progress("  " + line[:160])
        elif "%" in line and "|" in line:
            progress("  " + line[:120])
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("pip 装 %s 失败（退出码 %d）：\n%s"
                           % (", ".join(packages), proc.returncode,
                              "\n".join(tail[-12:])))
    progress("装好了：%s" % ", ".join(packages))


def has_package(venv_py, mod):
    r = subprocess.run([venv_py, "-c", "import %s" % mod],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=180, creationflags=_no_window())
    return r.returncode == 0


def torch_cuda_ok(venv_py):
    """检查装上的 torch 是不是**能用 CUDA**。

    返回 (状态, 说明) —— 状态分三档，这是刻意区分的：

        "cuda"          能用 GPU
        "cpu_wheel"     装的是 **CPU 版轮子**（PyPI 默认给的就是它）→ 重装就好
        "cuda_unusable" 装的是 **CUDA 版轮子**，但 is_available() 为 False
                        → 驱动太老 / 显卡不支持这个 CUDA 版本，重装同版本没用
        "none"          压根没装上（或 import 失败）

    ★ 以前这两种「不可用」都返回 "cpu"，上层无法判断该重装还是该让用户
      更新驱动 —— 用户实测撞上过「装的是 CPU 版」和「驱动太老」被一句话
      带过，不知道该干嘛（见 docs/DESIGN.md 第 9 节事故记录）。
    """
    code = ("import torch;"
            "print(torch.__version__);"
            "print(1 if torch.cuda.is_available() else 0);"
            "print(torch.version.cuda or 'None')")
    try:
        r = subprocess.run([venv_py, "-c", code], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=300,
                           creationflags=_no_window())
    except (OSError, subprocess.SubprocessError) as e:
        return "none", str(e)
    if r.returncode != 0:
        return "none", r.stdout.decode("utf-8", "replace")[-300:]
    lines = r.stdout.decode("utf-8", "replace").strip().splitlines()
    ver = lines[0].strip() if lines else "?"
    usable = len(lines) > 1 and lines[1].strip() == "1"
    cuda_build = lines[2].strip() if len(lines) > 2 else "None"
    if usable:
        return "cuda", ver
    if cuda_build and cuda_build != "None":
        return "cuda_unusable", "%s（CUDA %s 轮子，但驱动/显卡不可用）" % (
            ver, cuda_build)
    # 没有 CUDA 运行时版本 = 拿到的是 CPU 版轮子
    return "cpu_wheel", ver


# torch 的运行依赖。`--find-links` 源（平铺 wheel 目录）里只有 torch 本体，
# 加 `--no-index` 后这些依赖 pip 没法自己解析，只能先从普通 PyPI 镜像装好。
TORCH_DEPS = ("filelock", "typing-extensions", "sympy", "networkx",
              "jinja2", "fsspec")


def _ensure_torch_deps(venv_py, progress):
    """在走 `--find-links` 源之前，先把 torch 的依赖从普通镜像装好。

    失败不致命（可能已经存在），继续试；真缺依赖时 pip 会在下一步报
    「找不到 XXX」，上层换源/兜底逻辑照常生效。
    """
    progress("  先补 torch 的依赖（平铺源里没有依赖索引）……")
    for label, url in INDEXES:
        try:
            pip_install(venv_py, list(TORCH_DEPS), url, progress)
            return
        except RuntimeError:
            progress("  %s 不行，换下一个镜像补依赖……" % label)


def install_torch(venv_py, progress, index_url=None):
    """单独装 torch，**必须走 CUDA 索引**。

    返回 True 表示装上了 CUDA 版；False 表示只能用 CPU 版。
    """
    progress("正在安装 PyTorch（CUDA 版，约 2.5 GB，这一步最慢）……")

    # 用户显式指定了 PyPI 镜像时，仍然要给 torch 用 CUDA 源 ——
    # 否则一定装成 CPU 版。所以这里不吃 index_url。
    last_err = None
    for label, url, kind in TORCH_INDEXES:
        try:
            progress("用 %s" % label)
            if kind == "links":
                # 平铺目录 + --no-index：先把依赖装好，否则解析不出依赖
                _ensure_torch_deps(venv_py, progress)
            pip_install(venv_py, ["torch"], url, progress,
                        use_find_links=(kind == "links"))
            state, detail = torch_cuda_ok(venv_py)
            if state == "cuda":
                progress("PyTorch %s（CUDA 可用）✓" % detail, "ok")
                return True
            if state == "cuda_unusable":
                # 换 cuXXX 源有意义：可能只是这个 CUDA 版本比驱动新
                last_err = "装上了 %s，但 CUDA 不可用（驱动可能太老）" % detail
            elif state == "none":
                last_err = "torch 装上后导入失败：%s" % detail
            else:
                last_err = "装上了 %s，但它是 CPU 版轮子" % detail
            progress("  这个源给的不是可用的 CUDA 版（%s），换下一个……"
                     % detail)
        except RuntimeError as e:
            last_err = str(e)
            progress("  这个源不行，换下一个……")

    # ---- 退回 CPU 版 ----
    progress("CUDA 版都没装上（%s），退回 CPU 版……" % last_err, "warn")
    pyi = index_url or INDEXES[0][1]
    try:
        pip_install(venv_py, ["torch"], pyi, progress)
    except RuntimeError:
        # 有些 PyPI 镜像干脆不带 torch，再试官方
        pip_install(venv_py, ["torch"], "https://pypi.org/simple", progress)
    state, detail = torch_cuda_ok(venv_py)
    if state == "cuda":
        # 兜底反而装成了 CUDA 版（比如本机已有 CUDA 轮子）—— 如实报
        progress("PyTorch %s（CUDA 可用）✓" % detail, "ok")
        return True
    if state == "none":
        raise RuntimeError(
            "PyTorch 装上后导入失败：%s\n"
            "通常是 VC++ 运行库缺失或 wheel 与 Python 版本不匹配。\n"
            "可以手动装：pip install torch --index-url "
            "https://download.pytorch.org/whl/cu130" % detail)
    progress("装上了 PyTorch %s，但 **CUDA 不可用**（%s）。" % (detail, state),
             "warn")
    for line in CPU_FALLBACK_NOTE.splitlines():
        progress("  " + line, "warn")
    return False


# ==========================================================================
# 解包引擎源码
# ==========================================================================
def _source_stamp(src):
    """源码包的指纹。用来判断「exe 换代了，但 runtime/app 还是旧的」。"""
    h = hashlib.sha256()
    try:
        with open(src, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "unknown"
    return h.hexdigest()[:16]


#: app 下这些子目录是**纯源码**，重新解包前先删掉。
#: 留着旧文件会出现「模块已经删了但还在被 import」这种幽灵问题。
#: 刻意**不删** chat/ 和 plugins/ —— 那里面可能已经有用户的东西。
_SOURCE_ONLY = ("src", "gui", "docs")


def extract_app(root, progress, force=False):
    """把引擎源码解到 runtime/app。"""
    app = os.path.join(runtime_dir(root), "app")
    src = find_bundled_source()
    if src is None:
        raise RuntimeError("发布包里找不到引擎源码（应该是打包时漏了）")

    if os.path.isfile(src) and src.endswith(".zip"):
        marker = os.path.join(app, ".extracted")
        stamp = _source_stamp(src)
        old = ""
        if os.path.isfile(marker):
            try:
                with open(marker, "r", encoding="utf-8") as fp:
                    old = fp.read().strip()
            except OSError:
                old = ""
        # ★ 判断「要不要重新解包」必须看**内容指纹**，不能只看标记文件在不在。
        #   以前只要 `.extracted` 存在就跳过 —— 于是用户升级了 exe，
        #   跑的还是上一版的源码，**修好的 bug 一个都到不了他手里**。
        if old.startswith(stamp) and not force:
            progress("引擎源码已解包，跳过")
            return app
        if old:
            progress("引擎源码有新版本，重新解包……")
        else:
            progress("正在解包引擎源码……")
        os.makedirs(app, exist_ok=True)
        for sub in _SOURCE_ONLY:
            shutil.rmtree(os.path.join(app, sub), ignore_errors=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(app)
        with open(marker, "w", encoding="utf-8") as fp:
            fp.write("%s %s\n" % (stamp, time.strftime("%Y-%m-%d %H:%M:%S")))
        progress("引擎源码解到 %s" % app)
        return app

    # 开发模式（src 是项目目录）：直接指过去，不复制
    progress("开发模式：直接用项目目录 %s" % src)
    return src


#: 配置里哪些键的值是路径（复制配置时要把它们换成 runtime 下的绝对路径）
_PATH_KEYS = ("out_dir", "data_dir", "tokenizer_path", "data_path",
              "init_from", "raw_dir")


def _absolutize_config(text, rt, app):
    """把出厂配置里的**相对路径**换成 runtime 下的绝对路径。

    ## 为什么需要这一步

    出厂配置里写的是相对路径（`data1`、`out/small`）—— 这样仓库里不含
    任何「某台机器」的信息，别人 clone 下来不会看到一堆不存在的路径。

    但相对路径是相对**当前工作目录**的。打包版跑起来时工作目录是
    `runtime/app`，于是 `data1` 会变成 `runtime/app/data1` ——
    而用户的数据应该在 `runtime/data1`。

    所以在复制到 `runtime/configs/` 的时候，把路径**钉成绝对路径**，
    指向 runtime 下面。这样：
        - 用户双击启动器 -> 配置直接可用
        - 用户自己改 runtime/configs/*.ini -> 改的就是这份钉好的

    规则：
        data_dir = data1        -> <runtime>/data1
        out_dir  = out/small    -> <runtime>/out/small
        tokenizer_path          -> 跟着 data_dir 走（data1 -> <runtime>/data1/...）
        data_path = chat/xx.jsonl -> <runtime>/app/chat/xx.jsonl（它在引擎源码里）
    """
    out_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or "=" not in line:
            out_lines.append(line)
            continue
        key, _, rest = line.partition("=")
        k = key.strip()
        if k not in _PATH_KEYS:
            out_lines.append(line)
            continue
        comment = ""
        m = re.search(r"\s+#", rest)
        if m:
            comment, rest = rest[m.start():], rest[:m.start()]
        val = rest.strip()
        if not val or os.path.isabs(val):
            out_lines.append(line)
            continue
        # chat/ 下的东西在引擎源码里，其余都在 runtime 根下
        base = os.path.join(app, val) if val.startswith("chat/") else \
            os.path.join(rt, val)
        out_lines.append("%s= %s%s" % (k, os.path.normpath(base), comment))
    return "\n".join(out_lines) + "\n"


def seed_configs(root, app, progress, reset=False):
    """把出厂配置复制到 runtime/configs，并钉成 runtime 下的绝对路径。

    默认**不覆盖**用户改过的配置；但会检测「旧版残留/内容不同」并提示，
    用户可以用 `--reset-configs` 主动重置（重置前自动备份 .bak）。
    （审计 E23：出厂配置的修复以前永远到不了老用户，只能靠用户自己发现。）
    """
    dst = os.path.join(runtime_dir(root), "configs")
    os.makedirs(dst, exist_ok=True)
    src = os.path.join(app, "configs")
    if not os.path.isdir(src):
        return dst
    rt = runtime_dir(root)
    n = 0
    stale = []
    for name in os.listdir(src):
        if not name.endswith(".ini"):
            continue
        target = os.path.join(dst, name)
        with open(os.path.join(src, name), "r", encoding="utf-8-sig") as fp:
            text = fp.read()
        desired = _absolutize_config(text, rt, app)
        if os.path.isfile(target):
            if not reset:
                try:
                    with open(target, "r", encoding="utf-8-sig") as fp:
                        cur = fp.read()
                except OSError:
                    cur = None
                if cur is not None and cur.strip() != desired.strip():
                    stale.append(name)
                continue
            bak = target + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
            try:
                shutil.copy2(target, bak)
                progress("  已备份旧配置：%s" % os.path.basename(bak))
            except OSError:
                pass
        with open(target, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(desired)
        n += 1
    if n:
        progress("复制了 %d 份配置到 %s（路径已指向这里）" % (n, dst))
    if stale:
        progress("注意：%d 份 runtime\\configs 配置与新版出厂不同（旧版残留或你改过）："
                 % len(stale))
        for name in stale[:6]:
            progress("  - %s" % name)
        progress("  想更新为出厂版（会先备份 .bak）："
                 "LingYu.exe --console --reset-configs")
    return dst


def seed_defaults(root, progress):
    """给界面写一份默认设置，让数据/输出都落在 runtime 里。

    这样用户不会莫名其妙在「我的文档」里发现一堆 bin 文件 ——
    一切都规规矩矩待在安装目录下，删文件夹就等于卸载干净。

    ⚠️ 判断「用户设过没有」要看**值是不是空的**，不能只看键在不在。

    踩过的坑：这里原来用 `data.setdefault(key, 默认值)`，
    而 `gui.json` 里这些键**本来就存在**，只是值是空字符串
    （界面首次启动时写下的默认值）。setdefault 只认「键不存在」，
    于是打包版第一次跑完，`data_dir` 依然是空的 ——
    「所有东西都在 runtime 里」这个承诺**静默失效**，
    用户得自己去挑目录，而界面上没有任何线索说明为什么。
    """
    rt = runtime_dir(root)
    cfg_dir = os.path.join(rt, "configs")
    home = os.path.join(os.path.expanduser("~"), ".lingyu")
    os.makedirs(home, exist_ok=True)
    path = os.path.join(home, "gui.json")
    data = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp) or {}
        except (OSError, ValueError):
            data = {}

    wants = {
        "data_dir": os.path.join(rt, "data"),
        "out_dir": os.path.join(rt, "out"),
        "raw_dir": os.path.join(rt, "raw"),
        "chat_dir": os.path.join(rt, "chat"),
    }
    filled = []
    for key, val in wants.items():
        if not data.get(key):          # 缺失**或为空**都算没设过
            data[key] = val
            filled.append(key)

    # ---- config 要单独判「过不过期」，不能只看空不空 ----
    # 踩过的坑：开发机上界面把 config 存成了相对路径 `configs/small.ini`。
    # 打包版第一次跑时这个值非空，于是原样留着；界面再按源码根解析成
    # `runtime/app/configs/small.ini` —— 那是**带相对路径的源码副本**，
    # 里面的 `data_dir = data1` 落到 `runtime/app/data1`，
    # 而语料准备页写的是 `runtime/data`。结果刚准备完就说「还没准备语料」。
    cur = (data.get("config") or "").strip()
    if (not cur) or (not os.path.isabs(cur)) or (not os.path.isfile(cur)):
        cand = os.path.join(cfg_dir, os.path.basename(cur) or "small.ini")
        if os.path.isfile(cand) and cur != cand:
            data["config"] = cand
            filled.append("config")

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    if filled:
        progress("界面默认路径已写入（%s）" % "、".join(filled))
    for sub in ("data", "out", "raw", "chat"):
        os.makedirs(os.path.join(rt, sub), exist_ok=True)
    return path


# ==========================================================================
# 总流程
# ==========================================================================
def read_state(root):
    p = os.path.join(runtime_dir(root), STATE_FILE)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except (OSError, ValueError):
            return {}
    return {}


def write_state(root, data):
    rt = runtime_dir(root)
    os.makedirs(rt, exist_ok=True)
    p = os.path.join(rt, STATE_FILE)
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def environment_ready(root):
    """环境是不是已经装好了（快速判断，不做任何下载）。"""
    st = read_state(root)
    if st.get("state_version") != STATE_VERSION:
        return False, "首次使用"
    py = st.get("venv_python")
    if not py or not os.path.isfile(py):
        return False, "虚拟环境不见了"
    app = st.get("app_dir")
    if not app or not os.path.isdir(app):
        return False, "引擎源码不见了"
    return True, "ok"


def setup(root, progress, index_url=None, force=False, skip_deps=False):
    """完整的首次安装流程。返回 venv 里的 python.exe 路径。"""
    rt = runtime_dir(root)
    os.makedirs(rt, exist_ok=True)
    if not log.path:
        _start_log(root)
    progress("安装位置：%s" % rt)

    # ---- 1. Python ----
    py, why = pick_python(root, progress)
    if py is None and not force:
        for r in why[:4]:
            progress("  跳过：%s" % r)
    if py is None:
        py, why2 = install_python_with_uv(root, progress)
        if py is None:
            raise RuntimeError(
                "找不到可用的 Python，也没能下载一个。\n\n"
                "试过这些地方：\n  " + "\n  ".join(why[:6]) +
                "\n\n下载失败原因：%s\n\n"
                "你可以手动装一个 Python 3.10~3.12（64 位）再重新双击，\n"
                "或者设置环境变量 LINGYU_PYTHON 指向已有的 python.exe。"
                % why2)

    # ---- 2. 虚拟环境 ----
    venv_py = make_venv(py, root, progress)

    # ---- 3. 依赖 ----
    if skip_deps:
        progress("已指定 --skip-deps，跳过依赖安装（需要你自己保证环境里"
                 "有 torch / numpy / PySide6）")
    else:
        progress("检查依赖……")

        # 3a) 小包：从普通 PyPI 镜像装（快）
        need = []
        for mod, desc in REQUIREMENTS:
            probe = "PySide6" if mod.startswith("PySide6") else mod
            if has_package(venv_py, probe):
                progress("  已有：%s（%s）" % (mod, desc))
            else:
                progress("  缺：%s（%s）" % (mod, desc))
                need.append(mod)
        if need:
            progress("需要安装：%s" % "、".join(need))
            last_err = None
            for label, url in INDEXES:
                try:
                    progress("用镜像：%s" % label)
                    pip_install(venv_py, need, url, progress)
                    last_err = None
                    break
                except RuntimeError as e:
                    last_err = e
                    progress("这个镜像不行，换下一个……")
            if last_err is not None:
                raise last_err
        else:
            progress("  小依赖都装好了")

        # 3b) torch：**单独走 CUDA 索引**，装完还要验一次
        state, detail = ("none", "")
        if has_package(venv_py, "torch"):
            state, detail = torch_cuda_ok(venv_py)
        if state == "cuda":
            progress("  PyTorch %s（CUDA 可用）✓" % detail, "ok")
        elif state == "cpu_wheel":
            # 已经有 torch 但是 CPU 版轮子 —— 这正是踩过的坑，
            # 不能默默放过，直接重装成 CUDA 版
            progress("  发现装的是 CPU 版 PyTorch（%s），重装成 CUDA 版……"
                     % detail, "warn")
            install_torch(venv_py, progress, index_url)
        elif state == "cuda_unusable":
            # CUDA 轮子但 is_available() 为 False：可能是驱动太老。
            # 换更低 cuXXX 的源试试；仍不行的话 install_torch 会告诉用户更新驱动。
            progress("  装的是 CUDA 版 PyTorch（%s），但这台机器上 CUDA 不可用"
                     " —— 先试更低版本的 CUDA 源，仍不行就得更新显卡驱动。"
                     % detail, "warn")
            install_torch(venv_py, progress, index_url)
        else:
            install_torch(venv_py, progress, index_url)

    # ---- 4. 引擎源码 ----
    app = extract_app(root, progress, force=force)
    cfg_dir = seed_configs(root, app, progress)
    settings = seed_defaults(root, progress)
    progress("界面默认设置写到 %s" % settings)

    st = read_state(root)
    st.update({
        "state_version": STATE_VERSION,
        "base_python": py,
        "venv_python": venv_py,
        "app_dir": app,
        "configs_dir": cfg_dir,
        "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if skip_deps:
        st["skip_deps"] = True
    write_state(root, st)
    progress("全部准备完成 ✓")
    return venv_py


def ensure_app(root, progress):
    """**每次启动**都确认 runtime/app 里的源码和 exe 里带的是同一份。

    ★ 这一步不能只放在 `setup()` 里。

      踩过的坑：`extract_app()` 的哈希判断写对了，但 `setup()` 只在
      **首次安装**（或者环境坏掉）时才跑。环境装好之后每次启动都直接
      `launch()`，于是源码**永远不会更新** —— 用户升级了 exe，
      跑的还是几个月前那版代码，而版本号、界面、日志看着都是新的。

      实测就是这么发生的：我把分词器的修复写进了源码、重新打了包、
      双击运行，行为一点没变。查了半天才发现 `.extracted` 里还是
      上一轮的时间戳。

    哈希只是读一个几百 KB 的包，开销可以忽略；换来的是一条硬保证：
    **exe 是哪一版，跑的就是哪一版源码。**
    """
    src = find_bundled_source()
    if src is None or not (os.path.isfile(src) and src.endswith(".zip")):
        return          # 开发模式：源码就在项目目录里，没什么可同步的
    marker = os.path.join(runtime_dir(root), "app", ".extracted")
    stamp = _source_stamp(src)
    old = ""
    if os.path.isfile(marker):
        try:
            with open(marker, "r", encoding="utf-8") as fp:
                old = fp.read().strip()
        except OSError:
            old = ""
    if old.startswith(stamp):
        return
    extract_app(root, progress)


def _engine_log_path(root):
    """引擎 stderr 与退出记录的日志：runtime\\logs\\engine-YYYYmmdd.log"""
    d = os.path.join(runtime_dir(root), "logs")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return os.path.join(d, "engine-%s.log" % time.strftime("%Y%m%d"))


def _tail_lines(path, n=10, limit=65536):
    """读文件末尾 n 行（读不到就返回空列表，绝不因为日志问题再抛异常）。"""
    try:
        with open(path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            fp.seek(max(0, size - limit))
            data = fp.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [ln.rstrip() for ln in data.splitlines() if ln.strip()]
    return lines[-n:]


def launch(root, venv_py, args, progress):
    """启动引擎。app 目录加到 PYTHONPATH，这样 `-m lingyu` 找得到。

    ## ★ 工作目录必须是「安装根」，不能是 runtime\\app

    使用说明里的命令都写作 `--config runtime\\configs\\small.ini`（相对安装根）。
    这里曾经用 `cwd=app`，于是用户照着说明书敲的第一条命令就报
    「找不到配置文件：runtime\\configs\\small.ini」—— 实测踩过。
    引擎自己用 `paths.home()` / `LINGYU_HOME` 定位，不依赖 cwd，
    所以把 cwd 改成安装根是安全的。

    ## ★ 失败必须可见（曾经静默）

    以前用 `subprocess.call`：不记退出码、不抓 stderr，双击后启动失败
    就是「窗口闪一下/没反应」，`launcher.log` 里也只有「启动：…」。
    现在 stderr 落盘到 `runtime\\logs\\engine-*.log`，并记录退出码；
    非 0 时在终端打印「启动失败（退出码 N）」+ stderr 尾部 10 行。
    """
    st = read_state(root)
    app = st.get("app_dir")
    src = os.path.join(app, "src")
    env = dict(os.environ)
    paths = [p for p in (src, app) if p and os.path.isdir(p)]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(
            paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LINGYU_HOME"] = runtime_dir(root)

    cmd = [venv_py, "-m", "lingyu"] + list(args)
    cwd = root                     # ★ 安装根（见 docstring），不是 app
    progress("启动：" + " ".join(cmd[1:]))
    log_path = _engine_log_path(root)
    try:
        sink = open(log_path, "ab")
    except OSError:
        sink = open(os.devnull, "ab")
    try:
        label = "\n=== %s cwd=%s ===\n" % (
            time.strftime("%Y-%m-%d %H:%M:%S"), cwd)
        sink.write(label.encode("utf-8"))
        sink.write(("cmd=%s\n" % " ".join(cmd)).encode("utf-8"))
        sink.flush()
        try:
            proc = subprocess.Popen(cmd, env=env, cwd=cwd, stderr=sink)
        except OSError as e:
            log("启动失败：%s" % e, "error")
            return 1
        code = proc.wait()
    finally:
        sink.close()

    log("引擎退出码：%d（日志：%s）" % (code, log_path))
    if code != 0:
        msg = "启动失败（退出码 %d），日志：%s" % (code, log_path)
        tail = _tail_lines(log_path, 10)
        try:
            print(msg)
            for line in tail:
                print("  " + line)
        except Exception:
            pass
        log(msg, "error")
        for line in tail:
            log("  " + line)
    return code


def _launch_error_text(root, code):
    """失败对话框的文案：说清「哪一步失败 + 日志在哪 + 怎么重试 + 报 bug 带什么」。

    文案本身也是产品的一部分：用户看到 tkinter 灰框时，只有这里有指引。
    """
    return ("引擎启动失败（退出码 %d）。\n\n"
            "日志：%s\n\n"
            "常见原因：显卡驱动太旧、依赖没装全、被安全软件拦截。\n"
            "重试：LingYu.exe --console --reinstall\n"
            "报 bug 时请附上日志最后 10 行。" % (code, _engine_log_path(root)))


def show_launch_error(root, code):
    """双击路径（控制台已隐藏）下，用对话框让用户看见失败原因。"""
    msg = _launch_error_text(root, code)
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("%s 启动失败" % APP_ZH, msg)
        r.destroy()
    except Exception:
        pass


# ==========================================================================
# 界面（tkinter）
# ==========================================================================
class SetupWindow:
    """首次运行的向导窗口。

    只做三件事：显示在干什么、显示日志、出错了告诉用户怎么办。
    刻意做得简单 —— 它的职责是「让用户相信程序在动」，
    花哨的动画反而会让人怀疑是不是在假装。
    """

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("%s 首次运行向导" % APP_ZH)
        self.root.geometry("720x460")
        self.root.minsize(620, 380)
        self.root.configure(bg="#0F1115")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#0F1115")
        style.configure("TLabel", background="#0F1115", foreground="#E7EAF0",
                        font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Dim.TLabel", foreground="#9BA4B4",
                        font=("Microsoft YaHei UI", 9))
        style.configure("Thin.Horizontal.TProgressbar",
                        background="#5B8CFF", troughcolor="#1E232C",
                        bordercolor="#0F1115", lightcolor="#5B8CFF",
                        darkcolor="#5B8CFF")

        tk.Label(self.root, text="◆  " + APP_ZH, bg="#0F1115",
                 fg="#E7EAF0", font=("Microsoft YaHei UI", 18, "bold")
                 ).pack(anchor="w", padx=24, pady=(20, 0))
        tk.Label(self.root, text="第一次运行需要准备运行环境（约 2.5 GB，只做一次）",
                 bg="#0F1115", fg="#9BA4B4",
                 font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=24)

        self.step = tk.Label(self.root, text="准备中……", bg="#0F1115",
                             fg="#5B8CFF",
                             font=("Microsoft YaHei UI", 11, "bold"))
        self.step.pack(anchor="w", padx=24, pady=(16, 4))

        self.bar = ttk.Progressbar(self.root, mode="indeterminate", length=680,
                                   style="Thin.Horizontal.TProgressbar")
        self.bar.pack(fill="x", padx=24)
        self.bar.start(14)

        box = tk.Frame(self.root, bg="#0F1115")
        box.pack(fill="both", expand=True, padx=24, pady=(12, 8))
        self.text = tk.Text(box, bg="#171B22", fg="#9BA4B4", bd=0,
                            insertbackground="#E7EAF0", wrap="word",
                            font=("Consolas", 9), padx=12, pady=10)
        sb = tk.Scrollbar(box, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        self.foot = tk.Label(self.root, text="", bg="#0F1115", fg="#6A7382",
                             font=("Microsoft YaHei UI", 9), anchor="w",
                             justify="left")
        self.foot.pack(fill="x", padx=24, pady=(0, 16))

        self.done = False
        self.error = None

    def push(self, msg, level="info"):
        self.text.insert("end", msg + "\n")
        self.text.see("end")
        self.step.configure(text=msg[:90],
                            fg="#F87171" if level == "error"
                            else ("#34D399" if level == "ok" else "#5B8CFF"))
        # ★ 这里原来有 `self.root.update_idletasks()` / `self.root.update()`。
        #   那是为了让**工作线程**调用时窗口也能刷新而加的 —— 但正因为
        #   跨线程调 Tk 才导致偶发卡死（见 run_with_window 的说明）。
        #   现在 push 只在主线程（after 轮询）里被调用，
        #   Tk 自己会重绘，不需要手动 update；而且在 after 回调里调
        #   update() 会造成重入，反而更危险。
        log(msg, level)

    def finish(self, ok, venv_py=None, err=None):
        self.done = True
        self.error = err
        self.bar.stop()
        self.bar.configure(mode="determinate", value=100 if ok else 0)
        if ok:
            self.step.configure(text="准备完成，正在启动……", fg="#34D399")
            self.foot.configure(text="以后双击就会直接打开界面，不会再走这一步。")
        else:
            self.step.configure(text="出错了", fg="#F87171")
            self.foot.configure(
                text="日志文件：%s\n把上面这段和日志一起发出来，能帮上大忙。"
                     % (log.path or "(无)"), fg="#F87171")


def _start_log(root):
    """把日志文件挂上。

    ★ 必须在**每次运行的最开始**就挂，不能等到 setup() 里再挂。

      踩过的坑：原来只在 `setup()` 里设 `log.path`。环境装好之后第二次
      运行不会走 setup，于是 `runtime\\launcher.log` **一直是空的** ——
      而《使用说明》里明确写着「出问题看 runtime\\launcher.log」。
      用户界面起不来、翻日志发现是空的，比没有日志更让人火大。
    """
    try:
        rt = runtime_dir(root)
        os.makedirs(rt, exist_ok=True)
        log.path = os.path.join(rt, "launcher.log")
        with open(log.path, "a", encoding="utf-8") as fp:
            fp.write("\n=== %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    except OSError:
        pass


def run_with_window(root, args):
    """带向导窗口的启动。

    ## ★ 为什么消息要走队列，而不是让工作线程直接改界面

    这里踩过一个**偶发卡死**，很值得记下来，因为它只在「环境已经装好」
    这条最快的路径上出现 —— 也就是用户日常走的那条。

    原来是这样写的：

        threading.Thread(target=work).start()
        win.root.mainloop()

    而 `work()` 里直接 `win.push(...)` / `win.finish(...)` / `win.root.quit()`。
    两个问题叠在一起：

    1. **Tkinter 不是线程安全的**。`push()` 内部调 `self.root.update()`，
       从非主线程调它，Tk 的 C 层有时候会直接锁死。
    2. **`quit()` 早于 `mainloop()` 就失效**。环境已就绪时 `work()`
       几百毫秒就跑完了（`push` → `finish` → `sleep(0.4)` → `quit`），
       完全可能抢在主线程进入 `mainloop()` 之前把 `quit()` 调掉。
       `quit()` 只对**正在跑的** mainloop 有效，先调等于没调 ——
       于是 `mainloop()` 进去以后**永远不出来**，进程挂死。

    症状：双击 exe 有时能开、有时一直没反应；`launcher.log` 停在
    「环境已就绪，正在启动……」。而首次安装那一次反而**不会**卡 ——
    因为装环境要几分钟，mainloop 早就起来了。所以这个 bug 只在
    **装好之后**才暴露，特别容易被当成「偶发」。

    修法：工作线程**只往队列里塞消息**，一个 Tk 调用都不做；
    主线程用 `after()` 轮询队列。`quit()`/`destroy()` 也由主线程调。
    """
    _start_log(root)
    win = SetupWindow()
    q = queue.Queue()
    done = {}

    def work():
        def push(msg, level="info"):
            q.put(("msg", msg, level))        # ★ 只入队，绝不碰 Tk

        try:
            ready, why = environment_ready(root)
            if args.reinstall:
                ready, why = False, "指定了 --reinstall"
            if ready:
                push("环境已就绪，正在启动……", "ok")
            else:
                push("需要准备运行环境（%s）" % why)
                setup(root, push, index_url=args.index,
                      force=args.reinstall, skip_deps=args.skip_deps)
            done["venv_py"] = read_state(root).get("venv_python")
            q.put(("done", None, None))
        except Exception as e:
            tb = traceback.format_exc()
            log(tb)
            push(str(e), "error")
            push("—— 详细信息 ——")
            for line in tb.strip().splitlines()[-6:]:
                push("  " + line)
            done["error"] = e
            q.put(("fail", None, None))

    def pump():
        """主线程这边：把队列里的消息刷到界面上。"""
        try:
            while True:
                kind, msg, level = q.get_nowait()
                if kind == "msg":
                    win.push(msg, level)
                    continue
                win.finish(kind == "done", done.get("venv_py"),
                           err=done.get("error"))
                # destroy（而不是 quit）—— 窗口直接消失。
                # 用 quit 的话 mainloop 虽然退了，窗口还留在屏幕上，
                # 而下面 launch() 会一直阻塞到子进程结束，
                # 于是用户背后永远挂着一个「正在启动……」的向导窗口。
                win.root.destroy()
                return
        except queue.Empty:
            pass
        win.root.after(60, pump)

    threading.Thread(target=work, daemon=True).start()
    win.root.after(60, pump)
    win.root.mainloop()

    if done.get("error") is not None:
        return 1
    venv_py = done.get("venv_py")
    if not venv_py or not os.path.isfile(venv_py):
        return 1
    # ★ 每次启动都要对一次源码版本，不能只在 setup() 里做（见 ensure_app）。
    #   这一步发生在向导窗口已经关掉之后，所以直接写日志，不再往窗口推消息。
    ensure_app(root, log)
    code = launch(root, venv_py, args.app_args, log)
    if code:
        # 控制台已经被藏掉了：不给对话框，用户只会看到「窗口闪一下就没了」。
        show_launch_error(root, code)
    return code


def run_console(root, args):
    """--console：不弹窗，日志直接打终端（排查问题时用）。"""
    _start_log(root)
    try:
        ready, why = environment_ready(root)
        if args.reinstall:
            ready, why = False, "指定了 --reinstall"
        if ready:
            log("环境已就绪")
            venv_py = read_state(root)["venv_python"]
        else:
            log("需要准备运行环境（%s）" % why)
            venv_py = setup(root, log, index_url=args.index,
                            force=args.reinstall, skip_deps=args.skip_deps)
    except Exception as e:
        log("安装失败：%s" % e, "error")
        log(traceback.format_exc())
        return 1
    if args.setup_only:
        log("只装环境，不启动（--setup-only）", "ok")
        return 0
    # ★ 每次启动都要对一次源码版本，不能只在 setup() 里做（见 ensure_app）
    ensure_app(root, log)
    return launch(root, venv_py, args.app_args, log)


# ==========================================================================
# 入口
# ==========================================================================
def _fix_console_encoding():
    """让控制台输出不会因为编码问题炸掉。

    ## 为什么必须做

    中文 Windows 的控制台默认代码页是 **GBK(936)**。往它写 UTF-8 才有的
    字符（比如 `✓`、`→`、emoji）会直接抛：

        UnicodeEncodeError: 'gbk' codec can't encode character '\\u2713'

    这不是「显示成乱码」那么轻 —— 是**整个安装流程崩在最后一步**，
    而且崩在一个跟安装毫无关系的地方（打印进度）。实测踩过。

    两步一起做：
      1. 把控制台代码页切到 UTF-8（现代 Windows Terminal 能正确显示）
      2. 给 stdout/stderr 加 errors="replace" 兜底 ——
         万一代码页切不动（老系统、重定向到文件），
         顶多把那个字符显示成 `?`，绝不再抛异常。
    """
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _hide_console():
    """把控制台窗口藏起来。

    PyInstaller 用 `--console` 打包（这样 `LingYu.exe --console` 能看到
    完整输出，出问题好排查）。但双击时用户不该看到一个黑框 ——
    所以默认动作是把控制台藏掉，需要时用 `--console` 显式要它。

    为什么不直接打成 `--windowed`：那样 `print()` 就没有去处了，
    `--console` 也失去意义。留着控制台再按需隐藏，两种用法都能照顾到。
    """
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 0)      # SW_HIDE
    except Exception:
        pass


def _build_parser():
    ap = argparse.ArgumentParser(
        prog="LingYu",
        description="灵语启动器：准备运行环境并打开界面",
        # RawDescription：epilog 里的换行按原样保留（默认 formatter 会把
        # 换行折叠成空格，两层说明就挤成一段了）
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="启动器自己的参数要写在前面，引擎的参数写在后面。\n"
               "例如：LingYu.exe --console train --config runtime\\configs\\small.ini\n"
               "引擎命令：doctor / scan / prepare / params / plan / train / sample /\n"
               "          sft / chat / launch / gui\n"
               "（看某个命令的参数：--console plan --help）",
        # ★★ allow_abbrev=False：不许前缀匹配。理由见下面 parse_known_args 处。
        allow_abbrev=False)
    ap.add_argument("--console", action="store_true",
                    help="不弹向导窗口，日志直接输出到终端")
    ap.add_argument("--setup-only", action="store_true",
                    help="只准备运行环境，不启动界面")
    ap.add_argument("--reinstall", action="store_true",
                    help="重新检查并安装依赖（环境坏掉时用）")
    ap.add_argument("--skip-deps", action="store_true",
                    help="只准备 Python 环境和引擎源码，不装 torch 等依赖"
                         "（给已经自己管环境的人用，也方便调试）")
    ap.add_argument("--index", default=None,
                    help="指定 PyPI 镜像地址")
    ap.add_argument("--root", default=None,
                    help="安装根目录（默认 exe 所在目录）")
    ap.add_argument("--reset-configs", action="store_true",
                    help="把 runtime/configs 恢复成出厂版本（旧文件先备份成 .bak-时间戳）")
    # ★ --version 实际在 main() 里提前处理（不启动引擎）；这里只为让它
    #   出现在 --help 里，并避免被当成引擎参数透传。
    ap.add_argument("--version", action="store_true",
                    help="显示启动器与引擎版本")
    # 剩下的参数原样传给 lingyu（例如 `LingYu.exe doctor`）
    ap.add_argument("app_args", nargs="*", default=[],
                    help="传给 lingyu 的参数，如 doctor / train --config ...")
    return ap


# 引擎子命令（改引擎 CLI 时同步；仅用于 --help 提示）
ENGINE_COMMANDS = ("doctor", "scan", "prepare", "params", "plan", "train",
                   "sample", "sft", "chat", "launch", "gui")


def _launcher_help_text():
    """`LingYu.exe --help` 要同时交代两件事：启动器参数 + 引擎命令怎么看。"""
    return (_build_parser().format_help() +
            "\n引擎命令（写在启动器参数之后；不加 --console 会先弹向导窗口）\n"
            "  LingYu.exe --console doctor                                     体检环境\n"
            "  LingYu.exe --console plan  --config runtime\\configs\\small.ini  看显存够不够\n"
            "  LingYu.exe --console train --config runtime\\configs\\small.ini  开始训练\n"
            "\n命令清单：%s\n"
            "看某个引擎命令的全部参数：LingYu.exe --console plan --help（把 plan 换成别的命令）\n"
            % " / ".join(ENGINE_COMMANDS))


def _has_engine_arg(argv):
    """去掉启动器自己的参数后，是否还有引擎参数（用来区分两层 --help）。"""
    i = 0
    while i < len(argv):
        t = argv[i]
        if t in ("--console", "--setup-only", "--reinstall", "--skip-deps",
                 "--reset-configs", "--version"):
            i += 1
        elif t in ("--index", "--root"):
            i += 2
        elif t.startswith(("--index=", "--root=")):
            i += 1
        else:
            return True
    return False


def main(argv=None):
    # ★ 第一行就修控制台编码：argparse 的 --help/报错也在它之后才输出。
    #   以前放在解析之后，--help 在 GBK 控制台里全是乱码（实测踩过）。
    _fix_console_encoding()

    raw = list(sys.argv[1:] if argv is None else argv)
    # 版本号：不启动引擎、不装环境，直接打印一行就退出。
    # （引擎 CLI 自己没有 --version，所以这里吃掉；用户问「我是哪个版本」
    #   时最需要它，而且报 bug 时第一件事就是要版本号。）
    if "--version" in raw or "-V" in raw:
        print(version_text())
        return 0

    engine_help = False
    if any(t in ("-h", "--help") for t in raw):
        rest = [t for t in raw if t not in ("-h", "--help")]
        if _has_engine_arg(rest):
            engine_help = True          # 例：--console plan --help → 交给引擎
            raw = rest
        else:
            # 裸 --help：启动器参数 + 引擎命令怎么看，两层一起打印
            print(_launcher_help_text())
            return 0

    ap = _build_parser()
    # ★ 必须用 parse_known_args：`LingYu.exe train --config X` 里的
    #   `--config X` 是给 lingyu 的，不是给启动器的。
    #   用 parse_args 的话 argparse 会直接报
    #   「unrecognized arguments: --config ...」然后退出 ——
    #   而文档里明明写着可以这么用。实测踩过。
    #
    # ★★ 而且必须关掉**前缀匹配**（allow_abbrev=False），否则会静默吃掉
    #    lingyu 自己的参数：
    #
    #        LingYu.exe --console train --set train.max_steps=5
    #
    #    这里的 `--set` 是 `lingyu train` 的参数（临时覆盖配置项），
    #    但 argparse 默认允许缩写，于是把它匹配成了启动器的 `--setup-only`！
    #    结果：不训练，打印一句「只装环境，不启动」就退出了，
    #    **退出码还是 0**。用户看到的是「命令跑了，什么都没发生」。
    #
    #    这条尤其坑，因为 `--set` 正是出厂配置注释和使用说明里教用户用的写法。
    #    关掉 allow_abbrev（见函数开头）之后，`--set` 走 passthrough 传给 lingyu。
    args, passthrough = ap.parse_known_args(raw)
    args.app_args = list(args.app_args) + list(passthrough)
    if engine_help:
        # `--console plan --help`：帮助本身要交给引擎，而不是被 argparse 吃掉
        args.app_args.append("--help")

    root = os.path.abspath(args.root) if args.root else exe_dir()
    # --reset-configs：把 runtime/configs 恢复成出厂版（旧文件先备份成 .bak-时间戳）
    if getattr(args, "reset_configs", False):
        app_dir = os.path.join(runtime_dir(root), "app")
        if os.path.isdir(os.path.join(app_dir, "configs")):
            seed_configs(root, app_dir, print, reset=True)
        else:
            print("还没有安装过引擎源码：先正常启动一次完成安装，再重置配置。")
        if not args.app_args:
            return 0
    # 默认动作：打开图形界面（用户双击时期望的就是这个）
    if not args.app_args:
        args.app_args = ["gui"]

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # 开图形界面时藏控制台；跑命令行子命令时留着（用户要看输出）
    if not args.console and args.app_args == ["gui"]:
        _hide_console()

    if args.console or args.setup_only:
        return run_console(root, args)
    return run_with_window(root, args)


if __name__ == "__main__":
    sys.exit(main())
