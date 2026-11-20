# -*- coding: utf-8 -*-
r"""打包成发布用的文件夹。

## 产出长什么样

    dist/LingYu/
      LingYu.exe          启动器（双击这个）
      使用说明.txt
      LICENSE
      README.md
      runtime/            首次双击后才会出现（Python 环境、配置、数据）
        uv.exe            （从 exe 里解出来的，用来下载 Python）
        ...

## 为什么 exe 里要带 uv.exe

39 MB 的 uv 塞进 exe 会让启动器变成 50 MB 左右，但换来的是：
**用户机器上没有任何 Python 时也能一键装好**（uv 自己会下载 Python）。

对比另一种做法（不带 uv，让用户自己装 Python）：用户要先去 python.org
下载、安装、勾选 Add to PATH，然后回来重新双击 —— 那不叫「开箱即用」。
50 MB 换掉这一整段折腾，很值。

## 打包命令

    python packaging/build_exe.py

产物在 `dist/LingYu/`。加 `--zip` 会顺便压成一个 zip 方便上传 GitHub Release。
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "packaging")
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, "build")

APP_NAME = "LingYu"

# 发布文件夹里应该有的文件（不含 runtime/ 和 SHA256SUMS 自身）。
# SHA256SUMS 对这份清单算哈希；zip 条目也按它校验 —— 少一个文件、
# 多一个临时文件都算发布事故，宁可构建失败也别发出去。
RELEASE_FILES = ("LingYu.exe", "LICENSE", "README.md", "README.zh-CN.md",
                 "使用说明.txt")

README_TXT = """灵语（LingYu）—— 中文大模型训练引擎
================================================

怎么用
------
双击 LingYu.exe 就行。

第一次会弹出一个向导，自动准备运行环境（要下载约 2.5 GB，只做一次，
大概 5~20 分钟，取决于网速）。装完就直接打开界面，以后双击秒开。

准备工作做了什么
----------------
所有东西都装在这个文件夹的 runtime 子目录里，不会污染系统：

    runtime\\python     独立的 Python
    runtime\\venv       装了 PyTorch 等依赖的环境
    runtime\\app        引擎源码
    runtime\\configs    配置文件（可以直接用记事本改）
    runtime\\data       语料数据
    runtime\\out        训练输出（模型断点都在这里）
    runtime\\launcher.log   出问题时看这个

**想卸载？直接删掉整个文件夹就行**，程序本身不留别的东西。
（GUI 的窗口设置存在 %USERPROFILE%\\.lingyu\\gui.json，介意的话一并删掉。）

出问题了怎么办
--------------
1. 先看 runtime\\launcher.log 的最后几行
2. 用命令行跑一次，能看到详细过程：

       LingYu.exe --console --reinstall

3. 启动失败时看 runtime\\logs\\engine-<日期>.log（引擎自己的报错都在里面）
4. 环境装了一半断网了？再双击一次，它会接着装，不用从头来
5. 升级 exe 后如果提示「runtime\\configs 配置与新版出厂不同」（旧版残留或你改过），
   想恢复出厂配置可以执行（旧文件会先备份成 .bak-时间戳）：

       LingYu.exe --console --reset-configs

命令行也能用
------------
LingYu.exe 后面可以跟引擎的命令。建议加 --console：日志直接打到终端，
出问题时能看到完整过程（不加会先弹向导窗口，装好环境后再执行）。

    LingYu.exe --console doctor                                     体检环境
    LingYu.exe --console train --config runtime\\configs\\small.ini  开始训练
    LingYu.exe --console plan  --config runtime\\configs\\small.ini  先看显存够不够
    LingYu.exe --console sft   --config runtime\\configs\\sft.ini    监督微调
    LingYu.exe --console sft   --config runtime\\configs\\sft.ini --resume  接着上次的 SFT 断点训

    LingYu.exe --version                      看启动器/引擎版本
    LingYu.exe --help                         启动器参数 + 引擎命令清单
    LingYu.exe --console plan --help          看某个引擎命令的全部参数
    LingYu.exe --console --reset-configs      恢复出厂配置（先备份 .bak）

上面命令里的相对路径（如 runtime\\configs\\small.ini）是相对**这个文件夹**
的，在哪个目录敲都一样。

校验发布包（可选）
------------------
发布文件夹里带 SHA256SUMS，可以在 PowerShell 里核对 exe 有没有被改动：

    Get-FileHash .\\LingYu.exe -Algorithm SHA256

把结果和 SHA256SUMS 里 LingYu.exe 那一行对比即可。

许可
----
Apache-2.0，见 LICENSE。
"""


def check_prereqs():
    problems = []
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        problems.append("没装 PyInstaller：pip install pyinstaller")
    if not os.path.isdir(os.path.join(ROOT, "src", "lingyu")):
        problems.append("找不到 src/lingyu —— 请在项目根目录下运行")
    return problems


def find_uv():
    """找 uv.exe 用来打进发布包。"""
    p = shutil.which("uv")
    if p:
        return p
    cands = [
        os.path.join(os.environ.get("APPDATA", ""), "Python",
                     "Python314", "Scripts", "uv.exe"),
        os.path.join(os.path.expanduser("~"), ".local", "bin", "uv.exe"),
        os.path.join(os.path.expanduser("~"), ".cargo", "bin", "uv.exe"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def make_source_zip(progress):
    """把引擎源码打成一个 zip 塞进 exe。

    只打源码，不打 data/out/venv 那些 —— 用户的数据不该进发布包。

    加进包里的东西：
        src/lingyu/     引擎核心
        gui/            桌面界面
        configs/        出厂配置
        chat/           示例对话数据
        docs/           设计文档
        plugins/        插件目录（含说明）
    """
    out = os.path.join(BUILD, "lingyu_src.zip")
    os.makedirs(BUILD, exist_ok=True)
    include = ["src/lingyu", "gui", "configs", "chat", "docs", "plugins"]
    skip_dirs = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    skip_ext = {".pyc", ".pyo", ".pt", ".bin", ".png"}

    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in include:
            base = os.path.join(ROOT, rel)
            if not os.path.isdir(base):
                progress("  跳过（不存在）：%s" % rel)
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                for fn in filenames:
                    if os.path.splitext(fn)[1] in skip_ext:
                        continue
                    full = os.path.join(dirpath, fn)
                    arc = os.path.relpath(full, ROOT)
                    z.write(full, arc)
                    n += 1
        # 顶层再放一份版本信息，方便排查「用户装的是哪一版」
        z.writestr("BUILD_INFO.txt",
                   "built at %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    size = os.path.getsize(out) / 1024 / 1024
    progress("  源码包：%d 个文件，%.1f MB" % (n, size))
    return out


def _read_version():
    """读引擎版本号（`src/lingyu/version.py` 的 `__version__`）。"""
    try:
        with open(os.path.join(ROOT, "src", "lingyu", "version.py"),
                  "r", encoding="utf-8") as fp:
            m = re.search(r'__version__\s*=\s*"([^"]+)"', fp.read())
        return m.group(1) if m else "0.0.0"
    except OSError:
        return "0.0.0"


def write_version_file(progress):
    """生成 PyInstaller 的 Windows 版本资源（`--version-file`）。

    没有它时 exe 属性里的「文件版本/产品版本/公司名」全是空的 ——
    用户右键看不到版本，排查问题时也说不清装的是哪一版（审计 E09）。
    签名需要证书，这不在代码里解决；但版本资源不需要。
    """
    ver = _read_version()
    parts = [int(x) for x in re.findall(r"\d+", ver)[:3]] or [0, 0, 0]
    while len(parts) < 3:
        parts.append(0)
    quad = "(%d, %d, %d, 0)" % (parts[0], parts[1], parts[2])
    path = os.path.join(BUILD, "LingYu_version.txt")
    os.makedirs(BUILD, exist_ok=True)
    text = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=%s,
    prodvers=%s,
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'lingyu-lm'),
        StringStruct('FileDescription', 'LingYu - Chinese-first LLM training engine'),
        StringStruct('FileVersion', '%s'),
        StringStruct('InternalName', 'LingYu'),
        StringStruct('LegalCopyright', 'Apache-2.0'),
        StringStruct('OriginalFilename', 'LingYu.exe'),
        StringStruct('ProductName', 'LingYu'),
        StringStruct('ProductVersion', '%s')
      ])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""" % (quad, quad, ver, ver)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(text)
    progress("  版本资源：%s" % ver)
    return path


def run_pyinstaller(src_zip, uv_exe, progress, onedir=False):
    """调 PyInstaller 打启动器。

    用 `--onefile`：用户要的就是「一个 .exe」。启动器本身只有几十 MB，
    解包到临时目录只要一两秒，这个代价可以接受
    （真正 2.5 GB 的 torch 走 runtime 子目录，不进 exe）。
    """
    sep = ";" if os.name == "nt" else ":"
    add_data = ["%s%s." % (src_zip, sep)]
    if uv_exe:
        add_data.append("%s%s." % (uv_exe, sep))

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--name", APP_NAME,
           "--onefile" if not onedir else "--onedir",
           "--distpath", DIST, "--workpath", BUILD, "--specpath", BUILD,
           # 启动器只用标准库 + tkinter，把没用的都排掉能小不少
           "--exclude-module", "torch",
           "--exclude-module", "numpy",
           "--exclude-module", "PySide6",
           "--exclude-module", "matplotlib",
           "--exclude-module", "PIL",
           "--exclude-module", "pytest",
           # 控制台窗口：默认隐藏；想让 --console 也能看到输出就打开这个
           "--console",
           ]
    for d in add_data:
        cmd += ["--add-data", d]
    # Windows 版本资源（文件版本/产品版本/公司名）—— 没有它 exe 属性全是空的
    if os.name == "nt":
        vpath = write_version_file(progress)
        cmd += ["--version-file", vpath]
    cmd.append(os.path.join(PKG, "launcher.py"))

    progress("  运行 PyInstaller……")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       cwd=ROOT)
    txt = r.stdout.decode("utf-8", "replace")
    for line in txt.strip().splitlines()[-12:]:
        progress("    " + line[:150])
    if r.returncode != 0:
        raise RuntimeError("PyInstaller 失败：\n" + txt[-2000:])


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(rel_dir, progress):
    """给发布文件写 `SHA256SUMS`（`<hash>  <name>`，与 sha256sum 格式一致）。

    作用：
      - 用户下载后能确认文件没坏/没被替换（Release 页也放一份）；
      - zip 里带一份，解压后逐个对照就能发现缺文件；
      - 构建时顺带把「发布文件少了」变成硬错误，而不是发出去才发现。
    """
    lines = []
    for name in RELEASE_FILES:
        p = os.path.join(rel_dir, name)
        if not os.path.isfile(p):
            raise RuntimeError(
                "发布文件缺失：%s\n（先检查上面的构建步骤，别把半个发布包发出去）"
                % p)
        lines.append("%s  %s" % (_sha256_of(p), name))
    out = os.path.join(rel_dir, "SHA256SUMS")
    # UTF-8（无 BOM）+ LF：Linux `sha256sum -c` 直接可用；文件名含中文只能 UTF-8
    with open(out, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines) + "\n")
    progress("  SHA256SUMS（%d 个文件，含 exe 校验和）" % len(lines))
    return out


def verify_zip_entries(zpath):
    """校验发布 zip 里**恰好**是那几项（多一个临时文件都不行）。"""
    expected = {"%s/%s" % (APP_NAME, n)
                for n in list(RELEASE_FILES) + ["SHA256SUMS"]}
    with zipfile.ZipFile(zpath) as z:
        got = set(z.namelist())
    if got != expected:
        raise RuntimeError(
            "发布 zip 条目不对：\n  多了：%s\n  少了：%s"
            % (sorted(got - expected), sorted(expected - got)))
    return expected


def assemble(progress, zip_it=False, clean_runtime=False):
    """把发布文件夹拼出来。

    目标形态是用户明确要的「一个文件夹 + 一个 .exe」：

        dist/LingYu/
          LingYu.exe        ← 双击这个
          使用说明.txt
          LICENSE
          README.md
          runtime/          ← 首次双击后才出现
    """
    rel = os.path.join(DIST, APP_NAME)
    os.makedirs(rel, exist_ok=True)

    # 说明文件用 **UTF-8 with BOM** 写。
    # 不加 BOM 的话，老版本的记事本会按 GBK 解码，中文全是乱码 ——
    # 而这份文件的读者恰恰是最需要帮助的新手。（和 lingyu.ps1 同一个坑。）
    txt_path = os.path.join(rel, "使用说明.txt")
    with open(txt_path, "w", encoding="utf-8-sig", newline="\r\n") as fp:
        fp.write(README_TXT)
    progress("  使用说明.txt（UTF-8 with BOM）")

    for name in ("LICENSE", "README.md", "README.zh-CN.md"):
        src = os.path.join(ROOT, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(rel, name))
        else:
            progress("  （%s 还不存在，先跳过）" % name)

    # ★ 把 exe 复制进文件夹里。PyInstaller 默认放在 dist/ 根下，
    #   那样用户拿到的是「一个 exe」+「一个同名文件夹」，会困惑
    #   到底该点哪个。放进文件夹里就只有一种理解。
    exe = os.path.join(DIST, APP_NAME + ".exe")
    if not os.path.isfile(exe):
        raise RuntimeError("没找到 %s —— PyInstaller 没成功？" % exe)
    exe_in = os.path.join(rel, APP_NAME + ".exe")
    if os.path.normcase(exe) != os.path.normcase(exe_in):
        shutil.copy2(exe, exe_in)
        # ⚠️ 复制完必须把 dist/ 根下那个**删掉**。
        #
        #    实测踩过：留着它，用户会点 dist\LingYu.exe（就在同名文件夹旁边，
        #    看起来一模一样），而它以 dist\ 为安装根，于是环境装到了
        #    dist\runtime\ —— 和 dist\LingYu\runtime\ 完全两套。
        #    用户以为在用发布版，其实在用另一个副本，排查时也对不上日志。
        try:
            os.remove(exe)
        except OSError:
            progress("  ⚠ 没能删掉 dist\\%s.exe，请手动删（留着会误导用户）"
                     % APP_NAME)
    size = os.path.getsize(exe_in) / 1024 / 1024
    progress("  %s.exe：%.1f MB" % (APP_NAME, size))

    # ---- runtime\ 不进发布包，但**留在硬盘上** ----
    #
    # 「不该进 zip」和「不该留在硬盘上」是两件事，以前混成一件了。
    #
    # runtime\ 里确实有指向本机的绝对路径（venv、configs 都钉死了），
    # 所以它不能进发布包 —— 这条由下面的 os.walk 排除做到。
    #
    # 但它同时是**一份能直接跑的本地安装**。以前这里无条件 rmtree，
    # 于是「改一行代码 → 重新打包」会把正在用的安装连同 runtime\data\
    # 里准备好的语料一起删掉，而且不提示。实测就这么丢过一次语料。
    # 真要清干净加 --clean-runtime。
    stray = os.path.join(rel, "runtime")
    keep_runtime = os.path.isdir(stray) and not clean_runtime
    if os.path.isdir(stray):
        if keep_runtime:
            progress("  保留 runtime\\（不进 zip；要删掉加 --clean-runtime）")
        else:
            progress("  清掉 runtime\\（里面是本机路径，不该发出去）")
            shutil.rmtree(stray, ignore_errors=True)

    # 校验和必须在打包 zip 之前生成（zip 里要带上它）
    write_sha256sums(rel, progress)

    if zip_it:
        zpath = os.path.join(DIST, "%s-win64.zip" % APP_NAME)
        progress("  压缩成 %s……" % os.path.basename(zpath))
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, _dn, filenames in os.walk(rel):
                if "runtime" in os.path.relpath(dirpath, rel).split(os.sep):
                    continue          # ★ 打包时排除，硬盘上保留
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, DIST))
        verify_zip_entries(zpath)
        progress("  %s：%.1f MB（条目已校验）" % (
            os.path.basename(zpath), os.path.getsize(zpath) / 1024 / 1024))

    progress("  文件夹内容：")
    for fn in sorted(os.listdir(rel)):
        p = os.path.join(rel, fn)
        progress("    %-22s %10s" % (
            fn, ("%.1f MB" % (os.path.getsize(p) / 1024 / 1024))
            if os.path.getsize(p) > 1024 * 1024
            else ("%d KB" % (os.path.getsize(p) / 1024))))
    return rel


def main():
    ap = argparse.ArgumentParser(description="打包灵语发布版")
    ap.add_argument("--zip", action="store_true", help="顺便压成 zip")
    ap.add_argument("--skip-build", action="store_true",
                    help="跳过 PyInstaller，只重新拼发布文件夹（调试用）")
    ap.add_argument("--clean-runtime", action="store_true",
                    help="连 dist\\LingYu\\runtime 一起删掉（默认保留 —— 那是一份能直接跑的本地安装）")
    ap.add_argument("--onedir", action="store_true",
                    help="打成文件夹而不是单文件（启动快一点，但不好分发）")
    a = ap.parse_args()

    def progress(msg):
        print(msg, flush=True)

    progress("=== 打包灵语 ===")
    problems = check_prereqs()
    if problems:
        for p in problems:
            progress("✗ " + p)
        return 2

    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(DIST, exist_ok=True)

    if not a.skip_build:
        progress("[1/3] 打包引擎源码")
        src = make_source_zip(progress)
        progress("[2/3] 找 uv.exe")
        uv = find_uv()
        if uv:
            progress("  找到 %s（%.1f MB）" % (uv,
                                             os.path.getsize(uv) / 1024 / 1024))
        else:
            progress("  ⚠ 没找到 uv.exe —— 打出来的 exe 在「用户机器上没有"
                     "Python」时无法自动下载 Python")
        progress("[3/3] 打启动器 exe")
        run_pyinstaller(src, uv, progress, onedir=a.onedir)
    else:
        progress("跳过构建，直接拼发布文件夹")

    rel = assemble(progress, zip_it=a.zip, clean_runtime=a.clean_runtime)
    progress("\n完成：%s" % rel)
    progress("双击 %s\\%s.exe 试试。" % (rel, APP_NAME))
    return 0


if __name__ == "__main__":
    sys.exit(main())
