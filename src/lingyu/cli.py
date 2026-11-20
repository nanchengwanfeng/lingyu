# -*- coding: utf-8 -*-
'''命令行入口。所有功能都从这里进。

    python -m lingyu doctor                      体检：环境能不能训练
    python -m lingyu scan    --raw <目录>         只做语料体检，不写文件
    python -m lingyu prepare --raw <目录> --data <目录>   数据准备
    python -m lingyu params  --config <配置>      只算参数量与 KV cache
    python -m lingyu plan    --config <配置>      训练前预估显存与耗时
    python -m lingyu train   --config <配置>      开始训练
    python -m lingyu sample  --config <配置>      用训练好的模型写中文
'''

import argparse
import io
import os
import sys

from . import utils
from .version import __version__


def _project_root():
    """引擎安装根目录（= `paths.home()`，用户数据所在的那一层）。

    三种装法要都能算对：
      1. 开发：`<项目>/src/lingyu/cli.py` → 上三级是项目根
      2. pip install -e .：同上
      3. **打包版**：启动器设了 `LINGYU_HOME`，直接用它
         （那时源码在 runtime/app 下，靠 __file__ 推算会算到 runtime/app）

    ★ 规则只有一份，写在 `paths.py` 里。界面子进程、命令行、启动器
      全都走它 —— 以前界面自己算了一份，结果算到 `runtime/app`，
      和这里算的 `runtime` 不是一个地方，语料就这么「丢」了一次。
    """
    from . import paths
    return paths.home()


def _venv_python(root):
    """这个安装该用的 venv 解释器路径（可能不存在）。"""
    return os.path.join(root, "venv", "Scripts", "python.exe") \
        if os.name == "nt" else os.path.join(root, "venv", "bin", "python")


def _require_deps(cmd_name, need_torch=False, need_numpy=True, need_gui=False):
    """开工前先查依赖，缺了就立刻停下并说清原因。

    为什么不靠 import 报错：
        引擎顶层刻意不导入 numpy（为了轻量），所以缺 numpy 时
        要跑到 prepare 的第 5 步才崩 —— 前面白跑几分钟。
        更坑的是「venv 没激活、用了系统 Python」也会表现成缺包，
        报错信息完全看不出真正原因。

    ⚠️ 这里给出的每一条命令都必须是**当前这台机器上真的能跑的**。
       以前这里写死了 `C:\\path\\to\\lingyu\\lingyu.ps1` —— 那是开发者自己
       机器上的路径。别人 clone 下来或者用打包版，照着敲只会得到
       「找不到路径」，然后以为是引擎坏了。
       现在一律用 `sys.executable`（就是正在跑的这个解释器，永远正确）
       和 `_project_root()` 算出来的路径。
    """
    import sys
    missing = []
    if need_numpy:
        try:
            import numpy  # noqa: F401
        except ImportError:
            missing.append("numpy")
    if need_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch")
    gui_missing = False
    if need_gui:
        try:
            import PySide6  # noqa: F401
        except ImportError:
            gui_missing = True
    if not missing and not gui_missing:
        return True

    root = _project_root()
    venv_py = _venv_python(root)
    in_venv = os.path.normcase(venv_py) in os.path.normcase(sys.executable)
    packaged = bool(os.environ.get("LINGYU_HOME"))

    if missing:
        utils.err("缺少依赖：%s" % "、".join(missing))
    if gui_missing:
        utils.err("没装图形界面需要的 PySide6")

    print()
    utils.info("当前用的解释器：")
    utils.info("  %s" % sys.executable)

    if packaged:
        # 打包版：环境是启动器装的，缺依赖说明装坏了
        utils.info("这是打包版（运行时目录：%s）" % root)
        utils.info("修复办法 —— 重新装一遍依赖：")
        utils.info("  双击 %s.exe 时按住 Shift，或用命令行：" % "LingYu")
        utils.info("    LingYu.exe --console --reinstall")
    elif in_venv:
        utils.info("这是引擎的 venv，但依赖没装全。执行：")
        utils.info("  \"%s\" -m pip install %s"
                   % (sys.executable, " ".join(missing or ["PySide6-Essentials"])))
    else:
        utils.warn("这个 Python 不是引擎的虚拟环境 —— 多半是 venv 没激活。")
        utils.info("三种解法（任选一种）：")
        if os.path.isfile(venv_py):
            utils.info("  ① 直接用 venv 里的解释器（不用激活，最省事）：")
            utils.info("       \"%s\" -m lingyu %s" % (venv_py, cmd_name))
            utils.info("  ② 先激活再跑：")
            utils.info("       & \"%s\""
                       % os.path.join(root, "venv", "Scripts", "Activate.ps1"))
            utils.info("       python -m lingyu %s" % cmd_name)
        else:
            utils.info("  ① 先按 README 建好虚拟环境并装依赖")
        ps1 = os.path.join(root, "lingyu.ps1")
        if os.path.isfile(ps1):
            utils.info("  ③ 用启动脚本（它会自动用对解释器）：")
            utils.info("       & \"%s\" %s" % (ps1, cmd_name))
    if not packaged:
        print()
        utils.info("诊断命令（会打印解释器路径和依赖状态）：")
        utils.info("  \"%s\" -m lingyu doctor" % sys.executable)
    return False


def build_parser():
    p = argparse.ArgumentParser(
        prog="lingyu",
        description="灵语 —— 一个从零手写的、中文友好的小型语言模型训练引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # 帮助里的示例**不能写死开发者机器上的路径**。
        # 用通用写法：Windows 和 Linux 都看得懂，抄下来改一下就能用。
        epilog='''典型流程（把 <语料目录> 换成你自己的，例如 D:\\我的小说）：

  lingyu prepare --raw <语料目录> --data <数据目录>   # 准备数据
  lingyu plan    --config configs/small.ini           # 看显存够不够
  lingyu train   --config configs/small.ini           # 预训练
  lingyu sft     --config configs/sft.ini             # 教它回答问题
  lingyu chat    --config configs/sft.ini             # 和它聊天

想用图形界面（所有参数都能点）：

  lingyu gui
''')
    p.add_argument("-V", "--version", action="version",
                   version="灵语 LingYu %s" % __version__)

    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("doctor", help="体检：环境能不能训练")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")
    sp.add_argument("--raw", default=None, help="顺便检查这个语料目录里有多少可读语料（txt/md/json）")

    sp = sub.add_parser("scan", help="只做语料体检，算建议词表大小（不写任何文件）")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")
    sp.add_argument("--raw", required=True, help="放语料的目录（txt/md/json/jsonl）")
    sp.add_argument("--sample-mb", type=float, default=8.0,
                    help="最多读多少 MB 来统计（默认 8，语料很大时调小更快）")
    sp.add_argument("--exclude", action="append", default=[],
                    help="排除某个文件名，可写多次（默认已排除本引擎放的说明文件）")
    sp.add_argument("--json-extract", action="store_true",
                    help="抽 JSON 正文（默认把 .json/.jsonl 原样当语料，保留结构）")

    sp = sub.add_parser("prepare", help="数据准备：训练分词器 + 把语料编码成 bin 文件")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")
    sp.add_argument("--raw", required=True, help="放语料的目录（txt/md/json/jsonl，支持子文件夹）")
    sp.add_argument("--data", required=True, help="输出目录：分词器和 train.bin/val.bin 都放这里")
    sp.add_argument("--exclude", action="append", default=[],
                    help="排除某个文件名，可写多次（默认已排除本引擎放的说明文件）")
    sp.add_argument("--tokenizer", default="bpe", choices=["bpe", "char"],
                    help="bpe=子词，压缩比高（默认，推荐）；char=一字一 token，最直观")
    sp.add_argument("--vocab-size", type=int, default=None,
                    help="词表大小。默认自动（字符多则 8192，少则 4096）。调小省显存但压缩比会降")
    sp.add_argument("--val-ratio", type=float, default=0.02,
                    help="验证集占多少（默认 0.02 = 2%%）。验证集用来判断有没有过拟合")
    sp.add_argument("--tokenizer-sample-mb", type=float, default=3.0,
                    help="训练 BPE 时抽多少 MB 子集（默认 3）。越大越慢但词表更全")
    sp.add_argument("--no-clean", action="store_true",
                    help="不清洗文本（默认会统一换行、去零宽字符、合并空行）")
    sp.add_argument("--strip-ads", action="store_true",
                    help="顺便去掉网址和常见广告串（网文下载版建议开）")
    sp.add_argument("--json-extract", action="store_true",
                    help="抽 JSON 正文（默认把 .json/.jsonl 原样当语料，保留 {} 和键名）")
    sp.add_argument("--min-pair-count", type=int, default=2,
                    help="出现次数低于此值的字对不合并（默认 2，滤掉偶然组合）")
    sp.add_argument("--max-token-chars", type=int, default=4,
                    help="一个 token 最多几个字（默认 4 = 最多四字词）。调大压缩比更高但会有长串")
    sp.add_argument("--seed", type=int, default=1234, help="随机种子，固定后结果可复现")

    for name, helptext in (
            ("params", "只算参数量与 KV cache，不训练（改配置后先跑这个）"),
            ("plan", "训练前预估显存占用和耗时（改配置后先跑这个）")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--config", required=True, help="配置文件，如 configs/small.ini")
        sp.add_argument("--set", action="append", default=[],
                        help="临时覆盖配置，如 --set model.n_layer=12（可写多次）")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")

    sp = sub.add_parser("train", help="开始训练")
    sp.add_argument("--config", required=True, help="配置文件，如 configs/small.ini")
    sp.add_argument("--set", action="append", default=[],
                    help="临时覆盖配置，如 --set train.max_steps=2000（可写多次）")
    sp.add_argument("--resume", default=None, nargs="?", const="auto",
                    help="续训。只写 --resume 就用配置里的 out_dir/ckpt.pt；"
                         "也可直接给路径")
    sp.add_argument("--force", action="store_true",
                    help="out_dir 里已有存档时，强制从头重训（默认会停下提示你，"
                         "以免手滑覆盖掉练了很久的模型）")
    sp.add_argument("--init-from", default=None,
                    help="用另一个断点的权重作起点（例如 SFT 之后继续预训练）。"
                         "和 --resume 二选一；步数从 0 开始，不继承优化器状态")
    sp.add_argument("--device", default=None,
                    help="cuda 或 cpu，默认自动（有显卡就用显卡）")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")

    sp = sub.add_parser("sample", help="用训练好的模型写中文")
    sp.add_argument("--config", default=None,
                    help="配置文件；给了就用里面的 out_dir 找断点、sample 段定参数")
    sp.add_argument("--set", action="append", default=[], help="临时覆盖配置")
    sp.add_argument("--ckpt", default=None,
                    help="断点文件路径。不给就用配置里 out_dir 下的 best.pt 或 ckpt.pt")
    sp.add_argument("--prompt", default=None, help="开头文字，模型接着写")
    sp.add_argument("--max-new-tokens", type=int, default=None, help="最多生成多少 token（默认 200）")
    sp.add_argument("--temperature", type=float, default=None,
                    help="温度（默认 0.9）。调低=更稳更重复，调高=更新鲜更跑偏。0.7~1.0 合适")
    sp.add_argument("--top-k", type=int, default=None,
                    help="只在前 k 个候选里抽（默认 40）。调小更保守")
    sp.add_argument("--top-p", type=float, default=None,
                    help="核采样阈值（默认 0.95）。只保留累积概率到 0.95 的候选")
    sp.add_argument("--repetition-penalty", type=float, default=None,
                    help="重复惩罚（默认 1.05）。一直重复同一句就调到 1.15~1.3")
    sp.add_argument("--seed", type=int, default=None, help="随机种子，固定后每次结果一样")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")
    sp.add_argument("--no-stream", action="store_true",
                    help="不逐字打印，等生成完一次性输出（想直接复制文字时用）")
    sp.add_argument("--device", default=None, help="cuda 或 cpu")

    sp = sub.add_parser("sft", help="监督微调：教模型「回答问题」而不只是续写")
    sp.add_argument("--config", required=True, help="配置文件")
    sp.add_argument("--set", action="append", default=[],
                    help="临时覆盖配置，如 --set post.max_steps=500")
    sp.add_argument("--data", default=None,
                    help="对话数据集（jsonl）。给了就覆盖 post.data_path")
    sp.add_argument("--template", default=None,
                    help="对话模板：plain（中文标记，推荐）/ chatml / qwen")
    sp.add_argument("--init-from", default=None,
                    help="从哪个预训练断点开始。不给就自动找 [train] out_dir")
    sp.add_argument("--out-dir", default=None, help="SFT 输出目录")
    sp.add_argument("--force", action="store_true",
                    help="已存在 SFT 断点时也重新开始（默认会停下提示你）")
    sp.add_argument("--device", default=None, help="cuda 或 cpu")
    sp.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="接着上次的 SFT 断点继续。不给路径时用 "
                         "post.out_dir/ckpt_sft.pt；也可 --resume <断点路径>")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")

    sp = sub.add_parser("chat", help="和训练好的模型对话（需要先做 SFT）")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")
    sp.add_argument("--config", default=None, help="配置文件")
    sp.add_argument("--set", action="append", default=[], help="临时覆盖配置")
    sp.add_argument("--ckpt", default=None,
                    help="断点路径。不给就依次找 post.out_dir、train.out_dir")
    sp.add_argument("--prompt", default=None,
                    help="一上来就问这一句（不给就进入交互模式）")
    sp.add_argument("--no-stream", action="store_true",
                    help="不逐字打印，等一整段生成完再输出")
    sp.add_argument("--device", default=None, help="cuda 或 cpu")

    sp = sub.add_parser("launch", help="生成分布式训练启动脚本（本机跑不动时用）")
    sp.add_argument("--json", action="store_true",
                    help="输出 JSONL 事件（GUI 用；人类日志改走 stderr）")
    sp.add_argument("--config", required=True, help="配置文件")
    sp.add_argument("--set", action="append", default=[], help="临时覆盖配置")
    sp.add_argument("--backend", default=None,
                    help="目标后端：fsdp（实验性，仅生成脚本，分布式训练未实现）"
                         " / deepspeed / megatron。"
                         "不给就用配置里 parallel.backend")
    sp.add_argument("--out-dir", default="launch_scripts", help="脚本输出目录")
    sp.add_argument("--print-only", action="store_true",
                    help="只打印脚本内容，不写文件")

    sp = sub.add_parser("gui", help="打开图形界面（推荐新手用这个）")
    sp.add_argument("--config", default=None, help="启动时载入哪个配置")
    return p


class ConfigInvalid(Exception):
    """配置校验失败 —— main() 捕获后以退出码 2 结束，不吐 traceback。"""


def _validate_cfg(cfg, where="配置"):
    """校验已 resolve 的完整配置；有 error 就打人话并抛 ConfigInvalid。

    ★ 入口统一校验。以前只有 GUI 概览页调 validate，CLI 训练完全不设防：
      实测 `--set model.n_head=0` 会一路走到建模型才抛
      `ZeroDivisionError: integer modulo by zero`，`n_layer=-5` 则建出
      负层数模型；INI 里的 `[Model]`/笔误/`none` 也全是静默忽略。
    """
    from .config import validate
    errors, warnings, _ = validate(cfg, scale_notice=False)
    for w in warnings:
        utils.warn(w)
    if not errors:
        return
    utils.err("%s有 %d 处错误，先修好再跑：" % (where, len(errors)))
    for e in errors:
        utils.err(e)
    raise ConfigInvalid("%s校验失败（%d 处错误），已中止。"
                        % (where, len(errors)))


def load_cfg(path, sets, check=True):
    """读配置 → 应用 --set 覆盖 → **补全 spec 默认值** → **校验**。

    ★ 最后那步 `resolve()` 不能省。下游（train.py / sample.py）是直接下标
      取值的（`cfg["weight_decay"]`），配置里没写就抛
      `KeyError: 'weight_decay'` —— 用户只写想改的那几项是很自然的用法，
      不该因此崩掉。补全只做在入口这一处，后面就都能安心用下标。

    ★ `on_notice` 也必须接上：`apply_overrides` 用回调而不是直接 print，
      是为了让 GUI 也能拿到这些提示。不接回调的后果是「自动纠正」发生了
      但用户永远看不到 —— 他会以为自己真的写了个叫 `n_layers` 的键。

    ★ 校验放在 **--set 之后**：用户真正要跑的是覆盖后的配置，不能只校验文件。
    """
    from .config import load_config, apply_overrides, resolve
    cfg = load_config(path)
    if sets:
        apply_overrides(cfg, sets, on_notice=utils.info)
    cfg = resolve(cfg)
    if check:
        _validate_cfg(cfg)
    return cfg


# ---------------------------------------------------------------- doctor
def cmd_doctor(args):
    utils.set_json_mode(getattr(args, "json", False))
    utils.title("灵语环境体检")
    ok_all = True

    utils.info("[Python]")
    utils.info("  版本      : %s" % sys.version.split()[0])
    utils.info("  解释器    : %s" % sys.executable)
    if sys.version_info[:2] >= (3, 13):
        utils.warn("  Python 3.13+ 通常没有 PyTorch 轮子，建议 3.11 或 3.12")

    utils.info("[PyTorch]")
    try:
        import torch
        # ★ 先分清两种「没有 GPU」，它们的修法完全不同：
        #     ① 装的是 **CPU 版** torch（torch.version.cuda is None）
        #        -> 重装成 CUDA 版就好
        #     ② 装的是 CUDA 版但用不了（驱动太老 / 卡不支持）
        #        -> 要更新驱动或换 CUDA 版本
        #   以前两种情况都只说一句「CUDA 不可用」，用户根本不知道该干嘛。
        is_cpu_build = getattr(torch.version, "cuda", None) is None
        if is_cpu_build:
            utils.warn("  torch %s" % torch.__version__)
            ok_all = False
            utils.err("  这是 **CPU 版** PyTorch —— 它里面根本没有 CUDA 支持，"
                      "有显卡也用不上。")
            utils.info("  为什么：PyPI 上 Windows 的 torch 轮子就是 CPU 版，")
            utils.info("          CUDA 版在 PyTorch 自己的索引里，要单独指定源。")
            utils.info("  怎么修（复制整行执行）：")
            utils.info("    \"%s\" -m pip install torch --force-reinstall "
                       "--index-url https://download.pytorch.org/whl/cu130"
                       % sys.executable)
            utils.info("  装完再跑一次： lingyu doctor")
        else:
            utils.ok("  torch %s" % torch.__version__)
            utils.info("  CUDA 编译版 : %s" % torch.version.cuda)
            arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
            utils.info("  支持架构    : %s" % (", ".join(arch) if arch else "未知"))
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            utils.ok("  显卡        : %s (sm_%d%d, %.2f GB)"
                     % (pr.name, pr.major, pr.minor, pr.total_memory / 1024 ** 3))
            try:
                x = torch.randn(512, 512, device="cuda")
                _ = (x @ x).sum().item()
                torch.cuda.synchronize()
                utils.ok("  计算核测试  : 通过（能真正跑 GPU 计算）")
            except Exception as e:
                ok_all = False
                utils.err("  计算核测试  : 失败 -> %r" % (e,))
                utils.err("  这是「装了但算不起来」，通常是 较新的 GPU 架构 内核缺失")
                utils.err("  换 cu130 分支更新的 torch，或用 nightly")
            for dt in (torch.bfloat16, torch.float16):
                try:
                    a = torch.randn(128, 128, device="cuda", dtype=dt)
                    _ = (a @ a).float().sum().item()
                    torch.cuda.synchronize()
                    utils.ok("  %-10s : 可用" % dt)
                except Exception as e:
                    utils.warn("  %-10s : 不可用（%s）" % (dt, type(e).__name__))
        elif not is_cpu_build:
            ok_all = False
            utils.err("  CUDA 版 torch 装着，但用不了 —— 只能 CPU 训练（会很慢）")
            utils.info("  常见原因（按可能性排序）：")
            utils.info("    ① 显卡驱动太老 -> 去 NVIDIA 官网更新驱动")
            utils.info("    ② 这张卡太老，当前 CUDA 版本不支持了")
            utils.info("    ③ 没有 NVIDIA 显卡（核显/AMD 卡跑不了 CUDA）")
            utils.info("  想看更具体的原因： \"%s\" -c \"import torch;"
                       "torch.cuda.init()\"" % sys.executable)
    except ImportError:
        ok_all = False
        utils.err("  没装 PyTorch。装它：")
        utils.info("    \"%s\" -m pip install torch --index-url "
                   "https://download.pytorch.org/whl/cu130" % sys.executable)

    utils.info("[可选依赖]")
    for mod in ("numpy", "matplotlib"):
        try:
            __import__(mod)
            utils.ok("  %s" % mod)
        except ImportError:
            utils.warn("  %s 没装（画曲线要用，不影响训练）" % mod)

    utils.info("[引擎自身]")
    try:
        from .tokenizer import CharTokenizer, BPETokenizer  # noqa: F401
        from .model import LingYu, build_config  # noqa: F401
        utils.ok("  分词器 + 模型模块都能导入")
    except Exception as e:
        ok_all = False
        utils.err("  模块导入失败: %r" % (e,))

    if args.raw:
        utils.info("[语料目录]")
        from .prepare import find_text_files
        if os.path.isdir(args.raw):
            fs = find_text_files(args.raw)
            if fs:
                tot = sum(os.path.getsize(f) for f in fs)
                utils.ok("  %s：%d 个文件，合计 %.2f MB"
                         % (args.raw, len(fs), tot / 1024 / 1024))
            else:
                utils.warn("  %s 里没找到 .txt/.md/.json/.jsonl 文件" % args.raw)
        else:
            utils.warn("  目录不存在: %s" % args.raw)

    utils.info("[设备后端]")
    try:
        from . import devices as _devices
        for line in _devices.describe():
            utils.info(line)
        utils.info("  auto 选择 : %s" % _devices.pick("auto"))
    except Exception as exc:  # 探测失败不影响体检
        utils.warn("  设备探测失败: %r" % (exc,))

    print()
    utils.emit_event("doctor", task="doctor", ok=ok_all,
                     python=sys.version.split()[0], executable=sys.executable)
    if ok_all:
        utils.ok("体检通过，可以开始训练。")
        return 0
    utils.err("有项目不通过，先解决上面标 [错误] 的问题。")
    return 1


# ---------------------------------------------------------------- scan
def cmd_scan(args):
    utils.set_json_mode(getattr(args, "json", False))
    # 纯文本统计，**不需要 numpy / torch** —— 加检查反而会把一个本来
    # 能跑的命令拦下来。所以这里刻意不做 _require_deps
    # （下面的 inspect_corpus 只用标准库的 Counter）。
    from .prepare import find_text_files, read_corpus_file, clean_text
    from .tokenizer import inspect_corpus
    utils.title("语料体检（不写任何文件）")
    files = find_text_files(args.raw, extra_ignore=getattr(args, "exclude", None))
    json_mode = "extract" if getattr(args, "json_extract", False) else "raw"
    if not files:
        utils.err("在 %s 里没找到文本文件" % args.raw)
        return 1
    limit = int(args.sample_mb * 1024 * 1024)
    buf, used = [], 0
    n_read = 0
    n_skipped = 0
    shown = 0
    from collections import Counter as _Counter
    ext_counts = _Counter(os.path.splitext(f)[1].lower() or "(无扩展名)"
                          for f in files)
    json_rows = []
    for f in files:
        text, enc, n_json = read_corpus_file(f, json_mode=json_mode)
        is_json = os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".ndjson")
        if not text.strip():
            n_skipped += 1
            utils.emit_event("file", task="scan", path=os.path.basename(f),
                             enc=enc, chars=0, skipped=True)
            if is_json:
                utils.warn("  %-44s 跳过：这个 JSON 里没有可读文字"
                           % os.path.basename(f)[:42])
            else:
                utils.warn("  %-44s 跳过：空文件（没有正文）"
                           % os.path.basename(f)[:42])
            continue
        shown += 1
        tail = "（%d 条记录）" % n_json if n_json is not None else ""
        # ★ JSON/JSONL 必须**每个都打**：目录里混着几千个 md 时，
        #   用户第一眼要确认的就是「我的 jsonl 算不算」。
        if is_json or shown <= 10 or shown % 500 == 0:
            utils.info("  %-44s %-22s %s 字符%s"
                       % (os.path.basename(f)[:42], enc,
                          "{:,}".format(len(text)), tail))
        if is_json:
            json_rows.append((os.path.basename(f), len(text), n_json))
        n_read += 1
        utils.emit_event("file", task="scan", path=os.path.basename(f),
                         enc=enc, chars=len(text), records=n_json)
        text = clean_text(text)
        if used < limit:
            take = min(len(text), limit - used)
            buf.append(text[:take])
            used += take
    corpus = "".join(buf)
    info = inspect_corpus(corpus)
    utils.title("文件汇总")
    utils.info("按类型      : " + "  ".join(
        "%s %d 个" % (e, n) for e, n in ext_counts.most_common()))
    utils.info("读入/跳过   : %d 个 / %d 个%s"
               % (n_read, n_skipped,
                  "（明细只显示前 10 个和每 500 个一行）"
                  if n_read > 10 else ""))
    if json_rows:
        utils.info("JSON/JSONL  : %d 个，全部读入"
                   % len(json_rows))
        for name, chars, recs in json_rows[:10]:
            utils.info("  %-46s %s 字符%s"
                       % (name[:44], "{:,}".format(chars),
                          "（%d 条记录）" % recs if recs is not None else ""))
    utils.title("统计结果")
    utils.info("读入          : %s 字符" % "{:,}".format(info["chars"]))
    utils.info("不同字符种类  : %s 种" % "{:,}".format(info["distinct_chars"]))
    utils.info("抽样二字组    : %s 种" % "{:,}".format(info["distinct_pairs_sampled"]))
    utils.info("出现最多的字  : %s" % "".join(ch for ch, _ in info["top_chars"][:24]))
    print()
    utils.info("建议词表大小：")
    utils.info("  字符级 char : %s" % "{:,}".format(info["distinct_chars"] + 8))
    utils.info("  BPE         : %s" % ("8192" if info["distinct_chars"] > 3000 else "4096"))
    if info["distinct_chars"] < 2500:
        utils.warn("不同字符只有 %d 种，语料可能偏少或偏单一" % info["distinct_chars"])
    print()
    utils.emit_event("scan", task="scan", raw=args.raw, files=n_read,
                     skipped=n_skipped, chars=info["chars"],
                     distinct_chars=info["distinct_chars"],
                     vocab_hint=(8192 if info["distinct_chars"] > 3000 else 4096),
                     json_extract=bool(getattr(args, "json_extract", False)))
    utils.ok("下一步:  python -m lingyu prepare --raw %s --data <输出目录>" % args.raw)
    return 0


# ---------------------------------------------------------------- prepare
def cmd_prepare(args):
    utils.set_json_mode(getattr(args, "json", False))
    if not _require_deps("prepare", need_numpy=True):
        return 6
    from .prepare import run_prepare
    run_prepare(
        raw_dir=args.raw, data_dir=args.data,
        tokenizer_type=args.tokenizer, vocab_size=args.vocab_size,
        val_ratio=args.val_ratio, tokenizer_sample_mb=args.tokenizer_sample_mb,
        do_clean=not args.no_clean, do_strip_ads=args.strip_ads,
        seed=args.seed, min_pair_count=args.min_pair_count,
        json_mode="extract" if getattr(args, "json_extract", False) else "raw",
        max_token_chars=args.max_token_chars,
        exclude=getattr(args, "exclude", None))
    import json as _json
    meta = {}
    for _name in ("train", "val"):
        _p = os.path.join(args.data, _name + ".meta.json")
        if os.path.isfile(_p):
            try:
                with io.open(_p, encoding="utf-8") as _fp:
                    meta[_name] = _json.load(_fp)
            except Exception:
                pass
    _tr = meta.get("train") or {}
    _va = meta.get("val") or {}
    utils.emit_event(
        "prepare", task="prepare", data_dir=args.data,
        json_extract=bool(getattr(args, "json_extract", False)),
        vocab_size=_tr.get("vocab_size"),
        tokenizer_type=_tr.get("tokenizer_type"),
        train_tokens=_tr.get("n_tokens"), train_bytes=_tr.get("n_bytes"),
        val_tokens=_va.get("n_tokens"), val_bytes=_va.get("n_bytes"),
        corpus_chars=_tr.get("corpus_chars"),
        compression=_tr.get("compression"),
        tokenizer_fingerprint=_tr.get("tokenizer_fingerprint"))
    return 0


# ---------------------------------------------------------------- params/plan
def _load_tok_for_config(cfg):
    from .config import get
    from .tokenizer import load_tokenizer
    tp = get(cfg, "data", "tokenizer_path", None)
    if not tp:
        dd = get(cfg, "data", "data_dir", None)
        if dd:
            tp = os.path.join(dd, "tokenizer.txt")
    if not tp or not os.path.isfile(tp):
        utils.err("找不到分词器（配置里 tokenizer_path = %s）" % tp)
        utils.err("先运行:  python -m lingyu prepare --raw <语料目录> --data <数据目录>")
        return None
    return load_tokenizer(tp)


def _print_param_report_generic(model, mc):
    """非内置模型（自定义 [model] type）的参数明细：按顶层模块分组。

    内置 LingYu 有 param_groups()，能按「词嵌入/注意力/前馈」等语义分组；
    自定义模型没有这个约定，就按 named_parameters 的第一段名字分组 ——
    不假装知道它的内部结构。
    """
    groups = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(groups.values())
    utils.title("参数量明细（自定义模型：按顶层模块分组）")
    for key in sorted(groups):
        v = groups[key]
        utils.info("%-14s: %14s  (%5.2f M)" % (key[:14], "{:,}".format(v), v / 1e6))
    utils.info("总计          : %14s  (%5.2f M)"
               % ("{:,}".format(total), total / 1e6))
    if mc.get("use_moe"):
        utils.info("MoE 稀疏度    : 自定义模型未提供 active_param_count，跳过")
    return total


# 超过这个参数量就不实例化模型：fp32 下约 4 GB，再大会把机器拖死。
_INSTANTIATE_MAX_PARAMS = 1_000_000_000


def _print_param_report(mc, tok):
    """打印参数明细。

    ★ 所有数字都来自**真实模型**的参数分组（`model.param_groups()`），
      这里不再另写一套公式。

    为什么改掉手写公式（踩过的坑）：这里曾经按 `2 * d_model` 算归一化，
    而 RMSNorm 只有一个 weight、其实是 `d_model`，于是屏幕上永远挂着
    一句「公式与实测差 4,224，以实测为准」。更麻烦的是这种公式不会
    跟着配置走 —— 用户一切换 `norm=layernorm` 或换成插件组件，
    公式立刻失效，报出来的差值就成了假警报。

    教训：**能用实测就别用公式**；公式只该出现在给学生看的注释里。

    例外：模型大到本机装不下时不能真去建它（会吃几十 GB 内存甚至把机器
    拖死）。这种情况用 `estimate_param_count` 的解析式估算，并明确标注。
    """
    from .model import build_model, estimate_param_count
    est = estimate_param_count(mc)
    if est is not None and est > _INSTANTIATE_MAX_PARAMS:
        utils.title("参数量明细")
        utils.info("模型规模过大（约 %s 参数，%.1f B），不实例化实测："
                   % ("{:,}".format(est), est / 1e9))
        utils.info("  以下为结构估算值；显存与耗时见下一节。")
        utils.info("总计（估算）  : %14s  (%5.2f B)"
                   % ("{:,}".format(est), est / 1e9))
        return est
    model = build_model(mc)
    if not hasattr(model, "param_groups"):
        return _print_param_report_generic(model, mc)
    groups = model.param_groups()

    def g(*keys):
        return sum(v for k, v in groups.items() if any(x in k for x in keys))

    n_emb = g("词嵌入", "输出层")
    n_attn = g("注意力")
    n_mlp = g("前馈", "MoE")
    n_norm = g("归一化")
    n_other = g("其它")
    total = n_emb + n_attn + n_mlp + n_norm + n_other
    total_real = sum(p.numel() for p in model.parameters())

    utils.title("参数量明细")
    utils.info("词嵌入+输出   : %14s  (%5.2f M)   词表 %s × %d 维%s"
               % ("{:,}".format(n_emb), n_emb / 1e6, "{:,}".format(mc["vocab_size"]),
                  mc["d_model"], "（输入输出共享）" if mc["tie_weights"] else ""))
    utils.info("注意力        : %14s  (%5.2f M)   %d 层 × %d Q头 / %d KV头"
               % ("{:,}".format(n_attn), n_attn / 1e6, mc["n_layer"],
                  mc["n_head"], mc["n_kv_head"]))
    utils.info("前馈          : %14s  (%5.2f M)%s"
               % ("{:,}".format(n_mlp), n_mlp / 1e6,
                  "   ← MoE 稀疏专家" if mc["use_moe"] else ""))
    utils.info("归一化        : %14s  (%5.2f M)   %s，共 %d 处（每层 2 个 + 末尾 1 个）"
               % ("{:,}".format(n_norm), n_norm / 1e6, model.norm_name,
                  mc["n_layer"] * 2 + 1))
    if n_other:
        utils.info("其它          : %14s  (%5.2f M)"
                   % ("{:,}".format(n_other), n_other / 1e6))
    print()
    utils.info("总计          : %14s  (%5.2f M)"
               % ("{:,}".format(total_real), total_real / 1e6))
    if total != total_real:      # 分组逻辑漏了参数才会走到这里，属于引擎 bug
        utils.warn("参数分组漏算了 %s 个（分组和 %s ≠ 实测 %s），请报 issue"
                   % ("{:,}".format(total_real - total),
                      "{:,}".format(total), "{:,}".format(total_real)))

    if mc["use_moe"]:
        full = model.blocks[0].mlp_param_count(active=False)
        act = model.blocks[0].mlp_param_count(active=True)
        utils.info("MoE 稀疏度    : 每层总容量 %s / 每 token 实算 %s = %.2f 倍"
                   % ("{:,}".format(full), "{:,}".format(act), full / max(1, act)))
        ap = model.active_param_count()
        utils.info("激活参数      : %s (%.2f M)  ← 每个 token 实际只算这么多"
                   % ("{:,}".format(ap), ap / 1e6))
        # MoE 的激活参数**本来**就小于总参数，这是稀疏的意义，不是 bug。
        # 只检查「激活参数不该超过总参数」这个方向。
        if ap > total_real:
            utils.warn("激活参数估算 %s > 总参数 %s，差 %s，请报 issue"
                       % ("{:,}".format(ap), "{:,}".format(total_real),
                          "{:,}".format(ap - total_real)))
    elif model.active_param_count() != total_real:
        utils.warn("稠密模型激活参数应等于总参数，实测 %s ≠ %s，请报 issue"
                   % ("{:,}".format(model.active_param_count()),
                      "{:,}".format(total_real)))
    return total_real

    per_tok = 2 * mc["n_layer"] * mc["n_kv_head"] * mc["head_dim"] * 2
    utils.title("KV cache（只在生成时占用）")
    utils.info("每 token      : %s" % utils.fmt_bytes(per_tok))
    for L in (256, 512, 1024, 2048, 4096):
        utils.info("  %5d 上下文 : %s" % (L, utils.fmt_bytes(per_tok * L)))
    mha = 2 * mc["n_layer"] * mc["n_head"] * mc["head_dim"] * 2
    if mha != per_tok:
        utils.info("对比 MHA（%d 个 KV 头）每 token 要 %s —— GQA 省了 %.0f%%"
                   % (mc["n_head"], utils.fmt_bytes(mha), 100.0 * (1 - per_tok / mha)))
    return total_real


def cmd_params(args):
    if not _require_deps("params", need_torch=True, need_numpy=False):
        return 6
    from .model import config_from_section
    cfg = load_cfg(args.config, args.set)
    tok = _load_tok_for_config(cfg)
    if tok is None:
        return 1
    mc = config_from_section(cfg.get("model", {}), tok.vocab_size)
    _print_param_report(mc, tok)
    return 0


def _plan_scope(cfg, mc):
    """规划显存/耗时用的 (batch, grad_accum, max_steps, seq, is_post)。

    ★ 预训练配置读 [train]，SFT/DPO/LoRA 配置读 [post]（审计 B-17）：
    以前一律读 `train.batch_size`，而 sft.ini 根本没有这个键 → fallback
    到默认 32，把 SFT 的显存和耗时算成 98% 的假警告。后训练真正用的是
    `post.batch_size` / `post.max_seq_len` / `post.max_steps`。
    """
    from .config import get
    post = cfg.get("post", {}) or {}
    method = (post.get("method") or "").strip().lower()
    if method in ("sft", "dpo", "lora"):
        return (int(get(cfg, "post", "batch_size", 16) or 16), 1,
                int(get(cfg, "post", "max_steps", 2000) or 2000),
                int(get(cfg, "post", "max_seq_len", 512) or 512), True)
    return (int(get(cfg, "train", "batch_size", 32, required=True)),
            int(get(cfg, "train", "grad_accum", 1)),
            int(get(cfg, "train", "max_steps", 5000, required=True)),
            int(mc["block_size"]), False)


def cmd_plan(args):
    utils.set_json_mode(getattr(args, "json", False))
    if not _require_deps("plan", need_torch=True, need_numpy=False):
        return 6
    from .config import get
    from .model import config_from_section
    cfg = load_cfg(args.config, args.set)
    tok = _load_tok_for_config(cfg)
    if tok is None:
        return 1
    mc = config_from_section(cfg.get("model", {}), tok.vocab_size)
    total_real = _print_param_report(mc, tok)

    bs, ga, ms, blk, is_post = _plan_scope(cfg, mc)

    utils.title("显存与耗时预估")
    utils.info("规模口径      : %s"
               % ("后训练 [post]（batch %d × 最长 %d token × %d 步）"
                  % (bs, blk, ms) if is_post else
                  "预训练 [train]（batch %d × %d token × %d 步，梯度累积 %d）"
                  % (bs, blk, ms, ga)))
    # ★ 显存估算统一走 backends.plan —— 全项目只有这一份估算器。
    # 以前这里是自己手写的一行公式（params*16 + 激活），和 validate.py 里
    # 那份各算各的，迟早会给出两个互相矛盾的数字。
    from .backends import check_available_vram, estimate, fmt_bytes, suggestions
    free = check_available_vram()
    est = estimate(
        total_real, batch=bs, seq=blk, d_model=int(mc["d_model"]),
        n_layer=int(mc["n_layer"]),
        world_size=int(get(cfg, "parallel", "tensor_parallel", 1) or 1) *
                   int(get(cfg, "parallel", "pipeline_parallel", 1) or 1),
        zero_stage=int(get(cfg, "parallel", "zero_stage", 0) or 0),
        cpu_offload=bool(get(cfg, "parallel", "cpu_offload", False)),
        activation_checkpoint=bool(get(cfg, "parallel",
                                           "activation_checkpoint", False)),
        moe=bool(mc.get("use_moe")),
        per_card_bytes=free or None)

    utils.info("参数与优化器  : %s（每参数 %.0f 字节）"
               % (fmt_bytes(est["state_bytes"]), est["bytes_per_param"]))
    utils.info("激活值        : %s（批 %d × %d token × %d 层）"
               % (fmt_bytes(est["activation_bytes"]), bs, blk, mc["n_layer"]))
    utils.info("合计估算      : %s" % fmt_bytes(est["total_bytes"]))
    if free:
        utils.info("当前可用显存  : %s" % fmt_bytes(free))
        # ★ 规模提醒统一走 validate.scale_notices（审计 B-05：那段「规模感知」
        #   以前只在 GUI 里、而且唯一调用方还传 scale_notice=False → 死代码；
        #   它自己另算的公式还只算权重+优化器，和这里的完整明细差好几倍）。
        #   现在只做「翻译」，公式仍然只有 backends.plan 一份。
        from .config.validate import scale_notices
        for line in scale_notices(est):
            if line.startswith("⚠"):
                utils.warn(line)
            else:
                utils.info(line)
        if est.get("cards_needed", 1) > 1:
            print()
            utils.info("按省显存效果排序的调整建议：")
            # 一律关键字传参：这几个参数都是可选整数，按位置传极易搞反
            for t in suggestions(cfg.get("parallel", {}), est, total_real,
                                 active=None, n_layer=int(mc["n_layer"]),
                                 d_model=int(mc["d_model"])):
                utils.info("  " + t)

    tps = bs * ga * blk
    utils.info("每步 token 数 : %s" % "{:,}".format(tps))
    utils.info("总 token 数   : %s" % "{:,}".format(tps * ms))
    utils.info("耗时参考（按实测吞吐推算）：")
    for speed in (30000, 60000, 100000):
        utils.info("  若 %6s token/秒 → 约 %s"
                   % ("{:,}".format(speed), utils.fmt_duration(tps * ms / speed)))
    print()
    if est.get("cards_needed", 1) > 1:
        utils.warn("先解决显存问题再训练。下一步:  lingyu launch --config %s"
                   % args.config)
    tps = bs * ga * blk
    utils.emit_event(
        "plan", task="plan", config=args.config, params=total_real,
        scope="post" if is_post else "pretrain", batch=bs, seq=blk,
        max_steps=ms, tokens_per_step=tps, total_tokens=tps * ms,
        total_bytes=est["total_bytes"],
        usable=est.get("usable_per_card"),
        cards_needed=est.get("cards_needed", 1),
        fits=(est.get("cards_needed", 1) <= 1))
    if True:
        ratio = 100.0 * est["total_bytes"] / max(
            1.0, est.get("usable_per_card") or 1.0)
        if ratio >= 85:
            utils.warn("先跑几十步确认不会 OOM，再正式训练。"
                       "下一步:  python -m lingyu train --config %s"
                       % args.config)
        else:
            utils.ok("下一步:  python -m lingyu train --config %s" % args.config)
    return 0


# ---------------------------------------------------------------- train
def cmd_train(args):
    if not _require_deps("train", need_torch=True, need_numpy=True):
        return 6
    from .train import train
    from .config import get
    utils.set_json_mode(getattr(args, "json", False))
    cfg = load_cfg(args.config, args.set)
    resume = args.resume
    if resume == "auto":
        resume = os.path.join(get(cfg, "train", "out_dir", "out"), "ckpt.pt")
        if not os.path.isfile(resume):
            utils.err("配置里 out_dir 下没有 ckpt.pt，没法续训: %s" % resume)
            return 1
    try:
        train(cfg, device=args.device, resume=resume,
              force=getattr(args, "force", False),
              init_from=getattr(args, "init_from", None),
              config_path=args.config)
    except BaseException as exc:      # noqa: BLE001 - 结构化错误事件后继续抛
        utils.emit_event("error", task="train",
                         message=str(exc) or type(exc).__name__)
        raise
    return 0


# ---------------------------------------------------------------- sample
def cmd_sample(args):
    if not _require_deps("sample", need_torch=True, need_numpy=True):
        return 6
    from .sample import load_model, generate_text
    from .config import get
    cfg = load_cfg(args.config, args.set) if args.config else {}
    ckpt = args.ckpt
    if not ckpt:
        out_dir = get(cfg, "train", "out_dir", "out")
        for cand in ("best.pt", "ckpt.pt"):
            p = os.path.join(out_dir, cand)
            if os.path.isfile(p):
                ckpt = p
                break
    if not ckpt or not os.path.isfile(ckpt):
        utils.err("找不到断点文件: %s" % ckpt)
        utils.err("先训练，或用 --ckpt 指定路径")
        return 1
    device = args.device or ("cuda" if _cuda_ok() else "cpu")
    tok_path = get(cfg, "data", "tokenizer_path", None) if cfg else None
    model, tok, ck = load_model(ckpt, device=device, tokenizer_path=tok_path)
    utils.info("断点   : %s（第 %s 步）" % (ckpt, "{:,}".format(ck.get("step", 0))))
    utils.info("分词器 : 词表 %s" % "{:,}".format(tok.vocab_size))
    scfg = cfg.get("sample", {}) if cfg else {}

    def pick(v, key, default):
        return v if v is not None else scfg.get(key, default)

    prompt = pick(args.prompt, "prompt", "他")
    utils.title("生成")
    generate_text(
        model, tok, prompt=prompt,
        max_new_tokens=int(pick(args.max_new_tokens, "max_new_tokens", 200)),
        temperature=float(pick(args.temperature, "temperature", 0.9)),
        top_k=int(pick(args.top_k, "top_k", 40)),
        top_p=float(pick(args.top_p, "top_p", 0.95)),
        repetition_penalty=float(pick(args.repetition_penalty, "repetition_penalty", 1.05)),
        seed=args.seed, device=device, stream=not args.no_stream)
    return 0


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------- sft
def cmd_sft(args):
    """监督微调。

    四个命令行参数都是「覆盖配置」的快捷方式 —— 常见用法不该逼用户去改文件。
    """
    if not _require_deps("sft", need_torch=True, need_numpy=True):
        return 6
    cfg = load_cfg(args.config, args.set)
    post = cfg.setdefault("post", {})
    if args.data:
        post["data_path"] = args.data
    if args.template:
        post["template"] = args.template
    if args.init_from:
        post["init_from"] = args.init_from
    if args.out_dir:
        post["out_dir"] = args.out_dir
    post["method"] = "sft"          # 走到这个子命令就是要做 SFT，别再问
    # method/数据路径是刚改的，必须用改完的配置重新校验一次（--set 之后已校验过）
    _validate_cfg(cfg, "SFT 配置")
    from .sft import sft
    utils.set_json_mode(getattr(args, "json", False))
    try:
        sft(cfg, device=args.device, force=args.force, resume=args.resume,
            config_path=args.config)
    except BaseException as exc:      # noqa: BLE001
        utils.emit_event("error", task="sft",
                         message=str(exc) or type(exc).__name__)
        raise
    return 0


# ---------------------------------------------------------------- chat
def cmd_chat(args):
    if not _require_deps("chat", need_torch=True, need_numpy=True):
        return 6
    utils.set_json_mode(getattr(args, "json", False))
    cfg = load_cfg(args.config, args.set) if args.config else {}
    from .chat import run_chat
    return run_chat(cfg, ckpt=args.ckpt, device=args.device, prompt=args.prompt,
                    stream=not args.no_stream,
                    json_events=bool(getattr(args, "json", False)))


# ---------------------------------------------------------------- launch
def cmd_launch(args):
    utils.set_json_mode(getattr(args, "json", False))
    """生成分布式启动脚本。

    这个命令存在的意义：**本机跑不动的规模，得给用户一条真能走的路**。
    光说「太大了」是没用的 —— 那是把问题丢回给用户。
    """
    # 这个命令会建模型来算参数量，所以要 torch。
    # 少了这一句，缺 torch 时会吐一屏 ModuleNotFoundError 堆栈。
    if not _require_deps("launch", need_torch=True, need_numpy=False):
        return 6
    from .config import get
    from .backends import (backend_info, backend_is_implemented,
                           check_available_vram, estimate, fmt_bytes, generate,
                           list_backends, suggestions, write_outputs)
    from .model import build_model, config_from_section
    cfg = load_cfg(args.config, args.set)
    cfg["_path"] = args.config          # 脚本里要写清是从哪份配置生成的

    backend = (args.backend or get(cfg, "parallel", "backend", "local")
               or "local").strip().lower()
    known = [b["name"] for b in list_backends()]
    if backend not in known:
        # 以前这里直接 backend_info(backend) -> KeyError 一屏堆栈，
        # 用户看到的是一堆 pin 不了的 Python 内部帧，不知道哪个参数写错了。
        utils.err("不认识的后端：%r" % backend)
        utils.err("可选的有：%s" % "、".join(known))
        utils.err("本引擎真正实现训练循环的只有 local；"
                  "fsdp / deepspeed / megatron 目前只生成脚本。")
        return 2
    info = backend_info(backend)

    if backend == "fsdp":
        print()
        utils.warn("fsdp 后端目前是**实验性**的：引擎只会生成 torchrun 脚本，"
                   "训练代码里还没有分布式（FSDP2/DDP）实现。")
        utils.warn("脚本跑起来是多个进程各自做单卡训练（抢同一张卡/同一份 "
                   "out_dir），不要直接用于生产。")
        utils.warn("真正能直接跑的是 local（单卡）；多卡训练等后续版本。")

    tok = _load_tok_for_config(cfg)
    if tok is None:
        return 1
    mc = config_from_section(cfg.get("model", {}), tok.vocab_size)
    model = build_model(mc)
    params = sum(p.numel() for p in model.parameters())
    active = (model.active_param_count() if hasattr(model, "active_param_count")
              else params)

    # 与 plan 同一套口径（B-17）：后训练配置按 [post]，预训练按 [train]
    bs, _ga, _ms, seq, is_post = _plan_scope(cfg, mc)
    est = estimate(
        params,
        batch=bs,
        seq=seq,
        d_model=int(mc["d_model"]),
        n_layer=int(mc["n_layer"]),
        world_size=int(get(cfg, "parallel", "tensor_parallel", 1) or 1) *
                   int(get(cfg, "parallel", "pipeline_parallel", 1) or 1),
        zero_stage=int(get(cfg, "parallel", "zero_stage", 2) or 0),
        cpu_offload=bool(get(cfg, "parallel", "cpu_offload", False)),
        activation_checkpoint=bool(get(cfg, "parallel",
                                           "activation_checkpoint", False)),
        moe=bool(mc.get("use_moe")),
        # 传本机显存才能算出「需要几张卡」；没有 CUDA 就只给每卡估算
        per_card_bytes=check_available_vram() or None)

    utils.title("规模评估")
    utils.info("模型参数      : %s（%.1f M）"
               % ("{:,}".format(params), params / 1e6))
    utils.info("目标后端      : %s —— %s" % (info["label"], info["summary"]))
    utils.info("规模口径      : %s"
               % ("后训练 [post]（batch %d × 最长 %d token）" % (bs, seq)
                  if is_post else
                  "预训练 [train]（batch %d × %d token）" % (bs, seq)))
    utils.info("每卡显存估算  : %s（每参数 %.0f 字节）"
               % (fmt_bytes(est["total_bytes"]), est["bytes_per_param"]))
    utils.info("  优化器+权重 : %s" % fmt_bytes(est["state_bytes"]))
    utils.info("  激活值      : %s" % fmt_bytes(est["activation_bytes"]))
    if est.get("usable_per_card"):
        from .config.validate import scale_notices
        for line in scale_notices(est):
            if line.startswith("⚠"):
                utils.warn(line)
            else:
                utils.info(line)

    if not backend_is_implemented(backend):
        print()
        utils.warn("这个后端**不是本引擎实现的**，生成的是给外部框架用的脚本：")
        utils.info("  引擎只负责把你的配置翻译成对方需要的参数。")
        utils.info("  真正的训练由 %s 官方实现执行 —— 这是刻意的，"
                   "自己重写一个半成品对你没有好处。" % info["label"])
        utils.info("  脚本里所有 [TODO] 都是需要你人工确认的地方。")

    result = generate(backend, cfg, est, params, active)

    if args.print_only:
        print()
        print(result["script"])
        return 0

    written = write_outputs(args.out_dir, backend, result)
    utils.emit_event("launch", task="launch", backend=backend,
                     implemented=backend_is_implemented(backend),
                     files=[os.path.abspath(p) for p in written])
    print()
    utils.ok("已生成：")
    for p in written:
        utils.info("  %s" % os.path.abspath(p))
    print()
    utils.info("怎么用：")
    if backend in ("local", "fsdp"):
        utils.info("  1. 打开脚本，把标了 [TODO] 的地方改成你的机器")
        utils.info("  2. 每台机器都要装好同版本的 PyTorch 和这个项目")
        utils.info("  3. 执行脚本")
    else:
        utils.info("  1. 打开脚本，**先把开头那段前置条件读完**")
        utils.info("  2. 把每个 [TODO] 处理掉（多半要自己写几十行胶水代码）")
        utils.info("  3. 执行脚本")

    if est.get("cards_needed", 1) > 1:
        tips = suggestions(cfg.get("parallel", {}), est, params, active=active,
                           n_layer=int(mc["n_layer"]),
                           d_model=int(mc["d_model"]))
        if tips:
            print()
            utils.info("如果这个规模对你来说太大，按省显存效果排序：")
            for t in tips:
                utils.info("  " + t)
    return 0


# ---------------------------------------------------------------- gui
def cmd_gui(args):
    """打开图形界面。

    GUI 装在项目根目录的 `gui/` 包里（不在 `src/lingyu/` 下）——
    因为它是**可选依赖**：服务器上跑训练的人不需要装 PySide6。
    所以这里要把项目根加进 sys.path 才能 import 到它。
    """
    root = _project_root()
    if not _require_deps("gui", need_torch=False, need_numpy=False,
                         need_gui=True):
        return 6
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import gui
    except ImportError as e:
        utils.err("找不到 gui 包（应该在 %s 下）：%s" % (root, e))
        return 6
    if args.config:
        # 把「启动时用哪个配置」记进 GUI 的设置，界面起来后就用它
        try:
            from gui.context import ctx
            ctx.settings["config"] = args.config
            ctx.save()
        except Exception:
            pass
    return gui.run()


CMDS = {
    "doctor": cmd_doctor, "scan": cmd_scan, "prepare": cmd_prepare,
    "params": cmd_params, "plan": cmd_plan, "train": cmd_train,
    "sample": cmd_sample, "sft": cmd_sft, "chat": cmd_chat,
    "launch": cmd_launch, "gui": cmd_gui,
}


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = build_parser()
    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 0
    try:
        return CMDS[args.cmd](args) or 0
    except ConfigInvalid as e:
        # 配置错误已经在入口逐条打过了（[错误] ...），这里只给结论和退出码。
        utils.err(str(e))
        return 2
    except KeyboardInterrupt:
        print()
        utils.warn("已中断。")
        return 130
    except FileNotFoundError as e:
        utils.err(str(e))
        return 2
    except FileExistsError:
        # 覆盖保护触发：提示信息已经在 train() 里打过了，这里只需安静退出
        return 5
    except (ValueError, KeyError) as e:
        utils.err(str(e))
        return 3
    except BrokenPipeError:
        # 输出被管道提前关掉（例如 `... | Select-Object -First 5`）。
        # 这时训练/存档其实已经完成了，不该吐一堆 traceback 吓人。
        # 把 stdout 换成 devnull，避免解释器退出时再报一次。
        try:
            import os as _os
            _os.dup2(_os.open(_os.devnull, _os.O_WRONLY), 1)
        except OSError:
            pass
        return 0
    except OSError as e:
        utils.err("系统/文件错误：%s" % e)
        return 4
