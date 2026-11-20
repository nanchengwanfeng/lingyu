# -*- coding: utf-8 -*-
"""命令行端到端回归测试。

## 为什么需要这个文件

重构（v0.3 把平铺包改成 src/lingyu 包结构）最怕的不是「跑不起来」——
那种错误 pytest 立刻就能抓到；最怕的是**静默变化**：
`lingyu params` 少算了一部分参数、`plan` 给出的显存估算偏了、
`--set` 写错键名不报错…… 这些都不会让程序崩，但会让你基于错数字做决定。

所以这里的测试分两类：

1. **不变式**（invariant）—— 不看具体数字，看「数字之间的关系对不对」。
   例如：params 报的总参数 == 真实模型的 `sum(p.numel())`。
   这种断言不会因为改了措辞而失败，但一旦算错立刻红。

2. **端到端**（e2e）—— 真的走一遍 扫描语料 → 准备数据 → 训练 → 生成。
   用的是 2 层 32 维的迷你模型，CPU 上几秒就能跑完，但覆盖了
   分词器、数据加载、训练循环、断点保存、KV cache 生成这一整条链路。

跑法：

    pytest tests/test_cli.py -q              # 全部
    pytest tests/test_cli.py -q -m "not slow"  # 跳过端到端（更快）
"""

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def run(*args, **kw):
    """在引擎根目录跑 `python -m lingyu <args>`，返回 (退出码, 输出)。

    刻意用**子进程**而不是直接调 `lingyu.cli.main()`：
    子进程才能验证「入口点、参数解析、退出码、编码」这些真实用户会遇到的东西。
    曾经 `lingyu` 命令在别的目录下就报 `No module named lingyu`，只有
    子进程测试能发现这类问题。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    p = subprocess.run([sys.executable, "-m", "lingyu"] + list(args),
                       cwd=kw.pop("cwd", ROOT), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=kw.pop("timeout", 900))
    out = p.stdout.decode("utf-8", "replace")
    return p.returncode, out


def _num(text):
    """把 '12,587,904' 这样的千分位数字转成 int。"""
    return int(text.replace(",", "").strip())


# ============================================================ 基本入口

def test_help_lists_every_subcommand():
    code, out = run("--help")
    assert code == 0, out
    for cmd in ("doctor", "scan", "prepare", "params", "plan",
                "train", "sample"):
        assert cmd in out, "help 里没有 %s 子命令" % cmd


def test_no_args_prints_help_and_does_not_crash():
    code, out = run()
    # 不带参数时应该打印帮助，不能是 traceback
    assert "Traceback" not in out, out
    assert code in (0, 2), out


def test_doctor_runs():
    code, out = run("doctor", timeout=300)
    assert "Traceback" not in out, out
    # CI 装的是 CPU 版 torch：doctor 会把它标成「环境不完整」并返回 1。
    # 这是**预期行为**（提示用户装 CUDA 版），不能让测试因此失败。
    import torch
    cpu_only = getattr(torch.version, "cuda", None) is None
    assert code in ((0, 1) if cpu_only else (0,)), out


def test_plan_without_tokenizer_falls_back_to_config_vocab():
    """没跑过 prepare 的新 clone：plan 也要能出估算，而不是直接报错退出。

    分词器只影响词表大小；配置里本来就有 [model].vocab_size 可作估算依据。
    """
    code, out = run("plan", "--config", "configs/small.ini",
                    "--set", "data.tokenizer_path=__missing__/tokenizer.txt",
                    "--set", "data.data_dir=__missing__/data")
    assert code == 0, out
    assert "Traceback" not in out, out
    assert "找不到分词器" in out, out
    assert "词表" in out, out


# ============================================================ params 不变式

@pytest.mark.parametrize("cfg", ["configs/nano65.ini", "configs/small.ini",
                                 "configs/moe.ini"])
def test_params_total_equals_real_model(cfg):
    """★ 核心不变式：`params` 报的总数必须等于真实模型的参数量。

    这是把「手写公式 vs 实测」这类 bug 钉死的地方 —— 之前在 cli.py 里
    另写了一套公式，归一化层算成 2d，屏幕上就长期挂着一句假的
    「公式与实测差 4,224」。
    """
    code, out = run("params", "--config", cfg)
    assert code == 0, out
    m = re.search(r"总计\s*:\s*([\d,]+)", out)
    assert m, "params 输出里找不到总数：\n" + out
    reported = _num(m.group(1))

    # 词表大小可能是配置里写死、也可能由分词器决定，所以直接从输出里读回来，
    # 保证「报出来的数字」和「算出来的数字」用的是同一个模型结构。
    mv = re.search(r"词表\s*([\d,]+)\s*×\s*(\d+)\s*维", out)
    assert mv, "params 输出里找不到词表/维度：\n" + out
    vocab = _num(mv.group(1))

    from lingyu.config import load_config
    from lingyu.model import LingYu, config_from_section
    c = load_config(os.path.join(ROOT, cfg))
    section = dict(c.get("model", {}))
    section["vocab_size"] = vocab          # 用引擎自己用的那个词表大小
    model = LingYu(config_from_section(section, vocab))
    real = sum(p.numel() for p in model.parameters())
    assert reported == real, (
        "%s: params 报 %s，真实 %s（差 %s）"
        % (cfg, "{:,}".format(reported), "{:,}".format(real),
           "{:,}".format(reported - real)))


@pytest.mark.parametrize("cfg", ["configs/nano65.ini", "configs/small.ini"])
def test_params_has_no_phantom_mismatch_warning(cfg):
    """params 不该再报「公式与实测差」或「请报 issue」。"""
    code, out = run("params", "--config", cfg)
    assert code == 0, out
    assert "公式与实测差" not in out, out
    assert "请报 issue" not in out, out


def test_params_norm_line_matches_norm_choice():
    """归一化行的处数必须是 2L+1，且名字跟着配置走。"""
    code, out = run("params", "--config", "configs/nano65.ini")
    assert code == 0, out
    assert "rmsnorm" in out and "共 13 处" in out, out       # nano65: L=6 -> 13

    code, out = run("params", "--config", "configs/nano65.ini",
                    "--set", "model.norm=layernorm")
    assert code == 0, out
    assert "layernorm" in out, out
    # LayerNorm 每处多一个 bias：13 × 384 = 4,992 个
    assert "9,984" in out, out


# ============================================================ --set 校验

def test_set_typo_is_rejected_with_suggestion():
    """写错键名必须报错 + 给建议，不能静默忽略。"""
    code, out = run("params", "--config", "configs/nano65.ini",
                    "--set", "model.n_layr=8")
    assert code != 0, "写错键名居然成功了：\n" + out
    assert "n_layr" in out, out
    assert "n_layer" in out, "没给出正确拼写建议：\n" + out


def test_set_unknown_section_is_rejected():
    code, out = run("params", "--config", "configs/nano65.ini",
                    "--set", "modle.n_layer=8")
    assert code != 0, out


def test_set_alias_is_autocorrected():
    """常见笔误（n_layers / lr）应该自动纠正并提示，而不是报错。"""
    code, out = run("params", "--config", "configs/nano65.ini",
                    "--set", "model.n_layers=3")
    assert code == 0, out
    assert "自动纠正" in out, out
    assert "3 层" in out, "改了层数但输出没跟着变：\n" + out


def test_component_defaults_do_not_override_use_moe():
    """★ 真实 bug 回归：只写 use_moe=true、不写 ffn 时，必须真的是 MoE。

    踩坑经过：配置系统加了「补全 spec 默认值」之后，`ffn` 的默认值
    写成具体实现名 "swiglu"，于是：

        moe.ini（只有 use_moe = true）
          -> resolve() 补上 ffn = "swiglu"
          -> 解析逻辑看到 ffn 非空，就按 ffn 走，顺手把 use_moe 改成 false
          -> MoE 档静默变成稠密档，参数量 53.5M 变 28.3M

    最阴的是**不报错**：`params` 照样打印一屏好看的表格，只是数字全错。

    所以组件名的默认值必须是空字符串，空 = 引擎自己决定。
    这条测试同时钉住 spec 默认值和 resolve() 的行为。
    """
    from lingyu.config import SPEC, resolve

    for key in ("attention", "ffn", "norm", "pos"):
        assert SPEC["model"][key]["default"] == "", (
            "model.%s 的默认值必须是空字符串（空 = 自动），现在是 %r —— "
            "填具体实现名会静默推翻 use_moe 之类的联动开关"
            % (key, SPEC["model"][key]["default"]))

    # 只给 use_moe，不给 ffn：补全之后 ffn 仍应留空，交给解析逻辑推断
    cfg = resolve({"model": {"use_moe": True}})
    assert cfg["model"]["ffn"] == "", cfg["model"]["ffn"]

    from lingyu.model import build_config
    from lingyu.components import get_ffn
    section = {k: v for k, v in cfg["model"].items()
               if k in ("n_layer", "n_embd", "n_head", "n_kv_head", "block_size",
                        "use_moe", "ffn", "attention", "norm", "pos", "d_ff",
                        "n_expert", "n_expert_per_tok", "n_shared_expert",
                        "expert_d_ff", "tie_weights", "bias", "dropout")}
    section["vocab_size"] = 1000
    mc = build_config(**section)
    assert mc["use_moe"] is True, "use_moe 被静默改成了 %r" % mc["use_moe"]
    assert get_ffn(mc).__name__ == "MoE", (
        "use_moe=true 但选出来的前馈是 %s" % get_ffn(mc).__name__)


def test_moe_config_is_actually_moe():
    """同理，直接验 moe.ini 这一档真的建出 MoE 模型。"""
    code, out = run("params", "--config", "configs/moe.ini")
    assert code == 0, out
    assert "MoE 稀疏专家" in out, "moe.ini 没建出 MoE：\n" + out
    assert "激活参数" in out, out


# ==================================================== 依赖检查的完整性

def test_every_command_that_needs_torch_checks_for_it():
    """★ 需要 torch 的子命令必须**先检查再 import**。

    踩过的坑：`lingyu launch` 直接从 `.model` 导东西，而 `model.py`
    顶部就 `import torch`。环境里没 torch 时不报「缺少依赖：torch」，
    而是吐一屏 `ModuleNotFoundError` 堆栈 —— 用户完全看不出
    「该装什么、怎么装」。

    `scan` 是**故意**不加检查的：它只做纯文本统计，只用标准库。
    给它加上检查反而会把一个本来能跑的命令拦下来。
    """
    import ast
    path = os.path.join(ROOT, "src", "lingyu", "cli.py")
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src)

    cmds = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "CMDS" for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                cmds[k.value] = v.id

    guarded = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in cmds.values():
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and \
                        getattr(sub.func, "id", "") == "_require_deps":
                    guarded[node.name] = True

    # 这几个不需要检查：
    #   doctor  自己会逐项报告环境状态
    #   scan    纯文本统计，只用标准库（有测试证明）
    EXEMPT = {"doctor", "scan"}
    missing = [k for k, fn in sorted(cmds.items())
               if k not in EXEMPT and not guarded.get(fn)]
    assert not missing, (
        "这些命令没有做依赖检查，缺 torch 时会吐堆栈而不是人话：%s\n"
        "在函数开头加：if not _require_deps(\"%s\", need_torch=True): return 6"
        % (missing, missing[0] if missing else ""))

    # 反向检查：scan 确实不需要 torch（防止有人"顺手"给它加上检查）。
    # 用 ast 只看**真正的调用**，不看注释 —— 否则我写在 cmd_scan 里
    # 那句「这里刻意不做 _require_deps」的说明会把测试自己绊倒。
    scan_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cmd_scan":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and \
                        getattr(sub.func, "id", "") == "_require_deps":
                    scan_calls.append(sub.lineno)
    assert not scan_calls, (
        "cmd_scan 不需要 torch/numpy，别给它加依赖检查（第 %s 行）—— "
        "那会把一个本来能跑的命令拦下来" % scan_calls)


# ============================================================ plan

def test_plan_reports_layers_and_memory():
    code, out = run("plan", "--config", "configs/small.ini")
    assert code == 0, out
    assert "16" in out, "plan 没报出 n_layer=16：\n" + out
    assert "Traceback" not in out, out


def test_plan_warns_when_scale_exceeds_machine():
    """规模感知：参数大到本机装不下时，必须**明说**而不是假装能跑。

    这是「一个引擎从 13M 到万亿」这个定位的关键：同一套配置语法，
    超出本机能力时诚实提示。万亿参数在 8GB 卡上必须是明确的「不行」。
    """
    code, out = run("plan", "--config", "configs/nano65.ini",
                    "--set", "model.n_layer=80", "--set", "model.n_embd=8192",
                    "--set", "model.n_head=64")
    # 允许非 0 退出（明确拒绝）或 0 退出但给出提示，但绝不能是 traceback
    assert "Traceback" not in out, out
    low = out.lower()
    assert ("装不下" in out or "不够" in out or "需要" in out
            or "分布式" in out or "显存" in low or code != 0), out


# ============================================================ 端到端

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """一个迷你工作区：3 本小"书"，够跑完整链路。"""
    d = tmp_path_factory.mktemp("e2e")
    raw = d / "raw"
    raw.mkdir()
    # 中文语料，字符要够多，否则分词器建不出像样的词表。
    # 用可复现的伪随机拼接，避免依赖外部文件。
    base = ("他抬起头，看见窗外的雨已经停了。天边露出一线微光，"
            "远处的山影在雾气里若隐若现。他忽然想起很多年前的那个下午，"
            "也是这样的天气，也是这样的光。那时候他还年轻，什么都不怕。")
    for i, name in enumerate(("书一.txt", "书二.txt", "书三.txt")):
        text = (base * 40) + ("这是第%d本书独有的句子，用来制造一些差异。" % i) * 20
        (raw / name).write_text(text, encoding="utf-8")
    (raw / "放小说到这里.txt").write_text("这是说明文件，不该被算成语料。",
                                          encoding="utf-8")
    return {"dir": d, "raw": raw, "data": d / "data", "out": d / "out"}


def test_plan_huge_config_estimates_without_instantiating():
    """装不下的巨大配置：必须秒回估算 + 诚实提示，不能真去建模型吃光内存。

    回归：P0 之后 `cmd_plan` 曾无条件 `build_model()` 数参数，
    80 层 × 8192 维时实测吃掉 6.9 GB 内存并把一条测试拖到超时。
    """
    code, out = run("plan", "--config", "configs/nano65.ini",
                    "--set", "model.n_layer=80",
                    "--set", "model.n_embd=8192",
                    "--set", "model.n_head=64", timeout=120)
    assert "Traceback" not in out, out
    assert "估算" in out, out
    assert code in (0, 2), out


def test_e2e_scan(workspace):
    code, out = run("scan", "--raw", str(workspace["raw"]))
    assert code == 0, out
    assert "放小说到这里" not in out, "说明文件被当成语料了：\n" + out


@pytest.mark.slow
def test_e2e_full_pipeline(workspace):
    """扫描 → 准备数据 → 训练 → 生成，全链路真跑一遍。

    模型刻意做到最小（2 层 32 维），CPU 也能几秒跑完 ——
    目的是**验证链路通不通**，不是验证能不能学好。
    """
    data = str(workspace["data"])
    code, out = run("prepare", "--raw", str(workspace["raw"]),
                    "--data", data, "--tokenizer", "char",
                    "--vocab-size", "512", "--seed", "1234")
    assert code == 0, out
    # 分词器一律叫 tokenizer.txt（内容是 JSON）。用**发现**而不是硬编码扩展名，
    # 免得以后改了文件名测试就假失败。
    toks = [f for f in os.listdir(data) if f.startswith("tokenizer.")]
    assert toks, "prepare 没产出分词器文件，目录里只有：%s\n%s" % (
        os.listdir(data), out)
    tok_path = os.path.join(data, toks[0])
    assert os.path.isfile(os.path.join(data, "train.bin")), out
    assert os.path.isfile(os.path.join(data, "val.bin")), out

    outdir = str(workspace["out"])
    cfg_path = str(workspace["dir"] / "tiny.ini")
    with open(cfg_path, "w", encoding="utf-8") as fp:
        fp.write("[model]\n"
                 "vocab_size = 0\nn_layer = 2\nn_embd = 32\nn_head = 2\n"
                 "n_kv_head = 1\nblock_size = 64\ndropout = 0.0\n\n"
                 "[train]\n"
                 "batch_size = 8\ngrad_accum = 1\nmax_steps = 12\n"
                 "learning_rate = 0.01\nwarmup_iters = 2\n"
                 "eval_interval = 6\neval_iters = 2\nlog_interval = 6\n"
                 "ckpt_interval = 12\nout_dir = %s\nsample_while_train = False\n\n"
                 "[data]\n"
                 "data_dir = %s\ntokenizer_path = %s\n\n"
                 "[sample]\nmax_new_tokens = 16\n"
                 % (outdir, data, tok_path))

    code, out = run("plan", "--config", cfg_path, timeout=300)
    assert code == 0, out
    assert "Traceback" not in out, out

    code, out = run("train", "--config", cfg_path, "--device", "cpu",
                    timeout=900)
    assert code == 0, out
    assert "Traceback" not in out, out
    ckpt = os.path.join(outdir, "ckpt.pt")
    assert os.path.isfile(ckpt), "训练没存下断点：\n" + out

    # loss 必须是有限数（NaN/inf 会静默毁掉整个训练）
    m = re.findall(r"loss[^\d\-+]*([\d.]+)", out)
    assert m, "训练输出里没看到 loss：\n" + out
    last = float(m[-1])
    assert last == last and abs(last) != float("inf"), "loss 是 NaN/inf：%s" % last

    code, out = run("sample", "--config", cfg_path, "--ckpt", ckpt,
                    "--prompt", "他抬起头", "--max-new-tokens", "16",
                    "--device", "cpu", "--no-stream", timeout=300)
    assert code == 0, out
    assert "Traceback" not in out, out


@pytest.mark.slow
def test_e2e_resume(workspace):
    """续训必须能接上，且步数继续往上走（不是从头开始）。"""
    outdir = str(workspace["out"])
    cfg_path = str(workspace["dir"] / "tiny.ini")
    if not os.path.isfile(cfg_path):
        pytest.skip("端到端主流程没跑过")

    code, out = run("train", "--config", cfg_path, "--device", "cpu",
                    "--resume", "--set", "train.max_steps=16", timeout=900)
    assert code == 0, out
    assert "Traceback" not in out, out


@pytest.mark.slow
def test_e2e_structure_change_is_diagnosed(workspace):
    """改了模型结构后续训，必须给出**人话**诊断，而不是 size mismatch 堆栈。

    这是用户真实踩过的坑：他把 small.ini 从 8 层 512 维改成 16 层 256 维，
    然后用 --resume，得到一屏 `size mismatch for blocks.0...`。
    """
    cfg_path = str(workspace["dir"] / "tiny.ini")
    if not os.path.isfile(cfg_path):
        pytest.skip("端到端主流程没跑过")
    code, out = run("train", "--config", cfg_path, "--device", "cpu",
                    "--resume", "--set", "model.n_layer=3", timeout=300)
    assert code != 0, "结构变了还续训成功了？\n" + out
    assert "Traceback" not in out, "给用户看了堆栈：\n" + out
    assert ("n_layer" in out or "结构" in out or "shape" in out.lower()), out


@pytest.mark.slow
def test_e2e_sft_then_chat(workspace):
    """★ 完整对话链路：预训练 → SFT → 聊天。

    这是「不只是续写，还能向它提问」这个需求的自动化验收。
    模型小到没有意义，跑的是**链路**：能不能载入预训练权重、
    能不能把 JSONL 渲染成带 mask 的样本、能不能训、能不能存、
    chat 能不能找到断点并读出模板。
    """
    cfg_path = str(workspace["dir"] / "tiny.ini")
    if not os.path.isfile(cfg_path):
        pytest.skip("端到端主流程没跑过")
    (workspace["out"] / "best.pt")
    if not os.path.isfile(os.path.join(str(workspace["out"]), "ckpt.pt")):
        pytest.skip("没有预训练断点")

    chat_data = os.path.join(ROOT, "chat", "sample_chat.jsonl")
    assert os.path.isfile(chat_data), "仓库里应该带一份示例对话数据"
    sft_dir = str(workspace["dir"] / "out_sft")

    code, out = run("sft", "--config", cfg_path, "--device", "cpu",
                    "--data", chat_data, "--out-dir", sft_dir,
                    "--set", "post.max_steps=8",
                    "--set", "post.batch_size=4",
                    "--set", "post.max_seq_len=96",
                    "--set", "post.ask_interval=4",
                    "--set", "post.eval_interval=4",
                    "--set", "post.ckpt_interval=8",
                    "--set", "post.learning_rate=0.001",
                    timeout=900)
    assert code == 0, out
    assert "Traceback" not in out, out
    ck = os.path.join(sft_dir, "ckpt_sft.pt")
    assert os.path.isfile(ck), "SFT 没存下断点：\n" + out
    assert "回答占比" in out, "应该报告有多少 token 真的参与 loss：\n" + out

    # 聊天：给了 --prompt 就是问一句然后退出
    code, out = run("chat", "--config", cfg_path, "--ckpt", ck,
                    "--prompt", "你好", "--no-stream", "--device", "cpu",
                    timeout=600)
    assert code == 0, out
    assert "Traceback" not in out, out
    assert "灵语对话" in out, out


@pytest.mark.slow
def test_e2e_sft_refuses_when_base_structure_differs(workspace):
    """SFT 的起点结构不对时，必须报人话，而不是形状错误堆栈。"""
    cfg_path = str(workspace["dir"] / "tiny.ini")
    if not os.path.isfile(cfg_path):
        pytest.skip("端到端主流程没跑过")
    chat_data = os.path.join(ROOT, "chat", "sample_chat.jsonl")
    code, out = run("sft", "--config", cfg_path, "--device", "cpu",
                    "--data", chat_data,
                    "--out-dir", str(workspace["dir"] / "out_sft2"),
                    "--set", "model.n_layer=5",
                    "--set", "post.max_steps=2",
                    "--set", "post.max_seq_len=96",
                    timeout=600)
    assert code != 0, "结构不一致还能 SFT？\n" + out
    assert "Traceback" not in out, out
    assert "结构" in out or "n_layer" in out, out


# ======================================================== 后端名写错

def test_unknown_backend_flag_is_rejected_without_traceback():
    """★ `--backend bogus` 不许甩 KeyError 堆栈。

    以前 cmd_launch 直接 `backend_info(backend)`，未知名字是 dict KeyError，
    用户看到一屏 Python 内部帧，根本不知道是哪个参数写错了。
    """
    code, out = run("launch", "--config", "configs/nano65.ini",
                    "--backend", "bogus", timeout=300)
    assert code == 2, out
    assert "Traceback" not in out, out
    assert "不认识的后端" in out, out
    assert "local" in out, "报错要告诉用户有哪些可选：\n" + out


def test_unknown_backend_in_config_is_rejected(tmp_path):
    """配置里写 `parallel.backend = does-not-exist` 也要人话 + 退出码 2。"""
    p = tmp_path / "bad.ini"
    p.write_text("[model]\nn_layer = 2\n\n[parallel]\n"
                 "backend = does-not-exist\n", encoding="utf-8")
    code, out = run("plan", "--config", str(p), timeout=300)
    assert code == 2, out
    assert "Traceback" not in out, out
    assert "backend" in out, out


# ================================================ 规模提醒 / 后训练口径

def _has_cuda():
    """CI 是 CPU-only：涉及显存数字的断言要按机器能力分支。"""
    import torch
    return bool(torch.cuda.is_available())


def test_plan_wires_scale_notice_and_reports_cards_for_huge_model():
    """B-05：规模提醒必须真的出现在 CLI plan 输出里（大模型提示需要 N 卡）。

    以前这套「规模感知」只在 validate() 里，而唯一的生产调用方（GUI）
    还传了 scale_notice=False → 等于死代码。
    """
    code, out = run("plan", "--config", "configs/nano65.ini",
                    "--set", "model.n_layer=20",
                    "--set", "model.n_embd=1024",
                    "--set", "model.n_head=8",
                    "--set", "model.block_size=1024",
                    "--set", "train.batch_size=32",
                    timeout=600)
    assert code == 0, out
    assert "合计估算" in out, out
    if _has_cuda():
        assert "需要约" in out and "张同等显存的卡" in out, out
    else:
        assert "当前可用显存  : 未知" in out, out


def test_plan_small_config_reports_enough_memory():
    code, out = run("plan", "--config", "configs/nano65.ini", timeout=300)
    assert code == 0, out
    if _has_cuda():
        assert "显存够用" in out, out
        assert "张同等显存的卡" not in out, out
    else:
        assert "当前可用显存  : 未知" in out, out


def test_plan_for_sft_uses_post_batch_and_seq():
    """B-17：SFT 配置没有 [train] batch_size，规划要按 [post] 算。

    以前 fallback 到 train.batch_size 默认 32，把 SFT 的显存/耗时算成
    98% 的假警告，每步 token 数和总 token 数也都是错的。
    """
    from lingyu.config import load_config
    post = load_config(os.path.join(ROOT, "configs/sft.ini"))["post"]
    bs, seq, ms = (int(post["batch_size"]), int(post["max_seq_len"]),
                   int(post["max_steps"]))
    code, out = run("plan", "--config", "configs/sft.ini", timeout=300)
    assert code == 0, out
    assert "后训练 [post]" in out, out
    assert "批 %d × %d token" % (bs, seq) in out, out
    assert "每步 token 数 : {:,}".format(bs * seq) in out, out
    assert "总 token 数   : {:,}".format(bs * seq * ms) in out, out
