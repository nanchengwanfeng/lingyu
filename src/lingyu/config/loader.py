# -*- coding: utf-8 -*-
"""配置加载：INI / JSON / YAML 读写 + 命令行覆盖。

支持三种格式：
    .ini   主推。中文注释友好，记事本能改
    .json  程序生成用
    .yaml  进阶用户（需要 pyyaml，没装会给出明确提示）
"""

import ast
import json
import os
import re
import warnings

from .spec import SPEC, SECTIONS, all_keys

# 从 spec 自动推导类型集合，避免手工维护两份
_INT_KEYS = {k for s in SPEC.values() for k, p in s.items() if p["type"] is int}
_FLOAT_KEYS = {k for s in SPEC.values() for k, p in s.items() if p["type"] is float}
_BOOL_KEYS = {k for s in SPEC.values() for k, p in s.items() if p["type"] is bool}
_STR_KEYS = {k for s in SPEC.values() for k, p in s.items() if p["type"] is str}


def _unescape_double(inner):
    """还原双引号里的转义：`\\\\` -> `\\`、`\\"` -> `"`。

    只认这两种 —— `_fmt_value()` 写字符串时也只转义这两种，其余反斜杠
    （Windows 路径 `D:\\ai\\...`）原样保留。
    """
    out, i = [], 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and i + 1 < len(inner) and inner[i + 1] in ('\\', '"'):
            out.append(inner[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _to_scalar(text):
    """把配置里的字符串转成合适的 Python 值。

    带引号的字符串**原样去引号**（不会被当成数字/布尔/None），双引号里
    的 `\\\\` 和 `\\"` 按 `_fmt_value()` 的约定还原，保证写回再读回不变味。
    """
    t = text.strip()
    if t == "":
        return ""
    # ★ 先看引号：`"123"` 必须读成字符串 "123"，不能读成数字 123。
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
        inner = t[1:-1]
        return _unescape_double(inner) if t[0] == '"' else inner
    low = t.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", "~"):
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    if t.startswith("[") and t.endswith("]"):
        try:
            return ast.literal_eval(t)
        except Exception:
            pass
    return t


def _strip_comment(line):
    """去掉行内注释。引号里的 # 不算注释，引号内的 `\\"` 也不算结束。

    ★ 加转义处理是因为 `_fmt_value()` 会给含 `#` 的字符串加双引号，
    里面可能有 `\\"`（字符串本身就含引号时）。不认转义的话会在 `\\"` 上
    把引号状态搞反，后面的 `#` 就被误当注释了。
    """
    out, quote, escaped = [], None, False
    for ch in line:
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        else:
            if ch in "\"'":
                quote = ch
            elif ch == "#":
                break
        out.append(ch)
    return "".join(out)


def _coerce(key, value):
    """按 spec 的类型把值转对。转不了就原样返回（交给校验器报错）。"""
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if key in _BOOL_KEYS:
        return low in ("true", "yes", "on", "1")
    if key in _INT_KEYS:
        try:
            return int(float(low))
        except ValueError:
            return value
    if key in _FLOAT_KEYS:
        try:
            return float(low)
        except ValueError:
            return value
    return value


def _load_ini(path):
    cfg = {}
    cur = None
    first_seen = {}       # (段, 键) -> 第一次出现的行号，用于检测重复键
    # ⚠️ 必须用 utf-8-sig：Windows 上的记事本、PowerShell 的
    # `Set-Content -Encoding UTF8`、VS Code 的「带 BOM 保存」
    # 都会在文件最前面写三个字节的 BOM（EF BB BF）。
    # 用普通 utf-8 读的话，第一行会变成 "\ufeff[model]"，于是
    #   - 它不以 "[" 开头 -> 不被当成节名
    #   - 又不含 "=" -> 报「第 1 行看不懂（缺少 = 号）」
    # 用户看着一个明明正确的配置文件被拒绝，完全摸不着头脑。
    # utf-8-sig 在读无 BOM 文件时与 utf-8 完全一致，所以可以无脑用。
    with open(path, "r", encoding="utf-8-sig") as fp:
        for lineno, raw in enumerate(fp, 1):
            line = _strip_comment(raw).strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = line[1:-1].strip()
                cfg.setdefault(cur, {})
                continue
            if "=" not in line:
                raise ValueError(
                    "%s 第 %d 行看不懂（缺少 = 号）：%s" % (path, lineno, raw.rstrip()))
            if cur is None:
                raise ValueError(
                    "%s 第 %d 行出现在任何 [节] 之前：%s" % (path, lineno, raw.rstrip()))
            k, v = line.split("=", 1)
            key = k.strip()
            # ★ 重复键必须出声：INI 语义是「后面的赢」，用户复制粘贴时
            #   很容易留下两行，界面上看着改对了、实际用的还是旧值。
            if key in cfg[cur]:
                warnings.warn(
                    "%s 的 [%s] 段里 %s 出现了不止一次（第 %d 行和第 %d 行），"
                    "以后面那次为准 —— 如果前面那次才是你要的，删掉重复行。"
                    % (path, cur, key, first_seen[(cur, key)], lineno),
                    UserWarning, stacklevel=2)
            else:
                first_seen[(cur, key)] = lineno
            cfg[cur][key] = _to_scalar(v)
    return cfg


def load_config(path):
    """读配置文件。自动按扩展名选格式。

    一律用 `utf-8-sig` 打开：Windows 的记事本 / PowerShell 重定向 /
    VS Code 都可能写出带 BOM 的文件（详见 `_load_ini` 的注释）。
    `utf-8-sig` 对无 BOM 文件与 `utf-8` 行为一致，所以统一用没有副作用。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到配置文件：%s" % path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".ini", ".cfg", ".conf", ".txt", ""):
        cfg = _load_ini(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8-sig") as fp:
            cfg = json.load(fp)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ValueError(
                "读 YAML 配置需要 pyyaml。装它：\n"
                "  pip install pyyaml\n"
                "或者把配置改成 .ini 格式（推荐，记事本就能改）。")
        with open(path, "r", encoding="utf-8-sig") as fp:
            cfg = yaml.safe_load(fp) or {}
    else:
        raise ValueError("不认识的配置格式 %s（支持 .ini / .json / .yaml）" % ext)

    if not isinstance(cfg, dict):
        raise ValueError("配置文件的内容不是一个「节 -> 键值」的结构：%s" % path)
    out = {}
    for sec, params in cfg.items():
        if not isinstance(params, dict):
            raise ValueError(
                "配置里的 [%s] 段内容不对，应该是「键 = 值」的列表" % sec)
        out[sec] = {k: _coerce(k, v) for k, v in params.items()}
    return out


def resolve(cfg, sections=None):
    """把配置**补全**：spec 里有默认值、而你配置里没写的项，一律填上默认值。

    ## 为什么必须有这个函数（真实 bug）

    `train.py` 里读优化器参数是直接下标取值：`cfg["weight_decay"]`。
    用户手写一份精简配置、只写想改的那几项，就会得到：

        KeyError: 'weight_decay'

    这个报错完全看不出该补哪一项，也看不出合法取值范围。

    正确的修法**不是**在每个取值点写 `cfg.get(k, 默认值)` —— 那样默认值就
    散落到几十处、跟 spec.py 变成两份要同步维护的东西，迟早对不上
    （这正是当初 `--set` 静默忽略、组件名被忽略那批 bug 的成因）。
    正确做法是**在入口处一次补全**：spec.py 本来就写着每个参数的默认值，
    读进来补齐，后面所有代码就可以放心用下标。

    `sections=None` 表示全部 7 段都补。只想补某几段就传列表。
    """
    from .spec import SPEC
    out = {}
    for sec in (sections if sections is not None else SPEC.keys()):
        params = SPEC.get(sec, {})
        # 先放配置里的值（用户给的优先），再把缺的默认值补上
        got = dict(cfg.get(sec, {}) or {})
        for key, p in params.items():
            if key not in got:
                got[key] = p["default"]
            else:
                got[key] = _coerce(key, got[key])
        out[sec] = got
    # 配置里有、但 spec 里没有的段/键：原样保留。
    # 留着而不是丢掉，是为了不破坏插件自己加的配置项。
    for sec, params in cfg.items():
        if sec not in out:
            out[sec] = dict(params)
        else:
            for k, v in params.items():
                out[sec].setdefault(k, v)
    return out


def missing_required(cfg):
    """列出「必填但没填」的项，给报错信息用。

    spec 里 default 为 None 的参数视为必填（没有合理的默认值，
    例如 data_dir —— 引擎不可能猜出你的语料在哪）。
    """
    from .spec import SPEC
    out = []
    for sec, params in SPEC.items():
        for key, p in params.items():
            if p["default"] is None:
                val = (cfg.get(sec) or {}).get(key)
                if val in (None, ""):
                    out.append((sec, key, p))
    return out


def _fmt_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        # ★ 不能写 "None" —— 那是四个字母，读回来是个字符串，
        #   而 spec 里 default=None 的键（data_dir 之类）本意是「没设置」。
        return ""
    if isinstance(v, str):
        return _fmt_str(v)
    return str(v)


def _fmt_str(s):
    """把字符串写成 INI 值；**只在读回来会变味时才加引号**。

    ## 为什么必须加引号（真实事故）
    原来的实现一律 `str(v)` 不加引号，于是：
      - `prompt = 说 # 你好` 写回后读成 `"说"`（# 后面被当注释）
      - `prompt =   空格  ` 首尾空格被 strip 掉
      - `prompt = 123` / `= true` 读回来变成数字 / 布尔
    GUI 改一次配置、保存，值就悄悄变了。现在会写成
    `prompt = "说 # 你好"`，`_to_scalar()` 读回时按引号里的转义还原。
    """
    if s == "":
        return ""
    if "#" in s or s != s.strip() or _to_scalar(s) != s:
        return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')
    return s


#: INI 行：`  键 = 值   # 注释`。
#: 值可以是用双引号包起来的字符串（里面能放 `#`、首尾空格、像数字的文本），
#: 引号内用 `\\"` / `\\\\` 转义；不带引号的值里不含 `#`（那会被当注释）。
_INI_LINE = re.compile(
    r'^(\s*)([A-Za-z_][\w.\-]*)(\s*=\s*)'
    r'("(?:[^"\\]|\\.)*"|[^#]*?)'
    r'(\s*#.*)?$')
_INI_SECTION = re.compile(r"^\s*\[\s*([A-Za-z_][\w.\-]*)\s*\]\s*$")


def patch_config(path, changes):
    """**就地**改配置里的几个键，其余一个字节都不动。

    返回真正改掉的键数；文件不存在或格式看不懂时返回 0。

    ## 为什么要单独有这个函数

    `save_config()` 是「按 spec 重新生成整份文件」，会把用户手工写的
    中文注释**全部抹掉**。而出厂配置的价值有一大半就在那些注释里
    （「n_layer 堆几层」「n_kv_head 越小 KV cache 越省」）。

    界面里「改一个语料路径」「改一个学习率」这种小动作如果走
    `save_config()`，用户只是点了一下按钮，回头打开配置文件就发现
    自己看不懂了。所以界面改单个键时走这里。
    """
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig") as fp:
            lines = fp.read().splitlines()
    except OSError:
        return 0

    # (段, 键) -> 新值
    want = {}
    for (sec, key), val in changes.items():
        want[(sec, key)] = _fmt_value(val)

    changed = set()
    seen = set()          # ★ 和 changed 必须分开。
    # 只看 changed 的话，「值已经是对的」那些键会被算成「没找到」，
    # 于是再追加一行 —— 文件里出现两个 data_dir，读回来用哪个看运气。
    done = set()          # 已经写过的键；再遇到同一键就是重复行，必须删掉
    out = []
    sec = None
    for line in lines:
        m = _INI_SECTION.match(line)
        if m:
            sec = m.group(1)
            out.append(line)
            continue
        m = _INI_LINE.match(line)
        if m and sec is not None:
            k = (sec, m.group(2))
            if k in want:
                if k in done:
                    # ★ 同一个键出现第二次：INI 解析是「后面的赢」，
                    #   留着它，刚写进去的新值会被它静默覆盖回去。
                    changed.add(k)
                    continue
                done.add(k)
                seen.add(k)
                if want[k] != m.group(4).strip():
                    out.append("%s%s%s%s%s" % (m.group(1), m.group(2),
                                               m.group(3), want[k],
                                               m.group(5) or ""))
                    changed.add(k)
                    continue
        out.append(line)

    # 没找到的键：追加到它所属的段末尾（段不存在就新建一段）
    missing = [k for k in want if k not in seen]
    for sec_name in dict.fromkeys(s for s, _ in missing):
        keys = [(s, kk) for (s, kk) in missing if s == sec_name]
        idx = None
        for i, line in enumerate(out):
            m = _INI_SECTION.match(line)
            if m and m.group(1) == sec_name:
                idx = i
        if idx is None:
            out.extend(["", "[%s]" % sec_name])
            idx = len(out) - 1
        else:
            # 插到这一段最后一行非空内容之后
            j = idx + 1
            while j < len(out) and not _INI_SECTION.match(out[j]):
                if out[j].strip():
                    idx = j
                j += 1
        out[idx + 1:idx + 1] = ["%s = %s" % (kk, want[(s, kk)])
                                for (s, kk) in keys]
        changed.update(keys)

    if not changed:
        return 0
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(out).rstrip() + "\n")
    return len(changed)


def _ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _save_json(path, cfg):
    """写 JSON 配置。JSON 放不下注释（format 本身不支持），所以 comment 忽略。"""
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def _save_yaml(path, cfg):
    try:
        import yaml
    except ImportError:
        raise ValueError(
            "写 YAML 配置需要 pyyaml。装它：\n"
            "  pip install pyyaml\n"
            "或者把配置改成 .ini 格式（推荐，记事本就能改）。")
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)


def save_config(path, cfg, comment=None, preserve=True):
    """写配置文件。按扩展名选格式：.json / .yaml / 其余一律 INI。

    `preserve=True`（默认）时，如果目标文件已存在，会走 `patch_config()`
    **就地改值**，保住用户/出厂写的中文注释。只有在文件不存在、
    或者出现了新键时才退回「整份重写」。

    ## 以前的问题
    无视扩展名一律写 INI —— `save_config("x.json", ...)` 会往 .json 文件里
    写 `[model]`，之后再 `load_config()` 读它就直接抛 json 解析错误。
    现在 .json/.yaml 交给对应格式写。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        _save_json(path, cfg)
        return
    if ext in (".yaml", ".yml"):
        _save_yaml(path, cfg)
        return

    if preserve and os.path.isfile(path):
        changes = {}
        for sec, params in cfg.items():
            if not isinstance(params, dict):
                continue
            for k, v in params.items():
                if isinstance(v, (dict, list)):
                    continue
                changes[(sec, k)] = v
        n = patch_config(path, changes)
        if n:
            return
        # 一个键都没改到 => 文件已经是想要的，或者格式不认。后者少见，
        # 但也别把用户的原文件覆盖掉：只有在「值确实不同」时才重写。
        try:
            with open(path, "r", encoding="utf-8-sig") as fp:
                old = load_config(path)
        except Exception:
            old = None
        if old is not None and _same_values(old, cfg):
            return

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    lines = []
    if comment:
        for line in comment.split("\n"):
            lines.append("# " + line if line else "#")
        lines.append("")
    for sec in SECTIONS:
        if sec not in cfg or not cfg[sec]:
            continue
        lines.append("[%s]" % sec)
        width = max((len(k) for k in cfg[sec]), default=0)
        for k, v in cfg[sec].items():
            lines.append("%-*s = %s" % (width, k, _fmt_value(v)))
        lines.append("")
    # 不在 SPEC 里的自定义节也保留
    for sec, params in cfg.items():
        if sec in SECTIONS or not isinstance(params, dict):
            continue
        lines.append("[%s]" % sec)
        for k, v in params.items():
            lines.append("%s = %s" % (k, _fmt_value(v)))
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines).rstrip() + "\n")


def _same_values(a, b):
    """两份配置在「所有出现的键」上是否等价（忽略缺失键）。"""
    for sec, params in b.items():
        if not isinstance(params, dict):
            continue
        other = a.get(sec) or {}
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                continue
            if other.get(k) != v:
                return False
    return True


# ---------------------------------------------------------------- 覆盖项

# 常见笔误 -> 正确写法。**按段位分开**，否则会把 model.lr 错纠正成
# model.learning_rate（那个键属于 [train]）。
_KEY_ALIASES = {
    "model": {
        "n_embed": "n_embd", "n_emb": "n_embd", "n_embedding": "n_embd",
        "n_layers": "n_layer", "layers": "n_layer", "num_layers": "n_layer",
        "n_heads": "n_head", "heads": "n_head", "num_heads": "n_head",
        "n_kv_heads": "n_kv_head", "kv_head": "n_kv_head", "kv_heads": "n_kv_head",
        "blocksize": "block_size", "ctx": "block_size", "context": "block_size",
        "seq_len": "block_size", "max_seq_len": "block_size",
        "ffn_dim": "d_ff", "d_ffn": "d_ff", "hidden_dim": "d_ff",
        "moe": "use_moe", "experts": "n_expert", "top_k": "n_expert_per_tok",
    },
    "train": {
        "out.dir": "out_dir", "outdir": "out_dir", "out": "out_dir",
        "output_dir": "out_dir", "save_dir": "out_dir",
        "steps": "max_steps", "stepps": "max_steps", "iterations": "max_steps",
        "lr": "learning_rate", "learningrate": "learning_rate",
        "batch": "batch_size", "bs": "batch_size",
        "accum": "grad_accum", "gradient_accumulation": "grad_accum",
        "eval_every": "eval_interval", "save_interval": "ckpt_interval",
        "max_lr": "learning_rate", "warmup": "warmup_iters",
    },
    "data": {
        "dataroot": "data_dir", "data_path": "data_dir", "datadir": "data_dir",
        "raw_dir": "data_dir", "tokenizer": "tokenizer_path",
        "vocab_path": "tokenizer_path", "tok_path": "tokenizer_path",
    },
    "sample": {
        "temp": "temperature", "prompt_text": "prompt", "max_tokens": "max_new_tokens",
        "topk": "top_k", "topp": "top_p", "repeat_penalty": "repetition_penalty",
    },
    "post": {
        "template_name": "template", "chat_template": "template",
        "data": "data_path", "steps": "max_steps", "lr": "learning_rate",
    },
    "parallel": {
        "tp": "tensor_parallel", "pp": "pipeline_parallel", "ep": "expert_parallel",
        "zero": "zero_stage", "offload": "cpu_offload",
        "act_ckpt": "activation_checkpoint",
    },
    "runtime": {"language": "lang", "log": "log_level"},
}


def _suggest(section, key):
    import difflib
    known = list(all_keys(section))
    if not known:
        return []
    return difflib.get_close_matches(key, known, n=3, cutoff=0.6)


def apply_overrides(cfg, pairs, on_notice=None):
    """应用命令行覆盖项，形如 ["model.n_layer=12", "train.max_steps=2000"]。

    会校验节名和键名 —— 写错时立刻报错并给建议，**不静默忽略**。
    （静默忽略是最坑的：用户以为改成功了，其实还在用旧值。）
    """
    def notice(s):
        if on_notice:
            on_notice(s)

    for item in pairs or []:
        if "=" not in item:
            raise ValueError(
                "--set 的格式是 节名.键名=值，你给的是：%s\n"
                "  例子：--set train.max_steps=2000\n"
                "  改多项就写多个 --set：\n"
                "    --set train.max_steps=1000 --set train.out_dir=D:/out/新目录" % item)
        lhs, rhs = item.split("=", 1)
        if "." not in lhs:
            raise ValueError("--set 的左边要写成 节名.键名，你给的是：%s" % lhs)
        sec, key = lhs.split(".", 1)
        sec, key = sec.strip().lower(), key.strip().lower()

        if sec not in SECTIONS:
            raise ValueError("没有 [%s] 这个配置节。可用的有：%s"
                             % (sec, "、".join(SECTIONS)))

        known = all_keys(sec)
        if key not in known:
            good = _KEY_ALIASES.get(sec, {}).get(key)
            if good and good in known:
                notice("  [提示] --set 里的 %s.%s 已自动纠正为 %s.%s"
                       % (sec, key, sec, good))
                key = good

        if known and key not in known:
            sug = _suggest(sec, key)
            msg = ("[%s] 里没有 %s 这个配置项 —— 如果继续，这个 --set 会被忽略，"
                   "你会以为改成功了其实没改。\n" % (sec, key))
            if sug:
                msg += "  你是不是想写：%s\n" % "、".join("%s.%s" % (sec, s) for s in sug)
            msg += "  [%s] 可用的项：%s" % (sec, "、".join(sorted(known)))
            raise ValueError(msg)

        cfg.setdefault(sec, {})[key] = _coerce(key, rhs.strip())
    return cfg


def get(cfg, section, key, default=None, required=False):
    """取一个配置值。

    优先级：配置文件 > spec 默认值 > 调用方给的 default
    required=True 且配置和 spec 都没有时报错。
    """
    if section in cfg and key in cfg[section]:
        val = cfg[section][key]
        if val is not None and val != "":
            return _coerce(key, val)
    p = SPEC.get(section, {}).get(key)
    if p is not None:
        return p["default"]
    if required:
        raise KeyError(
            "配置缺少 [%s] 下的 %s。可以在配置文件里写 `%s = 值`，"
            "或用 --set %s.%s=值 临时指定。" % (section, key, key, section, key))
    return default


def as_dict(cfg):
    """深拷贝一份。"""
    return {s: dict(d) for s, d in cfg.items() if isinstance(d, dict)}
