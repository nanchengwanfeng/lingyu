# -*- coding: utf-8 -*-
"""杂项工具：控制台输出、随机种子、参数量与显存估算、格式化。

这里刻意把「估算」写得很直白，因为训练前最需要知道的就是
「我要花多少显存、大概跑多久」，而不是开跑之后才 OOM 崩掉。
"""

import json
import os
import random
import sys
import time

# ---------------------------------------------------------------- 控制台输出

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
}


def c(text, color):
    if not _USE_COLOR:
        return text
    return _C.get(color, "") + str(text) + _C["reset"]


def title(text):
    _say()
    _say(c("=" * 66, "cyan"))
    _say(c("  " + text, "cyan"))
    _say(c("=" * 66, "cyan"))


def info(msg):
    _say("  " + str(msg))


def ok(msg):
    _say("  " + c("[通过] ", "green") + str(msg))


def warn(msg):
    if _JSON_MODE:
        print("  " + c("[注意] ", "yellow") + str(msg),
              file=sys.stderr, flush=True)
        return
    _say("  " + c("[注意] ", "yellow") + str(msg))


def err(msg):
    print("  " + c("[错误] ", "red") + str(msg), file=sys.stderr)


def die(msg, code=1):
    err(msg)
    sys.exit(code)


# ---------------------------------------------------------- 结构化输出（--json）
# `--json` 模式下 stdout **只输出一行一个 JSON 事件**；人类可读的日志全部
# 改走 stderr（或直接不打印）。GUI 靠这个把「进度 / loss / 存档」当数据读，
# 而不是用正则去猜中文日志（审计 1.4）。
#
# 事件类型（数据契约，只加不改）：
#   start  {task, max_steps, batch_size, block_size, tokens_per_step, device}
#   step   {task, step, total, loss, lr, grad_norm, tok_s, elapsed}
#   eval   {task, step, val_loss, best}
#   ckpt   {task, step, path, best}
#   sample {task, step, text}
#   probe  {task, step, question, answer}
#   done   {task, steps, best_val, out_dir, stopped, interrupted, elapsed}
#   error  {task, message}
_JSON_MODE = False
_STDOUT = sys.stdout


class _JsonStdout:
    """JSON 模式下的 stdout 过滤层：只放行 JSON 行，人类日志转 stderr。"""

    def __init__(self, raw):
        self._raw = raw
        self._buf = ""

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                self._raw.write(line + "\n")
            else:
                try:
                    sys.stderr.write(line + "\n")
                except (OSError, ValueError):
                    pass
        return len(text)

    def flush(self):
        try:
            self._raw.flush()
        except (OSError, ValueError):
            pass
        try:
            sys.stderr.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._raw, name)


_JSON_STDOUT = None


def set_json_mode(on=True):
    global _JSON_MODE, _JSON_STDOUT
    on = bool(on)
    if on and not _JSON_MODE:
        base = sys.stdout
        _JSON_STDOUT = _JsonStdout(base)
        sys.stdout = _JSON_STDOUT
    elif not on and _JSON_MODE:
        if _JSON_STDOUT is not None:
            try:
                _JSON_STDOUT.flush()
            finally:
                sys.stdout = _JSON_STDOUT._raw
            _JSON_STDOUT = None
    _JSON_MODE = on


def json_mode():
    return _JSON_MODE


def emit_event(event, task=None, **fields):
    """JSON 模式下往 stdout 打一行事件；人类模式什么都不做。"""
    if not _JSON_MODE:
        return
    payload = {"event": str(event)}
    if task:
        payload["task"] = str(task)
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = round(value, 6)
        elif not isinstance(value, (int, str, bool, float)):
            value = str(value)
        payload[key] = value
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def _say(text="", end="\n"):
    """人类可读输出。JSON 模式下静默（GUI 会自己从事件里画日志）。"""
    if _JSON_MODE:
        return
    print(text, end=end, file=sys.stdout)


def fmt_int(n):
    return "{:,}".format(int(n))


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return "%.2f %s" % (n, unit)
        n /= 1024.0


def fmt_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return "%.1f 秒" % seconds
    if seconds < 3600:
        return "%d 分 %02d 秒" % (int(seconds // 60), int(seconds % 60))
    return "%d 小时 %02d 分" % (int(seconds // 3600), int((seconds % 3600) // 60))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------- 外部停止协议

_STOP_ENV = "LINGYU_STOP_FILE"


def stop_file():
    """返回外部停止请求文件路径（环境变量 LINGYU_STOP_FILE 的值）。

    未设置时返回空串。GUI / 调度器用「创建这个文件」来请求训练优雅停止。
    """
    return (os.environ.get(_STOP_ENV) or "").strip()


def stop_requested():
    """外部是否请求停止：LINGYU_STOP_FILE 非空且该路径存在。

    训练循环每步调用一次；返回 True 后由调用方保存断点并正常退出。
    """
    path = stop_file()
    return bool(path) and os.path.exists(path)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def reset(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0


def human_step_line(step, max_steps, loss, lr, dt, tokens_per_sec, extra="", width=30):
    frac = 0.0 if max_steps <= 0 else min(1.0, step / float(max_steps))
    filled = int(round(width * frac))
    bar = "#" * filled + "." * (width - filled)
    pct = frac * 100.0
    line = "  步 %6d/%-6d [%s] %5.1f%%  loss %s  lr %.2e  %.0f tok/s" % (
        step, max_steps, bar, pct,
        ("%.4f" % loss) if loss is not None else "  ---  ",
        lr, tokens_per_sec)
    if extra:
        line += "  " + extra
    return line


# ---------------------------------------------------------------- 参数与显存

def count_parameters(model):
    """返回 (全部参数, 可训练参数)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_kv_cache_bytes(n_layer, n_kv_head, head_dim, seq_len, batch=1, dtype_bytes=2):
    """KV cache 字节数 = 2(K和V) * 层数 * KV头数 * 每头维度 * 序列长 * 批大小 * 每元素字节数。"""
    return 2 * n_layer * n_kv_head * head_dim * seq_len * batch * dtype_bytes


def estimate_training_memory(n_param, batch_size, block_size, n_embd, n_layer,
                             bytes_per_param=16, activation_factor=22, moe_extra_gb=0.0):
    """粗估训练显存（字节）。

    参数侧：优化器状态是最贵的一块。
        bf16 混合精度 + AdamW：权重 2 + 梯度 2 + fp32 主权重 4 + 一阶动量 4 + 二阶动量 4 = 16 字节/参数
        （最省可以到 12 字节/参数，这里按保守的 16 算）
    激活侧：与 批大小 × 序列长 × 宽度 × 层数 成正比，用一个经验系数折算。
    """
    param_bytes = n_param * bytes_per_param
    act_bytes = batch_size * block_size * n_embd * n_layer * activation_factor
    return param_bytes + act_bytes + int(moe_extra_gb * (1024 ** 3))


def suggest_batch_size(n_param, block_size, n_embd, n_layer, free_bytes,
                       bytes_per_param=16, activation_factor=22, reserve=0.85):
    """按可用显存推荐一个尽量大的批大小。返回 (批大小, 估算占用字节)。"""
    fixed = n_param * bytes_per_param
    budget = max(0.0, free_bytes * reserve - fixed)
    per_sample = block_size * n_embd * n_layer * activation_factor
    if per_sample <= 0:
        return 1, fixed
    bs = int(budget // per_sample)
    bs = max(1, min(bs, 512))
    # 取 2 的幂，省显存碎片
    power = 1
    while power * 2 <= bs:
        power *= 2
    est = fixed + power * per_sample
    return power, est


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_run_meta(out_dir, kind, **fields):
    """在输出目录写 `run.json`，告诉 GUI / 分析脚本「这次跑的是什么」。

    为什么需要它：监控页要区分**预训练**和 **SFT** 的 tok/s、loss、总步数。
    以前只能从当前配置猜 —— 配置里同时有 `[train] out_dir` 和 `[post] out_dir`，
    猜错就会把预训练的 93k tok/s 配到 SFT 的 loss 上（用户实测撞过）。

    字段是**数据契约**：只加不改。`kind` 取 `"pretrain"` / `"sft"`。
    写不进去只告警，不影响训练本身（格式化磁盘/权限问题不该让训练挂掉）。
    """
    meta = {"kind": kind, "format": 1,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta.update({k: v for k, v in fields.items() if v is not None})
    path = os.path.join(out_dir, "run.json")
    try:
        save_json(path, meta)
    except OSError as e:
        warn("run.json 没写成（%s），GUI 监控页会退回按配置估算" % e)
    return meta


def sizeof_file(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
