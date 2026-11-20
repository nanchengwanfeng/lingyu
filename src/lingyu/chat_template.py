# -*- coding: utf-8 -*-
r"""对话模板：把「多轮问答」变成模型能学的一串 token。

## 要解决的问题

预训练模型只会**续写**：给它「从前有座山」，它接「山里有座庙」。
你没法问它问题，因为它没见过「问题」和「回答」这两种角色的区分。

SFT（监督微调）做的就是教它这个格式：

    用户：什么是光合作用？
    助手：光合作用是植物把光能变成化学能的过程……

教几万条这样的样本之后，你只要写出「用户：xxx\n助手：」，
它就知道该接着写**回答**，而不是继续编用户的话。

这个文件负责「怎么把问答渲染成 token」，以及一个很关键但常被忽略的细节：
**哪些 token 该算 loss，哪些不该**（见下面 loss mask 一节）。

## 支持的模板

| 名字 | 长什么样 | 需要特殊 token 吗 |
|---|---|---|
| `plain` | `用户：…\n助手：…\n` 纯中文标记 | 不需要 |
| `chatml` | `<\|im_start\|>user\n…<\|im_end\|>\n` | 需要 |
| `qwen` | 同 chatml，助手部分略作调整 | 需要 |

### 为什么默认不是 chatml

`plain` 用「用户：」「助手：」这种**本来就是普通汉字**的标记，因此：

- 零词表改动，任何老断点直接能用
- 对小模型更友好：中文标记在预训练语料里出现过千万次，
  embedding 是训练好的；而 `<|im_start|>` 这种只在 SFT 阶段见到，
  13M 的模型根本没足够数据把它学明白

ChatML 的价值在于**互通性**（和 HuggingFace 生态、其他工具对接），
所以两个都留着，按需要选。新建的分词器两种都能用。

## loss mask（新手最容易漏的一步）

SFT 时**不能对「用户：…」这几个字算 loss**。原因：

模型的任务是「学会怎么回答」。如果连问题也一起算 loss，
模型就在学「怎么复述问题」—— 而问题是用户给的，它不需要学。
更糟的是 Prompts 通常比回答长，不算 mask 的话大部分梯度都花在
「抄题」上，回答部分反而学不扎实。

所以我们要产出一个和 `ids` 等长的 `mask`：

    mask[i] = 1  ->  第 i 个位置要算 loss（回答部分）
    mask[i] = 0  ->  不算（问题部分、角色标记、padding）

`post.loss_on_prompt = true` 可以关掉这个行为（教学时用来做对比实验）。
"""

import json
import os

from . import utils
from .tokenizer import IM_END, IM_START

# ---------------------------------------------------------------------------
# 角色
# ---------------------------------------------------------------------------
ROLES = ("system", "user", "assistant")

# 中文角色名。`plain` 模板用它，也用于给人看的展示。
ROLE_ZH = {"system": "系统", "user": "用户", "assistant": "助手"}

# 模板名 -> 是否需要特殊 token
TEMPLATES = ("plain", "chatml", "qwen")

# qwen 模板与 chatml 结构完全相同，唯一区别是：**没有 system 消息时**
# 补一条官方默认系统提示（Qwen 系列 tokenizer 的默认 chat template 行为）。
# 旧版实现把这个参数完全忽略，导致 `--template qwen` 与 `chatml` 输出逐字节
# 相同 —— 用户以为选了 Qwen 格式，实际不是。
QWEN_DEFAULT_SYSTEM = "You are a helpful assistant."


def needs_special_tokens(template):
    """这个模板是否依赖 <|im_start|> 这类特殊 token。"""
    return (template or "plain").lower() not in ("plain",)


def available_templates(tok=None):
    """列出可用模板。给了分词器就顺便标出哪些当前用不了。"""
    out = []
    for name in TEMPLATES:
        if tok is None or not needs_special_tokens(name):
            out.append((name, True, ""))
            continue
        have = has_chat_special_tokens(tok)
        out.append((name, have,
                    "" if have else "当前分词器没有 <|im_start|>（需要重新跑 prepare）"))
    return out


def has_chat_special_tokens(tok):
    """分词器里有没有 ChatML 的起止标记。

    老分词器（这个功能之前 prepare 出来的）只有 4 个特殊 token，
    没有这两个。**不能靠 vocab_size 判断** —— 数量可能碰巧一样。
    """
    return ("<|im_start|>" in getattr(tok, "vocab", {})
            and "<|im_end|>" in getattr(tok, "vocab", {}))


def check_template_usable(template, tok):
    """模板能不能用这个分词器。不能用就抛一个说人话的错。

    这里刻意**报错而不是自动降级**：悄悄把 chatml 换成 plain，
    用户会以为自己在训 ChatML 格式，导出给别人却对不上。
    """
    if not needs_special_tokens(template):
        return
    if has_chat_special_tokens(tok):
        return
    raise ValueError(
        "模板 %r 需要特殊 token <|im_start|> / <|im_end|>，"
        "但当前分词器里没有这两个。\n"
        "你有两个选择：\n"
        "  1) 用纯文本模板（不改词表，任何断点都能用）：\n"
        "       --set post.template=plain\n"
        "  2) 重新准备数据，让词表带上这两个 token（推荐，长期做法）：\n"
        "       python -m lingyu prepare --raw <语料目录> --data <数据目录>\n"
        "     注意：重新 prepare 会让**旧断点失效**（ID 分配变了），需要重新预训练。"
        % template)


# ---------------------------------------------------------------------------
# 断点与分词器是否配套
# ---------------------------------------------------------------------------

def check_ckpt_tokenizer(ck, tok, where="断点"):
    """核对断点里记录的分词器指纹与当前分词器是否一致。

    为什么必须查：断点里只存了 `tokenizer_path`（一个路径），而用户完全
    可能显式指定另一份分词器、或者那份路径已经失效被换掉。两份分词器
    **词表大小可能一样**（比如都是 8192）但 token id 完全不同 —— 这时
    加载不会报任何错，loss 照降，输出全是乱码。只有指纹能发现。

    断点没有指纹（旧格式）时**不静默通过**：明确 warn「无法核对」，
    让用户自己知道这道防线没起作用。
    """
    want = (ck or {}).get("tokenizer_fingerprint")
    have = tok.fingerprint()
    if want:
        if want != have:
            raise ValueError(
                "%s里的分词器和当前分词器不一致（指纹 %s ≠ %s）。\n"
                "  token id 对不上，继续只会得到乱码。两条出路：\n"
                "  ① 换回当初做断点时用的分词器："
                " --set data.tokenizer_path=<那个 tokenizer.txt>\n"
                "  ② 换一个和当前分词器配套的断点："
                " --ckpt <断点路径>（SFT 用 --init-from）"
                % (where, want, have))
        return True
    utils.warn("这个%s没有记录分词器指纹（旧格式），无法核对 —— "
               "如果你换过分词器，训练/对话的结果会是乱码。" % where)
    return False


# ---------------------------------------------------------------------------
# 渲染：消息列表 -> token id 序列 + loss mask
# ---------------------------------------------------------------------------

def normalize_messages(messages):
    """检查并规范化消息列表。

    接受 [{"role": "user", "content": "..."}, ...]，
    也接受更松的写法（role 用中文、缺 role 视为 user），因为手写数据集
    难免写错，报错时要说清是哪一条的第几个。
    """
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages 必须是列表，现在是 %s" % type(messages).__name__)
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError("第 %d 条消息不是字典：%r" % (i, m))
        role = m.get("role") or m.get("from") or "user"
        role = str(role).strip().lower()
        # 常见别名：中文角色名、human/gpt 这类写法
        role = {"用户": "user", "助手": "assistant", "系统": "system",
                "human": "user", "gpt": "assistant", "ai": "assistant",
                "bot": "assistant", "model": "assistant"}.get(role, role)
        if role not in ROLES:
            raise ValueError(
                "第 %d 条消息的 role 是 %r，只能是 %s 之一"
                % (i, m.get("role"), "、".join(ROLES)))
        content = m.get("content")
        if content is None:
            content = m.get("value")
        if content is None:
            raise ValueError("第 %d 条消息没有 content" % i)
        out.append({"role": role, "content": str(content)})
    return out


def render(messages, tok, template="plain", add_generation_prompt=False,
           system=None):
    """把消息渲染成 (ids, mask)。

    参数
        messages            已经规范化的消息列表（推荐先过 normalize_messages）
        tok                 分词器
        template            'plain' / 'chatml' / 'qwen'
                            （chatml 与 qwen 结构相同；qwen 在没有 system
                            消息时补一条默认系统提示，见 QWEN_DEFAULT_SYSTEM）
        add_generation_prompt
                             True = 结尾补上「助手：」，用来让模型接着回答
                             （推理时用；训练时不需要）
        system              系统提示词（可选）。也可以直接写在 messages 里

    返回 (ids, mask)，都是等长的 list[int]。
    """
    template = (template or "plain").lower()
    if template not in TEMPLATES:
        raise ValueError("不认识的对话模板 %r，可用：%s"
                         % (template, "、".join(TEMPLATES)))
    check_template_usable(template, tok)

    msgs = normalize_messages(messages)
    if system:
        msgs = [{"role": "system", "content": system}] + \
               [m for m in msgs if m["role"] != "system"]

    # qwen 与 chatml 结构相同；差别只在「没有 system 时补默认系统提示」。
    if template == "qwen" and not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": QWEN_DEFAULT_SYSTEM}] + msgs

    if template == "plain":
        return _render_plain(msgs, tok, add_generation_prompt)
    return _render_chatml(msgs, tok, add_generation_prompt, template)


def _text_ids(tok, text):
    """把一段**普通文本**编码成 id（不加 bos/eos）。

    ⚠️ 特殊 token 在这里是**普通字符串**，所以模板自己负责把
    <|im_start|> 用 ID 直接插进去，不经过编码器 —— 编码器看到
    "<|im_start|>" 只会把它当一串普通字符来处理。
    """
    return list(tok.encode(text))


def _render_plain(msgs, tok, add_generation_prompt):
    """纯中文标记模板：

        系统：你是一个助手
        用户：你好
        助手：你好！
        <eos>

    末尾用 <eos> 收尾，教模型「一句话说完了要停」。
    没有 <eos> 的话模型会一直往下编，停不下来。
    """
    ids, mask = [], []
    for m in msgs:
        head = _text_ids(tok, "%s：" % ROLE_ZH[m["role"]])
        body = _text_ids(tok, m["content"]) + _text_ids(tok, "\n")
        # 角色标记不算 loss，回答内容算；问题和系统提示词都不算
        on = 1 if m["role"] == "assistant" else 0
        ids += head + body
        mask += [0] * len(head) + [on] * len(body)
    if add_generation_prompt:
        head = _text_ids(tok, "助手：")
        ids += head
        mask += [0] * len(head)
    elif any(m["role"] == "assistant" for m in msgs):
        # 只有**确实有回答**时才补 <eos> 教它收尾。
        # 如果这条样本只有 user、没有 assistant，补一个 mask=1 的 <eos>
        # 等于在教模型「被问了就别吭声」—— 这种样本本来就该整条丢掉，
        # 但前提是它的 mask 得真的是全 0（否则 render_dataset 不会丢它）。
        ids.append(2)          # EOS
        mask.append(1)         # 学会「该停了」也是回答的一部分
    return ids, mask


def _render_chatml(msgs, tok, add_generation_prompt, template):
    r"""ChatML：

        <|im_start|>system
        你是一个助手<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！<|im_end|>

    qwen 模板的结构与此完全相同；两者的唯一区别（没有 system 时补默认
    系统提示）在 `render()` 里完成，所以本函数不区分 template。
    """
    ids, mask = [], []
    for m in msgs:
        ids.append(IM_START)
        mask.append(0)
        head = _text_ids(tok, m["role"] + "\n")
        ids += head
        mask += [0] * len(head)
        body = _text_ids(tok, m["content"])
        on = 1 if m["role"] == "assistant" else 0
        ids += body
        mask += [on] * len(body)
        ids.append(IM_END)
        # <|im_end|> 对助手来说是「说完了」的信号，要学；对别的角色不用
        mask.append(on)
        nl = _text_ids(tok, "\n")
        ids += nl
        mask += [0] * len(nl)
    if add_generation_prompt:
        ids.append(IM_START)
        mask.append(0)
        head = _text_ids(tok, "assistant\n")
        ids += head
        mask += [0] * len(head)
    return ids, mask


def render_for_inference(messages, tok, template="plain", system=None):
    """推理用：渲染对话历史 + 「该你说了」的提示，返回 ids。"""
    ids, _ = render(messages, tok, template=template,
                    add_generation_prompt=True, system=system)
    return ids


# ---------------------------------------------------------------------------
# 数据集文件
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """读 JSONL 数据集：一行一条样本。

    每行的格式（三种都认）：

        {"messages": [{"role":"user","content":"…"}, …]}
        {"conversations": [{"from":"human","value":"…"}, …]}   # 常见开源格式
        {"prompt": "…", "response": "…"}                       # 最简写法

    为什么容忍这么多种：网上能找到的中文对话数据集格式五花八门，
    让用户先写脚本转换格式是很烦的事。这里都认下来，
    真正不认识的再报错并指出是第几行。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到数据集文件：%s" % path)
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, 1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "数据集第 %d 行不是合法 JSON：%s\n"
                    "  这一行的开头是：%s\n"
                    "  常见原因：字符串里的引号没转义、少了一个大括号。"
                    % (lineno, e.msg, line[:80]))
            msgs = _extract_messages(obj, lineno)
            rows.append(msgs)
    if not rows:
        raise ValueError("数据集 %s 里一条样本都没有" % path)
    return rows


def _extract_messages(obj, lineno):
    """从一行 JSON 里把消息列表抠出来。"""
    if not isinstance(obj, dict):
        raise ValueError("数据集第 %d 行不是 JSON 对象" % lineno)
    for key in ("messages", "conversations", "conversation", "dialog"):
        if key in obj:
            return normalize_messages(obj[key])
    if "prompt" in obj and "response" in obj:
        return normalize_messages([{"role": "user", "content": obj["prompt"]},
                                   {"role": "assistant", "content": obj["response"]}])
    if "instruction" in obj and "output" in obj:
        # 指令数据集（Alpaca 风格）
        msgs = []
        if obj.get("input"):
            msgs.append({"role": "user",
                         "content": "%s\n%s" % (obj["instruction"], obj["input"])})
        else:
            msgs.append({"role": "user", "content": obj["instruction"]})
        msgs.append({"role": "assistant", "content": obj["output"]})
        return normalize_messages(msgs)
    raise ValueError(
        "数据集第 %d 行认不出格式。需要下面任一种：\n"
        '  {"messages": [{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}\n'
        '  {"conversations": [{"from":"human","value":"…"},{"from":"gpt","value":"…"}]}\n'
        '  {"prompt": "…", "response": "…"}\n'
        '  {"instruction": "…", "input": "…", "output": "…"}\n'
        "  实际拿到的键是：%s" % (lineno, sorted(obj.keys())))


def render_dataset(rows, tok, template="plain", max_seq_len=None,
                   add_eos=True, system=None, loss_on_prompt=False):
    """把整个数据集渲染成样本列表，并统计有用的信息。

    system        : 给每条样本注入的系统提示（训练和推理必须用同一个，
                    否则模型学会了 A 却在实际对话时看到 B）
    loss_on_prompt: True 时问题/格式标记也参与 loss（默认只学回答）。
                    小数据下开它模型容易学会「复述问题」，一般不要开。

    返回 dict：
        samples    [{"ids": [...], "mask": [...], "n_supervised": n}, …]
        skipped    因为太长被丢掉的条数（附示例原因）
        stats      平均长度、被监督 token 占比等
    """
    samples, skipped, reasons = [], 0, []
    total_sup = total_len = 0
    for row in rows:
        ids, mask = render(row, tok, template=template, system=system)
        if loss_on_prompt:
            # 整条都学：问题、角色标记都算 loss。这是「续写式」训练，
            # 通常只在小数据/特定场景（比如教格式）才用。
            mask = [1] * len(mask)
        if max_seq_len and len(ids) > max_seq_len:
            skipped += 1
            if len(reasons) < 3:
                reasons.append("一条样本 %d 个 token，超过 max_seq_len=%d"
                               % (len(ids), max_seq_len))
            continue
        n_sup = sum(mask)
        if n_sup == 0:
            # 整条都没有要学的部分（比如只有 user 没有 assistant）—— 丢掉，
            # 否则这条样本对 loss 毫无贡献，纯浪费算力
            skipped += 1
            if len(reasons) < 3:
                reasons.append("这条样本没有助手回复，没有任何 token 需要学习")
            continue
        samples.append({"ids": ids, "mask": mask, "n_supervised": n_sup})
        total_sup += n_sup
        total_len += len(ids)
    stats = {
        "n_samples": len(samples),
        "n_skipped": skipped,
        "avg_len": (total_len / len(samples)) if samples else 0,
        "supervised_ratio": (total_sup / total_len) if total_len else 0,
    }
    return {"samples": samples, "skipped": skipped,
            "skip_reasons": reasons, "stats": stats}


def describe_templates(tok):
    """给 doctor / GUI 用：列出模板与可用状态。"""
    lines = []
    for name, ok, why in available_templates(tok):
        lines.append("  %-8s %s%s" % (name, "可用" if ok else "不可用",
                                      ("  ← " + why) if why else ""))
    return "\n".join(lines)
