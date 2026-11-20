# -*- coding: utf-8 -*-
r"""`lingyu chat` —— 和训练好的模型对话。

## 为什么要有这个（而不只是 `sample`）

`sample` 是「续写」：你给它一个开头，它往下编。它不能回答问题，
因为你没法告诉它「这句是问题，请回答」。

`chat` 是「对话」：维护一段多轮历史，每轮把历史按对话模板渲染好，
在「助手：」的位置让模型接着说。这需要模型做过 SFT（见 `sft.py`）。

## 三个实际会遇到的问题

### 1. 小模型经常会「自问自答」

它写着写着输出「用户：… 助手：…」，因为训练数据里就是这种一串。
`_clean_reply` 会在第一个角色标记处截断。

### 2. 历史越来越长，最后超出上下文

处理办法是**只保留最近几轮**。这里按 token 数算，而不是按轮数 ——
一轮可能很长，按轮数砍会突然爆掉。

### 3. 13M 的模型答不好是正常的

这不是 bug。它没读过互联网，只是把那几本书和 SFT 样本背下来了。
它能学会的是**格式**（该它说话了、说完要停），而不是知识。
想验证「SFT 生效了」，看的是「它是否在回答的位置上写出通顺的中文」，
而不是「它答得对不对」。
"""

import os
import sys

import torch

from . import utils
from .chat_template import (check_ckpt_tokenizer, check_template_usable,
                            render_for_inference)
from .config import get
from .sample import load_model
from .sft import _clean_reply

EXIT_WORDS = ("exit", "quit", "退出", "再见", ":q", "q")


def build_prompt_ids(history, tok, template, system=None, max_prompt_tokens=1536,
                     block_size=512):
    """把对话历史渲染成 token id，太长就丢掉最早的几轮。

    `max_prompt_tokens` 是**提示词**能用的 token 预算，由调用方按
    `block_size - 给回答预留的空间` 算出来（见 `chat_once`）。它必须
    <= 模型的 block_size —— 旧版这里默认写死 1536，而模型 block_size
    是 1024，导致「预算内」的 prompt 被模型静默截断、甚至一个 token 都
    生成不出来。

    保留策略（按优先级）：
      1. system 提示词必须保留；
      2. 最后一轮用户问题必须保留；
      3. 其余历史从**最近**往前加，装不下就丢最早的轮次，并提示丢了几轮；
      4. 最后兜底：连「system + 最后一个问题」都装不下时，从最老的 token
         裁（宁可截断问题开头，也不能让 prompt 超过 block_size）。

    为什么按 token 数而不是轮数：一轮可能是一句话，也可能是一整段粘贴。
    """
    budget = int(max_prompt_tokens)
    if budget <= 0:
        raise ValueError("max_prompt_tokens 必须大于 0（收到 %r）" % (max_prompt_tokens,))

    # 1) 先把 system 摘出来（它每轮都要带上，且不能因为裁剪被丢掉）
    sys_msg = system
    if sys_msg is None:
        for m in history:
            if m["role"] == "system":
                sys_msg = m["content"]
                break
    dialog = [m for m in history if m["role"] != "system"]

    def render_ids(msgs):
        return render_for_inference(msgs, tok, template=template, system=sys_msg)

    full = render_ids(dialog)
    if len(full) <= budget:
        return full, 0

    # 2) 最后一轮用户问题（及其后的消息）必须保留
    last_user = None
    for i in range(len(dialog) - 1, -1, -1):
        if dialog[i]["role"] == "user":
            last_user = i
            break
    if last_user is None:                    # 异常输入：没有 user 消息
        last_user = max(0, len(dialog) - 1)
    mandatory = dialog[last_user:]
    prefix = dialog[:last_user]

    # 3) 从最近往前加历史轮次
    kept = []
    for m in reversed(prefix):
        trial = [m] + kept
        if len(render_ids(trial + mandatory)) > budget:
            break
        kept = trial
    dropped = len(prefix) - len(kept)
    ids = render_ids(kept + mandatory)
    if dropped:
        utils.warn("已省略前 %d 轮对话（%d 条消息），本轮上下文 %d/%d token"
                   % ((dropped + 1) // 2, dropped, len(ids), budget))

    # 4) 兜底：连「system + 最后一个问题」都装不下，从最老的 token 裁
    if len(ids) > budget:
        if sys_msg:
            sys_ids = render_for_inference(
                [{"role": "system", "content": sys_msg}], tok, template=template)
        else:
            sys_ids = []
        q_ids = render_for_inference(mandatory, tok, template=template,
                                     system=None)
        gen_ids = render_for_inference([], tok, template=template, system=None)
        keep_q = budget - len(sys_ids) - len(gen_ids)
        if keep_q <= 0:
            keep_sys = max(0, budget - len(gen_ids))
            sys_ids = sys_ids[-keep_sys:] if keep_sys else []
            q_ids = []
            utils.warn("系统提示词和问题太长，已从开头截断到 %d token" % budget)
        elif keep_q < len(q_ids):
            q_ids = q_ids[-keep_q:]
            utils.warn("问题太长，已从开头截断（保留最后 %d 个 token）" % keep_q)
        ids = sys_ids + q_ids + gen_ids
        if len(ids) > budget:                # 最后一道保险
            ids = ids[-budget:]
    return ids, dropped


def chat_once(model, tok, history, template, device, temperature=0.8,
              top_k=40, top_p=0.9, max_new_tokens=256, repetition_penalty=1.1,
              system=None, stream=True):
    """生成一轮回答，返回 (回答文本, 本轮实际用掉的 token 数)。"""
    block_size = int(model.cfg["block_size"])
    max_new_tokens = int(max_new_tokens)

    # 按**真实 block_size** 算提示词预算：给回答预留空间，并留 8 token 余量。
    # （旧版 build_prompt_ids 上限写死 1536，而 block_size 可能只有 1024：
    #  prompt 看似在预算内，模型却只拿到最后 block_size 个 token，system
    #  被静默截掉；prompt >= block_size 时 model.generate 一个 token 都不生成。）
    reserve = max(16, max_new_tokens)
    reserve = min(reserve, max(1, block_size // 2))
    prompt_budget = block_size - reserve - 8
    prompt_budget = max(16, min(prompt_budget, max(16, block_size - 16)))
    ids, _dropped = build_prompt_ids(history, tok, template, system=system,
                                     max_prompt_tokens=prompt_budget,
                                     block_size=block_size)

    # 提示词本身可能已经接近上下文上限，留出回答的空间
    room = max(0, block_size - len(ids) - 8)
    n_new = max(1, min(max_new_tokens, room))
    if n_new < max_new_tokens:
        utils.warn("上下文快满了，这轮最多只能生成 %d 个 token" % n_new)

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # 生成到「下一个角色标记」为止：模型一旦开始写「用户：」，
    # 说明它这轮说完了，继续生成只会污染历史。
    # 流式/非流式共用同一份 stop 集（禁止 token == 遇到就停）。
    stop_ids = _stop_token_ids(tok, template)
    specials = _special_ids(tok)

    if not stream:
        out = model.generate(idx, max_new_tokens=n_new, temperature=temperature,
                             top_k=top_k, top_p=top_p, eos_id=2,
                             repetition_penalty=repetition_penalty,
                             forbid_ids=tuple(sorted(stop_ids)))
        new_ids = out[0].tolist()[len(ids):]
        text = tok.decode([i for i in new_ids if i not in specials])
        # 与流式一致：跨 token 边界的「用户：」停止串，出现就截断。
        # （非流式走 model.generate，没法中途回调停止；这里保证**输出**与
        # 流式相同，用户不会因为选了 --no-stream 就看到多出来的自问自答。）
        cut = _user_marker_cut(text)
        if cut is not None:
            text = text[:cut]
        reply = _clean_reply(text)
        # 非流式也要把回答显示出来 —— 以前只返回不打印，用户会看到
        # 「灵语 > 你 >」紧挨着，以为模型没说话（回答其实已进 history）。
        if reply:
            sys.stdout.write(reply + "\n")
            sys.stdout.flush()
        return reply, len(new_ids)

    # 流式：逐个 token 打印，边生成边看
    text = _generate_stream(model, tok, idx, n_new, temperature, top_k,
                            top_p, stop_ids, repetition_penalty, len(ids))
    return _clean_reply(text), n_new


def _stop_token_ids(tok, template):
    """哪些 token 一出现就该停下。

    两类：
      1. ChatML 的 <|im_start|> / <|im_end|> —— 模型开始写它们，说明
         它在「换角色」或者以为这轮结束了。
      2. 「用户：」的第一个字（只在 char 分词器上）。

    ⚠️ 特殊 token **不分模板一律要停**。踩过的坑：原来只在 chatml/qwen
    模板下才加，于是用 plain 模板时模型吐出 `<|im_end|>` 也照样
    一路生成下去，还把这个标记**原样打在屏幕上**给用户看。
    """
    stops = set()
    for name in ("<|im_start|>", "<|im_end|>", "<pad>", "<unk>"):
        if name in tok.vocab:
            stops.add(tok.vocab[name])
    # 「用户：」的第一个 token：char 分词器下就是「用」这个字。
    # BPE 下会切成别的，所以只在 char 分词器上用这一招，避免误停。
    if getattr(tok, "kind", "") == "char" and "用" in tok.vocab:
        stops.add(tok.vocab["用"])
    return stops


def _special_ids(tok):
    """词表里的所有特殊 token id —— 这些永远不该打印给用户看。"""
    from .tokenizer import SPECIAL_TOKENS
    return set(tok.vocab[t] for t in SPECIAL_TOKENS if t in tok.vocab)


def _user_marker_cut(text):
    """在解码后的文本里找「用户：」停止串，返回要保留的长度（没有则 None）。

    为什么要解码后再找、而不是只比对 stop_ids：
    BPE 会把「用户：」切成好几个 token（比如「用户」+「：」），逐 token
    比对要么漏停、要么误停。解码窗口天然跨 token 边界，字符串匹配是可靠的。

    两种命中位置：开头（模型直接开始自问自答）、换行之后（正常的多轮格式）。
    返回 0 表示整段都不要（开头的「用户：」）。
    """
    if not text:
        return None
    if text.startswith("用户："):
        return 0
    i = text.find("\n用户：")
    if i >= 0:
        return i
    return None


def _apply_sampling(logits, recent_ids, temperature, top_k, top_p,
                    repetition_penalty):
    """对最后一步 logits 做「重复惩罚 → 温度 → top_k → top_p」。

    这里的每一项都**逐行对齐 model.generate**（chat 自有采样循环曾经和
    model.generate 口径不同：温度 0 时一个跳过、一个除 1e-5；top_p 一个
    用 cum>p、一个用 cum_prev>p，同样的参数会得到不同的候选集）。
    以后改采样规则时，两处必须一起改。
    """
    # 重复惩罚：窗口 = 「prompt + 已生成」的最后 64 个 token（同 generate）
    if repetition_penalty and repetition_penalty != 1.0 and recent_ids:
        for t in set(int(x) for x in recent_ids):
            if logits[0, t] > 0:
                logits[0, t] = logits[0, t] / repetition_penalty
            else:
                logits[0, t] = logits[0, t] * repetition_penalty
    # 温度：generate 是 `temperature != 1.0`（0 也走 max(1e-5)），不是 `> 0`
    if temperature is not None and temperature != 1.0:
        logits = logits / max(1e-5, float(temperature))
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[:, [-1]]
        logits = torch.where(logits < thresh,
                             torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0 < top_p < 1.0:
        srt, srt_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(srt, dim=-1)
        cum = probs.cumsum(dim=-1)
        cut = cum > top_p
        cut[:, 0] = False               # 至少保留概率最高的一个
        srt = srt.masked_fill(cut, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, srt_idx, srt)
    return logits


def _generate_stream(model, tok, idx, n_new, temperature, top_k, top_p,
                     stop_ids, repetition_penalty, prompt_len):
    """流式生成。自己写循环而不用 model.generate，是为了能在
    生成途中根据「有没有出现停止词」提前退出 —— 提前退出能省一半时间。"""
    was_training = model.training
    model.eval()
    cfg = model.cfg
    B = idx.shape[0]
    dtype = next(model.parameters()).dtype
    cache = [
        {"k": torch.empty(B, cfg["n_kv_head"], 0, cfg["head_dim"],
                          device=idx.device, dtype=dtype),
         "v": torch.empty(B, cfg["n_kv_head"], 0, cfg["head_dim"],
                          device=idx.device, dtype=dtype)}
        for _ in range(cfg["n_layer"])
    ]
    out_ids = []
    n_past = 0
    printed = 0
    stop_text = None
    specials = _special_ids(tok)
    with torch.no_grad():
        for _ in range(n_new):
            if n_past == 0:
                inp = idx[:, -cfg["block_size"]:]
            else:
                inp = idx[:, -1:]
            logits, _, _ = model(inp, cache=cache, cache_pos=n_past,
                                 collect_aux=False)
            n_past += inp.shape[1]
            # 与 model.generate 一致：位置到 block_size 就停，避免在训练
            # 分布之外的位置继续生成（旧版没有这道保护，会长出 block_size
            # 之后的 token，质量不可控）。
            if n_past >= cfg["block_size"]:
                break
            logits = logits[:, -1, :].float()
            # 重复惩罚 / 温度 / top_k / top_p 全部走 _apply_sampling，
            # 与 model.generate 口径一致（以前是两套手写逻辑，会漂移）。
            logits = _apply_sampling(
                logits, idx[0, -64:].tolist(), temperature, top_k, top_p,
                repetition_penalty)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            t = int(nxt.item())
            # 与 model.generate 一致：先把这个 token 收进序列，再判断 EOS。
            # （decode 会跳过特殊 token，所以显示上没有差别，但序列口径一致。）
            out_ids.append(t)
            idx = torch.cat([idx, nxt], dim=1)
            if t == 2:                       # EOS
                break
            if t in stop_ids:
                break
            # 只把**非特殊 token** 拿去显示。特殊 token（<|im_end|> 之类）
            # 是引擎和模型之间的约定，用户不该在屏幕上看到它们。
            piece = tok.decode([i for i in out_ids if i not in specials])
            # 跨 token 边界的「用户：」停止串：BPE 可能把「用户」「：」切成
            # 两个 token，逐 token 比对会漏停，必须在解码窗口里找。
            cut = _user_marker_cut(piece)
            if cut is not None:
                if cut > printed:
                    sys.stdout.write(piece[printed:cut])
                    sys.stdout.flush()
                stop_text = piece[:cut]
                break
            # 先压住可能是停止串前缀的尾巴：「用户」「\n用」这类半截标记
            # 不能立刻显示（否则屏幕上会闪一下「用户」再消失），等下一个
            # token 到齐再决定是正常文字还是停止串。
            safe = len(piece)
            for marker in ("\n用户：", "用户："):
                for k in range(1, len(marker)):
                    if piece.endswith(marker[:k]):
                        safe = min(safe, len(piece) - k)
            if safe > printed:
                sys.stdout.write(piece[printed:safe])
                sys.stdout.flush()
                printed = safe
    if was_training:
        model.train()
    if stop_text is not None:
        print()
        return stop_text
    final = tok.decode([i for i in out_ids if i not in specials])
    if len(final) > printed:        # 把压住的尾巴补出来
        sys.stdout.write(final[printed:])
        sys.stdout.flush()
    print()
    return final


def run_chat(cfg, ckpt=None, device=None, prompt=None, stream=True,
             json_events=False):
    """交互式对话主循环。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok_path = get(cfg, "data", "tokenizer_path", None)
    data_dir = get(cfg, "data", "data_dir", None)
    if not tok_path and data_dir:
        tok_path = os.path.join(data_dir, "tokenizer.txt")

    # 优先用 SFT 的输出目录找断点 —— 用户问「和它聊天」通常指 SFT 之后的它
    if not ckpt:
        for cand in (get(cfg, "post", "out_dir", "") or "",
                     get(cfg, "train", "out_dir", "") or ""):
            if not cand:
                continue
            for name in ("best_sft.pt", "ckpt_sft.pt", "best.pt", "ckpt.pt"):
                p = os.path.join(cand, name)
                if os.path.isfile(p):
                    ckpt = p
                    break
            if ckpt:
                break
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(
            "找不到要聊天的断点（找过 post.out_dir 和 train.out_dir）。\n"
            "聊天需要一个已经训练过的模型，完整顺序是：\n"
            "  1) 先预训练： lingyu train --config configs/small.ini\n"
            "  2) 再做 SFT： lingyu sft   --config configs/sft.ini\n"
            "  3) 然后聊天： lingyu chat  --config configs/sft.ini\n"
            "如果已经有断点，请用 --ckpt 明确指定，例如：\n"
            "  lingyu chat --config configs/sft.ini --ckpt out_sft/ckpt_sft.pt")

    model, tok, ck = load_model(ckpt, device=device, tokenizer_path=tok_path)

    # 断点里记着当年那份分词器的指纹；和现在加载的对不上 -> 立刻停。
    # 这一步不做的话，同词表大小但 ID 不同的分词器会「静默加载成功」，
    # 输出全是乱码，用户完全不知道为什么。
    check_ckpt_tokenizer(ck, tok, where="对话断点")

    # 对话模板优先用断点里记的（训练时用的是哪个就用哪个）——
    # 用错模板模型会「答非所问」，因为它不认识你给的角色标记。
    template = ck.get("chat_template") or get(cfg, "post", "template", "plain") or "plain"
    template = template.lower()
    check_template_usable(template, tok)

    system = get(cfg, "post", "system_prompt", "") or None
    temperature = float(get(cfg, "sample", "temperature", 0.8) or 0.8)
    top_k = int(get(cfg, "sample", "top_k", 40) or 40)
    top_p = float(get(cfg, "sample", "top_p", 0.95) or 0.95)
    max_new = int(get(cfg, "sample", "max_new_tokens", 256) or 256)
    rp = float(get(cfg, "sample", "repetition_penalty", 1.1) or 1.1)

    utils.title("灵语对话")
    utils.info("断点      : %s" % ckpt)
    utils.info("模板      : %s" % template)
    utils.info("参数      : 温度 %.2f   top_k %d   top_p %.2f   最多 %d token"
               % (temperature, top_k, top_p, max_new))
    if ck.get("chat_template") is None and not ck.get("step"):
        utils.warn("这个断点看起来没做过 SFT —— 它只会续写，不会回答问题。")
        utils.info("先跑： lingyu sft --config <配置>")
    utils.emit_event("start", task="chat", ckpt=ckpt, template=template,
                     temperature=temperature, top_k=top_k, top_p=top_p,
                     max_new_tokens=max_new)
    if not json_events:
        print()
        utils.info("直接打字提问即可。输入 退出 / exit / :q 结束。输入 清空 重置对话。")
        print()

    history = []
    if prompt:
        history.append({"role": "user", "content": prompt})

    while True:
        if prompt:
            q, prompt = prompt, None
            if json_events:
                sys.stderr.write("你 > %s\n" % q)
            else:
                print("你 > %s" % q)
        else:
            try:
                if json_events:
                    sys.stderr.write("你 > ")
                    sys.stderr.flush()
                q = input("" if json_events else "你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not q:
            continue
        if q.lower() in EXIT_WORDS:
            break
        if q in ("清空", "clear", "reset"):
            history = []
            utils.ok("对话已清空")
            continue

        history.append({"role": "user", "content": q})
        if not json_events:
            sys.stdout.write("灵语 > ")
            sys.stdout.flush()
        try:
            reply, _ = chat_once(model, tok, history, template, device,
                                 temperature=temperature, top_k=top_k,
                                 top_p=top_p, max_new_tokens=max_new,
                                 repetition_penalty=rp, system=system,
                                 stream=(stream and not json_events))
        except KeyboardInterrupt:
            print()
            utils.warn("这轮生成被打断")
            history.pop()
            continue
        if not reply.strip():
            utils.warn("（模型没说话 —— 多训一会儿 SFT 通常就好了）")
            reply = ""
        history.append({"role": "assistant", "content": reply})
        utils.emit_event("reply", task="chat", question=q, text=reply)
    utils.emit_event("done", task="chat")
    utils.info("再见。")
    return 0
