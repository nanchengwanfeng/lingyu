# -*- coding: utf-8 -*-
'''用训练好的模型写中文。

采样（sampling）就是在每一步：
    模型给每个 token 一个分数 → 按分数变成概率 → 从概率里抽一个

四个旋钮决定「写得野还是稳」：

    temperature（温度）
        除以温度后再 softmax。温度 < 1 让分布更尖（稳、但容易重复），
        > 1 让分布更平（新颖、但容易跑偏）。0.7~1.0 比较合适。

    top_k
        只在概率最高的 k 个候选里抽，直接砍掉长尾里的垃圾 token。

    top_p（核采样）
        按概率从高到低累加，累积到 p 就停下，只在这批里抽。
        比 top_k 更自适应：模型有把握时候选少，没把握时候选多。

    repetition_penalty（重复惩罚）
        把已经出现过的 token 概率压低。中文小模型很容易陷入
        「一直重复同一句」，1.05~1.2 能明显缓解。

真正的采样循环在 model.py 的 LingYu.generate() 里，本模块只负责
「加载模型 + 处理提示 + 显示结果」。
'''

import os
import time

import torch

from . import utils
from .model import build_model
from .tokenizer import load_tokenizer


def load_model(ckpt_path, device="cuda", tokenizer_path=None):
    '''从断点加载模型与分词器。返回 (model, tokenizer, checkpoint)。'''
    from .train import load_checkpoint
    ck = load_checkpoint(ckpt_path, map_location="cpu")
    mcfg = dict(ck.get("config") or {})
    tp = tokenizer_path or ck.get("tokenizer_path")
    if not tp or not os.path.isfile(tp):
        cand = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "tokenizer.txt")
        if os.path.isfile(cand):
            tp = cand
    if not tp or not os.path.isfile(tp):
        raise FileNotFoundError(
            "找不到分词器。断点里记的是 %r。\n"
            "用 --config 指定配置文件，或把 tokenizer.txt 放到断点同目录。"
            % (ck.get("tokenizer_path"),))
    tok = load_tokenizer(tp)
    mcfg["vocab_size"] = tok.vocab_size
    model = build_model(mcfg)
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model, tok, ck


def generate_text(model, tok, prompt="", max_new_tokens=200, temperature=0.9,
                  top_k=40, top_p=0.95, seed=None, device="cuda",
                  repetition_penalty=1.05, stream=True, warmup=True):
    '''生成一段文本。stream=True 时逐字打印（像打字机）。

    warmup：先跑几个 token 把 CUDA 内核编译/缓存热起来。
    否则第一次调用会把「编译内核」的几秒算进生成速度里，
    报出一个严重偏低的 token/秒（实测：冷启动 3.8，热起来 113.9）。
    '''
    if seed is not None:
        torch.manual_seed(int(seed))
    ids = tok.encode(prompt) if prompt else [1]
    if not ids:
        ids = [1]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    if stream and prompt:
        print(prompt, end="", flush=True)

    if warmup:
        with torch.no_grad():
            model.generate(idx, max_new_tokens=2, temperature=1.0, use_cache=True)

    t0 = time.time()
    out = model.generate(
        idx, max_new_tokens=int(max_new_tokens),
        temperature=temperature, top_k=top_k, top_p=top_p,
        eos_id=None, use_cache=True, repetition_penalty=repetition_penalty)
    dt = time.time() - t0
    new_ids = out[0].tolist()[len(ids):]

    if stream:
        # 分小段打印，看起来像逐字生成
        step = max(1, len(new_ids) // 120)
        for i in range(0, len(new_ids), step):
            print(tok.decode(new_ids[i:i + step]), end="", flush=True)
        print()
    else:
        # 不逐字打印时，也要把结果完整打出来（否则等于什么都没看到）
        print(tok.decode(out[0].tolist()))
        print()
    if new_ids:
        utils.info("（生成 %d 个 token，耗时 %.2f 秒，%.1f token/秒）"
                   % (len(new_ids), dt, len(new_ids) / max(1e-6, dt)))
    return tok.decode(out[0].tolist())
