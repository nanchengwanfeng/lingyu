# -*- coding: utf-8 -*-
"""模型本体：一个从零手写的 Decoder-only Transformer，支持稠密与 MoE 两种前馈。

结构（自下而上）：

    token 嵌入  ──┐
                  ├─ 每层做两件事：先注意力（看别的 token），再前馈（自己想）
    位置编码(RoPE)┘

    每一层：
        x = x + 注意力(RMSNorm(x))        # 注意力：GQA 分组查询
        x = x + 前馈(RMSNorm(x))          # 前馈：稠密 SwiGLU  或  MoE 稀疏专家

    最后 RMSNorm + 线性层 -> 预测下一个 token

四个关键设计（都有原因，不是抄来的）：

1. **RMSNorm 前置（pre-norm）**
   归一化放在子层入口，训练更稳，小模型也不容易炸。

2. **RoPE 旋转位置编码**
   把位置信息「旋转」进 Q/K 向量里，不额外加参数，还能外推到更长上下文。

3. **GQA 分组查询注意力（KV 头数可配）**
   Q 头照旧，K/V 头变少，多个 Q 头共享一组 K/V。
   KV cache 大小 = 2 × 层数 × KV头数 × head_dim × 序列长 × 2字节
   把 n_kv_head 从 8 改成 2，cache 立刻省 4 倍，质量损失很小。

4. **SwiGLU 前馈 + 可选 MoE**
   稠密：两层 SwiGLU。MoE：一批「专家」+ 一个路由器，每个 token 只走其中几个，
   所以总参数可以很大，但每个 token 的实际计算量（激活参数）很小。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import components
from .components import (get_attention, get_ffn, get_norm, get_pos,
                         norm_dim_mult)
from typing import Protocol, runtime_checkable

from .registry import register


# ==========================================================================


class RMSNorm(nn.Module):
    """均方根归一化。比 LayerNorm 少一个「减均值」，更快，效果一样好。

    x 先除以自身的均方根（把数值缩放到统一尺度），再乘一个可学习的缩放向量。
    """

    PARAM_DIM_MULT = 1          # 只有 weight，参数量 = 特征维度

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class LayerNorm(nn.Module):
    """经典 LayerNorm。

    和 RMSNorm 的区别：多了「减均值」这一步（中心化）。
    RMSNorm 更快且效果相当，所以是本项目默认；这里提供 LayerNorm
    是为了让你能一行配置切换、亲手对比两者差异。
    """

    PARAM_DIM_MULT = 2          # weight + bias，参数量 = 2 × 特征维度

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (x * self.weight.float() + self.bias.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码。

    做法：把每个头的向量两两配对，按位置 t 旋转一个角度 t*theta_i。
    两个 token 做点积时，角度差自然编码了它们的「相对距离」，
    所以模型学的是相对位置，长度外推也更友好。

    这里按需缓存 cos/sin 表，序列变长就自动扩表（不用重新初始化模型）。
    """

    def __init__(self, head_dim, max_seq=8192, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "每个头的维度必须是偶数，否则没法两两配对"
        self.head_dim = head_dim
        self.base = base
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cos = None
        self._sin = None
        self._cached_len = 0
        self._build(max_seq)

    def _build(self, length, device=None, dtype=torch.float32):
        if device is None:
            device = self.inv_freq.device
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))     # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)              # (T, head_dim)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cached_len = length

    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._cached_len < seq_len or self._cos.device != device:
            need = max(seq_len, 64, self._cached_len * 2)
            self._build(need, device=device, dtype=torch.float32)
        cos = self._cos[:seq_len].to(dtype)
        sin = self._sin[:seq_len].to(dtype)
        return cos, sin


def _rotate_half(x):
    """把向量的前后两半交换并给后半取负 —— 旋转操作的一半。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin, offset=0):
    """把 RoPE 施加到 q, k 上。形状均为 (B, n_head, T, head_dim)。"""
    T = q.shape[-2]
    cos = cos[offset:offset + T].unsqueeze(0).unsqueeze(0)   # (1,1,T,D)
    sin = sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nn.Module):
    """多头 / 分组查询注意力，带 KV cache。

    **只注册一次，别加子类** —— 这个类按 n_head 与 n_kv_head 自动变成
    MHA / GQA / MQA，三者是同一份代码。加子类只会让人以为要分三处改。
    注册放在 `components.py` 里统一做。

    想写自己的注意力，请新建文件并在 plugins/ 里注册，不要改这里。
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.n_kv_head = cfg["n_kv_head"]
        self.head_dim = cfg["head_dim"]
        self.dropout = cfg["dropout"]
        self.bias = cfg["bias"]

        d = cfg["d_model"]
        self.q_proj = nn.Linear(d, self.n_head * self.head_dim, bias=self.bias)
        self.k_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=self.bias)
        self.v_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=self.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, d, bias=self.bias)

        self.rope = None          # 由 LingYu 注入（RoPE 表所有层共用）
        self._mask_cache = {}

    def _causal_mask(self, T, offset, device):
        """返回允许掩码（True = 允许被 attend），形状 (T, T + offset)。

        query i 的绝对位置是 offset + i，它只能看绝对位置 <= offset + i 的 key；
        offset == 0 时退化成标准下三角。
        只有「cache 里已有历史 + 一次喂多个 token（chunked prefill）」才需要它：
        此时 SDPA 的 is_causal 会按左上角对齐造方阵掩码，和真实偏移差 offset。
        """
        key = (T, offset, device.type)
        m = self._mask_cache.get(key)
        if m is None:
            total = T + offset
            m = torch.ones(T, total, device=device, dtype=torch.bool).tril(diagonal=offset)
            self._mask_cache[key] = m
        return m

    def _attend(self, q, k_all, v_all, T, offset):
        """注意力的统一出口（SDPA 调用只应出现在这里）。

        这是给**插件**留的扩展点：子类覆写这个方法就能换注意力内核
        （例如局部窗口、稀疏注意力），不用碰前向的 cache/RoPE 逻辑。
        示例见 `examples/plugins/attention_local_window.py`。

        默认实现 = 标准因果注意力：
          - 分块预填充（cache 里已有历史、一次喂多个 token）用显式矩形掩码；
          - 否则 `is_causal` 就够；T==1 时没有未来 token，不需要掩码。
        """
        if offset > 0 and T > 1:
            return F.scaled_dot_product_attention(
                q, k_all, v_all,
                attn_mask=self._causal_mask(T, offset, q.device),
                is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=(self.n_head != self.n_kv_head))
        return F.scaled_dot_product_attention(
            q, k_all, v_all,
            is_causal=(T > 1),
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=(self.n_head != self.n_kv_head))

    def forward(self, x, cache=None, cache_pos=0):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(T + cache_pos, x.device, x.dtype)
            q, k = apply_rope(q, k, cos, sin, offset=cache_pos)

        if cache is not None:
            q8 = bool(cache.get("int8"))
            if "pos" in cache:
                # 预分配 buffer（generate 用）：写入新 K/V，避免每步全量 torch.cat 复制。
                # buffer 形状 (B, n_kv_head, cap, head_dim)，cap >= 本层最多会缓存的 token 数。
                pos = int(cache["pos"])
                need = pos + T
                cap = cache["k"].shape[2]
                if need > cap:                     # 外部给太小 → 兜底扩容
                    grow = max(need, cap * 2)
                    nk = cache["k"].new_empty(B, self.n_kv_head, grow, self.head_dim)
                    nv = cache["v"].new_empty(B, self.n_kv_head, grow, self.head_dim)
                    nk[:, :, :pos] = cache["k"][:, :, :pos]
                    nv[:, :, :pos] = cache["v"][:, :, :pos]
                    cache["k"], cache["v"] = nk, nv
                    if q8:
                        sk = cache["k_scale"].new_empty(B, self.n_kv_head, grow)
                        sv = cache["v_scale"].new_empty(B, self.n_kv_head, grow)
                        sk[:, :, :pos] = cache["k_scale"][:, :, :pos]
                        sv[:, :, :pos] = cache["v_scale"][:, :, :pos]
                        cache["k_scale"], cache["v_scale"] = sk, sv
                if q8:
                    # KIVI（arXiv:2402.02750）的简化版：K/V 都按「每个 token 一个
                    # per-head absmax scale」量化到 int8。比 bf16 省一半，
                    # 误差集中在极少数离群值上，生成质量几乎不变。
                    ks = k.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
                    vs = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
                    cache["k"][:, :, pos:need] = torch.round(k / ks).clamp(-127, 127).to(torch.int8)
                    cache["v"][:, :, pos:need] = torch.round(v / vs).clamp(-127, 127).to(torch.int8)
                    cache["k_scale"][:, :, pos:need] = ks.squeeze(-1)
                    cache["v_scale"][:, :, pos:need] = vs.squeeze(-1)
                    k_all = cache["k"][:, :, :need].to(x.dtype) * \
                        cache["k_scale"][:, :, :need].to(x.dtype).unsqueeze(-1)
                    v_all = cache["v"][:, :, :need].to(x.dtype) * \
                        cache["v_scale"][:, :, :need].to(x.dtype).unsqueeze(-1)
                else:
                    cache["k"][:, :, pos:need] = k
                    cache["v"][:, :, pos:need] = v
                    k_all = cache["k"][:, :, :need]
                    v_all = cache["v"][:, :, :need]
                    if k_all.dtype != x.dtype:
                        k_all = k_all.to(x.dtype)
                        v_all = v_all.to(x.dtype)
                cache["pos"] = need
            else:
                # 兼容旧接口（chat.py 等外部调用方）：逐层 cat 追加历史
                pos = cache["k"].shape[2]
                if q8:
                    prev_k = (cache["k"].to(x.dtype) * cache["k_scale"].to(x.dtype).unsqueeze(-1)
                              if pos else cache["k"].to(x.dtype))
                    prev_v = (cache["v"].to(x.dtype) * cache["v_scale"].to(x.dtype).unsqueeze(-1)
                              if pos else cache["v"].to(x.dtype))
                    k_all = torch.cat([prev_k, k], dim=2)
                    v_all = torch.cat([prev_v, v], dim=2)
                    ks = k_all.abs().amax(dim=-1).clamp_min(1e-8) / 127.0
                    vs = v_all.abs().amax(dim=-1).clamp_min(1e-8) / 127.0
                    cache["k"] = torch.round(k_all / ks.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
                    cache["v"] = torch.round(v_all / vs.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
                    cache["k_scale"], cache["v_scale"] = ks, vs
                else:
                    k_all = torch.cat([cache["k"], k], dim=2)
                    v_all = torch.cat([cache["v"], v], dim=2)
                    cache["k"], cache["v"] = k_all, v_all
            if pos > 0 and T > 1:
                # 分块预填充：query 的绝对位置从 pos 开始，_attend 会显式给
                # 矩形掩码（is_causal 只对「左上角对齐的方阵」成立，这里用会错位）。
                out = self._attend(q, k_all, v_all, T, pos)
            else:
                # T == 1 时只有一个 query、能看到全部历史；T > 1 且 cache 为空
                # 时标准因果掩码就是对的。
                out = self._attend(q, k_all, v_all, T, pos)
        else:
            out = self._attend(q, k, v, T, 0)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(out)


def apply_repetition_penalty(logits, idx, penalty):
    """对 batch 里**每一行**已出现过的 token 施加重复惩罚（就地语义，返回新张量）。

    旧实现只处理 idx[0]，batch>1 时其余行完全不惩罚；这里改成 gather/scatter
    向量化，按行各看各的最近 64 个 token。
    """
    if penalty is None or float(penalty) == 1.0 or idx.numel() == 0:
        return logits
    window = idx[:, -64:].clamp(0, logits.size(-1) - 1)
    prev = logits.gather(1, window)
    prev = torch.where(prev > 0, prev / penalty, prev * penalty)
    return logits.scatter(1, window, prev)


class SwiGLU(nn.Module):
    """SwiGLU 前馈：三个矩阵而不是两个。

        gate = silu(W1 x)
        out  = W3( gate * (W2 x) )

    多出来的那条路径让「一个通道可以控制另一个通道的通过量」，
    表达力更强。为了参数量和普通 4× 前馈持平，中间维度取约 8/3 倍。
    """

    def __init__(self, d_model, d_ff, bias=False, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MoE(nn.Module):
    """稀疏专家前馈（DeepSeekMoE 风格：细粒度专家 + 共享专家）。

    每个 token 会：
      1. 被路由器打一个分（对每个专家算一个 logit）
      2. 只挑分数最高的 n_expert_per_tok 个专家来算（top-k）
      3. 按路由概率加权求和

    为什么要有「共享专家」：
        有些知识是所有 token 都需要的（比如基本语法）。如果全靠路由，
        每个专家都得学一遍，浪费容量。留 1 个永远激活的共享专家专门装这些。

    两个辅助损失（只在训练时用）：
        aux loss（负载均衡）：惩罚「某些专家被冷落」，系数默认 0.01
        z loss（路由器稳定）：惩罚路由 logit 过大，系数默认 0.001
    实测教训（论文 arXiv 2605.13769）：小模型上朴素 top-1 不做均衡会立刻坍缩，
        top-2 是质量提升的主要来源。
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.n_expert = cfg["n_expert"]
        self.top_k = min(cfg["n_expert_per_tok"], cfg["n_expert"])
        self.n_shared = cfg["n_shared_expert"]
        self.dropout = cfg["dropout"]
        self.bias = cfg["bias"]
        self.jitter = cfg["moe_router_jitter"]

        # 专家做成一个「打包」的参数张量：比 nn.ModuleList 更省显存、更好循环
        e_ff = cfg["expert_d_ff"]
        self.e_ff = e_ff
        self.w1 = nn.Parameter(torch.empty(self.n_expert, d, e_ff))
        self.w2 = nn.Parameter(torch.empty(self.n_expert, d, e_ff))
        self.w3 = nn.Parameter(torch.empty(self.n_expert, e_ff, d))

        self.router = nn.Linear(d, self.n_expert, bias=False)

        if self.n_shared > 0:
            s_ff = cfg["d_ff"]
            self.s_w1 = nn.Parameter(torch.empty(self.n_shared, d, s_ff))
            self.s_w2 = nn.Parameter(torch.empty(self.n_shared, d, s_ff))
            self.s_w3 = nn.Parameter(torch.empty(self.n_shared, s_ff, d))
            self.s_ff = s_ff
            if cfg["shared_expert_gate"]:
                self.s_gate = nn.Linear(d, 1, bias=False)

        self.reset_parameters(cfg)
        self.drop = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        # 最近一次前向的辅助损失与统计，训练循环直接读
        self.last_aux_loss = None
        self.last_z_loss = None
        self.last_load = None

    def reset_parameters(self, cfg):
        std = cfg["init_std"]
        for p in (self.w1, self.w2, self.w3):
            nn.init.normal_(p, mean=0.0, std=std)
        nn.init.normal_(self.router.weight, mean=0.0, std=std)
        if self.n_shared > 0:
            for p in (self.s_w1, self.s_w2, self.s_w3):
                nn.init.normal_(p, mean=0.0, std=std)
            if cfg["shared_expert_gate"]:
                nn.init.zeros_(self.s_gate.weight)

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        N = flat.shape[0]

        logits = self.router(flat)                       # (N, E)
        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter

        # z loss：约束路由 logit 不要太大（数值稳定 + 防止过早固化）
        self.last_z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

        probs = F.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)      # (N, k)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # aux loss：Switch Transformer 的负载均衡损失
        #   想让「每个专家分到的 token 比例」和「路由器给它的平均概率」都均匀
        if self.training:
            tokens_per_expert = torch.zeros(self.n_expert, device=x.device, dtype=torch.float32)
            ones = torch.ones(N, dtype=torch.float32, device=x.device)
            tokens_per_expert.scatter_add_(0, topi.reshape(-1), ones.repeat_interleave(self.top_k))
            f = tokens_per_expert / (N * self.top_k)
            P = probs.mean(dim=0)
            self.last_aux_loss = self.n_expert * torch.sum(f * P)
            self.last_load = (tokens_per_expert / max(1, N)).detach()
        else:
            self.last_aux_loss = None

        # 每个 token 在各个专家上的总路由权重：一次散点算完，
        # 避免在专家循环里反复做布尔比较和压缩（那是 8 倍的无用功）。
        #   topi (N,k) 里的专家编号 -> 把对应的 topv 累加到 (N,E) 的第 E 维上
        w_all = torch.zeros(N, self.n_expert, device=x.device, dtype=topv.dtype)
        w_all.scatter_add_(1, topi, topv)

        # 先统计每个专家分到多少 token，空的直接跳过（不做任何张量运算）
        counts = w_all.gt(0).sum(dim=0).tolist()

        out = torch.zeros_like(flat)
        for e in range(self.n_expert):
            if counts[e] == 0:
                continue
            rows = (w_all[:, e] > 0).nonzero(as_tuple=True)[0]
            w = w_all.index_select(0, rows)[:, e]
            xe = flat.index_select(0, rows)
            he = F.silu(xe @ self.w1[e]) * (xe @ self.w2[e])
            ye = (he @ self.w3[e]) * w.unsqueeze(-1).to(xe.dtype)
            out.index_add_(0, rows, ye)

        if self.n_shared > 0:
            hs = 0
            for i in range(self.n_shared):
                hs = hs + F.silu(flat @ self.s_w1[i]) * (flat @ self.s_w2[i])
            ys = hs @ self.s_w3[0]
            if hasattr(self, "s_gate"):
                ys = ys * torch.sigmoid(self.s_gate(flat))
            out = out + ys

        return self.drop(out.view(B, T, D))


class Block(nn.Module):
    """一层 Transformer：注意力 + 前馈，都带残差。

    **两个子层都是从注册表取的** —— 换实现只要改配置里的名字，
    不用动这个类：

        [model]
        attention = gqa        # 或 mha / mqa / 你自己的插件名
        ffn       = swiglu     # 或 moe / 你自己的
        norm      = rmsnorm    # 或 layernorm
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg_ref = cfg
        d = cfg["d_model"]

        # 归一化（注册表）
        NormImpl = get_norm(cfg)
        self.norm_name = (cfg.get("norm") or "rmsnorm").lower()
        self.ln1 = NormImpl(d, cfg["norm_eps"])
        self.ln2 = NormImpl(d, cfg["norm_eps"])

        # 注意力（注册表）
        AttnImpl = get_attention(cfg)
        self.attn_name = (cfg.get("attention") or "gqa").lower()
        self.attn = AttnImpl(cfg, layer_idx)

        # 前馈（注册表）。moe_every 控制隔几层放一个 MoE 层。
        FFNImpl = get_ffn(cfg)
        self.ffn_name = (cfg.get("ffn") or
                         ("moe" if cfg.get("use_moe") else "swiglu")).lower()
        use_moe_here = (self.ffn_name == "moe"
                        and cfg["moe_every"] > 0
                        and (layer_idx % cfg["moe_every"] == 0))
        if use_moe_here:
            self.mlp = FFNImpl(cfg)
            self.is_moe = True
        elif self.ffn_name == "moe":
            # 这层轮流到不放 MoE，退回全宽前馈
            self.mlp = SwiGLU(d, cfg["d_ff"], bias=cfg["bias"], dropout=cfg["dropout"])
            self.is_moe = False
        else:
            # 自定义前馈的统一签名约定是 FFNImpl(cfg)；内置 SwiGLU 是
            # SwiGLU(d_model, d_ff, ...)，所以这里分两路。
            if FFNImpl is SwiGLU:
                self.mlp = SwiGLU(d, cfg["d_ff"], bias=cfg["bias"],
                                  dropout=cfg["dropout"])
            else:
                self.mlp = FFNImpl(cfg)
            self.is_moe = False

    def forward(self, x, cache=None, cache_pos=0):
        # cache 传进来的是「本层自己的那一份」KV 缓存（由 LingYu.forward 按下标切好）
        x = x + self.attn(self.ln1(x), cache=cache, cache_pos=cache_pos)
        x = x + self.mlp(self.ln2(x))
        return x

    def attn_param_count(self):
        cfg = self.cfg_ref
        d = cfg["d_model"]
        hd = cfg["head_dim"]
        n = d * (cfg["n_head"] * hd) + 2 * d * (cfg["n_kv_head"] * hd) + d * (cfg["n_head"] * hd)
        if cfg["bias"]:
            # Q / O 各 d 个 bias，K / V 各 n_kv_head*head_dim 个。
            # 旧写法多算了 d（实测 49,536 vs 解析 49,664），已修。
            n += 2 * d + 2 * (cfg["n_kv_head"] * hd)
        return n

    def mlp_param_count(self, active=False):
        cfg = self.cfg_ref
        d = cfg["d_model"]
        if not self.is_moe:
            n = 3 * d * cfg["d_ff"]
            if cfg["bias"]:
                n += 2 * cfg["d_ff"] + d
            return n
        e = cfg["expert_d_ff"]
        n_experts = self.mlp.top_k if active else cfg["n_expert"]
        n = n_experts * (3 * d * e) + d * cfg["n_expert"]      # 专家 + 路由器
        if cfg["n_shared_expert"] > 0:
            n += cfg["n_shared_expert"] * (3 * d * cfg["d_ff"])
            if cfg["shared_expert_gate"]:
                n += d
        return n


class LingYu(nn.Module):
    """整个模型。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        V = cfg["vocab_size"]

        self.tok_emb = nn.Embedding(V, d)
        self.drop = nn.Dropout(cfg["dropout"]) if cfg["dropout"] > 0 else nn.Identity()
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg["n_layer"])])
        self.norm_name = (cfg.get("norm") or "rmsnorm").lower()
        NormImpl = get_norm(cfg)
        # 末尾归一化必须跟 Block 里用**同一个** NormImpl。
        # （曾经这里写死 RMSNorm，于是 model.norm=layernorm 时最后这一层
        #   悄悄还是 RMSNorm —— 配置看着生效了，其实只有一半生效。）
        self.ln_f = NormImpl(d, cfg["norm_eps"])
        self.lm_head = nn.Linear(d, V, bias=False)
        if cfg["tie_weights"]:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE 表所有层共用一份，但要作为 nn.Module 注册进来，
        # 这样 model.to(device) / state_dict 才会正确带上它
        if cfg["use_rope"]:
            self.rope = RotaryEmbedding(cfg["head_dim"],
                                        max_seq=max(8192, cfg["block_size"]),
                                        base=float(cfg.get("rope_base", 10000.0)))
        else:
            self.rope = None
        if self.rope is not None:
            for blk in self.blocks:
                blk.attn.rope = self.rope

        if cfg["use_learned_pos"]:
            self.pos_emb = nn.Embedding(cfg["block_size"], d)
        else:
            self.pos_emb = None

        self.apply(self._init)
        # 残差投影按 1/sqrt(2*n_layer) 缩小，深层残差累加才不会爆炸
        scale = cfg["residual_std"] / math.sqrt(2 * max(1, cfg["n_layer"]))
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w3.weight")                     or name.endswith("lm_head.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

        self._cache = None
        # 由训练循环按 [train] grad_checkpoint 打开；生成时永远是关的。
        self.grad_checkpoint = False
        # 由训练循环按 [train] loss_chunk 设置；0 = 不分块。
        self.loss_chunk = 0

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg["init_std"])
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg["init_std"])

    # ------------------------------------------------------------ 前向
    def forward(self, idx, targets=None, cache=None, cache_pos=0, collect_aux=True):
        B, T = idx.shape
        cfg = self.cfg

        x = self.tok_emb(idx)

        # 学习式位置嵌入（可选，默认关；开了就没法外推长度）
        if self.pos_emb is not None:
            pos = torch.arange(cache_pos, cache_pos + T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        # cache 是一个「每层一份」的列表；这里按层号切好再传下去。
        # （之前直接把整个列表传进每一层，导致 cache["k"] 报
        #   "list indices must be integers or slices, not str"）
        for i, blk in enumerate(self.blocks):
            sub = cache[i] if cache is not None else None
            if (self.grad_checkpoint and self.training and cache is None
                    and torch.is_grad_enabled()):
                # 激活重计算（Chen et al. 2016, arXiv:1604.06174）：前向只留
                # 每层输入，反向时重算中间激活 —— 用算力换显存。
                x = torch.utils.checkpoint.checkpoint(
                    lambda t, _b=blk, _s=sub: _b(t, cache=_s, cache_pos=cache_pos),
                    x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = blk(x, cache=sub, cache_pos=cache_pos)

        x = self.ln_f(x)

        if targets is None:
            # 生成时只需要最后一个位置的 logits，省算力
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            # 分块交叉熵：logits (B,T,V) 一次算 softmax 会再吃一份同样大的中间张量。
            # 长上下文（T 大）时这是显存大头，按 token 行分块算 sum 再平均，
            # 数学上等价，峰值显存降到 chunk 行的大小。
            chunk = int(getattr(self, "loss_chunk", 0) or 0)
            if chunk and targets.numel() > chunk:
                flat = logits.reshape(-1, logits.size(-1))
                tg = targets.reshape(-1)
                total = tg.ne(-1).sum().clamp(min=1)
                part = logits.new_zeros(())
                for st in range(0, tg.numel(), chunk):
                    part = part + F.cross_entropy(
                        flat[st:st + chunk], tg[st:st + chunk],
                        ignore_index=-1, reduction="sum")
                loss = part / total
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1)

        aux = None
        if collect_aux and cfg["use_moe"]:
            a = z = 0.0
            loads = []
            for blk in self.blocks:
                if getattr(blk, "is_moe", False):
                    if blk.mlp.last_aux_loss is not None:
                        a = a + blk.mlp.last_aux_loss
                    if blk.mlp.last_z_loss is not None:
                        z = z + blk.mlp.last_z_loss
                    if blk.mlp.last_load is not None:
                        loads.append(blk.mlp.last_load)
            aux = {
                "aux_loss": a if torch.is_tensor(a) else torch.tensor(0.0, device=idx.device),
                "z_loss": z if torch.is_tensor(z) else torch.tensor(0.0, device=idx.device),
                "loads": loads,
            }
        return logits, loss, aux

    # ------------------------------------------------------------ 生成
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 eos_id=None, use_cache=True, forbid_ids=(),
                 repetition_penalty=1.0):
        """自回归生成。返回完整的 token 序列（含输入的 prompt）。

        KV cache 让每步只算 1 个新 token 的 K/V，生成速度比不缓存快几十倍。
        """
        was_training = self.training
        self.eval()
        cfg = self.cfg
        B = idx.shape[0]
        dtype = next(self.parameters()).dtype
        limit = max(2, int(cfg["block_size"]))

        def _fresh_cache():
            if not use_cache:
                return None
            # 预分配整段上下文长度的 buffer：每步只写入 1 个 token，
            # 不再每步 torch.cat 复制全部历史（O(T²) → O(T)）。
            cap = limit
            kvq = str(cfg.get("kv_cache_dtype", "auto") or "auto").lower()
            if kvq not in ("auto", "f32", "bf16", "int8"):
                kvq = "auto"
            q8 = kvq == "int8"
            kd = {"f32": torch.float32, "bf16": torch.bfloat16}.get(kvq, dtype)
            caches = []
            for _ in range(cfg["n_layer"]):
                item = {
                    "k": torch.empty(B, cfg["n_kv_head"], cap, cfg["head_dim"],
                                     device=idx.device,
                                     dtype=torch.int8 if q8 else kd),
                    "v": torch.empty(B, cfg["n_kv_head"], cap, cfg["head_dim"],
                                     device=idx.device,
                                     dtype=torch.int8 if q8 else kd),
                    "pos": 0,
                    "int8": q8,
                }
                if q8:
                    item["k_scale"] = torch.empty(B, cfg["n_kv_head"], cap,
                                                  device=idx.device, dtype=torch.float32)
                    item["v_scale"] = torch.empty(B, cfg["n_kv_head"], cap,
                                                  device=idx.device, dtype=torch.float32)
                caches.append(item)
            return caches

        cache = _fresh_cache()
        n_past = 0
        # 每行是否已经生成过 EOS：出了 EOS 的行不再产出新内容（批量矩形用 EOS 填充）
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            if not use_cache:
                # 不缓存时每步喂完整窗口（截断最老的 token），历史不丢
                inp = idx[:, -limit:]
                logits, _, _ = self(inp, cache=None, cache_pos=0, collect_aux=False)
                n_past = inp.shape[1]
            elif n_past == 0:
                # 第一次：喂整个 prompt（太长就只留最后 block_size 个）
                inp = idx[:, -limit:]
                logits, _, _ = self(inp, cache=cache, cache_pos=0, collect_aux=False)
                n_past = inp.shape[1]
            elif n_past >= limit:
                # 上下文已满：丢掉最老的 token，用最后 limit-1 个 token 重建 cache
                # （窗口滚动，不再直接 break 导致 0 token 生成）
                keep = idx[:, -(limit - 1):]
                cache = _fresh_cache()
                logits, _, _ = self(keep, cache=cache, cache_pos=0, collect_aux=False)
                n_past = keep.shape[1]
            else:
                # 之后每步只喂 1 个新 token，历史全在 cache 里
                inp = idx[:, -1:]
                logits, _, _ = self(inp, cache=cache, cache_pos=n_past, collect_aux=False)
                n_past += 1

            logits = logits[:, -1, :].float()
            if forbid_ids:
                for fid in forbid_ids:
                    logits[:, fid] = -float("inf")
            # 重复惩罚：已出现过的 token 压低概率，缓解「一直重复同一句」。
            # 整批向量化：每行各看自己的最近 64 个 token（旧实现只看第 0 行）。
            if repetition_penalty and repetition_penalty != 1.0 and idx.shape[1] > 0:
                logits = apply_repetition_penalty(logits, idx, repetition_penalty)
            if temperature is not None and temperature != 1.0:
                logits = logits / max(1e-5, temperature)

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)

            if top_p is not None and 0 < top_p < 1.0:
                srt, srt_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(srt, dim=-1)
                cum = probs.cumsum(dim=-1)
                cut = cum > top_p
                cut[:, 0] = False
                srt = srt.masked_fill(cut, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(1, srt_idx, srt)

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if eos_id is not None:
                # 已出过 EOS 的行强制继续吐 EOS（保持 batch 矩形），
                # 保证 EOS 之后不会再有新的内容 token 进输出。
                nxt = torch.where(finished[:, None],
                                  torch.full_like(nxt, int(eos_id)), nxt)
                finished = finished | (nxt[:, 0] == int(eos_id))
            idx = torch.cat([idx, nxt], dim=1)
            # 所有行都结束后立刻停：batch>1 时不再陪着最长的那行空转
            if eos_id is not None and bool(finished.all()):
                break
        if was_training:
            self.train()
        return idx

    # ------------------------------------------------------------ 统计
    def param_groups(self):
        """按模块分组统计参数量，用来回答「参数都花在哪了」。"""
        groups = {}
        for name, p in self.named_parameters():
            if name.startswith("tok_emb"):
                k = "词嵌入 tok_emb"
            elif name.startswith("lm_head"):
                k = "输出层 lm_head"
            elif ".attn." in name:
                k = "注意力 attn"
            elif ".mlp.router" in name:
                k = "MoE 路由器 router"
            elif ".mlp.s_" in name:
                k = "MoE 共享专家"
            elif ".mlp.w1" in name or ".mlp.w2" in name or ".mlp.w3" in name:
                k = "MoE 专家"
            elif ".mlp." in name:
                k = "前馈 SwiGLU"
            elif "ln" in name or "norm" in name:
                k = "归一化 " + (self.norm_name or "norm")
            else:
                k = "其它"
            groups[k] = groups.get(k, 0) + p.numel()
        return groups

    def active_param_count(self):
        """每个 token 实际会用到多少参数。

        稠密模型里它等于总参数；MoE 里它远小于总参数 ——
        这正是稀疏模型的意义：总容量大，但每个 token 只付一部分计算。
        """
        cfg = self.cfg
        d = cfg["d_model"]
        V = cfg["vocab_size"]
        nd = norm_dim_mult(cfg)                      # RMSNorm=1, LayerNorm=2
        n = V * d * (1 if cfg["tie_weights"] else 2)
        n += nd * d                                  # 末尾归一化 ln_f
        if self.pos_emb is not None:
            n += cfg["block_size"] * d
        for blk in self.blocks:
            n += blk.attn_param_count() + blk.mlp_param_count(active=True) + 2 * nd * d
        return n

    # ------------------------------------------------------------ 扩词表
    def resize_vocab(self, new_vocab_size, state_dict=None):
        """扩大（或缩小）词表：tok_emb 和 lm_head 一起改。

        ## 什么时候需要

        SFT 时给分词器加了新 token（比如对话模板的 `<|im_start|>`），
        预训练断点的 embedding 就少几行。这时必须扩，否则加载直接形状不符。

        也可以反向用：先按小词表预训练，SFT 前再扩。

        ## 两个容易踩的点

        1. **tie_weights=True 时 tok_emb 和 lm_head 是同一个张量**。
           只改一个、另一个忘了重新绑定，模型会突然多出一份独立的
           输出矩阵（参数量翻倍，而且加载断点时形状对不上）。
        2. **新增的行是随机初始化的**，不是「自动学会」。扩完必须继续训练，
           否则模型对这几个新 token 的输出完全是噪声 —— 而它偏偏就是
           对话模板要用的标记，表现是「一进对话就胡言乱语」。

        返回新的词表大小。
        """
        old_w = self.tok_emb.weight
        old_v, dim = old_w.shape
        new_v = int(new_vocab_size)
        if new_v == old_v:
            return new_v
        dev, dt = old_w.device, old_w.dtype

        def _grow(src, rows):
            out = torch.empty(rows, src.shape[1], device=dev, dtype=dt)
            nn.init.normal_(out, mean=0.0, std=self.cfg["init_std"])
            n = min(rows, src.shape[0])
            out[:n] = src[:n].detach().to(device=dev, dtype=dt)
            return out

        if state_dict is not None and state_dict.get("tok_emb.weight") is not None:
            src_emb = state_dict["tok_emb.weight"]
        else:
            src_emb = old_w
        new_emb = _grow(src_emb, new_v)

        self.tok_emb = nn.Embedding(new_v, dim, device=dev, dtype=dt)
        self.tok_emb.weight = nn.Parameter(new_emb)

        if self.cfg["tie_weights"]:
            # 关键：重新绑定，让两者继续共享同一个张量
            self.lm_head = nn.Linear(dim, new_v, bias=False, device=dev, dtype=dt)
            self.lm_head.weight = self.tok_emb.weight
        else:
            if state_dict is not None and state_dict.get("lm_head.weight") is not None:
                src_head = state_dict["lm_head.weight"]
            else:
                src_head = self.lm_head.weight
            self.lm_head = nn.Linear(dim, new_v, bias=False, device=dev, dtype=dt)
            self.lm_head.weight = nn.Parameter(_grow(src_head, new_v))

        self.cfg["vocab_size"] = new_v
        return new_v


    # ------------------------------------------------------------ 结构指纹
    def structural_fingerprint(self):
        """续训/加载断点时用来判断「结构变没变」的字段集合。

        与训练循环里的结构一致性检查对齐：换了这些字段里的任何一个，
        旧断点的权重就对不上，必须重新训练（不是调参能救的）。
        """
        return model_fingerprint(self.cfg)


# ==========================================================================
# 模型协议与工厂
#
# 引擎其余部分（训练 / 微调 / 采样 / 计划 / GUI）只依赖下面这份最小接口，
# 不直接 import LingYu。要换模型实现：实现这些方法，注册进注册表，
# 再把配置里的 [model] type 改成你的名字即可。完整教程见 docs/EXTENDING.md。
# ==========================================================================

_FINGERPRINT_KEYS = (
    ("n_layer", "层数"),
    ("d_model", "宽度"),
    ("n_head", "头数"),
    ("n_kv_head", "KV 头数"),
    ("d_ff", "前馈维度"),
    ("block_size", "上下文长度"),
    ("vocab_size", "词表大小"),
    ("tie_weights", "是否共享权重"),
    ("bias", "是否带 bias"),
    ("use_moe", "是否 MoE"),
    ("attention", "注意力实现"),
    ("ffn", "前馈实现"),
    ("norm", "归一化实现"),
    ("pos", "位置编码实现"),
)


def model_fingerprint(model_cfg):
    """把「换了就必须重新训练」的结构字段整理成可比较的字典。"""
    cfg = model_cfg or {}
    return {k: cfg.get(k) for k, _label in _FINGERPRINT_KEYS}


@runtime_checkable
class ModelProtocol(Protocol):
    """模型实现的最小契约（对应训练循环的真实调用）。

    forward(idx, targets=None, cache=None, cache_pos=0, collect_aux=True)
        输入 token id（B, T）。给了 targets 返回 (logits, loss, aux)；
        - loss 必须用 ignore_index=-1 计算（SFT 的 loss mask 依赖它）
        - targets=None 时只需返回最后位置的 logits（生成用），loss/aux 为 None
        - collect_aux=False 允许跳过 MoE 统计
    generate(idx, max_new_tokens, ...)
        自回归续写，返回含 prompt 的完整序列；至少接受 max_new_tokens。
    structural_fingerprint()
        返回 dict，描述「换掉就必须重训」的结构字段。
    """

    def forward(self, idx, targets=None, cache=None, cache_pos=0,
                collect_aux=True):  # pragma: no cover - 协议声明
        ...

    def generate(self, idx, max_new_tokens, **kwargs):  # pragma: no cover
        ...

    def structural_fingerprint(self):  # pragma: no cover
        ...


def build_model(model_cfg, device=None):
    """按 [model] type 从注册表取模型类并实例化。

    model_cfg 是 `config_from_section()` 的产物，可带 ``type``
    （注册表名，默认 ``lingyu``）。device 给 None 时留在 CPU。
    """
    from . import registry as _registry
    cfg = dict(model_cfg or {})
    # 注意：type 不能 pop 掉 —— 断点里保存的 config 就是这份 dict，
    # 丢了 type，自定义模型续训时会被当成默认的 lingyu 重建。
    name = str(cfg.get("type", "") or "lingyu").strip().lower()
    cfg["type"] = name
    cls = _registry.get_or_die("model", name)
    model = cls(cfg)
    if device is not None:
        model = model.to(device)
    return model


def build_config(vocab_size, block_size=512, n_layer=8, n_embd=512, n_head=8,
                 n_kv_head=2, d_ff=None, dropout=0.0, bias=False, tie_weights=True,
                 use_rope=True, use_learned_pos=False, norm_eps=1e-6,
                 init_std=0.02, residual_std=0.02,
                 use_moe=False, n_expert=4, n_expert_per_tok=2,
                 n_shared_expert=1, expert_d_ff=None, moe_every=1,
                 moe_router_jitter=0.0, shared_expert_gate=False,
                 # ---- 注册表：组件选择（空 = 用内置默认）----
                 attention=None, ffn=None, norm=None, pos=None,
                 rope_base=10000.0, kv_cache_dtype="auto"):
    """把散参数整理成模型能吃的字典，顺便做合法性检查。

    nanoGPT 用户最关心的三个旋钮就是 n_layer / n_embd / n_head。

    attention / ffn / norm / pos 是**注册表名字**：填 mha、moe、layernorm
    这类内置名，或你在 plugins/ 里注册的名字。不填就用默认。
    """
    if n_embd % n_head != 0:
        raise ValueError("n_embd (%d) 必须能被 n_head (%d) 整除" % (n_embd, n_head))
    head_dim = n_embd // n_head
    if head_dim % 2 != 0:
        raise ValueError("每个头的维度 = n_embd/n_head = %d 是奇数，RoPE 要求偶数。"
                         "请调整 n_embd 或 n_head（例如 n_embd=384,n_head=6 -> 64）" % head_dim)
    if n_kv_head < 1:
        raise ValueError("n_kv_head 至少为 1")
    if n_kv_head > n_head:
        raise ValueError("n_kv_head (%d) 不能大于 n_head (%d)"
                         "（KV 头是给 Q 头共享的，比 Q 头还多没有意义）"
                         % (n_kv_head, n_head))
    if n_head % n_kv_head != 0:
        raise ValueError("n_head (%d) 必须是 n_kv_head (%d) 的整数倍，"
                         "否则 Q 头没法均分给 KV 头" % (n_head, n_kv_head))

    # ---- 位置编码：`pos` 是唯一真开关（审计 B-08）----
    # 以前 model 只读 use_rope / use_learned_pos，配置里的 `pos` 是个装饰品：
    # 写 `pos = learned` 仍然用 RoPE，而且 GUI 也改不了真正的开关。
    # 现在：显式给了 pos 就按它推导两个布尔值；没给就按旧布尔值反推 pos。
    pos_name = str(pos).strip().lower() if pos else ""
    if pos_name:
        if pos_name not in ("rope", "learned", "none"):
            raise ValueError("[model] pos = %r 不认识，可选：rope / learned / none"
                             % pos)
    else:
        if use_rope and use_learned_pos:
            raise ValueError("[model] use_rope 和 use_learned_pos 不能同时为 true；"
                             "建议改用 pos = rope / learned / none")
        if use_learned_pos:
            pos_name = "learned"
        elif use_rope:
            pos_name = "rope"
        else:
            pos_name = "none"
    use_rope = (pos_name == "rope")
    use_learned_pos = (pos_name == "learned")

    if d_ff is None:
        d_ff = int(round(8 * n_embd / 3 / 8) * 8)
    if expert_d_ff is None:
        # 专家做细粒度：每个专家大约是全宽前馈的 1/2
        expert_d_ff = max(8, int(round(d_ff / 2 / 8) * 8))

    cfg = dict(
        vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
        d_model=n_embd, n_head=n_head, n_kv_head=n_kv_head, head_dim=head_dim,
        d_ff=d_ff, dropout=dropout, bias=bias, tie_weights=tie_weights,
        use_rope=use_rope, use_learned_pos=use_learned_pos, norm_eps=norm_eps,
        init_std=init_std, residual_std=residual_std,
        use_moe=use_moe, n_expert=n_expert, n_expert_per_tok=n_expert_per_tok,
        n_shared_expert=n_shared_expert, expert_d_ff=expert_d_ff,
        moe_every=moe_every, moe_router_jitter=moe_router_jitter,
        shared_expert_gate=shared_expert_gate,
        # 组件选择（注册表名字；None 表示交给注册表用默认）
        attention=attention or "gqa",
        ffn=ffn or ("moe" if use_moe else "swiglu"),
        norm=norm or "rmsnorm",
        pos=pos_name,
        rope_base=rope_base,
        kv_cache_dtype=str(kv_cache_dtype or "auto").lower(),
    )
    return cfg


# build_config 接受的全部键（配置文件里多出来的键会被忽略，不会报错）
KNOWN_KEYS = (
    # "type" 是模型实现的注册表名（默认 lingyu）；由 config_from_section 单独取出
    "type", "vocab_size", "block_size", "n_layer", "n_embd", "n_head", "n_kv_head",
    "d_ff", "dropout", "bias", "tie_weights", "use_rope", "use_learned_pos",
    "norm_eps", "init_std", "residual_std", "use_moe", "n_expert",
    "n_expert_per_tok", "n_shared_expert", "expert_d_ff", "moe_every",
    "moe_router_jitter", "shared_expert_gate",
    # ⚠️ 组件选择（注册表名字）**必须在这里**，否则配置文件里写了
    #    norm = layernorm 会被静默忽略，模型还是用默认实现 —— 实测踩过。
    "attention", "ffn", "norm", "pos", "rope_base", "kv_cache_dtype",
)

# 这些键名在配置文件和 build_config 之间叫法不同
_KEY_ALIAS = {"d_model": "n_embd"}

# 字符串类配置（不参与 int/float/bool 转换）
_STR_KEYS = ("attention", "ffn", "norm", "pos", "type", "kv_cache_dtype")


def config_from_section(sec, vocab_size):
    """从配置文件的 [model] 段构造模型配置。

    多出来的键会被忽略（这样配置文件里可以写注释性字段，以后加字段也不炸）。
    """
    kw = {}
    for k, v in (sec or {}).items():
        k2 = _KEY_ALIAS.get(k, k)
        if k2 in KNOWN_KEYS and v is not None:
            kw[k2] = v
    # ---- vocab_size：0 = 跟随分词器；写了别的值就必须和分词器一致（审计 B-09）----
    # 以前这里无条件覆盖成分词器词表 —— 配置里写多少都静默无效，
    # 用户以为“我设了 32000”，实际模型还是 8192，断点加载时才炸。
    user_vocab = kw.get("vocab_size")
    try:
        user_vocab = int(user_vocab or 0)
    except (TypeError, ValueError):
        raise ValueError("[model] vocab_size 必须是整数（0 = 跟随分词器），拿到 %r"
                         % (kw.get("vocab_size"),))
    if user_vocab and user_vocab != int(vocab_size):
        raise ValueError(
            "[model] vocab_size = %d 和分词器词表 %d 不一致。\n"
            "  词表大小由分词器决定：留 0（自动跟随），或重新 prepare 生成对应词表的语料。"
            % (user_vocab, int(vocab_size)))
    kw["vocab_size"] = int(vocab_size)
    for key in ("n_layer", "n_embd", "n_head"):
        if key not in kw:
            raise KeyError("配置文件 [model] 段缺少必填项: %s" % key)
    for key in ("n_layer", "n_embd", "n_head", "n_kv_head", "block_size",
                "n_expert", "n_expert_per_tok", "n_shared_expert", "moe_every"):
        if key in kw:
            kw[key] = int(kw[key])
    for key in ("dropout", "norm_eps", "init_std", "residual_std",
                "moe_router_jitter", "rope_base"):
        if key in kw:
            kw[key] = float(kw[key])
    for key in ("bias", "tie_weights", "use_rope", "use_learned_pos", "use_moe",
                "shared_expert_gate"):
        if key in kw:
            kw[key] = bool(kw[key])
    for key in _STR_KEYS:
        if key in kw:
            kw[key] = str(kw[key]).strip().lower()
    for key in ("d_ff", "expert_d_ff"):
        if key in kw and kw[key]:
            kw[key] = int(kw[key])
        else:
            kw.pop(key, None)
    # type 不是 build_config 的参数：它决定「用哪个模型实现」，单独挂到 cfg 上
    model_type = str(kw.pop("type", "") or "lingyu").strip().lower()
    cfg = build_config(**kw)
    cfg["type"] = model_type
    return cfg


def _kv_elem_bytes(cfg, dtype_bytes=None):
    """每个 KV 元素占多少字节（int8 量化时外加每 token 一组 fp32 scale）。"""
    if dtype_bytes is not None:
        return float(dtype_bytes)
    kvq = str(cfg.get("kv_cache_dtype", "auto") or "auto").lower()
    if kvq == "int8":
        # 1 字节量化值 + 每 (token, head) 一个 fp32 scale，摊到 head_dim 个元素上
        return 1.0 + 4.0 / max(1, int(cfg["head_dim"]))
    if kvq == "f32":
        return 4.0
    return 2.0 if kvq in ("auto", "bf16") else 2.0


def kv_cache_bytes(cfg, seq_len, batch=1, dtype_bytes=None):
    """一个 batch 在给定序列长度下的 KV cache 字节数。"""
    n = (2 * cfg["n_layer"] * cfg["n_kv_head"] * cfg["head_dim"]
         * seq_len * batch * _kv_elem_bytes(cfg, dtype_bytes))
    return int(round(n))


def kv_cache_per_token(cfg, dtype_bytes=None):
    return int(round(2 * cfg["n_layer"] * cfg["n_kv_head"] * cfg["head_dim"]
                     * _kv_elem_bytes(cfg, dtype_bytes)))


# 给 Block 补一个「激活参数」方法，供 LingYu.active_param_count 调用
def _block_active(self):
    cfg = self.cfg_ref
    d = cfg["d_model"]
    n = 0
    n += 4 * d * d                      # q,k,v,o（GQA 下 k/v 更小，这里按上界算后校正）
    n -= 2 * d * (d - self.attn.k_proj.weight.shape[0])
    if self.is_moe:
        e = cfg["expert_d_ff"]
        n += self.mlp.top_k * (3 * d * e)
        if cfg["n_shared_expert"] > 0:
            n += cfg["n_shared_expert"] * (3 * d * cfg["d_ff"])
        n += d * cfg["n_expert"]
    else:
        n += 3 * d * cfg["d_ff"]
    n += 2 * norm_dim_mult(cfg) * d     # 本层两个归一化（RMSNorm 各 d，LayerNorm 各 2d）
    return n


# ==========================================================================
# 内置组件的注册
#
# 放在文件**末尾**是刻意的：components.py 刻意不 import 本模块（避免循环导入），
# 所以「实现」必须由本模块自己在所有类定义完之后登记进去。
#
# 加一个新组件要改两处：
#     1. 在这里（或 components.py 的 BUILTIN_* 常量）加名字
#     2. 写实现类
# 想知道引擎支持什么，看 components.py 的 BUILTIN_* 与下面这段。
# ==========================================================================
register("attention", "gqa", override=True)(Attention)
register("attention", "mha", override=True)(Attention)
register("attention", "mqa", override=True)(Attention)
register("ffn", "swiglu", override=True)(SwiGLU)
register("ffn", "moe", override=True)(MoE)
register("norm", "rmsnorm", override=True)(RMSNorm)
register("norm", "layernorm", override=True)(LayerNorm)
register("pos", "rope", override=True)(RotaryEmbedding)
register("model", "lingyu", override=True)(LingYu)

# 转出给外部用（`from lingyu.model import get_norm` 也能用）
describe_components = components.describe_all

__all__ = [
    "RMSNorm", "LayerNorm", "RotaryEmbedding", "apply_rope", "Attention",
    "SwiGLU", "MoE", "Block", "LingYu",
    "ModelProtocol", "build_model", "model_fingerprint", "estimate_param_count",
    "build_config", "config_from_section",
    "kv_cache_bytes", "kv_cache_per_token",
    "get_attention", "get_ffn", "get_norm", "get_pos", "describe_components",
]


# ------------------------------------------------------------ 纯估算
# 注意：这里刻意不实例化模型。`lingyu plan/params` 在「用户配了一个
# 装不下的巨大模型」时必须能瞬间回答，而不是先吃掉几十 GB 内存再被 OOM 杀掉。
def estimate_param_count(model_cfg):
    """按结构解析式估算参数量；只支持内置 lingyu，自定义模型返回 None。

    精度与 `param_groups()` 的实测总和一致（有单测钉住）；MoE 按每层
    `moe_every` 的轮换规则、共享专家、bias、归一化维度逐项计入。
    """
    if (model_cfg.get("type") or "lingyu").lower() != "lingyu":
        return None
    cfg = model_cfg
    d = int(cfg["d_model"])
    V = int(cfg["vocab_size"])
    nd = norm_dim_mult(cfg)
    n = V * d * (1 if cfg["tie_weights"] else 2)   # 词嵌入（+ 输出层）
    n += nd * d                                    # 末尾归一化 ln_f
    if cfg.get("use_learned_pos"):
        n += int(cfg["block_size"]) * d
    bias = bool(cfg.get("bias"))
    ffn = (cfg.get("ffn") or
           ("moe" if cfg.get("use_moe") else "swiglu")).lower()
    for i in range(int(cfg["n_layer"])):
        # 注意力：Q/K/V/O 四个投影
        hd = int(cfg["head_dim"])
        n += (d * (cfg["n_head"] * hd)
              + 2 * d * (cfg["n_kv_head"] * hd)
              + d * (cfg["n_head"] * hd))
        if bias:
            # Q / O 各 d 个 bias，K / V 各 n_kv_head*head_dim 个。
            # 旧写法多算了 d（实测 49,536 vs 解析 49,664），已修。
            n += 2 * d + 2 * (cfg["n_kv_head"] * hd)
        moe_here = (ffn == "moe" and int(cfg["moe_every"]) > 0
                    and i % int(cfg["moe_every"]) == 0)
        if moe_here:
            e = int(cfg["expert_d_ff"])
            n += int(cfg["n_expert"]) * (3 * d * e) + d * int(cfg["n_expert"])
            if int(cfg["n_shared_expert"]) > 0:
                n += int(cfg["n_shared_expert"]) * (3 * d * int(cfg["d_ff"]))
                if cfg.get("shared_expert_gate"):
                    n += d
        else:
            n += 3 * d * int(cfg["d_ff"])
            if bias:
                n += 2 * int(cfg["d_ff"]) + d
        n += 2 * nd * d                            # 本层两个归一化
    return int(n)


