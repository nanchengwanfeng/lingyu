# -*- coding: utf-8 -*-
"""示例：用 ~50 行实现一个能被引擎训练 / 采样 / 保存的「模型」。

这不是一个有用的语言模型（它连语序都看不见，只对整段 token 做平均），
用途是演示引擎的模型契约：

  - ``forward`` 返回 ``(logits, loss, aux)``
  - ``targets`` 里的 -1 必须被忽略（SFT loss mask）
  - ``generate`` 能自回归续写
  - ``structural_fingerprint`` 告诉引擎「哪些字段变了就不能续训」

把它复制到 ``plugins/tiny_lm.py``（或直接 import）后，配置里写
``[model] type = tiny`` 就能训练：

    lingyu plan  --config examples/configs/example_tiny.ini
    lingyu train --config examples/configs/example_tiny.ini --device cpu \
                 --set train.max_steps=10

真实的自定义架构见 ``docs/EXTENDING.md``。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lingyu.registry import register


class TinyBagLM(nn.Module):
    """词袋语言模型：所有位置共享同一个「平均上下文」预测。

    参数量小、跑得快，适合给契约测试当活体样例。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d_model = int(self.cfg["d_model"])
        vocab = int(self.cfg["vocab_size"])
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # 让输出分布初始时别太尖锐
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------ 契约 1/3
    def forward(self, idx, targets=None, cache=None, cache_pos=0,
                collect_aux=True):
        x = self.tok_emb(idx).mean(dim=1, keepdim=True)   # (B, 1, d)
        logits = self.head(x)                              # (B, 1, V)
        loss = None
        if targets is not None:
            logits = logits.expand(-1, targets.shape[1], -1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss, None

    # ------------------------------------------------------------ 契约 2/3
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, **kwargs):
        out = idx
        for _ in range(int(max_new_tokens)):
            logits, _, _ = self(out)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out

    # ------------------------------------------------------------ 契约 3/3
    def structural_fingerprint(self):
        return {
            "n_layer": self.cfg.get("n_layer"),
            "n_embd": self.cfg.get("d_model"),
            "vocab_size": self.cfg.get("vocab_size"),
            "block_size": self.cfg.get("block_size"),
        }


register("model", "tiny")(TinyBagLM)
