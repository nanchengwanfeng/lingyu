# -*- coding: utf-8 -*-
"""示例插件：局部窗口注意力（local window attention）。

用途
----
演示「只换注意力内核、不动模型其余部分」需要多小的改动量：
  - 不碰 model.py 的前向、KV cache、RoPE、量化缓存逻辑；
  - 只继承内置 `Attention`，覆写 `_attend()`（SDPA 的唯一出口）；
  - 注册成新名字 `local_window`，配置里写 `attention = local_window` 即生效。

窗口大小用环境变量 `LINGYU_ATTN_WINDOW` 控制（默认 512，0 = 全注意力）。
之所以不放进 [model] 配置：这是示例插件，不想为它改动引擎的参数契约；
真实产品里要暴露到配置，就在 spec.py 的 [model] 段加一个参数，再从这里读。

怎么用
------
1. 把这个文件复制到 `C:\\path\to\\lingyu\\plugins\\`（或你的引擎安装目录的 plugins/）；
2. 配置里写 `attention = local_window`；
3. 可选：`set LINGYU_ATTN_WINDOW=1024`。

注意
----
- 局部窗口会**改变模型能看到的上下文**：训练与推理必须用同一个窗口，
  否则效果会明显变差（等价于换了位置编码 / 掩码规则）。
- 窗口小于训练时的序列长度时，长距离依赖学不到；这是「省显存」的代价，
  不是 bug。窗口 0（默认之外的全注意力）不改变任何行为。
- 生成时的 KV cache 路径也已覆盖：单 token 解码时同样只看最近 window 个 key。
"""

import os

import torch
import torch.nn.functional as F

from lingyu.model import Attention
from lingyu.registry import register


def _window():
    try:
        return max(0, int(os.environ.get("LINGYU_ATTN_WINDOW", "512")))
    except (TypeError, ValueError):
        return 512


class LocalWindowAttention(Attention):
    """因果 + 滑窗：query 只能看最近 window 个 key（含自己）。"""

    def __init__(self, cfg, layer_idx):
        super().__init__(cfg, layer_idx)
        self.window = _window()

    def _attend(self, q, k_all, v_all, T, offset):
        total = k_all.shape[2]
        w = self.window
        # 窗口为 0，或整段历史都装得下：与默认实现完全一致（零开销）
        if w <= 0 or (T <= 1 and total <= w) or (T > 1 and offset == 0 and total <= w):
            return super()._attend(q, k_all, v_all, T, offset)
        qi = (torch.arange(T, device=q.device) + int(offset)).view(1, T, 1)
        ki = torch.arange(total, device=q.device).view(1, 1, total)
        mask = (ki <= qi) & (ki > qi - w)
        return F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=(self.n_head != self.n_kv_head))


register("attention", "local_window")(LocalWindowAttention)
