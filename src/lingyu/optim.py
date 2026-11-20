"""8-bit AdamW（分块动态量化）。

思想来自 Dettmers et al. 2021, "8-bit Optimizers via Block-wise Quantization"
(arXiv:2110.02871)：Adam 的一阶/二阶动量按 2048 个元素一组求 absmax 缩放，
量化成 int8 保存；更新时反量化到 fp32 计算，算完重新量化。
实现细节（实测教训）：只把**一阶动量 m** 压成 int8（每块一个 fp32 尺度）；
二阶动量 v 保留 bf16。原因：v 里有小到接近 0 的元素，int8 分块量化会把它们
直接压成 0，分母 sqrt(v) 变 0 会让那一步更新爆掉（实测单个参数一步跳 2.4）。
显存：每个参数 3 字节（m 1 + v 2）+ 每块一个尺度，相比 fp32 AdamW 的
8 字节/参数，约省 2.6 倍；如果以后找到对 v 更稳的量化方式，可以再压到 2 字节。

为什么不直接用 bitsandbytes：本项目要求纯 PyTorch、离线可装、CPU 上也能跑
测试；这里只依赖 torch，一百行，方便审计和改。
"""
import torch

BLOCK = 2048


class AdamW8bit(torch.optim.Optimizer):
    """AdamW + 分块 8-bit 动量状态。参数本身仍是 fp32/bf16，不量化。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _quantize(t):
        flat = t.detach().reshape(-1).float()
        n = flat.numel()
        if n == 0:
            return (torch.zeros(0, dtype=torch.int8),
                    torch.zeros(0, dtype=torch.float32), 0)
        nb = (n + BLOCK - 1) // BLOCK
        pad = nb * BLOCK - n
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blocks = flat.view(nb, BLOCK)
        absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
        scale = absmax / 127.0
        q = torch.round(blocks / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return q.contiguous(), scale.to(torch.float32), n

    @staticmethod
    def _dequantize(q, scale, n):
        return (q.float() * scale[:, None]).reshape(-1)[:n]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamW8bit 不支持稀疏梯度")
                state = self.state[p]
                n = p.numel()
                if len(state) == 0:
                    state["step"] = 0
                    m_q, m_scale, _ = self._quantize(
                        torch.zeros(n, dtype=torch.float32, device=p.device))
                    state["m_q"], state["m_scale"] = m_q, m_scale
                    state["v"] = torch.zeros(n, dtype=torch.bfloat16,
                                              device=p.device)
                state["step"] += 1
                m = self._dequantize(state["m_q"], state["m_scale"], n).reshape(p.shape)
                v = state["v"].float().reshape(p.shape)
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                state["m_q"], state["m_scale"], _ = self._quantize(m)
                state["v"] = v.reshape(-1).to(torch.bfloat16)
                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]
                if wd:
                    p.mul_(1.0 - lr * wd)
                denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss
