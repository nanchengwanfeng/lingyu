import pytest
import torch

from lingyu.optim import AdamW8bit
from lingyu.train import configure_optimizer


def _toy_cfg(**kw):
    cfg = {"weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
           "learning_rate": 1e-3}
    cfg.update(kw)
    return cfg


def test_adamw8bit_keeps_m_int8_and_v_bf16():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8193))
    p.grad = torch.randn_like(p)
    opt = AdamW8bit([p], lr=1e-3, betas=(0.9, 0.95))
    opt.step()
    st = opt.state[p]
    assert st["m_q"].dtype is torch.int8
    assert st["v"].dtype is torch.bfloat16
    nbytes = sum(t.numel() * t.element_size()
                 for t in (st["m_q"], st["v"], st["m_scale"]))
    # fp32 AdamW 要 2 份 4 字节/参数；这里 m 1 字节 + v 2 字节 + 每块尺度
    padded = ((p.numel() + 2047) // 2048) * 2048
    assert nbytes <= padded * 3 + 4096


def test_adamw8bit_close_to_fp32_adamw():
    torch.manual_seed(0)
    grads = [torch.randn(512) for _ in range(20)]
    p1 = torch.nn.Parameter(torch.randn(512))
    p2 = torch.nn.Parameter(p1.detach().clone())
    o1 = torch.optim.AdamW([p1], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    o2 = AdamW8bit([p2], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    for g in grads:
        for opt, pp in ((o1, p1), (o2, p2)):
            opt.zero_grad()
            pp.grad = g.clone()
            opt.step()
    diff = (p1 - p2).abs().max().item()
    # m 的量化误差约 absmax/127；v 保留 bf16。允许小量偏差但更新不能发散。
    assert torch.isfinite(p2).all()
    assert diff < 5e-3, "8-bit 与 fp32 的更新偏差过大: %g" % diff


def test_configure_optimizer_picks_8bit():
    model = torch.nn.Linear(16, 16)
    assert isinstance(configure_optimizer(model, _toy_cfg(optim_bits="8")),
                      AdamW8bit)
    assert not isinstance(configure_optimizer(model, _toy_cfg(optim_bits="32")),
                         AdamW8bit)


def test_configure_optimizer_rejects_other_bits():
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError):
        configure_optimizer(model, _toy_cfg(optim_bits="16"))
