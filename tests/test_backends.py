# -*- coding: utf-8 -*-
"""执行后端与规模估算测试。

这一组测试的职责很特别：**它们不验算得准不准，只验「有没有说谎」。**

规模估算必然有误差（激活值能差 2 倍），这没关系 —— 它的用途是
「提前发现这个配置根本不可能」。真正不能容忍的是**不诚实**：

- 说支持某个后端，其实没实现
- 让用户去跑一条不存在的命令
- 明明跑不动却说跑得动
- 明明跑得动却说跑不动

这些都会让用户对引擎失去信任，比算不准严重得多。
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lingyu.backends import (IMPLEMENTED, backend_info, backend_is_implemented,
                             bytes_per_param, estimate, fmt_bytes, generate,
                             list_backends, suggestions)
from lingyu.backends.launch import GENERATORS


# ==================================================== 显存模型

def test_unchanged_config_costs_16_bytes_per_param():
    """单卡不开启任何分片时，每参数 16 字节。

    2（bf16 权重）+ 2（bf16 梯度）+ 8（AdamW 两个 fp32 动量）
    + 4（fp32 主权重）= 16。这个数字是文档里反复出现的基准。
    """
    assert bytes_per_param(0, 1) == 16


def test_zero_stages_monotonically_reduce_memory():
    """ZeRO 阶段越高，每卡占用必须单调不增。"""
    vals = [bytes_per_param(s, 8) for s in (0, 1, 2, 3)]
    assert vals == sorted(vals, reverse=True), vals
    assert vals[0] > vals[-1]


def test_zero3_divides_by_world_size():
    """ZeRO-3 把一切都切开，每卡约 16/N。"""
    assert abs(bytes_per_param(3, 8) - 2.0) < 1e-9


def test_cpu_offload_removes_optimizer_bytes():
    """优化器挪到内存后，显存里只剩权重+梯度=4 字节。"""
    assert bytes_per_param(0, 1, cpu_offload=True) == 4


def test_more_cards_never_increase_per_card_memory():
    for stage in (1, 2, 3):
        a = bytes_per_param(stage, 2)
        b = bytes_per_param(stage, 16)
        assert b <= a, "stage %d: 卡多了每卡反而更占显存？" % stage


def test_activation_checkpoint_reduces_activations():
    """激活重算是「用时间换显存」，必须真的省下来。"""
    a = estimate(1e6, batch=8, seq=512, d_model=256, n_layer=8)
    b = estimate(1e6, batch=8, seq=512, d_model=256, n_layer=8,
                 activation_checkpoint=True)
    assert b["activation_bytes"] < a["activation_bytes"]
    assert b["total_bytes"] < a["total_bytes"]


def test_activation_scales_with_batch_and_layer():
    base = estimate(1e6, batch=1, seq=512, d_model=256, n_layer=8)
    more_b = estimate(1e6, batch=4, seq=512, d_model=256, n_layer=8)
    more_l = estimate(1e6, batch=1, seq=512, d_model=256, n_layer=16)
    assert more_b["activation_bytes"] == pytest.approx(base["activation_bytes"] * 4)
    assert more_l["activation_bytes"] == pytest.approx(base["activation_bytes"] * 2)


# ==================================================== 算几张卡

def test_small_model_fits_one_card():
    est = estimate(13e6, batch=8, seq=1024, d_model=256, n_layer=16,
                   per_card_bytes=6 * 1024 ** 3)
    assert est["fits_one_card"] is True
    assert est["cards_needed"] == 1
    assert "跑得动" in est["verdict"]


def test_huge_model_needs_many_cards_and_says_so():
    """★ 万亿参数必须诚实报出「需要多少张卡」。

    这是「一个引擎从 13M 到万亿」的核心：规模上去了不能假装还能跑，
    但也不能只说一句「太大了」—— 要给出具体数字。
    """
    est = estimate(1e12, batch=1, seq=2048, d_model=8192, n_layer=80,
                   per_card_bytes=80 * 1024 ** 3)      # 80GB 的卡
    assert est["fits_one_card"] is False
    assert est["cards_needed"] > 100, est["cards_needed"]
    assert "需要" in est["verdict"] and "张" in est["verdict"]


def test_fsdp_reduces_cards_needed():
    """同一个模型，开 FSDP 分片后需要的卡数必须变少。"""
    kw = dict(batch=4, seq=1024, d_model=2048, n_layer=24,
              per_card_bytes=24 * 1024 ** 3)
    plain = estimate(3e9, world_size=1, zero_stage=0, **kw)
    sharded = estimate(3e9, world_size=8, zero_stage=3, **kw)
    assert sharded["cards_needed"] < plain["cards_needed"]


def test_no_vram_info_means_no_false_verdict():
    """拿不到显卡信息时，不能编一个 verdict 出来。"""
    est = estimate(1e6, per_card_bytes=None)
    assert "cards_needed" not in est
    assert "verdict" not in est
    assert est["total_bytes"] > 0


def test_fmt_bytes_readable():
    assert fmt_bytes(0) == "0 B"
    assert "KB" in fmt_bytes(2048)
    assert "GB" in fmt_bytes(3 * 1024 ** 3)
    assert "TB" in fmt_bytes(5 * 1024 ** 4)


# ==================================================== 建议要能执行

def test_suggestions_only_when_it_does_not_fit():
    fits = estimate(1e6, per_card_bytes=8 * 1024 ** 3)
    assert suggestions({}, fits, 1e6) == []


def test_suggestions_are_concrete_flags():
    """每条建议都要带**具体怎么改**，不能只说「调小一点」。"""
    est = estimate(1e9, batch=8, seq=1024, d_model=2048, n_layer=24,
                   per_card_bytes=8 * 1024 ** 3)
    tips = suggestions({}, est, 1e9, n_layer=24, d_model=2048)
    assert tips, "跑不动却不给建议"
    joined = "\n".join(tips)
    assert "--set" in joined, "建议里应该有能直接抄的 --set"
    assert "lingyu launch" in joined, "应该告诉用户怎么生成分布式脚本"


def test_suggestions_report_correct_numbers_for_each_key():
    """★ 层数和宽度不能搞反。

    踩过的坑：调用方按位置传参
    `suggestions(cfg, est, params, n_layer, d_model)`，而签名里第 4 个是
    `active`、第 5 个才是 `n_layer` —— 于是建议写成
    「n_layer 从 2048 减半」（2048 其实是宽度），
    同时「减宽度」那条因为 d_model=None 直接消失，编号还断了一号。
    """
    est = estimate(1e9, batch=8, seq=1024, d_model=2048, n_layer=40,
                   per_card_bytes=8 * 1024 ** 3)
    joined = "\n".join(suggestions({}, est, 1e9, n_layer=40, d_model=2048))
    assert "n_layer 从 40" in joined, joined
    assert "n_embd 从 2048" in joined, joined


def test_suggestions_are_numbered_without_gaps():
    """编号必须连续 —— 「①②④」这种断号看着像漏了内容。"""
    est = estimate(1e9, batch=8, seq=1024, d_model=2048, n_layer=40,
                   per_card_bytes=8 * 1024 ** 3)
    tips = suggestions({}, est, 1e9, n_layer=40, d_model=2048)
    marks = [t[0] for t in tips]
    expect = [chr(0x2460 + i) for i in range(len(tips))]
    assert marks == expect, tips


def test_suggestions_have_no_unescaped_format_artifacts():
    """不能出现 `%%` 这种漏出来的格式转义 —— 那是代码 bug 的味道。"""
    est = estimate(1e9, batch=8, seq=1024, d_model=2048, n_layer=40,
                   per_card_bytes=8 * 1024 ** 3)
    joined = "\n".join(suggestions({}, est, 1e9, n_layer=40, d_model=2048))
    assert "%%" not in joined, joined


def test_suggestions_do_not_offer_cpu_offload_when_already_on():
    est = estimate(1e9, batch=8, seq=1024, d_model=2048, n_layer=24,
                   per_card_bytes=8 * 1024 ** 3)
    est["cpu_offload"] = True
    tips = "\n".join(suggestions({}, est, 1e9))
    assert "cpu_offload" not in tips, "已经开了就别再建议开"


def test_suggestions_never_present_unimplemented_switches_as_effective():
    """★ 建议里不许出现引擎**尚未实现**的开关。

    `train.py` 根本不读 `parallel.*`（没有分布式 / 激活重算 / CPU offload
    的实现）。旧版建议让用户 `--set parallel.cpu_offload=true` 之类 ——
    照着改显存一点不降，而且没有任何提示，比不给建议更坑。
    多卡脚本可以提，但必须明说「尚未实现」，不能包装成可生效手段。
    """
    est = estimate(1e9, batch=8, seq=1024, d_model=2048, n_layer=24,
                   per_card_bytes=8 * 1024 ** 3)
    joined = "\n".join(suggestions({}, est, 1e9, n_layer=24, d_model=2048))
    for bad in ("activation_checkpoint", "cpu_offload", "zero_stage",
                "parallel.backend", "fsdp"):
        assert bad not in joined, (
            "建议把未实现的开关 %r 当成可生效手段：\n%s" % (bad, joined))
    assert "lingyu launch" in joined, "该告诉用户怎么生成多卡脚本"
    assert "尚未实现" in joined, "提多卡脚本就必须标注它还没实现"


# ==================================================== 后端诚实性

def test_implemented_backends_are_truthful():
    """★ 只有 local 是引擎自己实现的。

    fsdp 目前只是**脚本生成（实验性）**：训练循环里没有分布式代码，
    生成的 torchrun 脚本会跑成 N 个独立单卡训练。这条测试是给未来的
    自己上的一道锁：等真实现了 FSDP2，再改 IMPLEMENTED 并补端到端测试。
    """
    assert IMPLEMENTED == ("local",)
    assert backend_is_implemented("local")
    assert not backend_is_implemented("fsdp")
    assert not backend_is_implemented("deepspeed")
    assert not backend_is_implemented("megatron")
    info = backend_info("fsdp")
    assert info["implemented"] == "config-only"
    assert "实验性" in info["summary"] or "尚未实现" in info["summary"]


def test_deepspeed_declares_itself_config_only():
    info = backend_info("deepspeed")
    assert info["implemented"] == "config-only"
    assert "不重写" in info["summary"] or "生成" in info["summary"]


def test_megatron_declares_itself_unimplemented():
    info = backend_info("megatron")
    assert info["implemented"] is False
    assert "不实现" in info["summary"]


def test_every_backend_has_generator_and_info():
    for name in ("local", "fsdp", "deepspeed", "megatron"):
        assert name in GENERATORS, name
        assert backend_info(name)["label"]


def test_list_backends_shape():
    rows = list_backends()
    assert [r["name"] for r in rows] == ["local", "fsdp", "deepspeed", "megatron"]
    assert all("summary" in r and "implemented" in r for r in rows)


# ==================================================== 生成的脚本

CFG = {
    "_path": "configs/small.ini",
    "model": {"n_layer": 16, "n_embd": 256, "n_head": 8, "n_kv_head": 2,
              "block_size": 1024, "vocab_size": 8192, "use_moe": False,
              "tie_weights": True},
    "train": {"batch_size": 8, "max_steps": 10000, "grad_clip": 1.0},
    "parallel": {"tensor_parallel": 1, "pipeline_parallel": 1,
                 "expert_parallel": 1, "zero_stage": 2},
}
EST = estimate(13e6, batch=8, seq=1024, d_model=256, n_layer=16,
               per_card_bytes=8 * 1024 ** 3)


def test_fsdp_script_is_runnable():
    """fsdp 是引擎真实现的路径，脚本里必须是能直接跑的 torchrun。"""
    r = generate("fsdp", CFG, EST, 13e6)
    s = r["script"]
    assert "torchrun" in s
    assert "lingyu train" in s
    assert "--set parallel.backend=fsdp" in s
    assert r["ds_config"] is None


def test_deepspeed_emits_config_file_and_honest_warning():
    r = generate("deepspeed", CFG, EST, 13e6)
    s = r["script"]
    assert r["ds_config"] is not None
    assert r["ds_config"]["zero_optimization"]["stage"] == 2
    assert "不能" in s and "lingyu train" in s, "必须说清这份脚本不能用 lingyu train 跑"
    assert "TODO" in s


def test_megatron_script_admits_it_is_a_reference():
    r = generate("megatron", CFG, EST, 13e6)
    s = r["script"]
    assert "没有实现" in s or "不实现" in s
    assert "TODO" in s
    assert "参考" in s


def test_scripts_list_todos():
    """每份脚本都必须标出需要人工确认的地方 —— 不许假装一键可用。"""
    for backend in ("fsdp", "deepspeed", "megatron"):
        s = generate(backend, CFG, EST, 13e6)["script"]
        assert "[TODO]" in s, "%s 的脚本没有 TODO 标记" % backend


def test_scripts_write_config_path_and_scale():
    for backend in ("fsdp", "deepspeed", "megatron"):
        s = generate(backend, CFG, EST, 13e6)["script"]
        assert "configs/small.ini" in s, backend
        assert "13.1M" in s or "13,082" in s or "13.0M" in s, backend


def test_moe_config_mentions_expert_parallel():
    cfg = dict(CFG)
    cfg = {**CFG, "model": dict(CFG["model"], use_moe=True, n_expert=8)}
    s = generate("megatron", cfg, EST, 13e6)["script"]
    assert "num-experts" in s or "MoE" in s


def test_unknown_backend_raises_with_options():
    with pytest.raises(ValueError) as ei:
        generate("nope", CFG, EST, 13e6)
    msg = str(ei.value)
    assert "fsdp" in msg and "megatron" in msg


def test_write_outputs_creates_files(tmp_path):
    from lingyu.backends import write_outputs
    r = generate("deepspeed", CFG, EST, 13e6)
    files = write_outputs(str(tmp_path / "o"), "deepspeed", r)
    assert len(files) == 2
    for f in files:
        assert os.path.isfile(f), f
        assert os.path.getsize(f) > 100, f
    # 脚本用 \n 行尾（Linux 集群上跑），不能是 Windows 的 \r\n
    sh = [f for f in files if f.endswith(".sh")][0]
    with open(sh, "rb") as fp:
        assert b"\r\n" not in fp.read(), "shell 脚本不该有 CRLF"


def test_write_outputs_sets_exec_bit(monkeypatch, tmp_path):
    """★ 非 Windows 上生成的 .sh 要带执行位（审计 B-16）。

    以前只写文件不 chmod，用户按脚本里写的 `./launch_x.sh` 跑会撞
    `Permission denied`。用假 os 在 Windows 上也能验证这条路径。
    """
    import types
    from lingyu.backends import launch as launch_mod
    calls = {}
    fake_os = types.SimpleNamespace(
        name="posix",
        path=os.path,
        makedirs=os.makedirs,
        chmod=lambda p, m: calls.__setitem__(p, m))
    monkeypatch.setattr(launch_mod, "os", fake_os)

    r = generate("megatron", CFG, EST, 13e6)
    files = launch_mod.write_outputs(str(tmp_path / "o"), "megatron", r)
    sh = [f for f in files if f.endswith(".sh")][0]
    assert calls.get(sh) == 0o755, calls
    assert os.path.isfile(sh)
