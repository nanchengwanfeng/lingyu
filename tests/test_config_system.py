# -*- coding: utf-8 -*-
"""配置系统测试：spec 完整性、校验、加载、覆盖。

这些测试的价值在于**防回归** —— 以后谁改了 spec 或校验逻辑，
跑一遍就知道有没有踩坏别人的配置。
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lingyu.config import (SPEC, SECTIONS, TIERS, all_keys, apply_overrides,
                           get, iter_params, load_config, make_ini_template,
                           save_config, validate)
from lingyu.config.spec import COMPONENT_CHOICES


# ============================================================== spec 自身

def test_spec_sections_complete():
    """七个配置段都要在。"""
    for sec in ("model", "train", "data", "sample", "post", "parallel", "runtime"):
        assert sec in SECTIONS, "缺少配置段 %s" % sec
    assert len(SECTIONS) == 7


def test_every_param_has_bilingual_help():
    """每个参数都必须有中英双语帮助 —— 开源要用，缺了 GUI 和文档就出不来。"""
    bad = []
    for sec, key, p in iter_params():
        if not p.get("help_zh"):
            bad.append("%s.%s 缺中文帮助" % (sec, key))
        if not p.get("help_en"):
            bad.append("%s.%s 缺英文帮助" % (sec, key))
    assert not bad, "以下参数帮助不全：\n  " + "\n  ".join(bad)


def test_every_param_has_valid_tier():
    for sec, key, p in iter_params():
        assert p["tier"] in TIERS, "%s.%s 的 tier=%r 不合法" % (sec, key, p["tier"])


def test_every_param_has_type_and_default():
    for sec, key, p in iter_params():
        assert "type" in p, "%s.%s 没有 type" % (sec, key)
        assert "default" in p, "%s.%s 没有 default" % (sec, key)
        assert isinstance(p["default"], p["type"]), \
            "%s.%s 的默认值类型不对：%r 不是 %s" % (sec, key, p["default"], p["type"].__name__)


def test_defaults_pass_validation():
    """**最重要的测试**：所有参数的默认值组合起来必须是合法配置。

    如果这条挂了，说明新用户一上手就是错的。
    """
    cfg = {sec: {k: p["default"] for k, p in SPEC[sec].items()}
           for sec in SECTIONS}
    # vocab_size=0 在真实使用时会被分词器填上，这里给个真值
    cfg["model"]["vocab_size"] = 8192
    errs, warns, _ = validate(cfg)
    assert not errs, "默认配置校验失败：\n  " + "\n  ".join(errs)


def test_component_choices_are_lists():
    for kind, choices in COMPONENT_CHOICES.items():
        assert isinstance(choices, list) and choices, "%s 的候选列表是空的" % kind


def test_ini_template_generates():
    """配置模板能生成，且生成的文本能被自己解析回来。"""
    for sec in SECTIONS:
        text = make_ini_template(sec)
        assert text.strip(), "[%s] 生成的模板是空的" % sec
        assert "#" in text, "[%s] 模板里应该有注释" % sec


# ============================================================== 校验逻辑

def _cfg(**model):
    cfg = {sec: {k: p["default"] for k, p in SPEC[sec].items()} for sec in SECTIONS}
    cfg["model"]["vocab_size"] = 8192
    cfg["model"].update(model)
    return cfg


def test_n_embd_must_divide_n_head():
    errs, _, _ = validate(_cfg(n_embd=384, n_head=7))
    assert any("整除" in e for e in errs), errs


def test_head_dim_must_be_even():
    # 390 / 6 = 65，奇数
    errs, _, _ = validate(_cfg(n_embd=390, n_head=6))
    assert any("偶数" in e for e in errs), errs


def test_even_head_dim_is_ok():
    errs, _, _ = validate(_cfg(n_embd=300, n_head=6))   # 50，偶数
    assert not any("偶数" in e for e in errs), errs


def test_small_head_dim_warns():
    _, warns, _ = validate(_cfg(n_embd=64, n_head=8))   # 每头 8 维
    assert any("偏小" in w for w in warns), warns


def test_n_kv_head_cannot_exceed_n_head():
    errs, _, _ = validate(_cfg(n_head=8, n_kv_head=16))
    assert any("不能大于" in e for e in errs), errs


def test_n_head_must_be_multiple_of_n_kv_head():
    errs, _, _ = validate(_cfg(n_head=8, n_kv_head=3))
    assert any("整数倍" in e for e in errs), errs


def test_type_mismatch_caught():
    cfg = _cfg()
    cfg["model"]["n_layer"] = "八层"
    errs, _, _ = validate(cfg)
    assert any("整数" in e for e in errs), errs


def test_range_caught():
    cfg = _cfg()
    cfg["model"]["n_layer"] = 99999
    errs, _, _ = validate(cfg)
    assert any("太大" in e for e in errs), errs


def test_choices_caught():
    cfg = _cfg()
    cfg["train"]["amp_dtype"] = "float64"
    errs, _, _ = validate(cfg)
    assert any("只能是" in e for e in errs), errs


def test_moe_top1_warns_about_collapse():
    """top-1 在小模型上会坍缩 —— 这是实测过的坑，必须警告。"""
    _, warns, _ = validate(_cfg(use_moe=True, n_expert=4, n_expert_per_tok=1))
    assert any("坍缩" in w for w in warns), warns


def test_moe_top_k_cannot_exceed_experts():
    errs, _, _ = validate(_cfg(use_moe=True, n_expert=4, n_expert_per_tok=8))
    assert any("不能大于" in e for e in errs), errs


def test_warmup_larger_than_steps_warns():
    cfg = _cfg()
    cfg["train"]["warmup_iters"] = 9999
    cfg["train"]["max_steps"] = 1000
    _, warns, _ = validate(cfg)
    assert any("预热" in w for w in warns), warns


def test_sft_needs_data_path():
    cfg = _cfg()
    cfg["post"]["method"] = "sft"
    cfg["post"]["data_path"] = ""
    errs, _, _ = validate(cfg)
    assert any("data_path" in e for e in errs), errs


def test_loss_on_prompt_warns():
    """开着 loss_on_prompt 会让模型自己生成问题 —— 必须警告。"""
    cfg = _cfg()
    cfg["post"].update({"method": "sft", "data_path": "x.jsonl",
                        "loss_on_prompt": True})
    _, warns, _ = validate(cfg)
    assert any("自己生成问题" in w for w in warns), warns


def test_unknown_section_warns_not_errors():
    cfg = _cfg()
    cfg["nosuchsection"] = {"a": 1}
    errs, warns, _ = validate(cfg)
    assert not errs
    assert any("不认识" in w for w in warns), warns


def test_unknown_key_warns_but_does_not_error():
    cfg = _cfg()
    cfg["model"]["n_layerr"] = 8
    errs, warns, _ = validate(cfg)
    assert not errs
    assert any("n_layerr" in w for w in warns), warns


def test_strict_mode_promotes_warnings():
    cfg = _cfg()
    cfg["model"]["n_layerr"] = 8
    errs, warns, _ = validate(cfg, strict=True)
    assert errs and not warns


# ============================================================== 加载与覆盖

def test_roundtrip_save_load():
    cfg = _cfg()
    cfg["model"]["n_layer"] = 12
    cfg["train"]["out_dir"] = "D:/tmp/out"
    cfg["sample"]["prompt"] = "测试开头"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ini")
        save_config(p, cfg, comment="测试配置\n第二行")
        back = load_config(p)
    assert back["model"]["n_layer"] == 12
    assert back["train"]["out_dir"] == "D:/tmp/out"
    assert back["sample"]["prompt"] == "测试开头"


def test_ini_comment_stripping():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ini")
        with open(p, "w", encoding="utf-8") as fp:
            fp.write("[model]\nn_layer = 8   # 这是注释\n")
            fp.write("prompt_like = \"含#号\"\n")
        cfg = load_config(p)
    assert cfg["model"]["n_layer"] == 8
    assert cfg["model"]["prompt_like"] == "含#号"


def test_ini_missing_equals_raises():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ini")
        with open(p, "w", encoding="utf-8") as fp:
            fp.write("[model]\nn_layer 8\n")
        with pytest.raises(ValueError, match="缺少 = 号"):
            load_config(p)


def test_ini_key_before_section_raises():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ini")
        with open(p, "w", encoding="utf-8") as fp:
            fp.write("n_layer = 8\n")
        with pytest.raises(ValueError):
            load_config(p)


def test_yaml_without_pyyaml_gives_helpful_error():
    """没装 pyyaml 时，报错要说清怎么办，不能只报 ImportError。"""
    try:
        import yaml  # noqa: F401
        pytest.skip("本机装了 pyyaml，跳过这条")
    except ImportError:
        pass
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.yaml")
        with open(p, "w", encoding="utf-8") as fp:
            fp.write("model:\n  n_layer: 8\n")
        with pytest.raises(ValueError, match="pyyaml"):
            load_config(p)


def test_override_basic():
    cfg = {}
    apply_overrides(cfg, ["model.n_layer=12", "train.max_steps=1000"])
    assert cfg["model"]["n_layer"] == 12
    assert cfg["train"]["max_steps"] == 1000


def test_override_typo_gets_corrected():
    """常见笔误自动纠正，并给出提示。"""
    notices = []
    cfg = {}
    apply_overrides(cfg, ["train.out.dir=D:/x"], on_notice=notices.append)
    assert cfg["train"]["out_dir"] == "D:/x"
    assert notices and "out_dir" in notices[0]


def test_override_unknown_key_raises_with_suggestion():
    """写错键名必须报错 —— 静默忽略是最坑的。"""
    with pytest.raises(ValueError) as ei:
        apply_overrides({}, ["train.foobar=1"])
    msg = str(ei.value)
    assert "foobar" in msg
    assert "可用的项" in msg


def test_override_similar_key_suggests():
    with pytest.raises(ValueError) as ei:
        apply_overrides({}, ["model.n_layerr=8"])
    assert "n_layer" in str(ei.value)   # 建议里应该有正确的名字


def test_override_unknown_section_raises():
    with pytest.raises(ValueError, match="配置节"):
        apply_overrides({}, ["nosuch.key=1"])


def test_override_no_equals_raises():
    with pytest.raises(ValueError, match="格式"):
        apply_overrides({}, ["train.max_steps"])


def test_override_no_dot_raises():
    with pytest.raises(ValueError, match="节名.键名"):
        apply_overrides({}, ["nodot=1"])


def test_override_message_shows_multi_set_example():
    """报错要示范「怎么改多项」，否则用户会像之前那样反复试错。"""
    with pytest.raises(ValueError) as ei:
        apply_overrides({}, ["train.max_steps"])
    assert "--set train.max_steps" in str(ei.value)


def test_override_repeated_edits():
    """多个 --set 连续生效（这是最常用的写法）。"""
    cfg = {}
    apply_overrides(cfg, ["train.max_steps=1000", "train.out_dir=D:/x",
                          "model.n_layer=12", "model.n_embd=384",
                          "model.n_head=6", "model.n_kv_head=2"])
    assert cfg["train"]["max_steps"] == 1000
    assert cfg["train"]["out_dir"] == "D:/x"
    assert cfg["model"]["n_layer"] == 12
    errs, _, _ = validate(cfg)
    assert not errs, errs


def test_override_does_not_cross_sections():
    """model.lr 不能被纠正成 model.learning_rate（那个键属于 train）。"""
    with pytest.raises(ValueError) as ei:
        apply_overrides({}, ["model.lr=1e-3"])
    assert "model.learning_rate" not in str(ei.value)


def test_override_bool_and_float():
    cfg = {}
    apply_overrides(cfg, ["model.use_moe=true", "model.dropout=0.15"])
    assert cfg["model"]["use_moe"] is True
    assert cfg["model"]["dropout"] == 0.15


def test_get_prefers_config_over_default():
    cfg = {"model": {"n_layer": 20}}
    assert get(cfg, "model", "n_layer") == 20
    assert get(cfg, "model", "n_head") == SPEC["model"]["n_head"]["default"]


def test_get_treats_empty_string_as_missing():
    """空字符串当作"没设置"，回落到默认值 —— 配置文件里留空是常见写法。"""
    cfg = {"model": {"n_layer": ""}}
    assert get(cfg, "model", "n_layer") == SPEC["model"]["n_layer"]["default"]


def test_all_keys_matches_spec():
    for sec in SECTIONS:
        assert set(all_keys(sec)) == set(SPEC[sec].keys())


# ============================================================== INI 重复键

def test_duplicate_ini_key_warns_with_section_key_and_line_numbers(tmp_path):
    """★ 同段同键写两次必须出声（审计 B-11）。

    INI 语义是「后面的赢」。复制粘贴留下的重复行会让用户以为改成功了，
    实际用的还是旧值 —— 以前完全静默，查起来只能一行行肉眼找。
    """
    p = tmp_path / "dup.ini"
    p.write_text("[model]\nn_layer = 4\n# 中间注释\nn_layer = 8\n",
                 encoding="utf-8")
    with pytest.warns(UserWarning) as rec:
        cfg = load_config(str(p))
    assert cfg["model"]["n_layer"] == 8      # 解析行为不变：后面赢
    msg = str(rec[0].message)
    assert "[model]" in msg and "n_layer" in msg, msg
    assert "第 2 行" in msg and "第 4 行" in msg, msg


def test_distinct_keys_in_same_section_do_not_warn(tmp_path):
    """只有重复键才警告，正常配置不许有噪音。"""
    import warnings as _warnings
    p = tmp_path / "ok.ini"
    p.write_text("[model]\nn_layer = 4\nn_embd = 128\n", encoding="utf-8")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")      # 有警告就失败
        cfg = load_config(str(p))
    assert cfg["model"]["n_layer"] == 4
