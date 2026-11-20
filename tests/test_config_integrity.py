# -*- coding: utf-8 -*-
"""配置完整性 + 数据/分词器配套检查。

这个文件里每一条测试都对应一个**真实踩过的坑**，共同点是：
它们都不会让程序崩，只会让你拿着错的结果继续往下走。

统一原则：**宁可开跑前拒绝，也不要悄悄跑出一个错的东西。**
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lingyu.config import SPEC, load_config, resolve
from lingyu.data import check_data_tokenizer


# ==================================================== resolve()：补全默认值

def test_resolve_fills_every_spec_param():
    """补全后，spec 里每一个参数都必须在配置里存在。

    这是「下游可以放心用下标取值」的前提 —— train.py 里 `cfg["weight_decay"]`
    曾经因为用户写了精简配置而抛 KeyError。
    """
    cfg = resolve({})
    missing = []
    for sec, params in SPEC.items():
        for key in params:
            if key not in cfg.get(sec, {}):
                missing.append("%s.%s" % (sec, key))
    assert not missing, "resolve() 没补上这些：" + "、".join(missing)


def test_resolve_keeps_user_values():
    """用户显式写的值不能被默认值顶掉。"""
    cfg = resolve({"train": {"max_steps": 42, "learning_rate": 0.5}})
    assert cfg["train"]["max_steps"] == 42
    assert cfg["train"]["learning_rate"] == 0.5


def test_resolve_keeps_unknown_keys():
    """spec 里没有的键要原样保留 —— 插件会用自己的配置项。"""
    cfg = resolve({"model": {"my_plugin_knob": 7}, "my_section": {"x": 1}})
    assert cfg["model"]["my_plugin_knob"] == 7
    assert cfg["my_section"]["x"] == 1


def test_resolve_is_idempotent():
    """补全两次 == 补全一次（否则 GUI 反复调用会漂移）。"""
    once = resolve({"train": {"max_steps": 10}})
    twice = resolve(once)
    assert once == twice


def test_resolve_coerces_types():
    """从 INI 读进来的是字符串，补全时要按 spec 的类型转好。"""
    cfg = resolve({"train": {"max_steps": "123", "learning_rate": "1e-3"},
                   "model": {"tie_weights": "true"}})
    assert cfg["train"]["max_steps"] == 123
    assert cfg["train"]["learning_rate"] == 1e-3
    assert cfg["model"]["tie_weights"] is True


# ==================================================== 出厂配置自检

def _shipped_configs():
    d = os.path.join(ROOT, "configs")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.endswith(".ini")]


@pytest.mark.parametrize("path", _shipped_configs(),
                         ids=lambda p: os.path.basename(p))
def test_shipped_config_data_dir_is_corpus_not_output(path):
    """★ data_dir 必须指向**语料目录**，不能是训练输出目录。

    真实 bug：configs/small.ini 里 `data_dir = C:/path/to/lingyu/out/small`
    —— 指向了 out_dir。跑 train 会直接找不到 train.bin，
    而用户看到配置里「明明路径是存在的」会一脸茫然。

    另外 data_dir 和 tokenizer_path 必须**同一个目录**：分属两次 prepare
    就会静默训出一堆乱码（见 check_data_tokenizer 的说明）。
    """
    cfg = load_config(path)
    data = cfg.get("data", {})
    data_dir = data.get("data_dir")
    if not data_dir:
        pytest.skip("该配置没写 data_dir")

    norm = os.path.normpath(os.path.abspath(data_dir))
    out_dir = cfg.get("train", {}).get("out_dir")
    if out_dir:
        assert norm != os.path.normpath(os.path.abspath(out_dir)), (
            "%s 的 data_dir 等于 out_dir（%s）—— data_dir 要写语料目录"
            % (os.path.basename(path), data_dir))

    tok = data.get("tokenizer_path")
    if tok:
        assert os.path.normpath(os.path.dirname(os.path.abspath(tok))) == norm, (
            "%s 的 tokenizer_path (%s) 不在 data_dir (%s) 里 —— "
            "两者必须来自同一次 prepare，否则训练不报错但结果是乱码"
            % (os.path.basename(path), tok, data_dir))


@pytest.mark.parametrize("path", _shipped_configs(),
                         ids=lambda p: os.path.basename(p))
def test_shipped_config_is_valid(path):
    """出厂配置必须能通过校验（有 error 就不行）。"""
    from lingyu.config import validate
    cfg = resolve(load_config(path))
    errors, warns, notices = validate(cfg)
    assert not errors, "%s 校验不通过：\n  %s" % (
        os.path.basename(path), "\n  ".join(errors))


@pytest.mark.parametrize("path", _shipped_configs(),
                         ids=lambda p: os.path.basename(p))
def test_shipped_config_data_exists(path):
    """配置指向的语料要真的在（数据没准备就跳过，CI 上不一定有语料）。"""
    cfg = load_config(path)
    data_dir = cfg.get("data", {}).get("data_dir")
    if not data_dir or not os.path.isdir(data_dir):
        pytest.skip("语料目录不存在（还没跑 prepare）")
    for f in ("train.bin", "val.bin"):
        assert os.path.isfile(os.path.join(data_dir, f)), (
            "%s 指向的 %s 里没有 %s" % (os.path.basename(path), data_dir, f))


# ==================================================== 语料/分词器配套

class _FakeTok:
    def __init__(self, vocab_size, kind):
        self.vocab_size = vocab_size
        self.kind = kind


def _make_data_dir(tmp_path, vocab_size=8192, kind="bpe", with_meta=True,
                   fingerprint=None):
    d = tmp_path / "d"
    d.mkdir(exist_ok=True)
    (d / "train.bin").write_bytes(b"\x00\x00" * 100)
    if with_meta:
        meta = {"dtype": "uint16", "n_tokens": 100, "vocab_size": vocab_size,
                "tokenizer_type": kind}
        if fingerprint:
            meta["tokenizer_fingerprint"] = fingerprint
        (d / "train.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(d)


def test_matching_pair_passes(tmp_path):
    d = _make_data_dir(tmp_path, 8192, "bpe")
    assert check_data_tokenizer(d, _FakeTok(8192, "bpe")) is None


def test_vocab_size_mismatch_is_refused(tmp_path):
    """词表大小不一致 —— 必须拒绝，不能只警告。"""
    d = _make_data_dir(tmp_path, 8192, "bpe")
    with pytest.raises(ValueError) as ei:
        check_data_tokenizer(d, _FakeTok(4096, "bpe"))
    assert "不配套" in str(ei.value)


def test_tokenizer_kind_mismatch_is_refused(tmp_path):
    """BPE 语料配 char 分词器：词表大小可能一样，一样会毁掉训练。"""
    d = _make_data_dir(tmp_path, 8192, "bpe")
    with pytest.raises(ValueError) as ei:
        check_data_tokenizer(d, _FakeTok(8192, "char"))
    assert "不配套" in str(ei.value)


def test_missing_meta_only_warns(tmp_path):
    """老数据目录没有 meta —— 查不了，但不能因此不让人训练。"""
    d = _make_data_dir(tmp_path, with_meta=False)
    msg = check_data_tokenizer(d, _FakeTok(8192, "bpe"))
    assert msg is not None and "无法核对" in msg


def test_non_strict_returns_message_instead_of_raising(tmp_path):
    d = _make_data_dir(tmp_path, 8192, "bpe")
    msg = check_data_tokenizer(d, _FakeTok(4096, "bpe"), strict=False)
    assert msg and "不配套" in msg


# ============================================ 词表指纹（最强的一道防线）

class _TokWithFp(_FakeTok):
    """带真实 fingerprint() 的分词器替身。"""
    def __init__(self, vocab_size, kind, fp):
        _FakeTok.__init__(self, vocab_size, kind)
        self._fp = fp

    def fingerprint(self):
        return self._fp


def test_fingerprint_mismatch_is_refused(tmp_path):
    """★ 词表大小、类型**全都一样**，只有 ID 分配不同 —— 也必须拦下来。

    这就是老分词器（4 个特殊 token）配新 prepare 的语料（6 个）的情形：
    vocab_size 都是 8192、都是 bpe，光看这两样永远发现不了，
    可每个汉字的 ID 都差了 2。不拦的后果是训练照跑、loss 照降、输出全是乱码。
    """
    d = _make_data_dir(tmp_path, 8192, "bpe", fingerprint="aaaa111122223333")
    with pytest.raises(ValueError) as ei:
        check_data_tokenizer(d, _TokWithFp(8192, "bpe", "bbbb444455556666"))
    msg = str(ei.value)
    assert "指纹" in msg and "aaaa111122223333" in msg
    assert "对不上" in msg


def test_fingerprint_match_passes(tmp_path):
    fp = "deadbeefcafe0001"
    d = _make_data_dir(tmp_path, 8192, "bpe", fingerprint=fp)
    assert check_data_tokenizer(d, _TokWithFp(8192, "bpe", fp)) is None


def test_missing_fingerprint_falls_back_to_weak_checks(tmp_path):
    """老数据目录没有指纹 —— 退化到旧检查，不能因此拒绝训练。"""
    d = _make_data_dir(tmp_path, 8192, "bpe")          # 无 fingerprint
    assert check_data_tokenizer(d, _TokWithFp(8192, "bpe", "whatever")) is None
    # 但弱检查该抓的还得抓
    with pytest.raises(ValueError):
        check_data_tokenizer(d, _TokWithFp(4096, "bpe", "whatever"))


def test_real_tokenizer_fingerprint_is_stable_across_save_load(tmp_path):
    """存盘再读回来，指纹必须一致 —— 否则每次加载都像换了词表。"""
    from lingyu.tokenizer import load_tokenizer, train_tokenizer
    text = "中文分词测试句子，用于验证指纹在存读之后保持一致。" * 30
    tok = train_tokenizer(text, 200, kind="char", verbose=False)
    p = str(tmp_path / "tokenizer.txt")
    tok.save(p)
    again = load_tokenizer(p)
    assert again.fingerprint() == tok.fingerprint()


# ==================================================== Windows BOM

def test_config_with_utf8_bom_is_readable(tmp_path):
    """★ 带 BOM 的配置文件必须能读。

    Windows 上太容易产生 BOM 了：记事本另存为 UTF-8、
    `Set-Content -Encoding UTF8`、VS Code 的「UTF-8 with BOM」。
    用普通 utf-8 读时第一行会变成 "\\ufeff[model]"，结果是
    「第 1 行看不懂（缺少 = 号）」—— 用户看着一个完全正确的配置文件
    被拒绝，根本猜不到是编码问题。
    """
    p = tmp_path / "bom.ini"
    body = "[model]\nn_layer = 4\n\n[train]\nmax_steps = 7\n"
    # 显式写 BOM
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    cfg = load_config(str(p))
    assert cfg["model"]["n_layer"] == 4, cfg
    assert cfg["train"]["max_steps"] == 7


def test_config_without_bom_still_works(tmp_path):
    """没有 BOM 的文件当然也要能读（utf-8-sig 对两者都兼容）。"""
    p = tmp_path / "plain.ini"
    p.write_bytes(b"[model]\nn_layer = 4\n")
    assert load_config(str(p))["model"]["n_layer"] == 4


def test_config_with_bom_survives_resolve_and_validate(tmp_path):
    from lingyu.config import validate
    p = tmp_path / "bom2.ini"
    p.write_bytes(b"\xef\xbb\xbf[model]\nn_layer = 4\nn_embd = 64\nn_head = 4\n")
    cfg = resolve(load_config(str(p)))
    errors, _, _ = validate(cfg)
    assert not errors, errors


# ============================================ SFT 配置必须能自动找到预训练断点

def _cfg(name):
    return resolve(load_config(os.path.join(ROOT, "configs", name)))


def test_sft_config_model_matches_pretrain_config():
    """★ SFT 的 [model] 必须和预训练的 [model] 完全一致。

    SFT 是「在已有模型上接着教」，结构不一样就接不上（引擎会拒绝，
    但那是运行时才发现）。出厂的两份配置应该开箱就对得上。
    """
    sft = _cfg("sft.ini")
    pre = _cfg("small.ini")
    keys = ("n_layer", "n_embd", "n_head", "n_kv_head", "block_size",
            "d_ff", "tie_weights", "bias", "use_moe")
    diffs = {k: (pre["model"].get(k), sft["model"].get(k))
             for k in keys if pre["model"].get(k) != sft["model"].get(k)}
    assert not diffs, (
        "sft.ini 和 small.ini 的 [model] 对不上：%s\n"
        "SFT 必须在预训练模型上接着教，结构必须一致。" % diffs)


def test_sft_config_train_out_dir_matches_pretrain():
    """★ 这条测试钉住一个真实的「开箱即坏」。

    踩过的坑：sft.ini 里没写 [train] out_dir，于是 find_base_checkpoint
    去默认的 `out/` 找断点，而 configs/small.ini 的断点其实在 `out/small/`。
    用户明明训过模型，跑 sft 却报「找不到预训练断点」，
    而且报错列出的四个路径里没有一个是他的模型所在处 —— 非常难猜。
    """
    sft = _cfg("sft.ini")
    pre = _cfg("small.ini")
    assert sft["train"]["out_dir"] == pre["train"]["out_dir"], (
        "sft.ini 的 [train] out_dir (%r) 和 small.ini (%r) 不一致，"
        "SFT 会找不到预训练断点" % (sft["train"]["out_dir"],
                                    pre["train"]["out_dir"]))


def test_sft_config_has_template_and_learning_rate():
    """SFT 配置得真的能跑：模板要写、学习率要不至于把模型训崩。"""
    sft = _cfg("sft.ini")
    assert sft["post"]["template"] in ("plain", "chatml", "qwen", "llama3")
    lr = float(sft["post"]["learning_rate"])
    assert 0 < lr <= 2e-3, (
        "SFT 学习率 %.1e 偏大 —— 预训练那种量级会让模型「灾难性遗忘」" % lr)


def test_sft_config_templates_usable_with_shipped_tokenizer():
    """如果 sft.ini 选了 chatml，那 preprare 出来的词表必须真的支持它。

    这条是防止「配置写着 chatml，但分词器没有那两个 token」——
    那种情况下运行时才报错，而且用户会以为是别的问题。
    """
    sft = _cfg("sft.ini")
    tpl = (sft["post"]["template"] or "plain").lower()
    if tpl == "plain":
        return
    from lingyu.chat_template import needs_special_tokens
    assert needs_special_tokens(tpl)
    tok_path = sft["data"]["tokenizer_path"]
    if not os.path.isfile(tok_path):
        pytest.skip("分词器还没准备")
    from lingyu.tokenizer import load_tokenizer
    from lingyu.chat_template import has_chat_special_tokens
    tok = load_tokenizer(tok_path)
    assert has_chat_special_tokens(tok), (
        "sft.ini 选了 %s 模板，但 %s 里没有 <|im_start|>；"
        "要么改成 plain，要么重新 prepare" % (tpl, tok_path))
