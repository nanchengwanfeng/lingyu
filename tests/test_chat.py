# -*- coding: utf-8 -*-
"""chat.py 的单测（审计 C 修复：非流式显示、prompt 预算、qwen 模板、stop 集）。

chat.py 以前完全没有测试，长 prompt 生成 0 token / --no-stream 不显示回答
这类问题只能靠用户撞上才发现。这里用「假分词器 + 假模型」把行为钉住，
不依赖真实断点，跑得也快。
"""

import pytest

torch = pytest.importorskip("torch")

from lingyu.chat import (_apply_sampling, _generate_stream, _stop_token_ids,
                         _user_marker_cut, build_prompt_ids, chat_once, run_chat)
from lingyu.chat_template import (QWEN_DEFAULT_SYSTEM, check_ckpt_tokenizer,
                                  render, render_for_inference)
from lingyu.tokenizer import IM_END, IM_START


class FakeTok:
    """最小的「按字」分词器：一个字一个 token，id = 100 + ord(ch)。"""

    kind = "char"
    vocab = {}

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [100 + ord(ch) for ch in text]
        if add_bos:
            ids = [1] + ids
        if add_eos:
            ids = ids + [2]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if 100 <= i <= 100 + 0x10FFFF:
                out.append(chr(i - 100))
        return "".join(out)


class BpeLikeTok(FakeTok):
    """带 ChatML 特殊 token 的假分词器（用来测 chatml/qwen 渲染）。"""

    kind = "bpe"
    vocab = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}


class FakeModel:
    """只实现 chat_once 用到的最小接口：cfg + generate()。"""

    def __init__(self, block_size=64, reply="好的", reply_ids=None):
        self.cfg = {"block_size": block_size}
        self.training = False
        self.reply = reply
        self.reply_ids = reply_ids

    def generate(self, idx, max_new_tokens, **kwargs):
        if self.reply_ids is not None:
            ids = list(self.reply_ids)[:max_new_tokens]
        else:
            ids = [100 + ord(ch) for ch in self.reply[:max_new_tokens]]
        extra = torch.tensor([ids], dtype=idx.dtype, device=idx.device)
        return torch.cat([idx, extra], dim=1)


# ---------------------------------------------------------------------------
# 1) 非流式必须把回答打印出来（审计 C04）
# ---------------------------------------------------------------------------

def test_chat_once_non_stream_prints_reply(capsys):
    tok = FakeTok()
    model = FakeModel(block_size=64, reply="好的")
    reply, n = chat_once(model, tok, [{"role": "user", "content": "你好"}],
                         "plain", "cpu", max_new_tokens=2, stream=False)
    out = capsys.readouterr().out
    assert reply == "好的"
    assert n == 2
    assert "好的" in out, "非流式分支必须把回答打印出来，否则用户以为模型没说话"


# ---------------------------------------------------------------------------
# 2) prompt 预算：不能超过 block_size - 给回答预留的空间（审计 C03）
# ---------------------------------------------------------------------------

def test_build_prompt_ids_respects_budget_keeps_system_and_last_question(capsys):
    tok = FakeTok()
    history = [{"role": "system", "content": "你是助手"}]
    for i in range(12):
        history.append({"role": "user", "content": "旧问题%02d" % i})
        history.append({"role": "assistant", "content": "旧回答%02d" % i})
    history.append({"role": "user", "content": "最后一个问题"})

    budget = 60
    ids, dropped = build_prompt_ids(history, tok, "plain",
                                    max_prompt_tokens=budget, block_size=128)
    text = tok.decode(ids)
    assert len(ids) <= budget, "prompt 超预算会让模型看不到 system，甚至生成 0 token"
    assert "你是助手" in text, "system 提示必须保留"
    assert "最后一个问题" in text, "最后一轮用户问题必须保留"
    assert dropped > 0
    assert "已省略前" in capsys.readouterr().out


def test_build_prompt_ids_short_history_untouched():
    tok = FakeTok()
    history = [{"role": "system", "content": "系统"},
               {"role": "user", "content": "你好"}]
    ids, dropped = build_prompt_ids(history, tok, "plain",
                                    max_prompt_tokens=200, block_size=256)
    assert dropped == 0
    assert ids == render_for_inference([{"role": "user", "content": "你好"}],
                                       tok, "plain", system="系统")


def test_build_prompt_ids_truncates_single_huge_message(capsys):
    tok = FakeTok()
    ids, _ = build_prompt_ids([{"role": "user", "content": "问" * 500}],
                              tok, "plain", max_prompt_tokens=50, block_size=64)
    assert len(ids) <= 50, "单条超长消息也必须压进预算，不能整段塞给模型"
    text = tok.decode(ids)
    assert "助手：" in text, "生成提示（助手：）不能被截掉"
    assert "（保留最后" in capsys.readouterr().out


def test_chat_once_long_history_still_generates(capsys):
    tok = FakeTok()
    model = FakeModel(block_size=32, reply="答")
    history = [{"role": "user", "content": "旧" * 200},
               {"role": "user", "content": "新问题"}]
    reply, _ = chat_once(model, tok, history, "plain", "cpu",
                         max_new_tokens=8, stream=False)
    assert reply, "长历史必须被裁到预算内，且仍然生成（不能 0 token）"
    assert "答" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 3) stop 集：流式/非流式共用；plain+BPE 也要停特殊 token（审计 C08）
# ---------------------------------------------------------------------------

def test_stop_ids_include_chat_specials_for_bpe():
    stops = _stop_token_ids(BpeLikeTok(), "plain")
    assert {IM_START, IM_END} <= stops, "plain+BPE 下模型写出 <|im_end|> 也必须停"


def test_stop_ids_char_adds_user_marker():
    class CharTok(FakeTok):
        vocab = {"用": 7}

    assert 7 in _stop_token_ids(CharTok(), "plain")


# ---------------------------------------------------------------------------
# 4) qwen 与 chatml 必须真的不同（审计 C11）
# ---------------------------------------------------------------------------

def test_qwen_adds_default_system_and_differs_from_chatml():
    tok = BpeLikeTok()
    msgs = [{"role": "user", "content": "你好"}]
    chatml_ids, chatml_mask = render(msgs, tok, template="chatml",
                                     add_generation_prompt=True)
    qwen_ids, qwen_mask = render(msgs, tok, template="qwen",
                                 add_generation_prompt=True)

    assert qwen_ids != chatml_ids, "qwen 与 chatml 不能逐字节相同"
    prefix = ([IM_START] + tok.encode("system\n") + tok.encode(QWEN_DEFAULT_SYSTEM)
              + [IM_END] + tok.encode("\n"))
    assert qwen_ids[:len(prefix)] == prefix
    assert qwen_ids[len(prefix):] == chatml_ids, "qwen 的其余结构与 chatml 相同"
    assert qwen_mask[:len(prefix)] == [0] * len(prefix)
    assert len(qwen_ids) == len(qwen_mask)
    assert len(chatml_ids) == len(chatml_mask)


def test_qwen_with_explicit_system_equals_chatml():
    tok = BpeLikeTok()
    msgs = [{"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"}]
    assert (render(msgs, tok, template="qwen", add_generation_prompt=True)[0]
            == render(msgs, tok, template="chatml",
                      add_generation_prompt=True)[0])


def test_qwen_render_dataset_roundtrip_ok():
    """qwen 插入的默认 system 也必须让 ids/mask 对齐。"""
    tok = BpeLikeTok()
    ids, mask = render([{"role": "user", "content": "问题"}], tok,
                       template="qwen")
    assert len(ids) == len(mask)
    assert sum(mask) == 0, "默认 system/user 都不该计入 loss"


# ---------------------------------------------------------------------------
# 5) plain + BPE：跨 token 边界的「用户：」停止串（第二批修复）
# ---------------------------------------------------------------------------

class SplittingBpeTok:
    """把「用户：」切成「用户」+「：」两个 token 的假 BPE 分词器。

    真实 BPE 下「用户」很可能是一个 token、「：」是另一个。只在逐 token
    层面比对停止串会漏停，这里刻意复现这种切分。
    """

    kind = "bpe"
    vocab = {"<|im_start|>": IM_START, "<|im_end|>": IM_END,
             "<pad>": 0, "<unk>": 3}
    PIECES = {"用户": 201, "：": 202, "\n": 203, "好的": 204}
    _ORDER = ("用户", "好的", "：", "\n")

    def __init__(self):
        self._rev = {v: k for k, v in self.PIECES.items()}

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        i = 0
        while i < len(text):
            for piece in self._ORDER:
                if text.startswith(piece, i):
                    ids.append(self.PIECES[piece])
                    i += len(piece)
                    break
            else:
                ids.append(300 + ord(text[i]))
                i += 1
        if add_bos:
            ids = [1] + ids
        if add_eos:
            ids = ids + [2]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i in self._rev:
                out.append(self._rev[i])
            elif 300 <= i <= 300 + 0x10FFFF:
                out.append(chr(i - 300))
        return "".join(out)

    def fingerprint(self):
        return "split-fp"


class FakeLM:
    """按预设序列逐 token 输出的假模型（只实现 _generate_stream 用的接口）。"""

    def __init__(self, next_ids):
        self.cfg = {"block_size": 64, "n_kv_head": 1, "head_dim": 4,
                    "n_layer": 1}
        self.training = False
        self.next_ids = list(next_ids)
        self.calls = 0

    def parameters(self):
        return iter([torch.zeros(1)])

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def __call__(self, inp, cache=None, cache_pos=0, collect_aux=False):
        self.calls += 1
        logits = torch.zeros(1, 1, 600)
        logits[0, 0, self.next_ids[self.calls - 1]] = 50.0
        return logits, None, None


def test_split_user_marker_really_splits_across_tokens():
    tok = SplittingBpeTok()
    assert tok.encode("用户：") == [201, 202], "假分词器必须真的切开「用户：」"


def test_user_marker_cut_direct():
    assert _user_marker_cut("用户：abc") == 0
    assert _user_marker_cut("好的\n用户：x") == 2
    assert _user_marker_cut("好的用户：x") is None
    assert _user_marker_cut("") is None


def test_stream_stops_at_split_user_marker_from_start(capsys):
    tok = SplittingBpeTok()
    model = FakeLM([201, 202, 204, 204])
    text = _generate_stream(model, tok, torch.tensor([[100]]), 6,
                            1.0, 0, 1.0, set(), 1.0, 1)
    assert text == ""
    assert model.calls == 2, "「用户」「：」到齐就该停，不能继续生成后面的 token"
    out = capsys.readouterr().out
    assert "用户" not in out, "半截停止串也不该显示到屏幕上"


def test_stream_stops_at_split_user_marker_after_reply(capsys):
    tok = SplittingBpeTok()
    model = FakeLM([204, 203, 201, 202, 204])
    text = _generate_stream(model, tok, torch.tensor([[100]]), 6,
                            1.0, 0, 1.0, set(), 1.0, 1)
    assert text == "好的"
    assert model.calls == 4
    out = capsys.readouterr().out
    assert "好的" in out
    assert "用户" not in out


def test_stream_stops_at_eos(capsys):
    tok = SplittingBpeTok()
    model = FakeLM([2, 204, 204])
    text = _generate_stream(model, tok, torch.tensor([[100]]), 6,
                            1.0, 0, 1.0, set(), 1.0, 1)
    assert text == ""
    assert model.calls == 1, "EOS 必须立刻停（与 model.generate 一致）"


def test_stream_flushes_held_tail_on_normal_end(capsys):
    """压住的前缀在正常结束时必须补出来，不能吞字。"""
    tok = SplittingBpeTok()
    model = FakeLM([204, 203, 300 + ord("x")])
    text = _generate_stream(model, tok, torch.tensor([[100]]), 3,
                            1.0, 0, 1.0, set(), 1.0, 1)
    assert text == "好的\nx"
    assert "好的\nx" in capsys.readouterr().out


def test_chat_once_non_stream_truncates_split_user_marker(capsys):
    tok = SplittingBpeTok()
    model = FakeModel(block_size=64, reply_ids=tok.encode("好的\n用户：继续"))
    reply, _ = chat_once(model, tok, [{"role": "user", "content": "你好"}],
                         "plain", "cpu", max_new_tokens=8, stream=False)
    out = capsys.readouterr().out
    assert reply == "好的"
    assert "好的" in out
    assert "用户" not in out, "--no-stream 也不能把自问自答显示出来"


# ---------------------------------------------------------------------------
# 6) 采样口径与 model.generate 对齐（第二批修复）
# ---------------------------------------------------------------------------

def test_apply_sampling_top_p_keeps_only_nucleus():
    # 概率 0.5 / 0.3 / 0.2，top_p=0.75 → 只该保留第一个（generate 的语义：
    # cum > p 的丢掉；旧 chat 逻辑用 cum_prev > p，会多留一个候选）。
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    out = _apply_sampling(logits.clone(), [], 1.0, None, 0.75, 1.0)
    finite = torch.isfinite(out[0])
    assert finite.tolist() == [True, False, False]


def test_apply_sampling_temperature_zero_uses_1e_5_like_generate():
    logits = torch.tensor([[1.0, 2.0]])
    out = _apply_sampling(logits.clone(), [], 0.0, None, None, 1.0)
    assert torch.allclose(out, torch.tensor([[1e5, 2e5]]))


def test_apply_sampling_repetition_penalty_both_signs():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = _apply_sampling(logits.clone(), [0, 1], 1.0, None, None, 2.0)
    assert torch.allclose(out, torch.tensor([[1.0, -4.0, 1.0]]))


def test_apply_sampling_top_k_keeps_k_highest():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    out = _apply_sampling(logits.clone(), [], 1.0, 2, None, 1.0)
    assert torch.isfinite(out[0]).tolist() == [False, False, True, True]


# ---------------------------------------------------------------------------
# 7) 找不到断点时的提示要讲清「先 train 再 sft」
# ---------------------------------------------------------------------------

def test_run_chat_missing_ckpt_explains_train_then_sft(tmp_path):
    cfg = {"post": {"out_dir": str(tmp_path / "no_sft")},
           "train": {"out_dir": str(tmp_path / "no_train")}}
    with pytest.raises(FileNotFoundError) as ei:
        run_chat(cfg)
    msg = str(ei.value)
    assert "lingyu train" in msg
    assert "lingyu sft" in msg
    assert "预训练" in msg and "SFT" in msg


# ---------------------------------------------------------------------------
# 8) 加载对话断点必须核对分词器指纹（C02）
# ---------------------------------------------------------------------------

def test_check_ckpt_tokenizer_rejects_mismatch():
    tok = SplittingBpeTok()
    with pytest.raises(ValueError) as ei:
        check_ckpt_tokenizer({"tokenizer_fingerprint": "bad"}, tok,
                             where="对话断点")
    assert "不一致" in str(ei.value)


def test_check_ckpt_tokenizer_accepts_match():
    tok = SplittingBpeTok()
    assert check_ckpt_tokenizer(
        {"tokenizer_fingerprint": tok.fingerprint()}, tok) is True


def test_check_ckpt_tokenizer_missing_fingerprint_warns(capsys):
    tok = SplittingBpeTok()
    assert check_ckpt_tokenizer({}, tok) is False
    assert "无法核对" in capsys.readouterr().out


def test_run_chat_rejects_ckpt_with_wrong_tokenizer_fingerprint(tmp_path):
    """★ 端到端负例：断点指纹对不上时 run_chat 必须拒绝，不能静默加载。"""
    from lingyu.model import LingYu, build_config
    from lingyu.tokenizer import train_tokenizer

    tok = train_tokenizer("用户：助手：你好光合作用植物能量" * 20, 120,
                          kind="char", verbose=False)
    tok_path = tmp_path / "tokenizer.txt"
    tok.save(str(tok_path))

    mcfg = build_config(vocab_size=tok.vocab_size, n_layer=1, n_embd=16,
                        n_head=2, n_kv_head=1, block_size=32)
    out = tmp_path / "out_sft"
    out.mkdir()
    torch.save({"config": mcfg, "model": LingYu(mcfg).state_dict(),
                "step": 1, "tokenizer_path": str(tok_path),
                "tokenizer_fingerprint": "0000000000000000"},
               str(out / "ckpt_sft.pt"))
    cfg = {"data": {"tokenizer_path": str(tok_path)},
           "post": {"out_dir": str(out), "template": "plain"},
           "train": {"out_dir": str(tmp_path / "train")}}
    with pytest.raises(ValueError) as ei:
        run_chat(cfg, device="cpu")
    msg = str(ei.value)
    assert "不一致" in msg
    assert "--set data.tokenizer_path" in msg, "报错必须给出人话出路"
