# -*- coding: utf-8 -*-
"""对话模板测试。

重点在两件事：
1. **loss mask 划得对**（只对助手回复算 loss）—— 划错了训练照跑、loss 照降，
   但模型学的是「复述问题」，你很难从 loss 曲线上看出来。
2. **老分词器要给出说人话的错**，不能悄悄降级成别的模板。
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lingyu.chat_template import (ROLES, TEMPLATES, _extract_messages,
                                  available_templates, check_ckpt_tokenizer,
                                  check_template_usable,
                                  has_chat_special_tokens, load_jsonl,
                                  needs_special_tokens, normalize_messages,
                                  render, render_dataset)
from lingyu.tokenizer import (CharTokenizer, IM_END, IM_START, SPECIAL_TOKENS,
                              train_tokenizer)

SAMPLE = [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user", "content": "光合作用是什么？"},
    {"role": "assistant", "content": "植物把光能变成化学能的过程。"},
]


@pytest.fixture(scope="module")
def tok():
    """新建的分词器 —— 带 <|im_start|> 等特殊 token。

    ⚠️ 训练文本里**必须包含角色标记用到的所有字**（用/户/助/手/系/统/：），
    否则角色标记会被编码成 <unk>，测试看到的就是一堆占位符。
    （这不是引擎的问题：char 分词器只认识训练语料里出现过的字。）
    """
    text = ("用户：助手：系统："
            "光合作用是植物把光能变成化学能的过程。你是一个乐于助人的助手。"
            "植物动物细胞能量阳光叶绿体二氧化碳水氧气葡萄糖猫娘。"
            "你好请问有什么可以帮你的吗很高兴为你解答这个一甲乙丙丁补充四"
            "只有问题没有回答坏掉的一行好行测试答问长期短") * 40
    return train_tokenizer(text, 400, kind="char", verbose=False)


@pytest.fixture(scope="module")
def old_tok():
    """模拟**老分词器**：词表里没有 <|im_start|>。"""
    t = CharTokenizer()
    t.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for i, ch in enumerate("光合作用是什么植物把能变成化学的过程你是一个乐于助人助手"):
        t.vocab[ch] = len(t.vocab)
    t._rebuild_reverse()
    return t


# ==================================================== 特殊 token

def test_special_tokens_are_appended_not_reordered():
    """★ 特殊 token 只能往后加。往中间插会让所有既有语料/断点错位。"""
    assert SPECIAL_TOKENS[:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]
    assert SPECIAL_TOKENS[IM_START] == "<|im_start|>"
    assert SPECIAL_TOKENS[IM_END] == "<|im_end|>"


def test_new_tokenizer_has_chat_tokens(tok):
    assert has_chat_special_tokens(tok)
    assert tok.vocab["<|im_start|>"] == IM_START


def test_old_tokenizer_lacks_chat_tokens(old_tok):
    assert not has_chat_special_tokens(old_tok)


def test_fingerprint_changes_when_ids_shift(tok, old_tok):
    """★ 指纹必须能发现「词表大小一样但 ID 分配不同」。

    这正是 check_data_tokenizer 原本漏掉的情况：老分词器 4 个特殊 token、
    新分词器 6 个，两边 vocab_size 可能都是 8192，可每个汉字的 ID 都差 2。
    """
    assert tok.fingerprint() != old_tok.fingerprint()
    # 同一个分词器反复算必须一样（否则每次 prepare 都像换了词表）
    assert tok.fingerprint() == tok.fingerprint()
    assert len(tok.fingerprint()) == 16


def test_fingerprint_covers_merges(tok):
    """BPE 合并规则也要进指纹：词表一样、合并顺序不同 = 切法不同。"""
    import copy
    t2 = copy.deepcopy(tok)
    t2.merges = list(t2.merges) + [("测", "试")]
    assert t2.fingerprint() != tok.fingerprint()


# ==================================================== loss mask

def test_mask_only_covers_assistant_reply(tok):
    """★ 核心语义：只有助手回复那一段 mask 是 1。"""
    ids, mask = render(SAMPLE, tok, template="plain")
    assert len(ids) == len(mask), "ids 和 mask 必须等长"

    # 逐段核对：把 mask 覆盖的 token 解码出来，应该正好是助手回复
    on = [i for i, m in enumerate(mask) if m == 1]
    assert on, "一个要学的 token 都没有"
    # 连续的 1
    assert on == list(range(on[0], on[-1] + 1)), "mask 应该是连续一段"

    learned = tok.decode([ids[i] for i in on])
    assert "化学能" in learned, "被监督的部分应该是助手回复，实际是 %r" % learned
    assert "光合作用是什么" not in learned, (
        "问题被算进 loss 了 —— 模型会去学「复述问题」：%r" % learned)


def test_mask_excludes_role_markers(tok):
    """「助手：」这几个字不算 loss —— 那是格式，不是内容。"""
    ids, mask = render(SAMPLE, tok, template="plain")
    text = tok.decode(ids)
    # 找到「助手：」在 ids 里的位置，它的 mask 必须是 0
    head = tok.encode("助手：")
    for start in range(len(ids) - len(head) + 1):
        if ids[start:start + len(head)] == head:
            assert all(m == 0 for m in mask[start:start + len(head)]), (
                "角色标记「助手：」被算进 loss 了")
            break
    assert "用户：" in text and "助手：" in text


def test_chatml_mask_learns_im_end_but_not_role(tok):
    """ChatML：助手的 <|im_end|> 要学（那是「说完了」的信号），
    角色名的 token 不学。"""
    ids, mask = render(SAMPLE, tok, template="chatml")
    assert IM_START in ids and IM_END in ids
    # 每个 <|im_start|> 都不算 loss
    for i, v in enumerate(ids):
        if v == IM_START:
            assert mask[i] == 0
    # 最后那个 <|im_end|>（在助手回复之后）要算
    last_end = len(ids) - 1 - list(reversed(ids)).index(IM_END)
    assert mask[last_end] == 1, "助手段落的 <|im_end|> 应该算 loss"


def test_generation_prompt_has_no_mask(tok):
    """推理时补的「助手：」不该带 mask（那儿还没有答案）。"""
    ids, mask = render(SAMPLE[:2], tok, template="plain",
                       add_generation_prompt=True)
    assert tok.decode(ids).rstrip().endswith("助手：")
    assert mask[-1] == 0


def test_plain_ends_with_eos_when_training(tok):
    """训练时不补生成提示，而是以 <eos> 收尾 —— 教模型「说完要停」。"""
    ids, mask = render(SAMPLE, tok, template="plain")
    assert ids[-1] == 2, "末尾应该是 EOS"
    assert mask[-1] == 1, "「该停了」也是要学的一部分"


def test_all_templates_produce_aligned_ids_and_mask(tok):
    for name in TEMPLATES:
        ids, mask = render(SAMPLE, tok, template=name)
        assert len(ids) == len(mask) > 0, name
        assert sum(mask) > 0, "%s 模板没产出任何要学的 token" % name
        assert all(v >= 0 for v in ids), name


# ==================================================== 老分词器

def test_chatml_on_old_tokenizer_raises_actionable_error(old_tok):
    """★ 不能自动降级 —— 要报错并给出两条明确出路。"""
    with pytest.raises(ValueError) as ei:
        render(SAMPLE, old_tok, template="chatml")
    msg = str(ei.value)
    assert "prepare" in msg and "plain" in msg
    assert "无效" in msg or "失效" in msg, "要提醒重新 prepare 会让旧断点失效"


def test_plain_works_on_old_tokenizer(old_tok):
    """plain 不需要特殊 token，老分词器必须能用。"""
    ids, mask = render(SAMPLE, old_tok, template="plain")
    assert sum(mask) > 0


def test_template_availability_reporting(old_tok, tok):
    rep_old = dict((n, ok) for n, ok, _ in available_templates(old_tok))
    assert rep_old["plain"] and not rep_old["chatml"]
    rep_new = dict((n, ok) for n, ok, _ in available_templates(tok))
    assert rep_new["plain"] and rep_new["chatml"]


def test_needs_special_tokens():
    assert not needs_special_tokens("plain")
    assert needs_special_tokens("chatml")
    assert needs_special_tokens("qwen")


# ==================================================== 消息规范化

def test_normalize_accepts_chinese_roles():
    msgs = normalize_messages([{"role": "用户", "content": "嗨"},
                               {"role": "助手", "content": "你好"}])
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_normalize_accepts_from_value_form():
    msgs = normalize_messages([{"from": "human", "value": "嗨"},
                               {"from": "gpt", "value": "你好"}])
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_normalize_rejects_bad_role_with_position():
    with pytest.raises(ValueError) as ei:
        normalize_messages([{"role": "user", "content": "a"},
                            {"role": "机器人", "content": "b"}])
    assert "第 1 条" in str(ei.value), str(ei.value)


def test_normalize_rejects_missing_content():
    with pytest.raises(ValueError):
        normalize_messages([{"role": "user"}])


def test_system_can_be_passed_separately(tok):
    ids, mask = render(SAMPLE[1:], tok, template="plain", system="你是猫娘。")
    text = tok.decode(ids)
    assert "你是猫娘。" in text
    assert "系统：" in text


# ==================================================== 数据集读取

def test_load_jsonl_accepts_all_documented_shapes(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join([
        '{"messages":[{"role":"user","content":"一"},{"role":"assistant","content":"甲"}]}',
        '{"conversations":[{"from":"human","value":"二"},{"from":"gpt","value":"乙"}]}',
        '{"prompt":"三","response":"丙"}',
        '{"instruction":"四","input":"补充","output":"丁"}',
        '// 这是注释，应该被跳过',
        '',
    ]), encoding="utf-8")
    rows = load_jsonl(str(p))
    assert len(rows) == 4
    assert all(len(r) == 2 for r in rows)
    assert rows[3][0]["content"] == "四\n补充"


def test_load_jsonl_reports_line_number_on_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"prompt":"好","response":"行"}\n{坏掉的一行}\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "第 2 行" in str(ei.value)


def test_load_jsonl_unknown_shape_lists_accepted_formats(tmp_path):
    p = tmp_path / "weird.jsonl"
    p.write_text('{"foo":"bar"}\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    msg = str(ei.value)
    assert "messages" in msg and "prompt" in msg and "foo" in msg


def test_load_jsonl_empty_file_is_an_error(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "一条样本都没有" in str(ei.value)


# ==================================================== 数据集渲染

def test_render_dataset_stats(tok):
    rows = [SAMPLE, SAMPLE, [{"role": "user", "content": "只有问题没有回答"}]]
    out = render_dataset(rows, tok, template="plain")
    assert out["stats"]["n_samples"] == 2
    assert out["skipped"] == 1, "没有助手回复的样本应该被丢掉"
    assert "没有助手回复" in " ".join(out["skip_reasons"])
    r = out["stats"]["supervised_ratio"]
    assert 0 < r < 1, "被监督比例应该在 0 和 1 之间，实际 %s" % r


def test_sample_without_assistant_has_empty_mask(tok):
    """★ 只有 user 的样本必须产出全 0 的 mask。

    踩过的坑：末尾无条件补 `<eos>` 且 mask=1，于是这种样本也有 1 个
    「要学的 token」。后果有两层：
      1. `render_dataset` 判定「有要学的部分」，不会丢掉它；
      2. 模型学到的是「用户说完话，我应该回一个 EOS」= 被问了就别吭声。

    mask 全 0 是「这条样本没价值」的信号，不能被污染。
    """
    ids, mask = render([{"role": "user", "content": "只有问题"}], tok,
                       template="plain")
    assert sum(mask) == 0, "只有 user 的样本不该有任何要学的 token"


def test_render_dataset_drops_too_long(tok):
    long_row = [{"role": "user", "content": "问"},
                {"role": "assistant", "content": "答" * 500}]
    out = render_dataset([long_row], tok, template="plain", max_seq_len=50)
    assert out["stats"]["n_samples"] == 0
    assert "max_seq_len" in " ".join(out["skip_reasons"])


def test_render_dataset_keeps_fitting_samples(tok):
    out = render_dataset([SAMPLE], tok, template="plain", max_seq_len=4096)
    assert out["stats"]["n_samples"] == 1
    assert out["skipped"] == 0


# ==================================================== 断点/分词器指纹（C02）

def test_check_ckpt_tokenizer_accepts_matching_fingerprint(tok):
    ck = {"tokenizer_fingerprint": tok.fingerprint()}
    assert check_ckpt_tokenizer(ck, tok) is True


def test_check_ckpt_tokenizer_rejects_mismatch(tok):
    """★ 同词表大小、不同 token id 的分词器必须挡住。

    这是最危险的一种错配：加载形状完全对得上，loss 照降，
    但每个汉字都指向别的 id —— 输出全是乱码，查不出原因。
    """
    with pytest.raises(ValueError) as ei:
        check_ckpt_tokenizer({"tokenizer_fingerprint": "deadbeefdeadbeef"}, tok,
                             where="预训练断点")
    msg = str(ei.value)
    assert "不一致" in msg and "预训练断点" in msg
    # 必须给两条人话出路
    assert "data.tokenizer_path" in msg
    assert "--ckpt" in msg or "--init-from" in msg


def test_check_ckpt_tokenizer_warns_when_fingerprint_missing(tok, capsys):
    """老断点没有指纹时不能静默通过，要明确说「无法核对」。"""
    assert check_ckpt_tokenizer({}, tok, where="对话断点") is False
    out = capsys.readouterr().out
    assert "无法核对" in out and "对话断点" in out


# ==================================================== system_prompt（C07）

SFT_ROW = [{"role": "user", "content": "光合作用是什么？"},
           {"role": "assistant", "content": "植物把光能变成化学能的过程。"}]


def test_render_dataset_injects_system_prompt(tok):
    """post.system_prompt 必须渲染进每条样本（训练与推理看到同一个前缀）。"""
    out = render_dataset([SFT_ROW], tok, template="plain",
                         system="你是一个乐于助人的助手。")
    s = out["samples"][0]
    text = tok.decode(s["ids"])
    assert "你是一个乐于助人的助手" in text
    # system 是格式不是内容，不该算进 loss
    assert "你是一个乐于助人的助手" not in tok.decode(
        [i for i, m in zip(s["ids"], s["mask"]) if m == 1])


def test_render_dataset_system_works_for_all_templates(tok):
    for tpl in ("plain", "chatml", "qwen"):
        out = render_dataset([SFT_ROW], tok, template=tpl, system="你是助手")
        assert out["stats"]["n_samples"] == 1, tpl
        text = tok.decode(out["samples"][0]["ids"])
        assert text.count("你是助手") == 1, "%s 模板下 system 应该只出现一次" % tpl


# ==================================================== loss_on_prompt（C06）

def test_loss_on_prompt_marks_every_token(tok):
    """loss_on_prompt=true 时整条样本都学（含问题/格式）。"""
    off = render_dataset([SFT_ROW], tok, template="plain")["samples"][0]
    on = render_dataset([SFT_ROW], tok, template="plain",
                        loss_on_prompt=True)["samples"][0]
    assert len(on["ids"]) == len(off["ids"])
    assert sum(on["mask"]) == len(on["ids"]), "开启后不该还有 mask=0 的位置"
    assert 0 < sum(off["mask"]) < len(off["ids"]), "默认只学回答"


def test_loss_on_prompt_affects_supervised_ratio(tok):
    out_off = render_dataset([SFT_ROW, SFT_ROW], tok, template="plain")
    out_on = render_dataset([SFT_ROW, SFT_ROW], tok, template="plain",
                            loss_on_prompt=True)
    assert out_on["stats"]["supervised_ratio"] == 1.0
    assert out_off["stats"]["supervised_ratio"] < 1.0
