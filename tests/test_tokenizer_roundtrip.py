# -*- coding: utf-8 -*-
"""分词器**存盘 → 读回**的一致性。

## 为什么单独一个文件

`tokenizer.txt` 是引擎里唯一一个「写下来再读回去」的格式，
而它用的是「注释 + TSV 混排」的纯文本 —— 这种格式最容易被
**内容本身**破坏。这里把每一个踩过的、以及差点踩到的字符都钉住。

## 这里挡下来的那个 P0

`prepare` 出来的分词器，`train` 一读就说「语料和分词器不配套」：

    词表指纹不一致：语料是 5ba29c265c7797f1，当前分词器是 070211f0f01f563b

根因：读盘时用「行首是不是 `#`」判注释，而词表里**真的有一个 token
就是 `#`**，它那一行 `#\t57\t123` 和注释长得一模一样，被静默跳过。
少一个 token → `fingerprint()` 变了 → `check_data_tokenizer` 拒绝开跑。

于是「刚 prepare 好的语料」**永远训不了** —— 而中文书里有 `#`
（章节号、Markdown、网址残留）再正常不过。

修法：`#` 也进转义表，写成 `\\#`；同时兼容旧文件（靠「第二列是不是数字」认）。
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lingyu.tokenizer import load_tokenizer, train_tokenizer  # noqa: E402


def _roundtrip(text, vocab_size=200, **kw):
    tok = train_tokenizer(text, vocab_size=vocab_size, kind="bpe",
                          verbose=False, min_pair_count=1,
                          max_token_chars=4, **kw)
    p = os.path.join(tempfile.mkdtemp(), "tok.txt")
    tok.save(p)
    return tok, load_tokenizer(p), p


def test_hash_token_survives_save_load():
    """★ 词表里有一个 token 就是 `#` 时，存盘再读回来不许少东西。"""
    tok, back, _ = _roundtrip("中华人民共和国成立了#第一章#今天天气很好。" * 50)
    assert "#" in tok.vocab, "构造的语料里应该有 # 这个 token"
    assert "#" in back.vocab, "`#` 读回来丢了 —— 注释判定误伤了数据行"
    assert back.vocab == tok.vocab, "词表和存盘前不一致"
    assert back.fingerprint() == tok.fingerprint(), (
        "指纹变了（%s -> %s）—— 训练会因此拒绝开跑"
        % (tok.fingerprint(), back.fingerprint()))


def test_hash_at_line_start_is_not_read_as_comment():
    """`#` 那一行在文件里必须**不再以 `#` 开头**。"""
    tok, _, path = _roundtrip("a#b#c" * 200, vocab_size=60)
    lines = open(path, encoding="utf-8").read().splitlines()
    # 找那个 id 对应 `#` 的行
    vid = tok.vocab["#"]
    row = [l for l in lines if l.split("\t")[1:2] == [str(vid)]]
    assert row, "文件里找不到 `#` 那一行"
    assert not row[0].startswith("#"), (
        "`#` 那一行还是以 # 开头 —— 又会被当成注释：%r" % row[0])


def test_every_awkward_token_round_trips():
    """制表符 / 换行 / 回车 / 反斜杠 / `#` 一起验。

    中文语料里换行和制表非常常见，每一个都曾经或差点把词表文件切碎。
    """
    text = "ab\tcd\nef\rgh\\ij#kl" * 40
    tok, back, _ = _roundtrip(text, vocab_size=120)
    assert back.vocab == tok.vocab, "有 token 在存读过程中丢了或串位了"
    assert back.fingerprint() == tok.fingerprint()
    assert back.merges == tok.merges, "合并规则也要原样回来"
    # 行为一致，不只是数据相等
    assert back.encode(text) == tok.encode(text)
    assert back.decode(tok.encode(text)) == tok.decode(tok.encode(text))


def test_fingerprint_is_stable_and_discriminating():
    """指纹得「同一份永远一样、改一点就不一样」。"""
    tok, back, path = _roundtrip("今天天气很好。" * 60, vocab_size=100)
    assert tok.fingerprint() == load_tokenizer(path).fingerprint()
    # 换个语料，指纹必须变
    other, _, _ = _roundtrip("明天要下雨了。" * 60, vocab_size=100)
    assert other.fingerprint() != tok.fingerprint()


def test_legacy_file_with_bare_hash_still_loads():
    """旧版写出来的分词器（`#` 没转义）也必须能正常读回来。

    兼容性承诺：现有分词器永久可用。老文件里那一行是裸的
    `#\\t57\\t3`，靠「第二列是不是数字」认出来。
    """
    p = os.path.join(tempfile.mkdtemp(), "old.txt")
    with open(p, "w", encoding="utf-8") as fp:
        fp.write("# 灵语分词器  JSON 头\n")
        fp.write('{"kind": "bpe", "vocab_size": 8, "meta": {}}\n')
        fp.write("# 词表\n")
        fp.write("<pad>\t0\t0\n")
        fp.write("a\t1\t5\n")
        fp.write("#\t2\t3\n")          # ← 旧格式：裸的 #，没有转义
        fp.write("b\t3\t4\n")
        fp.write("# 合并规则（按学习顺序）\n")
        fp.write("a\tb\n")
    tok = load_tokenizer(p)
    assert tok.vocab.get("#") == 2, "旧文件里的 `#` token 被当成注释吃掉了"
    assert tok.vocab.get("b") == 3
    assert len(tok.vocab) == 4, "读出来的词表项数不对：%s" % sorted(tok.vocab)
    assert tok.merges == [("a", "b")], "注释行不能被当成合并规则"
    assert tok.counts.get("#") == 3, "出现次数也要读对"


def test_comment_lines_still_ignored():
    """兼容旧文件的那条兜底**不能**把真正的注释也吃进来。"""
    p = os.path.join(tempfile.mkdtemp(), "c.txt")
    with open(p, "w", encoding="utf-8") as fp:
        fp.write("# 灵语分词器  JSON 头\n")
        fp.write('{"kind": "bpe", "vocab_size": 4, "meta": {}}\n')
        fp.write("# 词表（每行: token<TAB>id<TAB>出现次数）\n")
        fp.write("a\t0\t1\n")
        fp.write("b\t1\t1\n")
        fp.write("# 合并规则（按学习顺序，每行: 左<TAB>右）\n")
        fp.write("a\tb\n")
    tok = load_tokenizer(p)
    assert set(tok.vocab) == {"a", "b"}, "注释被当成 token 了：%s" % tok.vocab
    assert tok.merges == [("a", "b")]


def test_first_line_must_be_a_comment(tmp_path):
    """格式检查不能松掉 —— 文件被改坏时要报错，不能读出半个词表。"""
    p = tmp_path / "bad.txt"
    p.write_text('{"kind": "bpe"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(str(p))


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tokenizer(str(tmp_path / "nope.txt"))
