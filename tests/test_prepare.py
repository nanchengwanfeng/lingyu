"""数据准备里的抽样逻辑测试。

背景（P0 回归）：BPE 训练子集曾经用 `corpus[::step]` 跨步抽样，
把原文相隔很远的两个字拼成邻居，训练出的分词器在 held-out 文本上
压缩比只有 1.160（连续块 1.438）。这条 bug 不会报错、loss 照降，
只有对比压缩比才看得出来，所以必须用测试钉住「块内保持原文相邻」。
"""

from lingyu.prepare import _contiguous_sample


def test_contiguous_sample_keeps_original_adjacency():
    corpus = "".join(chr(0x4E00 + i % 500) for i in range(100_000))
    out = _contiguous_sample(corpus, 1000, n_blocks=4)

    assert len(out) == 1000
    # 4 块 × 250 字符，每一块都必须是原文里连续的一段
    block_len = 250
    for i in range(4):
        block = out[i * block_len:(i + 1) * block_len]
        assert block in corpus
        # 且不能是「跳着抽」：块内字符在原文中的下标必须连续
        start = corpus.index(block)
        assert corpus[start:start + block_len] == block
        assert out[i * block_len + 1] == corpus[start + 1]
    # 覆盖全文：首块贴开头，末块贴结尾（避免只学到开头几本书）
    assert out[:block_len] == corpus[:block_len]
    assert out[-block_len:] == corpus[-block_len:]


def test_contiguous_sample_returns_corpus_when_small_enough():
    corpus = "短语料" * 10
    assert _contiguous_sample(corpus, 10 ** 9) is corpus
    assert _contiguous_sample(corpus, 0) is corpus


def test_contiguous_sample_edge_cases():
    # 只有一块
    assert len(_contiguous_sample("abcdefghij", 4, n_blocks=1)) == 4
    # 配额比块数还小：块数自动收缩，绝不返回空
    out = _contiguous_sample("abcdefghij", 2, n_blocks=16)
    assert 0 < len(out) <= 2


# ---------------------------------------------------------------- JSON 语料
import json as _json
import os as _os

from lingyu.prepare import (extract_json_text, find_text_files,
                            read_corpus_file)


def test_find_text_files_accepts_json_and_markdown(tmp_path):
    for name in ("a.txt", "b.md", "c.json", "d.jsonl", "e.ndjson", "f.bin"):
        (tmp_path / name).write_text("语料", encoding="utf-8")
    names = [_os.path.basename(p) for p in find_text_files(str(tmp_path))]
    assert names == ["a.txt", "b.md", "c.json", "d.jsonl", "e.ndjson"]


def test_extract_json_text_prefers_content_keys():
    text, n = extract_json_text(_json.dumps(
        {"id": "abc", "url": "https://x/y", "text": "三国演义第一回"}))
    assert "三国演义第一回" in text
    assert "abc" not in text and "https" not in text
    assert n == 1


def test_extract_json_text_from_alpaca_jsonl():
    rows = [
        {"id": "1", "instruction": "写一句诗", "input": "", "output": "床前明月光"},
        {"id": "2", "instruction": "翻译", "input": "hello", "output": "你好"},
    ]
    raw = "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows)
    text, n = extract_json_text(raw)
    assert n == 2
    assert "床前明月光" in text and "你好" in text
    assert "写一句诗" in text and "hello" in text
    assert "instruction" not in text and "id" not in text


def test_extract_json_text_messages_and_broken_lines():
    raw = "\n".join([
        _json.dumps({"messages": [{"role": "user", "content": "你好"},
                                  {"role": "assistant", "content": "你好呀"}],
                     "id": "x"}, ensure_ascii=False),
        "{不是合法 json",
        "这一行是混进来的纯文本",
    ])
    text, n = extract_json_text(raw)
    assert n == 1
    assert "你好呀" in text and "这一行是混进来的纯文本" in text
    assert "assistant" not in text


def test_read_corpus_file_extracts_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(_json.dumps({"text": "灵语引擎"}, ensure_ascii=False),
                 encoding="utf-8")
    text, enc, n = read_corpus_file(str(p), json_mode="extract")
    assert text == "灵语引擎"
    assert "JSON" in enc and n == 1


def test_read_corpus_file_keeps_json_structure_by_default(tmp_path):
    payload = {"id": "abc", "text": "灵语引擎", "meta": {"v": 1}}
    p = tmp_path / "data.json"
    p.write_text(_json.dumps(payload, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    text, enc, n = read_corpus_file(str(p))
    # 默认原样：花括号、键名、缩进都在，不做正文抽取
    assert "灵语引擎" in text
    assert "{" in text and "}" in text and '"id"' in text
    assert "JSON" not in enc and n is None


# ==================================================== 2026-10-24 JSON 语料修复


def test_extract_json_text_finds_deep_strings():
    """B-07：JSON 抽正文不能用固定字段名，深层嵌套/任意键也要抽出来。"""
    from lingyu.prepare import extract_json_text
    text = '{"meta": {"rows": [{"inner": {"paragraph": "深层正文甲"}}]},' \
           ' "list": ["乙", 3]}'
    got, recs = extract_json_text(text)
    assert "深层正文甲" in got
    assert "乙" in got
    assert "meta" not in got


def test_read_corpus_file_json_raw_and_extract(tmp_path):
    """B-07：默认原样保留结构；--json-extract 才抽正文。"""
    from lingyu.prepare import read_corpus_file
    p = tmp_path / "a.json"
    p.write_text('{"steps": [{"title": "标题甲", "desc": "说明乙"}]}',
                 encoding="utf-8")
    raw, enc, n_rec = read_corpus_file(str(p))
    assert "{" in raw and "标题甲" in raw, "默认没保留 JSON 结构"
    text, _enc, recs = read_corpus_file(str(p), json_mode="extract")
    assert "标题甲" in text and "说明乙" in text
    assert "{" not in text
    assert recs and recs >= 1


def test_config_json_is_skipped(tmp_path):
    """B-07：mcp.json / package.json 这类配置文件不该进语料。"""
    from lingyu.prepare import extract_json_text, read_corpus_file
    cfg = tmp_path / "mcp.json"
    cfg.write_text('{"mcpServers": {"a": {"command": "npx", "args": ["-y"]}}}',
                   encoding="utf-8")
    text, recs = read_corpus_file(str(cfg), json_mode="extract")[:1] + \
        read_corpus_file(str(cfg), json_mode="extract")[2:3]
    assert text == "" and recs == 0
    assert extract_json_text(cfg.read_text(encoding="utf-8")) == ("", 0)


def test_json_extract_keeps_content_despite_config_like_keys():
    """数据集里同时有 name/version 和 content 时，先挑正文，别当配置丢弃。"""
    from lingyu.prepare import extract_json_text

    text, records = extract_json_text(
        '{"name": "foo", "version": "1.0", "content": "真正的正文"}')
    assert "真正的正文" in text
    assert records == 1


def test_json_extract_skips_package_json_by_filename(tmp_path):
    """按文件名能确定是配置文件时，extract 模式才跳过。"""
    from lingyu.prepare import read_corpus_file

    f = tmp_path / "package.json"
    f.write_text('{"name": "x", "dependencies": {"a": "1"}}', encoding="utf-8")
    text, enc, n = read_corpus_file(str(f), json_mode="extract")
    assert text == ""
    assert "配置文件" in enc


def test_json_extract_falls_back_to_raw_when_no_text(tmp_path):
    """没有正文字段又不是配置文件的 JSON，按原文收，不静默跳过。"""
    from lingyu.prepare import read_corpus_file

    f = tmp_path / "weird.json"
    f.write_text('{"nums": [1, 2, 3], "ok": true}', encoding="utf-8")
    text, enc, n = read_corpus_file(str(f), json_mode="extract")
    assert text.strip()
    assert "无正文字段" in enc


# ==================================================== 2026-11-20 并行读文件


def test_prepare_workers_env_override(monkeypatch):
    """LINGYU_PREPARE_WORKERS=1 强制串行；不设时不超过文件数。"""
    from lingyu.prepare import _prepare_workers

    monkeypatch.setenv("LINGYU_PREPARE_WORKERS", "1")
    assert _prepare_workers(100) == 1
    monkeypatch.setenv("LINGYU_PREPARE_WORKERS", "4")
    assert _prepare_workers(100) == 4
    monkeypatch.setenv("LINGYU_PREPARE_WORKERS", "4")
    assert _prepare_workers(2) == 2
    monkeypatch.delenv("LINGYU_PREPARE_WORKERS", raising=False)
    assert 1 <= _prepare_workers(100) <= 8


def test_parallel_file_reading_keeps_order_and_content(monkeypatch, tmp_path):
    """线程池只并行 I/O，拼接顺序必须与串行完全一致（否则语料顺序会变）。"""
    import lingyu.prepare as prep

    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(6):
        (raw / ("f%d.txt" % i)).write_text("第%d段语料。" % i * 50,
                                           encoding="utf-8")

    seen = []
    real = prep.read_corpus_file

    def wrapped(path, json_mode="raw"):
        seen.append(_os.path.basename(path))
        return real(path, json_mode=json_mode)

    monkeypatch.setattr(prep, "read_corpus_file", wrapped)
    monkeypatch.setenv("LINGYU_PREPARE_WORKERS", "4")
    txt = prep.run_prepare(str(raw), str(tmp_path / "data"), vocab_size=256,
                           verbose=False)
    assert seen == ["f0.txt", "f1.txt", "f2.txt", "f3.txt", "f4.txt", "f5.txt"]
    # run_prepare 返回 None，这里只确认整条管线跑通、产物存在
    assert _os.path.isfile(tmp_path / "data" / "train.bin")

