# -*- coding: utf-8 -*-
"""1.4 收尾：其余命令的 --json 事件契约测试。"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, cwd=ROOT):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, "-m", "lingyu"] + args,
                          cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          env=env, timeout=180)


def _events(stdout):
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def _corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.jsonl").write_text(
        '{"title":"标题甲","desc":"说明乙"}\n{"title":"第二篇","desc":"说明丙"}\n',
        encoding="utf-8")
    (raw / "b.json").write_text('{"title":"单篇","content":"正文"}',
                                encoding="utf-8")
    (raw / "c.txt").write_text("第一章 纯文本测试", encoding="utf-8")
    return raw


def test_scan_json_emits_file_and_summary(tmp_path):
    raw = _corpus(tmp_path)
    r = _run(["scan", "--raw", str(raw), "--sample-mb", "1", "--json",
              "--json-extract"])
    assert r.returncode == 0, r.stderr
    ev = _events(r.stdout)
    kinds = [e["event"] for e in ev]
    assert "summary" not in kinds
    summary = [e for e in ev if e["event"] == "scan"][-1]
    files = [e for e in ev if e["event"] == "file"]
    assert len(files) == 3
    assert summary["files"] == 3 and summary["json_extract"] is True
    assert summary["chars"] > 0
    a = [e for e in files if e["path"] == "a.jsonl"][0]
    assert a["records"] == 2 and a["chars"] > 0


def test_prepare_json_event_has_meta(tmp_path):
    raw = _corpus(tmp_path)
    data = tmp_path / "data"
    r = _run(["prepare", "--raw", str(raw), "--data", str(data),
              "--vocab-size", "256", "--tokenizer-sample-mb", "1",
              "--val-ratio", "0.1", "--json", "--json-extract"])
    assert r.returncode == 0, r.stderr
    ev = [e for e in _events(r.stdout) if e["event"] == "prepare"]
    assert ev, "missing prepare event"
    e = ev[-1]
    assert e["train_tokens"] > 0 and e["val_tokens"] >= 0
    assert e["vocab_size"] >= 256
    assert e["tokenizer_fingerprint"]
    assert e["json_extract"] is True
    assert (data / "train.bin").is_file()


def test_plan_json_event():
    r = _run(["plan", "--config", "configs/small.ini", "--json"])
    assert r.returncode == 0, r.stderr
    ev = [e for e in _events(r.stdout) if e["event"] == "plan"]
    assert ev, "missing plan event"
    e = ev[-1]
    assert e["params"] > 1_000_000
    assert e["total_bytes"] > 0
    assert e["scope"] in ("pretrain", "post")


def test_doctor_json_event():
    r = _run(["doctor", "--json"])
    assert r.returncode == 0, r.stderr
    ev = [e for e in _events(r.stdout) if e["event"] == "doctor"]
    assert ev and ev[-1]["ok"] is True
    assert ev[-1]["python"]
