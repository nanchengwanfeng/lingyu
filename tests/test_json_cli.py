# -*- coding: utf-8 -*-
"""--json 结构化输出（审计 1.4）。

引擎把进度输出成一行一个 JSON 事件，GUI 不用再靠正则猜中文日志。
这里测两件事：
1. emit_event 的格式（stdout、一行一个、可 JSON 解析）；
2. 真跑一次 `train --json`（CPU、2 步、最小模型），事件序列对得上。
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lingyu import utils  # noqa: E402


def test_emit_event_writes_one_json_line(capsys):
    utils.set_json_mode(True)
    try:
        utils.emit_event("step", task="train", step=3, total=10,
                         loss=4.25, lr=0.001, tok_s=1234)
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 1
        obj = json.loads(out[0])
        assert obj == {"event": "step", "task": "train", "step": 3,
                       "total": 10, "loss": 4.25, "lr": 0.001, "tok_s": 1234}
    finally:
        utils.set_json_mode(False)


def test_emit_event_is_silent_in_human_mode(capsys):
    utils.set_json_mode(False)
    utils.emit_event("step", task="train", step=1)
    assert capsys.readouterr().out == ""


def test_human_logs_stay_off_stdout_in_json_mode(capsys):
    utils.set_json_mode(True)
    try:
        utils.title("测试标题")
        utils.info("这条不该进 stdout")
        utils.emit_event("done", task="train", steps=2)
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 1 and json.loads(out[0])["event"] == "done"
    finally:
        utils.set_json_mode(False)


@pytest.mark.slow
def test_train_json_emits_start_step_done(tmp_path):
    """真跑 2 步（CPU、小模型）。输出里必须有 start/step/done。

    data/ 是 prepare 产物、不随仓库分发（.gitignore）。CI 的干净 checkout
    上没有它，这个端到端测试就没法真训练 —— 明确跳过而不是给出误导性的红灯。
    """
    if os.environ.get("CI") or not os.path.exists(
            os.path.join(ROOT, "data", "tokenizer.txt")):
        pytest.skip("需要本地 prepared data/（prepare 产物不进仓库）")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-m", "lingyu", "train", "--json",
           "--config", "configs/nano65.ini",
           "--device", "cpu",
           "--set", "train.max_steps=2",
           "--set", "train.batch_size=1",
           "--set", "train.grad_accum=1",
           "--set", "train.log_interval=1",
           "--set", "train.eval_interval=2",
           "--set", "train.ckpt_interval=0",
           "--set", "train.sample_interval=0",
           "--set", "model.block_size=64",
           "--set", "train.out_dir=" + (tmp_path / "out").as_posix(),
           "--set", "data.data_dir=data"]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                          timeout=240)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-3000:]
    events = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            events.append(json.loads(line))
    kinds = [ev.get("event") for ev in events]
    assert "start" in kinds
    assert "step" in kinds
    assert "done" in kinds
    start = events[kinds.index("start")]
    assert start["task"] == "train" and start["max_steps"] == 2
    steps = [ev for ev in events if ev.get("event") == "step"]
    assert [ev["step"] for ev in steps] == [1, 2]
    assert all(ev["total"] == 2 for ev in steps)
    done = events[kinds.index("done")]
    assert done["steps"] == 2
