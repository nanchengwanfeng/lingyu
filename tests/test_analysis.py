# -*- coding: utf-8 -*-
"""分析页测试：读 history.csv、算速度、KPI、发现 run 目录。"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("LINGYU_GUI_NO_PROBE", "1")
os.environ.setdefault("LINGYU_GUI_NO_ANIM", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_history(path, rows):
    import io
    with io.open(path, "w", encoding="utf-8", newline="") as fp:
        fp.write("step,train_loss,val_loss,lr,grad_norm,elapsed_sec\n")
        for r in rows:
            fp.write("%s,%s,%s,%s,%s,%s\n" % r)


def test_load_history_parses_rows_and_skips_broken(tmp_path):
    from gui.pages.analysis import load_history
    p = tmp_path / "history.csv"
    _write_history(p, [
        (500, 3.9, 4.2, 1e-3, 1.1, 10.0),
        (1000, 3.5, 4.0, 8e-4, 0.9, 20.0),
    ])
    with open(p, "a", encoding="utf-8") as fp:
        fp.write("not,a,row\n")       # 坏行（训练中途写坏的）
    rows = load_history(str(p))
    assert len(rows) == 2
    assert rows[0] == {"step": 500, "loss": 3.9, "val": 4.2, "lr": 1e-3,
                       "gnorm": 1.1, "elapsed": 10.0}
    assert rows[1]["val"] == 4.0


def test_load_history_missing_file_is_empty():
    from gui.pages.analysis import load_history
    assert load_history("Z:/definitely/not/here/history.csv") == []


def test_speed_series_uses_row_deltas():
    from gui.pages.analysis import load_history, speed_series
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "history.csv")
        _write_history(p, [
            (500, 3.9, 4.2, 1e-3, 1.1, 10.0),
            (1000, 3.5, 4.0, 8e-4, 0.9, 20.0),
            (1500, 3.3, 3.9, 6e-4, 0.8, 30.0),
        ])
        rows = load_history(p)
    sp = speed_series(rows, tokens_per_step=1000)
    # 每行差 500 步 × 1000 token，耗时差 10s → 50000 tok/s
    assert [s for s, _v in sp] == [1000, 1500]
    assert sp[0][1] == pytest.approx(50000.0)


def test_compute_kpis():
    from gui.pages.analysis import compute_kpis
    rows = [
        {"step": 500, "loss": 3.9, "val": 4.2, "lr": 1e-3, "gnorm": 1, "elapsed": 10.0},
        {"step": 1000, "loss": 3.5, "val": 3.8, "lr": 8e-4, "gnorm": 1, "elapsed": 20.0},
        {"step": 1500, "loss": 3.2, "val": 3.9, "lr": 6e-4, "gnorm": 1, "elapsed": 30.0},
    ]
    k = compute_kpis(rows, {"kind": "pretrain", "tokens_per_step": 1000})
    assert k["steps"] == 1500
    assert k["best"] == pytest.approx(3.8)
    assert k["best_step"] == 1000
    assert k["tokens"] == 1_500_000
    assert k["speed"] == pytest.approx(1_500_000 / 30.0)
    assert compute_kpis([], {}) == {}


def test_context_runs_discovers_history_dirs(tmp_path, monkeypatch):
    import json

    out = tmp_path / "out"
    pre = out / "chat_pretrain"
    sft = out / "chat_sft"
    for d, kind, step in ((pre, "pretrain", 10), (sft, "sft", 20)):
        d.mkdir(parents=True)
        _write_history(d / "history.csv", [(step, 3.0, 3.1, 1e-3, 1.0, 5.0)])
        (d / "run.json").write_text(json.dumps({"kind": kind}),
                                    encoding="utf-8")
    (out / "empty_dir").mkdir()     # 没有 history.csv，不该出现

    from gui.context import ctx
    monkeypatch.setattr(ctx, "out_dir", lambda: str(pre))
    monkeypatch.setattr(ctx, "sft_out_dir", lambda: str(sft))
    runs = ctx.runs()
    names = {r["name"]: r["kind"] for r in runs}
    assert names == {"chat_pretrain": "pretrain", "chat_sft": "sft"}
    assert all(r["history"].endswith("history.csv") for r in runs)


def test_analysis_page_builds_and_selects_run(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from gui.pages.analysis import AnalysisPage, load_history

    run = tmp_path / "out" / "demo"
    run.mkdir(parents=True)
    _write_history(run / "history.csv", [
        (100, 4.0, None, 1e-3, 1.0, 4.0),
        (200, 3.7, 3.9, 8e-4, 1.0, 9.0),
    ])
    (run / "run.json").write_text('{"kind": "pretrain", "tokens_per_step": 640}',
                                  encoding="utf-8")

    from gui.context import ctx
    monkeypatch.setattr(ctx, "runs", lambda: [{
        "dir": str(run), "name": "demo", "kind": "pretrain",
        "history": str(run / "history.csv"),
        "meta": {"kind": "pretrain", "tokens_per_step": 640}, "mtime": 1,
    }])

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841

    page = AnalysisPage()
    page.resize(1400, 900)
    page.ensure_built()
    page.on_show()
    app.processEvents()

    assert page.run_combo.count() == 1
    assert page.k_steps.value_label.text() == "200"
    assert page.chart_loss.losses == [4.0, 3.7]
    assert [v for v in page.chart_loss.vlosses] == [3.9]
    assert page.chart_lr.losses == [1e-3, 8e-4]
    assert len(page.chart_speed.losses) == 1
    page.deleteLater()
