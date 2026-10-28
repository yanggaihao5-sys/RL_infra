# -*- coding: utf-8 -*-
"""CI 产物闸门本身的自测：合格产物必须全过，坏一处必须失败（不静默放行）。"""
from __future__ import annotations

import json
import os

import verify_run


def _build_run(root, coverage=43.35, obs_coverage=43.35):
    run = root / "runs" / "demo"
    obs = root / "runs" / "obs" / "demo"
    run.mkdir(parents=True)
    obs.mkdir(parents=True)

    (run / "train.log").write_text(
        "链路标识 | test_mode=madqn | 单元测试\n"
        "EP | ep:    0 | R:    10.00 | cov:  5.00% | steps:  20 | loss:0.5000 | curQ:0.000 | td:0.2000\n",
        encoding="utf-8")
    (run / "eval.json").write_text(json.dumps({
        "coverage": coverage, "coverage_std": 1.0, "success_rate": 0.0,
        "steps": 200.0, "runs": 5, "success_rule": "覆盖率 ≥ 0.75",
    }), encoding="utf-8")
    (run / "madqn_best.pth").write_bytes(b"model")
    (run / "madqn_training_metrics_en.png").write_bytes(b"\x89PNG" + b"0" * 12_000)
    (obs / "run.json").write_text(json.dumps({"run_id": "demo", "status": "finished"}),
                                  encoding="utf-8")
    (obs / "metrics.jsonl").write_text(
        json.dumps({"ts": 1, "step": 0, "name": "test/coverage", "value": obs_coverage}) + "\n",
        encoding="utf-8")
    (obs / "events.jsonl").write_text("", encoding="utf-8")
    return run, obs


def test_valid_run_passes_all_checks(tmp_path):
    run, obs = _build_run(tmp_path)
    checks = verify_run.check_run(str(run), str(obs))
    assert len(checks) == 6
    assert all(ok for _, ok, _ in checks), checks


def test_skipped_eval_is_rejected(tmp_path):
    run, obs = _build_run(tmp_path)
    (run / "eval.json").write_text(json.dumps({"skipped": True}), encoding="utf-8")
    results = {name: ok for name, ok, _ in verify_run.check_run(str(run), str(obs))}
    assert results["② eval.json（真评测，非 skipped）"] is False
    assert results["⑥ test/coverage 与 eval.json 一致"] is False


def test_metric_mismatch_is_rejected(tmp_path):
    run, obs = _build_run(tmp_path, coverage=43.35, obs_coverage=40.0)
    results = {name: ok for name, ok, _ in verify_run.check_run(str(run), str(obs))}
    assert results["⑥ test/coverage 与 eval.json 一致"] is False
