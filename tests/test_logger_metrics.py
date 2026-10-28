# -*- coding: utf-8 -*-
"""日志契约：logger 写出的行必须能被诊断脚本与观测契约解析。"""
from __future__ import annotations

import json

import pytest

import analyze_training_log as atl
import log_to_metrics
from rlinfra.logger import TrainLogger


def _write_log(path):
    with TrainLogger(str(path), algo="MADQN", echo=False) as logger:
        logger.identity("madqn", "单元测试 | 日志契约")
        logger.overview("cpu", 123.4, 2, 2, 3)
        for ep in range(3):
            logger.ep_line(ep, 10.0 + ep, 5.0 * ep, 20, 0.5, None if ep == 0 else 0.1, 0.2)
        logger.train_line(2, 0.4, 0.3, 0.2, 0.9, 0.1)
        logger.reward_line({"new_cell_base": 1.0, "step_penalty": -0.01})
        logger.best_line(1, 12.5, 40.0, "runs/x/madqn_best.pth")
        logger.test_line(43.35, 58.9, 0.0, 200.0,
                         extra="测试轮数：5 | 地图 每轮随机 | 成功判据 覆盖率 ≥ 0.75")


def test_logger_contract_roundtrip(tmp_path):
    log = tmp_path / "train.log"
    _write_log(log)

    parsed = atl.analyze(str(log))
    assert [row[0] for row in parsed["ep"]] == [0, 1, 2]
    # 契约里 curQ 缺失写 0.000（诊断脚本按数值读，不区分 None）
    assert parsed["ep"][0][5] == pytest.approx(0.0)
    assert parsed["ep"][1][5] == pytest.approx(0.1)
    assert len(parsed["train"]) == 1
    assert parsed["test"]["coverage"] == pytest.approx(43.35)
    assert parsed["test"]["sr"] == pytest.approx(0.0)
    assert len(parsed["best"]) == 1
    assert parsed["reward_comp"]["new_cell_base"] == pytest.approx(1.0)

    metrics = log_to_metrics.parse_log(str(log), run_id="unit")["metrics"]
    n_ep = [m for m in metrics if m["name"] == "ep_reward"]
    assert len(n_ep) == 3 and n_ep[-1]["value"] == pytest.approx(12.0)
    assert any(m["name"] == "reward/step_penalty" for m in metrics)


def test_eval_json_is_single_source_for_test_metrics(tmp_path):
    log = tmp_path / "train.log"
    _write_log(log)
    (tmp_path / "eval.json").write_text(json.dumps({
        "coverage": 43.35, "coverage_std": 14.17, "reward": 58.9,
        "success_rate": 0.0, "steps": 200.0, "runs": 5,
        "success_rule": "覆盖率 ≥ 0.75",
    }), encoding="utf-8")

    parsed = log_to_metrics.parse_log(str(log), run_id="unit")
    test_metrics = [m for m in parsed["metrics"] if m["name"].startswith("test/")]
    names = {m["name"] for m in test_metrics}
    assert {"test/coverage", "test/coverage_std", "test/success_rate", "test/steps"} <= names
    # 测试汇总只进 meta，不再重复写 metrics
    assert parsed["meta"]["test_from_log"]["cov"] == pytest.approx(43.35)
    assert not any(m["name"] == "test_from_log" for m in parsed["metrics"])
