# -*- coding: utf-8 -*-
"""配置唯一源：派生量、校验闸门。"""
from __future__ import annotations

import os

import pytest

from config import Config


def test_default_derived_quantities():
    cfg = Config()
    cfg.validate()
    assert cfg.obs_dim == 54          # 2×(2·2+1)² + 4
    assert cfg.n_actions == 9
    assert cfg.eps_decay_episodes == int(300 * 0.8) - cfg.COMMON_random_episodes
    assert cfg.run_root == os.path.abspath(os.path.join("runs", "madqn_demo"))
    assert cfg.milestones == (0.5, 0.75, cfg.coverage_threshold)


def test_patch_resizes_observation():
    cfg = Config(COMMON_patch=3)
    assert cfg.obs_dim == 2 * 7 * 7 + 4 == 102


def test_validate_rejects_random_warmup_ge_episodes():
    cfg = Config(episodes=10, COMMON_random_episodes=10)
    with pytest.raises(AssertionError, match="随机预热回合"):
        cfg.validate()


def test_validate_rejects_unknown_mode_and_bad_threshold():
    with pytest.raises(AssertionError, match="未知 mode"):
        Config(mode="evaluate").validate()
    with pytest.raises(AssertionError, match="评测成功判据"):
        Config(eval_coverage_threshold=0.0).validate()


def test_validate_rejects_impossible_step_budget():
    cfg = Config(grid_size=20, init_max_train_steps=10)
    with pytest.raises(AssertionError, match="单回合步数上限"):
        cfg.validate()


def test_validate_rejects_unknown_best_metric():
    with pytest.raises(AssertionError, match="MADQN_best_metric"):
        Config(MADQN_best_metric="reward").validate()
