# -*- coding: utf-8 -*-
"""环境与智能体契约：观测/奖励分量/终止条件/保存加载。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from config import Config
from algos.madqn import MADroneSearchEnv, MADQNCTDEAgent, DroneSearchReplayBuffer


@pytest.fixture()
def cfg():
    c = Config(episodes=30)
    c.validate()
    return c


def test_reset_observation_shape(cfg):
    env = MADroneSearchEnv(cfg, seed=0, max_steps=10)
    obs = env.reset()
    assert len(obs) == cfg.agent_num
    for o in obs:
        assert o.shape == (cfg.obs_dim,)
        assert o.dtype == np.float32
        assert np.isfinite(o).all()


def test_reward_components_sum_to_reward_and_done_at_budget(cfg):
    env = MADroneSearchEnv(cfg, seed=0, max_steps=15)
    obs = env.reset()
    for _ in range(14):
        obs, rewards, done, info = env.step([8] * cfg.agent_num)   # 8 = 原地不动
        assert not done
    obs, rewards, done, info = env.step([8] * cfg.agent_num)

    assert done and info["done"]
    assert info["steps"] == 15
    assert len(rewards) == len(info["reward_components"]) == cfg.agent_num
    for r, comps in zip(rewards, info["reward_components"]):
        assert r == pytest.approx(sum(comps.values()))
    for comps in info["reward_components"]:
        assert set(comps) <= {
            "new_cell_base", "repeat_penalty", "obstacle_penalty", "collision_penalty",
            "step_penalty", "stagnation_penalty", "target_bonus",
            "milestone_50", "milestone_75", "milestone_90",
        }


def test_same_seed_same_trajectory(cfg):
    def rollout(seed):
        env = MADroneSearchEnv(cfg, seed=seed, max_steps=8)
        obs = env.reset()
        rs = []
        for i in range(8):
            obs, rewards, done, info = env.step([(i + j) % 9 for j in range(cfg.agent_num)])
            rs.append(rewards)
        return obs, rs

    obs_a, rs_a = rollout(7)
    obs_b, rs_b = rollout(7)
    for a, b in zip(obs_a, obs_b):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(rs_a, rs_b)


def test_action_count_is_checked(cfg):
    env = MADroneSearchEnv(cfg, seed=0, max_steps=5)
    env.reset()
    with pytest.raises(AssertionError, match="动作数量"):
        env.step([0])


def test_agent_act_and_save_load_roundtrip(cfg, tmp_path):
    env = MADroneSearchEnv(cfg, seed=0, max_steps=5)
    obs = np.stack(env.reset())
    agent = MADQNCTDEAgent(cfg, cfg.obs_dim, cfg.n_actions, device="cpu")
    actions = agent.act(obs, epsilon=0.0)
    assert actions.shape == (cfg.agent_num,)
    assert actions.dtype == np.int64
    assert ((actions >= 0) & (actions < cfg.n_actions)).all()

    path = tmp_path / "agent.pth"
    agent.save(str(path))
    other = MADQNCTDEAgent(cfg, cfg.obs_dim, cfg.n_actions, device="cpu")
    other.load(str(path), map_location="cpu")
    for a, b in zip(agent.online.state_dict().values(), other.online.state_dict().values()):
        assert torch.equal(a, b)
    assert agent.n_params() == 24713


def test_replay_buffer_push_sample_and_capacity(cfg, tmp_path):
    buf = DroneSearchReplayBuffer(capacity=10, obs_dim=cfg.obs_dim, seed=0)
    for i in range(20):
        buf.push(np.full(cfg.obs_dim, i, dtype=np.float32), i % cfg.n_actions,
                 float(i), np.full(cfg.obs_dim, i + 1, dtype=np.float32), False)
    assert len(buf) == 10
    assert buf.ready(10) and not buf.ready(11)

    batch = buf.sample(4)
    assert batch["obs"].shape == (4, cfg.obs_dim)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["dones"].shape == (4,)

    path = tmp_path / "buffer.npz"
    buf.save(str(path))
    loaded = DroneSearchReplayBuffer.load(str(path))
    assert len(loaded) == len(buf)
