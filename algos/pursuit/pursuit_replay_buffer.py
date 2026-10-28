# -*- coding: utf-8 -*-
"""经验回放池（ReplayBuffer 角色）。

数据结构，**不含任何算法逻辑** —— 所以可以独立单测、可以整体替换
（换成优先经验回放 SumTree 时，只改这个文件，agent 与训练循环零改动）。

存储单位是「单智能体单步转移」（obs, action, reward, next_obs, done），
因为参数共享 DQN 学的是「一个智能体在某种局部局面下该怎么动」。
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np


class PursuitSearchReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    # ───────────────────────── 写 ─────────────────────────
    def push(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(bool(done))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, obs, actions, rewards, next_obs, done) -> None:
        """一次落一整步的 N 个智能体转移（向量化写入，比 for 快）。"""
        n = len(rewards)
        idx = (np.arange(self.ptr, self.ptr + n)) % self.capacity
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.actions[idx] = np.asarray(actions, dtype=np.int64)
        self.rewards[idx] = np.asarray(rewards, dtype=np.float32)
        self.dones[idx] = float(bool(done))
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    # ───────────────────────── 读 ─────────────────────────
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        assert self.size > 0, "回放池为空"
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[idx], "actions": self.actions[idx],
            "rewards": self.rewards[idx], "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size

    def ready(self, min_size: int) -> bool:
        return self.size >= int(min_size)

    # ───────────────────────── 持久化 ─────────────────────────
    def save(self, path: str) -> None:
        """**原子写**（同 `PursuitCTDEAgent.save`）：npz 的目录表写在文件末尾，
        直写时读者会拿到「zip 无中央目录」的半个包，故一律 tmp + `os.replace`。"""
        tmp = f"{path}.tmp.npz"
        np.savez_compressed(tmp, obs=self.obs, next_obs=self.next_obs,
                            actions=self.actions, rewards=self.rewards, dones=self.dones,
                            ptr=np.array([self.ptr]), size=np.array([self.size]),
                            capacity=np.array([self.capacity]), obs_dim=np.array([self.obs_dim]))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "PursuitSearchReplayBuffer":
        z = np.load(path)
        buf = cls(int(z["capacity"][0]), int(z["obs_dim"][0]))
        buf.obs, buf.next_obs = z["obs"], z["next_obs"]
        buf.actions, buf.rewards, buf.dones = z["actions"], z["rewards"], z["dones"]
        buf.ptr, buf.size = int(z["ptr"][0]), int(z["size"][0])
        return buf
