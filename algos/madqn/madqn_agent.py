# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：读 COMMON_*

"""智能体与训练逻辑（agent 角色）：act / learn / 保存加载。

网络是它的组件（madqn_networks），数据从回放池来（madqn_replay_buffer）。

设计要点：
  · **三套网络**：online（学习）/ target（算 TD 目标）/ rollout（后台线程专用副本）。
    rollout 副本是为了让 actor 线程与 learner 线程**不共享可变权重**（只做单向同步），
    避免「边学边采样」下的隐性数据竞争 —— 代价是策略滞后，可接受。
  · **Double DQN**：动作由 online 选、价值由 target 估，抑制 Q 值过估计。
  · 参数共享：一次 forward 出 N 个智能体的 Q，天然批量化。
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .madqn_networks import MADQNQNetwork


class MADQNCTDEAgent:
    """CTDE 风格的参数共享 DQN 智能体（集中训练、分散执行）。"""

    def __init__(self, cfg, obs_dim: int, n_actions: int, device: Optional[str] = None):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online = MADQNQNetwork(self.obs_dim, self.n_actions, cfg.COMMON_hidden_dim).to(self.device)
        self.target = MADQNQNetwork(self.obs_dim, self.n_actions, cfg.COMMON_hidden_dim).to(self.device)
        self.rollout = MADQNQNetwork(self.obs_dim, self.n_actions, cfg.COMMON_hidden_dim).to(self.device)
        self.online.copy_to(self.target)
        self.online.copy_to(self.rollout)

        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.COMMON_lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.grad_steps = 0
        self.epsilon = float(cfg.COMMON_eps_start)

    # ══════════════════════ 执行侧（分散）══════════════════════
    def sync_rollout(self) -> None:
        """learner → actor 单向同步（actor 线程调用前先同步一次）。"""
        self.online.copy_to(self.rollout)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: Optional[float] = None,
            net: str = "rollout", greedy: bool = False) -> np.ndarray:
        eps = self.epsilon if epsilon is None else float(epsilon)
        obs = np.asarray(obs, dtype=np.float32)
        n = obs.shape[0]
        if greedy or eps <= 0.0:
            actions = self._q(net, obs).argmax(dim=1)
        else:
            explore = torch.rand(n, device=self.device) < eps
            random_a = torch.randint(0, self.n_actions, (n,), device=self.device)
            greedy_a = self._q(net, obs).argmax(dim=1)
            actions = torch.where(explore, random_a, greedy_a)
        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def _q(self, net: str, obs: np.ndarray) -> torch.Tensor:
        module = {"online": self.online, "target": self.target, "rollout": self.rollout}[net]
        return module(torch.as_tensor(obs, device=self.device))

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(np.clip(eps, 0.0, 1.0))

    # ══════════════════════ 学习侧（集中）══════════════════════
    def learn(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        dev = self.device
        obs = torch.as_tensor(batch["obs"], device=dev)
        next_obs = torch.as_tensor(batch["next_obs"], device=dev)
        actions = torch.as_tensor(batch["actions"], device=dev).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], device=dev).unsqueeze(1)
        dones = torch.as_tensor(batch["dones"], device=dev).unsqueeze(1)

        cur_q = self.online(obs).gather(1, actions)
        with torch.no_grad():
            next_a = self.online(next_obs).argmax(dim=1, keepdim=True)      # Double DQN：online 选动作
            next_q = self.target(next_obs).gather(1, next_a)                # target 估价值
            td_target = rewards + self.cfg.COMMON_gamma * next_q * (1.0 - dones)

        td = td_target - cur_q
        loss = self.loss_fn(cur_q, td_target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()

        self.grad_steps += 1
        if self.grad_steps % self.cfg.COMMON_target_update == 0:
            self.online.copy_to(self.target)                                # 硬更新

        return {"loss": float(loss.item()), "cur_q": float(cur_q.mean().item()),
                "tgt_q": float(td_target.mean().item()), "td": float(td.abs().mean().item())}

    # ══════════════════════ 持久化 ══════════════════════
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        """**原子写**：先写 `<path>.tmp` 再 `os.replace()` 换名。

        为什么必须这样（2026-09-30 加入）：
          高算作业**运行时**就要把 checkpoint 拉回本地看进度。`torch.save` 直写时，
          读者有 1~2 秒窗口拿到**半个文件**（`torch.load` 报 `unexpected EOF`），
          而且不报错、只是解不开 —— 极易被误判成「模型坏了」。
          `os.replace` 在同一文件系统内是原子操作，读者要么看到旧文件、要么看到新文件。
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "obs_dim": self.obs_dim, "n_actions": self.n_actions,
            "grad_steps": self.grad_steps, "epsilon": self.epsilon,
        }
        if extra:
            payload["extra"] = extra
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)          # 原子换名：读者永远看不到写了一半的文件

    def load(self, path: str, map_location: Optional[str] = None) -> dict:
        payload = torch.load(path, map_location=map_location or self.device, weights_only=False)
        assert payload["obs_dim"] == self.obs_dim, \
            f"模型观测维度 {payload['obs_dim']} != 当前环境 {self.obs_dim}（环境配置改过？）"
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload.get("target", payload["online"]))
        self.online.copy_to(self.rollout)
        self.grad_steps = int(payload.get("grad_steps", 0))
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        return payload.get("extra", {})

    def n_params(self) -> int:
        return self.online.n_params()
