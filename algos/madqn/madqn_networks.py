# -*- coding: utf-8 -*-
"""Q 网络结构（QNetwork 角色）。

**改结构 ≠ 改算法** —— 所以结构单独一个文件，便于版本化与替换
（换成 Dueling / Noisy / GRU 只动这里，agent 侧接口不变）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MADQNQNetwork(nn.Module):
    """共享参数的 Q 网络：输入单智能体观测，输出各动作 Q 值。

    参数共享（所有智能体用同一套权重）是本链路的设计选择：
    智能体同质 → 共享能显著加速收敛，且天然支持「训练 2 个、测试 3 个」。
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ───────────────────────── 工具 ─────────────────────────
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def copy_to(self, other: "MADQNQNetwork") -> None:
        other.load_state_dict(self.state_dict())

    @torch.no_grad()
    def soft_update_to(self, other: "MADQNQNetwork", tau: float) -> None:
        for p, q in zip(self.parameters(), other.parameters()):
            q.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
