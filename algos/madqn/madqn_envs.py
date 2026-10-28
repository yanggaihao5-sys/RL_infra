# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：读 COMMON_patch

"""栅格覆盖搜索环境（多智能体，离散 9 动作）。

**这是整个包里唯一放「领域知识」的文件** —— 想换任务（改成巡检/追逃/编队）只改这里。

核心规范（spec §6.3）：奖励在 `step()` 内、行为发生的**同一时刻**拆成**带名字的分量**，
随 `info["reward_components"]` 回传；训练循环**只累加、不重算**。
—— 这是唯一能诊断「奖励函数是否写错」的手段。

环境是**完全确定性的**：起点 / 障碍 / 目标点 / 初始朝向全部由 `seed` 决定，
每轮 reset 都一样 —— 两次跑出不同结果时，可以断言「差异来自算法，不是环境」。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 8 方向 + 原地不动（顺序即动作编号，别随便改：模型权重与编号绑定）
_MOVES: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0),
)


class MADroneSearchEnv:
    """单环境实例（不含并行）。并行/多进程见 madqn_parallel_env.py。"""

    def __init__(self, cfg, seed: Optional[int] = None, max_steps: Optional[int] = None):
        self.cfg = cfg
        self.H = self.W = int(cfg.grid_size)
        self.agent_num = int(cfg.agent_num)
        self.patch = int(cfg.COMMON_patch)
        self.obs_dim = int(cfg.obs_dim)
        self.n_actions = int(cfg.n_actions)
        self.max_steps = int(max_steps if max_steps is not None else cfg.init_max_train_steps)
        self.threshold = float(cfg.coverage_threshold)
        self.seed = int(cfg.seed if seed is None else seed)

        # ── 地图要素：由 seed 决定，每轮 reset 不变（保证环境可复现、可对比）──
        rng = np.random.default_rng(self.seed)
        self.obstacles = self._sample_cells(rng, cfg.obstacle_num, avoid=())
        self.targets = self._sample_cells(rng, cfg.target_num, avoid=tuple(map(tuple, self.obstacles)))
        starts = self._sample_starts(rng, self.agent_num, avoid=tuple(map(tuple, self.obstacles)))
        self.starts = starts
        self.initial_headings = [int(rng.integers(0, 8)) * 45 for _ in range(self.agent_num)]

        self.free_cells = self.H * self.W - len(self.obstacles)
        self._target_set = {tuple(t) for t in map(tuple, self.targets)}
        self.reset()

    # ══════════════════════ 地图生成 ══════════════════════
    def _sample_cells(self, rng, n: int, avoid: Tuple) -> np.ndarray:
        avoid_set = {tuple(map(int, a)) for a in avoid}
        cells: List[Tuple[int, int]] = []
        while len(cells) < n:
            c = (int(rng.integers(0, self.H)), int(rng.integers(0, self.W)))
            if c not in avoid_set and c not in cells:
                cells.append(c)
        return np.array(cells, dtype=np.int64).reshape(-1, 2)

    def _sample_starts(self, rng, n: int, avoid: Tuple) -> List[Tuple[int, int]]:
        avoid_set = {tuple(map(int, a)) for a in avoid}
        picked: List[Tuple[int, int]] = []
        guard = 0
        while len(picked) < n and guard < 10000:
            guard += 1
            c = (int(rng.integers(0, self.H)), int(rng.integers(0, self.W)))
            if c in avoid_set or c in picked:
                continue
            picked.append(c)
        assert len(picked) == n, "地图太小，放不下这么多智能体起点"
        return picked

    # ══════════════════════ reset ══════════════════════
    def reset(self) -> List[np.ndarray]:
        self.visited = np.zeros((self.H, self.W), dtype=np.int8)
        self.pos = [list(map(int, p)) for p in self.starts]
        self.headings = list(self.initial_headings)
        self.steps = 0
        self.done = False
        self.success = False
        self.stagnant = [0] * self.agent_num          # 连续未覆盖新格的步数
        self.hit_milestone = {m: False for m in self.cfg.milestones}
        for r, c in self.pos:                          # 起点视为已覆盖
            self.visited[r, c] = 1
        return self._obs()

    # ══════════════════════ step ══════════════════════
    def step(self, actions) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        assert actions.shape[0] == self.agent_num, "动作数量与 agent_num 不一致"

        comps: List[Dict[str, float]] = [{} for _ in range(self.agent_num)]

        def _acc(i: int, name: str, val: float) -> None:
            if abs(val) > 1e-12:
                comps[i][name] = comps[i].get(name, 0.0) + float(val)

        cov_before = self.coverage()
        new_cells = 0

        # ── ① 移动（撞障碍则原地不动）──
        for i, a in enumerate(actions):
            a = int(np.clip(a, 0, len(_MOVES) - 1))
            dr, dc = _MOVES[a]
            r = int(np.clip(self.pos[i][0] + dr, 0, self.H - 1))
            c = int(np.clip(self.pos[i][1] + dc, 0, self.W - 1))
            if self.obstacle_mask[r, c]:
                _acc(i, "obstacle_penalty", self.cfg.R_obstacle)   # 撞墙：惩罚 + 不动
                continue
            self.pos[i] = [r, c]
            self.headings[i] = self._heading_of(dr, dc, self.headings[i])

        # ── ② 覆盖判定（同一时刻记分量）──
        occupied: Dict[Tuple[int, int], int] = {}
        for i, (r, c) in enumerate(self.pos):
            occupied.setdefault((r, c), i)

        for i, (r, c) in enumerate(self.pos):
            if self.visited[r, c] == 1:
                self.stagnant[i] += 1
                _acc(i, "repeat_penalty", self.cfg.R_repeat)
            else:
                self.visited[r, c] = 1
                self.stagnant[i] = 0
                new_cells += 1
                _acc(i, "new_cell_base", self.cfg.R_new_cell)
                if (r, c) in self._target_set:
                    _acc(i, "target_bonus", self.cfg.R_target_bonus)
            if occupied.get((r, c)) != i:
                _acc(i, "collision_penalty", self.cfg.R_collision)

        # ── ③ 全局形状项 ──
        cov_after = self.coverage()
        for i in range(self.agent_num):
            _acc(i, "step_penalty", self.cfg.R_step)
            if self.stagnant[i] >= self.cfg.R_stagnation_window:
                _acc(i, "stagnation_penalty", self.cfg.R_stagnation)

        for m, done_flag in self.hit_milestone.items():
            if not done_flag and cov_after >= m:
                self.hit_milestone[m] = True
                bonus = {0.5: self.cfg.R_milestone_50,
                         0.75: self.cfg.R_milestone_75,
                         self.threshold: self.cfg.R_milestone_threshold}.get(m, 0.0)
                if bonus:
                    for i in range(self.agent_num):     # 里程碑是全局事件，均分给每个体
                        _acc(i, f"milestone_{int(m*100)}", bonus / self.agent_num)

        self.steps += 1
        if cov_after >= self.threshold:
            self.success = True
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

        rewards = [float(sum(comps[i].values())) for i in range(self.agent_num)]
        info = {
            "coverage": cov_after,
            "new_cells": new_cells,
            "steps": self.steps,
            "success": self.success,
            "done": self.done,
            "reward_components": comps,     # ★ 契约字段名，训练循环据此累加
        }
        return self._obs(), rewards, self.done, info

    # ══════════════════════ 观测 / 工具 ══════════════════════
    @property
    def obstacle_mask(self) -> np.ndarray:
        m = np.zeros((self.H, self.W), dtype=np.int8)
        for r, c in self.obstacles:
            m[r, c] = 1
        return m

    def coverage(self) -> float:
        return float(self.visited.sum()) / max(1, self.free_cells)

    def _patch_view(self, grid: np.ndarray, r: int, c: int) -> np.ndarray:
        p = self.patch
        rr = np.clip(np.arange(r - p, r + p + 1), 0, self.H - 1)
        cc = np.clip(np.arange(c - p, c + p + 1), 0, self.W - 1)
        return grid[np.ix_(rr, cc)].astype(np.float32).reshape(-1)

    def _obs(self) -> List[np.ndarray]:
        obs_mask = self.obstacle_mask
        cov = self.coverage()
        out: List[np.ndarray] = []
        for i, (r, c) in enumerate(self.pos):
            vis_patch = self._patch_view(self.visited, r, c)
            obs_patch = self._patch_view(obs_mask, r, c)
            ego = np.array([r / max(1, self.H - 1), c / max(1, self.W - 1),
                            cov, float(vis_patch.mean())], dtype=np.float32)
            v = np.concatenate([vis_patch, obs_patch, ego]).astype(np.float32)
            assert v.shape[0] == self.obs_dim, f"观测维度 {v.shape[0]} != obs_dim {self.obs_dim}"
            out.append(v)
        return out

    @staticmethod
    def _heading_of(dr: int, dc: int, default: int) -> int:
        if dr == 0 and dc == 0:
            return default
        return int((np.degrees(np.arctan2(dr, dc)) + 360) % 360 // 45) * 45

    # ══════════════════════ 环境快照 ══════════════════════
    def snapshot(self) -> List[Tuple[str, str]]:
        """给日志契约用的环境快照（每轮训练开头打一次）。对比实验前先对齐这些值。"""
        return [
            ("网格尺寸 GRID_SIZE", str(self.H)),
            ("智能体数量 agent_num", str(self.agent_num)),
            ("智能体起始位置", str([list(map(int, p)) for p in self.starts])),
            ("目标点位置", str([list(map(int, p)) for p in self.targets])),
            ("固定障碍物坐标", str([list(map(int, p)) for p in self.obstacles])),
            ("初始朝向 (度)", str(self.initial_headings)),
            ("覆盖率达成阈值", f"{self.threshold:.2f}"),
            ("单回合步数上限", str(self.max_steps)),
            ("随机种子", str(self.seed)),
        ]

    def render(self) -> str:
        """ASCII 地图（调试用）：. 未访 / * 已访 / # 障碍 / T 目标 / 1..n 智能体。"""
        ch = [["." if self.visited[r, c] == 0 else "*" for c in range(self.W)] for r in range(self.H)]
        for r, c in self.obstacles:
            ch[r][c] = "#"
        for r, c in self.targets:
            ch[r][c] = "T"
        for i, (r, c) in enumerate(self.pos):
            ch[r][c] = str((i + 1) % 10)
        return "\n".join("".join(row) for row in ch)
