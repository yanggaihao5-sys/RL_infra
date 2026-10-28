# -*- coding: utf-8 -*-
# 🕒 v2 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：读 COMMON_patch（原 MADQN_patch 的历史债已清） ｜ 上一版：v1 · 2026-10-17 16:43

"""追逃域（pursuit）：两机搜索一片地图并**围捕一个会移动的目标**。

这是 2026-10-17「迁移演练」的产物（见 `docs/迁移演练.md`）：
从 `algos/madqn/` 机械复制而来，**只有本文件是实质改写**，其余六个文件只做了前缀重命名。

与覆盖任务（`algos/madqn/`）的差别 —— 全部集中在“领域”这一层：

| 维度 | madqn（覆盖） | pursuit（追逃） |
|---|---|---|
| 目标 | 静态高价值点 `targets` | **会移动**的逃逸目标 `evader`（每 `PURSUIT_evader_period` 步走一步） |
| 成功 | 覆盖率 ≥ `threshold` | **捕获**（曼哈顿距离 ≤ `PURSUIT_capture_radius`） |
| 终止 | 达阈值 或 步数满 | 捕获 或 步数满 |
| `coverage` | 已覆盖格 / 可覆盖格 | **已搜查格 / 可搜查格**（仍是进度指标，字段名不变） |
| 观测 ego 4 维 | 行 / 列 / 覆盖率 / 局部已访均值 | 行 / 列 / **目标相对行差** / **目标相对列差** |

🔴 契约不变的部分（改它们会打断日志/训练设施）：
  · `reset()` / `step()` 的返回结构；`info` 里必须有 `coverage` / `success` / `reward_components`；
  · `obs_dim`（= 2×(2p+1)² + 4）与 `n_actions`（9）；
  · `snapshot()` 返回 (中文键, 字符串) 列表；
  · 奖励分量在 `step()` 内、行为发生**同一时刻**拆开（铁律 2）。

诚实短板（写进演练报告 §6）：目标位置是**全局可见**的（无部分可观性），
所以这不是完整的 POMDP 追逃，只是“搜索 + 追踪”的最小域切换实验。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 8 方向 + 原地不动（顺序即动作编号，别随便改：模型权重与编号绑定）
_MOVES: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0),
)


class PursuitSearchEnv:
    """单环境实例（不含并行）。并行/多进程见 pursuit_parallel_env.py。"""

    def __init__(self, cfg, seed: Optional[int] = None, max_steps: Optional[int] = None):
        self.cfg = cfg
        self.H = self.W = int(cfg.grid_size)
        self.agent_num = int(cfg.agent_num)
        self.patch = int(cfg.COMMON_patch)          # 🔴 参数名前缀是历史遗留，见演练报告 §6
        self.obs_dim = int(cfg.obs_dim)
        self.n_actions = int(cfg.n_actions)
        self.max_steps = int(max_steps if max_steps is not None else cfg.init_max_train_steps)
        self.threshold = float(cfg.coverage_threshold)   # 仅用于「搜查进度」里程碑；**不**触发终止
        self.seed = int(cfg.seed if seed is None else seed)
        self.capture_radius = int(getattr(cfg, "PURSUIT_capture_radius", 0))
        self.evader_period = max(1, int(getattr(cfg, "PURSUIT_evader_period", 2)))

        # ── 地图要素：由 seed 决定，每轮 reset 不变（保证环境可复现、可对比）──
        rng = np.random.default_rng(self.seed)
        self.obstacles = self._sample_cells(rng, cfg.obstacle_num, avoid=())
        # 目标点 + 逃逸目标起点一起采（都避开障碍），最后一个给逃逸目标用
        self.targets = self._sample_cells(rng, cfg.target_num + 1,
                                          avoid=tuple(map(tuple, self.obstacles)))
        self.evader_start = [int(self.targets[-1][0]), int(self.targets[-1][1])]
        self.targets = self.targets[:-1]                                     # 静态目标点（可留空）
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
        self.evader = list(self.evader_start)
        self.steps = 0
        self.done = False
        self.success = False
        self.stagnant = [0] * self.agent_num
        self.hit_milestone = {m: False for m in self.cfg.milestones}
        self.close_bonus_paid = False
        for r, c in self.pos:                            # 起点视为已搜查
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

        evader_ref = list(self.evader)                  # 逃逸目标**移动前**的位置：接近奖励的参照
        dists_before = [self._dist(p, evader_ref) for p in self.pos]
        new_cells = 0

        # ── ① 智能体移动（撞障碍则原地不动）──
        for i, a in enumerate(actions):
            a = int(np.clip(a, 0, len(_MOVES) - 1))
            dr, dc = _MOVES[a]
            r = int(np.clip(self.pos[i][0] + dr, 0, self.H - 1))
            c = int(np.clip(self.pos[i][1] + dc, 0, self.W - 1))
            if self.obstacle_mask[r, c]:
                _acc(i, "obstacle_penalty", self.cfg.R_obstacle)
                continue
            self.pos[i] = [r, c]
            self.headings[i] = self._heading_of(dr, dc, self.headings[i])

        # ── ② 接近奖励（用移动前的目标位置做参照，防止「目标跑掉」被算成智能体的惩罚）──
        for i in range(self.agent_num):
            d_now = self._dist(self.pos[i], evader_ref)
            _acc(i, "approach", self.cfg.PURSUIT_R_approach * (dists_before[i] - d_now))

        # ── ③ 搜查覆盖（同一时刻记分量，铁律 2）──
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

        # ── ④ 首次贴近目标：一次性奖励（沿用 R_target_bonus，概览行的理论上限才不失真）──
        if not self.close_bonus_paid:
            i_best = int(np.argmin([self._dist(p, self.evader) for p in self.pos]))
            if self._dist(self.pos[i_best], self.evader) <= self.capture_radius + 1:
                self.close_bonus_paid = True
                _acc(i_best, "target_bonus", self.cfg.R_target_bonus)

        # ── ⑤ 逃逸目标移动（每 N 步一步：走到「离最近智能体最远」的相邻格）──
        if self.steps > 0 and self.steps % self.evader_period == 0:
            self._move_evader()

        # ── ⑥ 全局形状项（与覆盖任务同一套名字 ⇒ 日志契约不变）──
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
                    for i in range(self.agent_num):     # 里程碑是全局事件，均分
                        _acc(i, f"milestone_{int(m*100)}", bonus / self.agent_num)

        # ── ⑦ 终止：捕获优先，其次步数打满 ──
        captured = any(self._dist(p, self.evader) <= self.capture_radius for p in self.pos)
        if captured:
            for i in range(self.agent_num):
                _acc(i, "capture", self.cfg.PURSUIT_R_capture)

        self.steps += 1
        if captured:
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
            "evader": list(self.evader),
            "capture": bool(captured),
            "reward_components": comps,     # ★ 契约字段名，训练循环据此累加
        }
        return self._obs(), rewards, self.done, info

    # ══════════════════════ 逃逸目标 ══════════════════════
    def _move_evader(self) -> None:
        """走到「离最近智能体最远」的相邻格；同分按距离和、再按 _MOVES 顺序打破（确定性）。"""
        mask = self.obstacle_mask
        r0, c0 = self.evader
        best_key, best_cell = None, (r0, c0)
        for dr, dc in _MOVES:
            r = int(np.clip(r0 + dr, 0, self.H - 1))
            c = int(np.clip(c0 + dc, 0, self.W - 1))
            if mask[r, c]:
                continue
            dmin = min(self._dist([r, c], p) for p in self.pos)
            dsum = sum(self._dist([r, c], p) for p in self.pos)
            key = (-dmin, -dsum)            # 先最大化最近距离，再最大化总距离
            if best_key is None or key < best_key:
                best_key, best_cell = key, (r, c)
        self.evader = [int(best_cell[0]), int(best_cell[1])]

    @staticmethod
    def _dist(a, b) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

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
        out: List[np.ndarray] = []
        for i, (r, c) in enumerate(self.pos):
            vis_patch = self._patch_view(self.visited, r, c)
            obs_patch = self._patch_view(obs_mask, r, c)
            # 4 维 ego：自己位置(2) + 目标相对位置(2)。**维度与覆盖任务完全一致**，收益是网络零改动
            ego = np.array([r / max(1, self.H - 1), c / max(1, self.W - 1),
                            (self.evader[0] - r) / max(1, self.H - 1),
                            (self.evader[1] - c) / max(1, self.W - 1)], dtype=np.float32)
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
        """给日志契约用的环境快照（与覆盖任务同一位置、同一用途）。"""
        return [
            ("网格尺寸 GRID_SIZE", str(self.H)),
            ("智能体数量 agent_num", str(self.agent_num)),
            ("智能体起始位置", str([list(map(int, p)) for p in self.starts])),
            ("逃逸目标起点", str(list(self.evader_start))),
            ("静态目标点位置", str([list(map(int, p)) for p in self.targets])),
            ("固定障碍物坐标", str([list(map(int, p)) for p in self.obstacles])),
            ("初始朝向 (度)", str(self.initial_headings)),
            ("捕获判据（曼哈顿 ≤）", str(self.capture_radius)),
            ("逃逸目标步频（每 N 步一动）", str(self.evader_period)),
            ("单回合步数上限", str(self.max_steps)),
            ("搜查进度阈值（仅里程碑）", f"{self.threshold:.2f}"),
            ("随机种子", str(self.seed)),
        ]

    def render(self) -> str:
        """ASCII 地图（调试用）：. 未搜查 / * 已搜查 / # 障碍 / T 静态目标 / E 逃逸目标 / 1..n 智能体。"""
        ch = [["." if self.visited[r, c] == 0 else "*" for c in range(self.W)] for r in range(self.H)]
        for r, c in self.obstacles:
            ch[r][c] = "#"
        for r, c in self.targets:
            ch[r][c] = "T"
        ch[self.evader[0]][self.evader[1]] = "E"
        for i, (r, c) in enumerate(self.pos):
            ch[r][c] = str((i + 1) % 10)
        return "\n".join("".join(row) for row in ch)
