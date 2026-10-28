# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-17 16:43 · dsh-agent ｜ 变更：新增：追逃域可解性探针（含难度旋钮 ---evader-period）

"""追逃域的可解性探针：用**无学习的「直线追击」启发式**跑 20 张图，看这任务到底能不能完成。

为什么需要它（迁移演练的一部分）：
  换域之后，最怕的不是「学习学得慢」，而是「这个域根本没法完成」——
  那样训练日志里的 0% 捕获率就无法归因。这个探针把「域是否可解」和「策略是否学到」分开测。

判据（跑之前写死）：
  · 若直线追击在 20 张图上捕获率 ≥ 50% ⇒ 域可解，训练没学会是**算法/奖励**问题，不是环境问题；
  · 若 < 50% ⇒ 先怀疑环境（逃逸策略过强 / 捕获半径 0 太苛刻 / 障碍堵死），再谈训练。

用法：
    python scripts/probe_pursuit.py --maps 20
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config                       # noqa: E402
from algos.pursuit.pursuit_envs import PursuitSearchEnv, _MOVES   # noqa: E402


def _chase_actions(env) -> List[int]:
    """每个智能体：朝目标走一步（曼哈顿距离最小），撞障碍的走法不选。"""
    mask = env.obstacle_mask
    acts: List[int] = []
    for r, c in env.pos:
        best_key, best_a = None, 8
        for a, (dr, dc) in enumerate(_MOVES):
            rr = int(np.clip(r + dr, 0, env.H - 1))
            cc = int(np.clip(c + dc, 0, env.W - 1))
            if mask[rr, cc]:
                continue
            key = (abs(rr - env.evader[0]) + abs(cc - env.evader[1]), 1 if a == 8 else 0, a)
            if best_key is None or key < best_key:
                best_key, best_a = key, a
        acts.append(best_a)
    return acts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="追逃域可解性探针（无学习）")
    ap.add_argument("--maps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="起始地图 seed（与评测协议一致：0…maps-1）")
    ap.add_argument("--evader-period", type=int, default=None,
                    help="覆盖 config 的逃逸步频（1=每步都动，最难追）；不给则用 config 值")
    args = ap.parse_args(argv)

    cfg = Config(test_mode="pursuit")
    cfg._derive()
    if args.evader_period is not None:
        cfg.PURSUIT_evader_period = int(args.evader_period)
    cfg.validate()

    caps, steps, covs = [], [], []
    for s in range(args.seed, args.seed + args.maps):
        env = PursuitSearchEnv(cfg, seed=s, max_steps=cfg.final_test_steps)
        env.reset()
        while True:
            _, _, done, info = env.step(_chase_actions(env))
            if done:
                break
        caps.append(1.0 if info["success"] else 0.0)
        steps.append(info["steps"])
        covs.append(info["coverage"] * 100.0)

    print(f"图数 {args.maps} · 捕获 {int(sum(caps))}/{args.maps} = {np.mean(caps)*100:.0f}% · "
          f"平均步数 {np.mean(steps):.1f} · 平均搜查 {np.mean(covs):.2f}%")
    for i, (c, st, cv) in enumerate(zip(caps, steps, covs)):
        print(f"  seed {args.seed + i:>2}: {'捕获' if c else '未捕获'} | 步数 {int(st):>3} | 搜查 {cv:.1f}%")
    ok = np.mean(caps) >= 0.5
    print(f"判定：{'✅ 域可解（直线追击能完成）' if ok else '❌ 域可疑（20 图捕获率 < 50%，先查环境）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
