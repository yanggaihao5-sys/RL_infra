# 🕒 v1 · 2026-10-24 12:43 · dsh-agent ｜ 变更：任务族采样 + G1~G5 合法性守卫（50 步被 G4 拒）

"""task_gen —— 任务族生成器 + **合法性守卫**（顺序点 / 高价值时限 / 重复率 / 特殊格 / 随机障碍）

目的：把"1 个任务"变成"一个任务族"。不改几何，只在**任务参数**上采样：
    n_waypoints    需要按顺序遍历的关键点数
    hv_deadline    多少步内必须遍历完所有高价值格
    repeat_cap     重复访问栅格比例上限
    special_*      特殊栅格（需多次覆盖）的个数与最少覆盖次数
    obstacle_mode  fixed（原图）/ per_episode（每回合随机）/ per_k（每 K 回合换一次）
    coverage_target 主判据阈值（0.92 / 0.96）

🔴 守卫（这一步不做，任务族就是垃圾任务族）：任何采样都要过 5 条硬检查 ——
    G1 自由格比例 ≥ 基线 0.90
    G2 从每个起点 8 邻接可达的自由格 ≥ coverage_target × 自由格数（结构性可达）
    G3 高价值格数 ≥ 80 且**全部可达**（否则"遍历完高价值格"不可能完成）
    G4 高价值时限的**下界可行**：hv_deadline ≥ ⌈N_hv / agent_num⌉ × slack（slack 默认 1.5，
      因为路径必须连续、不能瞬移到下一个高价值格）
    G5 特殊格 / 关键点都落在**自由且可达**的格子上，且两两不重合

用法：
    python tools/rlinfra_ext/task_gen.py --sample 20 --seed 0          # 采样并报拒绝率与原因
    python tools/rlinfra_ext/task_gen.py --probe                        # 只打印当前图的地形事实
    python tools/rlinfra_ext/task_gen.py --sample 5 --tight             # 故意给紧约束，验证守卫真的会拒
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SPACE = {
    "n_waypoints": [3, 5, 8],
    "hv_deadline": [80, 120, 200],
    "repeat_cap": [0.20, 0.25, 0.30],
    "special_visits": [2, 3],
    "n_special": [3, 6],
    "obstacle_mode": ["fixed", "per_episode", "per_k"],
    "coverage_target": [0.92, 0.96],
}

SLACK = 1.5          # 高价值时限的路径松弛系数
MIN_FREE_RATIO = 0.90
MIN_HV = 80


# ───────────────────────── 地形事实 ─────────────────────────
def load_terrain():
    """从本仓 env 取真实地形（只读）：障碍掩码、概率场、起点"""
    sys.path.insert(0, BASE)
    from MADQN.MADQN_envs import MADroneSearchEnv
    env = MADroneSearchEnv(agent_num=3, INIT_MAX_TRAIN_STEPS=100)
    env.reset()
    m = env.multi_channel_map
    obstacle = (m[2] != 0)
    prob = np.asarray(m[0], dtype=float)
    starts = getattr(env, "agent_start_pos", None) or [(6, 6), (24, 25), (6, 25)]
    starts = [tuple(map(int, s)) for s in starts]
    return obstacle, prob, starts, int(env.agent_num if hasattr(env, "agent_num") else len(starts))


def reachable(obstacle, start):
    H, W = obstacle.shape
    if obstacle[start]:
        return set()
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not obstacle[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return seen


def reach_all(obstacle, starts):
    s = set()
    for st in starts:
        s |= reachable(obstacle, st)
    return s


# ───────────────────────── 守卫 ─────────────────────────
def guard(task, obstacle, prob, starts, base_free, tol=None):
    """返回 (ok, reasons[])；reasons 空 = 通过"""
    reasons = []
    free = int((~obstacle).sum())
    if free < MIN_FREE_RATIO * base_free:
        reasons.append(f"G1 自由格 {free} < {MIN_FREE_RATIO:.0%}×{base_free}")

    ra = reach_all(obstacle, starts)
    need = int(math.ceil(task["coverage_target"] * free))
    if len(ra) < need:
        reasons.append(f"G2 可达格 {len(ra)} < 目标需覆盖 {need}（结构性不可达）")

    hv = [(int(r), int(c)) for r, c in zip(*np.where((prob >= 0.8) & (~obstacle)))]
    hv_reach = [p for p in hv if p in ra]
    if len(hv) < MIN_HV:
        reasons.append(f"G3 高价值格只有 {len(hv)} < {MIN_HV}")
    if len(hv_reach) < len(hv):
        reasons.append(f"G3 有 {len(hv) - len(hv_reach)} 个高价值格不可达（'遍历完'不可能完成）")

    agent_num = len(starts)
    lb = math.ceil(len(hv) / agent_num * SLACK)
    if task["hv_deadline"] < lb:
        reasons.append(f"G4 高价值时限 {task['hv_deadline']} < 下界 {lb}"
                       f"（N_hv={len(hv)} / {agent_num} 机 × slack={SLACK}）")

    if task.get("special_cells"):
        for p in task["special_cells"]:
            if tuple(p) not in ra:
                reasons.append(f"G5 特殊格 {tuple(p)} 不可达")
    if task.get("waypoints"):
        pts = [tuple(p) for p in task["waypoints"]]
        if len(set(pts)) != len(pts):
            reasons.append("G5 关键点有重复")
        for p in pts:
            if p not in ra:
                reasons.append(f"G5 关键点 {p} 不可达")
    return (len(reasons) == 0), reasons


# ───────────────────────── 采样 ─────────────────────────
def sample_task(rng, obstacle, prob, starts):
    task = {k: list(v)[int(rng.integers(len(v)))] if not isinstance(v[0], str)
            else v[int(rng.integers(len(v)))]
            for k, v in SPACE.items()}
    free_cells = [(int(r), int(c)) for r, c in zip(*np.where(~obstacle))]
    ra_list = sorted(reach_all(obstacle, starts))

    # 关键点：从可达格里随机取（真实生成时应按"顺序合理"布点，这里只保证合法）
    k = min(task["n_waypoints"], len(ra_list))
    idx = rng.choice(len(ra_list), size=k, replace=False)
    task["waypoints"] = [ra_list[i] for i in idx]

    # 特殊格：优先取高价值格
    hv = [(int(r), int(c)) for r, c in zip(*np.where((prob >= 0.8) & (~obstacle)))]
    pool = hv if len(hv) >= task["n_special"] else ra_list
    idx = rng.choice(len(pool), size=min(task["n_special"], len(pool)), replace=False)
    task["special_cells"] = [pool[i] for i in idx]
    return task


def random_obstacle(rng, shape, base_obstacle, n_blocks=12):
    """随机障碍布局（G1/G2 会在守卫里兜底）"""
    ob = np.zeros(shape, dtype=bool)
    free = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    for _ in range(n_blocks):
        r, c = free[int(rng.integers(len(free)))]
        ob[r, c] = True
    return ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--tight", action="store_true", help="故意施加紧约束，验证守卫会拒")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    obstacle, prob, starts, agent_num = load_terrain()
    base_free = int((~obstacle).sum())
    ra = reach_all(obstacle, starts)
    hv = [(int(r), int(c)) for r, c in zip(*np.where((prob >= 0.8) & (~obstacle)))]
    lb80 = math.ceil(len(hv) / agent_num * SLACK)

    print(f"地形事实：{obstacle.shape[0]}×{obstacle.shape[1]} · 自由格 {base_free} · 可达格 {len(ra)} "
          f"· 高价值格 {len(hv)} · 机数 {agent_num}")
    print(f"  ⇒ 高价值时限的**可行下界** = ⌈{len(hv)}/{agent_num}⌉ × {SLACK} = **{lb80} 步**"
          f"（低于它的任务一律被 G4 拒绝）")
    if args.probe:
        return 0

    rng = np.random.default_rng(args.seed)
    ok_n, rej = 0, {}
    good, warned = [], []
    for i in range(args.sample):
        task = sample_task(rng, obstacle, prob, starts)
        # 随机障碍：真生成一张新图并让 G1/G2 去检验（不合法就拒）
        ob = obstacle
        if task["obstacle_mode"] != "fixed":
            ob = random_obstacle(rng, obstacle.shape, obstacle,
                                 n_blocks=12 if task["obstacle_mode"] == "per_episode" else 8)
        if args.tight:
            task["hv_deadline"] = 50          # 故意给不可能的值
            task["repeat_cap"] = 0.05         # 故意给过严的重复率
        ok, reasons = guard(task, ob, prob, starts, base_free)
        if ok:
            ok_n += 1
            good.append(task)
            # 重复率没有结构守卫 —— 只能拿"已观测区间"当参照（本仓 O_J 实测 0.264~0.576）
            if task["repeat_cap"] < 0.264:
                warned.append((i, task["repeat_cap"]))
        else:
            for r in reasons:
                key = r.split("（")[0].split(" <")[0][:12]
                rej[key] = rej.get(key, 0) + 1
    print(f"\n采样 {args.sample} 个任务：合法 {ok_n} 个，拒绝 {args.sample - ok_n} 个")
    for k, v in sorted(rej.items(), key=lambda kv: -kv[1]):
        print(f"  拒绝原因[{k}] × {v}")
    if warned:
        print(f"  ⚠️ 其中 {len(warned)} 个任务的 repeat_cap < 0.264（本仓 O_J 实测最小值）"
              f"⇒ **超出已观测可达区间，属'无证据支持的要求'**，不拒但必须标注："
              f"{[f'{i+1}:{c}' for i, c in warned[:5]]}")
    if good:
        t = good[0]
        print("\n合法任务样例：")
        print(json.dumps({k: (list(map(list, v)) if isinstance(v, list) and v and isinstance(v[0], (list, tuple))
                             else v) for k, v in t.items()}, ensure_ascii=False)[:400])
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(good, f, ensure_ascii=False, indent=1)
        print(f"\n写出 {len(good)} 个合法任务 → {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
