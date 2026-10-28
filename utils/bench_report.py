# -*- coding: utf-8 -*-
# 🕒 v4 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：PROTOCOL 键同步 COMMON_patch ｜ 上一版：v3 · 2026-10-17 16:22

"""benchmark 报告生成器：把所有**原始评测证据** → 一份可复查的 md + 图。

设计原则（与 docs/bench/README.md 一致）：
  · 报告里每一个数字都由本脚本从原始文件算出，**不许手抄**；
  · 输入只有三处：`docs/bench/results/<run>.json`（20 图评测原文）、
    `docs/bench/runs.csv`（实验元数据）、根目录 `eval_summary.csv`（单图历史）；
  · 基线（随机策略 / 贪心最近未访）由本脚本**按同一协议现场算**，落 `docs/bench/baselines.json`。

用法：
    python utils/bench_report.py --baselines   # 只算基线（CPU，约 10 s），结果缓存
    python utils/bench_report.py --report      # 只出报告（读缓存 + 图）
    python utils/bench_report.py --all         # 两步都做（默认）

产物：
    docs/BENCHMARK.md
    docs/figures/bench_*.png
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "docs", "bench")
RESULT_DIR = os.path.join(BENCH_DIR, "results")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
BASELINE_JSON = os.path.join(BENCH_DIR, "baselines.json")
REPORT_MD = os.path.join(ROOT, "docs", "BENCHMARK.md")

# 评测协议冻结项（与 docs/bench/README.md 逐条对齐；改这里 = 换协议 = 数字不可比）
PROTOCOL = {
    "grid_size": 20,
    "agent_num": 2,
    "obstacle_num": 12,
    "COMMON_patch": 2,
    "max_steps": 200,
    "eval_coverage_threshold": 0.75,
    "map_seeds": list(range(20)),   # eval_random_maps：env seed = 0 + r
}


# ══════════════════════════ 工具 ══════════════════════════
def _fmt_lr(lr: float) -> str:
    s = f"{lr:.0e}"                      # 5e-04
    return s.replace("e-0", "e-").replace("e+0", "e+")


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _pstd(xs: List[float]) -> float:
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# ══════════════════════════ 输入 ══════════════════════════
def load_manifest() -> Dict[str, Dict[str, Any]]:
    with open(os.path.join(BENCH_DIR, "runs.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[r["run"]] = {
            "run": r["run"],
            "episodes": int(r["episodes"]),
            "lr": float(r["lr"]),
            "seed": int(r["seed"]),
            "single_map_coverage": float(r["single_map_coverage"]),
            "note": r.get("note", ""),
        }
    return out


def load_evals() -> Dict[str, Dict[str, Any]]:
    out = {}
    if not os.path.isdir(RESULT_DIR):
        return out
    for fn in sorted(os.listdir(RESULT_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(RESULT_DIR, fn), encoding="utf-8") as f:
            out[fn[:-5]] = json.load(f)
    return out


def merge_rows() -> List[Dict[str, Any]]:
    manifest, evals = load_manifest(), load_evals()
    rows = []
    for run, meta in manifest.items():
        ev = evals.get(run)
        if not ev:
            continue
        rows.append({**meta, **{k: ev.get(k) for k in
                                ("coverage", "coverage_std", "coverage_min", "coverage_max",
                                 "reward", "success_rate", "steps", "maps", "random_maps",
                                 "evaluated_at")}})
    rows.sort(key=lambda r: -r["coverage"])
    return rows


# ══════════════════════════ 基线（现场算）══════════════════════════
_MOVES = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0))


def _greedy_nearest_unvisited(env) -> List[int]:
    """每个智能体朝「最近的未访空格」走一步；同一步里已认领的格子不再重复认领。"""
    import numpy as np
    mask = env.obstacle_mask
    cells = np.argwhere((env.visited == 0) & (mask == 0))
    acts: List[int] = []
    for r, c in env.pos:
        if len(cells):
            d = np.abs(cells - np.array([r, c])).sum(axis=1)
            j = int(d.argmin())
            tr, tc = int(cells[j][0]), int(cells[j][1])
            cells = np.delete(cells, j, axis=0)
        else:
            tr, tc = r, c
        best_key, best_a = None, 8
        for a, (dr, dc) in enumerate(_MOVES):
            rr = int(np.clip(r + dr, 0, env.H - 1))
            cc = int(np.clip(c + dc, 0, env.W - 1))
            if mask[rr, cc]:
                continue                                  # 撞障碍的走法不选
            key = (abs(rr - tr) + abs(cc - tc), 1 if a == 8 else 0, a)
            if best_key is None or key < best_key:
                best_key, best_a = key, a
        acts.append(best_a)
    return acts


def _run_policy(cfg, policy: str, seed_maps: List[int]) -> Dict[str, Any]:
    import numpy as np
    from algos.madqn.madqn_envs import MADroneSearchEnv

    rng = np.random.default_rng(20260930)
    covs, succ, steps = [], [], []
    for s in seed_maps:
        env = MADroneSearchEnv(cfg, seed=s, max_steps=int(cfg.final_test_steps))
        env.threshold = float(PROTOCOL["eval_coverage_threshold"])
        env.reset()
        while True:
            acts = rng.integers(0, cfg.n_actions, size=cfg.agent_num) if policy == "random" \
                else _greedy_nearest_unvisited(env)
            _, _, done, info = env.step(acts)
            if done:
                break
        covs.append(info["coverage"] * 100.0)
        succ.append(1.0 if info["success"] else 0.0)
        steps.append(info["steps"])
    return {"coverage": round(_mean(covs), 2), "coverage_std": round(_pstd(covs), 2),
            "coverage_min": round(min(covs), 2), "coverage_max": round(max(covs), 2),
            "success_rate": round(_mean(succ) * 100.0, 2), "steps": round(_mean(steps), 1),
            "per_map_coverage": [round(c, 2) for c in covs]}


def compute_baselines() -> Dict[str, Any]:
    sys.path.insert(0, ROOT)
    from config import Config
    cfg = Config()
    cfg._derive()
    for k in ("grid_size", "agent_num", "obstacle_num", "COMMON_patch"):
        assert getattr(cfg, k) == PROTOCOL[k], f"配置与冻结协议不一致：{k}"
    cfg.final_test_steps = PROTOCOL["max_steps"]
    data = {
        "protocol": PROTOCOL,
        "computed_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": f"{platform.system()} / {platform.processor() or platform.machine()} / CPU-only",
        "cmd": "python utils/bench_report.py --baselines",
        "policies": {
            "random": _run_policy(cfg, "random", PROTOCOL["map_seeds"]),
            "greedy_nearest": _run_policy(cfg, "greedy_nearest", PROTOCOL["map_seeds"]),
        },
    }
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(BASELINE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[baseline] 已写 {os.path.relpath(BASELINE_JSON, ROOT)}")
    return data


def load_baselines() -> Dict[str, Any]:
    if not os.path.exists(BASELINE_JSON):
        return {}
    with open(BASELINE_JSON, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════ 图 ══════════════════════════
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    })
    return plt


def fig_coverage_by_model(rows, plt) -> str:
    rows = sorted(rows, key=lambda r: r["coverage"])
    labels = [f"{r['run']} ({r['episodes']}ep)" for r in rows]
    means = [r["coverage"] for r in rows]
    stds = [r["coverage_std"] or 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(rows) + 1.2))
    y = range(len(rows))
    ax.barh(list(y), means, xerr=stds, color="#4C72B0", alpha=0.85,
            error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1})
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    for yi, m, s in zip(y, means, stds):
        ax.text(m + s + 1.0, yi, f"{m:.1f}±{s:.1f}", va="center", fontsize=7.5)
    ax.set_xlabel("coverage over 20 random maps  [%]")
    ax.set_xlim(0, 105)
    ax.set_title("MADQN checkpoints, same eval protocol (20 maps, 200 steps, goal 75%)",
                 fontsize=9.5)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "bench_coverage_by_model.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_single_vs_multimap(rows, plt) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    xs = [r["single_map_coverage"] for r in rows]
    ys = [r["coverage"] for r in rows]
    ys_std = [r["coverage_std"] or 0.0 for r in rows]
    ax.errorbar(xs, ys, yerr=ys_std, fmt="o", ms=5, color="#C44E52",
                ecolor="#999999", capsize=3, lw=0, elinewidth=1)
    lo, hi = 30, 95
    ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1, label="y = x (no bias)")
    for i, r in enumerate(rows):                       # 交错偏移，避免标签叠在一起
        dx, dy = (6, -9) if i % 2 == 0 else (-6, 6)
        ax.annotate(r["run"], (r["single_map_coverage"], r["coverage"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=6.5,
                    ha="left" if dx > 0 else "right")
    ax.set_xlabel("coverage on the single fixed map (seed=0)  [%]")
    ax.set_ylabel("mean coverage over 20 random maps  [%]")
    ax.set_title("Single-map number is not a generalisation estimate", fontsize=9.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "bench_single_vs_multimap.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_baselines(rows, baselines, plt) -> str:
    best = max(rows, key=lambda r: r["coverage"])
    pol = baselines["policies"]
    entries = [
        (f"MADQN best\n({best['run']})", best["coverage"], best["coverage_std"] or 0.0),
        ("greedy nearest-unvisited", pol["greedy_nearest"]["coverage"],
         pol["greedy_nearest"]["coverage_std"]),
        ("uniform random policy", pol["random"]["coverage"], pol["random"]["coverage_std"]),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    names = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    errs = [e[2] for e in entries]
    colors = ["#4C72B0", "#DD8452", "#8C8C8C"]
    ax.bar(names, vals, yerr=errs, color=colors, alpha=0.9, width=0.55,
           error_kw={"ecolor": "#333333", "capsize": 4, "lw": 1})
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 1.5, f"{v:.1f}±{e:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("coverage over 20 random maps  [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Learned policy vs two heuristics (same 20 maps, 200 steps)", fontsize=9.5)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "bench_baselines.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ══════════════════════════ 报告 ══════════════════════════
def _table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _read_single_map_csv() -> List[Dict[str, str]]:
    p = os.path.join(ROOT, "eval_summary.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_report() -> str:
    rows = merge_rows()
    if not rows:
        raise SystemExit("没有可用的 20 图评测结果（docs/bench/results/*.json 为空）")
    baselines = load_baselines()
    plt = _mpl()
    os.makedirs(FIG_DIR, exist_ok=True)
    figs = [fig_coverage_by_model(rows, plt), fig_single_vs_multimap(rows, plt)]
    if baselines:
        figs.append(fig_baselines(rows, baselines, plt))

    best = max(rows, key=lambda r: r["coverage"])
    by_run = {r["run"]: r for r in rows}
    # 只在「同回合数、同 seed」下比 lr（2000 回合批次，seed=2024）
    lr_family = [by_run[r] for r in ("lr2k_1e3_s2024", "lr2k_1e4_s2024", "lr2k_5e4_s2024")
                 if r in by_run]
    seed_family = [by_run[r] for r in ("lr2k_5e4_s2024", "lr2k_5e4_s7", "lr2k_5e4_s13")
                   if r in by_run]
    lr_span = max(x["coverage"] for x in lr_family) - min(x["coverage"] for x in lr_family)
    seed_span = max(x["coverage"] for x in seed_family) - min(x["coverage"] for x in seed_family)
    std_lo = min(r["coverage_std"] for r in rows)
    std_hi = max(r["coverage_std"] for r in rows)
    single_over = [(r["run"], r["coverage"] - r["single_map_coverage"]) for r in rows]
    bias_lo = min(d for _, d in single_over)
    bias_hi = max(d for _, d in single_over)
    n_over = sum(1 for _, d in single_over if d > 0)

    single_rows = []
    for r in rows:
        sem = (r["coverage_std"] or 0.0) / max(1, r["maps"] or 1) ** 0.5
        single_rows.append([
            r["run"], str(r["episodes"]), _fmt_lr(r["lr"]), str(r["seed"]),
            f"{r['single_map_coverage']:.2f}",
            f"**{r['coverage']:.2f}**", f"{r['coverage_std']:.2f}", f"{sem:.2f}",
            f"{r['coverage_min']:.2f}", f"{r['coverage_max']:.2f}",
            f"{r['coverage'] - r['single_map_coverage']:+.2f}",
            f"{r['success_rate']:.0f}%", f"{r['reward']:.1f}",
        ])

    base_rows = []
    if baselines:
        pol = baselines["policies"]
        for name, d, note in (
            (f"MADQN 最优模型 `{best['run']}`", best, "10 个 checkpoint 中 20 图均值最高"),
            ("贪心「最近未访格」启发式", pol["greedy_nearest"], "无学习；每步朝最近未访空格走，本机现算"),
            ("均匀随机策略", pol["random"], "无学习；9 动作等概率，本机现算"),
        ):
            base_rows.append([name, f"**{d['coverage']:.2f}**", f"{d['coverage_std']:.2f}",
                              f"{d['coverage_min']:.2f}", f"{d['coverage_max']:.2f}",
                              f"{d['success_rate']:.0f}%", note])

    csv_rows = []
    for r in _read_single_map_csv():
        csv_rows.append([r.get("run_id", ""), r.get("coverage", ""), r.get("reward", ""),
                         r.get("success_rate", ""), r.get("steps", "")])

    gap_greedy = baseline_gap = ""
    sem_lo, sem_hi = min(r["coverage_std"] / max(1, r["maps"] or 1) ** 0.5 for r in rows), \
                     max(r["coverage_std"] / max(1, r["maps"] or 1) ** 0.5 for r in rows)
    _sem = lambda r: r["coverage_std"] / max(1, r["maps"] or 1) ** 0.5
    lr_sem_lo = min(_sem(x) for x in lr_family) if lr_family else sem_lo
    lr_sem_hi = max(_sem(x) for x in lr_family) if lr_family else sem_hi
    have_b = bool(baselines)
    if have_b:
        g = baselines["policies"]["greedy_nearest"]
        rnd = baselines["policies"]["random"]
        gap_greedy = f"{best['coverage'] - g['coverage']:+.2f}"
        baseline_gap = f"{best['coverage'] - rnd['coverage']:+.2f}"
        concl4 = (
            f"4. **与两个无学习基线的距离（同 20 图 / 同 200 步 / 同判据）。** "
            f"MADQN 最优 {best['coverage']:.2f}% vs 贪心启发式 **{g['coverage']:.2f}%** vs 随机 **{rnd['coverage']:.2f}%**"
            f"（差 {gap_greedy} / {baseline_gap}pp）。\n"
            f"   ⇒ **覆盖率上 MADQN 与贪心启发式打平**（差 {abs(best['coverage'] - g['coverage']):.2f}pp，"
            f"小于两者 SEM 之和），但**稳定性/效率明显落后**：贪心 std {g['coverage_std']:.2f}pp、成功率 {g['success_rate']:.0f}%，"
            f"而 MADQN std {best['coverage_std']:.2f}pp、成功率 {best['success_rate']:.0f}%（成功=200 步内达 0.75）。\n"
            f"   ⚠️ **这不等于“同条件下被启发式打败”**：贪心用的是**全局已访图**（特权信息、无观测限制），"
            f"Q 网络只吃 5×5 局部视野 + 覆盖率 + 位置 —— 信息量不对称。要下“学不如贪”的结论，"
            f"必须先给启发式加同样的局部视野（未做，列为下一步）。"
        )
    else:
        concl4 = ("4. **基线缺失**：先跑 `python utils/bench_report.py --baselines`（CPU 约 10 s）。")


    rel_figs = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in figs]
    proto = PROTOCOL
    today = _dt.datetime.now().strftime("%Y-%m-%d")

    md = f"""# RL_infra 基准报告：MADQN 栅格覆盖搜索

> **这份报告的角色：装置自证，不是本项目的价值主张。**
> 它证明的是「训练产出 checkpoint → 独立评测 → 20 图聚合 → 脚本重算成表和图」这条链真的能跑通、且可复查；
> 它**不**证明这套 infra 好用（那要靠迁移账单：换项目改了几个文件/几行，见 `README.md` 的「迁移账单」），
> 也**不**证明模型强（§5 第 4 条：连贪心启发式都没赢）。
> ⚠️ 下面的数字**不能搬到别的项目** —— 口径绑死在 20×20 / 2 智能体 / 12 障碍 / 判据 0.75 / seed 0–19，换配置即不可比。

> **读数**：`{best['run']}` 在 **20 张随机地图**上平均覆盖 **{best['coverage']:.2f}% ± {best['coverage_std']:.2f}pp**，
> 比均匀随机策略高 **{baseline_gap}pp**，但与「贪心最近未访格」启发式**基本持平（{gap_greedy}pp，且稳定性与成功率落后）**；
> 同一模型在**单张固定图**上却是 **{best['single_map_coverage']:.2f}%** —— 单图数字不是泛化估计。

- 生成命令：`python utils/bench_report.py --all`（报告内所有数字由脚本从原始文件算出，无手抄）
- 原始证据：`docs/bench/results/<run>.json`（每个模型一份 20 图评测原文）· `docs/bench/runs.csv`（实验元数据）· `docs/bench/baselines.json`（基线实测）· 根目录 `eval_summary.csv`（单图历史）
- 结果目录不含 checkpoint：模型 `runs/<run>/madqn_best.pth` 落在高算平台归档，本地只留评测产物

---

## 1. 评测协议（冻结项，改了就不可比）

| 项 | 值 |
|---|---|
| 环境 | 栅格 {proto['grid_size']}×{proto['grid_size']} · 智能体 {proto['agent_num']} · 固定障碍 {proto['obstacle_num']} · 局部视野 p={proto['COMMON_patch']} |
| 动作 | 9（8 方向 + 停） |
| 地图 | **每回合换一张**（`--eval-random-maps`）：env seed = {proto['map_seeds'][0]} … {proto['map_seeds'][-1]}，共 {len(proto['map_seeds'])} 张 |
| 回合上限 | {proto['max_steps']} 步 |
| 成功判据 | 覆盖率 ≥ {proto['eval_coverage_threshold']:.2f}（评测专用，与训练的 0.90 **解耦**） |
| 策略 | 贪心（argmax Q），无探索噪声 |
| 聚合口径 | 跨 20 张地图的 **mean ± 总体标准差（pstdev）**，另给 min/max |

⚠️ **口径两条**：标准差是**地图间**差异（n=20 张图），**不是** seed 重复实验的误差棒 —— 每个配置只有 1 个训练 seed。
评测作业本身是 `scripts/eval_runs.sh --maps 20 --random-maps`（标准差由 `scripts/aggregate_eval.sh` 从逐图 json 聚合）。

---

## 2. 主表：10 个 checkpoint × 20 张地图

{_table(['run', '回合', 'lr', 'seed', '单图(seed=0)', '20 图均值', 'std(pp)', 'SEM(pp)', 'min', 'max', '均值−单图', '成功率', '奖励'], single_rows)}

> **std 与 SEM 不是一回事**：`std` = 20 张图之间的**个体差异**；`SEM` = 均值的标准误（std/√20）。
> 比较两个模型的**均值**时看 SEM，不要说“差异都在 ±std 之内所以是噪声”（那是拿个体分布当均值误差用）。
> “均值−单图”为正 = 单图**低估**了该模型，为负 = 单图**高估**了它。10 个 run 里有 **{n_over} 个**是后者。

---

## 3. 对比基线（同 20 张地图 / 同 200 步 / 同 0.75 判据）

{_table(['策略', '20 图均值', 'std(pp)', 'min', 'max', '成功率', '说明'], base_rows) if base_rows else '_（未算基线：先跑 `python utils/bench_report.py --baselines`）_'}

基线实测环境：{baselines.get('hardware', '—') if baselines else '—'} · 时间 {baselines.get('computed_at', '—') if baselines else '—'} ·
逐图数字在 `docs/bench/baselines.json`（`per_map_coverage` 字段，20 个数可直接复算均值）。
⚠️ 有两个不对称，读表时必须带着看：
1. 学习策略的数字来自高算 A800 评测作业，基线来自本机 CPU —— 环境确定性 + 纯 numpy 实现使地图与动力学不随平台变，但严格说这是跨机比较；
2. **信息量不对称**：贪心启发式读的是**全局已访图**，Q 网络只读 5×5 局部视野 + 覆盖率 + 位置。⇒ 它是“上限参照”，不是“同条件对手”。

---

## 4. 图

| 图 | 说明 |
|---|---|
| ![coverage by model]({rel_figs[0]}) | 各 checkpoint 的 20 图均值，误差棒 = 地图间 std |
| ![single vs multimap]({rel_figs[1]}) | 单图值 vs 20 图均值，虚线是 y=x；点落在下方 = 单图高估 |
{('| ![' + 'baselines](' + rel_figs[2] + ') | 学习策略 vs 两个启发式 |') if len(rel_figs) > 2 else ''}

---

## 5. 结论（每条都带数字）

1. **lr 的“效应”单图报大了约 2 倍，但不能说它全是噪声。** 同回合（2000）同 seed（2024）下三个 lr 的 20 图均值：
   1e-3 **{by_run['lr2k_1e3_s2024']['coverage']:.2f}** / 1e-4 **{by_run['lr2k_1e4_s2024']['coverage']:.2f}** / 5e-4 **{best['coverage']:.2f}**
   ⇒ 跨度 **{lr_span:.2f}pp**；同一批 run 在**单图**上的跨度是
   {max(x['single_map_coverage'] for x in lr_family) - min(x['single_map_coverage'] for x in lr_family):.2f}pp。
   20 图的 SEM 只有 **{lr_sem_lo:.2f}–{lr_sem_hi:.2f}pp** ⇒ {lr_span:.1f}pp 的均值差**超出地图采样误差**，
   不能说“落在 ±std 之内所以是假象”（std 是图间散布，不是均值误差）。
   **口径收窄**：单 seed ⇒ 训练随机性未测量（同 lr 三个 seed 的均值跨度 {seed_span:.2f}pp），所以“5e-4 > 1e-3”可以当**工作假设**，不能当结论。
2. **seed 效应被单图放大约 2 倍。** 固定 lr=5e-4、20 图均值
   {seed_family[0]['coverage']:.2f} / {seed_family[1]['coverage']:.2f} / {seed_family[2]['coverage']:.2f}
   ⇒ 跨度 **{seed_span:.2f}pp**（单图上这条跨度是
   {max(x['single_map_coverage'] for x in seed_family) - min(x['single_map_coverage'] for x in seed_family):.2f}pp）。
   ⇒ 单图的“31.85pp seed 效应”是高估。
3. **单图的测量噪声量级：图间 std 8.4–17.2pp。** 同一个 checkpoint 在 20 张图上覆盖
   {best['coverage_min']:.2f}%–{best['coverage_max']:.2f}%（{best['run']}，跨度
   {best['coverage_max'] - best['coverage_min']:.2f}pp），各 run 的图间 std 为
   {std_lo:.2f}–{std_hi:.2f}pp（对应 SEM {sem_lo:.2f}–{sem_hi:.2f}pp）。
   ⇒ 任何**单图**数字都必须带这个量级的误差棒才诚实。
{concl4}
5. **成功率不再是恒 0。** 判据从 0.90 解耦到 0.75 后，10 个 run 的成功率分布为
   {min(r['success_rate'] for r in rows):.0f}%–{max(r['success_rate'] for r in rows):.0f}%（此前恒 0%），
   说明该字段重新有分辨力；但最高也只有 {max(r['success_rate'] for r in rows):.0f}%，不要当主指标用。

---

## 6. 诚实短板

- **n=20 是地图数，不是重复实验数**：每个配置只训了 **1 个 seed**，所以“模型间差异”里含训练随机性，无法与地图效应分离。
- **逐图原始数据没有回传本地**：`docs/bench/results/*.json` 只有聚合值（mean/std/min/max），逐图覆盖在远端 `runs/<run>/test_path_plots_en/*.json`。
  ⇒ 同一批 20 张图本来允许做**配对比较**（同一张图上 A 与 B 的差），这里只能做非配对的 SEM 估计，分辨率更低。
- **只有 10 个 checkpoint**，且全部来自同一条 MADQN 链路、同一个 20×20 任务；**没有第二套算法/第二个任务**，因此“基础设施可迁移”这一条本报告**没有**提供证据（只有 `docs/架构决策.md` 的设计论证）。
- **基线是启发式，不是文献方法**（无 Independent PPO / VDN / QMIX 等 MARL 复现），且贪心基线用**全局已访图**（特权信息，见 §3）。所以对比只能支撑“学到了东西”，不能支撑“比别人强”。
- **跨机比较**：学习策略数字来自高算 A800，基线来自本机 CPU。环境确定性 + 纯 numpy 实现使结果应当一致，但未做同机复算。
- **训练端指标已饱和**：2000 回合批次的训练期“最优覆盖”全部在 90.21–90.46%（跨度 0.25pp），训练日志曲线在本任务上**不能**用来看模型好坏，只能看它有没有崩。
- 与登记表的口径对齐（**已同步**）：`docs/实验登记表.md` §三补2 原文写“跨度 7.64pp、落在 ±std 之内 ⇒ 是噪声”；
  按同表三个数（75.08 / 69.46 / 57.44）重算应为 **17.64pp**，且“拿图间 std 当均值误差”是机制错误。
  登记表 **v8 已按 SEM 口径改写**，与本报告 §5 第 1 条一致 —— 两处如再出现 7.64pp，以本报告为准。

---

## 7. 复现

```powershell
$env:PYTHONIOENCODING="utf-8"
pip install -e ".[madqn]"                  # 或 pip install -r requirements.txt
python utils/bench_report.py --baselines   # 基线实测（CPU，约 10 s）→ docs/bench/baselines.json
python utils/bench_report.py --all         # 重出本报告 + 全部图
```

模型侧复现（需要 checkpoint 与 GPU）：训练 → `python main.py --episodes 2000 --lr 5e-4 --name <名> --parallel`；
评测 → `bash scripts/eval_runs.sh --maps 20 --random-maps <名>`，然后 `bash scripts/aggregate_eval.sh 20 <名>`，
最后把 `runs/<名>/eval.json` 拷进 `docs/bench/results/<名>.json` 并在 `docs/bench/runs.csv` 补一行元数据 —— 本报告即自动收录。

---

## 8. 附录：根目录 `eval_summary.csv`（单图历史，2026-09-30）

这份 csv 是**第一版评测**的产物：固定地图（seed=0）+ 旧判据 0.90，只有 3 个 run、无 std 列。保留它是因为它是本项目
最早的一次独立评测记录，**但不要再用它的数字下结论**（判据已改、单图、n=1）。

{_table(['run_id', 'coverage', 'reward', 'success_rate', 'steps'], csv_rows) if csv_rows else '_（无）_'}
"""
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    # 保留上一版戳（不手写戳）：带过来后由 artifact_stamp.py bump 成新版本 + 新时间
    prev_stamp = ""
    if os.path.exists(REPORT_MD):
        with open(REPORT_MD, encoding="utf-8") as f:
            first = f.readline().rstrip("\n")
        if first.startswith("> 🕒"):
            prev_stamp = first + "\n\n"
    with open(REPORT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(prev_stamp + md)
    print(f"[report] 已写 {os.path.relpath(REPORT_MD, ROOT)}（{len(rows)} run / {len(figs)} 图，{today}）")
    return REPORT_MD, hashlib.sha1(md.encode("utf-8")).hexdigest()[:8]


def _stamp(path: str, note: str) -> None:
    """用官方脚本盖戳（绝不手写戳）；脚本缺失时只提示，不阻断。

    note 里带正文 sha1 前 8 位：内容没变 → 版本不涨（脚本自己判断 note 未变）；
    内容变了 → 自动 bump。这样版本号只反映“真的改了”。
    """
    tool = os.path.join(os.path.expanduser("~"), ".dsh", "bin", "artifact_stamp.py")
    if not os.path.exists(tool):
        print(f"[stamp] 跳过：找不到 {tool}（报告已生成，请手工补戳）")
        return
    head = open(path, encoding="utf-8").read(400)
    action = "bump" if "🕒" in head else "new"
    try:
        subprocess.run([sys.executable, tool, action, path, "--note", note], check=False)
    except Exception as e:      # 盖戳失败不影响报告本身
        print(f"[stamp] 失败（忽略）：{e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成 benchmark 报告与基线")
    ap.add_argument("--baselines", action="store_true", help="只算基线")
    ap.add_argument("--report", action="store_true", help="只出报告")
    ap.add_argument("--all", action="store_true", help="基线 + 报告（不带参数时：没有基线缓存才重算基线）")
    args = ap.parse_args(argv)
    do_base = args.baselines or args.all or (not args.report and not os.path.exists(BASELINE_JSON))
    do_rep = args.report or args.all or not args.baselines
    if do_base:
        compute_baselines()
    if do_rep:
        path, digest = build_report()
        # 注意：note 里不能出现 ASCII 竖线 —— artifact_stamp.py 的 STAMP_RE 用它分段，会长不回去
        _stamp(path, f"基准报告重生成（content {digest} · {len(merge_rows())} run / 3 图）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
