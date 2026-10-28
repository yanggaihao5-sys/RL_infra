# 🕒 v1 · 2026-10-24 12:41 · dsh-agent ｜ 变更：多 seed × 固定预算的 best-so-far 对照 + MDE

"""M2/M3 的**同预算对照基准**：同一合成目标上比较 rs / bo / warmstart 的 best-so-far

为什么要有这个脚本：`sweep.py` 只跑一次，一次结果说明不了"BO 有效"——
必须 **多个外层 seed × 固定预算**，报 **均值±SD**，才有资格说"哪种策略更快"。
这就是 M3 的"噪声基线"纪律的小规模实现（不训练，纯逻辑验证）。

用法：
    python tools/rlinfra_ext/bench_sweep.py --seeds 5 --budget 12
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep as S  # noqa: E402


def run_once(strategy, budget, seed, prior_mean, noise):
    rng = np.random.default_rng(seed)
    X, y, curve = [], [], []
    for _ in range(budget):
        u = S.suggest(strategy,
                      np.array(X) if X else np.zeros((0, len(S.SPACE))),
                      np.array(y) if y else np.zeros(0),
                      prior_mean, rng)
        x = S.decode(u)
        val = S.objective_dry(x, rng, noise=noise)
        X.append(list(u))
        y.append(val)
        curve.append(max(y))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.03, help="合成目标的观测噪声 SD（≈我们 seed 噪声的相对量级）")
    ap.add_argument("--prior-csv", default=os.path.join("db", "runs.csv"))
    ap.add_argument("--wide", action="store_true",
                    help="把搜索空间放大（lr 10 档 × batch 8 档 = 80 格点），用来测 BO 优势是否随空间变大才出现")
    args = ap.parse_args()

    if args.wide:
        S.SPACE["lr"]["choices"] = [round(1e-4 * (2 ** i / 1.0), 6) for i in range(10)]
        S.SPACE["batch"]["choices"] = [32, 48, 64, 96, 128, 192, 256, 512]
        print(f"[wide] 搜索空间放大到 {len(S.SPACE['lr']['choices'])} × {len(S.SPACE['batch']['choices'])} "
              f"= {len(S.SPACE['lr']['choices']) * len(S.SPACE['batch']['choices'])} 个格点")

    prior_mean, n_prior = S.load_prior(args.prior_csv)
    print(f"同预算对照：预算 B={args.budget} 次评估 · 外层 {args.seeds} 个 seed · 噪声 SD={args.noise} "
          f"· 先验均值={prior_mean:.4f}（{n_prior} 行）")
    print("（合成目标理论最优 ≈ 0.90，最优点故意落在网格点之间）\n")

    curves = {}
    for strat, pm in (("rs", None), ("bo", None), ("warmstart", prior_mean)):
        cs = [run_once(strat, args.budget, s, pm, args.noise) for s in range(args.seeds)]
        curves[strat] = np.array(cs)

    hdr = f"{'budget k':>8}" + "".join(f"{s:>22}" for s in curves)
    print(hdr)
    print("-" * len(hdr))
    for k in range(args.budget):
        row = f"{k+1:>8}"
        for s, arr in curves.items():
            v = arr[:, k]
            row += f"{v.mean():>14.4f} ±{v.std():>6.4f}"
        print(row)
    print("-" * len(hdr))

    # 判定：在相同预算下，谁更早到达"较好值"（这里用 0.87 作参考线）
    ref = 0.87
    print(f"\n参考线 best-so-far ≥ {ref} 的最早评估次数（均值±SD，越小越好）：")
    for s, arr in curves.items():
        hits = []
        for row in arr:
            idx = np.argmax(row >= ref) + 1 if (row >= ref).any() else args.budget + 1
            hits.append(idx)
        print(f"  {s:<11} {np.mean(hits):>5.2f} ± {np.std(hits):.2f}")

    mde = 2.0 * float(curves["rs"][:, -1].std()) / np.sqrt(args.seeds)
    print(f"\n可辨识效应量（MDE，近似 2·SD/√N）：{mde:.4f} —— 小于它的差异**不能当结论**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
