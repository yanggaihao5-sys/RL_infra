# 🕒 v1 · 2026-10-24 12:41 · dsh-agent ｜ 变更：同预算搜参：rs/bo/warmstart，numpy GP-EI

"""同预算对照搜参器：`rs` / `bo` / `warmstart-bo` 三种策略，**零第三方依赖**（numpy 手写 GP-EI）

为什么不用 Optuna/sklearn：本机 system python 没有 sklearn（实测 ModuleNotFoundError），
且要能塞进 RL_infra（基座零第三方依赖的规矩）。

三种策略的区别**只在"下一步评估哪里"**：
  rs         —— 均匀随机采样（基线）
  bo         —— GP + Expected Improvement（冷启动，均值 = 观测均值）
  warmstart  —— 同上，但**均值函数来自先验观测**（如别的场景/别的仓的 run）⇒ 这就是"迁移先验"

同预算对照的纪律（M2/M3）：
  · 预算 B 固定（不是墙钟，是**评估次数**）
  · 报 **best-so-far @ 每次评估**（不是只报最终值）
  · 报告同配置重复的 SD ⇒ 才能算出"可辨识效应量"

用法（无训练也能验证逻辑）：
    python tools/rlinfra_ext/sweep.py --strategy bo --budget 12 --dry-run
    python tools/rlinfra_ext/sweep.py --strategy warmstart --budget 12 --prior-csv db/runs.csv --dry-run
真实运行（会起子进程训练；🔴 提交/占资源前必须先问用户）：
    python tools/rlinfra_ext/sweep.py --strategy bo --budget 8 --rl-infra __REPO_DIR__
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys

import numpy as np

# 搜索空间（键名 = 传给 RL_infra main.py 的 CLI 参数，见其 main.py:46-47）
SPACE = {
    "lr": {"flag": "--lr", "choices": [1e-4, 3e-4, 5e-4, 1e-3]},
    "batch": {"flag": "--batch-size", "choices": [64, 128, 256]},
}


def encode(x):
    """离散取值 → [0,1] 坐标（GP 在连续空间里工作）"""
    vec = []
    for k, spec in SPACE.items():
        c = spec["choices"]
        i = c.index(x[k]) if x[k] in c else 0
        vec.append(i / max(1, len(c) - 1))
    return np.array(vec, dtype=float)


def decode(u):
    out = {}
    for i, (k, spec) in enumerate(SPACE.items()):
        c = spec["choices"]
        idx = int(round(float(np.clip(u[i], 0, 1)) * max(1, len(c) - 1)))
        out[k] = c[idx]
    return out


# ───────────────────────────── GP + EI（numpy） ─────────────────────────────
def _kernel(A, B, ell=0.35):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / ell ** 2)


def gp_predict(X, y, xq, prior_mean, ell=0.35, jitter=1e-6):
    """返回 (mu, sigma)；先验均值 prior_mean 可以来自别的任务/别的仓（= warm-start）"""
    K = _kernel(X, X, ell) + jitter * np.eye(len(X))
    ks = _kernel(X, xq[None, :], ell).ravel()
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - prior_mean))
        v = np.linalg.solve(L, ks)
        mu = prior_mean + float(ks @ alpha)
        var = float(1.0 - v @ v)
    except np.linalg.LinAlgError:
        mu, var = float(np.mean(y)), 1.0
    return mu, math.sqrt(max(var, 1e-12))


def expected_improvement(mu, sigma, best, xi=0.01):
    """最小化问题：EI = (best - mu - xi)Φ(z) + σφ(z)"""
    z = (best - mu - xi) / max(sigma, 1e-12)
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return (best - mu - xi) * Phi + sigma * phi


# ───────────────────────────── 目标函数 ─────────────────────────────
def objective_dry(x, rng, noise=0.03):
    """合成目标：最优点在 lr=5e-4 / batch=128 附近（便于验证 BO 是否真在搜寻）

    真值函数故意做成"有一个清晰最优 + 观测噪声"，用来检验算法而不是运气。
    """
    lr_log = math.log10(x["lr"])
    a = (lr_log + 3.45) / 0.5               # 最优点 lr≈3.5e-4（**故意落在网格点之间**）
    b = (math.log2(x["batch"]) - 6.58) / 1.0  # 最优点 batch≈96
    true = 0.90 - 0.08 * (a ** 2) - 0.05 * (b ** 2)
    return true + rng.normal(0, noise)


def objective_real(x, cfg, rl_infra):
    """真跑：调 RL_infra main.py，然后从 runs/<name>/train.log 抽 best cov"""
    name = cfg["tag"]
    cmd = [sys.executable, "main.py", "--name", name, "--episodes", str(cfg["episodes"]),
           "--lr", str(x["lr"]), "--batch-size", str(x["batch"]),
           "--seed", str(cfg["seed"])]
    subprocess.run(cmd, cwd=rl_infra, check=True)
    log = os.path.join(rl_infra, "runs", name, "train.log")
    best = 0.0
    for line in open(log, encoding="utf-8", errors="replace"):
        if line.startswith("EP |"):
            for part in line.split("|"):
                if "cov:" in part:
                    best = max(best, float(part.split("cov:")[1].strip().rstrip("%")) / 100.0)
    return best


# ───────────────────────────── 主循环 ─────────────────────────────
def load_prior(csv_path):
    """从 db/runs.csv 抽先验：取有 cov_mean 的行，把 (lr,batch) 已知的当成观测

    现在表里没有 lr/batch 列 ⇒ 退化成一个"先验均值"（各线 cov_mean 的中位数）。
    """
    import csv as _csv
    vals = []
    if csv_path and os.path.exists(csv_path):
        for r in _csv.DictReader(open(csv_path, encoding="utf-8")):
            try:
                if r.get("cov_mean"):
                    vals.append(float(r["cov_mean"]))
            except ValueError:
                pass
    return (float(np.median(vals)) if vals else 0.80), len(vals)


def suggest(strategy, X, y, prior_mean, rng):
    if strategy == "rs" or len(X) < 3:
        return rng.uniform(0, 1, len(SPACE))
    if prior_mean is None:          # 冷启动：均值退化为"已观测的均值"（不引入外部先验）
        prior_mean = float(np.mean(y))
    best = float(np.min(y))
    cand = rng.uniform(0, 1, (512, len(SPACE)))
    ei = []
    for u in cand:
        mu, sd = gp_predict(X, y, u, prior_mean)
        ei.append(expected_improvement(mu, sd, best))
    return cand[int(np.argmax(ei))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["rs", "bo", "warmstart"], default="bo")
    ap.add_argument("--budget", type=int, default=8, help="评估次数预算（固定，不是墙钟）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="用合成目标验证逻辑（不训练）")
    ap.add_argument("--prior-csv", default=os.path.join("db", "runs.csv"))
    ap.add_argument("--rl-infra", default="", help="真跑时 RL_infra 仓路径")
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prng = random.Random(args.seed)
    prior_mean, n_prior = load_prior(args.prior_csv)
    if args.strategy != "warmstart":
        prior_mean = None            # 冷启动：不用外部先验

    pm_txt = ("无（冷启动，均值=观测均值）" if prior_mean is None
              else f"{prior_mean:.4f}（来自 {n_prior} 行 cov_mean）")
    print(f"策略={args.strategy} 预算={args.budget} 模式={'dry-run' if args.dry_run else '真跑'} 先验均值={pm_txt}")

    X, y, rows = [], [], []
    best_so_far = []
    for t in range(args.budget):
        u = suggest(args.strategy, np.array(X) if X else np.zeros((0, len(SPACE))),
                    np.array(y) if y else np.zeros(0), prior_mean, rng)
        x = decode(u)
        val = objective_dry(x, rng) if args.dry_run else objective_real(
            x, {"tag": f"sweep_{args.seed}_{t}", "episodes": args.episodes, "seed": args.seed},
            args.rl_infra)
        X.append(list(u))
        y.append(val)
        best_so_far.append(max(y))
        rows.append({"t": t + 1, **x, "value": round(val, 4), "best": round(max(y), 4)})
        print(f"  #{t+1:>2} lr={x['lr']:<7} batch={x['batch']:<4} value={val:.4f} best={max(y):.4f}")

    print("\nbest-so-far 曲线：" + " → ".join(f"{b:.3f}" for b in best_so_far))
    if args.dry_run:
        # 合成目标的最优值约 0.90（找出它 = BO 在起作用）
        print(f"合成目标理论最优 ≈ 0.90；本次最优 {max(y):.4f}")
    print(json.dumps({"strategy": args.strategy, "budget": args.budget,
                      "best_so_far": [round(b, 4) for b in best_so_far],
                      "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
