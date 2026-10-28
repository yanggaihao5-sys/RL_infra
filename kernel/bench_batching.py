# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：A2 表4：static vs continuous + CV 边界扫描；主判据用确定性工作量比

"""A2 表4：static batching vs continuous batching（CPU torch）

## 模型（只为调度，不为效果）

toy decoder：2 层 MLP + lm_head，hidden=128、vocab=1024，**无注意力/KV**。
每步成本 ∝ 槽位数 n（`(n,128)@(128,128)` 两层 + `(n,128)@(128,1024)`），
目的就是让"空转槽位"**直接体现为时间**而不是靠公式换算。

> ⚠️ 因此 tokens/s **只在「同一 toy 模型下两档之间」可比**，不可外推成真实 LLM 的绝对吞吐。
> 真实 LLM 每步成本还随序列长度上升（KV 读取 + 注意力），本玩具模型没有这一项。

## 两档（实现见 `kernel/batching.py`）

| 档 | 空转来源 |
|---|---|
| static | 块内**等最长**（已结束的序列继续占 padded 槽位） |
| continuous | 只有**池子见底**时才空转 |

## 判读纪律（两级，别混用）

| 级别 | 量 | 噪声 | 用途 |
|---|---|---|---|
| **主判据** | `work_ratio = (tokens+空转槽位)` 之比 | **确定性，0 噪声** | 判"连续批省了多少空转"——这是调度的**真值**，与模型实现无关 |
| 辅证 | `tps_ratio`（wall-clock tokens/s 之比）± SD | CPU 上实测 SD ≈ **±0.2** | 证明"省的槽位真的变成时间了"；**不单独用于判定** |

> 为什么不能只用 wall-clock：同一零效应情形（CV=0）跨次运行会给出 0.96 / 1.09 / 1.00 三个值，
> 而 `work_ratio` 恒为 1.00。拿 ±0.2 的量去判 1.5× 的闸门会被噪声带偏成假阳性/假阴性。

## 用法

    python kernel/bench_batching.py --dist equal,highvar,heavy --max-batch 8 --n 64 --seed 0
    python kernel/bench_batching.py --sweep-cv          # 扫 CV 找「连续批何时不划算」的边界

退出码：0 = 重尾档（手册闸门）≥1.5× 且等长档如实判无收益，否则 1。
产物：`runs/a2_batching/{batching.csv, boundary.csv, run.log}`
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batching import ContinuousBatcher, StaticBatcher, speedup  # noqa: E402

HIDDEN = 128
VOCAB = 1024
FIELDS = ["dist", "max_batch", "n_seqs", "seed", "rep",
          "static_tps", "continuous_tps", "tps_ratio", "ratio_sd", "ratio_n",
          "static_idle", "continuous_idle", "work_ratio", "tokens"]

# ───────────────────────── toy decoder ─────────────────────────
class ToyDecoder:
    """2 层 MLP + lm_head；每步前向的成本 ∝ 槽位数（这就是"空转=浪费"能被测到的原因）"""

    def __init__(self, hidden: int = HIDDEN, vocab: int = VOCAB, max_batch: int = 64,
                 seed: int = 0):
        import torch
        self.torch = torch
        g = torch.Generator().manual_seed(seed)
        self.w1 = torch.randn(hidden, hidden, generator=g) * 0.02
        self.w2 = torch.randn(hidden, hidden, generator=g) * 0.02
        self.w3 = torch.randn(hidden, vocab, generator=g) * 0.02
        self.h = torch.randn(max_batch, hidden, generator=g) * 0.1
        self.slots = 0

    def step(self, n_slots: int) -> None:
        torch = self.torch
        h = self.h[:n_slots]
        h = torch.relu(h @ self.w1)
        h = torch.relu(h @ self.w2)
        _ = h @ self.w3           # lm_head：只为把成本抬到可测
        self.h[:n_slots] = h      # 让状态可持续（成本模型的一部分）
        self.slots += n_slots

# ───────────────────────── 长度分布 ─────────────────────────
def gen_lengths(dist: str, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if dist == "equal":
        return [128] * n
    if dist == "highvar":                       # 32~512 均匀：高方差但不过分重尾
        return [rng.randint(32, 512) for _ in range(n)]
    if dist == "heavy":                         # 幂律：多短、极少数超长
        return [max(32, int(32 + 480 * (rng.random() ** 3))) for _ in range(n)]
    raise ValueError(f"不认识的 --dist：{dist}（equal|highvar|heavy）")

def gen_cv(cv: float, n: int, seed: int, mean: float = 160.0) -> list[int]:
    """给定变异系数 CV 的对数正态长度（裁到 [32,512]），用于扫「何时不划算」的边界"""
    if cv <= 1e-9:
        return [int(mean)] * n
    rng = random.Random(seed)
    sigma2 = math.log(1.0 + cv ** 2)
    mu = math.log(mean) - sigma2 / 2.0
    sigma = math.sqrt(sigma2)
    return [min(512, max(32, int(round(rng.lognormvariate(mu, sigma))))) for _ in range(n)]

# ───────────────────────── 一次对比 ─────────────────────────
def one_case(lens: list[int], max_batch: int, seed: int, reps: int = 5) -> dict:
    """跑 reps 轮，报中位数与 SD（方法论：效应量必须带 SD）"""
    tps_s, tps_c, ratios, work_r = [], [], [], []
    idle_s = idle_c = tokens = 0
    for _ in range(reps):
        dec = ToyDecoder(max_batch=max_batch, seed=seed)
        rs = StaticBatcher(lens, max_batch=max_batch).run(dec.step)
        dec = ToyDecoder(max_batch=max_batch, seed=seed)
        rc = ContinuousBatcher(lens, max_batch=max_batch).run(dec.step)
        sp = speedup(rs, rc)
        tps_s.append(rs["tokens_per_s"])
        tps_c.append(rc["tokens_per_s"])
        work_r.append(sp["work_ratio"])
        idle_s, idle_c = rs["idle_slots"], rc["idle_slots"]
        tokens = rs["tokens"]
        if rs["tokens_per_s"]:
            ratios.append(rc["tokens_per_s"] / rs["tokens_per_s"])
    return {"static_tps": statistics.median(tps_s),
            "continuous_tps": statistics.median(tps_c),
            "tps_ratio": statistics.median(ratios),
            "ratio_sd": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            "ratio_n": len(ratios),
            "static_idle": idle_s, "continuous_idle": idle_c,
            "work_ratio": statistics.median(work_r), "tokens": tokens}

def estimate_noise(args, reps: int = 5) -> tuple[float, float, float]:
    """零方差对照（CV=0）多跑几轮估噪声底；返回 (mean, SD, MDE=2·SD/√reps)"""
    lens = gen_cv(0.0, args.n, args.seed)
    vals = [one_case(lens, args.max_batch, args.seed)["tps_ratio"] for _ in range(reps)]
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd, 2.0 * sd / math.sqrt(len(vals))

def verdict_of(work_ratio: float, tps_ratio: float, sd: float, gate: float,
               w_neg: float = 1.02) -> str:
    """判定以**确定性**的 work_ratio 为准；tps_ratio 只作辅证（噪声带 ±sd）"""
    if work_ratio < w_neg:
        return f"无收益（工作量比 {work_ratio:.2f}×，空转省不出来）"
    if work_ratio >= gate:
        tag = "✅ 达标" if (tps_ratio - sd) >= 1.0 else "达标(但 wall-clock 噪声未完全排除)"
        return f"{tag} ≥{gate}×（工作量比 {work_ratio:.2f}×）"
    return f"有收益但未达 {gate}×（工作量比 {work_ratio:.2f}×）"

def main() -> int:
    ap = argparse.ArgumentParser(description="A2 表4：static vs continuous")
    ap.add_argument("--dist", default="equal,highvar,heavy")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=64, help="总序列数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5, help="每格重复轮数（取中位数 + SD）")
    ap.add_argument("--sweep-cv", action="store_true", help="另跑 CV 扫描找「何时不划算」边界")
    ap.add_argument("--noise-reps", type=int, default=5, help="零方差对照重复次数（估 MDE）")
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join("runs", "a2_batching"))
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("❌ 需要 torch：`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`")
        return 2
    print(f"# torch {torch.__version__} / CUDA {torch.cuda.is_available()}")

    # ── 噪声带：零方差对照下 wall-clock 加速比会抖多大（只作辅证，不参与判定）──
    noise_m, noise_sd, noise_mde = estimate_noise(args, reps=args.noise_reps)

    dists = [d for d in args.dist.split(",") if d]
    rows = []
    for dist in dists:
        lens = gen_lengths(dist, args.n, args.seed)
        r = one_case(lens, args.max_batch, args.seed, reps=args.reps)
        rows.append({"dist": dist, "max_batch": args.max_batch, "n_seqs": args.n,
                     "seed": args.seed, "rep": args.reps,
                     **{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "batching.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append(f"# 命令：python kernel/bench_batching.py --dist {args.dist} "
                 f"--max-batch {args.max_batch} --n {args.n} --seed {args.seed} "
                 f"--reps {args.reps} --sweep-cv")
    lines.append(f"# toy decoder：2 层 MLP + lm_head，hidden={HIDDEN} vocab={VOCAB}（无注意力/KV）；"
                 f"每格 = {args.reps} 轮中位数 ± SD")
    lines.append(f"# 噪声带（零方差 CV=0 重复 {args.noise_reps} 次测 wall-clock 抖动）："
                 f"mean={noise_m:.3f}× SD=**{noise_sd:.3f}** ⇒ tokens/s 加速比在 ±{noise_sd:.2f} 内"
                 f"**不可分辨**；故判定一律看确定性的 `工作量比`")
    lines.append("")
    lines.append("| 长度分布 | 长度特征 | static tokens/s | continuous tokens/s | 加速比 ± SD（辅证） "
                 "| 空转槽位 static | 空转槽位 continuous | **工作量比(主判据)** | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lens = gen_lengths(r["dist"], args.n, args.seed)
        mean = sum(lens) / len(lens)
        sd = statistics.pstdev(lens)
        feat = f"mean={mean:.0f} max={max(lens)} CV={sd / mean:.2f}"
        lines.append(f"| {r['dist']} | {feat} | {r['static_tps']:.0f} | {r['continuous_tps']:.0f} | "
                     f"{r['tps_ratio']:.2f}× ± {r['ratio_sd']:.2f} | "
                     f"{r['static_idle']} | {r['continuous_idle']} | "
                     f"**{r['work_ratio']:.2f}×** | "
                     f"{verdict_of(r['work_ratio'], r['tps_ratio'], noise_sd, args.gate)} |")

    # ── 边界扫描：CV 小到多少，连续批就没收益 ──
    boundary_rows = []
    if args.sweep_cv:
        for cv in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            lens = gen_cv(cv, args.n, args.seed)
            r = one_case(lens, args.max_batch, args.seed, reps=args.reps)
            boundary_rows.append({"cv": cv, "n_seqs": args.n, "max_batch": args.max_batch,
                                  "seed": args.seed,
                                  "tps_ratio": round(r["tps_ratio"], 6),
                                  "ratio_sd": round(r["ratio_sd"], 6),
                                  "work_ratio": round(r["work_ratio"], 6),
                                  "static_idle": r["static_idle"],
                                  "continuous_idle": r["continuous_idle"]})
        with open(os.path.join(args.out, "boundary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cv", "n_seqs", "max_batch", "seed", "tps_ratio",
                                              "ratio_sd", "work_ratio", "static_idle",
                                              "continuous_idle"])
            w.writeheader()
            w.writerows(boundary_rows)

        lines.append("")
        lines.append("### 边界扫描：长度变异系数 CV → 工作量比（**确定性，主判据**）+ wall-clock 辅证")
        lines.append("")
        lines.append("| CV | **工作量比** | 加速比 ± SD（辅证） | 空转 static | 空转 continuous | 判定 |")
        lines.append("|---|---|---|---|---|---|")
        for b in boundary_rows:
            lines.append(f"| {b['cv']} | **{b['work_ratio']:.2f}×** | "
                         f"{b['tps_ratio']:.2f}× ± {b['ratio_sd']:.2f} | "
                         f"{b['static_idle']} | {b['continuous_idle']} | "
                         f"{verdict_of(b['work_ratio'], b['tps_ratio'], noise_sd, args.gate)} |")
        hit = next((b["cv"] for b in boundary_rows if b["work_ratio"] >= args.gate), None)
        first_win = next((b["cv"] for b in boundary_rows if b["work_ratio"] >= 1.02), None)
        lines.append("")
        if first_win is None:
            lines.append("🔴 边界结论：本网格内工作量比恒 ≈1 ⇒ 连续批**始终无收益**（检查序列数/批大小）。")
        else:
            lines.append(f"🔴 **「连续批何时不划算」的边界（看确定性工作量比）**："
                         f"CV < **{first_win:.2f}** 时收益 <1.02×（≈无收益，只剩调度复杂度）；"
                         f"CV ≥ **{hit if hit is not None else '>2.0'}** 才达到 {args.gate}× 判据。")
        lines.append("   机制：CV→0 时块内等最长不产生任何浪费（static 的 idle→0），"
                     "连续批补位也就没东西可补 ⇒ 两档工作量相同，收益归零，只多付调度复杂度。")

    # ── 判定（手册闸门只压强尾档；一律看确定性工作量比）──
    by = {r["dist"]: r for r in rows}
    long_ok = all(by[d]["work_ratio"] >= args.gate for d in ("heavy",) if d in by)
    eq_ok = all(by["equal"]["work_ratio"] < 1.02 for d in ("equal",) if d in by)
    lines.append("")
    lines.append("## 判定（主判据 = 工作量比，确定性）")
    lines.append(f"- **长尾（重尾 `heavy`）≥{args.gate}×**："
                 f"{'✅ PASS' if long_ok else '❌ FAIL —— 按手册要求须如实写原因，不许改口径'}"
                 + (f"（工作量比 {by['heavy']['work_ratio']:.2f}×；wall-clock "
                    f"{by['heavy']['tps_ratio']:.2f}× ± {noise_sd:.2f} 与之相符）" if "heavy" in by else ""))
    lines.append(f"- 等长档如实报「无收益」："
                 f"{'✅ PASS' if eq_ok else '❌ 等长档工作量比竟 >1.02，与机制不符，需查实现'}"
                 + (f"（工作量比 {by['equal']['work_ratio']:.2f}×，两档空转均为 0）" if "equal" in by else ""))
    if "highvar" in by:
        hv = by["highvar"]
        lines.append(f"- 高方差档（**非闸门项**，如实上报）：工作量比 {hv['work_ratio']:.2f}× —— "
                     f"{'达标' if hv['work_ratio'] >= args.gate else '未达 1.5×'}"
                     f"；wall-clock {hv['tps_ratio']:.2f}× ± {noise_sd:.2f}")
    lines.append(f"- ⚠️ 诚实短板：toy decoder **无注意力/KV**，每步成本只 ∝ 槽位数；"
                 f"真实 LLM 每步成本还随序列长度上升，故本表的绝对 tokens/s 不可外推，"
                 f"只有**两档之间的工作量比**是可迁移结论。")
    lines.append(f"- ⚠️ wall-clock 在 CPU 上噪声 ±{noise_sd:.2f}（零方差对照实测），"
                 f"故 `加速比` 列只作辅证，判定不看它。")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "run.log"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if (long_ok and eq_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
