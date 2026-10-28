# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：A1 表1：内部碎片率 4档×3分布 + 闭式公式修正与适用域

"""A1 表1：内部碎片率 vs block_size，并检验闭式公式

用法（BENCH.md 表1 的命令就长这样）：
    python kernel/bench_internal_frag.py --block-sizes 8,16,32,64 --lens all --n 3000 --seed 0

三种长度分布（`--lens`）：
    all                   = uniform + lognormal + mixed 三种都跑（默认）
    uniform:20,300        = U[20,300] 均匀     （手册 A1 的参照分布）
    lognormal:160,0.8     = 对数正态，目标均值 160、sigma 0.8
    mixed:20,300          = 70% 均匀 + 30% 对数正态（真实 serving 的混合态）

两条闭式式（**分母不同，这是关键**）：
    原式  frag ≈ (b−1) / (2·mean_len)                ← 手册现行；分母用逻辑长度
    修正式 frag ≈ w / (mean_len + w),  w = (b−1)/2   ← 分母用**分配容量**，这才是碎片率的分母

推导（修正式）：
    · 序列长度 n 对 b 取模近似均匀 ⇒ 最后一块平均浪费 E[waste] = (b−1)/2 = w
    · 每条序列平均占用容量 E[cap] = E[⌈n/b⌉·b] ≈ mean_len + w
    · 内部碎片率 = Σ浪费 / Σ容量 = E[waste]/E[cap] = w / (mean_len + w)
    · 原式漏掉了分母里的 w，等价于假设 mean_len ≫ b —— 故它只在 mean_len/b 较大时成立

退出码：修正式在**参照分布**（uniform:20,300）上四档全部 <2pt ⇒ 0，否则 1。
产物：`runs/a1_internal_frag/{frag.csv, run.log}`（runs/ 不进 git，见 .gitignore）
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvpager import BlockManager  # noqa: E402

GATE = 0.02          # 判据：误差 < 2 个百分点（绝对差，不是相对误差）
REF_DIST = "uniform:20,300"
FIELDS = ["dist", "block_size", "n", "seed", "mean_len", "max_len", "ratio_m_over_b",
          "ratio_max_over_b", "measured", "closed_old", "delta_old", "closed_new",
          "delta_new", "verdict_new"]

# ───────────────────────── 闭式式 ─────────────────────────
def closed_old(block_size: int, mean_len: float) -> float:
    """手册现行式：分母用 mean_len（假设 mean_len ≫ b）"""
    return (block_size - 1) / (2.0 * mean_len) if mean_len > 0 else 0.0

def closed_new(block_size: int, mean_len: float) -> float:
    """修正式：分母用分配容量 mean_len + w，w=(b−1)/2"""
    w = (block_size - 1) / 2.0
    return w / (mean_len + w) if mean_len + w > 0 else 0.0

# ───────────────────────── 长度分布 ─────────────────────────
def gen_lengths(spec: str, n: int, seed: int) -> list[int]:
    """按 `--lens` 规格生成序列长度；同一 (spec, n, seed) 必须给同一串数与"""
    rng = random.Random(seed)
    kind, _, params = spec.partition(":")
    nums = [float(x) for x in params.split(",")] if params else []

    if kind == "uniform":
        lo, hi = int(nums[0]), int(nums[1])
        return [rng.randint(lo, hi) for _ in range(n)]

    if kind == "lognormal":
        target_mu, sigma = nums[0], nums[1]
        # 让 lognormal 的均值 ≈ target_mu
        mu = math.log(target_mu) - 0.5 * sigma ** 2
        return [max(1, int(round(rng.lognormvariate(mu, sigma)))) for _ in range(n)]

    if kind == "mixed":
        lo, hi = int(nums[0]), int(nums[1])
        w_uniform = nums[2] if len(nums) > 2 else 0.7
        sigma = 0.8
        mu = math.log((lo + hi) / 2) - 0.5 * sigma ** 2
        out = []
        for _ in range(n):
            if rng.random() < w_uniform:
                out.append(rng.randint(lo, hi))
            else:
                out.append(max(1, int(round(rng.lognormvariate(mu, sigma)))))
        return out

    raise ValueError(f"不认识的 --lens 规格：{spec}（见文件头）")

def expand_lens(spec: str) -> list[str]:
    if spec == "all":
        return [REF_DIST, "lognormal:160,0.8", "mixed:20,300"]
    return [spec]

# ───────────────────────── 测量 ─────────────────────────
def measure(block_size: int, lens: list[int]) -> float:
    """把 lens 全部建成序列，返回 BlockManager 的实测内部碎片率（无前缀共享）"""
    need = sum(math.ceil(L / block_size) for L in lens)
    bm = BlockManager(num_blocks=need, block_size=block_size)   # 恰好装下，不触发 OutOfBlocks
    for L in lens:
        bm.new_sequence(L)
    return bm.stats()["internal_frag"]

def run_cell(dist: str, block_size: int, lens: list[int], n: int, seed: int) -> dict:
    measured = measure(block_size, lens)
    mean_len = sum(lens) / len(lens)
    max_len = max(lens)
    c_old = closed_old(block_size, mean_len)
    c_new = closed_new(block_size, mean_len)
    d_old = abs(measured - c_old)
    d_new = abs(measured - c_new)
    return {
        "dist": dist, "block_size": block_size, "n": n, "seed": seed,
        "mean_len": round(mean_len, 2), "max_len": max_len,
        "ratio_m_over_b": round(mean_len / block_size, 2),
        "ratio_max_over_b": round(max_len / block_size, 2),
        "measured": round(measured, 6),
        "closed_old": round(c_old, 6), "delta_old": round(d_old, 6),
        "closed_new": round(c_new, 6), "delta_new": round(d_new, 6),
        "verdict_new": "PASS" if d_new < GATE else "FAIL",
    }

# ───────────────────────── 输出 ─────────────────────────
def md_tables(rows: list[dict], lens_specs: list[str]) -> str:
    buf = io.StringIO()
    for dist in lens_specs:
        sub = [r for r in rows if r["dist"] == dist]
        if not sub:
            continue
        buf.write(f"\n### 分布 `{dist}`（n={sub[0]['n']}, seed={sub[0]['seed']}）\n\n")
        buf.write("| block_size | mean_len | mean_len/b | max_len/b | 实测 | 原式 (b−1)/(2·m) | Δ原式 | "
                  "修正式 w/(m+w) | Δ修正式 | 判定 |\n")
        buf.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in sub:
            mark = "✅" if r["delta_new"] < GATE else "❌"
            old_mark = "" if r["delta_old"] < GATE else " ❌破闸"
            buf.write(f"| {r['block_size']} | {r['mean_len']} | {r['ratio_m_over_b']} | "
                      f"{r['ratio_max_over_b']} | "
                      f"{r['measured']:.4f} | {r['closed_old']:.4f} | {r['delta_old']:.4f}{old_mark} | "
                      f"{r['closed_new']:.4f} | {r['delta_new']:.4f} | {mark} {r['verdict_new']} |\n")
    return buf.getvalue()

def domain_report(rows: list[dict]) -> list[str]:
    """两条闭式式的适用域，附机制解释"""
    out = []
    for key, label in (("delta_old", "原式 (b−1)/(2·mean_len)"),
                       ("delta_new", "修正式 w/(mean_len+w)")):
        passed = [r for r in rows if r[key] < GATE]
        failed = [r for r in rows if r[key] >= GATE]
        if not failed:
            out.append(f"- {label}：本网格全部 <{GATE}（没扫到失效点）")
            continue
        # 原式失效率随 mean_len/b 下降单调 ⇒ 用"通过格里的最小比值"作下界
        lo_ok = min((r["ratio_m_over_b"] for r in passed), default=None)
        hi_bad = max(r["ratio_m_over_b"] for r in failed)
        worst = max(rows, key=lambda r: r[key])
        out.append(f"- {label}：mean_len/b ≥ {lo_ok} 时 <{GATE}；"
                   f"≤ {hi_bad} 时破闸（最坏 Δ={worst[key]:.4f} @ b={worst['block_size']}）")
    out.append("  · 机制：原式漏掉分母里的 w，等价假设 mean_len ≫ b ⇒ 只有大比值时才碰巧对；"
               "修正式把分母换成「分配容量」，靠的是 `n mod b 近似均匀 ⇒ E[浪费]=(b−1)/2`。")
    over = [r for r in rows if r["ratio_max_over_b"] < 1.0]
    if over:
        bmax = min(r["block_size"] for r in over)
        out.append(f"  · 修正式的上界机制：当 block_size 大到超过序列最大长度（本例 max_len="
                   f"{rows[0]['max_len']}，b={bmax} 起），每条序列只占 1 块、`n mod b` 不再均匀，"
                   f"`E[浪费]=(b−1)/2` 的前提破 ⇒ 修正式也开始偏小，需改用直接按实测分布求和。")
    return out

def main() -> int:
    global GATE
    ap = argparse.ArgumentParser(description="A1 表1：内部碎片率 vs block_size")
    ap.add_argument("--block-sizes", default="8,16,32,64")
    ap.add_argument("--lens", default="all", help="all | uniform:lo,hi | lognormal:mu,sigma | mixed:lo,hi[,w]")
    ap.add_argument("--n", type=int, default=3000, help="每条分布的序列条数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("runs", "a1_internal_frag"))
    ap.add_argument("--gate", type=float, default=GATE)
    args = ap.parse_args()

    GATE = args.gate
    block_sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]
    lens_specs = expand_lens(args.lens)

    t0 = time.perf_counter()
    rows: list[dict] = []
    for spec in lens_specs:
        lens = gen_lengths(spec, args.n, args.seed)   # 每条分布同一 seed ⇒ 可复现
        for b in block_sizes:
            rows.append(run_cell(spec, b, lens, args.n, args.seed))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "frag.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    cmd = "python kernel/bench_internal_frag.py " + " ".join(
        ["--block-sizes", args.block_sizes, "--lens", args.lens,
         "--n", str(args.n), "--seed", str(args.seed), "--gate", str(args.gate)])
    print(f"# 命令：{cmd}")
    print(f"# 耗时：{time.perf_counter() - t0:.2f}s  产物：{args.out}/frag.csv")
    body = md_tables(rows, lens_specs)
    print(body)

    # ── 闸门：修正式在参照分布上四档全过 ──
    ref = [r for r in rows if r["dist"] == REF_DIST]
    if not ref:
        print(f"⚠️ 本次没跑参照分布 {REF_DIST}，闸门未判定")
    ref_ok = bool(ref) and all(r["delta_new"] < GATE for r in ref)
    all_new_ok = all(r["delta_new"] < GATE for r in rows)
    print(f"## 判定（判据 Δ<{GATE}）= 绝对百分点")
    print(f"- 修正式全网格：{'✅ 全部通过' if all_new_ok else '❌ 有格破闸'}"
          f"（最坏 Δ={max(r['delta_new'] for r in rows):.4f}）")
    for line in domain_report(rows):
        print(line)
    print(f"- 参照分布 {REF_DIST}：{'✅ PASS' if ref_ok else '❌ FAIL'}")

    with open(os.path.join(args.out, "run.log"), "w", encoding="utf-8") as f:
        f.write(f"# 命令：{cmd}\n")
        f.write(body)
        f.write(f"\n修正式全网格：{'PASS' if all_new_ok else 'FAIL'}\n")
        f.write("适用域：\n" + "\n".join(domain_report(rows)) + "\n")
        f.write(f"参照分布判定：{'PASS' if ref_ok else 'FAIL'}\n")
    return 0 if ref_ok else 1

if __name__ == "__main__":
    sys.exit(main())
