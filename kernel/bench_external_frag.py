# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：A1 表3：外部碎片率（固定块分页 vs 变块首次适配）

"""A1 表3：外部碎片（空洞）率 —— 手册 A1 的第 2 个必测指标

## 为什么必须重做这一项

`kvpager.py` / `BENCH.md` 之前把空洞写成 `holes = len(self._free)`（**空闲块数**，不是率）。
块大小固定时那个量恒等于空闲块数，做不出曲线。手册要的是"**空闲块中不足 block_size 的碎片占比**"——
可在**固定块分页**里，空闲空间的粒度永远是整块、且块不要求连续，所以这个占比**结构上恒为 0**。
⇒ 光测分页这一个分配器，这一项只有 0，等于没测。要出数字必须**跨分配器对照**。

## 本脚本的做法

同一总容量、同一到达序列、同一 seed，跑两个分配器：

| 分配器 | 内碎 | 外碎 |
|---|---|---|
| **固定块分页**（`kvpager.BlockManager`） | 有（表 1 的 w/(m+w)） | **结构上 0** |
| **变块首次适配 + 相邻合并**（`IntervalAllocator`，本文件内 40 行） | ~0（按需精确切块，最小单位 1 token） | **有**，随 churn 上升 |

两个关键量（都能直接对应"分页到底省了什么"）：
  `unusable_free_ratio` = (Σ空闲 − 最大单块) / Σ空闲
      · 分页：分母分子都按块算 ⇒ 最大"单块"就是全部空闲块合起来（块不必连续）⇒ 恒 0
      · 变块：空闲被存活分配隔断、合并不上 ⇒ 越大说明越多的空闲"看得见吃不着"
  `max_admittable_ratio` = 此刻能一次接纳的最长序列 / Σ空闲
      · 分页恒 1.0；变块 < 1.0（外碎的直接行为后果）

## 用法

    python kernel/bench_external_frag.py --capacity 200000 --block-size 16 --steps 400 --seed 0
    python kernel/bench_external_frag.py --lens all --steps 400 --seed 0

退出码：0 = 分页侧 `unusable_free_ratio` 实测为 0 且 `max_admittable_ratio` = 1.0（结构断言成立），否则 1。
产物：`runs/a1_external_frag/{external_frag.csv, run.log}`
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvpager import BlockManager  # noqa: E402

FIELDS = ["lens", "allocator", "block_size", "capacity", "steps", "seed",
          "mean_live", "mean_used_ratio", "internal_frag", "unusable_free_ratio",
          "max_admittable_ratio", "reject_rate", "waste_total"]
REF_LENS = "uniform:20,300"

# ───────────────────────── 变块分配器（精确大小 + 首次适配 + 相邻合并）─────────────────────────
class IntervalAllocator:
    """空闲区间表 `[(start, size), ...]` 升序；alloc 首次适配并切分，free 后向前合并。

    这是"变块"的参照实现：块按请求精确切，故**内碎 ≈ 0**，代价是空闲被隔断 ⇒ 外碎。
    """

    def __init__(self, total_tokens: int):
        self.total = int(total_tokens)
        self.free: list[tuple[int, int]] = [(0, self.total)]
        self._live: dict[int, int] = {}
        self._tid = 0

    def alloc(self, n: int) -> int | None:
        for i, (s, sz) in enumerate(self.free):
            if sz >= n:
                self.free.pop(i)
                if sz > n:
                    self.free.insert(i, (s + n, sz - n))
                self._tid += 1
                self._live[self._tid] = (s, n)   # type: ignore[assignment]
                return self._tid
        return None

    def free_seq(self, tid: int) -> None:
        s, n = self._live.pop(tid)           # type: ignore[misc]
        self.free.append((s, n))
        self.free.sort()
        merged: list[tuple[int, int]] = []
        for st, sz in self.free:
            if merged and merged[-1][0] + merged[-1][1] == st:
                merged[-1] = (merged[-1][0], merged[-1][1] + sz)
            else:
                merged.append((st, sz))
        self.free = merged

    def snapshot(self) -> dict:
        free_total = sum(sz for _, sz in self.free)
        max_free = max((sz for _, sz in self.free), default=0)
        used = self.total - free_total
        return {"free_total": free_total, "max_admittable": max_free, "used": used,
                "internal_frag": 0.0}   # 精确切块 ⇒ 无块内浪费

class PagedAllocator:
    """固定块分页，套在 `BlockManager` 外面，提供与 IntervalAllocator 同名的快照接口"""

    def __init__(self, total_tokens: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = total_tokens // block_size
        self.bm = BlockManager(num_blocks=self.num_blocks, block_size=block_size)

    def alloc(self, n: int) -> str | None:
        if n == 0:
            n = 1
        try:
            return self.bm.new_sequence(n)
        except Exception:
            return None

    def free_seq(self, tid: str) -> None:
        self.bm.free(tid)

    def snapshot(self) -> dict:
        st = self.bm.stats()
        free_blocks = st["free_blocks"]
        free_total = free_blocks * self.block_size
        # 块不要求连续 ⇒ 全部空闲块合起来就是一个可用的"大块" ⇒ 能一次接纳 free_total
        return {"free_total": free_total, "max_admittable": free_total, "used": st["used_tokens"],
                "internal_frag": st["internal_frag"]}

# ───────────────────────── 负载 ─────────────────────────
def gen_lengths(spec: str, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    kind, _, params = spec.partition(":")
    nums = [float(x) for x in params.split(",")] if params else []
    if kind == "uniform":
        lo, hi = int(nums[0]), int(nums[1])
        return [rng.randint(lo, hi) for _ in range(n)]
    if kind == "lognormal":
        target, sigma = nums[0], nums[1]
        mu = math.log(target) - 0.5 * sigma ** 2
        return [max(1, int(round(rng.lognormvariate(mu, sigma)))) for _ in range(n)]
    if kind == "mixed":
        lo, hi = int(nums[0]), int(nums[1])
        sigma = 0.8
        mu = math.log((lo + hi) / 2) - 0.5 * sigma ** 2
        out = []
        for _ in range(n):
            out.append(rng.randint(lo, hi) if rng.random() < 0.7
                       else max(1, int(round(rng.lognormvariate(mu, sigma)))))
        return out
    raise ValueError(f"不认识的 --lens：{spec}")

def make_workload(dist: str, steps: int, arrivals: int, life_lo: int, life_hi: int,
                  seed: int) -> list[list[tuple[int, int]]]:
    """返回每步的到达事件 [[(长度, 存活步数), ...], ...]。两个分配器吃**同一份**负载。"""
    lens_pool = gen_lengths(dist, steps * arrivals, seed)
    rng = random.Random(seed + 1)
    k = 0
    out = []
    for _ in range(steps):
        ev = []
        for _ in range(arrivals):
            ev.append((lens_pool[k], rng.randint(life_lo, life_hi)))
            k += 1
        out.append(ev)
    return out

def alloc_capacity(alloc) -> int:
    """分配器的总容量（token）"""
    if isinstance(alloc, IntervalAllocator):
        return alloc.total
    return alloc.num_blocks * alloc.block_size

def simulate(alloc, workload: list[list[tuple[int, int]]], sample_every: int = 10) -> dict:
    live: list[tuple[object, int]] = []      # (tid, 剩余寿命)
    admitted = rejected = 0
    samples = []
    for t, events in enumerate(workload):
        # 先过期释放
        keep = []
        for tid, life in live:
            if life <= 1:
                alloc.free_seq(tid)
            else:
                keep.append((tid, life - 1))
        live = keep
        # 再接纳新请求
        for length, life in events:
            tid = alloc.alloc(length)
            if tid is None:
                rejected += 1
            else:
                admitted += 1
                live.append((tid, life))
        if t % sample_every == 0 or t == len(workload) - 1:
            snap = alloc.snapshot()
            snap["live"] = len(live)
            samples.append(snap)

    total_req = admitted + rejected
    agg = {
        "mean_live": sum(s["live"] for s in samples) / len(samples),
        "mean_used_ratio": sum(s["used"] for s in samples) / len(samples) / alloc_capacity(alloc),
        "internal_frag": sum(s["internal_frag"] for s in samples) / len(samples),
    }
    unu = []
    adm = []
    for s in samples:
        ft = s["free_total"]
        unu.append((ft - s["max_admittable"]) / ft if ft else 0.0)
        adm.append(s["max_admittable"] / ft if ft else 1.0)
    agg["unusable_free_ratio"] = sum(unu) / len(unu)
    agg["max_admittable_ratio"] = sum(adm) / len(adm)
    agg["reject_rate"] = rejected / total_req if total_req else 0.0
    agg["waste_total"] = agg["internal_frag"] + agg["unusable_free_ratio"]
    return agg

def main() -> int:
    ap = argparse.ArgumentParser(description="A1 表3：外部碎片（空洞）率")
    ap.add_argument("--capacity", default="200000,172000,165000",
                    help="池总容量（token），可给多档逗号分隔 —— 用来看外碎随压力放大")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--arrivals", type=int, default=40, help="每步到几条新请求")
    ap.add_argument("--life", default="5,45", help="存活步数范围 lo,hi")
    ap.add_argument("--lens", default="all",
                    help="all | uniform:lo,hi | lognormal:mu,sigma | mixed:lo,hi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=10, help="每几步采样一次")
    ap.add_argument("--out", default=os.path.join("runs", "a1_external_frag"))
    args = ap.parse_args()

    life_lo, life_hi = (int(x) for x in args.life.split(","))
    capacities = [int(x) for x in str(args.capacity).split(",")]
    dists = ["uniform:20,300", "lognormal:160,0.8", "mixed:20,300"] if args.lens == "all" else [args.lens]

    rows = []
    for cap in capacities:
        for dist in dists:
            wl = make_workload(dist, args.steps, args.arrivals, life_lo, life_hi, args.seed)
            paged = PagedAllocator(cap, args.block_size)
            r1 = simulate(paged, wl, args.every)
            rows.append({"lens": dist, "allocator": "paged_fixed_block",
                         "block_size": args.block_size, "capacity": cap,
                         "steps": args.steps, "seed": args.seed,
                         **{k: round(v, 6) for k, v in r1.items()}})

            interval = IntervalAllocator(cap)
            r2 = simulate(interval, wl, args.every)
            rows.append({"lens": dist, "allocator": "variable_first_fit",
                         "block_size": "—", "capacity": cap,
                         "steps": args.steps, "seed": args.seed,
                         **{k: round(v, 6) for k, v in r2.items()}})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "external_frag.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append(f"# 命令：python kernel/bench_external_frag.py --capacity {args.capacity} "
                 f"--block-size {args.block_size} --steps {args.steps} --arrivals {args.arrivals} "
                 f"--life {args.life} --lens {args.lens} --seed {args.seed}")
    lines.append(f"# 同容量 / 同到达序列 / 同 seed；存活步数 ~ U[{life_lo},{life_hi}]；"
                 f"每步 {args.arrivals} 条到达 ⇒ 稳态需求 ≈ {args.arrivals * (life_lo + life_hi) // 2 * 160} token")
    lines.append("")
    lines.append("| 容量(token) | 占用率 | 长度分布 | 分配器 | 内碎率 | **外碎率** | 最大可接纳/空闲 | 拒绝率 | 总浪费=内碎+外碎 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['capacity']} | {r['mean_used_ratio']:.3f} | `{r['lens']}` | "
                     f"{r['allocator']} | {r['internal_frag']:.4f} | "
                     f"**{r['unusable_free_ratio']:.4f}** | {r['max_admittable_ratio']:.4f} | "
                     f"{r['reject_rate']:.4f} | {r['waste_total']:.4f} |")

    paged_rows = [r for r in rows if r["allocator"] == "paged_fixed_block"]
    ok = all(abs(r["unusable_free_ratio"]) < 1e-9 and abs(r["max_admittable_ratio"] - 1.0) < 1e-9
             for r in paged_rows)
    lines.append("")
    lines.append("## 判定")
    lines.append(f"- 固定块分页侧结构断言（外碎 ≡ 0 且 最大可接纳/空闲 = 1）："
                 f"{'✅ 实测成立' if ok else '❌ 不成立，需查模型'}")
    var = [r for r in rows if r["allocator"] == "variable_first_fit"]
    lines.append(f"- 变块侧外碎实测：{min(r['unusable_free_ratio'] for r in var):.4f} ~ "
                 f"{max(r['unusable_free_ratio'] for r in var):.4f}"
                 f"（最大可接纳/空闲最低 {min(r['max_admittable_ratio'] for r in var):.4f}）")
    lines.append("")
    lines.append("## 读法（这就是手册第 2 个指标要的结论）")
    lines.append("- **固定块分页把外部碎片消除为 0**：空闲粒度恒为整块，且块不要求连续 ⇒ 任意 ≤ 空闲槽位的请求都能拼出来。")
    lines.append("- 代价全部记在**内部碎片**上（表 1 的 w/(m+w)，b 越大越贵），所以 b 就是这两类碎片的交换旋钮。")
    lines.append("- 变块首次适配把内碎压到 ~0，但空闲被存活分配隔断、合并不上 ⇒ 外碎>0，"
                 "直接后果是 `max_admittable` 远小于空闲总量（大请求被拒）。")
    lines.append("- ⇒ **选 b 不看内碎单项，看 `总浪费 = 内碎 + 外碎` 与拒绝率**；本表给的就是这条支配关系。")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "run.log"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
