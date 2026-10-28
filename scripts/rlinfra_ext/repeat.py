# 🕒 v1 · 2026-10-24 12:41 · dsh-agent ｜ 变更：×N 聚合并给出 MDE 与 Welch 判可区分性

"""同配置 ×N 重复：给"可辨识性"一个下限（M3 纪律）

为什么必须做：本仓实测同一配置重复的 SD 可以到 **±167~181 步**（`opt_exp/results/README.md:318`），
而多数"改进"的效应量就在这个量级 —— 没有 ×N 就分不清"改进"和"运气"。

输入：若干 run 目录（每个含 `eval.json`，或 `train.log` 里带 `测试汇总 | … 覆盖率：xx%`）
输出：
  · mean ± SD（跨 run，**不是**跨测试轮次 —— 后者是假重复）
  · **MDE**（最小可辨识效应量，近似 `2·SD/√N`）：小于它的差异不能当结论
  · 判定：给定两组各自的 (mean, SD, N)，用 Welch t 近似算能否区分

用法：
    python tools/rlinfra_ext/repeat.py runs/demo_300 runs/demo_60 ...
    python tools/rlinfra_ext/repeat.py --group A=runs/a1,runs/a2 --group B=runs/b1,runs/b2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

SUM_PAT = re.compile(r"测试汇总.*?全局平均覆盖率[：:]\s*([0-9.]+)%")


def read_run(d):
    """返回该 run 的评估覆盖率（优先 eval.json，退回 train.log 的测试汇总行）"""
    j = os.path.join(d, "eval.json")
    if os.path.exists(j):
        try:
            return float(json.load(open(j, encoding="utf-8"))["coverage"]) / 100.0
        except Exception:
            pass
    log = os.path.join(d, "train.log")
    if os.path.exists(log):
        hits = []
        for line in open(log, encoding="utf-8", errors="replace"):
            m = SUM_PAT.search(line)
            if m:
                hits.append(float(m.group(1)) / 100.0)
        if hits:
            return hits[-1]
    # 本仓（UAV opt）没有 eval.json / '测试汇总' 行 ⇒ 退回逐轮 JSON 的均值
    import glob
    for sub in ("test_path_plots_cn", "test_path_plots"):
        files = sorted(glob.glob(os.path.join(d, sub, "*.json")))
        if files:
            covs = []
            for p in files:
                try:
                    m = json.load(open(p, encoding="utf-8")).get("metrics", {})
                    if "global_coverage_rate" in m:
                        covs.append(float(m["global_coverage_rate"]))
                except Exception:
                    continue
            if covs:
                return float(np.mean(covs))
    return None


def stats(vals):
    v = np.array(vals, dtype=float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return float(v.mean()), sd, n, (2.0 * sd / math.sqrt(n) if n else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="run 目录列表（同一配置的 ×N 重复）")
    ap.add_argument("--group", action="append", default=[],
                    help="分组：name=dir1,dir2,...（给了就做两组对照）")
    args = ap.parse_args()

    if args.group:
        groups = {}
        for g in args.group:
            name, rest = g.split("=", 1)
            groups[name] = [read_run(d) for d in rest.split(",") if d.strip()]
        for name, vals in groups.items():
            got = [v for v in vals if v is not None]
            if not got:
                print(f"{name}: 没有可用结果")
                continue
            m, sd, n, mde = stats(got)
            print(f"{name:<12} N={n}  mean={m*100:.2f}%  SD={sd*100:.2f}pp  MDE(2SD/√N)={mde*100:.2f}pp")
        names = list(groups)
        if len(names) == 2:
            a = [v for v in groups[names[0]] if v is not None]
            b = [v for v in groups[names[1]] if v is not None]
            if len(a) > 1 and len(b) > 1:
                ma, sa, na, _ = stats(a)
                mb, sb, nb, _ = stats(b)
                se = math.sqrt(sa ** 2 / na + sb ** 2 / nb)
                t = (ma - mb) / se if se > 0 else float("inf")
                print(f"\nWelch 近似：Δ={(ma-mb)*100:+.2f}pp  SE={se*100:.2f}pp  t={t:.2f}  "
                      f"{'可区分(|t|>2.2)' if abs(t) > 2.2 else '不可区分(|t|≤2.2)'}")
                print("⚠️ 若 SD 极大（本仓见过 ±167 步），先把 N 加上去再谈差异")
        return 0

    got = [v for v in (read_run(d) for d in args.dirs) if v is not None]
    if not got:
        print("❌ 没有解析到任何结果（要 eval.json 或带'测试汇总'的 train.log）")
        return 2
    m, sd, n, mde = stats(got)
    print(f"N={n}  mean={m*100:.2f}%  SD={sd*100:.2f}pp")
    print(f"MDE（2·SD/√N）= {mde*100:.2f}pp ⇒ 小于它的差异**不能**当结论")
    print("单项：" + ", ".join(f"{v*100:.2f}%" for v in got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
