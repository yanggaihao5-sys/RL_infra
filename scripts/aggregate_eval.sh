#!/bin/bash
# 🕒 v1 · 2026-10-06 14:51 · dsh-agent ｜ 变更：新增：从逐图 test_path_plots_en/*.json 聚合 mean/std 并写回 eval.json

# 把「最近 N 次评测」的逐图结果聚合成 mean/std/min/max 并写回 runs/<name>/eval.json
#
# 为什么需要：本次 --maps 20 的「测试汇总」行里没有标准差（远端 trainer 版本较旧），
# 但逐图结果存在 runs/<name>/test_path_plots_en/*.json（含 run 序号 + coverage），
# 所以标准差可以**从原始数据算出来**，不必改远端代码。
#
# 用法：bash scripts/aggregate_eval.sh <N> <实验名...>
set -u
PROJ=/storage/public/home/__HPC_ACCOUNT__/RL_infra
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python
cd "$PROJ" || exit 1
N="${1:?用法: aggregate_eval.sh <每个run取最近N次> <实验名...>}"; shift
[ $# -gt 0 ] || { echo "需要至少一个实验名"; exit 2; }

"$PY" - "$N" "$@" <<'PYEOF'
import glob, json, statistics, sys, os
n = int(sys.argv[1]); runs = sys.argv[2:]
print(f"{'run':<20} {'n':>3} {'mean%':>8} {'std(pp)':>8} {'min%':>7} {'max%':>7}")
for r in runs:
    recs = []
    for f in glob.glob(f"runs/{r}/test_path_plots_en/*.json"):
        try:
            d = json.load(open(f)); recs.append((int(d.get("run", 0)), float(d["coverage"])))
        except Exception:
            pass
    recs.sort()
    covs = [c for _, c in recs[-n:]]
    if not covs:
        print(f"{r:<20} 无逐图数据"); continue
    m, s = statistics.mean(covs), (statistics.pstdev(covs) if len(covs) > 1 else 0.0)
    print(f"{r:<20} {len(covs):>3} {m:>8.2f} {s:>8.2f} {min(covs):>7.2f} {max(covs):>7.2f}")
    p = f"runs/{r}/eval.json"
    old = {}
    if os.path.exists(p):
        try: old = json.load(open(p))
        except Exception: old = {}
    old.update({"coverage": round(m, 2), "coverage_std": round(s, 2), "maps": len(covs),
                "coverage_min": round(min(covs), 2), "coverage_max": round(max(covs), 2)})
    tmp = p + ".tmp"
    json.dump(old, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, p)
print("已把 mean/std 写回各 runs/<name>/eval.json")
PYEOF
