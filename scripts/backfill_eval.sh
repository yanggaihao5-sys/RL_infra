#!/bin/bash
# 🕒 v1 · 2026-09-30 22:01 · dsh-agent ｜ 变更：一次性回填：把已完成评测写进 runs/<name>/eval.json

# 一次性回填：把已经跑出来的评测结果写进 runs/<name>/eval.json（F 的落地）
# 数据来源：已完成的评测作业 output.71823 的「测试汇总」行（不重新跑评测，不占 GPU）
set -u
PROJ=/storage/public/home/__HPC_ACCOUNT__/RL_infra
cd "$PROJ" || exit 1
declare -A COV=( [lr1e3_s2024]=70.62 [lr5e4_s2024]=80.41 [lr1e4_s2024]=77.06 [lr1e3_s7]=62.63 )
declare -A REW=( [lr1e3_s2024]=124.86 [lr5e4_s2024]=158.70 [lr1e4_s2024]=142.05 [lr1e3_s7]=108.60 )
for n in lr1e3_s2024 lr5e4_s2024 lr1e4_s2024 lr1e3_s7; do
    printf '{"coverage": %s, "reward": %s, "success_rate": 0.0, "steps": 200.0, "evaluated_at": "2026-09-30 job71823", "source": "output.71823"}\n' \
        "${COV[$n]}" "${REW[$n]}" > "runs/$n/eval.json"
    echo "写入 runs/$n/eval.json -> cov=${COV[$n]}% reward=${REW[$n]}"
done
echo "--- 校验 ---"
for n in lr1e3_s2024 lr5e4_s2024 lr1e4_s2024 lr1e3_s7; do cat "runs/$n/eval.json"; done
