#!/usr/bin/env bash
# 🕒 v1 · 2026-10-25 17:05 · dsh-agent ｜ 变更：B2 第三轮 —— CPU 段扫描，找 overlap 的收益交叉点
#
# 动机：91657 实测 GPU 利用率 98~100%、加速比 0.933×（overlap 更慢 7.2%）。
# 机制是「GPU 本来就满，没有空闲资源可叠」。那 overlap 到底在什么条件下才赚？
# 答案必须来自「rollout 侧非 GPU 段」有多大 —— 所以扫它。
set -euo pipefail

ROOT=/storage/public/home/__HPC_ACCOUNT__/RL_infra

jsub -q nonmig -n 6 -gpgpu 1 -R "span[hosts=1]" -m a40gpu01 \
     -J infr-cpu3-g \
     -o "output.%J" -e "error.%J" \
     -cwd "$ROOT" \
     bash "$ROOT/hpc/run_b_overlap.sh" \
       --device cuda --seed 0 --steps 40 --reps 2 --warmup 4 --batch 8 --max-new-tokens 32 \
       --sweep-cpu 0,5,10,20,40,80
