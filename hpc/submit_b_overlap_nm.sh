#!/usr/bin/env bash
# 🕒 v1 · 2026-10-24 18:28 · dsh-agent ｜ 变更：B2 第二轮 —— nonmig 整卡（锁 a40gpu01，避开 a100gpu01 的 glibc 2.17 坑）
#
# 为什么换 nonmig：① gpu 队列 63 PEND，nonmig 只 11；② **MIG 上 utilization 测不出来**，
#   整卡才有机会拿到 B2 要的第 2 个数；③ 模型 <1 GB，整卡属浪费但配额同为 u8 组 2 名额。
set -euo pipefail

ROOT=/storage/public/home/__HPC_ACCOUNT__/RL_infra

jsub -q nonmig -n 6 -gpgpu 1 -R "span[hosts=1]" -m a40gpu01 \
     -J infr-ovl2-g \
     -o "output.%J" -e "error.%J" \
     -cwd "$ROOT" \
     bash "$ROOT/hpc/run_b_overlap.sh" \
       --device cuda --seeds 3 --steps 60 --reps 3 --warmup 6 --max-new-tokens 32 --batch 8
