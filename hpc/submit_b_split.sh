#!/usr/bin/env bash
# 🕒 v1 · 2026-10-25 20:30 · dsh-agent ｜ 变更：推/训分离实验 —— 一个作业要 2 个 MIG 切片，rollout 与 trainer 分卡
#
# 兑现 docs/预注册_推训分离对照.md（**该文件在本次提交前已冻结**）。
# 与前三轮的区别：前几轮 rollout 与 trainer 在同卡上抢设备；这次分到 cuda:0 / cuda:1。
set -euo pipefail

ROOT=/storage/public/home/__HPC_ACCOUNT__/RL_infra

jsub -q gpu -n 6 -gpgpu "2 mig=1" -R "span[hosts=1]" \
     -J infr-splt-g \
     -o "output.%J" -e "error.%J" \
     -cwd "$ROOT" \
     bash "$ROOT/hpc/run_b_overlap.sh" \
       --device cuda:1 --rollout-device cuda:0 \
       --seeds 3 --steps 60 --reps 3 --warmup 6 --batch 8 --max-new-tokens 32
