#!/usr/bin/env bash
# 🕒 v1 · 2026-10-24 14:20 · dsh-agent ｜ 变更：B2 提交脚本（GPU 队列 mig=1，单节点）
# 用法：bash <本脚本>   —— 内含完整 jsub 命令，避免多层引号被 PowerShell/ssh 剥掉
set -euo pipefail

ROOT=/storage/public/home/__HPC_ACCOUNT__/RL_infra

jsub -q gpu -n 6 -gpgpu "1 mig=1" -R "span[hosts=1]" \
     -J infr-ovl-g \
     -o "output.%J" -e "error.%J" \
     -cwd "$ROOT" \
     bash "$ROOT/hpc/run_b_overlap.sh"
