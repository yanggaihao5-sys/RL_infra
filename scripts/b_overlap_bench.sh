#!/usr/bin/env bash
# 🕒 v1 · 2026-10-24 14:08 · dsh-agent ｜ 变更：B 阶段一键跑（Linux 薄壳，逻辑在 b_overlap_bench.py）
#
# B 阶段一键跑（给 481 / 高算用；逻辑全在 b_overlap_bench.py）
#   bash scripts/b_overlap_bench.sh                  # 本机 CPU，只验结构
#   bash scripts/b_overlap_bench.sh --device cuda     # 高算 GPU 出数
#   bash scripts/b_overlap_bench.sh --quick           # 极小规模
#   PYTHON=/path/to/python bash scripts/b_overlap_bench.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONIOENCODING=utf-8
exec "${PYTHON:-python}" "$HERE/b_overlap_bench.py" "$@"
