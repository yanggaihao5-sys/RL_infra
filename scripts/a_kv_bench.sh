#!/usr/bin/env bash
# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：A 阶段一键复现（Linux 薄壳，逻辑在 a_kv_bench.py）


# 🕒 v2 · 2026-10-24 15:30 · dsh-agent ｜ 变更：改为薄壳，逻辑统一在 a_kv_bench.py（本机无 bash 时也能验证） ｜ 上一版：v1
#
# A 阶段一键复现（给 481 / 高算等 Linux 机器用；逻辑全在 a_kv_bench.py）
#   bash scripts/a_kv_bench.sh              # 全跑
#   bash scripts/a_kv_bench.sh --fast        # 跳过 A2 的 CV 扫描
#   bash scripts/a_kv_bench.sh --no-torch    # 没 torch 时只跑 A1
#   PYTHON=/path/to/python bash scripts/a_kv_bench.sh   # 指定解释器
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONIOENCODING=utf-8
exec "${PYTHON:-python}" "$HERE/a_kv_bench.py" "$@"
