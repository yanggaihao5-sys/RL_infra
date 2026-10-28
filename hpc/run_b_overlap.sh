#!/usr/bin/env bash
# 🕒 v1 · 2026-10-25 20:14 · dsh-agent ｜ 变更：位置参数透传 + 设备探测打印 nvidia-smi -L 与全部 cuda 设备


# 🕒 v2 · 2026-10-24 18:25 · dsh-agent ｜ 变更：参数改为位置参数透传（支持换档不改脚本） ｜ 上一版：v1
# 由 hpc/submit_b_overlap*.sh 通过 jsub 调起；不直接手动跑（那样不在调度器里）
set -euo pipefail

ROOT=/storage/public/home/__HPC_ACCOUNT__/RL_infra
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python
cd "$ROOT"
export PYTHONIOENCODING=utf-8

echo "===== 设备探测 ====="
echo "--- nvidia-smi -L（判断两块卡是同一物理卡的两个 MIG 切片，还是两块物理卡）---"
nvidia-smi -L 2>&1 || echo "(nvidia-smi 不可用)"
"$PY" - <<'PYEOF'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda_available", ok)
if ok:
    print("device_count =", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)} | total_mem_GB {round(p.total_memory/1024**3,2)}")
PYEOF

echo "===== B2 基准（参数透传：$*）====="
"$PY" traintest/bench_overlap.py "$@"

echo "===== 产物 ====="
ls -l runs/b_overlap/ 2>/dev/null || echo "(没有 runs/b_overlap —— 检查 error 日志)"
echo "===== 完成 ====="
