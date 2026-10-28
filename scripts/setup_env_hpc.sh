#!/bin/bash
# 在高算登录节点补齐 transformers/peft 依赖（离线 whl，登录节点无外网）
set -u
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python
W=/storage/public/home/__HPC_ACCOUNT__/RL_infra/wheels

echo "--- 安装剩余依赖 ---"
"$PY" -m pip install --no-index --find-links "$W" \
    regex fsspec dill multiprocess requests tqdm pyyaml xxhash 2>&1 | tail -3

echo "--- 验证 import ---"
"$PY" - <<'PYEOF' 2>&1 | tail -12
mods = ["torch", "transformers", "peft", "accelerate", "datasets", "sentencepiece"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  {m:14s} {getattr(mod, '__version__', 'OK')}")
    except Exception as e:
        print(f"  {m:14s} 失败: {type(e).__name__}: {e}")
import torch
print("  cuda:", torch.cuda.is_available())
PYEOF

echo "--- 原子写实测（Linux，登录节点 = 与 a800 同 glibc 2.28）---"
cd /storage/public/home/__HPC_ACCOUNT__/RL_infra
"$PY" utils/_verify_atomic.py 2>&1 | tail -5
