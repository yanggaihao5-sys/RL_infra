#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# RL_infra · 高算平台只读探针（10 分钟内结束，不改任何远程文件）
#
# 回答两个悬而未决的问题（2026-09-30 提出）：
#   ① `-gpgpu 2` 到手的**整卡 A800 到底多少显存**（平台配置里查不到权威条目）
#   ② 计算节点（不是登录节点）**能否连外网**下模型权重 —— 决定权重是"直接拉"还是"本地传"
#
# 用法：
#   jsub -q gpu -n 2 -gpgpu 2 -R "span[hosts=1]" -W 10 -J rli-probe-g \
#        -o output.%J -e error.%J -cwd /storage/public/home/__HPC_ACCOUNT__/RL_infra/ \
#        bash /storage/public/home/__HPC_ACCOUNT__/RL_infra/scripts/probe.sh
# ════════════════════════════════════════════════════════════════════
set -u
export PYTHONIOENCODING=utf-8

echo "===== PROBE START job=${JOB_ID:-local} node=$(hostname) $(date '+%F %T') ====="

echo "--- GPU 清单（显存/型号/算力）---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version --format=csv || echo "nvidia-smi 失败"

echo "--- MIG 实例（若为切片则可见）---"
nvidia-smi -L 2>/dev/null || true

echo "--- CUDA 驱动能力 ---"
nvidia-smi | grep -i "CUDA Version" || true

echo "--- 节点内存 / 磁盘 ---"
free -g | head -2
df -h "$PWD" 2>/dev/null | tail -1
echo "TMPDIR=${TMPDIR:-未设置}"

echo "--- 外网连通性（各 8 秒超时）---"
for u in https://hf-mirror.com https://pypi.tuna.tsinghua.edu.cn/simple/ https://www.modelscope.cn https://github.com; do
    printf '%-45s ' "$u"
    curl -sS --max-time 8 -o /dev/null -w 'http=%{http_code} time=%{time_total}s\n' "$u" 2>&1 | tail -1
done

echo "--- DNS ---"
getent hosts hf-mirror.com 2>/dev/null || echo "DNS 解析 hf-mirror.com 失败"
getent hosts pypi.tuna.tsinghua.edu.cn 2>/dev/null || echo "DNS 解析 pypi 失败"

echo "--- 现有 python 环境关键包 ---"
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python
"$PY" - <<'EOF' 2>&1 | tail -20
import importlib, sys
print("python", sys.version.split()[0])
for m in ("torch", "transformers", "peft", "trl", "vllm", "accelerate", "datasets", "bitsandbytes"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod, '__version__', '已装(版本未知)')}")
    except Exception:
        print(f"  {m:14s} 未装")
try:
    import torch
    print("  cuda available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    gpu{i}: {p.name} {p.total_memory/1024**3:.1f}GB cc={p.major}.{p.minor}")
except Exception as e:
    print("  torch 自检失败:", e)
EOF

echo "--- 原子写实测（Linux：并发读不得出现坏文件）---"
# 🔴 不许用 `python - <<EOF`（stdin 模式）：实测在 a100gpu01 上会
#    `ImportError: Error importing numpy: you should not try to import numpy from its source directory`
#    —— 同一个 import 用 `python -c` / 脚本文件都正常。坑：stdin 模式下 sys.path[0]='' 语义不同。
if [ -f "$PWD/utils/_verify_atomic.py" ]; then
    ( cd "$PWD" && "$PY" "$PWD/utils/_verify_atomic.py" ) 2>&1 | tail -8
else
    echo "  跳过：utils/_verify_atomic.py 不存在（该文件没上传？）"
fi

echo "===== PROBE END $(date '+%F %T') ====="
