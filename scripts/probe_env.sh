#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# RL_infra · 高算环境探针（最小版，几分钟内结束，不改远程文件）
#
# 回答三个问题：
#   ① 我**实际拿到**的是几张卡、什么型号？（`-gpgpu 2` 到底给了什么）
#   ② 这个节点的 python/torch 到不到手？（glibc / 环境差异 —— a100 节点已确认坏）
#   ③ 内部有没有能用的 pip / 模型镜像？（计算节点外网 DNS 不通）
#
# 用法：
#   jsub -q gpu -n 2 -gpgpu 2 -R "span[hosts=1]" -W 10 -J rli-env-g \
#        -o 'output.%J' -e 'error.%J' -cwd /storage/public/home/__HPC_ACCOUNT__/RL_infra/ \
#        bash /storage/public/home/__HPC_ACCOUNT__/RL_infra/scripts/probe_env.sh
# ════════════════════════════════════════════════════════════════════
set -u
export PYTHONIOENCODING=utf-8
PROJ=/storage/public/home/__HPC_ACCOUNT__/RL_infra
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python

echo "===== ENV PROBE job=${JOB_ID:-local} node=$(hostname) $(date '+%F %T') ====="
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<未设置>}'   ← 分配到的卡号"
echo "SLURM/平台资源变量："
env | grep -iE 'JOB_ID|NSLOTS|CUDA|GPU|HOSTNAME' | sort | head -20

echo "--- 物理卡总数（未受限视角）---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null | head -10

echo "--- glibc ---"
ldd --version 2>/dev/null | head -1

echo "--- python 环境 ---"
"$PY" -c "import sys; print('python', sys.version.split()[0])" 2>&1 | tail -2
"$PY" -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | tail -2
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n', torch.cuda.device_count())" 2>&1 | tail -3
"$PY" -c "import torch; [print(f'  gpu{i}:', torch.cuda.get_device_properties(i).name, round(torch.cuda.get_device_properties(i).total_memory/1024**3,1),'GB') for i in range(torch.cuda.device_count())]" 2>&1 | tail -5

echo "--- 内部镜像可达性 ---"
for u in http://mirrors.aliyun.com https://pypi.org http://__HPC_LOGIN_IP__ https://mirrors.tuna.tsinghua.edu.cn; do
    printf '%-42s ' "$u"
    curl -sS --max-time 6 -o /dev/null -w 'http=%{http_code}\n' "$u" 2>&1 | tail -1
done
echo "--- pip 配置 ---"
cat "$HOME/.pip/pip.conf" 2>/dev/null || echo "（无 ~/.pip/pip.conf）"
cat "$HOME/.condarc" 2>/dev/null | head -8 || echo "（无 ~/.condarc）"

echo "--- 现有 conda 环境 ---"
ls /storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/ 2>/dev/null

echo "===== ENV PROBE END $(date '+%F %T') ====="
