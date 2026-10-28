#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# 长安大学高性能计算平台 · RL_infra 作业脚本（JH Unischeduler）
#
# ⚠️ 提交前必须：① 贴出完整命令给用户核对 ② 等用户明确同意 ③ 再执行
#    （jsub 占用真实 GPU 配额且不可逆，hpc-jobs 技能铁律 0）
#
# 用法（二选一）：
#   jsub < scripts/hpc_submit.sh                                  # 用脚本内 #JSUB 选项
#   jsub -q nonmig -n 6 -gpgpu 1 < scripts/hpc_submit.sh          # 命令行覆盖（优先级更高）
#
# 本机调用（远程命令必须包在 bash -lc 里）：
#   ssh_exec(alias="hpc", command="bash -lc 'jsub ... '")
# ════════════════════════════════════════════════════════════════════

#JSUB -q gpu                 # gpu 队列 = MIG 切片（A800）；整卡请改 nonmig
#JSUB -n 6                   # 核数：串行为主，6 核够（env_count=2 + torch 线程）
#JSUB -gpgpu 1 mig=1         # MIG 10GB（本项目显存约 600MB，mig=1 足够且更好排队）
#JSUB -R "span[hosts=1]"     # 强制单节点，避免被拆到多节点
#JSUB -J RL_infra-madqn
#JSUB -o output.%J           # 产物落在 -cwd 目录下
#JSUB -e error.%J
#JSUB -cwd /storage/public/home/__HPC_ACCOUNT__/RL_infra/

PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python

export PYTHONIOENCODING=utf-8      # ★ 必须：集群 locale 可能是 ASCII，中文日志会 UnicodeEncodeError
export LANG=${LANG:-en_US.UTF-8}

echo "作业号: $JOB_ID  节点: $(hostname)  开始: $(date '+%F %T')"
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

cd /storage/public/home/__HPC_ACCOUNT__/RL_infra || exit 1

# 冒烟自检：作业一开始就证明骨架没坏（约 30 秒），失败立刻退出别浪费卡
$PY utils/smoke_test.py || { echo "❌ 冒烟自检失败，终止"; exit 2; }

# 正式训练（产物落 runs/<name>/，作业收尾时归档 output/error 一起回传）
$PY main.py --name hpc_run1 --episodes 2000 --parallel --device cuda

echo "结束: $(date '+%F %T')"
