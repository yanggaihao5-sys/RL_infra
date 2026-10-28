#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# RL_infra · 高算平台「一个作业跑一批实验」骨架
#
# 为什么需要它（2026-09-30）：
#   高算 GPU 组（u8）每人只有 **2 个 RUN 作业名额**。把 10 个配置拆成 10 个作业
#   ⇒ 8 个永远排不上队，而且长跑作业一被时限杀掉，整批前功尽弃。
#   正确做法：**1 个作业 = 2 张整卡 = 一批实验**，配置表驱动 + 已完成跳过。
#
# 用法（远程命令必须包在 bash -lc 里，见 hpc-jobs 铁律 1）：
#   jsub -q gpu -n 16 -gpgpu 2 -R "span[hosts=1]" -W 1440 -J rli-madqn-g \
#        -o output.%J -e error.%J -cwd /storage/public/home/__HPC_ACCOUNT__/RL_infra/ \
#        bash /storage/public/home/__HPC_ACCOUNT__/RL_infra/scripts/run_batch.sh [选项]
#
# 选项：
#   --list  <文件>   实验清单，每行一个实验，`#` 开头为注释
#                    每行 = `实验名 | 额外参数...`（实验名决定 runs/<name>/ 目录）
#   --mode  seq|par  seq=单卡串行（默认，稳）；par=双卡并发（快，占满两张卡）
#   --dry-run        只打印将要执行什么，不真跑（提交前自查用）
#
# 断点续跑：某实验目录下有 `runs/<name>/.done` 就跳过 —— 作业被杀后重提不重跑。
#            正常跑完也会写 `.done`（里面是完成时间 + 退出码）。
# 原子产物：checkpoint 由 agent/buffer 的 save() 原子落盘（tmp + os.replace），
#            所以作业**运行中**随时 scp 拉 `runs/<name>/` 都是完整文件。
# ════════════════════════════════════════════════════════════════════
set -u

PROJ="${RLI_PROJ:-/storage/public/home/__HPC_ACCOUNT__/RL_infra}"
PY="${RLI_PY:-/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python}"
LIST="$PROJ/scripts/batch_list.txt"
MODE="seq"
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --list)    LIST="$2"; shift 2 ;;
        --mode)    MODE="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
done

# --- 环境（集群 locale 可能是 ASCII，中文日志会 UnicodeEncodeError）---
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-en_US.UTF-8}"
export TMPDIR="${PROJ}/.tmp"                    # 别用节点 /tmp：配额小、跨节点不可见
mkdir -p "$TMPDIR"

cd "$PROJ" || { echo "[FATAL] 工作目录不存在：$PROJ"; exit 1; }
[ -f "$LIST" ] || { echo "[FATAL] 实验清单不存在：$LIST"; exit 1; }

JOB="${JOB_ID:-local}"
say() { echo "[$(date '+%F %T')] $*"; }

say "===== 批实验开始 job=$JOB node=$(hostname) mode=$MODE dry=$DRY ====="
say "清单：$LIST"
command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=index,name,memory.total --format=csv \
    || say "（无 nvidia-smi：登录节点或 CPU 队列，忽略）"

# ---------------- 待跑清单（跳过已完成的） ----------------
TODO=(); SKIPPED=0
while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    name="${line%%|*}"
    name="$(echo "$name" | tr -d '[:space:]')"     # 实验名不许带空白（它同时是目录名）
    [ -n "$name" ] || continue
    if [ -f "runs/$name/.done" ]; then
        say "[SKIP] $name（已完成：$(cat "runs/$name/.done" 2>/dev/null)）"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    TODO+=("$name")
done < "$LIST"
say "待跑 ${#TODO[@]} 个，跳过 $SKIPPED 个"

# ---------------- 执行 ----------------
run_one() {   # $1 = GPU 序号（钉死 CUDA_VISIBLE_DEVICES）; $2 = 实验名
    gpu="$1"; name="$2"
    extra="$(grep -m1 "^[[:space:]]*${name}[[:space:]]*|" "$LIST" | cut -d'|' -f2-)"
    say "[MANIFEST] START name=$name gpu=$gpu job=$JOB"
    t0=$(date +%s)
    if [ "$DRY" = "1" ]; then
        say "[DRY] CUDA_VISIBLE_DEVICES=$gpu $PY main.py --name $name --device cuda $extra"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" $PY main.py --name "$name" --device cuda $extra
    rc=$?
    t1=$(date +%s)
    if [ "$rc" = "0" ]; then
        # 只有成功才写 .done：失败/被杀 → 下次重提会重跑这个实验
        printf 'finished=%s rc=0 gpu=%s seconds=%s\n' "$(date '+%F %T')" "$gpu" "$((t1 - t0))" \
            > "runs/$name/.done"
        say "[MANIFEST] DONE name=$name gpu=$gpu rc=0 elapsed=$((t1 - t0))s"
    else
        say "[MANIFEST] FAIL name=$name gpu=$gpu rc=$rc elapsed=$((t1 - t0))s"
    fi
    return 0
}

if [ "$MODE" = "par" ]; then
    # 双卡并发：固定 2 个槽位。**不用 `wait -n`**（部分 bash 4.x 不支持，会直接报错），
    # 用「每轮补满槽位 + 全部 wait」——语义等价，兼容性更好。
    i=0
    while [ $i -lt ${#TODO[@]} ]; do
        run_one $((i % 2)) "${TODO[$i]}" &
        i=$((i + 1))
        if [ $((i % 2)) -eq 0 ]; then wait; fi
    done
    wait
else
    i=0
    while [ $i -lt ${#TODO[@]} ]; do
        run_one 0 "${TODO[$i]}"
        i=$((i + 1))
    done
fi

say "===== 批实验结束 job=$JOB ====="
