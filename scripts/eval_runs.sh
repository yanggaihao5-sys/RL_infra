#!/bin/bash
# 🕒 v1 · 2026-10-06 14:51 · dsh-agent ｜ 变更：v3：加 --maps N/--random-maps/--fixed-maps，打印覆盖标准差，sed 锚定全局平均避免贪婪误抓

# 🕒 v3 · 2026-10-06 · dsh-agent ｜ 变更：加 --maps N / --random-maps / --fixed-maps，并打印覆盖标准差（多地图评测，给出均值±std） ｜ 上一版：v2 · 2026-09-30 22:01

# ════════════════════════════════════════════════════════════════════
# 对已训练好的 run 逐个跑 **独立评测**（--mode test），汇总成一行 CSV。
#
# 为什么要单独一个作业：`main.py --mode train` **不做最终评测**（评测是另一条命令），
# 而 GPU 组每人只有 2 个 RUN 名额 ⇒ 多个评测必须**塞进一个作业**顺序跑。
#
# 用法：
#   bash eval_runs.sh [选项] <实验名...>
#
# 选项：
#   --maps N        每个模型评 N 张地图（默认取 config.eval_runs=5）。
#                   🔴 **N 就是方差的样本量**：n=1 出的排序不可信（本批实测 seed 效应 31.85pp）。
#   --random-maps   评测时每轮换一张地图（默认；给出真实方差。等价 CLI --eval-random-maps）
#   --fixed-maps    评测时固定同一张地图（跨实验严格对齐环境用；多轮结果会逐字相同、无方差）
#
# 例（多地图评测，主推）：
#   jsub -q gpu -n 4 -gpgpu '1 mig=1' -R "span[hosts=1]" -W 60 -m a800gpu01 \
#        -J rli-maps-g -o "output.%J" -e "error.%J" \
#        -cwd /storage/public/home/__HPC_ACCOUNT__/RL_infra/ \
#        bash /storage/public/home/__HPC_ACCOUNT__/RL_infra/scripts/eval_runs.sh --maps 20 --random-maps \
#             lr2k_5e4_s2024 lr2k_1e4_s2024 lr2k_1e3_s2024
#
# 产物：eval_summary.csv（含 coverage_std）+ 每个 run 的 runs/<name>/eval.json + output.<jobid>
# ════════════════════════════════════════════════════════════════════
set -u
PROJ=/storage/public/home/__HPC_ACCOUNT__/RL_infra
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python
export PYTHONIOENCODING=utf-8
export TMPDIR="$PROJ/.tmp"
mkdir -p "$TMPDIR"
cd "$PROJ" || exit 1

MAPS=""
MAP_FLAG=""
NAMES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --maps)        MAPS="${2:?--maps 需要数字}"; shift 2 ;;
        --random-maps) MAP_FLAG="--eval-random-maps"; shift ;;
        --fixed-maps)  MAP_FLAG="--eval-fixed-maps"; shift ;;
        -*)            echo "未知选项：$1"; exit 2 ;;
        *)             NAMES+=("$1"); shift ;;
    esac
done
[ ${#NAMES[@]} -gt 0 ] || { echo "用法：eval_runs.sh [--maps N] [--random-maps|--fixed-maps] <实验名...>"; exit 2; }

CSV="$PROJ/eval_summary.csv"
[ -f "$CSV" ] || echo "run_id,coverage,coverage_std,reward,success_rate,steps" > "$CSV"

echo "[$(date '+%F %T')] 评测开始 job=${JOB_ID:-local} node=$(hostname) maps=${MAPS:-config默认} flag=${MAP_FLAG:-config默认}"

for name in "${NAMES[@]}"; do
    if [ ! -f "runs/$name/madqn_best.pth" ]; then
        echo "[SKIP] $name：没有 runs/$name/madqn_best.pth"
        continue
    fi
    echo "--- [MANIFEST] EVAL $name ---"
    # shellcheck disable=SC2086
    out="$("$PY" main.py --mode test --name "$name" ${MAPS:+--eval-runs $MAPS} $MAP_FLAG 2>&1)"
    line="$(printf '%s\n' "$out" | grep '测试汇总' | tail -1)"
    echo "$line"
    # 🔴 sed 是贪婪的：必须锚在「全局平均」上，否则会把「覆盖标准差 X pp」也当成覆盖率值
    cov="$(printf '%s' "$line" | sed -n 's/.*全局平均覆盖率：\([0-9.]*\)%.*/\1/p')"
    rew="$(printf '%s' "$line" | sed -n 's/.*全局平均奖励：\([0-9.\-]*\).*/\1/p')"
    suc="$(printf '%s' "$line" | sed -n 's/.*成功率：\([0-9.]*\)%.*/\1/p')"
    stp="$(printf '%s' "$line" | sed -n 's/.*平均完成步数：\([0-9.]*\).*/\1/p')"
    sd="$(printf '%s' "$line" | sed -n 's/.*覆盖标准差 \([0-9.]*\)pp.*/\1/p')"
    if [ -n "$cov" ]; then
        echo "$name,$cov,$sd,$rew,$suc,$stp" >> "$CSV"
        # 把评测结果**写回实验目录**（F：省掉"评测完还要手工跑转换器"这一步）
        # log_to_metrics.py 读到 eval.json 就会自动带上 test/* 指标
        printf '{"coverage": %s, "coverage_std": %s, "reward": %s, "success_rate": %s, "steps": %s, "maps": %s, "random_maps": %s, "evaluated_at": "%s"}\n' \
            "$cov" "${sd:-0}" "${rew:-0}" "${suc:-0}" "${stp:-0}" \
            "${MAPS:-null}" "$([ "$MAP_FLAG" = "--eval-random-maps" ] && echo true || ([ "$MAP_FLAG" = "--eval-fixed-maps" ] && echo false || echo null))" \
            "$(date '+%F %T')" > "runs/$name/eval.json"
        echo "[MANIFEST] EVAL_DONE $name cov=${cov}% ± ${sd:-?}pp reward=$rew success=${suc}% steps=$stp"
    else
        echo "[MANIFEST] EVAL_FAIL $name（没解析到测试汇总行）"
    fi
done

echo "--- 汇总表 $CSV ---"
cat "$CSV"
echo "[$(date '+%F %T')] 评测结束"
