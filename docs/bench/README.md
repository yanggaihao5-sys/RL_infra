> 🕒 v1 · 2026-10-17 16:14 · dsh-agent ｜ 变更：新增：基准报告原始证据目录说明与收录流程

# docs/bench —— 基准报告的原始证据目录

`docs/BENCHMARK.md` 里**每一个数字**都由 `utils/bench_report.py` 从这里算出来，没有手抄。
换数据只改本目录 + 跑一条命令，报告和图自动重出。

## 文件

| 文件 | 是什么 | 谁写的 |
|---|---|---|
| `runs.csv` | 实验元数据：`run` / 回合数 / lr / seed / **单图覆盖率** | 手工维护（每收录一个 run 补一行） |
| `results/<run>.json` | 该 run 的 **20 图评测原文**（`runs/<run>/eval.json` 的拷贝） | `scripts/eval_runs.sh` + `scripts/aggregate_eval.sh` 产出 |
| `baselines.json` | 随机策略 / 贪心最近未访策略的 20 图实测（含逐图数字） | `python utils/bench_report.py --baselines` |

## 冻结的评测协议（改 = 数字不可比）

栅格 20×20 · 2 智能体 · 12 障碍 · 局部视野 p=2 · 9 动作 · 每回合换图（env seed = 0…19）·
200 步上限 · 成功判据 0.75（与训练的 0.90 解耦）· 贪心策略 · 跨 20 图取 mean ± pstdev。

## 收录一个新 run

```bash
# 1) 高算上：训练 → 20 图评测 → 从逐图 json 聚合 std
bash scripts/eval_runs.sh --maps 20 --random-maps <run>
bash scripts/aggregate_eval.sh 20 <run>
# 2) 拉回本地，把 eval.json 拷进 docs/bench/results/<run>.json
# 3) docs/bench/runs.csv 补一行（回合数 / lr / seed / 单图覆盖率）
# 4) 重出报告
python utils/bench_report.py --all
```

## 注意

- `results/*.json` 里只有**聚合值**（mean/std/min/max），逐图原始覆盖在远端 `runs/<run>/test_path_plots_en/*.json`。
  ⇒ 同一批 20 张图**本来允许配对比较**，现在只能做非配对 SEM 估计（报告 §6 已登记为短板）。
- 这里**不放 checkpoint**（`*.pth` 被 `.gitignore` 挡住）：模型在高算归档，本地只留评测数字。
- 基线是**启发式**，不是文献方法；且贪心基线用**全局已访图**（特权信息）—— 读报告时必须带着这条看。
