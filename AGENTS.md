> 🕒 v8 · 2026-10-22 19:11 · dsh-agent ｜ 变更：§1 参数行明确 COMMON_* / <链路>_* 两档前缀 ｜ 上一版：v7 · 2026-10-22 17:08

# AGENTS.md — RL_infra（项目级常驻准则）

> 全局通用约定见 `~/.dsh/AGENTS.md`（每轮已自动注入）。**本文件只放本项目专属的东西。**
> 大块参考性认知在 `docs/`，这里只留一行指路。

## 0. 这个项目是什么

**可迁移的 RL 训练基础设施（最小完备版）**：一套「算法包 + 日志契约 + 产物隔离 + 集群提交」
的骨架，配一个能跑通的示例任务（多智能体栅格覆盖搜索，MADQN 链路）。

**目的不是刷某个任务的指标，而是把「换任务只改一个文件、换实验只改一行、出问题能脚本溯源」这件事固化下来。**
血统：《工程架构与组织规范（可迁移参考）》，源自 `D:\__projects__\UAV-area-search-DQN\docs\工程架构与组织规范.md`。

## 1. 四条隔离（本项目所有规矩的总纲）

| 维度 | 手段 | 落地位置 |
|---|---|---|
| **代码** | 一算法一包，包内固定六件套 | `algos/<algo>/` |
| **产物** | 唯一命名源 `name` + 两级目录封顶 | `config.py: name` → `runs/<name>/` |
| **参数** | 文件头一处集中 + **前缀即作用域**（`COMMON_*`=全链路共享，`<链路>_*`=仅该链路）+ 启动校验 | `config.py` |
| **日志** | 固定行前缀契约，业务代码不手拼字符串 | `rlinfra/logger.py` |

## 2. 目录速查

```
RL_infra/
├── main.py            ← 唯一入口（test_mode × MODE 二维 dispatch；主入口速查表在文件头）
├── config.py          ← 唯一配置源（每行带注释 + validate() + 派生量自证）
├── rlinfra/           ← 与算法无关的公共基座（logger / paths / metrics / registry）
├── algos/madqn/       ← 一条链路 = 一个包 = 六件套（envs/networks/agent/replay_buffer/parallel_env/trainer）
├── algos/pursuit/     ← 第二条链路（迁移演练产物：覆盖→追逃；只有 pursuit_envs.py 是实质改写）
├── utils/             ← 横切工具（analyze_training_log 诊断 / smoke_test 冒烟 / plot_metrics 画图 /
│                          log_to_metrics 转观测契约 / sync_hpc_runs 拉高算 run 回本地）
├── scripts/           ← 集群提交与批实验（run_batch.sh 批实验骨架 / eval_runs.sh 多 run 评测 /
│                          probe*.sh 探针 / lora_smoke.py 微调冒烟）
├── runs/<name>/       ← ★实验产物（代码与产物物理隔离；本目录不进 git；eval.json 存评测结果）
├── runs/obs/          ← 观测平台数据源（由 log_to_metrics / sync_hpc_runs 生成）
└── longago/           ← 历史产物归档（永不删除、也不参与当前工作）
```

## 3. 怎么跑（最常用七条）

```powershell
$env:PYTHONIOENCODING="utf-8"          # 跑 Python 前必设（全局规则）
python utils/smoke_test.py             # ① 骨架自检（9 项，含日志契约 ↔ 诊断脚本互证、删文件边界）
python main.py --episodes 300          # ② 训练（产物落 runs/<config.name>/）
python main.py --mode test --name <名> # ③ ★评测（**与训练是两条命令**；产物含 runs/<名>/eval.json）
python utils/analyze_training_log.py runs/<name>/train.log   # ④ 日志诊断（几 MB → 几十行报告）
python utils/plot_metrics.py runs/<name>/train.log           # ⑤ 画训练曲线
python utils/sync_hpc_runs.py          # ⑥ 拉高算全部 run 回本地并转观测契约（一条命令）
python main.py --print-config          # ⑦ 只看最终配置
python utils/bench_report.py --all     # ⑧ 重生成基准报告 + 图（docs/BENCHMARK.md；**别手改该 md**）
python utils/check_boundaries.py       # ⑨ 库边界自检（6 项；pip 装过可加 --installed）
```

> 一行跑通训练+评测：`python main.py --episodes 60 --name demo --then-test`（本机 CPU 约 100 s）。

**观测平台**：`cd ../UAV-area-search-crosscheck && python -m web.server --port 8077 --runs-dir <本仓库>/runs/obs`
（本地 127.0.0.1:8077；数据源只读 `runs/obs/`，新 run 同步后刷新即出现）

## 3.5 docs/ 索引（哪本讲什么，**别整读**）

| 文件 | 讲什么 | 什么时候读 |
|---|---|---|
| `函数级教程.md` | 逐函数走读（47KB，最长） | 要改某个函数前 |
| `学习路线.md` | 这套 infra 该怎么读、怎么改 | 第一次上手 |
| `架构决策.md` | 为什么这么搭（含代价） | 想推翻某个设计前 |
| `实验登记表.md` | ★每个实验的「唯一变量 / 证据路径 / 结论」 | **动手做实验前必看**，避免重复跑过的 |
| `微调与Agent能力-学习手册.md` | 微调→agent 七级阶梯 + 模型血缘 + 7B 取舍 | 走微调线时 |
| `BENCHMARK.md` | ★可对外证据：10 checkpoint × 20 图评测 + 基线 + 3 张图（`utils/bench_report.py` 生成，**别手改**） | 要报数字 / 复现评测前 |
| `库边界.md` | `rlinfra/`（库，零依赖）vs `algos/`（应用）vs 应用层；7 项自动检查 | 改接口 / 加算法前 |
| `bench/README.md` | 基准报告原始证据目录（runs.csv / results/*.json / baselines.json） | 收录新 run 时 |
| `迁移演练.md` | ★「换任务只改一个文件」的实测账单（覆盖→追逃）+ 撞出的 4 个设计缺陷 | 要动接口 / 评估"可迁移"说法前 |
| `学习进度看板.md` | 自动生成的学习进度 | 想知道学到哪 |
| `AgentHarness-战果与工具链.md` · `H2_实验报告.md` · `实验_上下文注入AB_设计.md` | Agent Harness 装置 + 上下文注入 A/B | 走 agent 线时 |
| `论文写作/README.md` | 论文写作目录说明 | 要写论文时 |

## 4. 🔴 铁律（动手前必看）

1. **改日志格式 → 必须改 `rlinfra/logger.py`，且同一次改动内跑 `utils/smoke_test.py` 第⑦项。**
   业务代码里**不许**出现 `print("EP | ...")` 这类手拼行 —— 格式一散，诊断脚本就瞎了。
2. **奖励分量必须在 `madqn_envs.py` 的 `step()` 内、行为发生的同一时刻拆分**，随 `info["reward_components"]` 回传；
   训练循环**只累加不重算**。这是唯一能诊断「奖励函数写错」的手段。
3. **加新算法 = `python scripts/new_chain.py --src algos/madqn --dst algos/<新名>`（复制 + 机械改名 + 记账）
   → 重写 `<新名>_envs.py` 一个文件 → 在 `rlinfra/registry.py` 登记一行（`train_fn` **和** `eval_fn` 都要填）。**
   **禁止**用「类重绑定」搞隐式切换（UAV 项目真实事故：import 成功、跑完不报错、却跑错链路）。
   ⚠️ 2026-10-17 迁移演练实测：`eval_fn` 曾被漏登记，导致 `--mode test` **静默**评测 madqn 的模型（见 `docs/迁移演练.md` §5 缺陷 #1）。
4. **函数内禁止硬编码路径与魔法数字**；路径一律由 `rlinfra/paths.py` 派生，数字一律来自 `config.py`。
5. **`obs_dim` 等派生量改参数后必须 `cfg._derive()`**（`main.py` 的 CLI 路径已自动做；写脚本调 Config 时要自己调）。
6. **受限环境留降级路径**：`mp` 走命名管道，沙箱里会被禁 → `parallel=False`（`SyncEnvPool`）必须始终可用。
   冒烟自检**永远**跑单进程分支。
7. **任何「删文件」逻辑都必须划定严格边界 + 单独的冒烟检查**（见 `utils/smoke_test.py` 第⑨项）：
   归档/清理只在自己实验目录内、只匹配精确文件名模式；**序号必须单调递增（max+1），不许复用空号**
   —— 复用空号会让「刚存的 checkpoint」当场被清理逻辑删掉（本项目实测踩到）。

## 5. 🖥 高算平台（HPC）—— 长安大学高性能计算平台

**能直接调**：`ssh_exec(alias="hpc", ...)`（已配密钥免密），完整协议见 `hpc-jobs` 技能。

### 🔴 提交作业 / 上传文件前必须先问用户
`jsub` 会占用真实 GPU 配额、`ssh_upload` 会覆盖服务器文件 —— **都属于破坏性操作**：
先贴出**完整命令原文** + 列出会动的远程文件 → **等用户明确同意** → 才执行。
（只读操作 `jjobs` / `jhosts` / `jctrl peek` / `ssh_download` / 远程 `ls|tail|grep` 不受此限。）

### 本项目提交模板（gpu 队列 MIG 切片，显存约 600MB 用 mig=1 足够且更好排）

```bash
# 1) 传代码（目录不能整体传，逐个文件）
#    ssh_upload(alias="hpc", localPath="__REPO_DIR__\\...", remotePath="/storage/public/home/__HPC_ACCOUNT__/RL_infra/...")
# 2) 提交
jsub -q gpu -n 6 -gpgpu '1 mig=1' -R "span[hosts=1]" -J RL_infra-madqn \
  -o output.%J -e error.%J -cwd /storage/public/home/__HPC_ACCOUNT__/RL_infra/ \
  /storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python main.py --name hpc_run1 --episodes 2000

# nonmig 整卡队列则写 -gpgpu 1（不带 mig）
```

- ⚠️ **命令必须包在 `bash -lc '...'` 里**（否则 `jhds.conf Not Found`）。
- ⚠️ 平台是校园网地址，换网络后先 `Test-NetConnection __HPC_LOGIN_IP__ -Port 22`。
- ⚠️ 作业收尾（认实验名 → 归档 output/error → 回传本地）见 `hpc-jobs` 技能「作业收尾」三步。

## 6. 已知代价（诚实记录，别当 bug 修）

- **rollout 线程用独立权重副本**：策略有滞后（同步粒度 = 每轮学习后一次），换来的是无数据竞争。
- **`runs/` 不进 git**：产物只留本地 + 平台归档；要对比就 diff 两个 `runs/<name>/` 目录。
- **镜像包靠复制同步**：目前只有一个算法包，还没有共享基类；复制第二个包时再抽基类（先照抄，后重构）。
- **评估地图与成功判据（2026-09-30 改）**：`eval_random_maps` **默认 True**（每轮换图 ⇒ 有真实方差）；
  要跨实验严格对齐环境用 `--eval-fixed-maps`（此时多轮结果逐字相同，是预期而非 bug）。
  评测成功判据 `eval_coverage_threshold`（默认 0.75）与训练终止阈值 `coverage_threshold`（0.90）**解耦**：
  0.90 高于测试图可达上限 ⇒ 成功率恒 0%、步数恒打满、指标无分辨力；解耦只动评测判据，**不改训练动力学，历史 run 仍可比**。
- **训练与评测是两条命令**：`--mode train` **不做**最终评测；评测必须 `--mode test --name <名>`，结果写 `runs/<名>/eval.json`
  （2026-10-17 起**本地也写**；此前只有高算 `scripts/eval_runs.sh` 会写）。
- **成功判据是链私有的（2026-10-17 起）**：覆盖域 = `覆盖率 ≥ eval_coverage_threshold`、追逃域 = `捕获`。
  两者都写进汇总行的 `成功判据 …` 与 `eval.json` 的 `success_rule` —— **不要**再假设所有链共用 0.75 这个数。
- **选优判据也是链私有的**：`MADQN_best_metric="coverage"` / `PURSUIT_best_metric="capture_steps"`（`probe_key` 白名单校验）。
  新增链路必须显式选一个 —— 否则会选出"扫得多"而不是"抓得快"的模型（实测差 **25pp**，见 `docs/迁移演练.md` §5.1）。
- **MIG 档位按实测选，不按猜测**：本项目 MADQN 约 600MB → `mig=1`；DeepSeek-R1-Distill-Qwen-7B LoRA 实测峰值 **17.92GB** → `mig=2` 即可（**不要**默认 `mig=4`，大切片极难排队：实测同一作业 PEND 106258s / RUN 81s）。
- **评估图用英文标签**：中文字体缺失时会变方框，`plot_metrics.py` 可用 `--lang cn` 自行覆盖。

## 7. 省 token 与收尾（继承全局规则）

- 大日志**不要人读**：`python utils/analyze_training_log.py <log>` 出结构化报告。
- 改完代码要摘要 → `change-submitter` 技能；看训练结果 → `output` 技能；查维度/算力 → `code-analyzer` 技能。
- **每轮回复末尾**写一行 `🔧 本轮：...` 声明用了哪些技能/插件/自建工具（全局硬规则）。
- 新踩到本项目专属的坑 → 追加进 `.dsh/skills/pitfall-guard-rlinfra/SKILL.md`（**不许**叫 `pitfall-guard`，同名会遮蔽用户级）。
