> 🕒 v4 · 2026-10-17 16:43 · dsh-agent ｜ 变更：迁移账单：换任务一项由「未取证」升级为实测（追逃演练）+ 链路数 1→2 ｜ 上一版：v3 · 2026-10-17 16:22

# RL_infra

[![ci](https://github.com/yanggaihao5-sys/RL_infra/actions/workflows/ci.yml/badge.svg)](https://github.com/yanggaihao5-sys/RL_infra/actions/workflows/ci.yml)

**一套把「换任务只改一个文件、换实验只改一行、出问题能脚本溯源」固化下来的 RL 训练基础设施。**
自带一条能跑通的示例链路：多智能体栅格覆盖搜索（MADQN / CTDE / 参数共享，2 智能体 · 20×20）。

| 状态 | |
|---|---|
| 代码 | ✅ 训练 → 评测 → 集群提交全链路跑通（Python 3.12 / torch 2.8 CPU 实测） |
| 评测 | ✅ 10 个 checkpoint × 20 张随机地图的可复现基准：[`docs/BENCHMARK.md`](docs/BENCHMARK.md) |
| 自动化 | ✅ **GitHub Actions CI 真跑并全绿**：静态（零依赖）/ pytest（py3.9 + 3.12）/ 端到端 30 回合 / 打包 + 干净 venv 安装；本地另可跑 `utils/smoke_test.py`（9 项）+ `utils/check_boundaries.py`（8 项） |
| 范围 | ⚠️ 两条链路（`madqn` 覆盖 / `pursuit` 追逃 —— 后者是迁移演练产物，见 [`docs/迁移演练.md`](docs/迁移演练.md)），仍是同一个「栅格多智能体」家族 |
| 交付物 | **可搬的骨架**（目录 / 产物 / 日志 / 注册表 / 提交），不是某个任务的模型。示例任务只用来证明装置跑得通 |

---

## 它解决什么

RL 实验烂掉通常不是因为算法错，而是因为下面三件事没有物理隔离：

| 痛点 | 本项目的手段 | 落地位置 |
|---|---|---|
| 产物和代码搅在一起，跑完找不到哪个数字是哪次实验的 | **唯一命名源** `config.name` → `runs/<name>/`，两级目录封顶；`runs/` 不进 git | `rlinfra/paths.py` |
| 日志格式随手改，事后 grep 不出来、画不出图 | **固定行前缀契约**，业务代码不许手拼日志行 | `rlinfra/logger.py` |
| 换个算法就靠「类重新赋值」偷懒，import 成功却跑错链路 | **显式工厂注册表**（`test_mode` → 一个 `AlgoSpec`），取不到就当场报错 | `rlinfra/registry.py` |

> 前两条是工程卫生；第三条来自真实事故（UAV 项目：`MADroneSearchEnv = ExpertMADroneSearchEnv` 之后
> import 正常、跑完不报错、结果全错）。`check_boundaries.py` 第 ⑥ 项就是防止它回来。

---

## 能搬什么、搬不动什么（迁移账单）

**这个仓库卖的是装置，不是成绩。** 装置能不能用，只能靠「换项目时要动多少」来证明，不能靠自己某个任务跑得多好，
更不能靠搬别人项目的结果来证明。账单如下——**没做到的也写进来**：

| 想搬的东西 | 证据 | 强度 |
|---|---|---|
| 目录骨架 / 产物隔离 / 日志契约 / 注册表（四件） | `docs/架构决策.md` §2「规范 §9 迁移清单 —— 逐条落地对照」（上游 = `UAV-area-search-DQN/docs/工程架构与组织规范.md`），逐条 `[x]` 落地 | ✅ 发生过一次（同作者、同领域；**不是**独立第三方搬运） |
| 产物 → 观测契约（`runs/<name>/` 与 `runs/harness/obs/` 同构） | `docs/AgentHarness-战果与工具链.md:182` 逐字：「harness 产物落在 `runs/harness/obs/<run_id>/`，与观测平台契约一致」——同一套观测平台直接读**非 RL** 实验 | ✅ 发生过一次（第二个任务类型，但仍在同一仓库） |
| 「换任务只改 envs 一个文件」 | ✅ **已实测 2026-10-17**：`algos/madqn/` → `algos/pursuit/`（覆盖→追逃），复制 + 重写 **1 个** envs（302 行）→ **2 分 18 秒**后冒烟通过，训练/评测/观测三条链路零改动接住。账单与撞出的 4 个设计缺陷 → [`docs/迁移演练.md`](docs/迁移演练.md) | ✅ 一次（同域内最小切换：仍是 2 体 / 20×20 / 9 动作 / 目标全可见） |
| 「换实验只改 `config.py` 的 `name` 一行」 | `rlinfra/paths.py` 唯一命名源 | ⚠️ 部分取证：目录随 `name` 变；没有「同代码两实验」的对照日志 |
| 模型数字（覆盖率 / 成功率 / std） | [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | ❌ **搬不动**：口径绑死 20×20 / 2 智能体 / 12 障碍 / 判据 0.75 / seed 0–19。换配置即不可比，搬到别的项目直接失效 |

> 缺口一句话：**「换任务只改一处」现在是实测结论（一次，同域）；「换项目」还不是。**
> 演练量到的真实代价不在基座（`logger.py` / `paths.py` **0 行改动**），而在**领域知识**：
> 奖励设计（探索奖励 9 倍于接近奖励 ⇒ 学到的策略 70% 捕获 vs 直线追击 100%）和选优判据（写死 coverage，选错了模型）。
> 附带的硬收获：注册表原来**只登记 train、没登记 eval** —— `--mode test` 会静默跑错链路，这次已修。

---

## 30 秒跑起来

前置：Python ≥ 3.9（实测 3.12）、能装 numpy / torch / matplotlib。

```powershell
$env:PYTHONIOENCODING="utf-8"                 # Windows 上跑 Python 前必设
pip install -e ".[madqn]"                     # 基座本身零第三方依赖；[madqn] 才装 torch
python main.py --episodes 60 --name demo --then-test   # ★ 一行命令：训练 + 评测
```

**预期输出**（本机 CPU，2026-10-17 实测，全程 **约 100 秒**）：

```
ℹ️ 阶段 test 完成 → {'coverage': 68.92, 'coverage_std': 5.81, 'success_rate': 40.0, 'steps': 181.0, 'runs': 5, ...}
```

产物全部落 `runs/demo/`（代码与产物物理隔离）：

```
runs/demo/
├── train.log                      全量日志（诊断脚本的输入）
├── madqn_best.pth                 最优模型（覆盖保存）
├── madqn_final_1.pth              周期模型（序号递增，不覆盖）
├── madqn_training_metrics_en.png  训练曲线
└── test_path_plots_en/            逐图轨迹 + 指标 json
```

后续三条常用命令（都是独立的，不需要连跑）：

```powershell
python utils/analyze_training_log.py runs/demo/train.log   # 几 MB 日志 → 几十行结构化报告
python utils/plot_metrics.py runs/demo/train.log           # 训练曲线
python utils/smoke_test.py                                 # 骨架自检 9 项
```

---

## CI：四道门禁（`.github/workflows/ci.yml`）

| job | 装什么 | 门禁 |
|---|---|---|
| 静态 | 什么都不装 | 语法编译 + 8 项库边界（`rlinfra` 零第三方、单向依赖、注册表懒加载） |
| 单测 | numpy + CPU torch + pytest | 23 项 pytest，Python **3.9 / 3.12 矩阵** |
| 端到端 | 再加 matplotlib | 冒烟 9 项 → **30 回合**训练+评测 → 日志诊断 → 出图 → 转观测契约 → `utils/verify_run.py` 6 项数字自洽 |
| 打包 | build | wheel 构建 → 内容边界（只含库+算法包）→ 干净 venv 安装 → 任意目录 import |

本地等价复现（三条命令，2026-10-28 实测 51.7s）：

```powershell
python -m pytest -q tests
python main.py --episodes 30 --name ci_smoke --then-test
python utils/verify_run.py --run runs/ci_smoke --obs runs/obs/ci_smoke
```

🔴 `--episodes` 必须 **> 25**（`COMMON_eval_interval`）：否则一次贪心评估都不触发、存不出 best，
评测会**静默 skipped**，CI 仍然“绿”。端到端 job 因此跑 30 回合，并用 `verify_run.py` 明确拒绝 skipped。
真跑证据、修掉的坑与未覆盖缺口见 [`docs/CI_交付报告.md`](docs/CI_交付报告.md)。

---

## 装置自证：示例任务跑通、并留下可复查的数字（**不是本项目的主张**）

这一节的作用只有两个：① 证明「训练 → 产生日志 → 独立评测 → 多图聚合 → 出图」这条链真的能跑；
② 给「换实验只改一行」提供一个具体例子。**模型效果不是本仓库的研究对象**，下面的数字**不能**搬到别的项目，
也不该被当成这套 infra 的价值证明（口径绑死在 20×20 / 2 智能体 / 12 障碍 / 判据 0.75 / seed 0–19）。

评测协议：20 张随机地图（env seed 0…19）· 每回合 200 步 · 成功判据覆盖率 ≥ 0.75 · 贪心策略 · mean ± 地图间 std。
数字来源与复现步骤见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

| 策略 | 20 图均值 | std | 成功率（≤200 步达 0.75） | 信息量 |
|---|---|---|---|---|
| **MADQN**（最优 checkpoint） | **75.08%** | ±12.30 | 0% | 局部 5×5 视野 + 覆盖率 + 位置 |
| 贪心「最近未访格」启发式 | **74.77%** | ±1.67 | **95%** | **全局已访图**（特权信息） |
| 均匀随机策略 | 34.63% | ±6.59 | 0% | 无 |

⚠️ 三条口径，别越过：

1. **MADQN 在覆盖率上与贪心启发式打平（+0.31pp），在稳定性与成功率上明显落后** —— 这是本项目最诚实的结论，不打算包装成胜利；
   公平性也不对等：启发式读全局已访图，Q 网络只有 5×5 局部视野。给启发式加局部视野后才算同条件（**未做**）。
2. **单张固定图上的数字不能当泛化**：同一个模型单图 88.40%、20 图均值 75.08%；10 个 run 里单图与 20 图的偏差最大 **13.32pp**。
3. 模型间差异里**含训练随机性**（每个配置只训 1 个 seed），所以只能当工作假设，不能当结论。

---

## 接口与边界（这个仓库最该被看的部分）

**`rlinfra/` 是库（只用标准库），`algos/` 是应用，依赖箭头只能向下**；`main.py` / `config.py` 是应用层，不随包分发。

```python
import rlinfra                                   # 库：日志 / 路径 / 计量 / 注册表，零第三方依赖
from rlinfra.registry import get_algo, register  # 换链路 = 登记一行，不做类重绑定
spec = get_algo("madqn")                         # 懒加载：包与类名对不上 → 立刻 AttributeError
```

```powershell
python utils/check_boundaries.py --installed      # 7 项边界检查：基座零依赖 / 不反向依赖 / 注册表显式工厂 …
```

- 完整规则与自动化检查：**[`docs/库边界.md`](docs/库边界.md)**
- 加一条新链路：`cp -r algos/madqn algos/<新名>` → 改前缀 → `rlinfra/registry.py` 加一行（5 步见该文档 §4）

---

## 骨架长什么样

```
main.py        唯一入口：test_mode（链路）× MODE（train/test/smoke）二维 dispatch
config.py      唯一配置源：超参一处定义 + 启动校验 + 派生量自证
rlinfra/       与算法无关的公共基座（logger / paths / metrics / registry）
algos/madqn/   一条链路 = 一个包 = 固定六件套
  madqn_envs.py           环境（唯一放「领域知识」的地方）
  madqn_networks.py       Q 网络结构
  madqn_agent.py          智能体与训练逻辑（act / learn / 存取）
  madqn_replay_buffer.py  经验回放
  madqn_parallel_env.py   环境池（多进程 + 单进程降级路径）
  madqn_trainer.py        后台 rollout 线程 + 训练主循环 + 贪心评估选 best + 评估
algos/pursuit/ 同构的第二条链路（迁移演练产物：覆盖 → 追逃；只有 pursuit_envs.py 是实质改写）
utils/         横切工具（日志诊断 / 冒烟自检 / 画图 / 基准报告 / 边界检查 / 拉高算产物）
scripts/       集群提交与批实验（高算平台；应用层脚本，路径按机器改）
docs/          设计决策 / 实验登记表 / 基准报告 / 函数级教程
```

## 换东西只动一处

| 想干什么 | 改哪里 |
|---|---|
| 换实验（新名字、新产物目录） | `config.py` 的 `name` 一行 |
| 换任务（巡检/追逃/编队） | `algos/madqn/madqn_envs.py`（**只改这一个文件**） |
| 换网络 / 回放策略 | `algos/madqn/madqn_networks.py` · `madqn_replay_buffer.py` |
| 加一条新链路 | 复制 `algos/madqn/` → 改前缀 → `rlinfra/registry.py` 登记一行 |
| 改日志格式 | `rlinfra/logger.py`（改完**必须**跑 `utils/smoke_test.py` 第⑦项） |
| 调奖励 | `config.py` 的 `R_*` 段（一处定义，不散在环境里） |

## 上高算平台

见 [`AGENTS.md` §5](AGENTS.md) 与 `scripts/hpc_submit.sh`。一句话：
`jsub -q gpu -n 6 -gpgpu '1 mig=1' -R "span[hosts=1]" ...` —— **提交前必须先确认**（占真实配额）。

## 文档

| 文档 | 什么时候看 |
|---|---|
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | 想知道这些数字怎么来的 / 要复现评测 |
| [`docs/库边界.md`](docs/库边界.md) | 要改接口、加算法、或评估这层抽象 |
| [`docs/实验登记表.md`](docs/实验登记表.md) | 动手做实验前（避免重复跑已经跑过的） |
| [`docs/架构决策.md`](docs/架构决策.md) | 想推翻某个设计前（含诚实的代价） |
| [`docs/函数级教程.md`](docs/函数级教程.md) | 要改某个具体函数前 |
| [`docs/学习路线.md`](docs/学习路线.md) | 第一次读这套代码 |
