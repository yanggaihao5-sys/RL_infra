> 🕒 v2 · 2026-10-24 12:43 · dsh-agent ｜ 变更：补 task_gen 实测：高价值时限下界 68 步 / 随机障碍 2-30 被 G5 拒 ｜ 上一版：v1 · 2026-10-24 12:41

# tools/rlinfra_ext —— 补 RL_infra 缺的四件（sweep / 早停 / ×N 重复 / 同预算对照）

> 为什么放在**本仓**而不是直接写进 `RL_infra`：那是我另一个仓，写入需要单独授权；
> 本目录是**可搬走**的一套脚本（零第三方依赖，只用 numpy + 标准库），确认后整目录拷进 `RL_infra/scripts/` 即可。
> 所有工具都能 **dry-run / 只读回测**，不占集群、不训练。

| 文件 | 干什么 | 依赖 |
|---|---|---|
| `sweep.py` | 同预算搜参：`rs`（随机）/ `bo`（numpy 手写 GP-EI）/ `warmstart`（先验均值来自 `db/runs.csv`） | numpy |
| `bench_sweep.py` | 同一合成目标上做 **多 seed × 固定预算** 的 best-so-far 对照，报均值±SD 与 MDE | numpy |
| `early_stop.py` | 从 `train.log` 的 `EP | ep: … cov: …%` 拟合**饱和模型**，判"还能涨多少"；`--backtest` 做截断回测 | numpy |
| `repeat.py` | 同配置 ×N 聚合：mean±SD + **MDE**，两组用 Welch 近似判可区分性 | numpy |
| `task_gen.py` | 任务族采样（顺序点 / 高价值时限 / 重复率 / 特殊格 / 随机障碍）+ **合法性守卫 G1~G5** | numpy |

## 🔴 实测结论（2026-10-24，都是**负/警示**结果，但这才是真材实料）

### 1. 搜索空间只有 12 个格点时，BO 与随机搜索**没有任何差别**
`bench_sweep.py --seeds 5 --budget 12`：三条 best-so-far 曲线在第 1~4 次评估内完全重合（`lr=5e-4, batch=128` 就是最优点）。
⇒ **在 2 维 × 12 格点上谈"算法贝叶斯"没有意义** —— 空间小到随机采样就能覆盖。

### 2. 空间放大到 80 格点后，BO/warmstart **仍然没赢随机搜索**
`bench_sweep.py --seeds 8 --budget 24 --wide`（lr 10 档 × batch 8 档）：

| 策略 | 达到 best-so-far ≥ 0.87 的最早评估次数（均值±SD，越小越好） |
|---|---|
| `rs`（随机） | **9.38 ± 8.31** |
| `bo` | 13.88 ± 10.20 |
| `warmstart` | 13.38 ± 11.17 |

观测噪声 SD=0.03 时，各策略最终值的 SD 高达 **0.06~0.11** ⇒ 差异全在噪声里。
🔴 **口径**：这是**本实现**（固定核宽 `ell=0.35`、未做输出标准化、未调 `xi`）的结果，**不代表 BO 的上限**；
但它精确复现了此前判断："**低 SNR 下，贝叶斯会被噪声淹没**" —— 要谈算法贝叶斯，先降噪、先把评估预算/任务数买上去。

### 3. 早停：线性外推完全失效；饱和模型**逐曲线才可用**
- 线性：用 SAC 前 2000 回合的斜率外推 3000 回合 ⇒ 预测再涨 **+126 pp**（上限只有 100%）⇒ 荒谬
- 饱和（`cov = A − B·e^(−ep/τ)`）截断回测：

| 臂 | 真实末 50 回合 | 渐近预测的平均绝对误差 | 判定 |
|---|---|---|---|
| `td3-s0`（早饱和） | 91.93% | **0.25 pp** | ✅ 可用来早停（500 回合起就准） |
| `sac-s0`（后段才爬升） | 85.45% | **15.50 pp** | ❌ 禁止用（前期误判为已饱和，锁死在 69%） |

⇒ **早停不是通用手段**：必须在**该算法自己的满预算曲线上先回测**，`MAE < 3pp` 才允许启用。
这条直接改写了"用早停把评估成本压到 1 h"的乐观假设 —— 它是**逐模型验证**后才成立的。

### 4. `repeat.py` 能立刻给"可辨识性"下限
例：opt9 的 `noms` vs `full`（各 2 seed，逐轮 JSON 均值）→ `Δ=+1.10pp, t=1.04` ⇒ **不可区分**，
与 `记录/五臂结果.md` 写的"测不出某分量承重"完全一致。

## 用法速查

```powershell
$env:PYTHONIOENCODING="utf-8"
python tools/rlinfra_ext/sweep.py --strategy warmstart --budget 12 --dry-run     # 逻辑验证
python tools/rlinfra_ext/bench_sweep.py --seeds 8 --budget 24 --wide             # 同预算对照
python tools/rlinfra_ext/early_stop.py <train.log> --backtest                    # 先回测，再决定能否早停
python tools/rlinfra_ext/repeat.py --group "A=run1,run2" --group "B=run3,run4"   # MDE + t
```

## 还没做（诚实清单）

- `sweep.py` 的**真跑**分支（`--rl-infra`）只写了调用骨架，**未实测**（要起子进程训练，须先问用户）
- GP 实现未调核：换成 Matérn + 输出标准化、或用 Optuna 复跑一次，才能对"BO 无效"下一般性结论
- `task_gen` 只做了"参数采样 + 守卫"，**任务奖励/观测的接线（bonus 分量、约束判定）还没写** —— 那是 envs 的事

## 5. `task_gen.py` 的实测（同样给的是"不行"的边界）

`--probe` 打印的地形事实（本仓 30×30 真实地图）：自由格 **893**、全部 8 邻达可、高价值格（p≥0.8）**136**、3 机
⇒ **高价值时限的可行下界 = ⌈136/3⌉ × 1.5 = 68 步**

- 🔴 **用户原提的"前 50 步遍历完高价值区"被 G4 明确拒绝**（`--tight` 实测 5/5 被拒）：50 < 68，
  而且 68 本身已经假设"每步都落在新的高价值格且零绕路"，实际只会更高。
- 🔴 **随机障碍模式下 30 个采样里 2 个被 G5 拒**：关键点/特殊格是按**原图**的可达集采的，换成随机障碍图后个别格子不可达
  ⇒ **生成器与守卫必须针对同一张图**，否则会造出"永远不可能完成"的任务。
- ⚠️ **`repeat_cap` 没有结构守卫**：重复率是**策略属性**不是地图属性。本仓 `O_J` 实测 **0.264~0.576**，
  30 个采样里有 **18 个** 落到 `<0.264` 区间 ⇒ 脚本会明确标注"**超出已观测可达区间，属无证据支持的要求**"（不拒，但要写进声明）。

```powershell
python tools/rlinfra_ext/task_gen.py --probe
python tools/rlinfra_ext/task_gen.py --sample 30 --seed 1 --json-out db/tasks_sample.json
python tools/rlinfra_ext/task_gen.py --sample 5 --tight      # 验证守卫真的会拒
```
