> 🕒 v2 · 2026-10-25 17:03 · dsh-agent ｜ 变更：填上 A40 整卡的 B2 三个数 + 预热坑 ｜ 上一版：v1 · 2026-10-24 14:05

# traintest/ —— B 阶段 · 训推一体最小系统

> 手册：`UAV-area-search-DQN-opt/docs/DS准备/01_手册_推理内核到训推一体.md` §3
> 判据：`03_排期与资源.md` §二（B1 10/29 · B2 10/30 · 归因+回传 10/31 · **11/1 前清零**）

## 是什么

| 文件 | 作用 |
|---|---|
| `rollout.py` | `RolloutEngine` 协议（`generate` / `load_weights`）+ `TorchRolloutEngine` + `WeightSync` |
| `pipeline.py` | `SyncPipeline` / `OverlapPipeline`（常驻 worker 线程）+ 三段计时 |
| `bench_overlap.py` | B2 基准：加速比 ± SD + MDE + 瓶颈归因 + `nvidia-smi` 采样 |

`kernel/` 与这里分开是有意的：`kernel/` 是 A 阶段**零第三方依赖**纯库
（`utils/check_boundaries.py` ⑧ 会检查），这里必须 import torch。

## 一条命令

```bash
python traintest/bench_overlap.py --device cpu  --seeds 3              # 本机：只验结构
python traintest/bench_overlap.py --device cuda --seeds 3 --steps 20   # 高算：出数
```

产物：`runs/b_overlap/{overlap.csv, run.log}`

## 已确认的三件事（2026-10-24/25）

1. **结构跑通**：两档都能完整跑完，`rollout / 权重同步 / 训练` 三段可分开计时，
   同步搬运量可测（每次约 1.14 MiB @ d=128/2 层）。
2. 🔴 **CPU 上 overlap 结构性更慢**（0.72~0.75×），原因是自回归 generate 是
   `max_new_tokens` 次 Python 循环 + 小算子，**GIL 基本不放** ⇒ 两线程串行化还多付切换。
   **这不是实现错，是 CPU 的必然结果。**
3. 🔴 **GPU（A40 整卡）上 overlap 也略慢**：`0.933× ± 0.005`（MDE 0.006，可辨识地慢 7.2%），
   而 **GPU 利用率 98.1~100.0** —— 单卡上生成与训练排在**同一块 GPU**，串行化由设备决定、不由线程决定
   ⇒ **单卡 overlap 的理论上限就是 ≈1.0**。给 rollout 加 0~80 ms 的 CPU 段也没救回来（全程 1±2%）。

## 数字（详见 `kernel/BENCH.md` 表 5）

| 项 | 值 | 设备 |
|---|---|---|
| overlap 加速比 | **0.933× ± 0.005**（MDE 0.006） | A40 整卡 |
| GPU 利用率 | **98.1 ~ 100.0** | A40 整卡 |
| 三段占比（同步档） | rollout **0.834** / 权重同步 **0.025** / 训练 **0.141** | A40 整卡 |
| 权重同步量 | 每次 **1139.2 KiB**（59 次共 65.64 MiB） | — |
| MIG 上利用率 | `—`（平台限制，测不出，**不猜**） | A800 MIG 1g.10gb |

🔴 **踩过的坑（别再踩）**：首跑没预热，CUDA 初始化 + cuDNN autotune 全压在第一档头上，
seed 0 的 sync 2.200 s vs 同档 seed 1/2 的 0.35 s ⇒ 报出 1.843× ± 1.812 的**废数**。
现在默认 `--warmup 4` + `--reps 3` + `--steps 60`，SD 从 1.812 掉到 0.005。

## 还没做的

| 项 | 状态 |
|---|---|
| 推/训分离到不同设备的 overlap | ❌ 未做 —— 这是唯一可能让 overlap 真为正的配置（单卡做不到） |
| 大模型级吞吐 | ❌ 不做 —— `TinyPolicy` 只量结构，百分比不可外推 |

## 与 `algos/uav_tcnm/` 的关系（说清楚，别背错锅）

手册 §4 把 `algos/uav_tcnm/` 列为 B 阶段需要的链路，但 opt 仓
`docs/实验工程/08_装置迁移方案_uav_tcnm.md` **第 1 行自己写着"本文件是方案，不含实现"**，
且那里标注 trainer 要从未仓 3879 行 `main.py` 里剥 ≈1500 行 —— 那是**独立的大工程**。

而手册 B2 要的三个数字（overlap 加速比 / GPU 利用率 / 三段占比）**不依赖 uav_tcnm**：
它们量的是「训推一体的结构」，和 A2 用 toy decoder 量调度同理。
⇒ 本目录先用 `TinyPolicy` 把这套结构钉死并出数；uav_tcnm 移植另立任务，不挡 B2。
