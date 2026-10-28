# 🕒 v1 · 2026-10-24 13:58 · dsh-agent ｜ 变更：B 阶段包入口（训推一体最小系统）

"""B 阶段 · 训推一体最小系统（需要 torch，CPU/GPU 同一份代码）

    traintest/rollout.py     rollout 引擎协议 + torch 实现 + 权重同步
    traintest/pipeline.py    同步 / overlap 两档 + 三段瓶颈归因
    traintest/bench_overlap.py   B2 基准（加速比 ± SD + MDE + 归因表）

为什么单独一个包而不放进 `kernel/`：`kernel/` 是 A 阶段的**零第三方依赖**纯库
（`utils/check_boundaries.py` ⑧ 会检查这一点），而这里必须 import torch。
分开之后那条边界检查才有意义，而不是被"开个例外"糊过去。
"""
