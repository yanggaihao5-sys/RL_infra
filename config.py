# -*- coding: utf-8 -*-
# 🕒 v3 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：13 个共用参数 MADQN_* → COMMON_*（前缀即作用域；取值逐字不变） ｜ 上一版：v2 · 2026-10-22 17:07

"""全项目唯一配置源。

规范（见 docs/架构决策.md）：
  ① 所有可调超参只在这里定义一处，函数内禁止魔法数字；
  ② 变量前缀即作用域：`COMMON_*` = 所有链路共享，`<链路>_*` = 只影响该链路
     （2026-10-17 #8：原来那批共享参数叫 `MADQN_*`，追逃链路复制后也在读 ⇒ 前缀宣布了虚假归属）；
  ③ 每行都有中文注释说明含义与调参后果；
  ④ 启动时 validate() 校验，派生量在日志里打印自证（防「看着正常实则致命」的值）。

改实验：只改 `name` + 下面某个参数；或用 CLI 覆盖（python main.py --episodes 500）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import List


@dataclass
class Config:
    # ══════════════════ ① 命名与产物目录（唯一命名源）══════════════════
    name: str = "madqn_demo"        # ★唯一命名源：换实验只改这一行；产物落 runs/<name>/
    runs_root: str = "runs"         # 产物根目录：与代码物理隔离（别让产物和代码平级）

    # ══════════════════ ② 链路与阶段（二维 dispatch）══════════════════
    test_mode: str = "madqn"        # 链路：跑哪套代码包（见 main.py 速查表）
    mode: str = "train"             # 阶段：train / test / smoke

    # ══════════════════ ③ 全局训练参数 ══════════════════
    agent_num: int = 2              # 智能体数量（改大→状态/奖励维度全变，需重训）
    grid_size: int = 20             # 栅格边长（20×20=400 格；30 更难，收敛更慢）
    episodes: int = 300             # ★总轮次。缩短会同时压缩探索衰减与步数上限→覆盖崩（UAV 项目实测 5000→1000 使覆盖 92%→8.7%）
    seed: int = 0                   # 随机种子：环境/网络/采样统一由此派生
    device: str = "auto"            # auto / cpu / cuda
    parallel: bool = False          # True=多进程环境池；受限环境（本机沙箱）保持 False
    env_count: int = 2              # 并行环境副本数（rollout 线程数），须 ≤ CPU 核数
    deterministic: bool = True      # True=torch 确定性算法（复现优先，略慢）

    # ══════════════════ ④ COMMON 参数（**两条链路共用**，2026-10-17 #8 重命名）══════════════════
    # 🔴 命名规则：`COMMON_*` = 所有链路共享（DQN 族通用：观测 patch / 网络宽度 / 学习率 / 回放池 / 探索）；
    #    `<链路>_*` = **只影响该链路**（`MADQN_best_metric` / `PURSUIT_*`）。
    #    为什么改：这批参数原名 `MADQN_*`，但追逃链路复制后也在读它们 —— 前缀宣布了一个虚假的归属。
    #    改名前后的**取值逐字不变**（`obs_dim=54`、`eps_decay_episodes=230` 已核对），历史上报的 run 仍可比；
    #    旧名与新名的 1:1 映射表见 `docs/库边界.md` §7。
    COMMON_patch: int = 2            # 局部视野半径 p → 观测含 (2p+1)² 的已访问/障碍双通道
    COMMON_hidden_dim: int = 128     # Q 网络隐层宽度
    COMMON_lr: float = 5e-4          # 学习率
    COMMON_gamma: float = 0.95       # 折扣因子
    COMMON_batch_size: int = 128     # 每次学习采样批大小
    COMMON_buffer_size: int = 50000  # 回放池容量（转移条数）
    COMMON_min_buffer: int = 1000    # 开始学习所需的最小样本数
    COMMON_learn_per_episode: int = 8  # 每收一个回合做几次梯度更新
    COMMON_target_update: int = 200  # 目标网络硬更新间隔（按梯度步计）
    COMMON_eval_interval: int = 25   # 每 N 回合做一次**贪心评估**用于选最优模型（0=关闭）
    COMMON_eps_start: float = 1.0    # 初始探索率
    COMMON_eps_end: float = 0.05     # 终值探索率
    COMMON_random_episodes: int = 10  # 前 N 轮纯随机（两段式探索：先铺满回放池再学）
    init_max_train_steps: int = 200  # 单回合步数上限（派生量会随 grid_size 自证打印）
    final_test_steps: int = 200     # 测试阶段单回合步数上限
    eval_runs: int = 5              # 评估跑几个回合
    eval_random_maps: bool = True   # 默认 True=每轮换一张地图（测泛化 + 给出真实方差）；
                                    #   要跨实验严格对齐环境请用 CLI --eval-fixed-maps
    eval_coverage_threshold: float = 0.75  # ★评测成功判据，**与训练终止阈值 coverage_threshold 解耦**。
                                    #   训练仍用 0.90（不动动力学 ⇒ 历史 run 可比）；评测按 0.75 判成功。
                                    #   起因：0.90 高于本任务测试图可达上限 → 成功率恒 0%、步数恒打满，指标无分辨力
    coverage_threshold: float = 0.90  # 覆盖率达成阈值（达到即回合结束 + 里程碑奖励）
    obstacle_num: int = 12          # 固定障碍物数量（位置由 seed 决定，每轮不变）
    target_num: int = 1             # 高价值目标点数量（覆盖时给 target_bonus）
    print_interval: int = 10        # 每多少回合打一行 TRAIN/REWARD 汇总
    save_interval: int = 50         # 每多少回合存一次 final 模型（递增序号不覆盖）
    save_keep_last: int = 5         # final 模型只保留最近 N 个（0=全部保留）。长跑不清理会堆成几百个文件

    # ══════════════════ ④b PURSUIT 链路参数（2026-10-17 迁移演练新增）══════════════════
    # 只放追逃域**真正新增**的东西；两条链路共用的那批在 ④（`COMMON_*`）
    PURSUIT_evader_period: int = 2      # 逃逸目标每 N 步移动一步（1=每步都动，最难追）
    PURSUIT_capture_radius: int = 0     # 捕获判定半径（曼哈顿距离，0=踩到同一格才算捕获）
    PURSUIT_R_approach: float = 0.5     # 每缩短 1 格曼哈顿距离的奖励（负值=只罚远离）
    PURSUIT_R_capture: float = 30.0     # 捕获成功的一次性奖励（每个智能体各得一份）

    # ══════════════════ ④c 选优判据（2026-10-17 修复 #10）══════════════════
    # 训练期「贪心评估 → 选 best」用哪个指标。写死覆盖率会让追逃域选出"扫得多"的模型，
    # 而实际要的是"抓得快"的那个（实测见 docs/迁移演练.md §5 缺陷 #4）。
    MADQN_best_metric: str = "coverage"         # 覆盖域：只有 coverage（1 维，与旧行为逐字等价）
    PURSUIT_best_metric: str = "capture_steps"  # 追逃域：先看是否捕获 → 再看步数 → 最后比覆盖

    # ══════════════════ ⑤ 奖励参数（每个分量一处定义）══════════════════
    R_new_cell: float = 1.0         # 每覆盖一个新格的基础正奖励
    R_repeat: float = -0.05         # 重复访问已覆盖格（抑制原地打转）
    R_obstacle: float = -0.20       # 撞固定障碍物（原地不动）
    R_collision: float = -0.10      # 两机同格（抑制挤在一起）
    R_step: float = -0.01           # 每步微小代价（鼓励尽快完成）
    R_target_bonus: float = 2.0     # 首次覆盖高价值目标点
    R_milestone_50: float = 5.0     # 全局覆盖率首破 50% 的一次性奖励
    R_milestone_75: float = 10.0    # 首破 75%
    R_milestone_threshold: float = 25.0  # 达成阈值的一次性奖励
    R_stagnation: float = -0.02     # 连续 window 步未覆盖新格（惩罚停滞）
    R_stagnation_window: int = 10   # 停滞判定窗口（步）

    # ══════════════════ 派生量 / 路径（只读，别手改）══════════════════
    def __post_init__(self) -> None:
        self._derive()

    def _derive(self) -> None:
        """所有派生量集中在这里，并在日志里自证（spec §9.4）。"""
        # 观测维度：局部已访问 patch + 局部障碍 patch + (行,列) + 覆盖率 + 邻域密度
        side = 2 * self.COMMON_patch + 1
        self.obs_dim: int = side * side * 2 + 4
        self.n_actions: int = 9                      # 8 方向 + 原地不动
        self.milestones = (0.5, 0.75, self.coverage_threshold)
        self.run_root: str = os.path.abspath(os.path.join(self.runs_root, self.name))
        self.eps_decay_episodes: int = max(1, int(self.episodes * 0.8) - self.COMMON_random_episodes)

    def derived_summary(self) -> List[str]:
        """派生量自证：这些数是「算出来的」，不是拍的——打印出来才能核对。"""
        return [
            f"obs_dim                 : {self.obs_dim}  (= 2×{2*self.COMMON_patch+1}² + 4)",
            f"n_actions               : {self.n_actions}",
            f"eps 衰减终点（回合）    : {self.COMMON_random_episodes + self.eps_decay_episodes}",
            f"随机预热回合            : {self.COMMON_random_episodes}",
            f"里程碑阈值              : {list(self.milestones)}",
            f"产物目录                : {self.run_root}",
        ]

    def validate(self) -> None:
        """启动即拦「看着正常实则致命」的值（spec §8.4）。"""
        assert self.mode in ("train", "test", "smoke"), f"未知 mode: {self.mode}"
        assert 1 <= self.agent_num <= 8, "agent_num 需在 1~8"
        assert 8 <= self.grid_size <= 200, "grid_size 需在 8~200"
        assert self.episodes >= 1, "episodes 必须 ≥ 1"
        assert 0.0 < self.COMMON_eps_end <= self.COMMON_eps_start <= 1.0, "探索率需 0<eps_end<=eps_start<=1"
        assert self.COMMON_random_episodes < self.episodes, \
            f"随机预热回合({self.COMMON_random_episodes}) 必须 < 总轮次({self.episodes})，否则全程随机不学习"
        assert 0.1 <= self.coverage_threshold <= 1.0, "覆盖率阈值需在 0.1~1.0"
        assert 0.1 <= self.eval_coverage_threshold <= 1.0, \
            "评测成功判据 eval_coverage_threshold 需在 0.1~1.0"
        assert self.COMMON_min_buffer <= self.COMMON_buffer_size, "min_buffer 不能大于 buffer_size"
        assert self.COMMON_eval_interval >= 0, "贪心评估间隔不能为负（0=关闭）"
        assert self.save_keep_last >= 0, "save_keep_last 不能为负（0=保留全部）"
        assert self.save_interval >= 1, "save_interval 至少为 1"
        assert self.PURSUIT_evader_period >= 1, "逃逸目标步频至少为 1（1=每步都动）"
        assert self.PURSUIT_capture_radius >= 0, "捕获半径不能为负"
        assert self.MADQN_best_metric in ("coverage",), \
            f"未知 MADQN_best_metric={self.MADQN_best_metric!r}（覆盖域只支持 coverage）"
        assert self.PURSUIT_best_metric in ("capture_steps", "coverage"), \
            f"未知 PURSUIT_best_metric={self.PURSUIT_best_metric!r}（可选 capture_steps / coverage）"
        assert self.obstacle_num < self.grid_size * self.grid_size * 0.2, "障碍物过多，地图不可覆盖"
        assert self.init_max_train_steps >= self.grid_size, \
            "单回合步数上限 < 栅格边长 → 物理上不可能覆盖完"
        if self.parallel:
            assert self.env_count >= 1, "并行环境数需 ≥ 1"

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
