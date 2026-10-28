# -*- coding: utf-8 -*-
# 🕒 v3 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：CLI dest 同步 COMMON_*，--lr/--batch-size 的 metavar 补回 ｜ 上一版：v2 · 2026-10-17 16:43

"""唯一入口：二维 dispatch（链路 × 阶段）。

════════════════════════ 主入口速查（与下表镜像同步，改动必须两处一起改）════════════════════════
  test_mode（链路） → 用哪个代码包、哪套类         |  MODE（阶段） → 跑到哪一步
  -------------------------------------------------|--------------------------------
  madqn             → algos/madqn/ 栅格覆盖搜索     |  train → 训练并保存模型
                                                   |  test  → 载入最优模型评估
                                                   |  smoke → 极短训练自检（冒烟）
════════════════════════════════════════════════════════════════════════════════════════════

用法：
    python main.py                                  # 用 config.py 的默认值训练
    python main.py --mode test                      # 评估
    python main.py --episodes 500 --name exp2       # 覆盖个别参数
    python main.py --parallel                       # 多进程环境池（集群上开）
    python main.py --print-config                   # 只打印最终配置（不跑）

⚠️ 环境池的并行分支必须在 `if __name__ == "__main__":` 保护下启动（多进程 spawn 要求），
   所以本文件所有动作都在 main() 里，模块级无副作用。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields

from config import Config
from rlinfra.logger import TrainLogger
from rlinfra.paths import RunPaths
from rlinfra.registry import get_algo, registered


# ══════════════════════ CLI → Config ══════════════════════
def build_config(argv=None) -> tuple[Config, argparse.Namespace]:
    ap = argparse.ArgumentParser(description="RL_infra：可迁移的 RL 训练基础设施")
    ap.add_argument("--name", type=str, help="实验名（唯一命名源，决定产物目录 runs/<name>/）")
    ap.add_argument("--mode", type=str, choices=["train", "test", "smoke"], help="阶段")
    ap.add_argument("--test-mode", dest="test_mode", type=str, help="链路（见下方 choices）")
    ap.add_argument("--episodes", type=int, help="总轮次")
    ap.add_argument("--grid-size", dest="grid_size", type=int)
    ap.add_argument("--agent-num", dest="agent_num", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", dest="COMMON_lr", type=float, metavar="LR", help="学习率")
    ap.add_argument("--batch-size", dest="COMMON_batch_size", type=int, metavar="N")
    ap.add_argument("--print-interval", dest="print_interval", type=int)
    ap.add_argument("--save-interval", dest="save_interval", type=int)
    ap.add_argument("--keep-last-final", dest="save_keep_last", type=int,
                    help="final 模型只保留最近 N 个（0=全部保留）")
    ap.add_argument("--device", type=str, help="auto / cpu / cuda")
    ap.add_argument("--parallel", action="store_true", default=None, help="开多进程环境池")
    ap.add_argument("--no-parallel", dest="parallel", action="store_false", help="关多进程环境池")
    ap.add_argument("--env-count", dest="env_count", type=int)
    ap.add_argument("--coverage-threshold", dest="coverage_threshold", type=float)
    ap.add_argument("--eval-runs", dest="eval_runs", type=int, help="评估跑几个回合")
    ap.add_argument("--eval-random-maps", dest="eval_random_maps", action="store_true",
                    default=None, help="评估时每轮换地图（config 默认已开）")
    ap.add_argument("--eval-fixed-maps", dest="eval_random_maps", action="store_false",
                    help="评估时固定地图（跨实验严格对齐环境用；多轮结果会逐字相同）")
    ap.add_argument("--eval-coverage-threshold", dest="eval_coverage_threshold", type=float,
                    help="评测成功判据（与训练阈值解耦，默认 0.75）")
    ap.add_argument("--print-config", action="store_true", help="打印最终配置后退出")
    ap.add_argument("--then-test", action="store_true",
                    help="训练完紧接着做评测（一行命令跑通：train + test）")
    args = ap.parse_args(argv)

    cfg = Config()
    for f in fields(Config):
        if hasattr(args, f.name) and getattr(args, f.name) is not None:
            setattr(cfg, f.name, getattr(args, f.name))
    cfg._derive()          # 参数改过 → 派生量必须重算（否则 obs_dim 会跟环境对不上）
    cfg.validate()
    return cfg, args


# ══════════════════════ 阶段实现 ══════════════════════
def run_smoke(cfg, paths: RunPaths, logger: TrainLogger) -> dict:
    """冒烟：把参数压到最小跑通一遍，证明「能 reset/step/学/存/打日志」。"""
    cfg.episodes = min(cfg.episodes, 3)
    cfg.COMMON_random_episodes = 1
    cfg.COMMON_min_buffer = min(cfg.COMMON_min_buffer, 200)
    cfg.COMMON_learn_per_episode = 1
    cfg.print_interval = 1
    cfg.save_interval = 2
    cfg.COMMON_eval_interval = 1     # 只跑几回合 → 每回合都做贪心评估，否则存不上 best
    cfg.parallel = False            # 冒烟永远走降级路径，保证受限环境也能跑
    cfg.env_count = min(cfg.env_count, 2)
    logger.warn(f"冒烟模式：episodes={cfg.episodes} grid={cfg.grid_size} "
                f"min_buffer={cfg.COMMON_min_buffer} parallel=False")
    spec = get_algo(cfg.test_mode)
    return spec.train(cfg, paths, logger)


def run_train(cfg, paths: RunPaths, logger: TrainLogger) -> dict:
    return get_algo(cfg.test_mode).train(cfg, paths, logger)


def run_test(cfg, paths: RunPaths, logger: TrainLogger) -> dict:
    # 2026-10-17（迁移演练）：原来这里硬编码 `from algos.madqn.madqn_trainer import evaluate_madqn`，
    # 换链路后 `--mode test` 会**静默地**评测 madqn 的模型。改为走注册表的 eval_fn。
    return get_algo(cfg.test_mode).evaluate(cfg, paths, logger)


MODES = {"train": run_train, "test": run_test, "smoke": run_smoke}


# ══════════════════════ 入口 ══════════════════════
def main(argv=None) -> int:
    cfg, args = build_config(argv)

    if args.print_config:
        for k, v in cfg.to_dict().items():
            print(f"{k:28s} = {v}")
        return 0

    paths = RunPaths.from_config(cfg).ensure()
    logger = TrainLogger(paths.log, algo=cfg.test_mode.upper())
    try:
        logger.emit("=" * 72)
        logger.emit(f"RL_infra | 链路 {cfg.test_mode} | 阶段 {cfg.mode} | 实验 {cfg.name}")
        logger.emit(f"主入口速查：链路 ∈ {sorted(registered())}；阶段 ∈ {sorted(MODES)}")
        logger.emit("=" * 72)
        result = MODES[cfg.mode](cfg, paths, logger)
        logger.info(f"阶段 {cfg.mode} 完成 → {result}")
        if args.then_test and cfg.mode == "train":
            # 训练阶段**不做**最终评测（见 AGENTS.md §6），要评测必须显式再跑一次
            logger.emit("-" * 72)
            logger.emit("--then-test：训练完成，接着做评测（等价于 --mode test）")
            result = run_test(cfg, paths, logger)
            logger.info(f"阶段 test 完成 → {result}")
        elif cfg.mode == "train":
            logger.info(f"下一步：python main.py --mode test --name {cfg.name}"
                        f"（或这次直接加 --then-test）")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
