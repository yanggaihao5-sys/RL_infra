# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-24 13:25 · dsh-agent ｜ 变更：新增 KV 与 BATCH 两种行前缀（kernel/kvlog.py 为镜像）

# 🕒 v1 · 2026-10-24 13:25 · dsh-agent ｜ 变更：新增 KV|/BATCH| 两种行前缀（kernel/kvlog.py 为镜像）

# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：新增 KV|/BATCH| 两种行前缀（kernel/kvlog.py 为镜像）

# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：新增 KV|/BATCH| 两种行前缀（kernel/kvlog.py 为镜像）

# 🕒 v2 · 2026-10-24 15:00 · dsh-agent ｜ 变更：新增 KV / BATCH 两种行前缀（推理内核侧契约，kernel/kvlog.py 为镜像） ｜ 上一版：v1
"""训练日志契约（spec §6）。

**日志不是给人逐行读的，是给脚本读的** —— 所以「行前缀 + 字段顺序」在这里一处定义，
业务代码只调方法、不自己拼字符串。这样换项目/改格式只改这一个文件，
`utils/analyze_training_log.py` 的正则也只需跟着这里对一次。

三层日志：
  ① 环境快照  `env_banner()`   开头 1 次  —— 对比两个实验前先对齐环境
  ② 每回合    `ep_line()`      每回合 1 行
  ③ 周期汇总  `train_line()` / `reward_line()`  每 print_interval 回合
运行身份：`identity()` 开头 1 行 —— 防「孪生符号静默跑错链路」（spec §8.2）
推理内核侧：`kv_line()` / `batch_line()` —— `KV |` / `BATCH |` 两种前缀（A 阶段 A1/A2），
  由 `kernel/kvlog.py` 镜像（kernel 要能独立跑），两边逐字互证见 `kernel/selfcheck_kv.py` ⑨。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Optional, Sequence


# ─────────────────── 推理内核侧行前缀（kernel/kvlog.py 的权威镜像）───────────────────
def format_kv_line(stats: Dict) -> str:
    """`KV |` 行：A1 分页 KV 的一行统计。"""
    return (f"KV | op:{stats.get('op', 'stats')} | seqs:{stats['seqs']} | "
            f"alloc:{stats['allocated_blocks']} | free:{stats['free_blocks']} | "
            f"frag:{stats['internal_frag']:.4f} | saved:{stats['shared_saved_blocks']}")


def format_batch_line(mode: str, result: Dict) -> str:
    """`BATCH |` 行：A2 批调度的一行结果。"""
    return (f"BATCH | mode:{mode} | steps:{result['steps']} | tokens:{result['tokens']} | "
            f"tps:{result['tokens_per_s']:.1f} | idle:{result['idle_slots']}")


class TrainLogger:
    """同时写 stdout 与文件（stdout 供 HPC 的 output.<jobid> 留痕，文件供本地诊断）。"""

    def __init__(self, log_path: Optional[str] = None, algo: str = "MADQN", echo: bool = True):
        self.algo = algo
        self.echo = echo
        self.log_path = log_path
        self._fh = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    # ───────────────────────── 底层 ─────────────────────────
    def emit(self, line: str = "") -> None:
        if self.echo:
            print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ───────────────────────── ① 身份 + 环境快照 ─────────────────────────
    def identity(self, chain: str, desc: str) -> None:
        """每条链路在日志开头打一行显式身份标识。"""
        self.emit(f"链路标识 | test_mode={chain} | {desc}")

    def env_banner(self, algo: str, items: Sequence[Sequence[str]]) -> None:
        """items: [(字段名, 值字符串), ...]，顺序即打印顺序。"""
        self.emit(f"环境配置一览（本次训练）[{algo}]")
        for k, v in items:
            self.emit(f"  {k:<24}: {v}")

    def derived(self, lines: Iterable[str]) -> None:
        """派生量自证块（spec §9.4：派生量要在日志里打印）。"""
        self.emit("派生量自证（算出来的，不是拍脑袋的）")
        for s in lines:
            self.emit(f"  {s}")

    def overview(self, device: str, prob_sum: float, env_count: int,
                 queue_size: int, episodes: int) -> None:
        self.emit(f"运行设备：{device} | 地图平均总奖励：{prob_sum:.2f} | "
                  f"并行环境数：{env_count} | 队列大小：{queue_size} | 总轮次={episodes}")

    # ───────────────────────── ② 每回合 ─────────────────────────
    def ep_line(self, ep: int, reward: float, coverage: float, steps: int,
                loss: float, cur_q: Optional[float], td: float) -> None:
        cur = "0.000" if cur_q is None else f"{cur_q:.3f}"
        self.emit(f"EP | ep:{ep:5d} | R:{reward:9.2f} | cov:{coverage:6.2f}% | "
                  f"steps:{steps:4d} | loss:{loss:.4f} | curQ:{cur} | td:{td:.4f}")

    # ───────────────────────── ③ 周期汇总 ─────────────────────────
    def train_line(self, ep: int, loss: float, cur_q: float, tgt_q: float,
                   eps: float, td: float, extra: str = "") -> None:
        tail = f" | {extra}" if extra else ""
        self.emit(f"TRAIN-{self.algo} | ep: {ep} | loss: {loss:.4f} | curQ: {cur_q:.3f} | "
                  f"tgtQ: {tgt_q:.3f} | eps: {eps:.4f} | td: {td:.4f}{tail}")

    def reward_line(self, comps: Dict[str, float]) -> None:
        """奖励分量均值行。分量名/值之间用空格分隔，值必须带正负号。"""
        body = " ".join(f"{k}:{v:+.5f}" for k, v in comps.items())
        self.emit(f"REWARD(每智能体每步均值) | {body}")

    # ───────────────────────── 事件 ─────────────────────────
    def best_line(self, ep: int, reward: float, coverage: float, path: str) -> None:
        self.emit(f"✅ 已更新 {self.algo} 最优模型（覆盖保存）| 回合：{ep} | "
                  f"全局奖励：{reward:.2f} | 覆盖率：{coverage:.2f}% | 路径：{path}")

    def saved(self, kind: str, path: str) -> None:
        self.emit(f"💾 模型+episode回放池保存完成：{path}  （类型：{kind}）")

    # ───────────────────────── 测试汇总 ─────────────────────────
    def test_line(self, coverage: float, reward: float, success_rate: float,
                  steps: float, extra: str = "") -> None:
        tail = f" | {extra}" if extra else ""
        self.emit(f"测试汇总 | 全局平均覆盖率：{coverage:.2f}% | 全局平均奖励：{reward:.2f} | "
                  f"成功率：{success_rate:.2f}% | 平均完成步数：{steps:.1f}{tail}")

    # ───────────────────────── 资源 ─────────────────────────
    def resources(self, start: str, end: str, cpu_sec: float, mem_mb: float) -> None:
        self.emit(f"Started at {start}")
        self.emit(f"Results reported at {end}")
        self.emit(f"CPU time: {cpu_sec:.6f} sec")
        self.emit(f"Max Memory: {mem_mb:.6f} MB")

    def info(self, msg: str) -> None:
        self.emit(f"ℹ️ {msg}")

    def warn(self, msg: str) -> None:
        self.emit(f"⚠️ {msg}")

    # ─────────────────── 推理内核侧（A1 / A2）───────────────────
    def kv_line(self, stats: Dict) -> None:
        """A1 分页 KV 统计行；格式权威在模块级 `format_kv_line`。"""
        self.emit(format_kv_line(stats))

    def batch_line(self, mode: str, result: Dict) -> None:
        """A2 批调度结果行；格式权威在模块级 `format_batch_line`。"""
        self.emit(format_batch_line(mode, result))
