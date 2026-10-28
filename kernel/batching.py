# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：实现 ContinuousBatcher 四条 TODO + StaticBatcher 分块与 idle 统计

"""批处理调度（A2）：static vs continuous 的**接口与参照实现**

本文件只做调度，不碰模型 —— 模型由调用方传入（CPU 上见 `bench_batching.py` 的 toy decoder）。
这样"调度带来的吞吐差"可以在**同一模型**下单独测量。**本文件零第三方依赖。**

约定：
  · 一个"步" = 对当前 batch 里每条活跃序列各前进一步（generate 1 token）
  · `run(step_fn)` 里 `step_fn(n_slots)` 表示"这次前向要跑 n_slots 个槽位"，由调用方决定开销
  · 返回同构 dict：steps / tokens / seconds / tokens_per_s / idle_slots / peak_live

两档的唯一差别（这就是 A2 要量的东西）：
  · **static**：先按到达顺序切成长度 `max_batch` 的块，块内**必须等最长的那条**结束；
                每步都按整块 padded 前向 ⇒ 已结束的序列继续占槽位（那是**空转**）
  · **continuous**：一个池子，某条结束立刻从等待队列补位到 `max_batch` 上限
                ⇒ 空转槽位只剩"队列确实空了"的那部分

`idle_slots` = Σ_每步 (槽位上限 − 该步活跃数)，两档直接可比：
  static 的 idle 来自**块内等最长**，continuous 的 idle 来自**池子见了底**。
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List

StepFn = Callable[[int], None]   # step_fn(n_slots) -> None

def _summary(steps: int, tokens: int, seconds: float, idle: int, peak: int) -> Dict:
    return {"steps": steps, "tokens": tokens, "seconds": seconds,
            "tokens_per_s": tokens / seconds if seconds else 0.0,
            "idle_slots": idle, "peak_live": peak}

class StaticBatcher:
    """静态批：块内所有序列必须等最长的那条结束（最朴素的实现对对照基线）"""

    def __init__(self, seqs: List[int], max_batch: int = 8):
        self.seqs = list(seqs)
        self.max_batch = int(max_batch)
        if self.max_batch <= 0:
            raise ValueError("max_batch 必须为正")

    def run(self, step_fn: StepFn) -> dict:
        t0 = time.perf_counter()
        tokens = idle = steps = peak = 0
        for i in range(0, len(self.seqs), self.max_batch):
            blk = self.seqs[i:i + self.max_batch]   # 该块的槽位上限 = 块大小
            cap = len(blk)
            rem = list(blk)
            while True:
                live = sum(1 for r in rem if r > 0)
                if live == 0:
                    break
                steps += 1
                step_fn(cap)          # 整块 padded 前向：已结束的序列仍占槽
                idle += cap - live
                peak = max(peak, live)
                for j, r in enumerate(rem):
                    if r > 0:
                        rem[j] = r - 1
                        tokens += 1
        dt = time.perf_counter() - t0
        return _summary(steps, tokens, dt, idle, peak)

class ContinuousBatcher:
    """连续批（token 级补位）：某序列结束立刻补进新序列，不等整批

    A2 的 4 条 TODO（已全部实现）：
      1. 维护 `running` / `waiting` 两个队列；每步结束后把 running 里完成的弹出   ← `_retire_finished`
      2. 从 waiting 补位到目标 batch 上限                                        ← `_refill`
      3. 统计空转槽位 = 每步 (上限 − 活跃数) 的累计                              ← `idle`
      4. 返回与 StaticBatcher 同构的 dict（含 `idle_slots`）                      ← `_summary`
    """

    def __init__(self, seqs: List[int], max_batch: int = 8):
        self.waiting: List[int] = list(seqs)
        self.max_batch = int(max_batch)
        if self.max_batch <= 0:
            raise ValueError("max_batch 必须为正")
        self.running: List[int] = []      # 每条在跑序列的**剩余 token 数**

    # ── TODO 1：把已完成的弹出 ──
    def _retire_finished(self) -> int:
        before = len(self.running)
        self.running = [r for r in self.running if r > 0]
        return before - len(self.running)

    # ── TODO 2：从等待队列补位到上限 ──
    def _refill(self) -> int:
        added = 0
        while len(self.running) < self.max_batch and self.waiting:
            self.running.append(self.waiting.pop(0))
            added += 1
        return added

    # ── TODO 3 + 4：空转统计 + 同构返回 ──
    def run(self, step_fn: StepFn) -> dict:
        t0 = time.perf_counter()
        self.running = []
        tokens = idle = steps = peak = 0
        self._refill()                       # 开局先填满
        while self.running:
            n = len(self.running)
            steps += 1
            idle += self.max_batch - n       # 池子见底才会 >0
            peak = max(peak, n)
            step_fn(n)
            for i in range(n):
                self.running[i] -= 1
            tokens += n
            self._retire_finished()          # 先退役
            self._refill()                   # 再补位
        dt = time.perf_counter() - t0
        return _summary(steps, tokens, dt, idle, peak)

def speedup(static: dict, continuous: dict) -> dict:
    """两档对比；`work_ratio` 用「槽位步数」算，与模型无关，是**纯调度收益**"""
    s_work = static["tokens"] + static["idle_slots"]
    c_work = continuous["tokens"] + continuous["idle_slots"]
    return {
        "wall_ratio": (static["seconds"] / continuous["seconds"]
                       if continuous["seconds"] else float("nan")),
        "tps_ratio": (continuous["tokens_per_s"] / static["tokens_per_s"]
                      if static["tokens_per_s"] else float("nan")),
        "work_ratio": s_work / c_work if c_work else float("nan"),
        "idle_static": static["idle_slots"],
        "idle_continuous": continuous["idle_slots"],
        "tokens": static["tokens"],
    }
