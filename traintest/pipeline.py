# 🕒 v3 · 2026-10-25 20:14 · dsh-agent ｜ 变更：支持推/训分离：device_rollout 独立；镜像搬运计入训练段；run_both 用 CPU 副本做初始化 ｜ 上一版：v2 · 2026-10-25 17:00

"""B1 · 训推一体的两档实现：**同步** 与 **overlap**

## 两档的定义（只差这一点）

| 档 | 一轮循环 | rollout 用的权重 |
|---|---|---|
| `sync` | generate → 权重同步 → train，**串行** | 每次同步后的最新权重 |
| `overlap` | 训练第 i 步的同时，rollout 线程跑第 i+1 步的 generate | **滞后一步**（副本，无数据竞争） |

## 为什么 overlap 一定保守报

overlap 的收益来自"生成与训练用不同硬件资源"。本项目是 **MIG 切片**，
生成与训练抢同一块 SM ⇒ CPU 上 overlap 基本不赚，GPU/MIG 上要实测。
**所以本文件不假设加速比为正**，它只负责把三段耗时分别量出来。

## 三段归因（手册 B2 第 3 条）

每轮拆成 `rollout / 权重同步 / 训练` 三段，最后给占比表。
判定规则（手册 §3 B2）：**若 rollout > 60%，说明优化推理引擎才对 RL 吞吐有杠杆。**
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional

import torch

from .rollout import RolloutEngine, TinyPolicy, WeightSync, sample_states


class _Timer:
    """累加三段耗时；线程安全（overlap 档 rollout 在别的线程里）"""

    def __init__(self):
        self._lock = threading.Lock()
        self.rollout = 0.0
        self.sync = 0.0
        self.train = 0.0

    def add(self, key: str, dt: float) -> None:
        with self._lock:
            setattr(self, key, getattr(self, key) + dt)

    def total(self) -> float:
        return self.rollout + self.sync + self.train

    def shares(self) -> Dict[str, float]:
        tot = self.total() or 1.0
        return {"rollout": self.rollout / tot, "sync": self.sync / tot, "train": self.train / tot}


class Pipeline:
    """两条 pipeline 的共同部分：训练步、负载生成、统计"""

    mode = "base"

    def __init__(self, trainer: TinyPolicy, engine: RolloutEngine, *,
                 steps: int = 20, batch: int = 8, prompt_len: int = 8,
                 max_new_tokens: int = 16, lr: float = 1e-3,
                 device: str = "cpu", device_rollout: str | None = None,
                 seed: int = 0, rollout_cpu_ms: float = 0.0):
        self.device = device
        self.device_rollout = device_rollout or device
        self.trainer = trainer.to(self.device)
        self.engine = engine
        self.sync = WeightSync(self.trainer, engine)
        self.steps = steps
        self.batch, self.prompt_len, self.max_new_tokens = batch, prompt_len, max_new_tokens
        self.device, self.seed = device, seed
        self.split = self.device != self.device_rollout
        # rollout 侧的**非 GPU 段**（真部署里是 tokenize / 后处理 / 网络往返）
        # 它不进 GPU，所以才有被训练段叠掉的可能 —— overlap 的收益来源就在这里
        self.rollout_cpu_ms = float(rollout_cpu_ms)
        self.opt = torch.optim.AdamW(self.trainer.parameters(), lr=lr)
        self.timer = _Timer()
        self.tokens = 0

    def _cpu_stall(self) -> None:
        if self.rollout_cpu_ms > 0:
            time.sleep(self.rollout_cpu_ms / 1000.0)

    # ── 训练一步：把 rollout 产出的动作当标签，跑一次前向+反向 ──
    def _train_step(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        t0 = time.perf_counter()
        self.trainer.train()
        # 推/训分离时 actions 在 rollout 设备上，必须先搬到训练设备（这段开销计入训练段）
        if actions.device != states.device:
            actions = actions.to(states.device)
        seq = torch.cat([states, actions], dim=1)
        logits = self.trainer(seq[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), seq[:, 1:].reshape(-1))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.timer.add("train", time.perf_counter() - t0)

    def _generate(self, states: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        self._cpu_stall()                  # rollout 侧非 GPU 段（计时含它，因为它属于 rollout）
        acts = self.engine.generate(states, self.max_new_tokens)
        self.timer.add("rollout", time.perf_counter() - t0)
        self.tokens += int(acts.numel())
        return acts

    def _add_tokens(self, n: int) -> None:
        """给常驻 worker 线程用（它在别的线程里累加 token 数）"""
        with self.timer._lock:
            self.tokens += n

    def _sync(self) -> None:
        t0 = time.perf_counter()
        self.sync.broadcast()
        self.timer.add("sync", time.perf_counter() - t0)

    def result(self) -> Dict:
        tot = self.timer.total()
        sh = self.timer.shares()
        return {
            "mode": self.mode, "steps": self.steps, "seconds": tot,
            "tokens": self.tokens,
            "tokens_per_s": self.tokens / tot if tot else 0.0,
            "rollout_s": self.timer.rollout, "sync_s": self.timer.sync, "train_s": self.timer.train,
            "share_rollout": sh["rollout"], "share_sync": sh["sync"], "share_train": sh["train"],
            "sync_bytes": self.sync.bytes_moved,
        }


class SyncPipeline(Pipeline):
    """串行档：generate → 同步 → train，一轮接一轮"""

    mode = "sync"

    def run(self) -> Dict:
        for i in range(self.steps):
            states = sample_states(self.batch, self.prompt_len,
                                   self.trainer.vocab, self.device, seed=self.seed + i)
            acts = self._generate(states)      # 用当前（最新）权重
            self._sync()                       # 生成完才同步给下一轮
            self._train_step(states, acts)
        return self.result()


class _RolloutWorker:
    """常驻 rollout 线程（**不是每步新建线程**）

    为什么常驻：每步 `threading.Thread(...)` 的创建/回收本身就是毫秒级开销，
    会把 overlap 的收益吃掉、把结论污染成"overlap 更慢"。常驻线程才对应真实部署里
    那个长期运行的 rollout 进程。
    """

    def __init__(self, engine: RolloutEngine, timer: _Timer, on_tokens, max_new: int,
                 cpu_ms: float = 0.0):
        self.engine = engine
        self.timer = timer
        self._on_tokens = on_tokens
        self._max_new = max_new
        self._cpu_ms = float(cpu_ms)
        self._in: "queue.Queue" = queue.Queue()
        self._out: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._err: List[BaseException] = []
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            states = item
            t0 = time.perf_counter()
            try:
                if self._cpu_ms > 0:
                    time.sleep(self._cpu_ms / 1000.0)   # rollout 侧非 GPU 段，计入 rollout 计时
                acts = self.engine.generate(states, self._max_new)
            except BaseException as e:
                self._err.append(e)
                self._out.put(None)
                return
            self.timer.add("rollout", time.perf_counter() - t0)
            self._on_tokens(int(acts.numel()))
            self._out.put(acts)

    def submit(self, states: torch.Tensor) -> None:
        self._in.put(states)

    def get(self) -> torch.Tensor:
        acts = self._out.get()
        if self._err:
            raise self._err[0]
        if acts is None:
            raise RuntimeError("rollout worker 异常退出")
        return acts

    def close(self) -> None:
        self._in.put(None)
        self._stop.set()
        self._th.join(timeout=3)


class OverlapPipeline(Pipeline):
    """overlap 档：训练第 i 步时，rollout worker 已在跑第 i+1 步的 generate

    权重滞后**一步**（`AGENTS.md` §6 已记录的代价）。worker 用独立权重副本，
    不碰 trainer 参数 ⇒ 无数据竞争。权重同步放在 worker 空闲时（拿到上一步结果之后）。
    """

    mode = "overlap"

    def run(self) -> Dict:
        w = _RolloutWorker(self.engine, self.timer, self._add_tokens, self.max_new_tokens,
                           cpu_ms=self.rollout_cpu_ms)
        try:
            states = sample_states(self.batch, self.prompt_len,
                                   self.trainer.vocab, self.device, seed=self.seed)
            w.submit(states)
            acts = w.get()                                  # 第 0 步没有可重叠的前置

            for i in range(self.steps):
                if i + 1 < self.steps:                      # 先派发下一步 → 与训练重叠
                    nxt = sample_states(self.batch, self.prompt_len,
                                        self.trainer.vocab, self.device, seed=self.seed + i + 1)
                    w.submit(nxt)
                self._train_step(states, acts)              # 主线程训练当前步
                if i + 1 < self.steps:
                    states = nxt
                    acts = w.get()                          # 等 worker 交出下一步动作
                    self._sync()                            # worker 空闲 ⇒ 在这里同步，滞后一步
        finally:
            w.close()
        return self.result()


def run_both(trainer: TinyPolicy, engine: RolloutEngine, **kw) -> Dict[str, Dict]:
    """同一份权重、同一 seed，跑两档；返回两者结果（用于算加速比）

    `base` 存成 **CPU 副本**：推/训分离时两档的模型落在不同设备上，
    用 CPU 副本做"同一份初始化"才不会被设备绑死（`load_state_dict` 的 `copy_` 支持跨设备）。
    """
    import copy
    base = {k: v.detach().cpu().clone() for k, v in trainer.state_dict().items()}

    def _fresh() -> tuple[TinyPolicy, RolloutEngine]:
        from .rollout import TorchRolloutEngine
        t = copy.deepcopy(trainer)
        t.load_state_dict(base)
        r = copy.deepcopy(engine.model)
        r.load_state_dict(base)
        r.to(engine.device).eval()
        return t, TorchRolloutEngine(r, device=engine.device)

    t1, e1 = _fresh()
    r_sync = SyncPipeline(t1, e1, **kw).run()
    t2, e2 = _fresh()
    r_over = OverlapPipeline(t2, e2, **kw).run()
    return {"sync": r_sync, "overlap": r_over}
