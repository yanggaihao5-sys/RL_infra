# -*- coding: utf-8 -*-
"""环境池（parallel_env 角色）：性能设施，**与算法完全解耦**。

两个实现，同一套接口 —— 上层训练循环 / rollout 线程**一个字都不用改**：

| 实现 | 何时用 | 代价 |
|---|---|---|
| `SyncEnvPool` | **受限环境 / 本机沙箱 / 冒烟自检**（默认） | 单进程串行，吞吐低 |
| `AsyncWorkerEnv` | 集群 / 本地终端（可开 `--parallel`） | 多进程有启动开销；受限沙箱可能禁命名管道 |

> spec §8.7 的教训：`mp.Queue` 走命名管道，**受限沙箱里会被禁** ——
> 所以「单进程降级路径」不是可选项，是必须预留的。
> `make_pool(cfg, ...)` 是唯一入口，调用方不许直接 new。

接口约定（两个实现都必须满足）：
    reset()                -> List[List[obs]]         # [env_idx][agent_idx]
    step(actions)          -> (obs, rewards, dones, infos)
    snapshot()             -> List[(字段, 值)]         # 取自 env 0，供环境快照打印
    close()
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

Obs = List[np.ndarray]


class SyncEnvPool:
    """单进程池：串行持有 n 个环境副本（降级路径，行为与多进程版一致）。"""

    def __init__(self, cfg, env_cls: Callable, n: int, seed: int = 0,
                 max_steps: Optional[int] = None, **env_kwargs):
        self.cfg = cfg
        self.num_envs = int(n)
        self.envs = [env_cls(cfg, seed=seed + i, max_steps=max_steps, **env_kwargs)
                     for i in range(self.num_envs)]

    def reset(self) -> List[Obs]:
        return [env.reset() for env in self.envs]

    def reset_envs(self, idxs) -> List[Obs]:
        return [self.envs[k].reset() for k in idxs]

    def step(self, actions: List[np.ndarray]):
        obs, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, d, i = env.step(a)
            obs.append(o); rewards.append(r); dones.append(d); infos.append(i)
        return obs, rewards, dones, infos

    def snapshot(self) -> List[Tuple[str, str]]:
        return self.envs[0].snapshot()

    def close(self) -> None:
        self.envs = []


# ══════════════════════ 多进程实现 ══════════════════════
def _worker_main(rank: int, cfg, seed: int, max_steps: Optional[int],
                 env_path: str, env_cls_name: str, q_in, q_out) -> None:
    """子进程主体：持有一个环境，按命令（reset/step/snapshot/close）响应。

    必须是**模块级函数**（可 pickle）—— 这也是为什么环境类要在子进程里按
    「模块路径 + 类名」重新 import，而不是直接把类对象递进来。
    """
    import importlib
    env_cls = getattr(importlib.import_module(env_path), env_cls_name)
    env = env_cls(cfg, seed=seed, max_steps=max_steps)
    try:
        while True:
            cmd, payload = q_in.get()
            if cmd == "close":
                break
            if cmd == "reset":
                q_out.put(("ok", env.reset()))
            elif cmd == "step":
                o, r, d, i = env.step(payload)
                q_out.put(("ok", (o, r, d, i)))
            elif cmd == "snapshot":
                q_out.put(("ok", env.snapshot()))
            else:
                q_out.put(("err", f"unknown cmd {cmd}"))
    finally:
        q_out.put(("bye", None))


class AsyncWorkerEnv:
    """多进程环境池：**先给所有 worker 发命令，再统一收结果** → 真并行。"""

    def __init__(self, cfg, env_cls: Callable, n: int, seed: int = 0,
                 max_steps: Optional[int] = None, **_):
        self.cfg = cfg
        self.num_envs = int(n)
        self.ctx = mp.get_context("spawn")
        self.env_path = f"{env_cls.__module__}"
        self.env_cls_name = env_cls.__name__

        self._q_in: List[Any] = []
        self._q_out: List[Any] = []
        self._procs: List[Any] = []
        for rank in range(self.num_envs):
            q_in, q_out = self.ctx.Queue(), self.ctx.Queue()
            p = self.ctx.Process(
                target=_worker_main,
                args=(rank, cfg, seed + rank, max_steps, self.env_path, self.env_cls_name, q_in, q_out),
                daemon=True,
            )
            p.start()
            self._q_in.append(q_in); self._q_out.append(q_out); self._procs.append(p)

    # ── 广播 + 收集 ──
    def _broadcast(self, cmd: str, payloads: Optional[List[Any]] = None,
                   targets: Optional[List[int]] = None) -> List[Any]:
        idxs = list(range(self.num_envs)) if targets is None else list(targets)
        for k in idxs:
            self._q_in[k].put((cmd, None if payloads is None else payloads[k]))
        out = []
        for k in idxs:
            tag, val = self._q_out[k].get()
            if tag == "err":
                raise RuntimeError(f"worker 报错：{val}")
            out.append(val)
        return out

    def reset(self) -> List[Obs]:
        return self._broadcast("reset")

    def reset_envs(self, idxs) -> List[Obs]:
        return self._broadcast("reset", targets=list(idxs))

    def step(self, actions: List[np.ndarray]):
        obs, rewards, dones, infos = [], [], [], []
        for o, r, d, i in self._broadcast("step", [np.asarray(a) for a in actions]):
            obs.append(o); rewards.append(r); dones.append(d); infos.append(i)
        return obs, rewards, dones, infos

    def snapshot(self) -> List[Tuple[str, str]]:
        return self._broadcast("snapshot")[0]

    def close(self) -> None:
        for q in self._q_in:
            q.put(("close", None))
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def make_pool(cfg, env_cls: Callable, n: Optional[int] = None, seed: Optional[int] = None,
              max_steps: Optional[int] = None) -> Any:
    """环境池唯一入口：按 cfg.parallel 返回多进程或单进程实现。"""
    n = int(n if n is not None else cfg.env_count)
    seed = int(seed if seed is not None else cfg.seed)
    cls = AsyncWorkerEnv if cfg.parallel else SyncEnvPool
    return cls(cfg, env_cls, n=n, seed=seed, max_steps=max_steps)
