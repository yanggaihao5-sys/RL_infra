# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-17 16:43 · dsh-agent ｜ 变更：AlgoSpec 补 eval_fn + evaluate：修 --mode test 硬编码 madqn 的静默错链路；登记 pursuit

"""链路注册表：`test_mode` → 一套类与训练函数。

**为什么不用「类重绑定」**（spec §5.3 的隐式注入）：
  UAV 项目用 `MADroneSearchEnv = ExpertMADroneSearchEnv` 实现零侵入切换，
  代价是「import 成功、跑完不报错、却跑错链路」，只能靠日志事后发现（spec §8.2 真实事故）。
  这里改成**显式工厂**：链路必须在表里登记，取不到的链路直接报错。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class AlgoSpec:
    chain: str            # 链路名（= test_mode）
    package: str          # 代码包（import 路径）
    algo_tag: str         # 产物文件名用的算法短名（如 madqn）
    env_cls: str          # 环境类名
    agent_cls: str        # 智能体类名
    buffer_cls: str       # 回放池类名
    pool_cls: str         # 环境池类名
    train_fn: str         # 训练函数名
    eval_fn: str          # 评估函数名（**迁移演练 2026-10-17 补**：原来只登记了 train，
                          #   main.py 里 `--mode test` 硬编码 madqn ⇒ 换链路后评测跑的还是 madqn）
    map_desc: str         # 人读的地图/动作描述（打印到链路身份行）

    # ───────────────────────── 懒加载（避免循环 import）─────────────────────────
    def _load(self, attr: str) -> Any:
        mod = importlib.import_module(self.package)
        try:
            return getattr(mod, getattr(self, attr))
        except AttributeError as e:  # 孪生符号防线的第二道：名字对不上立刻炸
            raise AttributeError(
                f"链路 {self.chain} 在 {self.package} 里找不到 {getattr(self, attr)}；"
                f"检查 rlinfra/registry.py 的登记名与 algos/ 的实际类名是否一致"
            ) from e

    @property
    def env(self) -> Any:
        return self._load("env_cls")

    @property
    def agent(self) -> Any:
        return self._load("agent_cls")

    @property
    def buffer(self) -> Any:
        return self._load("buffer_cls")

    @property
    def pool(self) -> Any:
        return self._load("pool_cls")

    @property
    def train(self) -> Callable:
        return self._load("train_fn")

    @property
    def evaluate(self) -> Callable:
        return self._load("eval_fn")


_ALGOS: Dict[str, AlgoSpec] = {}


def register(spec: AlgoSpec) -> AlgoSpec:
    _ALGOS[spec.chain] = spec
    return spec


def get_algo(chain: str) -> AlgoSpec:
    if chain not in _ALGOS:
        raise KeyError(f"未知链路 test_mode={chain!r}；已注册：{sorted(_ALGOS)}（见 main.py 速查表）")
    return _ALGOS[chain]


def registered() -> Dict[str, AlgoSpec]:
    return dict(_ALGOS)


# 官方链路登记（新增链路 = 加一行 + 复制一个 algos/<算法>/ 包）
register(AlgoSpec(
    chain="madqn",
    package="algos.madqn",
    algo_tag="madqn",
    env_cls="MADroneSearchEnv",
    agent_cls="MADQNCTDEAgent",
    buffer_cls="DroneSearchReplayBuffer",
    pool_cls="make_pool",
    train_fn="train_madqn_ctde",
    eval_fn="evaluate_madqn",
    map_desc="栅格覆盖搜索 | 9 动作（8 方向 + 停）",
))

# 追逃链路（2026-10-17 迁移演练：由 algos/madqn 复制 + 重写 envs 得到，见 docs/迁移演练.md）
register(AlgoSpec(
    chain="pursuit",
    package="algos.pursuit",
    algo_tag="pursuit",
    env_cls="PursuitSearchEnv",
    agent_cls="PursuitCTDEAgent",
    buffer_cls="PursuitSearchReplayBuffer",
    pool_cls="make_pool",
    train_fn="train_pursuit_ctde",
    eval_fn="evaluate_pursuit",
    map_desc="追逃搜索 | 9 动作（8 方向 + 停）| 目标每 N 步移动",
))
