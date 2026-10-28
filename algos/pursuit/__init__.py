# -*- coding: utf-8 -*-
"""PURSUIT 链路（多智能体 DQN + 参数共享，栅格覆盖搜索）。

包内固定六件套（角色固定，只换前缀）：
    pursuit_envs            环境（唯一「领域知识」集中地，最常改）
    pursuit_networks        网络结构（QNetwork 角色）
    pursuit_agent           智能体与训练逻辑（act / learn / 保存加载）
    pursuit_replay_buffer   经验回放（数据结构，可独立测试与替换）
    pursuit_parallel_env    多进程环境池（性能设施，与算法解耦）
    pursuit_trainer         后台 rollout 线程 + 训练主循环（并发设施，可整体旁路）
"""
from .pursuit_agent import PursuitCTDEAgent
from .pursuit_envs import PursuitSearchEnv
from .pursuit_parallel_env import AsyncWorkerEnv, SyncEnvPool, make_pool
from .pursuit_replay_buffer import PursuitSearchReplayBuffer
from .pursuit_networks import PursuitQNetwork
from .pursuit_trainer import RolloutActorThread, train_pursuit_ctde, evaluate_pursuit

__all__ = [
    "PursuitSearchEnv", "PursuitQNetwork", "PursuitCTDEAgent", "PursuitSearchReplayBuffer",
    "SyncEnvPool", "AsyncWorkerEnv", "make_pool",
    "RolloutActorThread", "train_pursuit_ctde", "evaluate_pursuit",
]
