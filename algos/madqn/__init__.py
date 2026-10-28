# -*- coding: utf-8 -*-
"""MADQN 链路（多智能体 DQN + 参数共享，栅格覆盖搜索）。

包内固定六件套（角色固定，只换前缀）：
    madqn_envs            环境（唯一「领域知识」集中地，最常改）
    madqn_networks        网络结构（QNetwork 角色）
    madqn_agent           智能体与训练逻辑（act / learn / 保存加载）
    madqn_replay_buffer   经验回放（数据结构，可独立测试与替换）
    madqn_parallel_env    多进程环境池（性能设施，与算法解耦）
    madqn_trainer         后台 rollout 线程 + 训练主循环（并发设施，可整体旁路）
"""
from .madqn_agent import MADQNCTDEAgent
from .madqn_envs import MADroneSearchEnv
from .madqn_parallel_env import AsyncWorkerEnv, SyncEnvPool, make_pool
from .madqn_replay_buffer import DroneSearchReplayBuffer
from .madqn_networks import MADQNQNetwork
from .madqn_trainer import RolloutActorThread, train_madqn_ctde, evaluate_madqn

__all__ = [
    "MADroneSearchEnv", "MADQNQNetwork", "MADQNCTDEAgent", "DroneSearchReplayBuffer",
    "SyncEnvPool", "AsyncWorkerEnv", "make_pool",
    "RolloutActorThread", "train_madqn_ctde", "evaluate_madqn",
]
