# -*- coding: utf-8 -*-
"""rlinfra：与算法无关的公共基座（日志契约 / 产物路径 / 资源计量 / 链路注册表）。

规范：这一层不许出现任何「某个具体算法」的词；算法专属逻辑一律放 algos/<算法>/。
"""
from .logger import TrainLogger
from .metrics import ResourceMeter
from .paths import RunPaths
from .registry import AlgoSpec, get_algo, register, registered

__all__ = [
    "TrainLogger", "ResourceMeter", "RunPaths",
    "AlgoSpec", "get_algo", "register", "registered",
]
