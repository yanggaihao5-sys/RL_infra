# -*- coding: utf-8 -*-
"""资源计量：墙钟 / CPU 核时 / 峰值内存。

输出格式正好喂给日志契约的 `Started at ... / Results reported at ... /
CPU time: ... sec / Max Memory: ... MB` 四行（analyze_training_log.py 会解析）。
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Optional, Tuple

try:  # Linux / macOS 有 resource；Windows 没有
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None

import tracemalloc


class ResourceMeter:
    def __init__(self) -> None:
        self.t0: float = 0.0
        self.t1: Optional[float] = None
        self.c0: float = 0.0
        self._tracemalloc_started = False

    def start(self) -> "ResourceMeter":
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()
        if _resource is None and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started = True
        return self

    def stop(self) -> "ResourceMeter":
        self.t1 = time.perf_counter()
        return self

    # ───────────────────────── 指标 ─────────────────────────
    @property
    def wall_sec(self) -> float:
        return (self.t1 if self.t1 is not None else time.perf_counter()) - self.t0

    @property
    def cpu_sec(self) -> float:
        return time.process_time() - self.c0

    @property
    def mem_mb(self) -> float:
        if _resource is not None:
            ru = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            # Linux 单位 KB；macOS 单位 bytes
            return ru / 1024.0 if sys.platform != "darwin" else ru / 1024.0 / 1024.0
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            return peak / 1024.0 / 1024.0
        return 0.0

    def stamps(self) -> Tuple[str, str]:
        """(开始时刻, 结束时刻)，格式对齐 START_RE / END_RE。"""
        now = datetime.now()
        end = now
        start = datetime.fromtimestamp(now.timestamp() - self.wall_sec)
        return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")
