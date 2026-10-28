# -*- coding: utf-8 -*-
"""pytest 路径夹具：把仓库根与 utils/ 放进 sys.path，测试不依赖安装状态。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
