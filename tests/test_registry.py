# -*- coding: utf-8 -*-
"""注册表：显式工厂、懒加载、两条链路的 train/eval 都登记。"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from rlinfra.registry import get_algo, registered

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_both_chains_register_train_and_eval():
    assert {"madqn", "pursuit"} <= set(registered())
    for chain in ("madqn", "pursuit"):
        spec = get_algo(chain)
        assert spec.package == f"algos.{chain}"
        assert spec.train_fn and spec.eval_fn
        assert callable(spec.train)
        assert callable(spec.evaluate)


def test_unknown_chain_fails_loudly():
    with pytest.raises(KeyError, match="未知链路"):
        get_algo("no_such_chain")


def test_get_algo_is_lazy_and_does_not_import_algorithm_package():
    code = (
        "import sys; from rlinfra.registry import get_algo; "
        "s = get_algo('madqn'); "
        "print(s.package, 'algos.madqn' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "algos.madqn False"
