# -*- coding: utf-8 -*-
"""产物路径：唯一命名源、final 序号单调递增、清理边界。"""
from __future__ import annotations

from rlinfra.paths import RunPaths


def _touch(path):
    path.write_bytes(b"x")


def test_next_final_uses_max_plus_one_not_first_gap(tmp_path):
    p = RunPaths(name="exp", runs_root=str(tmp_path / "runs")).ensure()
    for n in (1, 2, 5):
        _touch(tmp_path / "runs" / "exp" / f"madqn_final_{n}.pth")
    assert p.next_final_model("madqn").endswith("madqn_final_6.pth")


def test_prune_only_numbered_final_and_keeps_latest(tmp_path):
    p = RunPaths(name="exp", runs_root=str(tmp_path / "runs")).ensure()
    for n in (1, 2, 3, 4, 5):
        _touch(tmp_path / "runs" / "exp" / f"madqn_final_{n}.pth")
    for name in ("madqn_final.pth", "madqn_best.pth", "madqn_pretrain.pth", "notes.txt"):
        _touch(tmp_path / "runs" / "exp" / name)

    removed = p.prune_final_models("madqn", keep=2)

    assert sorted(removed) == sorted(
        str(tmp_path / "runs" / "exp" / f"madqn_final_{n}.pth") for n in (1, 2, 3)
    )
    left = {q.name for q in (tmp_path / "runs" / "exp").iterdir()}
    assert {"madqn_final_4.pth", "madqn_final_5.pth"} <= left
    assert {"madqn_final.pth", "madqn_best.pth", "madqn_pretrain.pth", "notes.txt"} <= left


def test_prune_with_zero_keeps_everything(tmp_path):
    p = RunPaths(name="exp", runs_root=str(tmp_path / "runs")).ensure()
    _touch(tmp_path / "runs" / "exp" / "madqn_final_1.pth")
    assert p.prune_final_models("madqn", keep=0) == []
    assert (tmp_path / "runs" / "exp" / "madqn_final_1.pth").exists()
