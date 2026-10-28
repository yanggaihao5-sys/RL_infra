# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-22 17:07 · dsh-agent ｜ 变更：新增：选优判据对照（coverage vs capture_steps，验收缺陷 #4）

"""选优判据对照：同一批 checkpoint，用**两种判据**各选一次 best，看它俩选的是不是同一个模型。

为什么需要（2026-10-17 修复 #10 的验收）：
  训练期"选 best"的判据原来写死 coverage。追逃域里这个判据会选出"扫得最多"的模型，
  而不是"抓得最快"的模型。本脚本把这件事**事后量出来**：
  对同一组 checkpoint 逐个跑 20 图评测，然后分别按 `coverage` 与 `capture_steps` 选优，比较结果。

用法：
    python scripts/compare_ckpt_selection.py --run pursuit_drill --maps 20
    python scripts/compare_ckpt_selection.py --run pursuit_drill --chain pursuit --map-name pursuit

判据（跑之前写死）：
  · 两种判据选出的 checkpoint 不同，且 `capture_steps` 选出的那个 20 图捕获率更高 ⇒ 缺陷 #4 成立（修复有意义）；
  · 两者相同 ⇒ 这个 run 上该缺陷**没有实际后果**（也只是"这次没踩到"）。

⚠️ 本脚本只读 checkpoint + 跑评测，**不改** runs/<run>/ 下的 best，也不写 eval.json。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config                                  # noqa: E402
from rlinfra.logger import TrainLogger                     # noqa: E402
from rlinfra.paths import RunPaths                         # noqa: E402
from rlinfra.registry import get_algo                      # noqa: E402


def _checkpoints(run_dir: str, tag: str) -> List[str]:
    best = os.path.join(run_dir, f"{tag}_best.pth")
    finals = sorted(glob.glob(os.path.join(run_dir, f"{tag}_final_*.pth")))
    return ([best] if os.path.exists(best) else []) + finals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="选优判据对照（coverage vs capture_steps）")
    ap.add_argument("--run", required=True, help="实验名，如 pursuit_drill")
    ap.add_argument("--chain", default="pursuit", help="链路名（默认 pursuit）")
    ap.add_argument("--maps", type=int, default=20)
    ap.add_argument("--out", default="", help="把结果另存成 json（可选）")
    args = ap.parse_args(argv)

    cfg = Config(test_mode=args.chain)
    cfg.name = args.run
    cfg._derive()
    cfg.validate()
    paths = RunPaths.from_config(cfg)
    run_dir = paths.root
    tag = args.chain
    ckpts = _checkpoints(run_dir, tag)
    if not ckpts:
        print(f"❌ {run_dir} 下没有 {tag}_best.pth / {tag}_final_*.pth")
        return 2

    evaluate = get_algo(args.chain).evaluate
    logger = TrainLogger(os.path.join(run_dir, "_select_compare.log"), algo=args.chain.upper())
    rows: List[Dict[str, Any]] = []
    try:
        for p in ckpts:
            r = evaluate(cfg, paths, logger, runs=args.maps, model_path=p)
            rows.append({"ckpt": os.path.basename(p),
                         "capture_rate": float(r.get("success_rate", 0.0)),
                         "steps": float(r.get("steps", 0.0)),
                         "coverage": float(r.get("coverage", 0.0)),
                         "coverage_std": float(r.get("coverage_std", 0.0))})
    finally:
        logger.close()

    print(f"\n{'checkpoint':<24}{'捕获率':>8}{'平均步数':>10}{'覆盖':>9}{'±std':>8}")
    for r in rows:
        print(f"{r['ckpt']:<24}{r['capture_rate']:>7.0f}%{r['steps']:>10.1f}"
              f"{r['coverage']:>8.2f}%{r['coverage_std']:>8.2f}")

    # 两种判据各自会选谁（与 trainer 里的 probe_key 同构，但这里用 20 图结果当"探针"）
    by_cov = max(rows, key=lambda r: r["coverage"])
    by_cap = max(rows, key=lambda r: (1 if r["capture_rate"] > 0 else 0,
                                      r["capture_rate"], -r["steps"]))
    print(f"\n按 coverage 选      → {by_cov['ckpt']}（捕获率 {by_cov['capture_rate']:.0f}%）")
    print(f"按 capture 选        → {by_cap['ckpt']}（捕获率 {by_cap['capture_rate']:.0f}%，"
          f"{by_cap['steps']:.1f} 步）")
    same = by_cov["ckpt"] == by_cap["ckpt"]
    gain = by_cap["capture_rate"] - by_cov["capture_rate"]
    print(f"判定：{'两者相同' if same else f'选出的不是同一个模型，捕获率差 {gain:+.0f}pp'}"
          f" → {'缺陷 #4 在本 run 无后果' if same else '缺陷 #4 成立'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"run": args.run, "maps": args.maps, "rows": rows,
                       "picked_by_coverage": by_cov["ckpt"],
                       "picked_by_capture": by_cap["ckpt"]}, f, ensure_ascii=False, indent=1)
        print(f"已写 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
