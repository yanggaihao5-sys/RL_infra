# -*- coding: utf-8 -*-
"""CI 产物闸门：训练 → 评测 → 诊断 → 出图 → 观测契约，五件产物齐且数字自洽。

用法：
    python utils/verify_run.py --run runs/ci_smoke --obs runs/obs/ci_smoke

退出码：0=全过；1=有失败项。这个脚本只读产物，不跑训练、不改任何文件。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Tuple


def _read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_run(run_dir: str, obs_dir: Optional[str] = None) -> List[Tuple[str, bool, str]]:
    checks: List[Tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # ① 训练日志：存在、非空、含契约行
    log = os.path.join(run_dir, "train.log")
    ok = os.path.isfile(log) and os.path.getsize(log) > 0
    n_ep = 0
    if ok:
        with open(log, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        n_ep = sum(1 for line in text.splitlines() if line.startswith("EP |"))
        ok = "链路标识" in text and n_ep > 0
    add("① train.log（链路标识 + EP 行）", ok, f"{n_ep} 行 EP" if ok else "缺失或格式不符")

    # ② 评测结果：eval.json 必须是真评测（不是 skipped）
    ev_path = os.path.join(run_dir, "eval.json")
    ev = {}
    if os.path.isfile(ev_path):
        try:
            ev = _read_json(ev_path)
        except (json.JSONDecodeError, OSError) as exc:
            ev = {"__error__": str(exc)}
    ev_ok = bool(ev) and not ev.get("skipped") and all(
        isinstance(ev.get(k), (int, float)) for k in ("coverage", "success_rate", "steps")
    ) and bool(ev.get("success_rule"))
    add("② eval.json（真评测，非 skipped）", ev_ok,
        f"coverage={ev.get('coverage')} success_rule={ev.get('success_rule')}" if ev_ok
        else f"内容={ev}")

    # ③ 最优模型
    best = glob.glob(os.path.join(run_dir, "*_best.pth"))
    add("③ *_best.pth", bool(best) and os.path.getsize(best[0]) > 0,
        os.path.basename(best[0]) if best else "没有 best 模型")

    # ④ 训练曲线图
    pngs = [p for p in glob.glob(os.path.join(run_dir, "*_training_metrics_en.png"))
            if os.path.getsize(p) > 10_000]
    add("④ *_training_metrics_en.png", bool(pngs),
        os.path.basename(pngs[0]) if pngs else "没有有效 PNG（>10KB）")

    # ⑤ 观测契约
    if obs_dir is None:
        obs_dir = os.path.join(os.path.dirname(run_dir), "obs", os.path.basename(run_dir))
    run_json = os.path.join(obs_dir, "run.json")
    metrics = os.path.join(obs_dir, "metrics.jsonl")
    events = os.path.join(obs_dir, "events.jsonl")
    meta = {}
    metric_rows = []
    if os.path.isfile(run_json):
        try:
            meta = _read_json(run_json)
        except (json.JSONDecodeError, OSError):
            meta = {}
    if os.path.isfile(metrics):
        with open(metrics, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    metric_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    metric_rows = []
                    break
    obs_ok = (meta.get("status") == "finished" and bool(metric_rows)
              and os.path.isfile(events))
    add("⑤ 观测契约（run.json/metrics.jsonl/events.jsonl）", obs_ok,
        f"{len(metric_rows)} 条指标" if obs_ok else f"obs_dir={obs_dir}")

    # ⑥ 评测数字与观测契约一致（防两条链路各写一份、对不上）
    same = False
    if ev_ok and obs_ok:
        got = [m["value"] for m in metric_rows if m.get("name") == "test/coverage"]
        same = bool(got) and abs(float(got[-1]) - float(ev["coverage"])) < 1e-6
    add("⑥ test/coverage 与 eval.json 一致", same,
        f"{ev.get('coverage')}" if same else "缺 test/coverage 或数值不一致")

    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CI 产物闸门")
    ap.add_argument("--run", required=True, help="runs/<name> 目录")
    ap.add_argument("--obs", default=None, help="观测目录（默认 <runs>/obs/<name>）")
    args = ap.parse_args(argv)

    run_dir = os.path.abspath(args.run)
    checks = check_run(run_dir, os.path.abspath(args.obs) if args.obs else None)
    failed = 0
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} 项通过 —— "
          + ("产物链路完好" if failed == 0 else "产物链路有断点"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
