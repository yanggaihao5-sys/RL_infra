# -*- coding: utf-8 -*-
# 🕒 v2 · 2026-09-30 22:01 · dsh-agent ｜ 变更：同步 eval.json（否则评测结果进不了平台）+ 输出显示测试覆盖率 ｜ 上一版：v1 · 2026-09-30 21:08

"""一条命令把高算的 run 拉回本地并接进观测平台（runs/obs/）。

为什么需要（2026-09-30）：
    观测平台只读**本地** `runs/obs/`，而训练产物全在高算。
    "刷新页面没反应"的根因就是这个 —— 缺的不是刷新，是**搬运**。
    本脚本把「找 run → 拉日志 → 转契约 → 落 runs/obs」串成一条命令。

流程（按需分步，每一步都可单独跑）：
    1) 探测远端 runs/ 下有哪些实验目录（`ssh ... ls -d runs/*/`）
    2) 每个 run 只拉 `train.log`（几 MB），**不拉 checkpoint**（几百 MB 且平台不读）
    3) 调 `utils/log_to_metrics.py` 转成 run.json / metrics.jsonl / events.jsonl
    4) 产物落 `<本地 runs 根>/obs/<run_id>/`，平台刷新即出现

用法：
    python utils/sync_hpc_runs.py                 # 同步全部 run
    python utils/sync_hpc_runs.py --only lr1e3_s2024 lr5e4_s2024
    python utils/sync_hpc_runs.py --dry-run       # 只列远端有哪些 run，不下载
    python utils/sync_hpc_runs.py --skip-existing # 本地已有 run.json 的跳过

🔴 边界：**只读远端**（`ls` / `cat` 日志）；只写本地 `runs/obs/` 与临时目录。
    不删远端任何东西，不动 checkpoint。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SSH_KEY = os.environ.get("RLI_SSH_KEY", "C:/Users/__user__/.ssh/id_ed25519")
SSH_HOST = os.environ.get("RLI_SSH_HOST", "__HPC_ACCOUNT__@__HPC_LOGIN_IP__")
REMOTE_PROJ = os.environ.get("RLI_REMOTE_PROJ", "/storage/public/home/__HPC_ACCOUNT__/RL_infra")


def ssh(cmd: str, timeout: int = 60) -> str:
    """在登录节点执行一条命令（**必须 bash -lc**，否则平台 CLI 环境变量不全）。"""
    full = ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", f"ConnectTimeout=20",
            SSH_HOST, f"bash -lc '{cmd}'"]
    r = subprocess.run(full, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh 失败（{r.returncode}）：{(r.stderr or '').strip()[:300]}")
    return r.stdout


def scp_down(remote: str, local: str, timeout: int = 300) -> None:
    os.makedirs(os.path.dirname(local), exist_ok=True)
    full = ["scp", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            f"{SSH_HOST}:{remote}", local]
    r = subprocess.run(full, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scp 失败（{r.returncode}）：{(r.stderr or '').strip()[:300]}")


def list_remote_runs() -> List[str]:
    out = ssh(f"cd {REMOTE_PROJ}/runs && ls -d */ 2>/dev/null | tr -d '/'")
    names = [x.strip() for x in out.splitlines() if x.strip()]
    return [n for n in names if re.fullmatch(r"[\w.\-]+", n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只同步这些 run_id")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "obs"), help="本地观测根目录")
    ap.add_argument("--tmp", default=os.path.join(ROOT, ".tmp", "sync"), help="下载暂存目录")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="本地已有 run.json 就跳过")
    args = ap.parse_args()

    print(f"远端：{SSH_HOST}:{REMOTE_PROJ}/runs")
    runs = list_remote_runs()
    if args.only:
        runs = [r for r in runs if r in set(args.only)]
    print(f"发现 {len(runs)} 个 run：{', '.join(runs) if runs else '（无）'}")

    if args.dry_run or not runs:
        return 0

    ok = skipped = failed = 0
    for rid in runs:
        dst_dir = os.path.join(args.out, rid)
        if args.skip_existing and os.path.exists(os.path.join(dst_dir, "run.json")):
            print(f"  [SKIP] {rid}（本地已有 run.json）")
            skipped += 1
            continue
        log_local = os.path.join(args.tmp, rid, "train.log")
        try:
            scp_down(f"{REMOTE_PROJ}/runs/{rid}/train.log", log_local)
        except Exception as e:                       # noqa: BLE001
            print(f"  [FAIL] {rid}：拉日志失败 —— {e}")
            failed += 1
            continue
        size_kb = os.path.getsize(log_local) / 1024
        # 🔴 eval.json 必须一起拉（2026-09-30 实测教训）：转换器优先用它写 test/* 指标；
        #    只拉 train.log 的话，`--mode test` 的评测结果永远进不了平台。
        try:
            scp_down(f"{REMOTE_PROJ}/runs/{rid}/eval.json", os.path.join(args.tmp, rid, "eval.json"))
            eval_tag = " +eval"
        except Exception:                            # noqa: BLE001
            eval_tag = ""                            # 没评测过就没有这个文件，属正常
        # 转换（复用项目自己的转换器，保证契约只有一处实现）
        conv = subprocess.run([sys.executable, os.path.join(ROOT, "utils", "log_to_metrics.py"),
                               "--log", log_local, "--out", args.out, "--run-id", rid, "--force"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if conv.returncode != 0:
            print(f"  [FAIL] {rid}：转换失败 —— {(conv.stdout or conv.stderr).strip()[:200]}")
            failed += 1
            continue
        # 回读校验：契约三件套是否都在
        missing = [f for f in ("run.json", "metrics.jsonl", "events.jsonl")
                   if not os.path.exists(os.path.join(dst_dir, f))]
        if missing:
            print(f"  [FAIL] {rid}：产物缺 {missing}")
            failed += 1
            continue
        meta = json.load(open(os.path.join(dst_dir, "run.json"), encoding="utf-8"))
        cov = (meta.get("eval") or {}).get("coverage")
        cov_tag = f" / 测试覆盖 {cov}%" if cov is not None else ""
        print(f"  [OK] {rid}  日志 {size_kb:.0f}KB{eval_tag} → {meta.get('metric_count')} 条指标 / "
              f"算法={meta.get('algorithm')} / 步数={meta.get('total_timesteps')}{cov_tag}")
        ok += 1

    print(f"\n同步完成：成功 {ok} · 跳过 {skipped} · 失败 {failed}")
    print(f"观测平台数据源：{args.out}（刷新 http://127.0.0.1:8077 即出现）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
