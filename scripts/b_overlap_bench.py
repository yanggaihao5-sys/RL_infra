# 🕒 v1 · 2026-10-24 14:08 · dsh-agent ｜ 变更：B 阶段一键跑（结构自检 + 本机 CPU 基线），GPU 数在高算出

"""B 阶段一键跑（跨平台单实现）

    python scripts/b_overlap_bench.py                 # 默认 --device cpu（本机只验结构）
    python scripts/b_overlap_bench.py --device cuda    # 高算 GPU 出数
    python scripts/b_overlap_bench.py --quick          # 极小规模，只证明结构跑通

为什么和 `a_kv_bench.py` 一样是 Python：本机没有可用 bash，纯 shell 版没法本地验证。
`b_overlap_bench.sh` 是三行薄壳，给 481 / 高算用。

退出码：0 = 结构跑通（**不对加速比大小做通过判定**，那是数据结论）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description="B 阶段一键跑")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="极小规模，只验结构")
    args = ap.parse_args()

    steps, batch = (4, 2) if args.quick else (20, 8)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    argv = [PY, "traintest/bench_overlap.py", "--device", args.device,
            "--seeds", str(args.seeds), "--steps", str(steps), "--batch", str(batch)]

    print(f"仓库：{ROOT}")
    print(f"Python：{sys.version.split()[0]}（{PY}）")
    print(f"命令：{' '.join(argv[1:])}")
    print("")
    r = subprocess.run(argv, cwd=ROOT, env=env)
    print("")
    if r.returncode != 0:
        print(f"❌ 结构未跑通（exit={r.returncode}）")
        return r.returncode
    print("✅ B 阶段结构跑通。加速比与归因见 runs/b_overlap/run.log")
    if args.device == "cpu":
        print("   ⚠️ 本次是 CPU：overlap 结构性更慢（GIL），不是实现错；"
              "「值不值」要上 GPU 回答。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
