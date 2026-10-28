# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：A 阶段一键复现（Python 单实现，跨平台）

"""A 阶段一键复现（D1 分页 KV + D2 连续批）—— **跨平台单实现**

    python scripts/a_kv_bench.py             # 全跑
    python scripts/a_kv_bench.py --fast       # 跳过 A2 的 CV 扫描（省一半时间）
    python scripts/a_kv_bench.py --no-torch   # 没 torch 时只跑 A1
    bash   scripts/a_kv_bench.sh              # Linux 上的薄壳，等价于上面

为什么是 Python 而不是纯 shell：本机（Windows 开发机）没有可用的 bash，
shell 版没法本地验证 = 没法证明"一键复现"真的能复现。故逻辑只写在 Python 一处，
`.sh` 只是三行壳（给 481 / 高算用），两边不会走偏。

产物：runs/a1_internal_frag/ · runs/a1_internal_frag_domain/ · runs/a1_prefix_share/
      runs/a1_external_frag/ · runs/a2_batching/
退出码：0 = 全部闸门通过；1 = 有项未过（逐项打印 PASS/FAIL/SKIP）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

def run(name: str, want: int, argv: list[str]) -> tuple[str, bool]:
    print("")
    print(f"════════ {name} ════════", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY] + argv, cwd=ROOT, env=env)
    ok = r.returncode == want
    print(f"---- {name}：exit={r.returncode}（期望 {want}）"
          f"{'  ✅' if ok else '  ❌'}", flush=True)
    return (f"{'PASS' if ok else 'FAIL'}  {name}（exit={r.returncode}，期望 {want}）", ok)

def main() -> int:
    ap = argparse.ArgumentParser(description="A 阶段一键复现")
    ap.add_argument("--fast", action="store_true", help="跳过 A2 的 CV 边界扫描")
    ap.add_argument("--no-torch", action="store_true", help="跳过 A2（没装 torch 时）")
    args = ap.parse_args()

    print(f"仓库：{ROOT}")
    print(f"Python：{sys.version.split()[0]}（{PY}）")

    results: list[str] = []
    fails = 0

    def add(name: str, want: int, argv: list[str]) -> None:
        nonlocal fails
        line, ok = run(name, want, argv)
        results.append(line)
        if not ok:
            fails += 1

    add("A1 自检 kernel/selfcheck_kv.py（11 项，含行前缀互证）", 0,
        ["kernel/selfcheck_kv.py"])

    add("表1 内部碎片率（4 档 block_size × 3 分布）", 0,
        ["kernel/bench_internal_frag.py", "--block-sizes", "8,16,32,64",
         "--lens", "all", "--n", "3000", "--seed", "0"])

    # 探边命令故意越界，exit=1 是预期结果
    add("表1f 闭式式适用域探边（越界格存在 ⇒ exit=1 是预期）", 1,
        ["kernel/bench_internal_frag.py", "--block-sizes", "8,16,32,64,128,256,512",
         "--lens", "uniform:20,300", "--n", "3000", "--seed", "0",
         "--out", "runs/a1_internal_frag_domain"])

    add("表2 前缀共享收益（省块数 = (N−1)×⌈p/b⌉）", 0,
        ["kernel/bench_prefix_share.py", "--prefix", "128,512,1024",
         "--n", "10,50", "--suffix", "64", "--block-size", "16"])

    add("表3 外部碎片率（固定块分页 vs 变块首次适配）", 0,
        ["kernel/bench_external_frag.py", "--capacity", "200000,172000,165000",
         "--block-size", "16", "--steps", "400", "--seed", "0"])

    has_torch = subprocess.run([PY, "-c", "import torch"], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if args.no_torch or not has_torch:
        results.append("SKIP  表4（未装 torch；A1 不受影响）")
        print("\n════════ 表4 跳过：没有 torch ════════")
        print("安装：python -m pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
    else:
        argv = ["kernel/bench_batching.py", "--dist", "equal,highvar,heavy",
                "--max-batch", "8", "--n", "64", "--seed", "0"]
        if not args.fast:
            argv.append("--sweep-cv")
        add("表4 static vs continuous" + ("" if args.fast else " + CV 边界扫描"), 0, argv)

    add("库边界自检 utils/check_boundaries.py（含 kernel ⑧⑨）", 0,
        ["utils/check_boundaries.py"])

    print("")
    print("════════════════════════ 汇总 ════════════════════════")
    for line in results:
        print(line)
    print("-----------------------------------------------------")
    if fails == 0:
        print("✅ A 阶段全部闸门通过。数字出处见 kernel/BENCH.md 表 1~表 4 与 runs/ 下各目录。")
        return 0
    print(f"❌ {fails} 项未过 —— 按 kernel/BENCH.md 的判据逐项查，不许改口径凑数。")
    return 1

if __name__ == "__main__":
    sys.exit(main())
