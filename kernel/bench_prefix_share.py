# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：A1 表2：前缀共享收益（省块数=(N-1)×前缀块数）

"""A1 表2：前缀共享收益

场景：一条共享前缀（长度 p）被 N 条请求复用，每条请求 = 前缀 + 私有后缀（长度 s）。
    父序列只存前缀（相当于"前缀缓存条目"），N 条请求全部 fork 它再接后缀。

    不共享：N 条请求各自从零存              ⇒ N × ⌈(p+s)/b⌉
    共享后：前缀只存一份 + 每条各自加后缀    ⇒ ⌈p/b⌉ + N × (⌈(p+s)/b⌉ − ⌈p/b⌉)
    省下   = (N − 1) × ⌈p/b⌉                 ← 闭式；脚本逐格核对，不等就报错并 exit 1

用法：
    python kernel/bench_prefix_share.py --prefix 128,512,1024 --n 10,50 --block-size 16

产物：runs/a1_prefix_share/{prefix_share.csv, run.log}；退出码 0 = 逐格吻合。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvpager import BlockManager  # noqa: E402

FIELDS = ["prefix_len", "suffix_len", "N", "block_size", "no_share_blocks",
          "shared_blocks", "saved_blocks", "saved_ratio", "expected_saved", "match"]

def run_case(prefix_len: int, suffix_len: int, n: int, block_size: int) -> dict:
    total_len = prefix_len + suffix_len
    per_seq_noshare = math.ceil(total_len / block_size)

    # 共享路径：父序列只存前缀（前缀缓存条目），N 条请求全部 fork 它再接后缀
    parent_blocks = math.ceil(prefix_len / block_size)
    suffix_blocks = math.ceil(total_len / block_size) - parent_blocks   # fork 后每条新增的块
    need = parent_blocks + n * suffix_blocks
    bm = BlockManager(num_blocks=need + 8, block_size=block_size)

    p = bm.new_sequence(prefix_len)                      # 父序列 = 纯前缀
    kids = [bm.new_sequence(total_len, prefix_from=p) for _ in range(n)]
    st = bm.stats()

    no_share = n * per_seq_noshare                       # N 条请求各自从零存
    shared = st["allocated_blocks"]
    saved = no_share - shared
    expected = (n - 1) * parent_blocks                   # 闭式
    for k in kids:
        bm.free(k)
    bm.free(p)
    return {
        "prefix_len": prefix_len, "suffix_len": suffix_len, "N": n, "block_size": block_size,
        "no_share_blocks": no_share,
        "shared_blocks": shared,
        "saved_blocks": saved,
        "saved_ratio": round(saved / no_share, 4),
        "expected_saved": expected,
        "match": "OK" if saved == expected else f"MISMATCH({saved}!={expected})",
    }

def main() -> int:
    ap = argparse.ArgumentParser(description="A1 表2：前缀共享收益")
    ap.add_argument("--prefix", default="128,512,1024", help="共享前缀长度（token）")
    ap.add_argument("--n", default="10,50", help="同前缀序列数 N（逗号分隔）")
    ap.add_argument("--suffix", type=int, default=64, help="每条序列的私有后缀长度")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out", default=os.path.join("runs", "a1_prefix_share"))
    args = ap.parse_args()

    block_size = args.block_size
    prefixes = [int(x) for x in args.prefix.split(",")]
    ns = [int(x) for x in args.n.split(",")]

    rows = [run_case(p, args.suffix, n, block_size) for p in prefixes for n in ns]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "prefix_share.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append(f"# 命令：python kernel/bench_prefix_share.py --prefix {args.prefix} "
                 f"--n {args.n} --suffix {args.suffix} --block-size {block_size}")
    lines.append(f"# 后缀固定 {args.suffix} token；块大小 b={block_size}")
    lines.append("")
    lines.append("| 前缀 token | 同前缀序列数 N | 不共享所需块 | 共享后实际块 | 省下块数 | 省下比例 | 闭式 (N−1)×⌈p/b⌉ | 核对 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['prefix_len']} | {r['N']} | {r['no_share_blocks']} | {r['shared_blocks']} | "
                     f"{r['saved_blocks']} | {r['saved_ratio']:.3f} | {r['expected_saved']} | "
                     f"{'✅' if r['match'] == 'OK' else '❌ ' + r['match']} |")
    bad = [r for r in rows if r["match"] != "OK"]
    lines.append("")
    lines.append(f"## 判定：{len(rows) - len(bad)}/{len(rows)} 格省块数与闭式 (N−1)×⌈p/b⌉ 吻合"
                 + ("" if not bad else " ❌ 有不吻合"))
    lines.append("· 机制：共享把**前缀块**由 N 份压成 1 份，省下的量与后缀无关；"
                 "后缀每条约 ⌈(p+s)/b⌉−⌈p/b⌉ 块，本来就不共享，省不掉。")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "run.log"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(main())
