# 🕒 v2 · 2026-10-24 13:30 · dsh-agent ｜ 变更：⑨ 改为：无 rlinfra 时显式 SKIP（镜像副本），格式器坏掉仍判 FAIL ｜ 上一版：v1 · 2026-10-24 13:22


"""A1 自检（≥8 项）：`python kernel/selfcheck_kv.py` 全绿才算 A1 交付

设计纪律：自检**不依赖随机**（除第 7 项固定 seed 的统计项），失败即退出码非 0。
第 ⑨ 项是**跨文件互证**：`kernel/kvlog.py` 与 `rlinfra/logger.py` 的两种行前缀必须逐字相同。
"""
from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kvpager import BlockManager, OutOfBlocks, theoretical_internal_frag  # noqa: E402

_ok = 0
_fail = []

def check(name, cond, detail=""):
    global _ok
    if cond:
        _ok += 1
        print(f"✅ {name}  {detail}")
    else:
        _fail.append(name)
        print(f"❌ {name}  {detail}")

def main():
    # ① 分配/释放往返：释放后块数回到初始
    bm = BlockManager(num_blocks=64, block_size=16)
    s = bm.new_sequence(50)                      # ceil(50/16)=4 块
    check("① 分配块数 = ceil(tokens/block_size)", len(bm._table[s]) == 4, f"{len(bm._table[s])} 块")
    bm.free(s)
    check("① 释放后空闲块复原", len(bm._free) == 64, f"free={len(bm._free)}")

    # ② 块复用：连续分配/释放 200 次不泄漏
    for i in range(200):
        sid = bm.new_sequence(10 + i % 40)
        bm.free(sid)
    check("② 200 次分配/释放无泄漏", len(bm._free) == 64 and not bm._table, f"free={len(bm._free)}")

    # ③ 边界：恰好整数倍不多占块
    sid = bm.new_sequence(64)                    # 16*4
    check("③ 整数倍不浪费块", len(bm._table[sid]) == 4, f"{len(bm._table[sid])} 块")
    bm.free(sid)

    # ④ 前缀共享：fork 后共享块，引用计数正确
    base = bm.new_sequence(64)
    child = bm.fork_prefix(base)
    st = bm.stats()
    check("④ fork 共享前缀（不新增块）", st["allocated_blocks"] == 4, f"allocated={st['allocated_blocks']}")

    # ⑤ 释放子序列不影响父序列的表
    bm.free(child)
    check("⑤ 释放子序列后父序列块仍在", len(bm._table[base]) == 4 and len(bm._free) == 60,
          f"free={len(bm._free)}")

    # ⑥ 容量耗尽抛 OutOfBlocks，且失败无副作用（块没被吃掉）
    bm2 = BlockManager(num_blocks=2, block_size=16)
    bm2.new_sequence(32)                          # 占满 2 块
    try:
        bm2.new_sequence(16)
        raised = False
    except OutOfBlocks:
        raised = True
    check("⑥ 容量耗尽抛 OutOfBlocks 且不泄漏", raised and len(bm2._free) == 0,
          f"raised={raised} free={len(bm2._free)}")

    # ⑦ 内部碎片率与闭式近似一致（误差 <2 个百分点），固定 seed
    rng = random.Random(0)
    bm3 = BlockManager(num_blocks=80000, block_size=16)
    lens = [rng.randint(20, 300) for _ in range(3000)]
    for n in lens:
        bm3.new_sequence(n)
    st = bm3.stats()
    mean_len = sum(lens) / len(lens)
    theo = theoretical_internal_frag(16, mean_len)      # 修正式 w/(m+w)
    err = abs(st["internal_frag"] - theo)
    check("⑦ 内部碎片率 vs 修正式 w/(m+w)（<2%）", err < 0.02,
          f"实测 {st['internal_frag']:.4f} vs 理论 {theo:.4f}（Δ={err:.4f}）")

    # ⑧ 前缀共享收益可计量：N 条同前缀序列，省下的块 = (N−1)×前缀块数
    bm4 = BlockManager(num_blocks=1000, block_size=16)
    p = bm4.new_sequence(128)                     # 8 块前缀
    kids = [bm4.fork_prefix(p) for _ in range(9)]
    st4 = bm4.stats()
    check("⑧ 前缀共享省块数 = (N−1)×前缀块数", st4["shared_saved_blocks"] == 9 * 8,
          f"saved={st4['shared_saved_blocks']}（期望 {9*8}）")
    for k in kids:
        bm4.free(k)

    # ⑨ 行前缀互证：kernel/kvlog.py 必须与 rlinfra/logger.py 逐字相同
    #     · 在 RL_infra 仓内：必须 PASS
    #     · 镜像副本（如 opt 仓 tools/kernel_ref/，那边没有 rlinfra 包）：判 SKIP 并**显式打印**
    #       —— 跳过 ≠ 通过，所以不是静默略过，而是少一项、字数也对不上。
    try:
        from kvlog import batch_line as k_batch, kv_line as k_kv
        sys.path.insert(0, os.path.dirname(_HERE))          # 让 rlinfra 可导入
        try:
            from rlinfra.logger import format_batch_line as r_batch, format_kv_line as r_kv
        except ImportError:
            print("⏭ ⑨ 行前缀互证 SKIP —— 本目录没有 rlinfra 包（镜像副本），"
                  "本项只在 RL_infra 仓内有效")
        else:
            st5 = bm4.stats()
            res5 = {"steps": 12, "tokens": 96, "tokens_per_s": 1234.5, "idle_slots": 7}
            same = ((k_kv(st5) == r_kv(st5))
                    and (k_batch("continuous", res5) == r_batch("continuous", res5)))
            check("⑨ KV|/BATCH| 前缀与 rlinfra/logger.py 逐字一致", same,
                  k_kv(st5) if same else f"kernel: {k_kv(st5)} ｜ rlinfra: {r_kv(st5)}")
    except Exception as e:                                  # 格式器本身坏了：明说不通过
        check("⑨ KV|/BATCH| 前缀与 rlinfra/logger.py 逐字一致", False,
              f"互证失败：{type(e).__name__}: {e}")

    # ⑩ 封闭容量：恰好装下时一次成功，多要 1 token 就抛（边界不漏不溢）
    bm6 = BlockManager(num_blocks=10, block_size=8)
    sid6 = bm6.new_sequence(80)                   # 10×8 恰好装满
    ok_full = len(bm6._free) == 0
    try:
        bm6.new_sequence(1)
        over = False
    except OutOfBlocks:
        over = True
    check("⑩ 恰好装满成功 / 再多 1 token 必抛", ok_full and over,
          f"free={len(bm6._free)} over_raised={over}")
    bm6.free(sid6)

    print("-" * 60)
    if _fail:
        print(f"❌ {len(_fail)}/{_ok + len(_fail)} 项失败：{_fail}")
        return 1
    print(f"✅ 全部 {_ok} 项自检通过 —— A1 可以进 BENCH")
    return 0

if __name__ == "__main__":
    sys.exit(main())
