# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：KV/BATCH 行前缀的 kernel 侧镜像格式器（与 logger.py 互证）

"""kernel 侧的两种行前缀（契约权威在 `rlinfra/logger.py`）

为什么这里还有一份：`kernel/` 要能 `python kernel/xxx.py` 独立跑，**不 import 项目基座**。
但日志前缀必须只有一处定义 —— 否则两边一改就散。

所以：**`rlinfra/logger.py` 是权威**，本文件是同构镜像；
`kernel/selfcheck_kv.py` 的 ⑨ 项逐字比对两边输出，**对不上就红**。
这条互证链和「日志契约 ↔ 分析脚本互证」是同一个做法。

格式（字段顺序固定，`|` 分隔，`k:v` 且值不带空格）：
    KV | op:<操作> | seqs:<在跑序列数> | alloc:<占用块> | free:<空闲块> | frag:<内部碎片率> | saved:<前缀共享省块>
    BATCH | mode:<static|continuous> | steps:<步数> | tokens:<产出 token> | tps:<tokens/s> | idle:<空转槽位数>
"""
from __future__ import annotations

from typing import Dict

def kv_line(stats: Dict) -> str:
    """A1 分页 KV 的一行统计。`stats` 即 `BlockManager.stats()` 的返回（可带 `op` 键）。"""
    return (f"KV | op:{stats.get('op', 'stats')} | seqs:{stats['seqs']} | "
            f"alloc:{stats['allocated_blocks']} | free:{stats['free_blocks']} | "
            f"frag:{stats['internal_frag']:.4f} | saved:{stats['shared_saved_blocks']}")

def batch_line(mode: str, result: Dict) -> str:
    """A2 批调度的一行结果。`result` 即 `batcher.run()` 的返回。"""
    return (f"BATCH | mode:{mode} | steps:{result['steps']} | tokens:{result['tokens']} | "
            f"tps:{result['tokens_per_s']:.1f} | idle:{result['idle_slots']}")
