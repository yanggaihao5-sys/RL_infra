# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：闭式式改修正式 w/(m+w)；旧式改名 naive 并写明失效域


"""分页 KV Cache 参考实现（A1 · 本文件是**可运行的参考实现**，不是骨架）

设计目标：把 PagedAttention 的**内存管理**部分单独抽出来，不依赖任何推理框架、
不依赖 GPU，纯 Python 即可运行与测量。这样"分页到底省了什么"可以**算出来**而不是猜出来。

核心概念（与 vLLM 一致的最小集合）：
  block      固定大小（block_size 个 token 槽）的存储单元；块内可以不装满
  block_table 每条序列持有的一串 block id（逻辑位置 → 物理块）
  refcount   块被几条序列引用（前缀共享时 >1）；减到 0 才回 free list

指标定义（`stats()` 返回）：
  internal_frag   已分配块中**未被使用**的槽位占比（块内碎片）—— 表 1 测的就是它
  holes           空闲块**个数**（⚠️ 是计数，不是碎片率！块大小固定时空闲粒度恒为整块，
                  所以"不足 block_size 的空洞占比"结构上恒为 0；外部碎片必须跨分配器对照才测得出，
                  见 `kernel/bench_external_frag.py` 与 `BENCH.md` 表 3）
  shared_saved    因前缀共享而**少分配**的块数

↔️ 与闭式式的关系：`internal_frag` 应逼近 `theoretical_internal_frag(b, mean_len) = w/(mean_len+w)`，
   **不是** `theoretical_internal_frag_naive`（旧式，仅在 mean_len/b ≥ 5 时可用）。

⚠️ 本实现刻意不做：copy-on-write、swap、preemption。做到这一步足以回答"分页 vs 连续分配"的问题，
   再多就是重写 vLLM —— 那属于读懂之后的事。
"""
from __future__ import annotations

import math
from typing import Dict, List

class OutOfBlocks(RuntimeError):
    pass

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks / block_size 必须为正")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self._free: List[int] = list(range(self.num_blocks))
        self._ref: Dict[int, int] = {i: 0 for i in range(self.num_blocks)}
        self._table: Dict[str, List[int]] = {}
        self._tokens: Dict[str, int] = {}
        self._seq = 0

    # ───────────────────────── 基本操作 ─────────────────────────
    def _blocks_needed(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def new_sequence(self, n_tokens: int, prefix_from: str | None = None) -> str:
        """建一条序列；`prefix_from` 给了就共享那条序列的**全部**块（前缀共享的最小版）"""
        if n_tokens < 0:
            raise ValueError("n_tokens 不能为负")
        self._seq += 1
        sid = f"s{self._seq}"
        if prefix_from is not None:
            base = self._table[prefix_from]
            for b in base:
                self._ref[b] += 1
            extra = max(0, self._blocks_needed(n_tokens) - len(base))
            need = extra
        else:
            base, need = [], self._blocks_needed(n_tokens)
        if need > len(self._free):
            # 回滚已加的引用，保证失败不留副作用
            for b in base:
                self._ref[b] -= 1
            raise OutOfBlocks(f"需要 {need} 块，只剩 {len(self._free)}")
        take = [self._free.pop() for _ in range(need)]
        for b in take:
            self._ref[b] += 1
        self._table[sid] = base + take
        self._tokens[sid] = int(n_tokens)
        return sid

    def free(self, sid: str) -> None:
        for b in self._table.pop(sid):
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)
        self._tokens.pop(sid, None)

    def fork_prefix(self, src: str) -> str:
        """等价于 new_sequence(同长度, prefix_from=src)"""
        return self.new_sequence(self._tokens[src], prefix_from=src)

    # ───────────────────────── 指标 ─────────────────────────
    def stats(self) -> dict:
        allocated_blocks = sum(1 for v in self._ref.values() if v > 0)   # 被占用（含共享）的**不同**块数
        used_tokens = sum(self._tokens.values())            # 逻辑 token 总数
        capacity = allocated_blocks * self.block_size
        internal_frag = (capacity - used_tokens) / capacity if capacity else 0.0

        # 前缀共享省了多少：不共享时每条序列都要自己的块
        naive = sum(self._blocks_needed(n) for n in self._tokens.values())
        shared_saved = max(0, naive - allocated_blocks)

        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "seqs": len(self._table),
            "allocated_blocks": allocated_blocks,
            "free_blocks": len(self._free),
            "holes": len(self._free),                       # 块大小固定 ⇒ 空闲块即"不可再切"的碎片
            "used_tokens": used_tokens,
            "internal_frag": round(internal_frag, 6),
            "shared_saved_blocks": shared_saved,
        }

def theoretical_internal_frag(block_size: int, mean_len: float) -> float:
    """闭式近似（**修正式**）：内部碎片 ≈ w / (mean_len + w)，w = (block_size − 1) / 2

    推导：序列长度 n 对 b 取模近似均匀 ⇒ 最后一块平均浪费 E[waste] = (b−1)/2 = w；
    每条序列平均占用**容量** E[cap] = E[⌈n/b⌉·b] ≈ mean_len + w；
    内部碎片率 ≡ Σ浪费 / Σ容量 = w / (mean_len + w)。

    ⚠️ 分母必须是**分配容量**，不是逻辑长度 —— 这正是旧式（见下）错的地方。

    适用域（实测，见 `BENCH.md` 表 1f）：需要 `mean_len/b ≳ 0.6`（长度支撑集至少盖住 1 个块宽）；
    当 block_size 超过最长序列时 `n mod b` 退化，本式会偏小，应改为按实测分布直接求和。
    """
    if mean_len <= 0:
        return 0.0
    w = (block_size - 1) / 2.0
    return w / (mean_len + w)

def theoretical_internal_frag_naive(block_size: int, mean_len: float) -> float:
    """闭式近似（**旧式，仅作对照**）：内部碎片 ≈ (block_size − 1) / (2·mean_len)

    ⚠️ 它漏掉了分母里的 w，等价于假设 `mean_len ≫ block_size`；
    实测只在 `mean_len/block_size ≥ 5` 时误差 <2 个百分点，b=64 时可差 **3.1 个百分点**。
    **不要拿它当判据基准**，保留它只为让 `BENCH.md` 表 1 的"原式 vs 修正式"可复现。
    """
    return (block_size - 1) / (2.0 * mean_len) if mean_len > 0 else 0.0
