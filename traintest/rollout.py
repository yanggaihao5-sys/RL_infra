# 🕒 v2 · 2026-10-25 20:14 · dsh-agent ｜ 变更：make_pair 支持 device_rollout 双设备 ｜ 上一版：v1 · 2026-10-24 13:45

"""B1 · rollout 引擎接口与同步实现（训推一体最小系统）

## 为什么需要接口而不是直接调模型

手册 §3 B1 要的第一条是**引擎可插拔**：`generate(states, max_new_tokens) -> actions` 定住之后，
「torch 本地生成」和「vLLM server」可以互换，而**测 overlap 的代码一行不改**。
所以本文件只定义协议 + 一个 torch 实现，不掺 pipeline 逻辑（那在 `pipeline.py`）。

## 与 A2 的关系

A2 用 toy decoder 只为调度；这里的 `TinyPolicy` 要**同时**被 rollout 和 trainer 使用
（训推一体的前提是同一套权重），所以它是一个真的 decoder block 栈（带因果自注意力），
只是体积被压到很小 —— 目的仍是**测结构，不是测效果**。

## 权重同步

`WeightSync.broadcast()` 把 trainer 的参数拷进 rollout 的**独立副本**。
用副本而不是共享，代价是策略滞后一步（本项目 `AGENTS.md` §6 已记录这条代价），
换来的是无数据竞争 —— overlap 档能跑起来的前提。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# ───────────────────────── 被训/被推的同一套权重 ─────────────────────────
class TinyPolicy(nn.Module):
    """最小因果解码器：TokenEmb + 2 层 pre-LN 自注意力 + lm_head

    成本 ∝ batch × seq（有真注意力，故 GPU 上能看到利用率变动），但参数量很小。
    """

    def __init__(self, vocab: int = 64, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 64):
        super().__init__()
        self.vocab, self.d_model, self.max_len = vocab, d_model, max_len
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(b, t)
        h = self.emb(idx) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(h)


# ───────────────────────── 引擎协议 ─────────────────────────
class RolloutEngine:
    """rollout 引擎协议。换成 vLLM server 时只需另写一个子类，pipeline 不改。"""

    name = "base"

    def generate(self, states: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        raise NotImplementedError

    def load_weights(self, src_state: Dict[str, torch.Tensor]) -> float:
        """把外部权重装进来；返回搬运的字节数（瓶颈归因要用）"""
        raise NotImplementedError


class TorchRolloutEngine(RolloutEngine):
    """实现 1：torch 本地生成（batch 8~32）。同步实现，先跑通再谈 overlap。"""

    name = "torch_local"

    def __init__(self, model: TinyPolicy, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def generate(self, states: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """states: (B, prompt_len) int64。逐 token 自回归到 max_new_tokens 步。"""
        idx = states.to(self.device)
        for _ in range(max_new_tokens):
            logits = self.model(idx)                     # (B, T, V)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx[:, states.shape[1]:].detach()

    def load_weights(self, src_state: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            self.model.load_state_dict(src_state)
        return float(sum(v.numel() * v.element_size() for v in src_state.values()))


# ───────────────────────── 权重同步 ─────────────────────────
class WeightSync:
    """trainer → rollout 的权重广播；记录搬运量与耗时可分别归因。"""

    def __init__(self, trainer: nn.Module, engine: RolloutEngine):
        self.trainer = trainer
        self.engine = engine
        self.calls = 0
        self.bytes_moved = 0.0
        self.seconds = 0.0

    def broadcast(self) -> float:
        t0 = time.perf_counter()
        state = {k: v.detach() for k, v in self.trainer.state_dict().items()}
        nbytes = self.engine.load_weights(state)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.bytes_moved += nbytes
        self.seconds += dt
        return dt

    def summary(self) -> Dict[str, float]:
        return {"calls": self.calls, "bytes": self.bytes_moved, "seconds": self.seconds}


# ───────────────────────── 工具 ─────────────────────────
def make_pair(seed: int = 0, device: str = "cpu", device_rollout: str | None = None,
              **kw) -> tuple[TinyPolicy, TinyPolicy, TorchRolloutEngine]:
    """建一对**同构但独立**的权重：trainer 与 rollout（副本，故允许滞后一步）

    `device` = 训练设备；`device_rollout` = rollout 设备。
    两者给**不同**的卡就是「推/训分离」—— 生成与训练不再抢同一块 GPU。
    """
    torch.manual_seed(seed)
    trainer = TinyPolicy(**kw).to(device)
    rollout = TinyPolicy(**kw)
    rollout.load_state_dict({k: v.detach().cpu() for k, v in trainer.state_dict().items()})
    return trainer, rollout, TorchRolloutEngine(rollout, device=device_rollout or device)


def sample_states(batch: int, prompt_len: int, vocab: int, device: str = "cpu",
                  seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, prompt_len), generator=g).to(device)
