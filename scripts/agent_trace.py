# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-01 11:04 · dsh-agent ｜ 变更：新增：agent 轨迹记录契约（TRACE 行前缀 + 观测契约）

"""Agent 轨迹记录契约（trajectory trace）：把「跑了一次 agent」变成可归因的证据。

🔴 为什么必须有它（`docs/实验_上下文注入AB_设计.md` §4.3）
    "人肉跑 = 不可复现" —— 手工开三次会话就没有可比性。
    轨迹是 agent harness 的**唯一**证据来源：出事能回放，改版能对比，出结论能引用。

两条输出（与 RL_infra「产物隔离 + 日志契约」一致）
    ① 人类可读 `<out>/<run_id>.log`      固定行前缀 `TRACE | ...`，可 grep
    ② 观测契约 `<out>/obs/<run_id>/metrics.jsonl`  {ts, step, name, value}
       `<out>/obs/<run_id>/events.jsonl`           {ts, step, type, payload}
       `<out>/obs/<run_id>/run.json`               运行身份 + 汇总

🔴 铁律：业务代码不许手拼 trace 行 —— 格式一散，回放与统计就瞎了。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

# 行前缀契约：新增事件类型只改这里，不要在各处手写字符串
PREFIX = "TRACE"


class AgentTrace:
    """一次 agent 运行的轨迹记录器（上下文管理器）。"""

    def __init__(self, run_id: str, out_dir: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.run_id = run_id
        self.out_dir = out_dir
        self.obs_dir = os.path.join(out_dir, "obs", run_id)
        os.makedirs(self.obs_dir, exist_ok=True)
        self.t0 = time.time()
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.update({"run_id": run_id, "created_at": self.t0, "kind": "agent-harness"})

        self._log = open(os.path.join(out_dir, f"{run_id}.log"), "w", encoding="utf-8")
        self._metrics = open(os.path.join(self.obs_dir, "metrics.jsonl"), "w", encoding="utf-8")
        self._events = open(os.path.join(self.obs_dir, "events.jsonl"), "w", encoding="utf-8")
        self._summary: List[Dict[str, Any]] = []

    # ───────────────────────── 低层写入 ─────────────────────────
    def _emit(self, line: str) -> None:
        """唯一允许拼 trace 行的地方。"""
        self._log.write(f"{PREFIX} | {line}\n")
        self._log.flush()

    def event(self, turn: int, kind: str, **fields: Any) -> None:
        body = " | ".join(f"{k}:{v}" for k, v in fields.items())
        self._emit(f"turn:{turn} | event:{kind}" + (f" | {body}" if body else ""))
        self._events.write(json.dumps(
            {"ts": time.time(), "step": turn, "type": kind, "payload": fields},
            ensure_ascii=False) + "\n")
        self._events.flush()

    def metric(self, step: int, name: str, value: float) -> None:
        self._metrics.write(json.dumps(
            {"ts": time.time(), "step": step, "name": name, "value": float(value)},
            ensure_ascii=False) + "\n")
        self._metrics.flush()

    # ───────────────────────── 领域快捷方法 ─────────────────────────
    def task_start(self, task_id: str, capability: str, attempt: int) -> None:
        self.event(0, "task_start", task=task_id, capability=capability, attempt=attempt)

    def model_out(self, turn: int, content: str, tool_name: Optional[str],
                  tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.event(turn, "model_out",
                   tool=tool_name or "-",
                   content_len=len(content or ""),
                   tokens_in=tokens_in, tokens_out=tokens_out)

    def tool_call(self, turn: int, name: str, args: Dict[str, Any], ok: bool,
                  error: str = "", duration_ms: int = 0) -> None:
        self.event(turn, "tool_call", name=name, ok=int(ok),
                   args=json.dumps(args, ensure_ascii=False)[:200],
                   error=(error or "")[:200], ms=duration_ms)
        # 工具成功/失败率是 harness 健康度的核心指标
        self.metric(turn, f"tool/{name}/ok", 1.0 if ok else 0.0)

    def task_end(self, task_id: str, passed: bool, reason: str, turns: int,
                 seconds: float) -> None:
        self.event(turns, "task_end", task=task_id, passed=int(passed),
                   reason=reason[:200], turns=turns, seconds=round(seconds, 2))
        self.metric(turns, f"task/{task_id}/pass", 1.0 if passed else 0.0)
        self.metric(turns, f"task/{task_id}/turns", float(turns))
        self._summary.append({"task": task_id, "passed": bool(passed),
                              "reason": reason, "turns": turns,
                              "seconds": round(seconds, 2)})

    # ───────────────────────── 收尾 ─────────────────────────
    def finish(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n = len(self._summary)
        npass = sum(1 for s in self._summary if s["passed"])
        acc = (npass / n) if n else 0.0
        total_turns = sum(s["turns"] for s in self._summary)
        self.metric(0, "run/acc", acc)
        self.metric(0, "run/n_tasks", float(n))
        self.metric(0, "run/avg_turns", (total_turns / n) if n else 0.0)
        self.metric(0, "run/wall_s", time.time() - self.t0)

        out = dict(self.meta)
        out.update({
            "status": "finished",
            "finished_at": time.time(),
            "n_tasks": n, "n_passed": npass, "acc": round(acc, 4),
            "avg_turns": round((total_turns / n) if n else 0.0, 2),
            "wall_s": round(time.time() - self.t0, 2),
            "results": self._summary,
        })
        if extra:
            out.update(extra)
        # 原子写（tmp + os.replace）：观测平台可能同时读它
        tmp = os.path.join(self.obs_dir, "run.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(self.obs_dir, "run.json"))

        self.event(0, "run_end", acc=round(acc, 4), n=n, passed=npass)
        self._log.close()
        self._metrics.close()
        self._events.close()
        return out
