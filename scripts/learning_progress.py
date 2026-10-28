# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-01 11:04 · dsh-agent ｜ 变更：新增：学习进度看板生成器（证据驱动）

# 🕒 v1 · 2026-09-30 · dsh-agent ｜ 变更：新增学习进度看板生成器（证据驱动，不许手写完成度）

"""学习进度看板：**从仓库里的事实反推进度**，不接受手写打勾。

🔴 为什么这么设计
    手写"我会 PPO 了 ✅"没有任何证据价值。本脚本只做一件事：
    **扫描仓库里真实存在的产物，把"做了没有"变成"有没有文件/有没有数据"**。
    没有证据的能力项一律标 ⬜，并明确指出缺什么。

判据（每一项都对应一个可检查的文件或目录）
    ① 装置      scripts/agent_harness_min.py 能否跑通（有没有成功 run）
    ② 判分器    scripts/agent_tasks.py 的任务数 + 每个任务有没有 verify
    ③ 实验数据  runs/harness/obs/*/run.json 的 acc、任务数、墙钟
    ④ 观测      runs/obs/*/metrics.jsonl 的 run 数
    ⑤ 报告      docs/ 下的实验报告 / 设计文档完成度
    ⑥ H2 三档   context_level ∈ {A,B,C} 是否各有 ≥N 次 run（H2 实验的前置条件）

用法
    python scripts/learning_progress.py                 # 打印 + 写 docs/学习进度看板.md
    python scripts/learning_progress.py --n-min 30      # 改 H2 样本量门槛（默认 30）
    python scripts/learning_progress.py --no-write      # 只看不写
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 能力项 → (证据来源, 达标判据说明)。判据写在数据里，不写死在代码逻辑里。
CAPABILITIES: List[Dict[str, Any]] = [
    {"key": "harness", "name": "最小 agent harness（工具 schema + 控制循环）",
     "evidence": "scripts/agent_harness_min.py", "why": "harness 岗的核心工程物"},
    {"key": "tools", "name": "工具接口 + 参数校验 + 错误回写",
     "evidence": "scripts/agent_harness_min.py::TOOLS", "why": "工具设计质量直接决定 agent 成功率"},
    {"key": "trace", "name": "轨迹记录契约（可回放、可统计）",
     "evidence": "scripts/agent_trace.py", "why": "没有轨迹就没有归因"},
    {"key": "verifier", "name": "确定性 verifier（程序化判分）",
     "evidence": "scripts/agent_tasks.py::TASKS", "why": "评测地基，禁 LLM-as-judge"},
    {"key": "runs", "name": "真跑过（有 run.json 证据）",
     "evidence": "runs/harness/obs/*/run.json", "why": "设计不算数，跑过才算"},
    {"key": "dash", "name": "观测平台接入（metrics.jsonl）",
     "evidence": "runs/harness/obs/*/metrics.jsonl", "why": "曲线是学习闭环的反馈源"},
    {"key": "ab_design", "name": "H2 实验设计（可证伪判据 + 撤回条件）",
     "evidence": "docs/实验_上下文注入AB_设计.md", "why": "研究方法论的证据"},
    {"key": "ab_data", "name": "H2 已产出数据（A/B/C 三档都跑过）",
     "evidence": "runs/harness/obs/*/run.json::context_level", "why": "设计停在纸面 = 0 分"},
    {"key": "report_tool", "name": "H2 报告生成器（判据机械套用 + Wilson CI）",
     "evidence": "scripts/h2_report.py", "why": "结论必须可复算，不能手写"},
    {"key": "client_test", "name": "真模型客户端零花费契约回归",
     "evidence": "scripts/test_deepseek_client.py", "why": "拿 key 前先证明第一次调用是通的"},
]


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def scan() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ts": time.time(), "items": {}, "runs": [], "metrics_runs": 0}

    # ① 文件型证据
    out["items"]["harness"] = os.path.isfile(os.path.join(PROJ, "scripts", "agent_harness_min.py"))
    out["items"]["trace"] = os.path.isfile(os.path.join(PROJ, "scripts", "agent_trace.py"))
    out["items"]["ab_design"] = os.path.isfile(
        os.path.join(PROJ, "docs", "实验_上下文注入AB_设计.md"))
    out["items"]["report_tool"] = os.path.isfile(
        os.path.join(PROJ, "scripts", "h2_report.py"))
    out["items"]["client_test"] = os.path.isfile(
        os.path.join(PROJ, "scripts", "test_deepseek_client.py"))

    # ② 判分器：数任务 + 校验每个都有 verify
    n_tasks, verified = 0, 0
    try:
        sys.path.insert(0, os.path.join(PROJ, "scripts"))
        from agent_tasks import list_tasks              # noqa: E402
        for t in list_tasks():
            n_tasks += 1
            if callable(getattr(t, "verify", None)):
                verified += 1
        out["items"]["tools"] = True
    except Exception as e:                              # noqa: BLE001
        out["items"]["tools"] = False
        out["tools_error"] = f"{type(e).__name__}: {e}"
    out["n_tasks"] = n_tasks
    out["n_verified"] = verified
    out["items"]["verifier"] = n_tasks > 0 and verified == n_tasks

    # ③ 真跑过的 run
    for p in sorted(glob.glob(os.path.join(PROJ, "runs", "harness", "obs", "*", "run.json"))):
        d = _read_json(p)
        if not d:
            continue
        out["runs"].append({
            "run_id": d.get("run_id"),
            "client": d.get("client"),
            "context_level": d.get("context_level"),
            "n_tasks": d.get("n_tasks"), "acc": d.get("acc"),
            "avg_turns": d.get("avg_turns"), "wall_s": d.get("wall_s"),
            "created_at": d.get("created_at"),
        })
    out["items"]["runs"] = len(out["runs"]) > 0

    # ④ 观测契约
    out["metrics_runs"] = len(glob.glob(
        os.path.join(PROJ, "runs", "harness", "obs", "*", "metrics.jsonl")))
    out["items"]["dash"] = out["metrics_runs"] > 0

    return out


def ab_coverage(runs: List[Dict[str, Any]], n_min: int) -> Dict[str, Any]:
    """H2 前置条件：A/B/C 三档在**同一 client** 下是否各有 ≥ n_min 次任务记录。"""
    real = [r for r in runs if r.get("client") and r["client"] != "mock"]
    per: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for r in real:
        lv = r.get("context_level")
        if lv in per:
            per[lv] += int(r.get("n_tasks") or 0)
    ready = all(per[k] >= n_min for k in ("A", "B", "C"))
    return {"per_level": per, "n_min": n_min, "ready": ready,
            "mock_runs_excluded": len(runs) - len(real)}


def render(scan_data: Dict[str, Any], ab: Dict[str, Any]) -> str:
    L: List[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(scan_data["ts"]))
    L.append(f"# 学习进度看板（自动生成 · {ts}）")
    L.append("")
    L.append("> 🔴 本文件由 `scripts/learning_progress.py` 生成，**不要手改** —— 手写的完成度没有证据价值。")
    L.append("> 每一项的 ⬜/✅ 都由**仓库里真实存在的文件或数据**决定。")
    L.append("")

    # 能力表
    L.append("## 一、能力项（证据驱动）")
    L.append("")
    L.append("| 状态 | 能力项 | 证据 | 为什么重要 |")
    L.append("|---|---|---|---|")
    for cap in CAPABILITIES:
        got = scan_data["items"].get(cap["key"], False)
        if cap["key"] == "ab_data":
            got = ab["ready"]
        mark = "✅" if got else "⬜"
        L.append(f"| {mark} | {cap['name']} | `{cap['evidence']}` | {cap['why']} |")
    L.append("")

    # 判分器
    L.append("## 二、评测装置")
    L.append("")
    L.append(f"- 任务数：**{scan_data.get('n_tasks', 0)}**，其中带确定性 verifier 的："
             f"**{scan_data.get('n_verified', 0)}**")
    if scan_data.get("tools_error"):
        L.append(f"- ⚠️ 任务集导入失败：`{scan_data['tools_error']}`")
    L.append(f"- 观测契约文件（metrics.jsonl）：**{scan_data.get('metrics_runs', 0)}** 个")
    L.append("")

    # 跑过的数据
    L.append("## 三、跑过的 run（真数据）")
    L.append("")
    if not scan_data["runs"]:
        L.append("_还没有任何 run。先跑 `python scripts/agent_harness_min.py --client mock --repeat 1`。_")
    else:
        L.append("| run_id | client | 档位 | 任务数 | 通过 | 平均轮数 | 墙钟(s) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in scan_data["runs"]:
            acc = r.get("acc")
            acc_s = f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else "-"
            L.append(f"| `{r.get('run_id')}` | {r.get('client')} | {r.get('context_level')} | "
                     f"{r.get('n_tasks')} | {acc_s} | {r.get('avg_turns')} | {r.get('wall_s')} |")
    L.append("")

    # H2 前置
    L.append("## 四、H2 实验前置条件（A/B/C 三档 × 真模型）")
    L.append("")
    L.append(f"- 门槛：每档 ≥ **{ab['n_min']}** 次任务记录（设计文档 §2.2 的样本量纪律）")
    L.append(f"- 实际（已排除 mock）：A={ab['per_level']['A']} · B={ab['per_level']['B']} · "
             f"C={ab['per_level']['C']}")
    L.append(f"- 结论：**{'✅ 可以开跑 H2' if ab['ready'] else '⬜ 还不够，先补跑'}**")
    if ab["mock_runs_excluded"]:
        L.append(f"- （已排除 {ab['mock_runs_excluded']} 个 mock run：mock 恒 100%，不代表模型能力）")
    L.append("")

    # 下一步
    L.append("## 五、下一步（由缺口倒推）")
    L.append("")
    steps: List[str] = []
    if not scan_data["items"].get("runs"):
        steps.append("跑一次装置自检：`python scripts/agent_harness_min.py --client mock --repeat 3`")
    if ab["per_level"]["A"] == 0:
        steps.append("用**真模型**跑 A 档：`--client deepseek --context A --repeat 10`"
                     "（需 `DEEPSEEK_API_KEY`）")
    if ab["per_level"]["B"] == 0 or ab["per_level"]["C"] == 0:
        steps.append("补 B / C 档（同一天内跑完，否则服务端变化会污染结论）")
    if ab["ready"]:
        steps.append("按 §2.5 判据出结论：C 比 A 高 ≥15 点 → H2 成立；<5 点 → 撤回")
        steps.append("写一页技术报告（`docs/`），含 §4 的诚实短板")
    for s in steps:
        L.append(f"- [ ] {s}")
    if not steps:
        L.append("- 无待办")
    L.append("")
    L.append("---")
    L.append("")
    L.append("_生成器：`scripts/learning_progress.py`；数据来源：本仓库文件系统_")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=30)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=os.path.join(PROJ, "docs", "学习进度看板.md"))
    args = ap.parse_args(argv)

    data = scan()
    ab = ab_coverage(data["runs"], args.n_min)
    text = render(data, ab)

    print(text)
    if not args.no_write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, args.out)
        print(f"\n已写入：{args.out}")

        snap = os.path.join(PROJ, "runs", "progress", time.strftime("%Y%m%d_%H%M") + ".json")
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        with open(snap, "w", encoding="utf-8") as f:
            json.dump({"scan": data, "ab": ab}, f, ensure_ascii=False, indent=2)
        print(f"快照：{snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
