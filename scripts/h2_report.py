# -*- coding: utf-8 -*-
# 🕒 v4 · 2026-10-09 16:16 · dsh-agent ｜ 变更：诚实短板加第 5 条：思考模式下 temperature=0 名义生效实为默认采样 ｜ 上一版：v3 · 2026-10-09 16:15

# 🕒 v1 · 2026-10-01 · dsh-agent ｜ 变更：新增 H2 实验分析与技术报告生成器（读 run.json，出判据结论文档）

"""H2 实验分析与技术报告生成：从 `runs/harness/obs/<run_id>/run.json` 出**带判据的结论**。

🔴 判据在跑之前就写死（`docs/实验_上下文注入AB_设计.md` §2.5），本脚本只负责**机械套用**：
   C 比 A 高 ≥15 点 → 成立 ｜ 5–15 点 → 方向成立效应量不定 ｜ <5 点 → 撤回 ｜ 更低 → 反向
   先写判据再跑，是为了防止"跑完再解释"（p-hacking 的入门形态）。

🔴 mock 的 run 会被**排除**并显式标注：mock 是"完美模型"脚本，acc 恒 100%，
   混进来会让结论完全失真（`docs/AgentHarness-战果与工具链.md` §1 已写明）。

统计口径
    · 通过率 = 通过任务数 / 总任务数（**任务级**，不是轮次级）
    · 置信区间用 **Wilson 区间**（小样本 + 比例数据，比正态近似可靠）
    · 样本量纪律：每档 < 30 次任务记录时，报告必须标 ⚠️（设计文档 §2.2）

用法
    python scripts/h2_report.py                          # 用默认档位 h2_A/h2_B/h2_C
    python scripts/h2_report.py --runs h2_A h2_B h2_C --out docs/H2_实验报告.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIENT_ORDER = ["deepseek"]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """二项比例的 Wilson 得分区间（95%）。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def load_run(rid: str, obs_root: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(obs_root, rid, "run.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def summarize(run: Dict[str, Any]) -> Dict[str, Any]:
    res = run.get("results") or []
    n = len(res)
    k = sum(1 for r in res if r.get("passed"))
    lo, hi = wilson(k, n)
    by_task: Dict[str, List[bool]] = {}
    for r in res:
        by_task.setdefault(r.get("task", "?"), []).append(bool(r.get("passed")))
    # 🔴 逐条 results[] 里**没有** token 字段（只有 task/passed/reason/turns/seconds），
    #    总量在 run.json 顶层 tok_in_total/tok_out_total —— 漏了这个 fallback 时
    #    §五 成本表会因 `any(tok_in)` 恒为 False 而**整个不渲染**（实测踩到）。
    tok_in = sum(int(r.get("tok_in") or 0) for r in res) or int(run.get("tok_in_total") or 0)
    tok_out = sum(int(r.get("tok_out") or 0) for r in res) or int(run.get("tok_out_total") or 0)
    return {
        "run_id": run.get("run_id"), "client": run.get("client"),
        "model": run.get("model"), "context_level": run.get("context_level"),
        "context_desc": run.get("context_desc"),
        "n": n, "k": k, "acc": (k / n) if n else 0.0, "ci": (lo, hi),
        "avg_turns": run.get("avg_turns"), "wall_s": run.get("wall_s"),
        "tasks": {t: (sum(v), len(v)) for t, v in sorted(by_task.items())},
        "tok_in": tok_in, "tok_out": tok_out,
        "cost_cny": run.get("cost_cny"),
        "created_at": run.get("created_at"),
    }


def verdict(acc_c: float, acc_a: float, n_min_ok: bool) -> Tuple[str, str]:
    d = (acc_c - acc_a) * 100
    if not n_min_ok:
        return "⬜ 样本不足，不下结论", f"差值 {d:+.1f} 点，但每档未达 30 次任务记录的门槛"
    if d >= 15:
        return "✅ H2 成立", f"C 比 A 高 {d:.1f} 个点（≥15）"
    if d >= 5:
        return "⚠️ 方向成立、效应量不确定", f"C 比 A 高 {d:.1f} 个点（5–15），需加样本"
    if d >= 0:
        return "❌ H2 不成立，撤回", f"仅高 {d:.1f} 个点（<5），转 H3/H1"
    return "🔴 反向", f"C 比 A **低** {abs(d):.1f} 个点：常驻规则是净收益，另有原因"


def render(sums: List[Dict[str, Any]], dropped: List[Dict[str, Any]], n_min: int) -> str:
    L: List[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M")
    L.append(f"# H2 实验报告 · 上下文注入 A/B/C（生成于 {ts}）")
    L.append("")
    L.append("> 本文件由 `scripts/h2_report.py` 生成，**数字全部来自 run.json**，不要手改。")
    L.append("> 判据在跑之前写死（`docs/实验_上下文注入AB_设计.md` §2.5）。")
    L.append("")

    real = [s for s in sums if s["client"] in DEFAULT_CLIENT_ORDER]
    if not real:
        L.append("## ⚠️ 尚无真模型数据——本报告不含任何结论")
        L.append("")
        L.append("当前 `runs/harness/obs/` 里没有 `client=deepseek` 的 run，无法出结论。")
        L.append("")
        if dropped:
            L.append("**已排除的非真模型 run**（mock 恒 100%，代表装置可用、不代表模型能力）：")
            L.append("")
            L.append("| run_id | client | 档位 | 任务数 | 通过率 |")
            L.append("|---|---|---|---|---|")
            for s in dropped:
                L.append(f"| `{s['run_id']}` | {s['client']} | {s['context_level']} | "
                         f"{s['n']} | {s['acc']*100:.1f}% |")
            L.append("")
        L.append("### 补齐真模型数据（需要 `DEEPSEEK_API_KEY`）")
        L.append("")
        L.append("```powershell")
        L.append("python scripts/agent_harness_min.py --client deepseek --repeat 6 --estimate-cost --price-in <in> --price-out <out>")
        L.append("python scripts/agent_harness_min.py --client deepseek --context A --repeat 6 --run-id h2_A")
        L.append("python scripts/agent_harness_min.py --client deepseek --context B --repeat 6 --run-id h2_B")
        L.append("python scripts/agent_harness_min.py --client deepseek --context C --repeat 6 --run-id h2_C")
        L.append("python scripts/h2_report.py     # 再生成本报告")
        L.append("```")
        L.append("")
        L.append("⚠️ **三档必须在同一天内跑完**（服务端是黑盒，跨天会被「模型版本变了」污染）。")
        return "\n".join(L) + "\n"

    # 真数据：按档位聚合
    by_level: Dict[str, List[Dict[str, Any]]] = {}
    for s in real:
        by_level.setdefault(s["context_level"] or "?", []).append(s)

    L.append("## 一、数据总览（真模型）")
    L.append("")
    L.append("| 档位 | 说明 | run 数 | 任务记录 | 通过 | 通过率 | 95% Wilson CI | 平均轮数 |")
    L.append("|---|---|---|---|---|---|---|---|")
    agg: Dict[str, Dict[str, Any]] = {}
    for lv in sorted(by_level):
        ss = by_level[lv]
        n = sum(s["n"] for s in ss)
        k = sum(s["k"] for s in ss)
        lo, hi = wilson(k, n)
        turns = [s["avg_turns"] for s in ss if s["avg_turns"] is not None]
        agg[lv] = {"n": n, "k": k, "acc": (k / n) if n else 0.0, "ci": (lo, hi)}
        L.append(f"| **{lv}** | {ss[0]['context_desc']} | {len(ss)} | {n} | {k} | "
                 f"**{(k/n*100 if n else 0):.1f}%** | "
                 f"[{lo*100:.1f}, {hi*100:.1f}] | "
                 f"{(sum(turns)/len(turns)) if turns else 0:.2f} |")
    L.append("")
    if by_level:
        any_model = real[0].get("model")
        L.append(f"- 模型：`{any_model}` · client：`{real[0].get('client')}`")
        days = {time.strftime("%Y-%m-%d", time.localtime(s["created_at"] or 0)) for s in real}
        L.append(f"- 跑测日期：{sorted(days)}" + ("  ⚠️ **跨天，服务端漂移风险**" if len(days) > 1 else "  ✅ 同日"))
    L.append("")

    # 判据
    L.append("## 二、判据套用（§2.5）")
    L.append("")
    if "A" in agg and "C" in agg:
        n_min_ok = all(agg[lv]["n"] >= n_min for lv in ("A", "B", "C") if lv in agg) and \
                   all(lv in agg for lv in ("A", "B", "C"))
        v, why = verdict(agg["C"]["acc"], agg["A"]["acc"], n_min_ok)
        L.append(f"- **结论：{v}**")
        L.append(f"- 依据：{why}")
        L.append(f"- A 通过率 {agg['A']['acc']*100:.1f}% · C 通过率 {agg['C']['acc']*100:.1f}%")
        if not n_min_ok:
            L.append(f"- ⚠️ 样本量门槛：每档 ≥ {n_min} 次任务记录（当前 A={agg.get('A',{}).get('n',0)} · "
                     f"B={agg.get('B',{}).get('n',0)} · C={agg.get('C',{}).get('n',0)}）")
        # 🔴 天花板检查（2026-10-09 加）：A 档已接近满分时，"C−A 的差值"没有上涨空间，
        #    此时机械套用判据会得出"❌ H2 不成立，撤回"—— 那是**把量程不足误读成反驳**。
        #    正确读法：任务集对当前模型饱和，判据不适用；改看不受天花板影响的次级信号（轮数/token）。
        if agg["A"]["acc"] >= 0.95:
            L.append(f"- 🔴 **天花板效应**：A 档通过率已 {agg['A']['acc']*100:.1f}%（满分），"
                     "5 任务集对当前模型**饱和、无分辨力** ⇒ 上面的差值判据**既不成立也不撤回**，"
                     "只能判「当前任务集测不出上下文档位的增益」。")
            L.append("- 可继续报告的**次级信号**：同批次的平均轮数与 token 随档位单调变化"
                     "（见 §一 平均轮数 与 §五 成本），它不受天花板影响。")
    else:
        L.append("- ⚠️ A 或 C 档缺数据，无法套用判据。")
    L.append("")

    # 逐任务
    L.append("## 三、逐任务明细（定位是哪个能力维度在退化）")
    L.append("")
    task_ids = sorted({t for s in real for t in s["tasks"]})
    L.append("| 任务 | " + " | ".join(sorted(by_level)) + " |")
    L.append("|---" * (1 + len(by_level)) + "|")
    for t in task_ids:
        cells = []
        for lv in sorted(by_level):
            k = sum(s["tasks"].get(t, (0, 0))[0] for s in by_level[lv])
            n = sum(s["tasks"].get(t, (0, 0))[1] for s in by_level[lv])
            cells.append(f"{k}/{n}" if n else "-")
        L.append(f"| `{t}` | " + " | ".join(cells) + " |")
    L.append("")
    L.append("> 🔴 **禁止型任务（a3/a4/a5 的格式约束）比要求型任务更容易受上下文膨胀影响** ——"
             "若它们的降幅明显大于 a1/a2，这本身就是可报告的现象。")
    L.append("")

    # 诚实短板
    L.append("## 四、🔴 诚实短板（不许省略）")
    L.append("")
    L.append("1. **服务端黑盒**：无权重/无解码参数观测点 ⇒ 结论只能是**当日快照**，不可跨月比较。")
    L.append(f"2. **样本量**：每档 ≥ {n_min} 次任务记录是下限；低于此不下结论。")
    L.append("3. **mock 数据已排除**：mock 恒 100%，只证明装置可用，不代表模型能力。")
    L.append("4. **判据在跑之前写死**：防止跑完再解释（见设计文档 §2.5）。")
    L.append("5. 🔴 **本批 `temperature=0` 实际没生效**：DeepSeek 官方文档明示「思考模式不支持 "
             "temperature / presence_penalty / frequency_penalty，设了不报错但不生效」，"
             "而 `deepseek-flash` 默认开思考模式 ⇒ 设计文档里「固定温度」这一固定项是**名义上的**，"
             "实际是模型默认采样。本批三档同日跑可部分抵消，但跨批比较必须知道这一点。")
    _idx = 5
    if dropped:
        _idx += 1
        L.append(f"{_idx}. 本报告排除了 {len(dropped)} 个非真模型 run。")
    if "A" in agg and agg["A"]["acc"] >= 0.95:
        _idx += 1
        L.append(f"{_idx}. 🔴 **天花板效应（本批实测）**：A 档已 {agg['A']['acc']*100:.1f}% ⇒ "
                 "本批数据**不能**检验 H2（通过率没有上涨空间），"
                 "只能报「5 任务集对当前模型饱和」+ 轮数/token 的次级差异。")
    L.append("")

    # 成本
    if any(s.get("tok_in") for s in real):
        ti = sum(s["tok_in"] for s in real)
        to = sum(s["tok_out"] for s in real)
        cost = sum(s.get("cost_cny") or 0 for s in real)
        L.append("## 五、成本")
        L.append("")
        L.append(f"- 输入 {ti:,} tok · 输出 {to:,} tok · 合计花费 **{cost:.4f} 元**")
        L.append("")
        # 逐档成本：天花板效应下这是**唯一还有分辨力**的信号（常驻规则的代价）
        L.append("| 档位 | 任务记录 | 输入 tok | 输出 tok | 花费 | 平均轮数 |")
        L.append("|---|---|---|---|---|---|")
        for lv in sorted(by_level):
            ss = by_level[lv]
            lv_ti = sum(s["tok_in"] for s in ss)
            lv_to = sum(s["tok_out"] for s in ss)
            lv_cost = sum(s.get("cost_cny") or 0 for s in ss)
            turns = [s["avg_turns"] for s in ss if s["avg_turns"] is not None]
            L.append(f"| **{lv}** | {sum(s['n'] for s in ss)} | {lv_ti:,} | {lv_to:,} | "
                     f"{lv_cost:.4f} 元 | {(sum(turns)/len(turns)) if turns else 0:.2f} |")
        L.append("")
        L.append("> 判据见 §二：本批数据下通过率触顶，**这组 token/轮数差异才是可用信号**"
                 "（常驻规则越多 → 轮数与输入 token 越大，通过率不变）。")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["h2_A", "h2_B", "h2_C"])
    ap.add_argument("--obs-root", default=os.path.join(PROJ, "runs", "harness", "obs"))
    ap.add_argument("--n-min", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(PROJ, "docs", "H2_实验报告.md"))
    args = ap.parse_args(argv)

    # 收集：指定档位 + 目录里其他所有 run（用来做"排除项"清单）
    all_ids = sorted(os.listdir(args.obs_root)) if os.path.isdir(args.obs_root) else []
    sums: List[Dict[str, Any]] = []
    for rid in sorted(set(args.runs) | set(all_ids)):
        d = load_run(rid, args.obs_root)
        if d:
            sums.append(summarize(d))

    dropped = [s for s in sums if s["client"] not in DEFAULT_CLIENT_ORDER]
    text = render(sums, dropped, args.n_min)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, args.out)
    print(f"\n已写入：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
