# -*- coding: utf-8 -*-
# 🕒 v2 · 2026-10-01 16:02 · dsh-agent ｜ 变更：补 commission（要求）型指令约束：新增 v_must_include + com-1/com-2，与既有 constraint（omission 型）配对，使 omission/commission 不对称可测（依据 arXiv:2604.20911） ｜ 上一版：v1 · 2026-09-30 20:59

"""评测集 + 自动判分（verifier）：把「能力」变成可复现的数字。

🔴 为什么必须有这个文件
    `scripts/lora_smoke.py` 的 loss 曲线**不能**回答"模型能力变了没有" ——
    8 步 loss 从 2.84 降到 2.50，只是拟合了 4 条样本（见 `docs/实验登记表.md` §二）。
    学习手册级 4 的命题「**loss 下降 ≠ 学到对的东西**」要能验证，就必须有一个
    **自己造的、能自动判对错**的评测集。这就是本文件。

设计纪律（与 RL_infra 四条隔离一致）
    · 任务与判分**同处一个文件**（`TASKS`），换任务只改这一处。
    · 产物走同一观测契约：`<out>/obs/<run_id>/{metrics.jsonl,run.json}`，
      指标名 `eval/<capability>/acc` 与 `eval/acc`，前端/平台无需改。
    · 判分**只依赖程序化规则**（精确匹配 / 正则 / JSON 解析），不用模型当裁判。
      模型裁判（LLM-as-judge）是另一件事，别混进来。

用法
    # ① 看有哪些任务与能力维度（不需要 GPU/torch）
    python scripts/eval_set.py --list

    # ② 评一个基座模型
    python scripts/eval_set.py --model <权重目录> --tag base

    # ③ 评微调后的 adapter（级 4 / 级 6 的判据）
    python scripts/eval_set.py --model <权重目录> --adapter <run_dir>/adapter --tag lora

    # ④ 长上下文退化：在题目前塞 N 段无关填充（"感觉水平下降"的可测版本）
    python scripts/eval_set.py --model <权重目录> --tag pad8k --context-pad 8000

    # ⑤ 只跑某个能力维度
    python scripts/eval_set.py --model <权重目录> --only constraint

产物
    <out>/<tag>.log                            人类可读（含每题 ✓/✗ 与模型原文前 200 字）
    <out>/obs/<tag>/metrics.jsonl              {ts, step, name, value}
    <out>/obs/<tag>/run.json                   运行身份 + 汇总（供平台/登记表引用）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# 评价集：一条任务 = 一个 prompt + 一个程序化判分器
#   判分器签名 verify(text: str) -> tuple[bool, str]   （第二项是判分理由，进日志）
# 🔴 新增任务只改这里。判分器必须是**确定性**的：同样的输出永远给同样的分。
# ─────────────────────────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    return s.strip().replace("，", ",").replace("：", ":")


def v_exact_int(expected: int):
    """纯算术：回复里出现的第一个整数必须等于 expected。"""
    def _v(text: str):
        m = re.search(r"-?\d+", text)
        if not m:
            return False, "回复里没有整数"
        got = int(m.group(0))
        return got == expected, f"期望 {expected}，得到 {got}"
    return _v


def v_boxed(expected: str):
    """格式遵循：答案必须包在 \\boxed{} 里（格式不对 = 不通过，不管内容对不对）。"""
    def _v(text: str):
        m = re.search(r"\\boxed\{([^}]*)\}", text)
        if not m:
            return False, "没有用 \\boxed{} 包住答案"
        got = _norm(m.group(1))
        return got == _norm(expected), f"期望 {expected}，盒子里是 {got!r}"
    return _v


def v_bare_number(expected: int):
    """指令约束（omission 型）：去掉首尾空白后**只能**是一个数字，多一个字符就失败。

    🔴 这条专治「模型爱解释」：能力上它算得对，但约束没守住。
       长上下文实验里，**禁止型约束的衰减比要求型更快** —— 所以要有独立维度测它。
    """
    def _v(text: str):
        got = text.strip()
        if not re.fullmatch(r"-?\d+", got):
            return False, f"只允许输出一个数字，实际输出 {got[:60]!r}"
        return int(got) == expected, f"期望 {expected}，得到 {got}"
    return _v


def v_must_include(keyword: str, min_chars: int = 15):
    """指令约束（**commission 型**：必须**额外包含**某些内容）。

    🔴 与 `v_bare_number`（omission 型：不许有多余字符）**成对**。
       依据 `Omission Constraints Decay While Commission Constraints Persist in Long-Context
       LLM Agents`（arXiv:2604.20911）：禁止型约束随上下文加深从 73% 掉到 33%，
       而要求型稳定在 100%。要测出这个**不对称**，两类必须**分维度独立统计**
       —— 所以 commission 单独一个 cap，绝不与 constraint 混在一个分母里。
    """
    def _v(text: str):
        t = text.strip()
        if len(t) < min_chars:
            return False, f"回复太短（{len(t)} < {min_chars} 字符），没按要求作答"
        if keyword not in t:
            return False, f"缺少必须包含的关键词「{keyword}」"
        return True, "OK"
    return _v


def v_tool_json(required_keys):
    """工具调用格式：输出必须是合法 JSON，且含指定键。"""
    def _v(text: str):
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return False, "找不到 JSON 对象"
        try:
            obj = json.loads(m.group(0))
        except Exception as e:  # noqa: BLE001
            return False, f"JSON 解析失败：{e}"
        if not isinstance(obj, dict):
            return False, "顶层不是对象"
        missing = [k for k in required_keys if k not in obj]
        return (not missing), (f"缺键 {missing}" if missing else "OK")
    return _v


def v_contains(*needles: str):
    """长上下文针：回复里必须同时出现全部 needle。"""
    def _v(text: str):
        miss = [n for n in needles if n not in text]
        return (not miss), (f"缺 {miss}" if miss else "OK")
    return _v


# capability 是分析维度，不是难度分级。改名要同步 docs/实验登记表.md。
TASKS = [
    # ① 算术（最短上下文，作为"装置是否正常"的基线）
    dict(id="arith-1", cap="arith", prompt="问题：37 加 58 等于几？\n只回答数字。",
         verify=v_bare_number(95)),
    dict(id="arith-2", cap="arith", prompt="问题：一箱 24 个，7 箱共几个？\n只回答数字。",
         verify=v_bare_number(168)),
    # ② 格式遵循（答案是次要的，格式是主要的）
    dict(id="fmt-1", cap="format", prompt="计算 13×14，把最终答案放进 \\boxed{} 里。",
         verify=v_boxed("182")),
    dict(id="fmt-2", cap="format", prompt="计算 25 的平方，把最终答案放进 \\boxed{} 里。",
         verify=v_boxed("625")),
    # ③ 指令约束（omission 型：不许有多余内容）
    dict(id="con-1", cap="constraint", prompt="回答下面的问题。硬性要求：你的整段回复必须只有数字，不得有任何其他字符。\n问题：9 乘 9 等于几？",
         verify=v_bare_number(81)),
    dict(id="con-2", cap="constraint", prompt="回答下面的问题。硬性要求：你的整段回复必须只有数字，不得有任何其他字符。\n问题：100 减 37 等于几？",
         verify=v_bare_number(63)),
    # ③b 指令约束（**commission 型：必须额外包含**）—— 与 ③ 配对
    #    🔴 关键词刻意选**与答案无关**的词（校验/复核），否则不用"守约束"也能蒙对，
    #       测的就不是约束遵循能力了。两者合起来才是「omission vs commission」两臂。
    dict(id="com-1", cap="commission",
         prompt="回答下面的问题。硬性要求：除了答案之外，你的回复必须**额外包含关键词「校验」**，且总长度不少于 15 个字符。\n问题：9 乘 9 等于几？",
         verify=v_must_include("校验", 15)),
    dict(id="com-2", cap="commission",
         prompt="回答下面的问题。硬性要求：除了答案之外，你的回复必须**额外包含关键词「复核」**，且总长度不少于 15 个字符。\n问题：100 减 37 等于几？",
         verify=v_must_include("复核", 15)),
    # ④ 工具调用 JSON（Agent 骨架的最小能力）
    dict(id="tool-1", cap="tool",
         prompt='请输出一个工具调用，调用 read_file 读取 /tmp/a.txt。\n格式要求：只输出一个 JSON 对象，含键 "tool" 和 "args"。',
         verify=v_tool_json(["tool", "args"])),
    dict(id="tool-2", cap="tool",
         prompt='请输出一个工具调用，调用 calc 计算 12*8。\n格式要求：只输出一个 JSON 对象，含键 "tool" 和 "args"。',
         verify=v_tool_json(["tool", "args"])),
    # ⑤ 长上下文针（配合 --context-pad 使用；不打 pad 时是普通短上下文题）
    dict(id="needle-1", cap="needle",
         prompt="阅读上文后回答：那串校验码是什么？\n只回答校验码本身。",
         verify=v_contains("K7QX-2291")),
]

NEEDLE = "\n【环境记录】本批任务的校验码为 K7QX-2291，请在后文需要时使用。\n"
FILLER = "【无关日志】本条记录与任务无关，仅用于占用上下文长度，请忽略其内容。\n"


def build_context(task_prompt: str, pad_tokens: int) -> str:
    """把 needle 藏在长上下文里（pad_tokens 是近似 token 数，按 2 字符/token 估）。"""
    if pad_tokens <= 0:
        return task_prompt
    n = max(1, pad_tokens // 12)          # 每段填充约 12 token
    return NEEDLE + FILLER * n + task_prompt


# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/public/home/__HPC_ACCOUNT__/RL_infra/model_cache/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录（不传 = 评基座）")
    ap.add_argument("--out", default="model_cache/_eval_logs")
    ap.add_argument("--tag", default="base", help="本次运行的短名（决定产物目录）")
    ap.add_argument("--only", default="", help="只跑某个 capability")
    ap.add_argument("--context-pad", type=int, default=0, help="在题目前塞多少 token 的无关填充")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--list", action="store_true", help="只列任务，不加载模型")
    args = ap.parse_args()

    if args.list:
        caps: dict[str, int] = {}
        for t in TASKS:
            caps[t["cap"]] = caps.get(t["cap"], 0) + 1
        print(f"共 {len(TASKS)} 条任务，{len(caps)} 个能力维度：")
        for c, n in sorted(caps.items()):
            print(f"  {c:<12} {n} 条   ({', '.join(t['id'] for t in TASKS if t['cap'] == c)})")
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_id = args.tag
    run_dir = os.path.join(args.out, "obs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(args.out, f"{run_id}.log"), "w", encoding="utf-8")
    m_f = open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8")
    t0 = time.time()

    def say(line: str = "") -> None:
        print(line, flush=True)
        if not log_f.closed:
            log_f.write(line + "\n")
            log_f.flush()

    def metric(step: int, name: str, value: float) -> None:
        m_f.write(json.dumps({"ts": time.time(), "step": step, "name": name,
                              "value": float(value)}) + "\n")
        m_f.flush()

    say(f"===== EVAL SET | run_id={run_id} | adapter={args.adapter or '(base)'} | "
        f"context_pad={args.context_pad} =====")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model = model.to(dev)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        say(f"已挂载 adapter：{args.adapter}")
    model.eval()

    tasks = [t for t in TASKS if not args.only or t["cap"] == args.only]
    per_cap: dict[str, list[int]] = {}
    details = []
    step = 0

    for t in tasks:
        step += 1
        prompt = build_context(t["prompt"], args.context_pad)
        if args.context_pad > 0 and not prompt.startswith(NEEDLE):
            prompt = NEEDLE + prompt          # 非 needle 题也把针放进上下文，测"被淹没"
        msgs = [{"role": "user", "content": prompt}]
        try:
            text_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            text_in = prompt
        enc = tok(text_in, return_tensors="pt", truncation=True, max_length=8192).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        ok, why = t["verify"](gen)
        per_cap.setdefault(t["cap"], []).append(1 if ok else 0)
        details.append(dict(id=t["id"], cap=t["cap"], ok=bool(ok), why=why,
                            output=gen[:400]))
        say(f"[{'✓' if ok else '✗'}] {t['id']:<10} ({t['cap']:<10}) {why}")
        say(f"      模型输出：{gen[:200]!r}")
        metric(step, f"eval/{t['cap']}/pass", 1.0 if ok else 0.0)

    for cap, flags in sorted(per_cap.items()):
        acc = sum(flags) / len(flags)
        say(f"—— 能力 {cap:<12} 通过 {sum(flags)}/{len(flags)} = {acc:.1%}")
        metric(0, f"eval/{cap}/acc", acc)
    total = [f for flags in per_cap.values() for f in flags]
    overall = sum(total) / max(len(total), 1)
    say(f"===== 总通过率 {sum(total)}/{len(total)} = {overall:.1%} "
        f"| 用时 {time.time()-t0:.1f}s ===== ")
    metric(0, "eval/acc", overall)
    metric(0, "elapsed_s_total", time.time() - t0)

    with open(os.path.join(run_dir, "detail.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "status": "finished", "kind": "rlinfra-eval-set",
                   "model_path": args.model, "adapter": args.adapter or None,
                   "context_pad": args.context_pad, "n_tasks": len(tasks),
                   "overall_acc": round(overall, 4),
                   "per_capability": {c: round(sum(v) / len(v), 4) for c, v in per_cap.items()},
                   "created_at": t0, "finished_at": time.time()}, f,
                  ensure_ascii=False, indent=2)

    log_f.close()
    m_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
