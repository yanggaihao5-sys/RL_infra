# -*- coding: utf-8 -*-
# 🕒 v5 · 2026-10-09 16:03 · dsh-agent ｜ 变更：真模型三处协议修复：tool/assistant 顺序、reasoning_content 按键存在回传、run_command 自述 Windows 环境 ｜ 上一版：v4 · 2026-10-09 16:00

# 🕒 v1 · 2026-09-30 · dsh-agent ｜ 变更：新增最小 agent harness（工具 schema + 控制循环 + 轨迹契约 + 确定性 verifier 接线）

"""最小 agent harness：模型外面那层「驾驶员」。

🔴 为什么要有它（而不是直接调库/用现成框架）
    Agent Harness 岗位考的核心不是"会用框架"，是**能把模型行为变成可测的数字**：
    上下文怎么组装、工具接口怎么设计、失败怎么重试、产出怎么判分、轨迹怎么留证。
    这五件事全在这一个文件里，各 ~20 行，看得见、改得动。

结构（与 `docs/实验_上下文注入AB_设计.md` §1 一一对应）
    ① 工具接口   TOOLS：JSON Schema 声明 + 参数校验 + **错误信息回写**
    ② 控制循环   run_task()：模型 → tool call → 执行 → 结果回写 → 直到无 tool call 或超轮次
    ③ 轨迹记录   AgentTrace（scripts/agent_trace.py）：每步一行，可回放、可统计
    ④ 产出校验   scripts/agent_tasks.py 的确定性 verifier
    ⑤ 上下文构建  build_context()：常驻规则档位（A/B/C）—— H2 实验的自变量

用法
    python scripts/agent_harness_min.py --list                     # 看任务集
    python scripts/agent_harness_min.py --client mock --repeat 3   # 不需要 key，验证装置
    python scripts/agent_harness_min.py --client deepseek --repeat 5 --task a1 --task a2
    python scripts/agent_harness_min.py --client deepseek --check  # 只测连通性

环境
    DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL（默认 https://api.deepseek.com）
    DEEPSEEK_MODEL（默认 deepseek-chat）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_tasks import Task, get_tasks, list_tasks          # noqa: E402
from agent_trace import AgentTrace                            # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════════════
# ① 工具接口
#    每个工具 = { schema（给模型看）, run(args, ws) -> (ok, text) }
#    🔴 失败必须返回**可行动的**错误信息（模型只能靠这句话自我修正）
# ══════════════════════════════════════════════════════════════════════


def _safe_join(ws: str, rel: str) -> str:
    """workspace 边界：不许 `..` 逃出去（harness 的第一条安全约束）。"""
    p = os.path.abspath(os.path.join(ws, rel))
    if not p.startswith(os.path.abspath(ws) + os.sep) and p != os.path.abspath(ws):
        raise ValueError(f"路径越出工作区：{rel}")
    return p


def _t_read_file(args: Dict[str, Any], ws: str) -> Tuple[bool, str]:
    rel = str(args.get("path", ""))
    try:
        p = _safe_join(ws, rel)
    except ValueError as e:
        return False, str(e)
    if not os.path.exists(p):
        return False, f"文件不存在：{rel}（先用 list_dir 看目录）"
    max_bytes = int(args.get("max_bytes", 20000))
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        return True, data[:max_bytes] + f"\n...（已截断，文件更大；用 max_bytes 调大或 grep）"
    return True, data


def _t_write_file(args: Dict[str, Any], ws: str) -> Tuple[bool, str]:
    rel = str(args.get("path", ""))
    content = args.get("content")
    if content is None:
        return False, "缺少 content 参数"
    try:
        p = _safe_join(ws, rel)
    except ValueError as e:
        return False, str(e)
    mode = args.get("mode", "overwrite")
    os.makedirs(os.path.dirname(p) or ws, exist_ok=True)
    if mode == "append":
        with open(p, "a", encoding="utf-8") as f:
            f.write(str(content))
    else:
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(content))
    return True, f"已写入 {rel}（{len(str(content))} 字符，mode={mode}）"


def _t_list_dir(args: Dict[str, Any], ws: str) -> Tuple[bool, str]:
    rel = str(args.get("path", "."))
    try:
        p = _safe_join(ws, rel)
    except ValueError as e:
        return False, str(e)
    if not os.path.isdir(p):
        return False, f"不是目录：{rel}"
    entries = sorted(os.listdir(p))
    return True, "\n".join(entries) if entries else "（空目录）"


def _t_run_command(args: Dict[str, Any], ws: str) -> Tuple[bool, str]:
    cmd = str(args.get("command", ""))
    if not cmd:
        return False, "缺少 command 参数"
    banned = ("rm -rf /", "format ", "shutdown")
    if any(b in cmd for b in banned):
        return False, "命令被 harness 拒绝执行"
    try:
        r = subprocess.run(cmd, shell=True, cwd=ws, capture_output=True,
                           text=True, timeout=int(args.get("timeout", 30)))
    except subprocess.TimeoutExpired:
        return False, f"命令超时（>{args.get('timeout', 30)}s）"
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    # 真模型第一次跑就发了 `ls -la ... | od -c`（POSIX 习惯），在 cmd.exe 下 rc=1 白烧两轮。
    # 错误信息必须**可行动**：直接告诉它本机是什么 shell、该换什么。
    if r.returncode != 0 and "is not recognized" in out:
        out += ("\n（本机 shell 是 Windows cmd.exe：无 ls/grep/cat/tail/od；"
                "请改用 dir / findstr / type，或直接用 read_file / list_dir）")
    return r.returncode == 0, f"rc={r.returncode}\n{out[:3000]}"


TOOLS: Dict[str, Dict[str, Any]] = {
    "read_file": {
        "run": _t_read_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取工作区内一个文本文件的内容（超长会截断）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作区的文件路径"},
                        "max_bytes": {"type": "integer", "description": "最多读取字节数，默认 20000"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "write_file": {
        "run": _t_write_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "把内容写入工作区内的文件。mode=overwrite 覆盖（默认），mode=append 追加",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "string", "enum": ["overwrite", "append"]},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "list_dir": {
        "run": _t_list_dir,
        "schema": {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "列出工作区内目录的文件",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    },
    "run_command": {
        "run": _t_run_command,
        "schema": {
            "type": "function",
            "function": {
                "name": "run_command",
                # 🔴 工具必须自述**自己所在的环境**：真模型第一次跑就发了 POSIX 命令（见 _t_run_command）。
                "description": "在工作区内执行 shell 命令并返回输出（用于跑测试、检索日志）。"
                               "⚠️ 本机 shell = Windows cmd.exe：没有 ls/grep/cat/tail/od，"
                               "请用 dir / findstr / type，或直接用 read_file / list_dir",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    },
}


def tool_schemas() -> List[Dict[str, Any]]:
    return [t["schema"] for t in TOOLS.values()]


# ══════════════════════════════════════════════════════════════════════
# ⑤ 上下文构建：常驻规则档位（H2 实验的自变量）
#     A=全量（复现现场） B=中间 C=最小
# ══════════════════════════════════════════════════════════════════════

RULE_LAYERS = {
    "global": (
        "【全局约定】\n"
        "- 回答用简体中文；先给结论再给说明，能不解释就不解释。\n"
        "- 产物必须落盘，不要只在回复里描述。\n"
        "- 改动前先读原文；不要臆测文件内容。\n"
        "- 遇到不确定的事，先查证再动手。\n"
    ),
    "workspace": (
        "【工作区约定】\n"
        "- 本工作区的任务都要用工具完成，不许只说「我会做」。\n"
        "- 写入文件前确认路径在工作区内。\n"
        "- 完成后必须自我检查一次产出。\n"
        "- 回答保持简短，不要复述任务。\n"
        "- 如果用了 run_command，必须看返回值再下结论。\n"
    ),
    "project": (
        "【项目约定】\n"
        "- 任务描述里的格式要求是硬约束，多一个字符都算失败。\n"
        "- 只写要求你写的内容，不要添加解释、标点或额外文件。\n"
    ),
}

# 技能按需注入：档位 A 全给，B/C 不给（H2 的另一个自变量）
SKILLS_TEXT = (
    "【技能清单（常驻）】\n"
    "- 文件操作技能：read_file / write_file / list_dir 的用法与边界。\n"
    "- 日志检索技能：大日志优先用 run_command + grep，不要整读。\n"
    "- JSON 技能：产出 JSON 前先读 schema，禁止额外键。\n"
    "- 测试技能：修代码后必须跑测试并把退出码写进结论。\n"
)


def build_context(level: str) -> Tuple[str, str]:
    """返回 (system_prompt, 档位描述)。level ∈ A/B/C。"""
    if level == "A":
        sp = RULE_LAYERS["global"] + "\n" + RULE_LAYERS["workspace"] + "\n" + \
             RULE_LAYERS["project"] + "\n" + SKILLS_TEXT
        return sp, "A=全量（全局+工作区+项目规则 + 全部技能）"
    if level == "B":
        sp = RULE_LAYERS["workspace"] + "\n" + RULE_LAYERS["project"]
        return sp, "B=中间（工作区+项目规则，无技能）"
    sp = RULE_LAYERS["project"]
    return sp, "C=最小（只有项目规则）"


# ══════════════════════════════════════════════════════════════════════
# 模型客户端（可插拔）
# ══════════════════════════════════════════════════════════════════════


class MockClient:
    """确定性脚本客户端：**只用来验证装置**（循环/工具/轨迹/判分），不代表模型能力。

    ⚠️ 它会主动完成每个任务（等价于"完美模型"），所以 mock 跑出的 acc 恒为 1.0。
       这是有意的：装置调试时不希望被模型能力干扰。

    设计要点：
      · **多 tool call**：a1/a2 的第一轮一次给两个调用（write 后立刻 read 自检），
        专门用来验证 harness 的"一轮执行全部 tool call"分支。
      · **故意失败一次**：a5 先发一个不存在的工具名，逼 harness 走
        "不存在的工具 → 不重试 → 让模型自己改"的分支。
    """

    name = "mock"

    def complete(self, messages, tools) -> Dict[str, Any]:
        user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        prompt = user["content"] if user else ""
        # 🔴 step 数「轮」而不是「tool 消息条数」：一轮里可以并发多个 tool call，
        #    用 tool 消息条数会在多调用轮之后错位（这正是本文件要演示的真实坑）。
        step = len([m for m in messages if m.get("role") == "assistant"])

        def done(text: str) -> Dict[str, Any]:
            return {"content": text, "tool_calls": [], "tokens_in": 0, "tokens_out": 0}

        def calls(*items) -> Dict[str, Any]:
            return {"content": "", "tokens_in": 0, "tokens_out": 0,
                    "tool_calls": [{"name": n, "arguments": a} for n, a in items]}

        # a1：追加一行 OK（第一轮就并发 write + read 自检）
        if "log.txt" in prompt and "追加" in prompt:
            if step == 0:
                return calls(("write_file", {"path": "log.txt", "content": "OK\n", "mode": "append"}),
                             ("read_file", {"path": "log.txt"}))
            return done("已追加 OK 并自检")
        # a2：读 → 修 → 跑测试（修完立刻读回来自检 = 两个 call）
        if "calc.py" in prompt:
            if step == 0:
                return calls(("read_file", {"path": "calc.py"}))
            if step == 1:
                return calls(("write_file", {"path": "calc.py",
                                             "content": "def add(a, b):\n    return a + b\n"}),
                             ("read_file", {"path": "calc.py"}))
            if step == 2:
                return calls(("run_command", {"command": "python test_calc.py"}))
            return done("已修复并跑通测试")
        # a3：找不一致
        if "cfg_a.json" in prompt:
            if step == 0:
                return calls(("run_command", {"command": "cat cfg_a.json cfg_b.json cfg_c.json"}))
            if step == 1:
                return calls(("write_file", {"path": "answer.txt", "content": "cfg_c.json"}))
            return done("已写出 answer.txt")
        # a4：schema 服从
        if "schema.json" in prompt:
            if step == 0:
                return calls(("read_file", {"path": "notes.txt"}),
                             ("read_file", {"path": "schema.json"}))
            if step == 1:
                return calls(("write_file", {"path": "tool_call.json",
                                             "content": json.dumps({"city": "杭州", "days": 3},
                                                                   ensure_ascii=False)}))
            return done("已按 schema 产出")
        # a5：先发一个不存在的工具（验证"不重试、让模型自纠"分支），再 grep
        if "server.log" in prompt:
            if step == 0:
                return calls(("search_log", {"pattern": "WARN"}))       # 故意不存在
            if step == 1:
                return calls(("run_command", {"command": "grep WARN server.log"}))
            if step == 2:
                return calls(("write_file", {"path": "answer.txt", "content": "ERR-7731"}))
            return done("已写出 code 值")
        return done("（mock 不认识这个任务）")



class DeepSeekClient:
    """真模型客户端：OpenAI 兼容的 /chat/completions（DeepSeek 官方即此协议）。

    🔴 解析策略：优先用原生 tool_calls；若模型把 tool call 写在正文里（JSON），
       走 `_parse_text_tool_call` 兜底 —— 这是 harness 必须处理的真实失败模式。
    """

    name = "deepseek"

    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def complete(self, messages, tools) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "tools": tools, "tool_choice": "auto"}
        data = self._post(payload)
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        calls = []
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__raw__": fn.get("arguments")}
            calls.append({"name": fn.get("name", ""), "arguments": args})
        content = msg.get("content") or ""
        if not calls and content:
            calls = _parse_text_tool_call(content)
        return {"content": content, "tool_calls": calls,
                # 思考模式的 CoT：请求带 tools 时必须回传（见 run_task 的 _asst）。
                # 🔴 用 None 表示"服务端没给这个字段"，"" 表示"给了但为空"—— 两者必须区分：
                #    实测 deepseek-flash 有时返回 content+tool_calls 而 reasoning_content 为空串，
                #    此时服务端**仍**要求该键存在，缺键照样 400。
                "reasoning_content": msg.get("reasoning_content"),
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0)}

    def check(self) -> str:
        data = self._post({"model": self.model, "messages":
                           [{"role": "user", "content": "ping"}], "max_tokens": 4})
        return json.dumps(data["choices"][0]["message"], ensure_ascii=False)[:200]


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _parse_text_tool_call(text: str) -> List[Dict[str, Any]]:
    """兜底：从正文里抠出 {"name": ..., "arguments": {...}} 形式。"""
    candidates = _FENCE_RE.findall(text)
    if not candidates:
        start, depth = None, 0
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments") or obj.get("args") or {}
        if name in TOOLS and isinstance(args, dict):
            return [{"name": name, "arguments": args}]
    return []


# ══════════════════════════════════════════════════════════════════════
# ② 控制循环（含三个 harness 必备件：多 tool call · 重试自愈 · 上下文预算）
# ══════════════════════════════════════════════════════════════════════


def est_tokens(text: str) -> int:
    """粗略 token 估算：CJK 约 1 token/字，ASCII 约 1 token/4 字符。

    🔴 不引入 tokenizer：harness 的预算控制要**便宜且在模型无关**下可用；
       误差用 `--ctx-budget` 留出的安全边际吸收。真实项目里应换成目标模型的 tokenizer。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


def msgs_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(est_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)


def trim_context(messages: List[Dict[str, Any]], budget: int,
                 keep_recent: int = 3) -> Tuple[List[Dict[str, Any]], int, int]:
    """上下文预算控制：超预算时**从最旧的工具结果开始压**（保留最近 N 条完整）。

    返回 (新 messages, 估算 tokens, 被裁掉条数)。
    策略选择理由：system/user 是任务本身不能动；assistant 的 tool_calls 删了会破坏
    协议配对；只有 tool 结果是可以安全替换成摘要的"可压缩内容"。
    """
    est = msgs_tokens(messages)
    if budget <= 0 or est <= budget:
        return messages, est, 0

    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    trimmed = 0
    for i in tool_idx[:-keep_recent] if keep_recent else tool_idx:
        content = str(messages[i].get("content", ""))
        if content.startswith("[已裁剪"):
            continue
        head = content[:200]
        messages[i] = dict(messages[i],
                           content=f"[已裁剪 原 {len(content)} 字符] {head}...")
        trimmed += 1
        est = msgs_tokens(messages)
        if est <= budget:
            break
    return messages, est, trimmed


class Harness:
    def __init__(self, client, trace: AgentTrace, context_level: str = "C",
                 max_turns: int = 8, rule_feedback: bool = True,
                 max_retries: int = 2, ctx_budget: int = 0,
                 keep_tool_results: int = 3, max_calls_per_turn: int = 4) -> None:
        self.client = client
        self.trace = trace
        self.system_prompt, self.context_desc = build_context(context_level)
        self.max_turns = max_turns
        self.rule_feedback = rule_feedback      # 失败时是否把 verifier 理由回灌（H2 的第二个旋钮）
        self.max_retries = max_retries          # 同一个工具失败后的重试次数（0=不重试）
        self.ctx_budget = ctx_budget            # 估算 token 预算，0=不裁剪
        self.keep_tool_results = keep_tool_results
        self.max_calls_per_turn = max_calls_per_turn

    # ───────────── 工具执行（带重试自愈）─────────────
    def _exec_one(self, turn: int, name: str, args: Dict[str, Any],
                  workspace: str) -> Tuple[bool, str, int]:
        """执行一个 tool call。返回 (ok, 结果文本, 实际尝试次数)。

        重试规则：
          · 工具**不存在** → 不重试（模型写错了名字，重试也不会变对，让它自己改）
          · 工具**执行失败** → 重试至 max_retries，每次退避 50ms × 尝试次数
          · 成功 → 立即返回
        """
        if name not in TOOLS:
            res = f"不存在的工具：{name}。可用工具：{list(TOOLS)}"
            self.trace.tool_call(turn, name, args, False, res, 0)
            return False, res, 1

        last = ""
        for attempt in range(1, self.max_retries + 2):
            tk0 = time.time()
            try:
                ok, result = TOOLS[name]["run"](args, workspace)
            except Exception as e:                            # noqa: BLE001
                ok, result = False, f"工具内部异常：{type(e).__name__}: {e}"
            ms = int((time.time() - tk0) * 1000)
            if ok or attempt > self.max_retries:
                self.trace.tool_call(turn, name, args, ok, "" if ok else result, ms)
                if attempt > 1:
                    self.trace.metric(turn, f"tool/{name}/retries", attempt - 1)
                return ok, result, attempt
            last = result
            self.trace.event(turn, "tool_retry", name=name, attempt=attempt,
                             error=str(result)[:120])
            time.sleep(0.05 * attempt)
        return False, last, self.max_retries + 1

    def run_task(self, task: Task, workspace: str, attempt: int = 1) -> Dict[str, Any]:
        t0 = time.time()
        self.trace.task_start(task.id, task.capability, attempt)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.prompt},
        ]
        verifier_reason = ""
        turn = 0
        n_calls = 0
        n_retries = 0
        tok_in = 0
        tok_out = 0

        while turn < self.max_turns:
            turn += 1

            # 每轮开始先做上下文预算检查（把最旧的工具结果压掉）
            messages, est, trimmed = trim_context(messages, self.ctx_budget,
                                                  self.keep_tool_results)
            self.trace.metric(turn, "ctx/est_tokens", est)
            if trimmed:
                self.trace.event(turn, "ctx_trimmed", dropped=trimmed, est_tokens=est)

            try:
                resp = self.client.complete(messages, tool_schemas())
            except Exception as e:                                # noqa: BLE001
                self.trace.event(turn, "model_error", error=str(e)[:200])
                break

            calls = (resp.get("tool_calls") or [])[:self.max_calls_per_turn]
            tok_in += int(resp.get("tokens_in") or 0)
            tok_out += int(resp.get("tokens_out") or 0)
            self.trace.metric(turn, "ctx/tok_in_cum", float(tok_in))
            self.trace.metric(turn, "ctx/tok_out_cum", float(tok_out))
            self.trace.model_out(turn, resp.get("content", ""),
                                 calls[0]["name"] if calls else None,
                                 resp.get("tokens_in", 0), resp.get("tokens_out", 0))
            if not calls:
                break

            # 一轮里的**全部** tool call 都要执行（多调用支持）
            # 🔴 顺序铁律：assistant(tool_calls) 必须**先于**它对应的 tool 结果入列。
            #    之前是先 append tool 再 append assistant，mock 不校验所以 7 个 run 全绿；
            #    真模型第一次跑就在第 2 轮被服务端拒绝：
            #    HTTP 400 "Messages with role 'tool' must be a response to a preceding
            #    message with 'tool_calls'" —— 这正是"假服务端测不出协议顺序"的实例。
            assistant_calls = [
                {"id": f"call_{turn}_{k}", "type": "function",
                 "function": {"name": c.get("name", ""),
                              "arguments": json.dumps(c.get("arguments") or {},
                                                      ensure_ascii=False)}}
                for k, c in enumerate(calls)
            ]
            _asst: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", ""),
                                     "tool_calls": assistant_calls}
            # 🔴 DeepSeek 思考模式 + tools：**上一轮的 reasoning_content 必须原样回传**，
            #    否则第二次请求 400 "The reasoning_content in the thinking mode must be
            #    passed back to the API."（mock 无此字段，所以同样是只有真模型才暴露的坑）
            #    ⚠️ 判据是"服务端给过这个键"（is not None），**不是**"值非空"：
            #       实测有响应返回空串 reasoning_content，缺键仍然 400。
            if resp.get("reasoning_content") is not None:
                _asst["reasoning_content"] = resp.get("reasoning_content") or ""
            messages.append(_asst)
            for k, c in enumerate(calls):
                name, args = c.get("name", ""), (c.get("arguments") or {})
                cid = f"call_{turn}_{k}"
                ok, result, tries = self._exec_one(turn, name, args, workspace)
                n_calls += 1
                n_retries += max(0, tries - 1)
                messages.append({"role": "tool", "tool_call_id": cid, "name": name,
                                 "content": ("OK\n" if ok else "ERROR\n") + result})

        # ── ④ 产出校验（确定性 verifier）──
        try:
            passed, verifier_reason = task.verify(workspace)
        except Exception as e:                                    # noqa: BLE001
            passed, verifier_reason = False, f"verifier 异常：{type(e).__name__}: {e}"

        seconds = time.time() - t0
        self.trace.metric(turn, "loop/n_tool_calls", float(n_calls))
        self.trace.metric(turn, "loop/n_retries", float(n_retries))
        self.trace.metric(turn, "ctx/tok_in", float(tok_in))
        self.trace.metric(turn, "ctx/tok_out", float(tok_out))
        self.trace.task_end(task.id, passed, verifier_reason, turn, seconds)
        if not passed and self.rule_feedback:
            self.trace.event(turn, "verifier_feedback", reason=verifier_reason[:200])
        return {"task": task.id, "passed": passed, "reason": verifier_reason,
                "turns": turn, "seconds": round(seconds, 2),
                "tool_calls": n_calls, "retries": n_retries,
                "tok_in": tok_in, "tok_out": tok_out}


def estimate_run_cost(client, tasks: List[Task], repeat: int, price_in: float,
                      price_out: float, ctx_level: str, args) -> Dict[str, Any]:
    """**先花钱后跑**：用 1 次真实调用校准「每任务 token 量级」，再外推整轮成本。

    🔴 为什么值得单独做：A/B/C × 6 次 = 90 次任务调用，DeepSeek 是**付费 API**。
       先报预算再决定跑不跑，是工程习惯，也是面试常问的"你怎么控制成本"。
    """
    tb, _ = build_context(ctx_level)
    task = tasks[0]
    padding = "填充。" * max(0, args.ctx_budget // 3) if args.ctx_budget else ""
    msgs = [{"role": "system", "content": tb},
            {"role": "user", "content": task.prompt + padding}]
    resp = client.complete(msgs, tool_schemas())
    ti = int(resp.get("tokens_in") or 0)
    to = int(resp.get("tokens_out") or 0)
    if ti == 0:
        ti, to = est_tokens(msgs[0]["content"] + msgs[1]["content"]), 80
        src = "（服务端未返回 usage，用估算值）"
    else:
        src = "（服务端真实 usage）"
    per_task_in = ti * 3          # 多轮里输入会被反复重发，3 是保守下限
    per_task_out = max(to, 60) * 3
    n = len(tasks) * repeat
    cost = (per_task_in * n * price_in + per_task_out * n * price_out) / 1e6
    return {"calib_tokens_in": ti, "calib_tokens_out": to, "source": src,
            "n_calls_est": n, "tok_in_est": per_task_in * n, "tok_out_est": per_task_out * n,
            "cost_est": cost}



# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def make_client(kind: str, args) -> Any:
    if kind == "mock":
        return MockClient()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("❌ 未设置 DEEPSEEK_API_KEY（可先用 --client mock 验证装置）", file=sys.stderr)
        sys.exit(2)
    return DeepSeekClient(
        model=args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=args.base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=key, temperature=args.temperature)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="最小 agent harness")
    ap.add_argument("--list", action="store_true", help="列出任务集")
    ap.add_argument("--client", default="mock", choices=["mock", "deepseek"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--task", action="append", default=None, help="只跑指定任务（可多次）")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--context", default="C", choices=["A", "B", "C"], help="常驻规则档位（H2 自变量）")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=2,
                    help="同一个工具失败后的重试次数（0=不重试）")
    ap.add_argument("--ctx-budget", type=int, default=0,
                    help="估算 token 预算，超出则裁剪最旧的工具结果（0=不裁剪）")
    ap.add_argument("--keep-tool-results", type=int, default=3,
                    help="裁剪时保留最近 N 条工具结果完整")
    ap.add_argument("--max-calls-per-turn", type=int, default=4,
                    help="单轮最多执行几个 tool call")
    ap.add_argument("--out", default=os.path.join(PROJ, "runs", "harness"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--keep-workspace", action="store_true")
    ap.add_argument("--check", action="store_true", help="只测模型连通性")
    # 成本控制：默认 0 = 不估（真实单价请按所用模型填，这里不写死任何厂商价格）
    ap.add_argument("--price-in", type=float, default=0.0, help="输入价（元/百万 token）")
    ap.add_argument("--price-out", type=float, default=0.0, help="输出价（元/百万 token）")
    ap.add_argument("--estimate-cost", action="store_true",
                    help="只做 1 次校准调用，外推整轮 token 与费用后退出（不跑评测）")
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'id':4s} {'能力':10s} 说明")
        for t in list_tasks():
            print(f"{t.id:4s} {t.capability:10s} {t.why}")
        print(f"\n共 {len(list_tasks())} 个任务，判分器全部确定性（见 scripts/agent_tasks.py）")
        return 0

    client = make_client(args.client, args)
    if args.check:
        print(f"模型：{getattr(client, 'model', 'mock')}")
        if hasattr(client, "check"):
            print("连通性 OK：", client.check())
        else:
            print("mock 客户端无需连通性检查")
        return 0

    tasks = get_tasks(args.task)
    if not tasks:
        print("❌ 没有匹配的任务", file=sys.stderr)
        return 2

    # ── 成本预估（只花 1 次调用）──
    if args.estimate_cost:
        try:
            est = estimate_run_cost(client, tasks, args.repeat,
                                    args.price_in, args.price_out, args.context, args)
        except Exception as e:                                  # noqa: BLE001
            print(f"❌ 校准调用失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 3
        print("═══ 成本预估（先花钱后跑）═══")
        print(f"校准调用：输入 {est['calib_tokens_in']} tok / 输出 {est['calib_tokens_out']} tok "
              f"{est['source']}")
        print(f"预计规模：{len(tasks)} 任务 × {args.repeat} 次 = {est['n_calls_est']} 次任务调用")
        print(f"预计 token：输入 ≈{est['tok_in_est']:,} / 输出 ≈{est['tok_out_est']:,}"
              f"（按 每任务 3 轮 保守外推）")
        if args.price_in or args.price_out:
            print(f"预计费用：**{est['cost_est']:.4f} 元**"
                  f"（按 输入 {args.price_in}/M · 输出 {args.price_out}/M）")
        else:
            print("预计费用：未给单价（--price-in/--price-out），只报 token 量")
        return 0

    run_id = args.run_id or f"h{args.context}_{args.client}_{time.strftime('%m%d_%H%M%S')}"
    trace = AgentTrace(run_id, args.out, meta={
        "client": args.client, "model": getattr(client, "model", "mock"),
        "context_level": args.context, "context_desc": "",
        "max_turns": args.max_turns, "repeat": args.repeat,
        "tasks": [t.id for t in tasks], "temperature": args.temperature,
        "max_retries": args.max_retries, "ctx_budget": args.ctx_budget,
        "keep_tool_results": args.keep_tool_results,
        "max_calls_per_turn": args.max_calls_per_turn,
    })
    harness = Harness(client, trace, context_level=args.context, max_turns=args.max_turns,
                      max_retries=args.max_retries, ctx_budget=args.ctx_budget,
                      keep_tool_results=args.keep_tool_results,
                      max_calls_per_turn=args.max_calls_per_turn)
    trace.meta["context_desc"] = harness.context_desc

    print(f"═══ agent harness ═══")
    print(f"run_id   : {run_id}")
    print(f"client   : {args.client}（{getattr(client, 'model', 'mock')}）")
    print(f"上下文档 : {harness.context_desc}")
    print(f"任务     : {[t.id for t in tasks]} × {args.repeat} 次")
    print(f"harness  : 重试≤{args.max_retries} · 上下文预算={args.ctx_budget or '不限'}"
          f" · 单轮≤{args.max_calls_per_turn} 个 tool call")
    print(f"产物     : {os.path.join(args.out, 'obs', run_id)}")
    print("─" * 60)

    results = []
    for attempt in range(1, args.repeat + 1):
        for t in tasks:
            ws = os.path.join(args.out, "_ws", f"{run_id}_{t.id}_{attempt}")
            if os.path.exists(ws):
                shutil.rmtree(ws, ignore_errors=True)
            os.makedirs(ws, exist_ok=True)
            if t.setup:
                t.setup(ws)
            r = harness.run_task(t, ws, attempt)
            results.append(r)
            flag = "✅" if r["passed"] else "❌"
            print(f"{flag} {t.id} #{attempt} | turns={r['turns']} | {r['seconds']}s | {r['reason'][:70]}")
            if not args.keep_workspace:
                shutil.rmtree(ws, ignore_errors=True)

    # ── token / 成本核算 ──
    tok_in = sum(r.get("tok_in", 0) for r in results)
    tok_out = sum(r.get("tok_out", 0) for r in results)
    cost = (tok_in * args.price_in + tok_out * args.price_out) / 1e6
    trace.metric(0, "ctx/tok_in_total", float(tok_in))
    trace.metric(0, "ctx/tok_out_total", float(tok_out))
    if args.price_in or args.price_out:
        trace.metric(0, "ctx/cost_cny", float(cost))

    summary = trace.finish({"tok_in_total": tok_in, "tok_out_total": tok_out,
                            "price_in": args.price_in, "price_out": args.price_out,
                            "cost_cny": round(cost, 6)})
    print("─" * 60)
    print(f"通过 {summary['n_passed']}/{summary['n_tasks']} = {summary['acc']*100:.1f}%"
          f" | 平均轮数 {summary['avg_turns']} | 墙钟 {summary['wall_s']}s")
    if tok_in or tok_out:
        msg = f"token：输入 {tok_in:,} / 输出 {tok_out:,}"
        if args.price_in or args.price_out:
            msg += f" | 实际花费 {cost:.4f} 元"
        print(msg)
    print(f"轨迹：{os.path.join(args.out, run_id + '.log')}（grep 'TRACE' 可回放）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
