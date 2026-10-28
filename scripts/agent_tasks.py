# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-01 11:04 · dsh-agent ｜ 变更：新增：5 个 agent 任务 + 确定性 verifier

"""Agent 任务集 + **确定性 verifier**（agent harness 的判分地基）。

🔴 为什么判分器必须确定性（`docs/实验_上下文注入AB_设计.md` §2.3）
    "看起来对不对"不可复现，同一份产出必须永远给同样的分。
    模型裁判（LLM-as-judge）是另一件事，不许混进来。

契约（与 `scripts/eval_set.py` 保持一致，换任务只改这一处的 `TASKS`）
    每个任务 = { id, prompt, setup, verify(workspace) -> (passed: bool, reason: str) }
    · `setup(ws)` 造初始现场（可能不存在）
    · `verify(ws)` 只读 workspace，返回 (bool, 理由)

任务类型（覆盖 agent harness 的核心失败面）
    a1 精确写入      —— 格式服从
    a2 修 bug        —— 读代码 + 回归验证（verifier 真跑测试）
    a3 找不一致      —— 跨文件比较 + **只报告文件名**（禁止型约束）
    a4 结构化输出    —— tool-call schema 服从（禁止型）
    a5 长日志取字段  —— 长上下文 + 精确定位（禁止型）
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════
# 工具函数（都是确定性的）
# ══════════════════════════════════════════════════════════════════════


def _read(ws: str, rel: str) -> str:
    p = os.path.join(ws, rel)
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _write(ws: str, rel: str, text: str) -> None:
    p = os.path.join(ws, rel)
    os.makedirs(os.path.dirname(p) or ws, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def _run(ws: str, cmd: str, timeout: int = 30) -> Tuple[int, str]:
    r = subprocess.run(cmd, shell=True, cwd=ws, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


# ══════════════════════════════════════════════════════════════════════
# 任务定义
# ══════════════════════════════════════════════════════════════════════


# ---------- a1：精确写入 ----------
def _a1_setup(ws: str) -> None:
    _write(ws, "log.txt", "第一行\n")


def _a1_verify(ws: str) -> Tuple[bool, str]:
    p = os.path.join(ws, "log.txt")
    if not os.path.exists(p):
        return False, "log.txt 不存在"
    lines = _read(ws, "log.txt").splitlines()
    if len(lines) != 2:
        return False, f"期望 2 行，实际 {len(lines)} 行"
    if lines[1] != "OK":
        return False, f"第 2 行应为 'OK'，实际 {lines[1]!r}"
    return True, "第 2 行精确等于 OK"


# ---------- a2：修 bug（verifier 真跑测试）----------
_BUGGY = '''def add(a, b):
    return a - b          # BUG
'''

_BUG_TEST = '''from calc import add
assert add(2, 3) == 5, "add(2,3) 应为 5"
assert add(-1, 1) == 0, "add(-1,1) 应为 0"
print("TESTS PASSED")
'''


def _a2_setup(ws: str) -> None:
    _write(ws, "calc.py", _BUGGY)
    _write(ws, "test_calc.py", _BUG_TEST)


def _a2_verify(ws: str) -> Tuple[bool, str]:
    try:
        rc, out = _run(ws, "python test_calc.py")
    except subprocess.TimeoutExpired:
        return False, "测试超时"
    if rc != 0:
        return False, f"测试失败(rc={rc})：{out.strip()[:120]}"
    return "TESTS PASSED" in out, out.strip()[:120] or "测试通过"


# ---------- a3：找不一致，只报告文件名 ----------
def _a3_setup(ws: str) -> None:
    _write(ws, "cfg_a.json", json.dumps({"port": 8080, "host": "a"}, indent=2))
    _write(ws, "cfg_b.json", json.dumps({"port": 8080, "host": "b"}, indent=2))
    _write(ws, "cfg_c.json", json.dumps({"port": 9090, "host": "c"}, indent=2))


def _a3_verify(ws: str) -> Tuple[bool, str]:
    """答案文件必须是 `answer.txt`，内容恰好是 `cfg_c.json`（允许前后空白）。"""
    p = os.path.join(ws, "answer.txt")
    if not os.path.exists(p):
        return False, "answer.txt 不存在"
    got = _read(ws, "answer.txt").strip()
    if got != "cfg_c.json":
        return False, f"应为 'cfg_c.json'（只写文件名），实际 {got!r}"
    return True, "精确报告了不一致的文件名"


# ---------- a4：tool-call schema 服从（禁止型）----------
_A4_SCHEMA = {
    "type": "object",
    "required": ["city", "days"],
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "minimum": 1, "maximum": 14},
    },
    "additionalProperties": False,
}


def _a4_setup(ws: str) -> None:
    _write(ws, "schema.json", json.dumps(_A4_SCHEMA, indent=2, ensure_ascii=False))
    _write(ws, "notes.txt", "用户说：下周去杭州开会，来回一共待 3 天。\n")


def _a4_verify(ws: str) -> Tuple[bool, str]:
    p = os.path.join(ws, "tool_call.json")
    if not os.path.exists(p):
        return False, "tool_call.json 不存在"
    raw = _read(ws, "tool_call.json")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"不是合法 JSON：{e}"
    if not isinstance(data, dict):
        return False, "顶层必须是对象"
    extra = set(data) - set(_A4_SCHEMA["properties"])
    if extra:
        return False, f"出现 schema 禁止的额外键：{sorted(extra)}"
    for k in _A4_SCHEMA["required"]:
        if k not in data:
            return False, f"缺少必填键：{k}"
    if data.get("city") != "杭州":
        return False, f"city 应为 '杭州'，实际 {data.get('city')!r}"
    if data.get("days") != 3:
        return False, f"days 应为整数 3，实际 {data.get('days')!r}"
    return True, "schema 完全服从（无额外键、类型正确）"


# ---------- a5：长日志取字段（长上下文）----------
_A5_NEEDLE = "ERR-7731"


def _a5_setup(ws: str) -> None:
    lines = []
    for i in range(4000):
        if i == 2717:
            lines.append(f"2026-09-30 10:00:00 WARN node=a800gpu01 code={_A5_NEEDLE} msg=heartbeat timeout")
        else:
            lines.append(f"2026-09-30 10:00:{i % 60:02d} INFO node=cpu{i % 9:02d} code=OK msg=tick {i}")
    _write(ws, "server.log", "\n".join(lines) + "\n")


def _a5_verify(ws: str) -> Tuple[bool, str]:
    p = os.path.join(ws, "answer.txt")
    if not os.path.exists(p):
        return False, "answer.txt 不存在"
    got = _read(ws, "answer.txt").strip()
    if got != _A5_NEEDLE:
        return False, f"应为 {_A5_NEEDLE!r}，实际 {got!r}"
    return True, "从 4000 行日志中精确取出了 code"


# ══════════════════════════════════════════════════════════════════════
# 注册表
# ══════════════════════════════════════════════════════════════════════


class Task:
    def __init__(self, tid: str, prompt: str, setup: Optional[Callable[[str], None]],
                 verify: Callable[[str], Tuple[bool, str]], capability: str,
                 why: str = "") -> None:
        self.id = tid
        self.prompt = prompt
        self.setup = setup
        self.verify = verify
        self.capability = capability
        self.why = why


TASKS: Dict[str, Task] = {
    "a1": Task(
        "a1",
        "在文件 `log.txt` 的末尾**追加一行**，内容严格为 `OK`（不要多余空格、不要引号）。"
        "完成后简要说明你做了什么。",
        _a1_setup, _a1_verify, "format",
        "测格式服从：多写一个字符就失败",
    ),
    "a2": Task(
        "a2",
        "当前目录下有 `calc.py` 和 `test_calc.py`。`test_calc.py` 会失败，"
        "请修复 `calc.py` 让测试通过（不要改测试文件）。完成后说明改了什么。",
        _a2_setup, _a2_verify, "debug",
        "测读代码+回归验证：verifier 真跑测试",
    ),
    "a3": Task(
        "a3",
        "当前目录有 `cfg_a.json` / `cfg_b.json` / `cfg_c.json`，其中**恰好有一个**与另外两个不一致。"
        "把**不一致的那个文件名**写入 `answer.txt` —— **只写文件名本身**，不要写路径、不要任何解释或标点。",
        _a3_setup, _a3_verify, "constraint",
        "测禁止型约束：多写一个字就失败",
    ),
    "a4": Task(
        "a4",
        "读 `schema.json` 与 `notes.txt`，按 notes 里的信息产出一个 flight_search 的调用参数，"
        "写入 `tool_call.json`。必须**严格**符合 schema：不许出现 schema 未定义的键。",
        _a4_setup, _a4_verify, "constraint",
        "测结构化输出服从：schema 是硬约束",
    ),
    "a5": Task(
        "a5",
        "`server.log` 有 4000 行。找出唯一一行 `WARN`，把它的 `code=` 值写入 `answer.txt`"
        "（**只写值本身**，不要写 `code=`、不要引号）。",
        _a5_setup, _a5_verify, "longctx",
        "测长上下文定位：4000 行里找一根针",
    ),
}


def list_tasks() -> List[Task]:
    return list(TASKS.values())


def get_tasks(ids: Optional[List[str]] = None) -> List[Task]:
    if not ids:
        return list_tasks()
    return [TASKS[i] for i in ids if i in TASKS]
