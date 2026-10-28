# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-01 11:07 · dsh-agent ｜ 变更：新增：DeepSeekClient 契约回归（本地假服务端，7 用例，零 API 花费）

# 🕒 v1 · 2026-10-01 · dsh-agent ｜ 变更：新增 DeepSeekClient 契约回归（本地假服务端，零 API 花费）

"""DeepSeekClient 的契约回归测试：**不起真 API、不花一分钱**验证请求/解析逻辑。

🔴 为什么需要一个假服务端
    `DeepSeekClient` 只有在真 key 下才会被执行到；没有 key 时它就是**一行都没跑过的代码**。
    而它承载了 harness 最脆弱的部分：请求体形状、tool_calls 解析、正文 JSON 兜底、
    usage 记账、HTTP 错误处理。用一个本地 HTTP 假服务端就能把这些全部覆盖。

覆盖用例
    ① 原生 tool_calls  → 正确解析出 name/arguments
    ② 正文 JSON 兜底    → tool_call 写在 content 里（fenced 与裸 JSON 各一）
    ③ 纯文本回复        → tool_calls 为空、content 保留
    ④ usage 记账        → tokens_in / tokens_out 透传
    ⑤ HTTP 401         → 抛 RuntimeError 且带状态码
    ⑥ 请求体形状        → model / messages / tools / tool_choice / temperature 都在

运行：python scripts/test_deepseek_client.py     （无外部依赖，除标准库 + requests）
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_harness_min import DeepSeekClient, tool_schemas   # noqa: E402

_CAPTURED: list = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # 安静
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _CAPTURED.append(body)
        mode = self.headers.get("X-Test-Mode", "native")

        if mode == "http401":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "bad key"}).encode())
            return

        if mode == "native":
            msg = {"role": "assistant", "content": "",
                   "tool_calls": [{"id": "c1", "type": "function",
                                   "function": {"name": "write_file",
                                                "arguments": json.dumps({"path": "a.txt",
                                                                         "content": "OK"})}}]}
        elif mode == "fenced":
            msg = {"role": "assistant",
                   "content": '我来写文件：\n```json\n{"name": "write_file", '
                              '"arguments": {"path": "a.txt", "content": "OK"}}\n```'}
        elif mode == "bare":
            msg = {"role": "assistant",
                   "content": '{"name": "list_dir", "arguments": {"path": "."}}'}
        else:
            msg = {"role": "assistant", "content": "已完成，无需工具。"}

        resp = {"choices": [{"message": msg}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve() -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    srv = _serve()
    base = f"http://127.0.0.1:{srv.server_port}"
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{'✅' if cond else '❌'} {name}" + (f"  | {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    # ① 原生 tool_calls
    c = DeepSeekClient(model="test-model", base_url=base, api_key="k", temperature=0.0)
    c._post_headers_hack = None
    r = c.complete(msgs, tool_schemas())
    check("① 原生 tool_calls 解析", bool(r["tool_calls"]) and
          r["tool_calls"][0]["name"] == "write_file" and
          r["tool_calls"][0]["arguments"].get("path") == "a.txt",
          json.dumps(r["tool_calls"][0], ensure_ascii=False)[:80])

    # ④ usage 记账 + ⑥ 请求体形状
    check("④ usage 记账（in/out）", r["tokens_in"] == 123 and r["tokens_out"] == 45,
          f"in={r['tokens_in']} out={r['tokens_out']}")
    sent = _CAPTURED[-1]
    check("⑥ 请求体形状", sent.get("model") == "test-model" and "tools" in sent
          and sent.get("tool_choice") == "auto" and sent.get("temperature") == 0.0
          and isinstance(sent.get("messages"), list),
          f"keys={sorted(sent)}")

    # ② 正文 JSON 兜底（fenced / bare）—— 通过自定义 header 切换假服务端行为
    class _Fenced(DeepSeekClient):
        pass

    # 直接改假服务端行为：用 header 注入，所以这里手工发一次
    import requests
    def raw_call(mode: str) -> dict:
        payload = {"model": "test-model", "messages": msgs,
                   "temperature": 0.0, "tools": tool_schemas(), "tool_choice": "auto"}
        rr = requests.post(f"{base}/chat/completions", json=payload,
                           headers={"Content-Type": "application/json",
                                    "Authorization": "Bearer k", "X-Test-Mode": mode})
        return rr.json()

    # 用同一个 client 的解析路径：临时把假服务端行为换成 fenced
    orig = _Handler.do_POST

    def patched(self):
        self.headers.__class__ = type(self.headers)      # no-op，保持接口
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _CAPTURED.append(body)
        msg = {"role": "assistant",
               "content": '```json\n{"name": "write_file", "arguments": {"path": "b.txt"}}\n```'}
        data = json.dumps({"choices": [{"message": msg}],
                           "usage": {"prompt_tokens": 9, "completion_tokens": 9}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    _Handler.do_POST = patched
    r2 = c.complete(msgs, tool_schemas())
    check("② fenced 正文 JSON 兜底", bool(r2["tool_calls"]) and
          r2["tool_calls"][0]["name"] == "write_file", json.dumps(r2["tool_calls"], ensure_ascii=False)[:80])

    def patched_bare(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _CAPTURED.append(body)
        msg = {"role": "assistant", "content": '{"tool": "list_dir", "args": {"path": "."}}'}
        data = json.dumps({"choices": [{"message": msg}],
                           "usage": {"prompt_tokens": 9, "completion_tokens": 9}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    _Handler.do_POST = patched_bare
    r3 = c.complete(msgs, tool_schemas())
    check("②b 裸 JSON 兜底（tool/args 别名）", bool(r3["tool_calls"]) and
          r3["tool_calls"][0]["name"] == "list_dir", json.dumps(r3["tool_calls"], ensure_ascii=False)[:80])

    def patched_text(self):
        n = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(n) or b"{}")
        msg = {"role": "assistant", "content": "已完成，无需工具。"}
        data = json.dumps({"choices": [{"message": msg}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 6}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    _Handler.do_POST = patched_text
    r4 = c.complete(msgs, tool_schemas())
    check("③ 纯文本回复", r4["tool_calls"] == [] and "已完成" in r4["content"])

    # ⑤ HTTP 401
    _Handler.do_POST = orig
    bad = DeepSeekClient(model="test-model", base_url=base, api_key="bad")
    try:
        import requests as _rq

        def patched_401(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')

        _Handler.do_POST = patched_401
        bad.complete(msgs, tool_schemas())
        check("⑤ HTTP 401 抛错", False, "没有抛错")
    except RuntimeError as e:
        check("⑤ HTTP 401 抛错", "401" in str(e), str(e)[:60])
    finally:
        _Handler.do_POST = orig

    srv.shutdown()
    print("─" * 60)
    if failures:
        print(f"❌ 失败 {len(failures)} 项：{failures}")
        return 1
    print("✅ DeepSeekClient 契约全部通过（未消耗任何真实 API 额度）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
