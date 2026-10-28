# -*- coding: utf-8 -*-
# 🕒 v2 · 2026-09-30 22:01 · dsh-agent ｜ 变更：评测指标唯一来源改为 eval.json（修 test/* 重复两条）；日志里的测试汇总只记 meta ｜ 上一版：v1 · 2026-09-30 21:08

"""把 RL_infra 的 `train.log` 转成**通用观测契约**（run.json / metrics.jsonl / events.jsonl）。

为什么需要（2026-09-30）：
    训练日志是给人看的文本（日志契约见 `rlinfra/logger.py`），而观测平台要的是
    逐行 JSON 时序。两者之间需要一个**只读日志、只写新文件**的翻译层，
    这样换任何观测前端都不用动训练代码。

契约（与 `UAV-area-search-crosscheck/infra/telemetry.py` 的读侧一致）：
    runs/<run_id>/run.json        {"run_id","status","created_at",...}
    runs/<run_id>/metrics.jsonl   每行 {"ts","step","name","value"}
    runs/<run_id>/events.jsonl    每行 {"ts","step","type","payload"}

用法：
    python utils/log_to_metrics.py --log runs/demo_300/train.log [--out runs/obs] [--force]
    # 输出目录默认是 <日志同级目录的父目录>/obs/<run_id>/，可用 --out 改

🔴 边界：**不覆盖 train.log、不动 checkpoint**；目标目录已存在时须显式 `--force`。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

EP_RE = re.compile(
    r"^EP\s*\|\s*ep:\s*(?P<ep>\d+)\s*\|\s*R:\s*(?P<R>-?[\d.]+)\s*\|\s*cov:\s*(?P<cov>[\d.]+)%\s*\|"
    r"\s*steps:\s*(?P<steps>\d+)\s*\|\s*loss:\s*(?P<loss>-?[\d.]+)\s*\|\s*curQ:\s*(?P<curq>-?[\d.]+)\s*\|"
    r"\s*td:\s*(?P<td>-?[\d.]+)")
TRAIN_RE = re.compile(
    r"^TRAIN-\w+\s*\|\s*ep:\s*(?P<ep>\d+)\s*\|\s*loss:\s*(?P<loss>-?[\d.]+)\s*\|\s*curQ:\s*(?P<curq>-?[\d.]+)\s*\|"
    r"\s*tgtQ:\s*(?P<tgtq>-?[\d.]+)\s*\|\s*eps:\s*(?P<eps>[\d.]+)\s*\|\s*td:\s*(?P<td>-?[\d.]+)")
REWARD_RE = re.compile(r"^REWARD\(每智能体每步均值\)\s*\|\s*(?P<body>.+)$")
REWARD_COMP_RE = re.compile(r"(?P<k>[\w\u4e00-\u9fa5]+):(?P<v>[+-]?\d+\.\d+)")
BEST_RE = re.compile(
    r"已更新\s*\w+\s*最优模型.*?回合：\s*(?P<ep>\d+).*?全局奖励：\s*(?P<R>-?[\d.]+)"
    r".*?覆盖率：\s*(?P<cov>[\d.]+)%")
TEST_RE = re.compile(
    r"测试汇总\s*\|\s*全局平均覆盖率：\s*(?P<cov>[\d.]+)%\s*\|\s*全局平均奖励：\s*(?P<R>-?[\d.]+)\s*\|"
    r"\s*成功率：\s*(?P<succ>[\d.]+)%\s*\|\s*平均完成步数：\s*(?P<steps>[\d.]+)")
IDENT_RE = re.compile(r"链路标识\s*\|\s*test_mode=(?P<algo>\w+)")   # 算法名
EPISODES_RE = re.compile(r"总轮次\s*[=:：]\s*(?P<n>\d+)")             # 总步数


def parse_log(path: str, run_id: str) -> Dict[str, Any]:
    metrics: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            m = EP_RE.match(line)
            if m:
                ep = int(m.group("ep"))
                for key, name in (("R", "ep_reward"), ("cov", "coverage"), ("steps", "steps"),
                                  ("loss", "loss"), ("curq", "cur_q"), ("td", "td")):
                    metrics.append({"ts": t0, "step": ep, "name": name, "value": float(m.group(key))})
                continue
            m = TRAIN_RE.match(line)
            if m:
                ep = int(m.group("ep"))
                for key, name in (("loss", "loss"), ("curq", "cur_q"), ("tgtq", "tgt_q"),
                                  ("eps", "epsilon"), ("td", "td")):
                    metrics.append({"ts": t0, "step": ep, "name": name, "value": float(m.group(key))})
                continue
            m = REWARD_RE.match(line)
            if m:
                for k, v in REWARD_COMP_RE.findall(m.group("body")):
                    metrics.append({"ts": t0, "step": 0, "name": f"reward/{k}", "value": float(v)})
                continue
            m = TEST_RE.search(line)
            if m:
                # 🔴 只记进 meta，**不写 metrics**（2026-09-30）：评测指标的唯一来源是
                # `runs/<name>/eval.json`（下面统一追加）。否则同一份评测会被记两遍
                # —— 本机实测出现过 test/coverage 重复两条，平台图上就是重复点。
                meta["test_from_log"] = {k: float(m.group(k)) for k in ("cov", "R", "succ", "steps")}
                continue
            m = BEST_RE.search(line)
            if m:
                events.append({"ts": t0, "step": int(m.group("ep")), "type": "best_model",
                               "payload": {"reward": float(m.group("R")),
                                           "coverage": float(m.group("cov"))}})
                continue
            if line.startswith("运行设备"):
                meta["device_line"] = line.strip()
                continue
            m = IDENT_RE.search(line)                 # 链路标识 | test_mode=madqn | ...
            if m:
                meta["algo"] = m.group("algo")
                continue
            m = EPISODES_RE.search(line)              # 总轮次=300 / 总轮次 : 300
            if m:
                meta["total_episodes"] = int(m.group("n"))
                continue
    meta["run_id"] = run_id
    # F（2026-09-30）：评测结果回写文件 `runs/<name>/eval.json`（由 scripts/eval_runs.sh 产出）。
    # 有了它就不必再从 output.<jobid> 里 grep —— 评测完直接 sync_hpc_runs.py 就能带出 test/* 指标。
    ev = os.path.join(os.path.dirname(os.path.abspath(path)), "eval.json")
    if os.path.isfile(ev):
        try:
            with open(ev, encoding="utf-8") as fh:
                e = json.load(fh)
            meta["eval"] = e
            for key, nm in (("coverage", "test/coverage"),
                            ("coverage_std", "test/coverage_std"),
                            ("reward", "test/reward"),
                            ("success_rate", "test/success_rate"), ("steps", "test/steps")):
                if e.get(key) is not None:
                    metrics.append({"ts": t0, "step": 0, "name": nm, "value": float(e[key])})
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"⚠️ eval.json 读取失败（忽略）：{exc}")
    return {"metrics": metrics, "events": events, "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="RL_infra 的 train.log")
    ap.add_argument("--out", default=None, help="输出根目录（默认 runs/obs）")
    ap.add_argument("--run-id", default=None, help="默认取日志所在目录名")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log = os.path.abspath(args.log)
    if not os.path.isfile(log):
        print(f"❌ 日志不存在：{log}")
        return 2
    run_id = args.run_id or os.path.basename(os.path.dirname(log))
    out_root = args.out or os.path.join(os.path.dirname(os.path.dirname(log)), "obs")
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    mpath = os.path.join(out_dir, "metrics.jsonl")
    if os.path.exists(mpath) and not args.force:
        print(f"❌ 已存在 {mpath}（要覆盖加 --force）")
        return 3

    p = parse_log(log, run_id)
    if not p["metrics"]:
        print("❌ 没解析出任何指标 —— 日志格式与契约不符（先查格式，别入库）")
        return 4

    # 原子落盘：先写 .tmp 再换名（与 checkpoint 同一约定，读者不会看到半个文件）
    def _dump(path_: str, records: List[Dict[str, Any]], obj: bool = False) -> None:
        tmp = f"{path_}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            if obj:
                json.dump(records, fh, ensure_ascii=False, indent=2)
            else:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path_)

    _dump(mpath, p["metrics"])
    _dump(os.path.join(out_dir, "events.jsonl"), p["events"])
    now = time.time()
    meta = dict(p["meta"])
    meta.update({"status": "finished", "created_at": now, "finished_at": now,
                 "source_log": log, "metric_count": len(p["metrics"]),
                 "event_count": len(p["events"])})
    # ── 观测平台前端读的字段名（web/static/app.js:35-38）──
    # 它找的是 algorithm / total_timesteps / pid；我们的名字不同 ⇒ 表格全显示 "-"。
    # 这里做一次映射（**离线导入的 run 没有 PID，故 pid 留空**，平台会显示 "-"）。
    meta["algorithm"] = meta.get("algo") or meta.get("kind") or "-"
    meta["algo"] = meta.get("algo") or "-"
    # 总步数：日志里的「总轮次=300」那行是**宽字符**，按不同编码读会变乱码（本项目实测），
    # 所以优先用 ASCII 行统计出来的回合数（EP 行里最大 ep），其次才是正则抓到的数字。
    if not meta.get("total_timesteps"):
        eps = [int(r["step"]) for r in p["metrics"] if r["name"] == "ep_reward"]
        meta["total_timesteps"] = max(eps) if eps else meta.get("total_episodes") or "-"
    _dump(os.path.join(out_dir, "run.json"), meta, obj=True)

    names = sorted({r["name"] for r in p["metrics"]})
    print(f"✅ {run_id}：{len(p['metrics'])} 条指标 / {len(p['events'])} 个事件 → {out_dir}")
    print(f"   指标名（{len(names)}）：{', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
