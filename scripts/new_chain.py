# -*- coding: utf-8 -*-
# 🕒 v2 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：复制器提示文案同步 COMMON_* ｜ 上一版：v1 · 2026-10-17 16:43

"""链路复制器：`algos/<src>` → `algos/<dst>`，只做**机械重命名**并记账。

为什么要有它：AGENTS.md 铁律 3 说「加新算法 = 复制一个包 + 改前缀 + 登记一行」，
但手工 mv + 全局替换正是最容易出错的一步（漏一处 → import 成功、跑错类）。
本脚本把这一步变成可复现、可记账的动作：改了几个文件、每个文件改了几行，跑完直接打印。

用法（唯一一次演练用得上的参数已写死在 `--profile pursuit`）：

    python scripts/new_chain.py --src algos/madqn --dst algos/pursuit --profile pursuit
    python scripts/new_chain.py --src algos/madqn --dst algos/xxx --rename a=b ...

记账口径：
  · 「机械改名行」= 只由替换规则造成的差异（本脚本无法逐行判定语义，只给出 diff 行数）；
  · 「实质改写」= 复制完**人工**改的那些行（不在本脚本的统计里，要人自己补进演练报告）。

🔴 本脚本不写 registry、不写 config：登记与参数必须人工确认（显式工厂的意义就在这）。
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 项目 + 任务两级的替换规则；顺序敏感：长的先换，否则 "MADQNQNetwork" 会被 "MADQN" 先吃掉
PROFILES = {
    "pursuit": [
        ("MADroneSearchEnv", "PursuitSearchEnv"),
        ("DroneSearchReplayBuffer", "PursuitSearchReplayBuffer"),
        ("MADQNCTDEAgent", "PursuitCTDEAgent"),
        ("MADQNQNetwork", "PursuitQNetwork"),
        ("train_madqn_ctde", "train_pursuit_ctde"),
        ("evaluate_madqn", "evaluate_pursuit"),
    ],
}
# 任何 profile 都追加的通用规则：模块名/文件名 → 只换小写前缀；链标签 → 只换不被 _ 保护的 MADQN
COMMON = [("madqn_", "{dst}_"), ("madqn", "{dst}")]


def _apply(text: str, pairs, dst_name: str) -> str:
    for old, new in pairs:
        text = text.replace(old, new)
    # 链标签（"链路 MADQN"、"MADQN 链路"）→ 新链路名；`COMMON_patch` 这类参数名**故意不动**
    return re.sub(r"MADQN(?!_)", dst_name.upper(), text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="复制一个链路包并机械重命名")
    ap.add_argument("--src", required=True, help="源包，如 algos/madqn")
    ap.add_argument("--dst", required=True, help="目标包，如 algos/pursuit")
    ap.add_argument("--profile", choices=sorted(PROFILES), help="预置的类名映射")
    ap.add_argument("--rename", nargs="*", default=[], help="附加 old=new 替换")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    dst_name = os.path.basename(args.dst).strip("/\\")
    pairs = list(PROFILES.get(args.profile, [])) + [
        (o, n) for o, n in (p.split("=", 1) for p in args.rename)]
    pairs = [(o, n.format(dst=dst_name)) for o, n in pairs]
    common = [(o, n.format(dst=dst_name)) for o, n in COMMON]

    src_abs = os.path.join(ROOT, args.src)
    dst_abs = os.path.join(ROOT, args.dst)
    if os.path.exists(dst_abs) and not args.dry_run:
        print(f"❌ 目标已存在：{args.dst}（拒绝覆盖 —— 复制器不是同步器）")
        return 2

    rows = []
    changed_total = 0
    for dirpath, _, files in os.walk(src_abs):
        if "__pycache__" in dirpath:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            src_file = os.path.join(dirpath, fn)
            rel = os.path.relpath(src_file, src_abs)
            new_rel = _apply(fn, pairs + common, dst_name)
            with open(src_file, encoding="utf-8") as f:
                before = f.read()
            after = _apply(before, pairs + common, dst_name)
            diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0))
            changed = sum(1 for d in diff if d.startswith(("+", "-")) and not d.startswith(("+++", "---")))
            changed_total += changed
            n_lines = before.count("\n") + (0 if before.endswith("\n") else 1)
            rows.append((rel, new_rel, n_lines, changed))
            if not args.dry_run:
                out = os.path.join(dst_abs, new_rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8", newline="\n") as f:
                    f.write(after)

    print(f"{'源文件':<26}{'目标文件':<28}{'行数':>6}{'机械改行':>9}")
    for rel, new_rel, n, c in rows:
        print(f"{rel:<26}{new_rel:<28}{n:>6}{c:>9}")
    print(f"\n合计：{len(rows)} 个文件 · {sum(r[2] for r in rows)} 行 · 机械改名 {changed_total} 行")
    print("下一步（人工，必须你自己确认）：")
    print("  1) 重写 <dst>_envs.py 的领域逻辑（其余五个文件应保持零语义改动）")
    print(f"  2) rlinfra/registry.py 加一行 register(AlgoSpec(chain=\"{dst_name}\", package=\"algos.{dst_name}\", ...))")
    print("  3) config.py 只加真正新增的参数（🔴 别改 MADQN_* 的物理含义）")
    return 0 if not args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
