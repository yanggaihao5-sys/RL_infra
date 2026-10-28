# 🕒 v1 · 2026-10-25 17:25 · dsh-agent ｜ 变更：新增全项目唯一打包脚本（拒绝式 tar + tar 清单对账 + utf-8 解包）
"""产物打包 / 回传 / 解包 / 对账 —— **全项目只此一个打包脚本**

    python scripts/pack_results.py all            # 打包 → 回传 → 解包 → 对账（一条命令）
    python scripts/pack_results.py pack           # 只远程打包 + 生成清单
    python scripts/pack_results.py fetch          # 只回传（包 + 清单）
    python scripts/pack_results.py unpack         # 只本地解包（utf-8，见下）
    python scripts/pack_results.py verify         # 只对账（远程清单 vs 本地解包结果）

## 三条铁律（来自 `hpc-jobs` 技能 §六，都是踩过的坑）

1. 🔴 **拒绝式打包，绝不用白名单**：`tar --exclude` 打**整个** `RL_infra`，只排除明确不要的。
   白名单（`find -name '*.json'` 之类）**天生滞后** —— 产物类型会长出来，漏一个类型就静默丢一批文件，
   而 `tar` 照样 exit 0 报"打包成功"。
2. 🔴 **一个项目只许有一个打包脚本**，清单只在本文件的 `EXCLUDES` 一处维护。
   按实验各写一份 `_packN.sh` 正是"修了 e19 忘了 e23"的成因。
3. 🔴 **回传后必须对账**：`tar` 报 OK 只证明包能读，不证明**文件齐**。
   本脚本用 **`tar -tzf` 自己的成员清单**当基准（不是另跑 `find`，避免两套规则漂移），
   与本地解包后的实际文件列表逐条比对，**差集非空即判失败**。

## 为什么本地解包不用 `tar -xzf`

Windows 的 bsdtar 按 **GBK** 解码 UTF-8 文件名 ⇒ 中文目录变成**建不出来的乱码名、其下文件静默全丢**，
且**退出码仍是 0**。所以本脚本用 Python `tarfile(encoding="utf-8")` 逐个 `extract`。

## 排除清单（`EXCLUDES`，唯一维护处）

| 排除 | 为什么 |
|---|---|
| `model_cache` | 18 G HF 权重；项目规矩：只放高算、不回传本地 |
| `wheels` | 预下载的轮子，可再生 |
| `longago` | 历史归档，不参与当前工作 |
| `.git` `.dsh` `.idea` `.tmp` `__pycache__` `*.pyc` `*.pyo` | 版本库 / 编辑器 / 缓存 |
| `*.pth` `*.pt` | 模型权重，按规矩先瘦身再搬（A/B 阶段本仓无自有权重） |

> ⚠️ **故意不排除** `runs/`、`output.*`、`error.*`、`docs/`、`kernel/`、`traintest/`、`hpc/`、`scripts/`
> —— 那些就是交付物本身。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile

# ── 连接信息（默认值取自 hpc-jobs 技能实测；都可用参数覆盖）──
DEFAULT_HOST = "__HPC_ACCOUNT__@__HPC_LOGIN_IP__"
DEFAULT_KEY = os.path.expanduser("~/.ssh/id_ed25519")
DEFAULT_REMOTE_ROOT = "/storage/public/home/__HPC_ACCOUNT__/RL_infra"
# 🔴 包**必须放在被包的树之外**：放在 RL_infra/ 里面会让 tar 报
#    `file changed as we read it`（边写边读自己）并以 exit 1 失败。
DEFAULT_PKG_DIR = "/storage/public/home/__HPC_ACCOUNT__/_deliver"
DEFAULT_PKG_NAME = "rlinfra_deliver.tgz"
DEFAULT_DEST = os.path.join("_deliver", "rlinfra")

# ── 排除清单：唯一维护处 ──
EXCLUDES = [
    "model_cache", "wheels", "longago",
    ".git", ".dsh", ".idea", ".tmp",
    "__pycache__", "*.pyc", "*.pyo",
    "*.pth", "*.pt",
    "_pack", "_deliver",          # 万一有人把包/暂存放进树里
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ssh(args, cmd_list, check=True, capture=True):
    base = ["ssh", "-i", args.key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", args.host]
    full = base + ["bash -lc " + _dq(" ".join(cmd_list))]
    return subprocess.run(full, check=check, text=True, capture_output=capture,
                          encoding="utf-8", errors="replace")


def _scp(args, remote, local, check=True):
    """远程 → 本地"""
    return subprocess.run(["scp", "-i", args.key, "-o", "BatchMode=yes",
                           f"{args.host}:{remote}", local],
                          check=check, text=True, capture_output=True,
                          encoding="utf-8", errors="replace")


def _scp_up(args, local, remote, check=True):
    """本地 → 远程"""
    return subprocess.run(["scp", "-i", args.key, "-o", "BatchMode=yes",
                           local, f"{args.host}:{remote}"],
                          check=check, text=True, capture_output=True,
                          encoding="utf-8", errors="replace")


def _dq(s: str) -> str:
    """把命令包成单引号串（内部单引号转义），供远程 `bash -lc` 吃"""
    return "'" + s.replace("'", "'\\''") + "'"


def _remote_script(args) -> str:
    """远程打包脚本内容。写成文件上传再 `bash <path>` —— 避免多层引号被剥（技能铁律 5）"""
    ex = " ".join(f"--exclude={_dq(e)}" for e in EXCLUDES)
    r = args.remote_root
    pkg = args.remote_pkg
    lst = pkg + ".list"
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {_dq(r)}
mkdir -p {_dq(args.pkg_dir)}
rm -f {_dq(pkg)} {_dq(lst)}
# 拒绝式：打整个目录，只排除 EXCLUDES
tar {ex} -czf {_dq(pkg)} .
# 用 tar 自己的成员清单当对账基准（不另跑 find，避免两套规则漂移）
tar -tzf {_dq(pkg)} | grep -v '/$' | sed 's|^\\./||' | sort > {_dq(lst)}
echo "PKG_BYTES=$(stat -c %s {_dq(pkg)})"
echo "LIST_FILES=$(wc -l < {_dq(lst)})"
"""


def cmd_pack(args) -> int:
    script = _remote_script(args)
    local_sh = os.path.join(args.dest, "_remote_pack.sh")
    os.makedirs(args.dest, exist_ok=True)
    with open(local_sh, "w", encoding="utf-8", newline="\n") as f:   # 必须 LF
        f.write(script)
    remote_sh = f"{args.remote_root}/_pack_remote.sh"
    _scp_up(args, local_sh, remote_sh)
    r = _ssh(args, ["bash", remote_sh], check=False)
    print((r.stdout or "").strip())
    if r.returncode != 0:
        print("❌ 远程打包失败（exit=%d）：\n%s" % (r.returncode, (r.stderr or "")[-1500:]))
        return 1
    if args.clean_remote_script:
        _ssh(args, ["rm", "-f", remote_sh], check=False)
    return 0


def cmd_fetch(args) -> int:
    pkg = args.remote_pkg
    local_pkg = os.path.join(args.dest, args.pkg_name)
    os.makedirs(args.dest, exist_ok=True)
    _scp(args, pkg, local_pkg)
    _scp(args, pkg + ".list", local_pkg + ".list")
    size = os.path.getsize(local_pkg)
    print(f"本地包：{local_pkg}（{size/1024/1024:.2f} MiB）")
    return 0


def cmd_unpack(args) -> int:
    local_pkg = os.path.join(args.dest, args.pkg_name)
    if not os.path.isfile(local_pkg):
        print(f"❌ 本地没有包：{local_pkg}（先跑 fetch）")
        return 1
    if os.path.isdir(args.local_dest) and os.listdir(args.local_dest) and not args.force:
        print(f"❌ 目标目录非空：{args.local_dest}（要覆盖加 --force）")
        return 1
    os.makedirs(args.local_dest, exist_ok=True)
    n = 0
    # 🔴 encoding='utf-8' 是关键：Windows bsdtar 用 GBK 会建出乱码名并静默丢文件
    with tarfile.open(local_pkg, "r:gz", encoding="utf-8") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            try:
                tf.extract(m, args.local_dest, filter="data")
            except TypeError:                      # Python < 3.12 没有 filter 参数
                tf.extract(m, args.local_dest)
            n += 1
    print(f"解包 {n} 个文件 → {args.local_dest}")
    return 0


def _walk_files(base: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(base):
        for f in files:
            p = os.path.relpath(os.path.join(dirpath, f), base)
            out.append(p.replace(os.sep, "/"))
    return sorted(out)


def cmd_verify(args) -> int:
    local_pkg = os.path.join(args.dest, args.pkg_name)
    lst_path = local_pkg + ".list"
    if not os.path.isfile(lst_path):
        print(f"❌ 缺少远程清单：{lst_path}")
        return 1
    with open(lst_path, encoding="utf-8") as f:
        remote = sorted({ln.strip() for ln in f if ln.strip()})
    local = _walk_files(args.local_dest)

    remote_set, local_set = set(remote), set(local)
    missing = sorted(remote_set - local_set)      # tar 里有、本地没有 ⇒ 漏了
    extra = sorted(local_set - remote_set)        # 本地多出来 ⇒ 解包越界/污染

    print(f"远程清单 {len(remote)} 个文件 · 本地解包 {len(local)} 个文件")
    if missing:
        print(f"🔴 漏 {len(missing)} 个（tar 里有、本地没有）：")
        for m in missing[:20]:
            print("   -", m)
        if len(missing) > 20:
            print(f"   …… 还有 {len(missing)-20} 个")
    if extra:
        print(f"⚠️ 多 {len(extra)} 个（本地有、tar 清单里没有）：")
        for e in extra[:20]:
            print("   +", e)
    if not missing and not extra:
        print("✅ 对账通过：差集为空（远程包内容 ⇔ 本地解包结果 逐条一致）")
        return 0
    print("❌ 对账失败 —— 按技能铁律：补齐重传，不许当成打包成功")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="产物打包/回传/解包/对账（全项目唯一打包脚本）")
    ap.add_argument("action", choices=["all", "pack", "fetch", "unpack", "verify"])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    ap.add_argument("--pkg-dir", default=DEFAULT_PKG_DIR,
                    help="远程暂存目录（必须在被包的树之外，否则 tar 报 file changed）")
    ap.add_argument("--pkg-name", default=DEFAULT_PKG_NAME)
    ap.add_argument("--dest", default=os.path.join(ROOT, "_deliver"),
                    help="本地暂存目录（放包与清单）")
    ap.add_argument("--local-dest", default=os.path.join(ROOT, "_deliver", "rlinfra"))
    ap.add_argument("--force", action="store_true", help="允许解包到非空目录")
    ap.add_argument("--keep-remote-script", action="store_true")
    args = ap.parse_args()
    args.clean_remote_script = not args.keep_remote_script
    args.remote_pkg = f"{args.pkg_dir}/{args.pkg_name}"

    steps = {"pack": cmd_pack, "fetch": cmd_fetch, "unpack": cmd_unpack, "verify": cmd_verify}
    seq = ["pack", "fetch", "unpack", "verify"] if args.action == "all" else [args.action]
    for s in seq:
        print(f"\n════════ {s} ════════")
        if steps[s](args) != 0:
            return 1
    if args.action == "all":
        print("\n✅ 打包 → 回传 → 解包 → 对账 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
