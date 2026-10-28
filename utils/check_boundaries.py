# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-24 13:22 · dsh-agent ｜ 变更：新增 ⑧ kernel 库零第三方、⑨ A1 自检全绿；BOM 与 utf-8 子进程修复


"""库边界自检：`rlinfra/` 是**库**，`algos/` 是**应用**，`kernel/` 是**纯 CPU 推理内核**，依赖单向。

九项检查（每项打印 PASS / FAIL，任一项失败 → 退出码 1）：

  ① `import rlinfra` 成功，且导出的公共符号没被改坏；
  ② `rlinfra/` 里**没有任何第三方 import**（只用标准库）—— 装基座不该需要 numpy/torch；
  ③ `rlinfra/` 里**不出现 `algos`**（基座不知道有哪个算法存在）；
  ④ `algos/` 对 `rlinfra/` 的依赖是**向下**的（允许 import rlinfra，反之禁止）；
  ⑤ `algos/madqn/madqn_envs.py` **不 import config**（环境吃鸭子类型的 cfg，不绑项目配置）；
  ⑥ 注册表是**懒加载**：未登记的链路当场 KeyError，登记的链路能取到；
  ⑦ 装了包以后**换目录 `import rlinfra` 仍可用**（`--installed` 时才检查；验证 `pip install -e .`）；
  ⑧ `kernel/` 的**库文件**（非 `bench_*.py`）零第三方依赖 —— 内核必须能裸跑；
  ⑨ `kernel/selfcheck_kv.py` **全部通过**（A1 的 11 项，含行前缀与 logger 互证）。

用法：
    python utils/check_boundaries.py             # ①–⑥ + ⑧⑨（仓库内自检，无需安装）
    python utils/check_boundaries.py --installed # 加上 ⑦（先 pip install -e .）
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PKG = "rlinfra"
ALGO_PKG = "algos"
KERNEL_PKG = "kernel"
# 基座允许的 import：全部标准库；出现这些就不是「纯基座」了
ALLOWED_THIRD_PARTY = {"torch", "numpy", "matplotlib", "pandas", "scipy", "tensorboard"}
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}

results = []

def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))

def _py_files(pkg: str):
    base = os.path.join(ROOT, pkg)
    for dirpath, _, files in os.walk(base):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)

def _imports(path: str):
    """返回 (绝对顶层模块名列表, 相对 import 列表)。

    用 `utf-8-sig` 读：有的文件带 BOM（U+FEFF），Python 解释器能容忍，但 `ast.parse` 会报
    `invalid non-printable character` —— 这里替它吃掉。
    """
    with open(path, encoding="utf-8-sig") as f:
        tree = ast.parse(f.read(), filename=path)
    tops, relatives = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # 相对 import（from .x import y）
                relatives.append((node.module or "").split(".")[0])
            elif node.module:
                tops.append(node.module.split(".")[0])
    return tops, relatives

# ══════════════════════════ ① 公共符号 ══════════════════════════
def check_public_api() -> None:
    sys.path.insert(0, ROOT)
    import rlinfra
    expected = {"TrainLogger", "ResourceMeter", "RunPaths",
                "AlgoSpec", "get_algo", "register", "registered"}
    missing = expected - set(rlinfra.__all__)
    check("① import rlinfra + 公共符号", not missing, f"缺 {sorted(missing)}" if missing else
          f"__all__ = {len(rlinfra.__all__)} 个")

# ══════════════════════════ ② 基座零第三方依赖 ══════════════════════════
def check_no_third_party() -> None:
    bad = []
    for path in _py_files(LIB_PKG):
        tops, _ = _imports(path)
        for t in tops:
            if t.startswith(".") or t in STDLIB or t == LIB_PKG:
                continue
            bad.append(f"{os.path.relpath(path, ROOT)} → {t}")
    check("② rlinfra/ 只用标准库", not bad,
          "; ".join(bad) if bad else "无第三方 import（基座可裸装）")

# ══════════════════════════ ③ 基座不依赖算法 ══════════════════════════
def check_lib_ignores_algos() -> None:
    hits = []
    for path in _py_files(LIB_PKG):
        tops, rel = _imports(path)
        if ALGO_PKG in tops:
            hits.append(os.path.relpath(path, ROOT))
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                code = line.split("#", 1)[0]
                if f"import {ALGO_PKG}" in code or f"from {ALGO_PKG}" in code:
                    hits.append(f"{os.path.relpath(path, ROOT)}:{i}")
    check("③ rlinfra/ 不 import algos", not hits, ", ".join(sorted(set(hits))[:5]))

# ══════════════════════════ ④ 依赖方向向下 ══════════════════════════
def check_direction() -> None:
    allowed, bad = set(), []
    for path in _py_files(ALGO_PKG):
        tops, _ = _imports(path)
        if LIB_PKG in tops:
            allowed.add(os.path.relpath(path, ROOT))
    # algos 里唯一允许「向上」碰的是 rlinfra（库），不是 main.py / config.py / utils
    for path in _py_files(ALGO_PKG):
        tops, _ = _imports(path)
        for t in tops:
            if t in ("main", "config", "utils", "scripts"):
                bad.append(f"{os.path.relpath(path, ROOT)} → {t}")
    check("④ algos → rlinfra 单向", not bad,
          "; ".join(bad) if bad else f"{len(allowed)} 个文件 import rlinfra（合法方向）")

# ══════════════════════════ ⑤ 环境不绑项目配置 ══════════════════════════
def check_env_duck_typed() -> None:
    p = os.path.join(ROOT, ALGO_PKG, "madqn", "madqn_envs.py")
    tops, _ = _imports(p)
    check("⑤ madqn_envs 不吃 config.Config", "config" not in tops,
          "环境只按属性名读 cfg，换项目不用带 config.py")

# ══════════════════════════ ⑥ 注册表懒加载 ══════════════════════════
def check_registry() -> None:
    sys.path.insert(0, ROOT)
    from rlinfra.registry import get_algo, registered
    ok = True
    detail = ""
    try:
        get_algo("no_such_chain")
        ok, detail = False, "未知链路竟然没报错"
    except KeyError as e:
        detail = f"未知链路 KeyError：{str(e)[:40]}…"
    spec = get_algo("madqn")
    # 🔴 不触发 `spec.train` / `spec.env`：那会 import torch/numpy，把「无第三方依赖的
    #    边界 job」变成又一个装 torch 的 job。这里只验证登记字段与包可解析；
    #    「登记的名字真能加载出可调用对象」由 tests/test_registry.py 在装了依赖的 job 里验证。
    import importlib.util
    ok = ok and spec.package == "algos.madqn" and bool(spec.train_fn) and bool(spec.eval_fn)
    ok = ok and importlib.util.find_spec(spec.package) is not None
    ok = ok and "algos.madqn" not in sys.modules      # 懒加载：取 spec 不应 import 包
    check("⑥ 注册表显式工厂 + 懒加载", ok, detail)

# ══════════════════════════ ⑦ 装完换目录仍可 import ══════════════════════════
def check_installed_import() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        code = ("import rlinfra, algos; "
                "print('rlinfra', rlinfra.__all__[:3], 'algos', algos.__doc__.splitlines()[0][:12])")
        r = subprocess.run([sys.executable, "-c", code], cwd=tmp,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    check("⑦ 任意 cwd 下 import rlinfra / algos", ok,
          (r.stdout or r.stderr).strip()[:120])

# ══════════════════════════ ⑧ kernel 库文件零第三方 ══════════════════════════
def check_kernel_pure() -> None:
    """`kernel/` 的**库文件**（kvpager / batching / kvlog …）必须只用标准库。

    `bench_*.py` 例外：它们是基准脚本、不是库 —— `bench_batching.py` 按手册要求用 CPU torch。
    """
    bad, bench = [], []
    for path in _py_files(KERNEL_PKG):
        name = os.path.basename(path)
        if name.startswith("bench_"):
            bench.append(name)
            continue
        tops, _ = _imports(path)
        for t in tops:
            if t.startswith(".") or t in STDLIB or t in (LIB_PKG, KERNEL_PKG, "kvpager",
                                                         "batching", "kvlog", "selfcheck_kv"):
                continue
            bad.append(f"{os.path.relpath(path, ROOT)} → {t}")
    check("⑧ kernel/ 库文件只用标准库（bench_* 例外）", not bad,
          "; ".join(bad) if bad else f"库文件零第三方；{len(bench)} 个 bench_*.py 不受此约束")

# ══════════════════════════ ⑨ A1 自检必须全绿 ══════════════════════════
def check_kernel_selfcheck() -> None:
    """直接跑 `kernel/selfcheck_kv.py`：A1 的判据不许只写在文档里。"""
    p = os.path.join(ROOT, KERNEL_PKG, "selfcheck_kv.py")
    if not os.path.exists(p):
        check("⑨ kernel/selfcheck_kv.py 全绿", False, "文件不存在")
        return
    r = subprocess.run([sys.executable, p], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = [ln for ln in (r.stdout or "").strip().splitlines() if ln.strip()]
    failed = [ln for ln in tail if ln.startswith("❌")]
    check("⑨ kernel/selfcheck_kv.py 全绿", r.returncode == 0,
          (f"{len(tail)} 行输出，失败项 {len(failed)}" if r.returncode == 0
           else (r.stdout + r.stderr).strip()[-300:]))

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="库边界自检")
    ap.add_argument("--installed", action="store_true", help="额外检查 pip install -e . 后的跨目录 import")
    args = ap.parse_args(argv)
    check_public_api()
    check_no_third_party()
    check_lib_ignores_algos()
    check_direction()
    check_env_duck_typed()
    check_registry()
    check_kernel_pure()
    check_kernel_selfcheck()
    if args.installed:
        check_installed_import()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} 项通过"
          + (f"；失败：{failed}" if failed else " —— 边界完好"))
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
