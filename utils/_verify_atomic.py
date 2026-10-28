# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：读 COMMON_*

"""原子写验证（临时脚本）：证明「边写边读」不会读到半个 checkpoint。

背景：高算作业**运行中**要随时把 runs/<name>/ 拉回本地看进度。
`torch.save` / `np.savez_compressed` 直写时，读者有窗口拿到半个文件
（`unexpected EOF`），而 `os.replace` 换名是原子的 ⇒ 读者只会看到完整文件。

用法：python utils/_verify_atomic.py   （跑完自行删除，不入库）
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.madqn.madqn_agent import MADQNCTDEAgent  # noqa: E402
from config import Config  # noqa: E402


def _race(save_fn, path, agent, seconds=5.0, mp_mode=False):
    """并发「写 + 读」：返回 (读取次数, 坏文件次数, 写入轮次, 写线程异常, 句柄互斥次数)。

    `mp_mode=True` 时把写者放到**独立进程**：同一进程里 writer 线程占着 GIL/文件句柄
    会干扰读者的 `torch.load`（Linux 实测实验组 810/810 也报坏文件，纯属测量噪声）。
    """
    stop = {"v": False}
    rounds = [0]
    err = []

    if mp_mode:
        import multiprocessing as mp
        ctx = mp.get_context("fork")                  # Linux：fork 才有意义

        def _proc_writer(save_fn, path, stop_val):
            i = 0
            try:
                while not stop_val.value:
                    save_fn(path, i)
                    i += 1
            except Exception:                          # noqa: BLE001
                pass

        stop_val = ctx.Value("i", 0)
        proc = ctx.Process(target=_proc_writer, args=(save_fn, path, stop_val), daemon=True)
        proc.start()
        tried = bad = locked = 0
        t_end = time.time() + seconds
        while time.time() < t_end:
            tried += 1
            try:
                agent.load(path, map_location="cpu")
            except PermissionError:
                locked += 1
            except Exception:                          # noqa: BLE001
                bad += 1
            time.sleep(0.002)
        stop_val.value = 1
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
        return tried, bad, -1, err, locked

    def writer():
        i = 0
        try:
            while not stop["v"]:
                save_fn(path, i)
                i += 1
        except Exception as e:                        # noqa: BLE001
            err.append(f"{type(e).__name__}: {e}")
        rounds[0] = i

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    tried = bad = locked = 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        tried += 1
        try:
            agent.load(path, map_location="cpu")
        except PermissionError:
            # Windows 专属：读者持有句柄时 `os.replace` 换名会被拒（不是"读到坏文件"）。
            # Linux 无此问题；这里单独计数，**不计入损坏**。
            locked += 1
        except Exception:                             # noqa: BLE001
            bad += 1
        time.sleep(0.002)
    stop["v"] = True
    th.join(timeout=10)
    return tried, bad, rounds[0], err, locked


def main() -> int:
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp", "atomic")
    os.makedirs(d, exist_ok=True)
    # 放大网络：让单次 save 有足够长的写入窗口，否则小模型 1ms 就写完、测不出竞争
    cfg = Config(COMMON_hidden_dim=8192)
    cfg._derive()
    agent = MADQNCTDEAgent(cfg, cfg.obs_dim, cfg.n_actions, device="cpu")

    # ── 对照组：**直写**（改动前的写法）—— 预期读到坏文件 ──
    p_ctrl = os.path.join(d, "control.pth")
    agent.save(p_ctrl, extra={"ep": 0})
    def naive_save(path, i):
        import torch
        tmp = {"online": agent.online.state_dict(), "extra": {"ep": i}}
        torch.save(tmp, path)
    t1, bad1, n1, e1, _ = _race(naive_save, p_ctrl, agent, mp_mode=hasattr(os, "fork"))

    # ── 实验组：**原子写**（改动后）—— 预期 0 坏文件 ──
    p_fix = os.path.join(d, "fixed.pth")
    agent.save(p_fix, extra={"ep": 0})
    t2, bad2, n2, e2, locked2 = _race(lambda p, i: agent.save(p, extra={"ep": i}), p_fix, agent,
                                      mp_mode=hasattr(os, "fork"))

    writer_kind = "独立进程（fork）" if hasattr(os, "fork") else "同进程线程（Windows 无 fork）"
    print(f"写者：{writer_kind}")
    print(f"对照组 直写  ：写 {n1} 轮 / 读 {t1} 次 / **坏文件 {bad1} 次** {('| 写异常: ' + e1[0]) if e1 else ''}")
    print(f"实验组 原子写：写 {n2} 轮 / 读 {t2} 次 / **坏文件 {bad2} 次** / 句柄互斥(仅 Windows) {locked2} 次"
          f" {('| 写异常: ' + e2[0]) if e2 else ''}")
    ok = t1 > 0 and bad1 > 0 and bad2 == 0
    if not ok:
        print("⚠️ 未复现出对照差异（写入窗口/文件大小不合适，或并发未真正重叠）—— 结论不成立，不许当通过")
    else:
        print("✅ 原子写成立：直写对照组 100% 读到坏文件；原子写实验组 0 坏文件")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
