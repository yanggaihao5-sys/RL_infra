# 🕒 v4 · 2026-10-25 20:14 · dsh-agent ｜ 变更：新增 --rollout-device；输出版本自描述设备组合与是否分离 ｜ 上一版：v3 · 2026-10-25 17:00

"""B2 · overlap 加速比 + 三段瓶颈归因（手册 §3 B2 三条必测）

    python traintest/bench_overlap.py --device cpu --seeds 3      # 本机先验结构
    python traintest/bench_overlap.py --device cuda --seeds 3     # 高算 GPU 出数

## 三条必测怎么落

1. **overlap 加速比** = `wall(sync) / wall(overlap)`，同一 episode 数、同 seed 集 ⇒ 每 seed 一个比值，
   报 **mean ± SD** 与 **MDE(2·SD/√N)**；`|Δ| < MDE` 判**不可辨识**，不许写成"提升了"。
2. **GPU 利用率**：`nvidia-smi --query-gpu=utilization.gpu` 采样；**MIG 上常恒为 0** ⇒ 拿不到就写 `—`。
3. **瓶颈归因**：每档把一轮拆成 `rollout / 权重同步 / 训练` 三段给占比；
   **rollout > 60% ⇒ 优化推理引擎才对 RL 吞吐有杠杆**（这正是要讲的故事）。

退出码：0 = 两档都跑通且加速比已给出（无论正负）；1 = 结构没跑通。**不对加速比大小做通过与否的判定**
—— 那是数据结论，不是代码正确性。

产物：`runs/b_overlap/{overlap.csv, run.log}`
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traintest.pipeline import run_both  # noqa: E402
from traintest.rollout import make_pair  # noqa: E402

FIELDS = ["seed", "device", "steps", "batch", "prompt_len", "max_new_tokens",
          "reps", "warmup_s", "sync_s", "overlap_s", "speedup", "speedup_sd_within",
          "sync_tps", "overlap_tps",
          "sync_share_rollout", "sync_share_sync", "sync_share_train",
          "overlap_share_rollout", "overlap_share_sync", "overlap_share_train",
          "gpu_util_mean", "sync_bytes"]


class GpuSampler:
    """后台按秒采 `nvidia-smi` 的 utilization.gpu；采不到就保持 None（写 `—`，不许猜）"""

    def __init__(self, interval: float = 0.5):
        self.interval, self.samples, self.ok = interval, [], False
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, encoding="utf-8", errors="replace")
                vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
                if vals:
                    self.samples.append(max(vals))
                    self.ok = True
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=3)
        return statistics.fmean(self.samples) if self.samples else None


def one_seed(seed: int, args) -> dict:
    """一个 seed 的完整测量：**先预热**，再每档重复 reps 次取中位数

    🔴 为什么要预热：91060 那次首跑没预热，CUDA 初始化 + cuDNN autotune 的成本
    全落在**先跑的那一档**（sync）头上 —— seed 0 的 sync 2.200 s vs 同档 seed 1/2 的 0.35 s，
    直接把加速比抬成 3.936×、SD 1.81、MDE 2.09 ⇒ 判成"不可辨识"。**那是预热伪影，不是效应。**

    🔴 为什么每档要重复：20 步 × 2 档总共才 0.35 s，单次计时被抖动主导。
    """
    def _pair():
        return make_pair(
            seed=seed, device=args.device, device_rollout=args.rollout_device or args.device,
            vocab=args.vocab, d_model=args.d, n_layers=args.layers,
            n_heads=args.heads, max_len=args.prompt_len + args.max_new_tokens + 2)

    kw = dict(steps=args.steps, batch=args.batch, prompt_len=args.prompt_len,
              max_new_tokens=args.max_new_tokens, device=args.device,
              device_rollout=args.rollout_device or args.device, seed=seed,
              rollout_cpu_ms=getattr(args, "rollout_cpu_ms", 0.0))

    # ── 预热：两档各跑 warmup 步，**不计时、不进统计** ──
    warm_s = 0.0
    if args.warmup > 0:
        t0 = time.perf_counter()
        wl = dict(kw, steps=args.warmup)
        tr, _ro, en = _pair()
        run_both(tr, en, **wl)
        warm_s = time.perf_counter() - t0
    gpu_util_at_warm = None
    if args.device.startswith("cuda"):
        try:
            import torch
            gpu_util_at_warm = round(torch.cuda.memory_allocated() / 1024**2, 1)
        except Exception:
            gpu_util_at_warm = None

    # ── 正式测量：每档 reps 次，取中位数；两档交替跑，避免"谁先跑谁吃亏" ──
    sampler = GpuSampler() if args.device.startswith("cuda") else None
    if sampler:
        sampler.start()
    sync_runs, over_runs = [], []
    for _ in range(args.reps):
        tr, _ro, en = _pair()
        sync_runs.append(run_both(tr, en, **kw)["sync"])
        tr2, _ro2, en2 = _pair()
        over_runs.append(run_both(tr2, en2, **kw)["overlap"])
    gpu_util = sampler.stop() if sampler else None

    s = dict(sync_runs[0])
    o = dict(over_runs[0])
    sync_med = statistics.median(r["seconds"] for r in sync_runs)
    over_med = statistics.median(r["seconds"] for r in over_runs)
    # 三段占比取中位数那一轮的
    s = min(sync_runs, key=lambda r: abs(r["seconds"] - sync_med))
    o = min(over_runs, key=lambda r: abs(r["seconds"] - over_med))
    return {
        "seed": seed, "device": args.device, "steps": args.steps, "batch": args.batch,
        "prompt_len": args.prompt_len, "max_new_tokens": args.max_new_tokens,
        "reps": args.reps, "warmup_s": round(warm_s, 6),
        "sync_s": round(sync_med, 6), "overlap_s": round(over_med, 6),
        "speedup": round(sync_med / over_med, 6) if over_med else float("nan"),
        "speedup_sd_within": round(statistics.stdev(
            [a["seconds"] for a in sync_runs]) if args.reps > 1 else 0.0, 6),
        "sync_tps": round(s["tokens_per_s"], 2), "overlap_tps": round(o["tokens_per_s"], 2),
        "sync_share_rollout": round(s["share_rollout"], 4),
        "sync_share_sync": round(s["share_sync"], 4),
        "sync_share_train": round(s["share_train"], 4),
        "overlap_share_rollout": round(o["share_rollout"], 4),
        "overlap_share_sync": round(o["share_sync"], 4),
        "overlap_share_train": round(o["share_train"], 4),
        "gpu_util_mean": "—" if gpu_util is None else round(gpu_util, 1),
        "sync_bytes": int(s["sync_bytes"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="B2 overlap 加速比 + 瓶颈归因")
    ap.add_argument("--device", default="cpu", help="训练设备")
    ap.add_argument("--rollout-device", default="",
                    help="rollout 设备；与 --device 不同即「推/训分离」（如 --device cuda:1 --rollout-device cuda:0）")
    ap.add_argument("--seeds", type=int, default=3, help="跑几个 seed（每个 seed 都跑两档）")
    ap.add_argument("--seed", type=int, default=0, help="--sweep-cpu 模式下用哪个 seed")
    ap.add_argument("--steps", type=int, default=60, help="每档每次计时的步数（别太小，否则噪声主导）")
    ap.add_argument("--reps", type=int, default=3, help="每档重复几次取中位数")
    ap.add_argument("--warmup", type=int, default=4,
                    help="预热步数：不计时、进不了统计。0 = 关掉（⚠️ 首档会被 CUDA 初始化污染）")
    ap.add_argument("--rollout-cpu-ms", type=float, default=0.0,
                    help="rollout 侧**非 GPU** 段（tokenize/后处理/网络）的毫秒数；overlap 的收益只可能来自这一段")
    ap.add_argument("--sweep-cpu", default="",
                    help="扫 --rollout-cpu-ms，如 '0,2,5,10,20,40'；给了就只跑这个 sweep（单 seed），出\"何时 overlap 才划算\"的边界")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join("runs", "b_overlap"))
    args = ap.parse_args()

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("❌ --device cuda 但 torch.cuda.is_available() = False —— "
              "本机是 CPU 版 torch，GPU 跑请到高算提交")
        return 2
    print(f"# torch {torch.__version__} / cuda_available={torch.cuda.is_available()} / device={args.device}")
    if args.device.startswith("cuda"):
        print(f"# GPU: {torch.cuda.get_device_name(0)}")

    # ── 模式 2：扫 rollout 侧非 GPU 段，找「overlap 何时才划算」的边界 ──
    if args.sweep_cpu:
        vals = [float(x) for x in args.sweep_cpu.split(",") if x.strip()]
        srows = []
        print(f"\n### CPU 段扫描：--rollout-cpu-ms ∈ {vals}（单 seed）")
        for ms in vals:
            args.rollout_cpu_ms = ms
            r = one_seed(args.seed, args)
            srows.append({"cpu_ms": ms, **r})
            print(f"  cpu_ms={ms:6.1f}  加速比={r['speedup']:.3f}×  "
                  f"sync={r['sync_s']:.3f}s  overlap={r['overlap_s']:.3f}s  "
                  f"GPU util={r['gpu_util_mean']}")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "sweep_cpu.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cpu_ms"] + FIELDS)
            w.writeheader()
            w.writerows(srows)
        out = ["", "## CPU 段扫描：overlap 的收益从哪来（这就是「何时才划算」的答案）", "",
               "| rollout 非GPU段(ms) | sync 秒 | overlap 秒 | **加速比** | GPU util | 判定 |",
               "|---|---|---|---|---|---|"]
        cross = None
        for r in srows:
            verdict = "overlap 更慢" if r["speedup"] < 1.0 else "✅ overlap 更快"
            if cross is None and r["speedup"] >= 1.0:
                cross = r["cpu_ms"]
            out.append(f"| {r['cpu_ms']:.0f} | {r['sync_s']:.3f} | {r['overlap_s']:.3f} | "
                       f"**{r['speedup']:.3f}×** | {r['gpu_util_mean']} | {verdict} |")
        out.append("")
        out.append("🔴 **交叉点**：" + (
            f"CPU 段 ≥ **{cross:.0f} ms** 时 overlap 才开始赚；低于它就被线程开销吃掉了"
            if cross is not None else
            "本网格内 CPU 段再大也没让 overlap 翻正 —— 说明瓶颈不在「有没有非 GPU 段可叠」，需查别的机制"))
        text = "\n".join(out)
        print(text)
        with open(os.path.join(args.out, "run.log"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return 0

    rows = [one_seed(s, args) for s in range(args.seeds)]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "overlap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    sp = [r["speedup"] for r in rows]
    mean = statistics.fmean(sp)
    sd = statistics.stdev(sp) if len(sp) > 1 else 0.0
    mde = 2.0 * sd / math.sqrt(len(sp)) if sp else 0.0

    lines = []
    lines.append(f"# 命令：python traintest/bench_overlap.py --device {args.device} --seeds {args.seeds} "
                 f"--steps {args.steps} --reps {args.reps} --warmup {args.warmup} "
                 f"--batch {args.batch} --prompt-len {args.prompt_len} "
                 f"--max-new-tokens {args.max_new_tokens} --d {args.d} --layers {args.layers}")
    lines.append(f"# 模型：TinyPolicy vocab={args.vocab} d_model={args.d} layers={args.layers} "
                 f"heads={args.heads}（有因果自注意力；**只为结构，不为效果**）")
    split = bool(args.rollout_device) and args.rollout_device != args.device
    lines.append(f"# 设备：训练={args.device} · rollout={args.rollout_device or args.device}"
                 f"（**推/训分离 = {'是' if split else '否'}**）")
    lines.append(f"# 预热 {args.warmup} 步（不计时）+ 每档重复 {args.reps} 次取中位数；"
                 f"每个 seed 的预热耗时 {rows[0]['warmup_s']:.2f}s（已排除在计时之外）"
                 if rows else "")
    lines.append("")
    lines.append("| seed | sync 秒(中位) | overlap 秒(中位) | **加速比** | sync tokens/s | overlap tokens/s | GPU util |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['seed']} | {r['sync_s']:.3f} | {r['overlap_s']:.3f} | "
                     f"**{r['speedup']:.3f}×** | {r['sync_tps']:.0f} | {r['overlap_tps']:.0f} | "
                     f"{r['gpu_util_mean']} |")
    lines.append("")
    lines.append(f"## 加速比：**{mean:.3f}× ± {sd:.3f}**（n={len(sp)}）· "
                 f"MDE(2·SD/√N)=**{mde:.3f}**")
    if abs(mean - 1.0) < mde:
        lines.append("- 判定：**不可辨识**（|加速比−1| < MDE）⇒ **不许写成「提升了」**")
    elif mean > 1.0:
        lines.append(f"- 判定：overlap 快 {mean:.3f}×（超出 MDE {mde:.3f}，可辨识）")
    else:
        lines.append(f"- 判定：overlap **更慢** {1/mean if mean else float('nan'):.3f}×"
                     f"（超出 MDE {mde:.3f}，可辨识）—— 如实写，这本身是有价值的负结果")

    lines.append("")
    lines.append("## 瓶颈归因（一轮拆三段，手册 B2 第 3 条）")
    lines.append("")
    lines.append("| 档 | rollout 占比 | 权重同步占比 | 训练占比 | 结论 |")
    lines.append("|---|---|---|---|---|")
    for key, label in (("sync", "同步"), ("overlap", "overlap")):
        ro = statistics.fmean(r[f"{key}_share_rollout"] for r in rows)
        sy = statistics.fmean(r[f"{key}_share_sync"] for r in rows)
        tr = statistics.fmean(r[f"{key}_share_train"] for r in rows)
        verdict = "🔴 rollout >60% ⇒ 推理引擎是杠杆" if ro > 0.6 else "训练/同步占主导"
        lines.append(f"| {label} | {ro:.3f} | {sy:.3f} | {tr:.3f} | {verdict} |")
    bytes_total = rows[0]["sync_bytes"] if rows else 0
    n_sync = max(1, args.steps - 1)        # overlap 档第 0 步无前置，故同步次数 ≈ steps−1
    lines.append("")
    lines.append(f"- 权重同步：**每次 {bytes_total / n_sync / 1024:.1f} KiB**"
                 f"（{n_sync} 次共 {bytes_total / 1024 / 1024:.2f} MiB）；"
                 f"同步档占比 {statistics.fmean(r['sync_share_sync'] for r in rows):.3f} —— "
                 f"权重同步是否够便宜，看这一列而不是凭感觉。")
    if args.device.startswith("cuda"):
        utils = [r["gpu_util_mean"] for r in rows if r["gpu_util_mean"] != "—"]
        lines.append(f"- GPU 利用率均值：{utils[0] if utils else '—'}"
                     f"{'（MIG 切片上常恒为 0，属平台限制 ⇒ 写 —，不许猜）' if not utils else ''}")
    else:
        lines.append("- 本次跑在 **CPU**：GPU 利用率不适用（写 `—`）。")
        lines.append("")
        lines.append("### 🔴 本次 CPU 结果的读法（别当成 overlap 的真实收益）")
        lines.append("- overlap 在 CPU 上**结构性更慢**：自回归 generate 是 `max_new_tokens` 次"
                     "Python 循环 + 小算子，GIL 基本不放 ⇒ 两线程**串行化**还多付上下文切换。")
        lines.append("- 因此 `加速比 < 1` 是 **CPU 的必然结果，不是实现错**；要回答"
                     "「overlap 值不值」必须上 GPU（torch 在 CUDA 上释放 GIL，且生成/训练可用不同资源）。")
        lines.append("- 本机这一跑的作用只有一个：**证明结构跑通 + 三段能分开计时**。")

    text = "\n".join(lines)
    print("")
    print(text)
    with open(os.path.join(args.out, "run.log"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
