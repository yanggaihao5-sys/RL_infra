# 🕒 v1 · 2026-10-24 12:41 · dsh-agent ｜ 变更：饱和模型外推 + 截断回测（线性外推已证不可用）

"""早停 / 曲线外推：从 `train.log` 的 `EP | ep: … cov: …%` 行估计"还能涨多少"

为什么需要：单次评估 4~32 h，若每次都要跑满 5000 回合，同预算对照（M2）根本跑不动。

🔴 2026-10-24 实测教训（本文件第二版的由来）：
   **线性外推在这条曲线上完全失效** —— 用 SAC 前 2000 回合的窗口斜率（+0.042 pp/回合）
   线性外推 3000 回合 ⇒ 预测再涨 **+126 pp**（物理上不可能，覆盖率上限 100%）。
   覆盖率是**单调有界且饱和**的，必须用**饱和模型**拟合，不是直线。

模型：`cov(ep) = A − B·exp(−ep/τ)`；τ 网格搜索，A/B 用最小二乘闭式解，A 截到 ≤ 1.0。
  预测"最终值" = A（渐近覆盖率）；"还能涨多少" = A − 当前窗口均值。

用法：
    # 完整日志
    python tools/rlinfra_ext/early_stop.py <train.log> --target-ep 5000
    # 回测：假装只跑到第 N 回合，用前半段预测（再和真实末值比）
    python tools/rlinfra_ext/early_stop.py <train.log> --upto-ep 2000 --target-ep 5000
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

PAT = re.compile(r"^EP \| ep:\s*(\d+).*?cov:\s*([0-9.]+)%")


def parse(path):
    eps, covs = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PAT.match(line)
            if m:
                eps.append(int(m.group(1)))
                covs.append(float(m.group(2)) / 100.0)
    return np.array(eps, dtype=float), np.array(covs, dtype=float)


def fit_saturating(ep, cov):
    """cov = A - B·exp(-ep/τ)；τ 网格 + (A,B) 线性最小二乘"""
    best = None
    x = ep - ep.min()
    span = max(float(ep.max() - ep.min()), 1.0)
    for tau in np.linspace(span / 20.0, span * 2.0, 120):
        k = np.exp(-x / tau)
        M = np.column_stack([np.ones_like(k), -k])
        try:
            (A, B), *_ = np.linalg.lstsq(M, cov, rcond=None)
        except np.linalg.LinAlgError:
            continue
        sse = float(((cov - M @ np.array([A, B])) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(A), float(B), float(tau))
    _, A, B, tau = best
    return min(A, 1.0), B, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window", type=int, default=50, help="窗口（回合）——报当前水平与 SD")
    ap.add_argument("--noise-pp", type=float, default=3.0,
                    help="噪声基线（pp）——同配置重复的 SD；不知道就保守给 3")
    ap.add_argument("--target-ep", type=int, default=0, help="原本打算跑到第几回合（0=日志末回合）")
    ap.add_argument("--upto-ep", type=int, default=0, help="回测用：假装只跑到第 N 回合")
    args = ap.parse_args()

    ep, cov = parse(args.log)
    if args.upto_ep:
        keep = ep <= args.upto_ep
        ep, cov = ep[keep], cov[keep]
    if len(cov) < 20:
        print(f"❌ 解析到的 EP 行太少（{len(cov)}）：{args.log}")
        return 2

    w = min(args.window, len(cov))
    tail = cov[-w:]
    cur = float(tail.mean())
    sd_pp = float(np.std(tail) * 100.0)
    A, B, tau = fit_saturating(ep, cov)
    gain_pp = (A - cur) * 100.0
    last_ep = int(ep[-1])
    target = args.target_ep or last_ep

    print(f"日志：{args.log}")
    print(f"  回合数 {len(cov)}（末回合 {last_ep}）｜窗口 {w} 回合均值 {cur*100:.2f}%｜窗口 SD {sd_pp:.2f} pp")
    print(f"  饱和拟合：渐近覆盖率 A = {A*100:.2f}% ｜τ = {tau:.0f} 回合")
    print(f"  距渐近还剩 {gain_pp:+.2f} pp ｜目标回合 {target}")
    verdict = abs(gain_pp) < args.noise_pp
    print(f"  噪声基线 {args.noise_pp:.1f} pp ⇒ "
          f"{'✅ 建议早停（剩余增益 < 噪声）' if verdict else '⏳ 不建议早停（剩余增益仍 > 噪声）'}")
    return 0


def backtest(args):
    """截断回测：预测的渐近覆盖率 vs 真实末段均值 —— 这是"能不能早停"的唯一判据"""
    ep, cov = parse(args.log)
    if len(cov) < 200:
        print(f"❌ 太短（{len(cov)} 行），无法回测")
        return 2
    w = min(args.window, len(cov))
    actual = float(cov[-w:].mean()) * 100.0
    print(f"日志：{args.log}")
    print(f"真实末 {w} 回合均值 = {actual:.2f}%（这就是要预测的目标）")
    print("  只看到  渐近预测A   误差(pp)  判定")
    errs = []
    for upto in (500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000):
        if upto >= len(cov):
            continue
        keep = ep <= upto
        A, _, _ = fit_saturating(ep[keep], cov[keep])
        err = A * 100.0 - actual
        errs.append(abs(err))
        flag = "OK(<2pp)" if abs(err) < 2 else ("WARN(2~5pp)" if abs(err) < 5 else "FAIL(>5pp)")
        print(f"{upto:>8}{A*100:>11.2f}%{err:>+10.2f}  {flag}")
    mae = sum(errs) / len(errs)
    print(f"平均绝对误差 = {mae:.2f} pp ⇒ "
          f"{'可用（<3pp）' if mae < 3 else '不可用（预测偏差过大，禁止用它早停）'}")
    return 0


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("log")
        ap.add_argument("--backtest", action="store_true")
        ap.add_argument("--window", type=int, default=50)
        sys.exit(backtest(ap.parse_args()))
    sys.exit(main())
