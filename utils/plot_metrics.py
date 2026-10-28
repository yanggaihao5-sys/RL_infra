# -*- coding: utf-8 -*-
"""从训练日志画曲线图（不重跑训练，只读 runs/<name>/train.log）。

用法：
    set PYTHONIOENCODING=utf-8 && python utils/plot_metrics.py runs/demo_60/train.log
    python utils/plot_metrics.py runs/demo_60/train.log --lang cn

产出：<实验目录>/<algo>_training_metrics_{en|cn}.png（与日志同目录，随实验一起归档）
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "utils"))

import numpy as np  # noqa: E402
import analyze_training_log as atl  # noqa: E402

# 中文标签需要系统有中文字体，缺字体会变成方框 —— 默认用英文
LABELS = {
    "en": dict(reward="Episode reward", coverage="Coverage (%)", loss="Loss",
               curve="Training curves", ep="Episode"),
    "cn": dict(reward="每回合奖励", coverage="覆盖率 (%)", loss="损失",
               curve="训练曲线", ep="回合"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="训练日志路径（runs/<name>/train.log）")
    ap.add_argument("--lang", choices=["en", "cn"], default="en")
    ap.add_argument("--algo", default="madqn")
    args = ap.parse_args()

    res = atl.analyze(args.log)
    if "error" in res:
        print(f"❌ {res['error']}")
        return 1
    ep = res["ep"]
    if not ep:
        print("❌ 日志里没有 EP 行，无法画图")
        return 1

    eps = np.array([x[0] for x in ep])
    reward = np.array([x[1] for x in ep])
    cov = np.array([x[2] for x in ep])
    loss = np.array([x[4] for x in ep])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = LABELS[args.lang]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(eps, reward, lw=1.2)
    axes[0].set_xlabel(L["ep"]); axes[0].set_ylabel(L["reward"]); axes[0].grid(alpha=.3)
    axes[1].plot(eps, cov, lw=1.2, color="tab:green")
    axes[1].set_xlabel(L["ep"]); axes[1].set_ylabel(L["coverage"]); axes[1].grid(alpha=.3)
    axes[2].plot(eps, loss, lw=1.2, color="tab:red")
    axes[2].set_xlabel(L["ep"]); axes[2].set_ylabel(L["loss"]); axes[2].grid(alpha=.3)

    name = os.path.basename(os.path.dirname(os.path.abspath(args.log))) or "run"
    fig.suptitle(f"{L['curve']} | {name} | {len(ep)} episodes | "
                 f"cov peak {cov.max():.1f}%")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(args.log)),
                       f"{args.algo}_training_metrics_{args.lang}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"✅ 已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
