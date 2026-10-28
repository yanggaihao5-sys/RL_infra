# -*- coding: utf-8 -*-
"""产物路径：唯一命名源 `name` → 两级封顶的产物目录（spec §3）。

```
runs/<name>/                    ← 一级：本次实验全部产物
├── <algo>_best.pth             ← 覆盖保存
├── <algo>_final_<N>.pth        ← 递增序号，不覆盖
├── <algo>_training_metrics_en.png
├── train.log
└── <downstream>/               ← 二级：下游产物共用同一处（不各自嵌套）
```

铁律：**函数内禁止硬编码输出路径**，一律由 RunPaths 派生。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class RunPaths:
    name: str
    runs_root: str = "runs"

    # ───────────────────────── 构造 ─────────────────────────
    @classmethod
    def from_config(cls, cfg) -> "RunPaths":
        return cls(name=cfg.name, runs_root=cfg.runs_root)

    # ───────────────────────── 一级：实验根 ─────────────────────────
    @property
    def root(self) -> str:
        return os.path.abspath(os.path.join(self.runs_root, self.name))

    def ensure(self) -> "RunPaths":
        os.makedirs(self.root, exist_ok=True)
        return self

    # ───────────────────────── 二级：下游目录（复用同一处，不嵌套）─────────────────────────
    def child(self, sub: str) -> str:
        """下游产物目录。多个模块要用同一个下游目录时，全部调这里传同一个 sub。"""
        p = os.path.join(self.root, sub)
        os.makedirs(p, exist_ok=True)
        return p

    # ───────────────────────── 具名产物 ─────────────────────────
    @property
    def log(self) -> str:
        return os.path.join(self.root, "train.log")

    def best_model(self, algo: str) -> str:
        return os.path.join(self.root, f"{algo}_best.pth")

    def pretrain_model(self, algo: str) -> str:
        return os.path.join(self.root, f"{algo}_pretrain.pth")

    def final_model(self, algo: str, n: int = 0) -> str:
        """n=0 → <algo>_final.pth；n>0 → <algo>_final_<n>.pth（自动避让已存在文件）。"""
        if n <= 0:
            return os.path.join(self.root, f"{algo}_final.pth")
        return os.path.join(self.root, f"{algo}_final_{n}.pth")

    def next_final_model(self, algo: str) -> str:
        """下一个 final 序号 = **已有最大序号 + 1**（序号单调递增，永不复用）。

        🔴 必须是「max+1」而**不是**「第一个空号」：配合 `prune_final_models()` 时，
        空号会落在已被清掉的旧号上 —— 存完 `final_1` 后按序号排序它成了最旧的，
        立刻被清理删掉 → **刚存的 checkpoint 当场丢失**（本项目实测踩到）。
        max+1 保证新存的永远是最大号，永远排在「保留区」里。
        """
        pat = re.compile(rf"^{re.escape(algo)}_final_(\d+)\.pth$")
        nums = [int(m.group(1)) for name in os.listdir(self.root)
                if (m := pat.match(name))] if os.path.isdir(self.root) else []
        return self.final_model(algo, (max(nums) if nums else 0) + 1)

    def figure(self, algo: str, lang: str = "en") -> str:
        return os.path.join(self.root, f"{algo}_training_metrics_{lang}.png")

    def test_plot_dir(self, lang: str = "en") -> str:
        return self.child(f"test_path_plots_{lang}")

    # ───────────────────────── 产物清理 ─────────────────────────
    def prune_final_models(self, algo: str, keep: int) -> List[str]:
        """只保留最近 `keep` 个 `<algo>_final_<N>.pth`，返回被删掉的路径。

        **为什么需要**：`next_final_model()` 只递增、从不覆盖 —— 长跑时每
        `save_interval` 回合存一个，2000 回合就能堆出 40 个文件（实测），
        几万回合会失控。序号仍然单调递增（= 第几次保存），只是旧文件被清掉。

        🔴 安全约束（删文件必须先划边界）：
          · 只删**本实验目录 root 下、文件名严格匹配 `<algo>_final_<数字>.pth`** 的文件；
          · `*_best.pth` / `*_final.pth`（无序号）/ `*_pretrain.pth` / 任何其他文件**一律不碰**；
          · `keep <= 0` → 不清理（保留全部）。
        """
        if keep is None or keep <= 0:
            return []
        pat = re.compile(rf"^{re.escape(algo)}_final_(\d+)\.pth$")
        items = []
        for name in os.listdir(self.root):
            m = pat.match(name)
            if m:
                items.append((int(m.group(1)), os.path.join(self.root, name)))
        items.sort()
        removed = []
        for _, path in (items[:-keep] if len(items) > keep else []):
            os.remove(path)
            removed.append(path)
        return removed

    def __str__(self) -> str:
        return self.root
