# -*- coding: utf-8 -*-
# 🕒 v1 · 2026-09-30 20:59 · dsh-agent ｜ 变更：指标名与语义对齐：elapsed_s/gpu_mem_gb 拆成 *_total 与当步值；修 requires_grad 警告

"""最小 LoRA 冒烟：证明「高算 + A800 MIG + transformers/peft + 本地 DeepSeek 权重」这条链路能跑。

它**不是**训练脚本，是链路体检：
  ① 本地权重能否被 transformers 加载（离线，无 Hub 访问）
  ② 模型能否搬到 CUDA（MIG 切片）
  ③ peft 能否把 LoRA 挂上去、前向+反向+优化器一步是否真的改了参数
  ④ 训练指标能否写进 RL_infra 的观测契约（同时写 train.log 与 metrics.jsonl）

用法（作业内）：
  python scripts/lora_smoke.py --model <本地权重目录> --steps 6 --max-len 256

产物（在 --out，默认 model_cache/_smoke_logs）：
  <run_id>.log           人类可读日志（复用 rlinfra 日志契约的行前缀）
  obs/<run_id>/metrics.jsonl   观测契约（每行 {ts, step, name, value}）
  obs/<run_id>/run.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLES = [
    "问题：1+1 等于几？\n回答：2。\n推理：两个 1 相加得到 2。",
    "问题：把 3 个苹果分给 3 个人，每人几个？\n回答：1 个。\n推理：3 除以 3 等于 1。",
    "问题：如果一个任务需要 5 步，每步 2 秒，共几秒？\n回答：10 秒。\n推理：5 乘以 2 等于 10。",
    "问题：搜索 4x4 网格需要几条边？\n回答：24 条。\n推理：4x4 网格有 4*5*2=40 条边，去掉外边界 16 条。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/public/home/__HPC_ACCOUNT__/RL_infra/model_cache/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="/storage/public/home/__HPC_ACCOUNT__/RL_infra/model_cache/_smoke_logs")
    ap.add_argument("--run-id", default=None, help="默认取模型目录名 → lora_smoke_1p5b / lora_smoke_7b")
    ap.add_argument("--batch", type=int, default=4, help="训练批大小（大模型显存紧张时调小）")
    ap.add_argument("--r", type=int, default=8, help="LoRA 秩（越大容量越大、越吃显存）")
    args = ap.parse_args()
    if args.run_id is None:
        base = os.path.basename(os.path.normpath(args.model))
        tag = "7b" if "7B" in base else ("1p5b" if "1.5B" in base else base)
        args.run_id = f"lora_smoke_{tag}"

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_dir = os.path.join(args.out, "obs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(args.out, f"{args.run_id}.log"), "w", encoding="utf-8")
    m_f = open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8")
    t0 = time.time()

    def say(line: str) -> None:
        print(line, flush=True)
        if not log_f.closed:                     # 收尾时可能已关闭，不要再写
            log_f.write(line + "\n")
            log_f.flush()

    def metric(step: int, name: str, value: float) -> None:
        m_f.write(json.dumps({"ts": time.time(), "step": step, "name": name,
                              "value": float(value)}) + "\n")
        m_f.flush()

    import torch
    say(f"运行设备：{'cuda' if torch.cuda.is_available() else 'cpu'} | torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        say(f"链路标识 | test_mode=lora_smoke | {p.name} | 显存 {p.total_memory/1024**3:.1f}GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    say(f"加载权重：{args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    say(f"权重加载完成（{sum(p.numel() for p in model.parameters())/1e9:.2f}B 参数），用时 {time.time()-t0:.1f}s")

    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say(f"LoRA 挂载完成：可训练参数 {trainable/1e6:.2f}M（占比 {100*trainable/sum(p.numel() for p in model.parameters()):.2f}%）")

    enc = tok(SAMPLES, return_tensors="pt", padding=True,
              truncation=True, max_length=args.max_len).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    say(f"批次：input_ids {tuple(enc['input_ids'].shape)} | 设备 {enc['input_ids'].device}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad][0]

    model.train()
    last_t = time.time()
    for step in range(1, args.steps + 1):
        out = model(**enc, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        # 🔴 指标名与语义必须一致（2026-09-30 修正）：
        #   旧名 elapsed_s / gpu_mem_gb 实际是「从启动起的累计墙钟」与「累计峰值显存」，
        #   与手册级 2「看 gpu_mem_gb + elapsed_s」的说法不符 —— 会被读成每步耗时/当步占用。
        #   现在拆成四个：*_total = 累计，step_time_s / gpu_mem_cur_gb = 当步。
        loss_v = float(loss.detach())          # 避免 requires_grad 张量转标量的 UserWarning
        now = time.time()
        step_time = now - last_t
        last_t = now
        say(f"TRAIN-smoke | ep: {step} | loss: {loss_v:.4f} | 用时 {now-t0:.1f}s")
        metric(step, "loss", loss_v)
        metric(step, "elapsed_s_total", now - t0)
        metric(step, "step_time_s", step_time)
        if device == "cuda":
            metric(step, "gpu_mem_peak_gb", torch.cuda.max_memory_allocated() / 1024**3)
            metric(step, "gpu_mem_cur_gb", torch.cuda.memory_allocated() / 1024**3)

    after = [p for p in model.parameters() if p.requires_grad][0]
    changed = not torch.equal(before, after.detach())
    max_mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    say(f"参数是否真的更新：{'✅ 是' if changed else '❌ 否'}")
    say(f"峰值显存：{max_mem:.2f} GB")
    metric(0, "param_changed", 1.0 if changed else 0.0)
    metric(0, "peak_mem_gb", max_mem)
    metric(0, "trainable_ratio", 100 * trainable / sum(p.numel() for p in model.parameters()))

    save_dir = os.path.join(run_dir, "adapter")
    model.save_pretrained(save_dir)
    say(f"💾 LoRA adapter 已存：{save_dir}")

    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": args.run_id, "status": "finished",
                   "created_at": t0, "finished_at": time.time(),
                   "kind": "rlinfra-lora-smoke", "model_path": args.model,
                   "steps": args.steps, "param_changed": changed,
                   "peak_mem_gb": round(max_mem, 3),
                   "trainable_params_m": round(trainable / 1e6, 3)}, f, ensure_ascii=False, indent=2)
    say(f"===== SMOKE END 用时 {time.time()-t0:.1f}s =====")
    log_f.close()
    m_f.close()
    return 0 if changed else 1


if __name__ == "__main__":
    sys.exit(main())
