#!/bin/bash
# 登录节点冒烟：transformers 能否真正加载 DeepSeek-R1-Distill-Qwen-1.5B（只读权重，不做训练）
set -u
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python
M=/storage/public/home/__HPC_ACCOUNT__/RL_infra/model_cache/DeepSeek-R1-Distill-Qwen-1.5B
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8

"$PY" - <<PYEOF 2>&1 | tail -20
import time, torch
from transformers import AutoTokenizer, AutoConfig
t0 = time.time()
tok = AutoTokenizer.from_pretrained("$M")
cfg = AutoConfig.from_pretrained("$M")
print("tokenizer + config OK", round(time.time() - t0, 1), "s")
print("  arch:", cfg.architectures, "| layers:", cfg.num_hidden_layers,
      "| hidden:", cfg.hidden_size, "| vocab:", cfg.vocab_size)
ids = tok.apply_chat_template(
    [{"role": "user", "content": "1+1=?"}], add_generation_prompt=True, return_tensors="pt")
print("  chat_template 渲染后长度:", ids.shape)
print("  解码前 60 字符:", tok.decode(ids[0][:60])[:120].replace("\n", "\\n"))
PYEOF
