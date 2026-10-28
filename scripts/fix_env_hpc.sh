#!/bin/bash
# 高算登录节点：补齐 transformers 依赖的正确版本（离线 whl）
set -u
PY=/storage/public/home/__HPC_ACCOUNT__/anaconda3/envs/pytorch/bin/python
W=/storage/public/home/__HPC_ACCOUNT__/RL_infra/wheels

"$PY" -m pip install --no-index --no-deps --find-links "$W" \
    "tokenizers-0.19.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "fsspec-2024.6.1-py3-none-any.whl" \
    "pyarrow-20.0.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" 2>&1 | tail -3

echo "--- 验证 import ---"
"$PY" -c "import torch, transformers, peft, accelerate, datasets, sentencepiece, tokenizers; print('torch', torch.__version__); print('transformers', transformers.__version__); print('peft', peft.__version__); print('accelerate', accelerate.__version__); print('datasets', datasets.__version__); print('tokenizers', tokenizers.__version__)" 2>&1 | tail -8
