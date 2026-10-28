#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# 🔴 结论（2026-09-30 实测）：**高算平台整台机器都没有外网** —— 登录节点 ln01 与
#    计算节点一样：DNS 解析失败、pypi/hf-mirror/github/1.1.1.1 全部 http=000。
#    所以本脚本放在高算上**没有意义**，保留只为留个证据。
#
# 正确做法：在**本地**（有外网的机器）下载，再 scp 上传到
#    /storage/public/home/__HPC_ACCOUNT__/RL_infra/model_cache/<模型名>/
#    实测：本机 curl hf-mirror 约 6 MB/s，1.5B 权重 3.4GB 约 10 分钟。
#    ssh_upload 走 __HPC_LOGIN_IP__（校园网地址）可达。
#
# 本机下载命令（PowerShell，已实测可用）：
#   $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
#   curl.exe -L --retry 5 -C - -o model.safetensors `
#     "https://hf-mirror.com/<owner>/<name>/resolve/main/model.safetensors"
#   # 小文件（config.json / tokenizer.json / tokenizer_config.json /
#   #        generation_config.json）逐个同法下载，别用 hf CLI 的 --exclude 列表
#   # ⚠️ hf CLI 的 Xet 传输走 xethub.hf.co，国内 401 —— 必须 HF_HUB_DISABLE_XET=1 且卸载 hf_xet
#   # ⚠️ `--exclude "*.jpg" "*.png"` 会被 PowerShell 解析坏（Invalid argument），
#   #    正确写法：每个模式各写一个 --exclude
# ════════════════════════════════════════════════════════════════════
echo "❌ 高算平台无外网，无法在此下载模型。请在本地下载后 scp 上传（见脚本头说明）。"
exit 1
