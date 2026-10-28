> 🕒 v2 · 2026-10-28 21:07 · dsh-agent ｜ 变更：复扫：敏感项改为不列原值；快照文件数 136 ｜ 上一版：v1 · 2026-10-28 21:04

# CI 交付报告：四道门禁，GitHub Actions 真跑全绿

**仓**：<https://github.com/yanggaihao5-sys/RL_infra>（脱敏单提交快照，`ef98841`）
**CI run**：<https://github.com/yanggaihao5-sys/RL_infra/actions/runs/40692704072>（`success`，53 秒）
**本地对应提交**：`698ba1b`（公开仓为脱敏快照，sha 不同、内容同源）

---

## 0. 结论（四句）

1. **CI 不是只有 yml**：2026-10-28 20:01（UTC 13:01）真实触发，5 个 job（含 py3.9/3.12 矩阵）**全部 success**，总耗时 53 秒。
2. **测试是真测**：23 项 pytest 在 py3.9 与 py3.12 上均为 `23 passed`；端到端 job 真跑 30 回合训练 + 5 张随机图评测，并逐项校验产物，不是“进程退出码 0 就算过”。
3. **部署闸门是真装**：wheel 在干净 venv 里安装成功，从 `/tmp` 任意目录 `import rlinfra / algos` 成功；分发包只含库与算法包（25 个文件），应用层 `config.py/main.py` 未混入。
4. **诚实边界**：端到端只证“装置通”（30 回合、成功率 0%），不是收敛实验；同一命令在本地与 runner 上数字不同（43.09% vs 33.45%），跨环境逐字复现**未做到、也未作为本轮判据**。

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `.github/workflows/ci.yml` | 四类 job：静态 / 单元 / 端到端 / 打包安装 |
| `tests/`（7 个文件，23 项） | 配置派生与校验、日志契约、产物路径清理边界、注册表懒加载、环境/智能体/回放池契约、产物闸门自测 |
| `utils/verify_run.py` | 6 项产物闸门：日志 / 真评测 / best 模型 / 曲线图 / 观测契约 / 数字自洽 |
| `utils/check_boundaries.py` | ⑥ 改为不 import torch，边界 job 真正做到零第三方依赖 |
| `pyproject.toml` | 新增 `dev = ["pytest>=8.0"]` optional extra |
| `README.md` | 自动化状态改为“CI 真跑全绿”，补 4 道门禁表与本地复现命令 |
| `.gitignore` | 补 `build/`、`dist/` |

---

## 2. 流水线结构

| job | 安装 | 门禁 | runner 耗时 |
|---|---|---|---|
| `static` 静态检查 | **无第三方依赖** | 全源码语法编译 + 8 项库边界 | **7s** |
| `unit` 单元测试（3.9 / 3.12） | numpy + CPU torch + pytest | 23 项 pytest，两个 Python 版本 | **46s / 48s** |
| `e2e` 端到端 | + matplotlib | 冒烟 9 项 → 30 回合训练+评测 → 日志诊断 → 出图 → 观测契约 → `verify_run` 6 项 | **44s** |
| `package` 部署闸门 | build | wheel/sdist → 内容边界 → 干净 venv 安装 → 任意目录 import | **17s** |

触发：`push main` / 任意 PR / 手动 `workflow_dispatch`；`concurrency` 同 ref 自动取消旧 run。

---

## 3. 真跑证据

### 3.1 run 级

| 项 | 值 |
|---|---|
| run id | `40692704072` |
| 结论 | **success**（5/5 job） |
| 快照 SHA | `ef98841eec8acd20cfabfd71b31d0eaaa58e823e` |
| 起止（UTC） | 13:01:25 → 13:02:18（53s） |
| 产物 | `e2e-ci_smoke`（295,087 B）· `dist`（123,216 B） |

### 3.2 逐 job 关键输出（原日志可点 run 页核对）

- **static**：`8/8 项通过 —— 边界完好`。
- **unit / py3.9**：`23 passed in 2.03s`；**unit / py3.12**：`23 passed in 4.09s`。
- **e2e**：
  - 冒烟 9 项 `✅ 全部通过`；
  - `训练结束：30 回合 / 墙钟 2.9s / 最优覆盖 36.60%（第 25 回合）`；
  - `测试汇总 | 全局平均覆盖率：33.45% | … | 测试轮数：5 | 覆盖标准差 5.52pp`；
  - `6/6 项通过 —— 产物链路完好`。
- **package**：
  - `Successfully built rlinfra-0.1.0.tar.gz and rlinfra-0.1.0-py3-none-any.whl`；
  - `wheel: dist/rlinfra-0.1.0-py3-none-any.whl | 文件数: 25`；
  - `clean-install OK: /tmp/clean-venv/.../rlinfra/__init__.py`。

### 3.3 发布快照复扫

- 快照文件数 **136**（`git ls-files` 138，脱敏管线按规则 drop 2 个知识库派生物）。
- 敏感项零命中：高算账号 / 内网登录 IP / 内网主机名 / 本机项目路径 / 用户名 / 学号邮箱 —— **0 hits**。
- 快照提交作者使用 GitHub noreply，不带本机邮箱。

---

## 4. 本地等价验证（2026-10-28，Windows + Python 3.12.4 + torch 2.8.0+cpu）

| 步骤 | 命令 | 结果 |
|---|---|---|
| 边界 | `python utils/check_boundaries.py` | 8/8，exit 0 |
| 冒烟 | `python utils/smoke_test.py` | 9/9，`✅ 全部通过` |
| 单测 | `python -m pytest -q tests` | **23 passed in 7.08s** |
| 端到端 | 训练→评测→诊断→出图→转观测→闸门 | 30 回合 **51.7s**；eval `43.09% ±4.39`；223 条指标；**6/6** |
| 打包 | `pip wheel .` + 干净 venv 安装 | 25 文件；任意 cwd import 成功；**49.3s** |

> 本地执行时 Windows 低完整性沙箱不允许受限进程写 `runs/`、`tests/__pycache__`、系统临时目录，
> 因此本地验证用完全权限跑；**CI 在 Linux runner 上无此限制**，两端命令完全一致。

---

## 5. 修掉的真问题（不是“把 yml 写好看”）

### 5.1 边界 job 在 runner 上直接失败 —— ⑥ 项偷偷 import 了 numpy/torch

- 旧 run（<https://github.com/yanggaihao5-sys/RL_infra/actions/runs/40517036335>）里 `冒烟` job 绿、`库边界` job **红**。
- 日志：`ModuleNotFoundError: No module named 'numpy'`，位置在边界检查 ⑥ `spec.train`（触发 `algos.madqn` import）。
- 修法：⑥ 改为**静态验证**登记字段 + `find_spec` + 断言 `algos.madqn` 未进 `sys.modules`（真正验证懒加载），
  “登记的名字真能加载出可调用对象”交给装了 torch 的 unit/e2e job。边界 job 现在真的零第三方依赖（7s）。

### 5.2 旧的端到端“绿”是假绿 —— 12 回合一次评估都不触发

- `COMMON_eval_interval = 25`，旧 CI 只跑 12 回合 → 不存 `madqn_best.pth` → `evaluate_madqn` 直接
  `return {"skipped": True}`（`madqn_trainer.py:386-388`），`--then-test` 仍退出 0，job 仍绿。
- 修法：端到端改 **30 回合**（跨过 25），并新增 `utils/verify_run.py`：
  **显式拒绝 `eval.json.skipped`**，还要求 `test/coverage` 与 `eval.json` 数值一致。
- 自测覆盖：`tests/test_verify_run.py` 专门构造 `skipped` 与数值不一致两种坏产物，断言闸门必须失败。

### 5.3 README 自述“没有 CI”

已改为“GitHub Actions CI 真跑并全绿”，并补 CI 徽章与 4 道门禁表。

### 5.4 打包产物边界以前没有自动化

`package` job 现在检查 wheel 里必须有 `rlinfra/__init__.py`、`algos/madqn/__init__.py`，
必须**没有** `config.py`、`main.py`，再进干净 venv 验证可安装、可 import。

---

## 6. 判据与未覆盖缺口（诚实清单）

1. **端到端是装置验证，不是效果结论**：30 回合、成功率 0%、覆盖率 33–43%，只证明“训练→评测→日志→图→观测契约”这条链不断。
2. **跨环境数值不稳定**：同一命令本地 `43.09%`、runner `33.45%`；本地最佳覆盖 42.01%，runner 36.60%。
   两端 torch 2.8.0 vs 2.15.0、numpy 1.26.4 vs 2.5.4，**原因未做受控取证**；因此 CI 不能做“数字回归”门禁。
3. **没有 GPU/CUDA job**：全部 CPU；`device=auto` 的 CUDA 分支未被 CI 覆盖。
4. **没有 Windows runner**：本地 Windows 跑过，但 CI 只跑 ubuntu-latest；Windows 专属问题（路径/编码/多进程）无自动门禁。
5. **e2e 只跑 py3.12**：py3.9 只覆盖单测，不覆盖训练链路。
6. **公开仓是脱敏快照**：单提交、135 文件，不含本地历史；CI 跑在快照上，因此“本地 HEAD 与公开 HEAD sha 不同”是预期。
7. **GitHub 注解两条非失败告警**：`actions/checkout@v4`/`setup-python@v5` 的 Node20 弃用提示、`ubuntu-latest` 将迁移 Ubuntu 26；当前不影响结论，后续升级 action 大版本即可。

---

## 7. 复现（照抄即可）

```powershell
# 本地四道门禁
python utils/check_boundaries.py
python -m pytest -q tests
python main.py --episodes 30 --name ci_smoke --then-test
python utils/analyze_training_log.py runs/ci_smoke/train.log
python utils/plot_metrics.py runs/ci_smoke/train.log
python utils/log_to_metrics.py --log runs/ci_smoke/train.log --out runs/obs --force
python utils/verify_run.py --run runs/ci_smoke --obs runs/obs/ci_smoke
python -m build
```

```bash
# 远端：点一下就会跑（或 push main）
gh workflow run ci -R yanggaihao5-sys/RL_infra
gh run watch -R yanggaihao5-sys/RL_infra
```

---

## 8. 下一步（可选，不在本轮范围）

- 加 tag 触发的 Release job（把 `dist` 挂到 GitHub Release），把“安装包可下载”也变成门禁。
- 给端到端加固定地图模式做**可比的数字回归**（先解决跨 torch 版本的可复现性）。
- 加 Windows job（只跑单测 + 冒烟，控制耗时）。
- 需要 GPU 时，用自托管 runner 加一个 `gpu-smoke` job；公共 runner 无 GPU。
