> 🕒 v4 · 2026-10-09 16:15 · dsh-agent ｜ 变更：真模型已接：补 §五之二（首跑 3 个只有真服务端会打的 400）+ 修正 §3.3.1 过度承诺 ｜ 上一版：v3 · 2026-10-01 11:07

# Agent Harness：战果、装置与 H2 执行计划

> 这份文档回答三个问题：**我建了什么**（可验证）· **怎么复现**（命令原文）· **接下来怎么把它变成简历上的东西**。
> 进度看板（自动生成）→ [`学习进度看板.md`](学习进度看板.md)。

---

## 一、战果清单（每一项都有对应文件，不是描述）

| # | 交付物 | 文件 | 作用 |
|---|---|---|---|
| 1 | **最小 agent harness** | [`agent_harness_min.py`](../scripts/agent_harness_min.py) | 工具 schema + 控制循环 + 上下文档位 + 客户端可插拔 |
| 2 | **工具接口层** | 同上 `TOOLS` | 4 个工具、JSON Schema 声明、参数校验、**可行动的错误回写** |
| 3 | **轨迹契约** | [`agent_trace.py`](../scripts/agent_trace.py) | 固定行前缀 `TRACE \| ...`，可 grep 回放；同时落观测契约 |
| 4 | **确定性 verifier 集** | [`agent_tasks.py`](../scripts/agent_tasks.py) | 5 个任务 / 5 个判分器，**禁 LLM-as-judge** |
| 5 | **进度看板生成器** | [`learning_progress.py`](../scripts/learning_progress.py) | 扫仓库事实反推进度，手写打勾不算 |
| 6 | **H2 实验设计** | [`实验_上下文注入AB_设计.md`](实验_上下文注入AB_设计.md) | 可证伪判据 + 撤回条件 + 样本量纪律 |

**装置自检已通过**：`mock` 客户端 × 5 任务 × 3 次 = **15/15 通过，平均 3.0 轮，墙钟 0.88s**。

> ⚠️ **口径收窄**：mock 是"完美模型"脚本客户端，**只证明装置（循环/工具/轨迹/判分）能用，不代表任何模型能力**。
> 它的 acc 恒为 100%，任何用它算出来的"模型成绩"都是无效的。

---

## 二、复现命令（三条，从零到有数据）

```powershell
$env:PYTHONIOENCODING="utf-8"

# ① 看任务集与判分维度（不需要任何 key / GPU）
python scripts/agent_harness_min.py --list

# ② 装置自检（mock，不需要 key）—— 证明循环/工具/轨迹/判分全通
python scripts/agent_harness_min.py --client mock --repeat 3 --run-id h_smoke

# ③ 跑真模型（需要 key）—— 这一步才开始产生"模型能力"数据
$env:DEEPSEEK_API_KEY="sk-..."
python scripts/agent_harness_min.py --client deepseek --check
python scripts/agent_harness_min.py --client deepseek --context C --repeat 10
```

**产物落在哪**：
```
runs/harness/<run_id>.log                      人类可读轨迹（grep 'TRACE' 回放）
runs/harness/obs/<run_id>/metrics.jsonl        观测契约（每步指标）
runs/harness/obs/<run_id>/events.jsonl         结构化事件流
runs/harness/obs/<run_id>/run.json             运行身份 + 汇总（acc / 平均轮数 / 逐任务结果）
```

**看进度**：`python scripts/learning_progress.py` → 刷新 [`学习进度看板.md`](学习进度看板.md)

---

## 三、H2 执行计划（把设计变成数字）

### 3.1 自变量与固定项

| 项 | 内容 |
|---|---|
| **自变量** | 常驻规则档位 `--context A / B / C` |
| **A（全量）** | 全局规则 + 工作区规则 + 项目规则 + 全部技能（复现"水平下降"现场）|
| **B（中间）** | 工作区 + 项目规则，无技能 |
| **C（最小）** | 只有项目规则 |
| **固定项** | 同一 client、同一模型、**同一天内跑完**、`temperature=0`、同一批任务 |

### 3.2 目标数据量（设计文档 §2.2 的纪律）

```
5 任务 × 每档 ≥ 30 次任务记录  ⇒  三档合计 ≥ 90 次任务
命令：--repeat 6   （5 任务 × 6 次 = 30 次/档）
```

```powershell
python scripts/agent_harness_min.py --client deepseek --context A --repeat 6 --run-id h2_A
python scripts/agent_harness_min.py --client deepseek --context B --repeat 6 --run-id h2_B
python scripts/agent_harness_min.py --client deepseek --context C --repeat 6 --run-id h2_C
python scripts/learning_progress.py      # 看板会判断三档是否达标
```

### 3.3 ⚠️ 跑之前先算钱：成本预估与 token 核算

DeepSeek 是**付费 API**，A/B/C × 6 次 = **90 次任务调用**。所以 harness 内置两道闸：

```powershell
# ① 只花 1 次调用，外推整轮 token 与费用（不跑评测）
python scripts/agent_harness_min.py --client deepseek --repeat 6 --estimate-cost `
       --price-in <输入价> --price-out <输出价>

# ② 真跑时自动核算，落到 run.json 与 metrics.jsonl：
#    ctx/tok_in · ctx/tok_out · ctx/tok_in_total · ctx/tok_out_total · ctx/cost_cny
```
> 🔴 **单价由你填**（元/百万 token）：**代码里不写死任何厂商价格** —— 价格会变，写死就会过期撒谎。

### 3.3.1 `DeepSeekClient` 契约回归（**零花费**）

没有 key 时 `DeepSeekClient` 是"一行都没跑过的代码"，而它承载了最脆弱的部分。所以配了**本地假服务端**测试：

```powershell
python scripts/test_deepseek_client.py
# ✅ 7/7：原生 tool_calls · fenced 兜底 · 裸 JSON 兜底（tool/args 别名）
#         纯文本回复 · usage 记账 · 请求体形状 · HTTP 401 抛错
```
⇒ 保证**客户端解析这一层**是通的。🔴 **但这句话已被实测修正**：它只覆盖"请求/解析"，没覆盖"harness 组装的对话协议"，
真模型首跑仍连吃 3 个 400 —— 见 **§五之二**。

### 3.4 判据（跑之前就写死，见设计文档 §2.5）

| 结果 | 判定 |
|---|---|
| C 比 A 高 **≥15 个百分点** | ✅ **H2 成立**，效应量大到值得当结论 |
| C 比 A 高 **5–15 个点** | ⚠️ 方向成立、效应量不确定，加样本 |
| C 比 A 高 **<5 个点** | ❌ **不成立，撤回**；转 H3（工具膨胀）|
| C 比 A **更低** | 🔴 **反向**：常驻规则是净收益，"水平下降"另有原因 |

### 3.5 🔴 必须写进报告的诚实短板

1. **服务端是黑盒**：无权重/无解码参数观测点 ⇒ 若当天服务端变了，结论受污染。
   缓解：三档同一天跑完 + 记录日期与模型标识 + **只当"当日快照"**，不做跨月比较。
2. **样本量**：每档 < 30 时，"高 10 个点"很可能是噪声（本项目实测：只换 seed，best 覆盖率差 13.14 点）。
3. **mock 数据无效**：任何混入 mock 的统计都必须剔除（看板已自动排除）。

---

## 四、这套东西怎么用在简历/面试上

### 一句话版本（可背）

> 观察到模型"水平下降"但无法归因 → 拆成 6 条**互相独立**的假设 → 自建 **agent harness + 确定性 verifier + 轨迹契约**（因为现成框架测不了"上下文注入"这个自变量）→ 用 A/B/C 三档实测通过率差 X 个百分点 → 因服务端黑盒，结论只覆盖当日快照。

### 这句话里有三个考核点

| 考核点 | 证据 |
|---|---|
| **能做装置**（不是只会用框架） | `agent_harness_min.py` + `agent_trace.py` + 5 个 verifier |
| **研究方法论** | 可证伪判据、撤回条件、样本量纪律、**主动写短板** |
| **工程习惯** | 产物隔离（`runs/`）、观测契约（`metrics.jsonl`）、原子写、可回放轨迹 |

### 面试可能被追问的三个真问题（先想清楚）

1. **"为什么不用现成框架？"** —— 因为要测的自变量是"上下文注入档位"，框架里是黑盒；且需要确定性 verifier 与轨迹契约，框架给不了可比的数字。
2. **"tool call 解析失败怎么办？"** —— 已实现两路：原生 `tool_calls` + 正文 JSON 兜底（`_parse_text_tool_call`）；未实现的是多 tool call 并行与部分失败重试，**这是已知缺口**。
3. **"怎么保证评测不骗自己？"** —— verifier 全程序化，禁 LLM-as-judge；mock run 在统计里自动剔除；三档同天跑以对抗服务端漂移。

---

## 五、已知缺口（不许藏）

| 缺口 | 状态 | 影响 / 补法 |
|---|---|---|
| 只执行**第一个** tool call | ✅ **已补** | 一轮执行全部 call（`--max-calls-per-turn`）；mock 的 a1/a2/a4 已用「并发 write+read 自检」验证 |
| 无**多轮重试/自愈** | ✅ **已补** | `--max-retries`：工具失败退避重试；**工具名写错不重试**（让模型自纠）；重试次数进指标 `tool/<name>/retries` |
| 无**上下文预算控制** | ✅ **已补** | `--ctx-budget` + `est_tokens()`：超预算优先压最旧的工具结果，保留最近 `--keep-tool-results` 条；指标 `ctx/est_tokens`、事件 `ctx_trimmed` |
| 观测平台未指向 harness | ✅ **已补** | 见 §六，8078 已在跑，22 个指标名可选 |
| 未接**真模型** | ✅ **已补**（2026-10-09） | `deepseek-flash` × A/B/C × 6 次 = **90 次任务记录**，同日跑完；数据与结论见 [`H2_实验报告.md`](H2_实验报告.md)。**结果是通过率触顶（三档全 100%）⇒ 本批不能检验 H2**，可用信号只剩轮数/token |

> 这三个补丁不是"锦上添花"：**上下文预算**正是 H3（工具/技能膨胀）的自变量，
> **重试**是 agent harness 与"一次性脚本"的分界线，**多 tool call** 是并行工具的前提。

---

## 五之二、真模型首跑暴露的 3 个坑（**mock 永远测不出来**）

> 🔴 这一节是 §3.3.1 那句"一旦拿到 key，第一次调用就是通的"的**更正**：
> 假服务端只验证了**客户端解析**，没验证**harness 组装的对话协议**，所以真模型第一跑就 400 了三次。

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | 第 2 轮起 `HTTP 400 Messages with role 'tool' must be a response to a preceding message with 'tool_calls'` | `run_task()` 先 append `tool` 结果、后 append `assistant(tool_calls)`；mock 不看顺序 | 顺序反过来：先 assistant(tool_calls) 再逐条 tool（`agent_harness_min.py` 控制循环内） |
| 2 | 第 2 轮起 `HTTP 400 The reasoning_content in the thinking mode must be passed back to the API` | DeepSeek 思考模式 + `tools` 时，历史轮的 `reasoning_content` **必须原样回传**；harness 只存了 `content` | `DeepSeekClient` 透传 `reasoning_content`，assistant 消息带上它 |
| 3 | 同一 400 在"服务端返回空串 reasoning_content"时**仍然**出现 | 判据错：按"值非空"决定是否回传；服务端要求的是**键存在** | 改为 `is not None`（`None`=服务端没给，`""`=给了但空）——实测修完 a3 从 0/3 变 3/3 |
| 4 | `run_command` 发 `ls -la ... \| od -c`，rc=1 白烧两轮重试 | 工具没自述环境：本机 shell 是 Windows cmd.exe | 工具 description 写明 + 报错时回一句可行动提示（`dir/findstr/type` 或改用 `read_file`） |

**教训（可直接进面试）**：*"零花费回归"只能覆盖**你说得出形状的东西**；协议**顺序**与**必填字段**这两类失败，只有真服务端会打你。*

---

## 六、观测平台指向 harness（已跑通）

harness 产物落在 `runs/harness/obs/<run_id>/`，与观测平台契约一致（`run.json` / `metrics.jsonl` / `events.jsonl`），
**不用改任何代码**，只要把平台的 `--runs-dir` 指过去：

```powershell
# 独立实例（不动原来的 8077），端口 8078
python -m web.server --port 8078 --runs-dir "__REPO_DIR__\runs\harness\obs"
# 打开 http://127.0.0.1:8078
```

**已验证**：
```
GET /api/runs              → 4 个 run（h_smoke / h_verify / h_gaps / h_gaps2）
GET /api/runs/h_gaps2/names → 22 个指标名，含：
  ctx/est_tokens · loop/n_tool_calls · loop/n_retries
  tool/<name>/ok · tool/<name>/retries · task/<id>/pass · run/acc
```

⚠️ **一个实现细节值得记住**（也是给面试讲的真话）：
`a5` 任务里 mock 先发了一个不存在的工具名 `search_log`，再发 `run_command`。
harness 的处理是 **不重试、把可用工具列表回写给它**，模型下一轮自己改对。
⇒ 这是"**错误信息要可行动**"的具体落地：错误回写里带 `可用工具：[...]`，
比只回 `unknown tool` 的自我修正率高得多。

