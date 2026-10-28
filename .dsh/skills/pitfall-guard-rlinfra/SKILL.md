> 🕒 v4 · 2026-10-22 17:08 · dsh-agent ｜ 变更：新增 1.12：PowerShell Get-Content -Raw|Set-Content 静默毁 UTF-8 中文（本次实测踩到，靠 git 恢复） ｜ 上一版：v3 · 2026-10-06 13:03

## 🕒 产物戳

规则与戳形见 `~/.dsh/AGENTS.md`「🕒 产物戳」；**戳放正文最前（有 frontmatter 则其后），戳行不许手改、版本只增不复用**。戳形：`> 🕒 vN · YYYY-MM-DD HH:MM · <作者> ｜ 变更：<一句话> ｜ 上一版：vN-1 · <时间>`

```powershell
$S = "C:\Users\__user__\.dsh\bin\artifact_stamp.py"
python $S new  <file> --note "初稿"          # 新建 → v1
python $S bump <file> --note "改了什么"       # 改完 → v2/v3…（先改内容再 bump）
python $S check <目录>                        # 体检：缺戳 / 改过没盖戳（退出码 1 = 违规）
```

"""RL_infra 项目级踩坑档案（pitfall-guard-rlinfra）。

⚠️ 命名规则：项目级技能**必须**叫 `pitfall-guard-<项目短名>`，绝**不能**直接叫 `pitfall-guard`
—— DSH 里「最近一层的同名技能直接胜出」，同名会把用户级那个整体遮蔽掉（实测：项目级建好后用户级的自检清单整个消失）。

用户级（通用坑）见 `~/.dsh/skills/pitfall-guard/SKILL.md`；高算平台作业见 `hpc-jobs` 技能。
**本文件只放「只有本项目才会踩」的坑。**

---

## 1.1 日志格式一漂移，诊断脚本会「静默少解析」（不是报错！）

**现象**：`utils/analyze_training_log.py` 报告看着正常，但环境快照块是空的、
奖励分量 top 只有零星几个、测试汇总没有 —— **它不报错，只是少解析**。

**根因**：诊断脚本靠**正则匹配行前缀**（`EP | ep:` / `TRAIN-` / `REWARD(每智能体每步均值) |` …）。
业务代码里手拼字符串 `print(f"EP|ep:{ep}...")`（少个空格、换个字段顺序）就会匹配不上，
而 `analyze()` 遇到匹配不上只是跳过，不抛异常。

**判据**：报告里某一块整体为空 → 十有八九是格式漂了，不是「这次训练真的没有该数据」。

**解法（本项目已固化）**：`utils/smoke_test.py` 第⑦项「日志契约 ↔ 诊断脚本」——
用真 `TrainLogger` 造日志 → `analyze()` 解析 → **逐字段断言**。
**改任何日志相关代码后，必须跑它。** 业务代码里禁止手拼日志行（一律走 `rlinfra/logger.py`）。

---

## 1.2 `mp` 多进程环境池在受限沙箱里会挂 —— 别把默认值改成 `parallel=True`

**现象**：本机 DSH agent 沙箱里跑 `--parallel` 报 `EPERM` / 卡死 / 子进程起不来。

**根因**：`multiprocessing.Queue` 走**命名管道**，受限沙箱禁止打开命名管道（规范 §8.7 同源）。

**解法**：`config.py` 的 `parallel` **默认 False**（走 `SyncEnvPool`）；
`run_smoke()` 里**强制** `parallel=False`；冒烟自检永远只验单进程分支。
真要验多进程：`python utils/smoke_test.py --mp`（在正常终端里跑）。

---

## 1.3 改参数后忘了 `cfg._derive()` → `obs_dim` 与环境对不上

**现象**：`AssertionError: 观测维度 54 != obs_dim 30` 或模型加载时报维度不一致。

**根因**：`obs_dim` 是由 `MADQN_patch` **派生**的，不是写死的。dataclass 构造时自动算，
但**手工 `setattr` 改参数后不会重算**。

**解法**：`main.py` 的 CLI 路径已自动调 `_derive()`；**自己写脚本/notebook 调 Config 时，
改完参数必须显式 `cfg._derive()`**（`smoke_test.py` 的 `t_end2end` 是范例）。

---

## 1.4 奖励分量的名字是「接口」，改名会静默废掉诊断规则

**现象**：奖励诊断块不再报「里程碑从未触达」/「负惩罚主导」，即使问题确实存在。

**根因**：`analyze_training_log.py` 的启发式规则**按名字找分量**
（`milestone*`、`stagnation_penalty`、`repeat_penalty`）。

**解法**：新增/改名分量时，同步确认诊断脚本的规则还能命中；
`milestone_*` 前缀与 `*_penalty` 后缀建议保留（语义清晰且被规则覆盖）。

---

## 1.5 产物目录只有两级 —— 别再嵌套第三层

**现象**：`runs/<name>/a/b/c/` 越套越深，归档和对比都变麻烦。

**规范**：`RunPaths.child(sub)` 是**二级目录的唯一入口**，多个模块要用同一个下游目录时
**全部调它传同一个 `sub`**（复用同一处，不各自嵌套）。这是规范 §3.1 的硬要求。

---

## 1.6 🔴 训练日志里写着 90.5% 的 best，`--mode test` 只有 53.9% —— 差 36 个点

**现象**：`runs/demo_300/train.log` 里 best 覆盖率 90.5%（第 245 回合），
`python main.py --mode test --name demo_300` 实测 **53.9%**，成功率 0%。

**根因（两条主因，都不是「训练不够」）**：
1. 🔴 **判据与产物对不上**：训练回合的覆盖率由 **rollout 权重副本**（actor 线程持有）产生，
   而 `save()` 存的是 **online 权重** —— 用 A 的表现挑 B。两者每轮只同步一次，本来就不是同一套权重。
2. **max 选择偏差**：在 300 个**单回合样本**里取最大值，挑中的是「运气好的一回合」，不是「最好的模型」。

（ε-greedy 的探索红利是第三位因素，且**方向不固定** —— 实测同一阶段里 +14 点与 −30 点都出现过，
所以**别把它当成「训练端一定偏高」的经验公式**，要看日志里打出的「探索红利」实测值。）

**判据**：`--mode test` 的结果与日志 best 记录差 > 5 个点 → 就是这个坑。

**解法（已固化）**：`madqn_trainer.GreedyProbe` —— 每 `MADQN_eval_interval`（默认 25）回合，
用 **online 权重 + 固定地图 + 贪心策略**（= 与 `--mode test` **完全同一套判据**）跑一回合，按它存 best。
修复后实测（`runs/demo_probe`）：best 90.21% ↔ test 90.21%，**完全一致**，成功率 100%。
**别改回**「按训练回合覆盖率选 best」。

---

## 1.8 🔴 清理旧 checkpoint 时，**刚存的那个被自己删掉了**

**现象**：日志交替打印
`保存完成：madqn_final_1.pth` → `清理旧 final 模型 1 个：['madqn_final_1.pth']` → 又存 `madqn_final_1.pth`…
即：**每次保存后，刚写的文件立刻被清理逻辑删掉**，序号永远停在 1，最新存档全部丢失。

**根因**（两个改动单独看都对，凑一起就错）：
① `next_final_model()` 找的是「**第一个空号**」；② 清理按**序号从小到大**删最旧的。
清理删掉 `final_1` 后空号又变成 1 → 重新写 `final_1` → 而 `final_1` 恰好是最小号 → **下一轮就被当最旧的删掉**。

**判据**：日志里同一个文件名反复出现「保存完成」+「清理」成对出现。

**解法（已固化）**：`next_final_model()` 改成「**已有最大序号 + 1**」（序号单调递增、永不复用空号），
新存的永远是最大号，永远落在保留区。已在 `paths.next_final_model()` 的 docstring 写明原因。
**别改回「第一个空号」** —— 只要开了清理就会复现。

**配套教训**：**删文件的逻辑必须单独写一条冒烟检查**（`utils/smoke_test.py` 第⑨项）：
造假文件 → 调用清理 → 断言「该删的删了、`best`/`final`(无序号)/`pretrain`/别的算法的文件全部幸存」。
只测 happy path 会漏掉这一类。

---

## 1.9 冒烟自检第⑧项会真开线程跑训练 —— 别在它里面改全局随机种子状态

**现象**：连续调用 `train_madqn_ctde` 后，后续测试的随机性不如预期。

**根因**：训练函数会 `torch.manual_seed(cfg.seed)` / `np.random.seed(cfg.seed)`（这是**有意**的可复现设计）。

**解法**：需要独立随机性的测试，放在端到端项**之前**，或自己重新播种。
（当前 `smoke_test.py` 已按此顺序排：①~⑦ 在 ⑧ 之前。）

---

## 1.10 🔴 本机有**全局 git-lfs filter**：一次 `git add` 就把 3.4GB 塞进 `.git/`

**现象**：仓库只有 64 个小文件、`git ls-files` 里没有任何大文件，
但 `.git/` 占 **3390 MB**；`git rev-list --objects` 找不到大 blob，`pack/` 也是空的。

**根因**：本机（用户级）装了 git-lfs 且配置了 filter，`git add` 时把大文件转成 LFS 指针，
实体落进 **`.git/lfs/objects/`**。本项目**没有 `.gitattributes`**，所以指针没进索引
—— 结果这个 3.4GB 对象成了**彻底孤立、任何已跟踪文件都不引用**的缓存。

**判据**：`.git` 体积与 `git ls-files` 体量严重不符 → 先查
`Get-ChildItem .git -Recurse -File | Where-Object { $_.Length -gt 100MB }`，别只看 pack/。

**核查「能否安全删」**：用对象哈希去搜全仓——
`git grep -l "<lfs对象哈希>"` 无命中 ⇒ 孤儿缓存，可移走。
（本例对象哈希 `58858233513d...` 与 DeepSeek 1.5B 权重的 sha256 形态一致，疑为下载缓存误入。）

**解法**：移出仓库到项目外备份（**可逆**，别直接删），并确认 `.gitignore` 已排除大目录
（本项目已排 `model_cache/`、`.tmp/`、`runs/`、`output/_kb/`）。

---

## 1.11 🔴 `git commit --amend` 不带 `-- <path>` 会**改写 HEAD 的提交信息**

**现象**：想修正「上一条」提交的信息，执行
`git commit --amend -F msg1.txt`，结果 **HEAD（最新那条）的信息被换成了 msg1 的内容**
—— 而 HEAD 的**文件内容没变**，于是提交历史里出现一条「内容和信息完全对不上」的提交。

**根因**：`--amend` 永远针对 **HEAD**；`git commit -F <msg> -- <path>` 的定向只在
「新建提交」时有效。想改的若是 HEAD 的**父提交**，`--amend` 帮不上忙。

**判据**：`git log --oneline` 里某条的信息与 `git show --stat` 的文件对不上 → 就是这个坑。

**解法**：
1. 动手前先 `git branch backup_xxx HEAD` 存档；
2. 只改 HEAD 信息 → `git commit --amend -F <msg>`（**不加路径**）；
3. 改更早的提交 → 用 `git rebase -i` 的 `reword`（或 `GIT_SEQUENCE_EDITOR` + `GIT_EDITOR` 脚本化），
   **别**用 `--amend`；
4. 改完必须 `git show --stat --oneline HEAD` 复核「信息 ↔ 内容」是否仍匹配。

**配套**：本项目提交走**定向路径** `git commit -F msg -- <path...>`，
目的是不带走别的会话预暂存的文件；但这条规矩**不适用于 `--amend`**（见上）。

---

## 1.12 PowerShell 里 `Get-Content -Raw | Set-Content` 会**静默毁掉 UTF-8 中文**（2026-10-22 实测）

**现象**：想批量替换 md 里的一个标题，执行

```powershell
(Get-Content "docs/迁移演练.md" -Raw).Replace('### 4.4 修复验收','### 5.1 修复验收') |
    Set-Content "docs/迁移演练.md" -Encoding utf8 -NoNewline
```

之后文件里中文全变成 `淇楠屾敹` 这类乱码（用 `read` 工具直接能看出）。**命令不报错**，
`git status` 只显示"已修改"，`Select-String` 也照样显示（它按同一套错编码解码，肉眼分不清）。

**根因**：Windows PowerShell 5.1 的 `Get-Content` **默认按 ANSI/GBK 解码**（不是 UTF-8），
`Set-Content -Encoding utf8` 再把已经错的字符串编回 UTF-8 ⇒ **二次编码**，原文不可逆。

**判据**：改完用 `python -c "print(open(f,encoding='utf-8').read()[:80])"` 或 `read` 工具看前 80 字；
**不要**用 `Select-String` / `Get-Content` 的输出判断文件本身好坏（那只是控制台编码）。

**解法**：
1. 本仓库文本文件一律用**文件工具**改（`edit` / `write`），不要用 PowerShell 管道重写；
2. 必须脚本化时用 Python 且显式 `encoding="utf-8"`（`newline="\n"`）；
3. 已经毁了 → `git checkout -- <file>` 回滚，再用文件工具重做（本次靠 git 救回，损失 = 一次未提交编辑）。
