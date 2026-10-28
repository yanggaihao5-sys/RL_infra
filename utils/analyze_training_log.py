# -*- coding: utf-8 -*-
"""
训练日志多维诊断脚本（通用版）

设计目标：
  1. 全量读入大日志，但只输出精简的结构化诊断报告（省 token）。
  2. 多维度分析：环境 / 训练趋势 / 覆盖率 / 奖励分量 / 测试 / 资源 / 模型大小。
  3. 自动诊断「训练问题」与「奖励函数问题」，并给出通用解决方向。
  4. 通用性：日志格式对齐后（见文末「格式契约」），本脚本无需改动即可套用到其他项目。

用法：
    PYTHONIOENCODING=utf-8 python utils/analyze_training_log.py <日志1> [日志2 ...]

依赖：仅 Python 标准库（re/os/argparse/datetime），可直接复制到其他项目。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
格式契约（其他项目对齐这些「行前缀」即可复用本脚本）：
  EP | ep: N | R: X | cov:Y% | steps: N | loss:X | [curQ:X |] td:X      # 每回合汇总（必选）
  TRAIN[|-XXX] | ep: N | loss:X | curQ:X | tgtQ:X | ... | eps:X | td:X  # 训练指标（可选）
  REWARD(每智能体每步均值) | name:+v name:-v ...                        # 奖励分量（可选，强烈建议）
  回合：N | 全局奖励：X | 覆盖率：Y%                                    # 最优模型（可选）
  运行设备：xxx | 地图平均总奖励：X                                      # 概览（可选）
  环境配置一览 / 网格尺寸 / 智能体起始位置 ...                          # 环境快照（可选）
  全局平均覆盖率：X% / 成功率：X% / 平均完成步数：X                     # 测试汇总（可选）
  Started at ... / Results reported at ... / CPU time: ... / Max Memory: ...  # 资源（可选）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import os
import re
from datetime import datetime

# ────────────────────────── 解析正则（格式对齐处） ──────────────────────────
# 每回合汇总：MADQN 有 curQ，独立 DQN 无 curQ（可选组）
EP_RE = re.compile(
    r"EP \| ep:\s*(\d+)\s*\|\s*R:\s*([\d.eE+-]+)\s*\|\s*cov:\s*([\d.eE+-]+)%\s*\|\s*steps:\s*(\d+)"
    r"\s*\|\s*loss:\s*([\d.eE+-]+)\s*\|(?:\s*curQ:\s*([\d.eE+-]+)\s*\|)?\s*td:\s*([\d.eE+-]+)"
)
# 训练指标（统一抓核心 6 字段，gru_norm/neigh 单独可选抓）
TRAIN_CORE_RE = re.compile(
    r"TRAIN(?:-[A-Za-z0-9]+)? \| ep:\s*(\d+)\s*\| loss:\s*([\d.eE+-]+)\s*\| curQ:\s*([\d.eE+-]+)\s*\| tgtQ:\s*([\d.eE+-]+)"
    r".*?eps:\s*([\d.eE+-]+)\s*\| td:\s*([\d.eE+-]+)"
)
GRU_NORM_RE = re.compile(r"gru_norm:\s*([\d.eE+-]+)")
# 最优模型行
BEST_RE = re.compile(r"回合：(\d+)\s*\|\s*全局奖励：([\d.eE+-]+)\s*\|\s*覆盖率：([\d.eE+-]+)%")
# 概览
DEVICE_RE = re.compile(r"运行设备：(\w+)")
PROB_SUM_RE = re.compile(r"地图平均总奖励：([\d.]+)")
EPISODES_RE = re.compile(r"总轮次=(\d+)")
# 奖励分量行：name:+v name:-v ...
REWARD_COMP_RE = re.compile(r"REWARD\(每智能体每步均值\)\s*\|\s*(.+)")
# 测试汇总
TEST_COV_RE = re.compile(r"全局平均覆盖率：([\d.]+)%")
TEST_RWD_RE = re.compile(r"全局平均奖励：([\d.eE+-]+)")
TEST_SR_RE = re.compile(r"成功率[（(]?[^）)]*[）)]?：([\d.]+)%")
TEST_STEPS_RE = re.compile(r"平均完成步数：([\d.]+)")
PAUSE_CNT_RE = re.compile(r"全局总暂停次数：(\d+)")
PAUSE_STEPS_RE = re.compile(r"全局总暂停步数：(\d+)")
# 资源
START_RE = re.compile(r"Started at\s+(.+)")
END_RE = re.compile(r"Results reported at\s+(.+)")
CPU_TIME_RE = re.compile(r"CPU time:\s*([\d.]+)\s*sec")
MEM_RE = re.compile(r"Max Memory:\s*([\d.]+)\s*(\w+)")
# 环境快照字段
ENV_GRID_RE = re.compile(r"网格尺寸\s*\w*\s*:\s*(\d+)")
ENV_AGENT_RE = re.compile(r"智能体数量\s*\w*\s*:\s*(\d+)")
ENV_START_RE = re.compile(r"智能体起始位置\s*:\s*(\[\[.*?\]\])")
ENV_TARGET_RE = re.compile(r"目标点位置\s*:\s*(\[\[.*?\]\])")
ENV_OBST_RE = re.compile(r"固定障碍物坐标\s*:\s*(\[\[.*?\]\])")
ENV_HEADING_RE = re.compile(r"初始朝向\s*\(?\w*\)?\s*:\s*(\[.*?\])")
ENV_THRESH_RE = re.compile(r"覆盖率达成阈值\s*:\s*([\d.]+)")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt_time(s):
    """把 '2026-09-10 10:50:43' 解析成 datetime，失败返回 None。"""
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _stat(seq):
    """对一个 float 序列返回 (n, mean, min, max, head_mean, tail_mean)。head/tail 各取 20%。"""
    if not seq:
        return (0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    k = max(1, len(seq) // 5)
    head = seq[:k]
    tail = seq[-k:]
    return (len(seq), sum(seq) / len(seq), min(seq), max(seq),
            sum(head) / len(head), sum(tail) / len(tail))


def analyze(path):
    """全量读入单个日志，返回结构化字典。"""
    if not os.path.exists(path):
        return {"path": path, "error": "文件不存在"}

    ep = []          # (ep, reward, coverage, steps, loss, curQ_or_None, td)
    train = []       # (ep, loss, curQ, tgtQ, eps, td, gru_norm_or_None)
    best = []        # (ep, reward, coverage)
    reward_comp = {}  # 分量名 -> 值（取最后一次区间均值，或累加求平均）
    reward_comp_count = 0
    device = None
    prob_sum = None
    episodes = None
    env = {}
    test = {}
    res_meta = {}

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            # EP 每回合
            e = EP_RE.search(line)
            if e:
                ep_, r, cov, steps, loss, curQ, td = e.groups()
                ep.append((int(ep_), _f(r), _f(cov), int(steps), _f(loss),
                           _f(curQ) if curQ is not None else None, _f(td)))
                continue
            # TRAIN
            t = TRAIN_CORE_RE.search(line)
            if t:
                ep_, loss, curQ, tgtQ, eps, td = t.groups()
                g = GRU_NORM_RE.search(line)
                train.append((int(ep_), _f(loss), _f(curQ), _f(tgtQ), _f(eps), _f(td),
                              _f(g.group(1)) if g else None))
                continue
            # BEST
            b = BEST_RE.search(line)
            if b:
                best.append((int(b.group(1)), _f(b.group(2)), _f(b.group(3))))
                continue
            # 奖励分量（累加后求平均，避免只取最后一次）
            rc = REWARD_COMP_RE.search(line)
            if rc:
                for name, val in re.findall(r"([\w.]+):([+-]?[\d.eE+-]+)", rc.group(1)):
                    reward_comp[name] = reward_comp.get(name, 0.0) + _f(val)
                reward_comp_count += 1
                continue
            # 概览
            if device is None:
                d = DEVICE_RE.search(line)
                if d:
                    device = d.group(1)
            if prob_sum is None:
                p = PROB_SUM_RE.search(line)
                if p:
                    prob_sum = _f(p.group(1))
            if episodes is None:
                n = EPISODES_RE.search(line)
                if n:
                    episodes = int(n.group(1))
            # 环境快照
            for key, rx in [("grid", ENV_GRID_RE), ("agent", ENV_AGENT_RE),
                            ("start", ENV_START_RE), ("target", ENV_TARGET_RE),
                            ("obstacle", ENV_OBST_RE), ("heading", ENV_HEADING_RE),
                            ("threshold", ENV_THRESH_RE)]:
                if key not in env:
                    m = rx.search(line)
                    if m:
                        env[key] = m.group(1)
            # 测试汇总
            if "coverage" not in test:
                m = TEST_COV_RE.search(line)
                if m:
                    test["coverage"] = _f(m.group(1))
            if "reward" not in test:
                m = TEST_RWD_RE.search(line)
                if m:
                    test["reward"] = _f(m.group(1))
            if "sr" not in test:
                m = TEST_SR_RE.search(line)
                if m:
                    test["sr"] = _f(m.group(1))
            if "steps" not in test:
                m = TEST_STEPS_RE.search(line)
                if m:
                    test["steps"] = _f(m.group(1))
            if "pause_cnt" not in test:
                m = PAUSE_CNT_RE.search(line)
                if m:
                    test["pause_cnt"] = int(m.group(1))
            if "pause_steps" not in test:
                m = PAUSE_STEPS_RE.search(line)
                if m:
                    test["pause_steps"] = int(m.group(1))
            # 资源
            if "start" not in res_meta:
                m = START_RE.search(line)
                if m:
                    res_meta["start"] = m.group(1).strip()
            if "end" not in res_meta:
                m = END_RE.search(line)
                if m:
                    res_meta["end"] = m.group(1).strip()
            if "cpu_time" not in res_meta:
                m = CPU_TIME_RE.search(line)
                if m:
                    res_meta["cpu_time"] = _f(m.group(1))
            if "mem" not in res_meta:
                m = MEM_RE.search(line)
                if m:
                    res_meta["mem"] = _f(m.group(1)), m.group(2)

    # 奖励分量求平均
    if reward_comp_count > 0:
        reward_comp = {k: v / reward_comp_count for k, v in reward_comp.items()}

    return {
        "path": path,
        "name": os.path.basename(path),
        "device": device, "prob_sum": prob_sum, "episodes": episodes,
        "env": env, "test": test, "meta": res_meta,
        "ep": ep, "train": train, "best": best, "reward_comp": reward_comp,
    }


# ────────────────────────── 诊断规则（通用，阈值可调） ──────────────────────────

def diagnose_training(res):
    """诊断训练问题，返回 [(问题, 方向)] 列表。"""
    issues = []
    ep = res["ep"]
    train = res["train"]
    test = res.get("test", {})

    if not ep:
        return issues

    r = [x[1] for x in ep]
    cov = [x[2] for x in ep]
    steps = [x[3] for x in ep]
    loss = [x[4] for x in ep if x[4] == x[4]]  # 去 NaN

    # 1) loss 爆炸 / 不降
    if train:
        tr_loss = [t[1] for t in train]
        head_loss = sum(tr_loss[:3]) / max(1, len(tr_loss[:3]))
        tail_loss = sum(tr_loss[-10:]) / max(1, len(tr_loss[-10:]))
        if max(tr_loss) > 100 * max(tail_loss, 1e-6):
            issues.append(("loss 早期爆炸（峰值是末段的 100 倍+），训练初期不稳定",
                           "检查初始学习率、reward scale、梯度裁剪；可先 warmup 或降低 lr"))
        elif tail_loss > 2 * head_loss:
            issues.append(("loss 不降反升（末段是开头的 2 倍+），可能未收敛或发散",
                           "降低学习率 / 检查目标网络更新频率 / 检查 reward 是否发散"))

    # 2) 覆盖率停滞
    if cov:
        k = max(1, len(cov) // 4)
        early = sum(cov[:k]) / len(cov[:k])
        late = sum(cov[-k:]) / len(cov[-k:])
        if max(cov) - late < 2.0 and max(cov) < 97:
            issues.append((f"覆盖率停滞在 ~{max(cov):.1f}%（后期均值 {late:.1f}%，未继续提升）",
                           "最后一段区域缺乏激励 → 检查里程碑奖励/稀缺性奖励是否触达；或探索策略太弱"))
        if late < early:
            issues.append((f"覆盖率后期倒挂（后期 {late:.1f}% < 前期 {early:.1f}%）",
                           "策略在后期退化或过拟合 → 检查 replay 分布偏移 / 目标网络 / 学习率衰减"))

    # 3) 奖励波动 / 不升
    if r:
        r_mean = sum(r) / len(r)
        r_head = sum(r[:max(1, len(r)//5)]) / max(1, len(r)//5)
        r_tail = sum(r[-max(1, len(r)//5):]) / max(1, len(r[-max(1, len(r)//5):]))
        if r_tail < r_head:
            issues.append((f"总奖励后期低于前期（尾 {r_tail:.0f} < 头 {r_head:.0f}）",
                           "奖励设计或策略退化 → 检查是否有负奖励主导 / 稀疏奖励"))

    # 4) Q 值发散
    if train:
        curQ = [t[2] for t in train]
        tgtQ = [t[3] for t in train]
        if curQ and tgtQ:
            gap_head = abs(curQ[0] - tgtQ[0])
            gap_tail = abs(curQ[-1] - tgtQ[-1])
            if max(curQ) > 5 * max(tgtQ):
                issues.append(("curQ 远超 tgtQ（Q 值高估/发散）",
                               "检查 bootstrapping 过高估计、target 网络更新是否过慢、reward 缩放"))

    # 5) 测试成功率低
    if test.get("sr") is not None and test["sr"] < 50:
        issues.append((f"测试成功率低（{test['sr']:.1f}%）",
                       "覆盖率目标设定可能过难，或最后阶段区域未被覆盖 → 见奖励分量诊断"))
    if test.get("coverage") is not None and test["coverage"] < 90:
        issues.append((f"测试覆盖率偏低（{test['coverage']:.1f}% < 90%）",
                       "覆盖率未达标 → 结合奖励分量看是哪类奖励缺失"))

    return issues


def diagnose_reward(res):
    """诊断奖励函数问题，返回 [(问题, 方向)] 列表。"""
    issues = []
    comp = res.get("reward_comp", {})
    if not comp:
        return [("日志未记录奖励分量，无法诊断奖励函数",
                 "在环境 step 内把奖励拆成带名字的分量并通过 info 回传（见方法论文档第 2 节）")]

    pos = {k: v for k, v in comp.items() if v > 0}
    neg = {k: v for k, v in comp.items() if v < 0}
    pos_sum = sum(pos.values())
    neg_sum = sum(neg.values())

    # 1) 里程碑全为 0
    milestones = {k: v for k, v in comp.items() if k.startswith("milestone")}
    if milestones and all(abs(v) < 1e-9 for v in milestones.values()):
        issues.append(("里程碑奖励 milestone_* 全部为 0（从未触发）",
                       "阈值可能设得覆盖率永远够不到，或里程碑奖励逻辑未真正发放 → 检查发放条件"))

    # 2) 负奖励主导
    if pos_sum > 0 and abs(neg_sum) > 0.5 * pos_sum:
        issues.append((f"负奖励占比过高（负和 {neg_sum:.3f} vs 正和 {pos_sum:.3f}）",
                       "惩罚项过重会抑制探索 → 检查 repeat/stagnation/boundary 惩罚幅度"))

    # 3) 停滞/重复惩罚过大
    for k in ("stagnation_penalty", "repeat_penalty"):
        if k in comp and abs(comp[k]) > 0.03:
            issues.append((f"{k} 绝对值偏大（{comp[k]:.3f}），智能体大量停滞/重复访问",
                           "可能探索不足或区域分配不合理 → 检查探索策略、区域划分、奖励引导"))

    # 4) 主导分量判断（正奖励里哪个占比最大，是否健康）
    if pos:
        top_name, top_val = max(pos.items(), key=lambda x: x[1])
        if pos_sum > 0 and top_val / pos_sum > 0.6:
            issues.append((f"正奖励高度集中在 {top_name}（占比 {top_val/pos_sum:.0%}）",
                           "奖励结构单一 → 可能导致策略只追逐单一目标，忽略其他行为"))

    # 5) 某分量缺失（正奖励里应有的项为 0）
    for k in ("new_cell_base", "new_cell_share"):
        if k in comp and abs(comp[k]) < 1e-9:
            issues.append((f"{k} 为 0（覆盖新格奖励未触达）",
                           "覆盖奖励可能被屏蔽 → 检查 is_new_cell 判断与发放位置"))

    return issues


# ────────────────────────── 输出 ──────────────────────────

def _pct(x):
    return "N/A" if x != x else f"{x:.1f}%"


def print_single(res):
    if "error" in res:
        print(f"[{res['name']}] {res['error']}")
        return

    ep, train, best = res["ep"], res["train"], res["best"]
    test, meta, env = res.get("test", {}), res.get("meta", {}), res.get("env", {})

    print("=" * 72)
    print(f"日志：{res['name']}")
    print("=" * 72)

    # 概览 + 资源
    wall = None
    if meta.get("start") and meta.get("end"):
        s, e = _fmt_time(meta["start"]), _fmt_time(meta["end"])
        if s and e:
            wall = (e - s).total_seconds()
    parts = []
    if res["device"]:
        parts.append(f"设备:{res['device']}")
    if res["episodes"]:
        parts.append(f"总轮次:{res['episodes']}")
    if res["prob_sum"] is not None:
        parts.append(f"理论最大奖励:{res['prob_sum']}")
    print(f"[概览] {' | '.join(parts)}")
    if wall is not None:
        h = wall / 3600
        cpu = meta.get("cpu_time")
        line = f"[资源] 墙钟 {h:.2f}h"
        if cpu is not None:
            line += f" | CPU 核时 {cpu/3600:.2f}h"
            # CPU 利用率 = CPU核时 / (墙钟 × 进程数)；进程数未知时仅给核时
        if meta.get("mem"):
            line += f" | 峰值内存 {meta['mem'][0]:.1f}{meta['mem'][1]}"
        print(line)
    else:
        print("[资源] 日志未记录起止时间")

    # 环境快照
    if env:
        print(f"[环境] {env.get('grid','?')}x{env.get('grid','?')} | {env.get('agent','?')}机 "
              f"| 阈值 {env.get('threshold','?')} | 朝向 {env.get('heading','?')}")
        print(f"       障碍物 {env.get('obstacle','?')}")

    # 每回合统计
    if ep:
        r_st = _stat([x[1] for x in ep])
        c_st = _stat([x[2] for x in ep])
        s_st = _stat([float(x[3]) for x in ep])
        print(f"[训练{len(ep)}回合] "
              f"奖励 均值{r_st[1]:.0f}(头{r_st[4]:.0f}/尾{r_st[5]:.0f}) "
              f"覆盖率 均值{c_st[1]:.1f}% 峰值{c_st[3]:.1f}%(头{c_st[4]:.1f}/尾{c_st[5]:.1f}) "
              f"步数 均值{s_st[1]:.0f}")

    # TRAIN 指标首尾
    if train:
        t0, t1 = train[0], train[-1]
        g0 = f"{t0[6]:.1f}" if t0[6] is not None else "-"
        g1 = f"{t1[6]:.1f}" if t1[6] is not None else "-"
        print(f"[指标] loss {t0[1]:.1f}→{t1[1]:.2f} | curQ {t0[2]:.1f}→{t1[2]:.1f} | "
              f"tgtQ {t0[3]:.1f}→{t1[3]:.1f} | eps {t0[4]:.3f}→{t1[4]:.3f} | td {t0[5]:.1f}→{t1[5]:.2f}"
              + (f" | gru_norm {g0}→{g1}" if t0[6] is not None else ""))

    # 覆盖率提升
    if best:
        print(f"[覆盖提升] {len(best)}次 | 峰值 {max(x[2] for x in best):.1f}%"
              f"@ep{max(best, key=lambda x: x[2])[0]}")

    # 奖励分量（取绝对值 top5）
    comp = res.get("reward_comp", {})
    if comp:
        top = sorted(comp.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        print("[奖励分量 top|绝对|] " + " ".join(f"{k}:{v:+.3f}" for k, v in top))

    # 测试汇总
    if test:
        t = [f"覆盖率 {_pct(test.get('coverage'))}"]
        if test.get("reward") is not None:
            t.append(f"奖励 {test['reward']:.1f}")
        if test.get("sr") is not None:
            t.append(f"成功率 {test['sr']:.1f}%")
        if test.get("steps") is not None:
            t.append(f"步数 {test['steps']:.0f}")
        if test.get("pause_cnt") is not None:
            t.append(f"暂停 {test['pause_cnt']}次/{test.get('pause_steps', 0)}步")
        print("[测试] " + " | ".join(t))

    # 诊断
    tr_issues = diagnose_training(res)
    rw_issues = diagnose_reward(res)
    if tr_issues or rw_issues:
        print("-" * 72)
        print("[诊断]")
        for label, issues in [("训练", tr_issues), ("奖励函数", rw_issues)]:
            for prob, fix in issues:
                print(f"  ⚠️ [{label}] {prob}")
                print(f"        → {fix}")
    else:
        print("[诊断] 未发现明显异常")
    print()


def print_compare(results):
    """多文件对比 + 差异摘要。"""
    print("=" * 72)
    print("多日志对比表")
    print("=" * 72)
    hdr = f"{'日志':<28}{'设备':<6}{'轮次':<7}{'覆盖率峰值':<11}{'测试覆盖':<10}{'成功率':<8}{'墙钟':<9}{'内存'}"
    print(hdr)
    for res in results:
        if "error" in res:
            print(f"{res['name']:<28} {res['error']}")
            continue
        ep = res["ep"]
        cov_peak = max((x[2] for x in ep), default=float("nan"))
        test = res.get("test", {})
        meta = res.get("meta", {})
        wall = ""
        if meta.get("start") and meta.get("end"):
            s, e = _fmt_time(meta["start"]), _fmt_time(meta["end"])
            if s and e:
                wall = f"{(e-s).total_seconds()/3600:.2f}h"
        mem = f"{meta['mem'][0]:.1f}{meta['mem'][1]}" if meta.get("mem") else "-"
        print(f"{res['name']:<28}{res['device'] or '?':<6}{res['episodes'] or '?':<7}"
              f"{_pct(cov_peak):<11}{_pct(test.get('coverage')):<10}"
              f"{_pct(test.get('sr')):<8}{wall:<9}{mem}")

    # 差异摘要
    print()
    print("[关键差异]")
    ok = [r for r in results if "error" not in r]
    if len(ok) >= 2:
        for res in ok:
            ep = res["ep"]
            cov_peak = max((x[2] for x in ep), default=float("nan"))
            test = res.get("test", {})
            tr_issues = diagnose_training(res)
            rw_issues = diagnose_reward(res)
            print(f"  • {res['name']}: 覆盖峰值 {_pct(cov_peak)} / 测试覆盖 {_pct(test.get('coverage'))} "
                  f"/ 成功率 {_pct(test.get('sr'))}")
            for prob, _ in tr_issues[:2]:
                print(f"      - 训练问题: {prob}")
            for prob, _ in rw_issues[:2]:
                print(f"      - 奖励问题: {prob}")
    print()


def main():
    ap = argparse.ArgumentParser(description="训练日志多维诊断脚本（通用版）")
    ap.add_argument("paths", nargs="+", help="一个或多个日志文件路径")
    args = ap.parse_args()

    results = [analyze(p) for p in args.paths]

    if len(results) == 1:
        print_single(results[0])
    else:
        for res in results:
            print_single(res)
        print_compare(results)


if __name__ == "__main__":
    main()
