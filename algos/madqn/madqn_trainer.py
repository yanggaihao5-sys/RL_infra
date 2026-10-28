# -*- coding: utf-8 -*-
# 🕒 v2 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：读 COMMON_*（行为不变） ｜ 上一版：v1 · 2026-10-22 17:07

"""训练设施（trainer 角色）：后台 rollout 线程 + 训练主循环 + 评估。

并发结构（与 UAV 项目同构，但去掉框架依赖）：

```
┌── RolloutActorThread（后台线程）──┐        ┌── 主线程（learner）──┐
│ 环境池 step → 写回放池            │        │ 从回放池采样 → learn │
│ 每回合结束 → stats 队列           │ ─────► │ 打日志 / 存模型 / 调 ε│
│ 按需同步 rollout 权重             │ ◄───── │ request_sync()       │
└──────────────────────────────────┘        └──────────────────────┘
```

**为什么拆线程**：环境 step（纯 numpy/Python）与梯度更新（torch）天然是两种负载，
分开后 CPU 不空转；且 rollout 线程用**独立权重副本**，与 learner 只做单向同步，
不存在「边学边采样」的隐性竞争。权重拷贝用同一把锁保护。

整条链路可整体旁路：把 `RolloutActorThread` 换成同步 for 循环即可（接口就 3 个方法）。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rlinfra.logger import TrainLogger
from rlinfra.metrics import ResourceMeter
from rlinfra.paths import RunPaths

from .madqn_agent import MADQNCTDEAgent
from .madqn_envs import MADroneSearchEnv
from .madqn_parallel_env import make_pool
from .madqn_replay_buffer import DroneSearchReplayBuffer


# ════════════════════════════════════════════════════════════════
# 后台 rollout 线程
# ════════════════════════════════════════════════════════════════
class RolloutActorThread(threading.Thread):
    def __init__(self, pool, agent: MADQNCTDEAgent, buffer: DroneSearchReplayBuffer,
                 cfg, stats_q: "queue.Queue[Dict[str, Any]]", weight_lock: threading.Lock,
                 name: str = "rollout-actor"):
        super().__init__(name=name, daemon=True)
        self.pool = pool
        self.agent = agent
        self.buffer = buffer
        self.cfg = cfg
        self.stats_q = stats_q
        self.weight_lock = weight_lock
        self.stop_event = threading.Event()
        self._sync_request = threading.Event()
        self.episodes_done = 0
        self.env_steps = 0
        self.error: Optional[BaseException] = None

    # ── 外部接口（主线程调用）──
    def request_sync(self) -> None:
        """请 actor 下次迭代前把 online 权重同步到 rollout 副本。"""
        self._sync_request.set()

    def stop(self) -> None:
        self.stop_event.set()

    # ── 主体 ──
    def run(self) -> None:  # pragma: no cover - 线程体，由冒烟测试间接覆盖
        try:
            n = self.pool.num_envs
            obs = self.pool.reset()
            ep_reward = [0.0] * n
            ep_comps: List[Dict[str, float]] = [{} for _ in range(n)]
            ep_steps = [0] * n

            while not self.stop_event.is_set():
                if self._sync_request.is_set():
                    with self.weight_lock:
                        self.agent.sync_rollout()
                    self._sync_request.clear()

                flat = np.stack([o for per_env in obs for o in per_env])       # (n*agent_num, obs_dim)
                acts = self.agent.act(flat, net="rollout")
                per_env_acts = acts.reshape(n, self.cfg.agent_num)

                nobs, rewards, dones, infos = self.pool.step([per_env_acts[k] for k in range(n)])
                self.env_steps += n

                for k in range(n):
                    for i in range(self.cfg.agent_num):
                        self.buffer.push(obs[k][i], per_env_acts[k][i], rewards[k][i],
                                         nobs[k][i], bool(dones[k]))
                        ep_reward[k] += float(rewards[k][i])
                        for name, val in infos[k]["reward_components"][i].items():
                            ep_comps[k][name] = ep_comps[k].get(name, 0.0) + float(val)
                    ep_steps[k] += 1
                    if dones[k]:
                        self.stats_q.put(self._pack(infos[k], ep_reward[k], ep_steps[k],
                                                    ep_comps[k]))
                        ep_reward[k], ep_steps[k], ep_comps[k] = 0.0, 0, {}

                obs = nobs
                done_idx = [k for k in range(n) if dones[k]]
                if done_idx:
                    for k, o in zip(done_idx, self.pool.reset_envs(done_idx)):
                        obs[k] = o
        except BaseException as e:      # 线程内异常必须能被主线程看见，不许静默死掉
            self.error = e
            self.stats_q.put({"_error": repr(e)})

    def _pack(self, info: Dict[str, Any], reward: float, steps: int,
              comps: Dict[str, float]) -> Dict[str, Any]:
        self.episodes_done += 1
        denom = max(1, self.cfg.agent_num * steps)
        return {
            "ep": self.episodes_done,
            "reward": reward / self.cfg.agent_num,
            "coverage": float(info["coverage"]) * 100.0,
            "steps": int(steps),
            "success": bool(info["success"]),
            "components": {k: v / denom for k, v in comps.items()},
        }


# ════════════════════════════════════════════════════════════════
# 训练主循环
# ════════════════════════════════════════════════════════════════
def theoretical_max_reward(cfg, free_cells: int, n_targets: int) -> float:
    """理论最大单回合奖励（概览行要用）。用来判断「学到的奖励离上限还差多远」。"""
    base = free_cells * cfg.R_new_cell + n_targets * cfg.R_target_bonus
    milestone = cfg.R_milestone_50 + cfg.R_milestone_75 + cfg.R_milestone_threshold
    return base + milestone


class GreedyProbe:
    """固定地图上的**贪心评估**：用来选最优模型。

    🔴 为什么必须单独做：训练回合的覆盖率来自 **ε-greedy + 滞后的 rollout 权重**，
    会**系统性偏高** —— 随机探索本身就能多覆盖几格。直接拿它选 best，会选出「运气好」的模型。
    本项目实测（300 回合 / 20×20 / 2 机）：训练回合 best = **90.5%**，
    同一权重贪心评估只有 **53.9%** —— 差 36 个点，全是探索红利。
    选 best 的判据必须是「**部署时会用的那个策略**」的表现，不是训练时的表现。
    """

    def __init__(self, cfg, agent: MADQNCTDEAgent, seed: Optional[int] = None):
        self.cfg = cfg
        self.agent = agent
        self.env = MADroneSearchEnv(cfg, seed=cfg.seed if seed is None else seed,
                                    max_steps=cfg.final_test_steps)

    def run(self) -> Dict[str, Any]:
        obs = self.env.reset()
        total_r, steps = 0.0, 0
        while True:
            acts = self.agent.act(np.stack(obs), net="online", greedy=True)
            obs, rewards, done, info = self.env.step(acts)
            total_r += float(np.sum(rewards)); steps += 1
            if done:
                break
        return {"coverage": info["coverage"] * 100.0, "success": bool(info["success"]),
                "steps": steps, "reward": total_r / self.cfg.agent_num}


def probe_key(cfg, pr: Dict[str, Any]) -> tuple:
    """把一次贪心评估压成**可比较的元组**（字典序，越大越好）。

    覆盖域的判据就是 1 维覆盖率，与修复前 `pr["coverage"] > best_cov` **逐字等价**。
    （2026-10-17 #10：把「选优判据」从写死的比较里抽出来，让不同域各自定义 ——
      追逃域用 `capture_steps`，见 `algos/pursuit/pursuit_trainer.py`。）
    """
    metric = getattr(cfg, "MADQN_best_metric", "coverage")
    if metric == "coverage":
        return (float(pr["coverage"]),)
    raise ValueError(f"未知 MADQN_best_metric={metric!r}；覆盖域只支持 coverage")


def _prune(cfg, paths: RunPaths, logger: TrainLogger) -> None:
    """清理旧的 final 模型（只保留最近 cfg.save_keep_last 个），删了什么要如实报。"""
    removed = paths.prune_final_models(cfg.test_mode, cfg.save_keep_last)
    if removed:
        names = [os.path.basename(p) for p in removed]
        logger.info(f"清理旧 final 模型 {len(removed)} 个（只保留最近 {cfg.save_keep_last} 个）：{names}")


def epsilon_at(cfg, ep: int) -> float:
    """两段式探索：前 random_episodes 纯随机铺回放池，之后线性衰减。"""
    if ep <= cfg.COMMON_random_episodes:
        return cfg.COMMON_eps_start
    frac = min(1.0, (ep - cfg.COMMON_random_episodes) / max(1, cfg.eps_decay_episodes))
    return cfg.COMMON_eps_start + frac * (cfg.COMMON_eps_end - cfg.COMMON_eps_start)


def resolve_device(cfg) -> str:
    if cfg.device and cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_madqn_ctde(cfg, paths: RunPaths, logger: TrainLogger) -> Dict[str, Any]:
    """MADQN 链路训练主循环。返回训练摘要 dict（供 main 打印/自检）。"""
    device = resolve_device(cfg)
    if cfg.deterministic:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    meter = ResourceMeter().start()

    os.makedirs(paths.root, exist_ok=True)
    pool = make_pool(cfg, MADroneSearchEnv, n=cfg.env_count, seed=cfg.seed)
    agent = MADQNCTDEAgent(cfg, cfg.obs_dim, cfg.n_actions, device=device)
    buffer = DroneSearchReplayBuffer(cfg.COMMON_buffer_size, cfg.obs_dim, seed=cfg.seed)

    # ── 快照 / 身份 / 概览（三层日志的第①②层）──
    logger.identity(cfg.test_mode, f"{MADroneSearchEnv.__name__} | 地图 {cfg.grid_size}×{cfg.grid_size} "
                                   f"| 动作 {cfg.n_actions} 向 | 设备 {device} | 并行 {cfg.parallel}")
    logger.env_banner(cfg.test_mode.upper(), pool.snapshot())
    n_targets = int(len(pool.envs[0].targets)) if hasattr(pool, "envs") else cfg.target_num
    free_cells = int(pool.envs[0].free_cells) if hasattr(pool, "envs") else \
        cfg.grid_size * cfg.grid_size - cfg.obstacle_num
    logger.derived(cfg.derived_summary())
    logger.overview(device, theoretical_max_reward(cfg, free_cells, n_targets),
                    cfg.env_count, cfg.COMMON_buffer_size, cfg.episodes)
    logger.info(f"网络参数量：{agent.n_params()}（共享参数，所有智能体同一套权重）")

    # ── 起 actor 线程 ──
    stats_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=50)
    weight_lock = threading.Lock()
    actor = RolloutActorThread(pool, agent, buffer, cfg, stats_q, weight_lock)
    actor.start()
    _probe = GreedyProbe(cfg, agent)     # 固定地图贪心评估器（选 best 的判据）

    best_key = None                     # 选优键（元组，字典序）；判据由 MADQN_best_metric 决定
    best_metric = getattr(cfg, "MADQN_best_metric", "coverage")
    best_cov = -1.0                     # 只用于日志/摘要的"最优覆盖"，**不**参与比较
    best_ep = 0
    comp_sum: Dict[str, float] = {}
    comp_n = 0
    win_rewards: List[float] = []
    last_metrics = {"loss": 0.0, "cur_q": 0.0, "tgt_q": 0.0, "td": 0.0}
    t0 = time.perf_counter()

    try:
        while True:
            stat = stats_q.get()
            if "_error" in stat:
                raise RuntimeError(f"rollout 线程崩溃：{stat['_error']}")
            ep = int(stat["ep"])
            if ep > cfg.episodes:
                break

            # ① 学习
            eps = epsilon_at(cfg, ep)
            agent.set_epsilon(eps)
            loss_acc = 0.0
            n_learn = 0
            if buffer.ready(cfg.COMMON_min_buffer):
                for _ in range(cfg.COMMON_learn_per_episode):
                    batch = buffer.sample(cfg.COMMON_batch_size)
                    with weight_lock:
                        last_metrics = agent.learn(batch)
                    loss_acc += last_metrics["loss"]
                    n_learn += 1
                actor.request_sync()
            avg_loss = loss_acc / n_learn if n_learn else 0.0

            if n_learn == 0:
                last_metrics = {"loss": 0.0, "cur_q": 0.0, "tgt_q": 0.0, "td": 0.0}

            # ② 每回合汇总行
            logger.ep_line(ep, stat["reward"], stat["coverage"], stat["steps"],
                           avg_loss, last_metrics["cur_q"], last_metrics["td"])

            # ③ 周期汇总行（训练指标 + 奖励分量）
            win_rewards.append(stat["reward"])
            for k, v in stat["components"].items():
                comp_sum[k] = comp_sum.get(k, 0.0) + v
            comp_n += 1
            if ep % cfg.print_interval == 0:
                logger.train_line(ep, avg_loss, last_metrics["cur_q"], last_metrics["tgt_q"],
                                  eps, last_metrics["td"],
                                  extra=f"buf:{len(buffer)} eps:{eps:.3f} "
                                        f"Rbar:{np.mean(win_rewards[-cfg.print_interval:]):.2f}")
                if comp_n:
                    logger.reward_line({k: v / comp_n for k, v in comp_sum.items()})
                    comp_sum, comp_n = {}, 0

            # ④ 最优模型（覆盖保存）+ 周期模型（递增不覆盖）
            #    判据用 **贪心评估**（部署时的策略），且判据本身由 MADQN_best_metric 决定（修复 #10）
            if cfg.COMMON_eval_interval > 0 and ep % cfg.COMMON_eval_interval == 0 \
                    and buffer.ready(cfg.COMMON_min_buffer):
                pr = _probe.run()
                gap = stat["coverage"] - pr["coverage"]
                logger.info(f"贪心评估 ep{ep}：覆盖率 {pr['coverage']:.2f}% | 步数 {pr['steps']} | "
                            f"成功 {pr['success']} | 训练回合覆盖 {stat['coverage']:.2f}%"
                            f"（探索红利 {gap:+.1f} 点）")
                key = probe_key(cfg, pr)
                if best_key is None or key > best_key:
                    best_key, best_cov, best_ep = key, pr["coverage"], ep
                    agent.save(paths.best_model(cfg.test_mode),
                               extra={"ep": ep, "greedy_coverage": best_cov,
                                      "best_metric": best_metric, "best_key": list(key),
                                      "train_coverage": stat["coverage"], "cfg": cfg.to_dict()})
                    logger.best_line(ep, pr["reward"], pr["coverage"],
                                     paths.best_model(cfg.test_mode))
            elif cfg.COMMON_eval_interval <= 0 and buffer.ready(cfg.COMMON_min_buffer):
                key = probe_key(cfg, {"coverage": stat["coverage"],
                                      "success": stat["success"], "steps": stat["steps"]})
                if best_key is None or key > best_key:
                    best_key, best_cov, best_ep = key, stat["coverage"], ep
                    agent.save(paths.best_model(cfg.test_mode),
                               extra={"ep": ep, "coverage": best_cov,
                                      "best_metric": best_metric, "best_key": list(key),
                                      "cfg": cfg.to_dict()})
                    logger.best_line(ep, stat["reward"], stat["coverage"],
                                     paths.best_model(cfg.test_mode))

            if ep % cfg.save_interval == 0:
                p = paths.next_final_model(cfg.test_mode)
                agent.save(p, extra={"ep": ep, "coverage": stat["coverage"]})
                logger.saved("final", p)
                _prune(cfg, paths, logger)

            if ep >= cfg.episodes:
                break

            # 线程静默死亡保护：actor 报错要能立刻冒出来
            if actor.error is not None:
                raise RuntimeError(f"rollout 线程异常：{actor.error!r}")
    finally:
        actor.stop()
        actor.join(timeout=10)
        try:
            if best_key is None:
                # 回合数 < eval_interval → 探针一次都没跑过。收尾**必须补一次**，
                # 否则「训练完了却没有 best 模型」，而日志里只写个 -1.00% 让人摸不着头脑。
                pr = _probe.run()
                best_key, best_cov, best_ep = probe_key(cfg, pr), pr["coverage"], cfg.episodes
                agent.save(paths.best_model(cfg.test_mode),
                           extra={"ep": best_ep, "greedy_coverage": best_cov,
                                  "best_metric": best_metric, "best_key": list(best_key),
                                  "note": "收尾补评估（回合数不足一个 eval_interval）"})
                logger.best_line(best_ep, pr["reward"], pr["coverage"], paths.best_model(cfg.test_mode))
            p = paths.next_final_model(cfg.test_mode)
            agent.save(p, extra={"ep": cfg.episodes, "coverage": best_cov})
            logger.saved("final", p)
            _prune(cfg, paths, logger)
            buffer.save(os.path.join(paths.root, f"{cfg.test_mode}_final_buffer.npz"))
        except Exception as e:      # 收尾失败不该吞掉训练结果，但要如实说
            logger.warn(f"收尾保存失败：{e!r}")
        pool.close()

    meter.stop()
    start_s, end_s = meter.stamps()
    logger.resources(start_s, end_s, meter.cpu_sec, meter.mem_mb)
    logger.info(f"训练结束：{cfg.episodes} 回合 / 墙钟 {meter.wall_sec:.1f}s / "
                f"最优覆盖 {best_cov:.2f}%（第 {best_ep} 回合，选优判据 {best_metric}）/ "
                f"回放池 {len(buffer)} 条")

    return {"best_coverage": best_cov, "best_ep": best_ep, "best_metric": best_metric,
            "best_key": list(best_key) if best_key is not None else None,
            "wall_sec": meter.wall_sec,
            "buffer_size": len(buffer), "device": device, "paths": paths.root}


# ════════════════════════════════════════════════════════════════
# 评估
# ════════════════════════════════════════════════════════════════
def evaluate_madqn(cfg, paths: RunPaths, logger: TrainLogger, runs: Optional[int] = None,
                   model_path: Optional[str] = None) -> Dict[str, Any]:
    """载入最优模型，贪心策略跑 runs 个回合，打测试汇总行 + 存路径图。

    环境是**确定性设计**。默认 `eval_random_maps=True`（每轮换一张地图 ⇒ 有真实方差）；
    要**跨实验严格对齐环境**时用 `--eval-fixed-maps` —— 此时固定地图 + 贪心策略会让
    多轮结果**逐字相同**，这是预期行为，不是 bug（有效 n=1，算不出方差）。
    🔴 `eval_coverage_threshold`（默认 0.75）与训练的 `coverage_threshold`（0.90）**解耦**：
    只覆盖评测阶段的成功判据与提前终止，**不动训练动力学**，所以历史 run 仍可比。
    """
    device = resolve_device(cfg)
    runs = int(runs if runs is not None else cfg.eval_runs)
    model_path = model_path or paths.best_model(cfg.test_mode)
    if not os.path.exists(model_path):
        logger.warn(f"找不到模型 {model_path}，跳过评估（先跑 train）")
        return {"skipped": True}

    agent = MADQNCTDEAgent(cfg, cfg.obs_dim, cfg.n_actions, device=device)
    agent.load(model_path, map_location=device)
    logger.identity(cfg.test_mode, f"评估模式 | 模型 {os.path.basename(model_path)} | 设备 {device} "
                                   f"| 地图 {'每轮随机' if cfg.eval_random_maps else f'固定 seed={cfg.seed}'}")

    covs, rewards, succ, steps_list, paths_all, env0 = [], [], [], [], [], None
    for r in range(runs):
        env = MADroneSearchEnv(cfg, seed=cfg.seed + r if cfg.eval_random_maps else cfg.seed,
                               max_steps=cfg.final_test_steps)
        # 评测成功判据与训练阈值解耦（只在评测期生效，不改训练动力学）
        env.threshold = float(cfg.eval_coverage_threshold)
        if env0 is None:
            env0 = env
            logger.env_banner(cfg.test_mode.upper(), env.snapshot())
        obs = env.reset()
        ep_r, step = 0.0, 0
        traj = [[] for _ in range(cfg.agent_num)]
        while True:
            acts = agent.act(np.stack(obs), net="online", greedy=True)
            for i in range(cfg.agent_num):
                traj[i].append(list(map(int, env.pos[i])))
            obs, rs, done, info = env.step(acts)
            ep_r += float(np.sum(rs)); step += 1
            if done:
                break
        covs.append(info["coverage"] * 100.0); rewards.append(ep_r / cfg.agent_num)
        succ.append(1.0 if info["success"] else 0.0); steps_list.append(step)
        paths_all.append(traj)
        logger.info(f"run {r + 1}/{runs}：覆盖率 {info['coverage']*100:.2f}% | "
                    f"步数 {step} | 成功 {info['success']}")

    if not cfg.eval_random_maps and len(set(covs)) == 1 and runs > 1:
        logger.info(f"{runs} 次跑结果完全相同 —— 确定性环境 + 贪心策略的**预期行为**"
                    f"（对比实验环境对齐的代价）；要测泛化请加 --eval-random-maps")

    # 🔴 新增字段名刻意避开「覆盖率：」—— `scripts/eval_runs.sh` 用**贪婪** sed 抓
    #    `.*覆盖率：([0-9.]*)%`，若这里再出现一次「覆盖率：」会把标准差误抓成均值。
    # 🔴 成功判据是**链私有**的（修复 #9）：覆盖域 = 覆盖率 ≥ 阈值；追逃域 = 捕获。
    #    写进日志与 eval.json，避免"两个域共用一行元数据"的误导。
    success_rule = f"覆盖率 ≥ {cfg.eval_coverage_threshold:.2f}"
    map_note = "每轮随机" if cfg.eval_random_maps else f"固定 seed={cfg.seed}"
    logger.test_line(float(np.mean(covs)), float(np.mean(rewards)),
                     float(np.mean(succ)) * 100.0, float(np.mean(steps_list)),
                     extra=f"测试轮数：{runs} | 覆盖标准差 {float(np.std(covs)):.2f}pp"
                           f" | 地图 {map_note} | 成功判据 {success_rule}")
    _dump_test_artifacts(cfg, paths, env0, covs, steps_list, paths_all, runs)
    result = {"coverage": float(np.mean(covs)), "coverage_std": float(np.std(covs)),
              "success_rate": float(np.mean(succ)) * 100.0,
              "steps": float(np.mean(steps_list)), "runs": runs,
              "maps": runs, "eval_random_maps": bool(cfg.eval_random_maps),
              "success_rule": success_rule,
              "eval_coverage_threshold": float(cfg.eval_coverage_threshold),
              "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    # 本地 `--mode test` 也落一份 eval.json（修复 #9：此前只有高算 scripts/eval_runs.sh 写）
    with open(os.path.join(paths.root, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    return result


def _dump_test_artifacts(cfg, paths: RunPaths, env, covs, steps_list, paths_all, runs) -> None:
    """测试路径图 + JSON。**文件名内嵌关键指标**（步数/覆盖率/run 号），看名知内容。"""
    out_dir = paths.test_plot_dir("en")
    for r in range(runs):
        tag = f"t_s{int(steps_list[r])}_avg{covs[r]:.1f}%_r{r}_multi_agent_en"
        with open(os.path.join(out_dir, tag + ".json"), "w", encoding="utf-8") as f:
            json.dump({"run": r, "steps": int(steps_list[r]), "coverage": covs[r],
                       "paths": paths_all[r]}, f, ensure_ascii=False, indent=2)
        try:
            _plot_one_path(cfg, env, paths_all[r], out_dir, tag, covs[r], steps_list[r])
        except Exception as e:
            print(f"⚠️ 路径图绘制失败（不影响训练）：{e!r}")


def _plot_one_path(cfg, env, traj, out_dir: str, tag: str, cov: float, steps: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.zeros((env.H, env.W))
    for r, c in env.obstacles:
        grid[r, c] = 1
    for r, c in env.targets:
        grid[r, c] = 2
    ax.imshow(grid, cmap=ListedColormap(["white", "black", "gold"]), origin="upper")
    for i, p in enumerate(traj):
        arr = np.asarray(p)
        ax.plot(arr[:, 1], arr[:, 0], "-", linewidth=1.2, label=f"agent {i + 1}")
        ax.plot(arr[0, 1], arr[0, 0], "o", markersize=5)
    ax.set_title(f"coverage {cov:.1f}% | steps {steps}")
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, tag + ".png"), dpi=110)
    plt.close(fig)
