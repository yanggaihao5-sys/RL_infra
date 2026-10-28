# -*- coding: utf-8 -*-
# 🕒 v2 · 2026-10-22 19:11 · dsh-agent ｜ 变更：#8：读 COMMON_* ｜ 上一版：v1 · 2026-10-17 16:16

"""冒烟自检：不训练出结果，只证明「这套骨架没坏」。

检查项（每项独立、失败即报，退出码非 0）：
  1. 配置：校验通过、派生量与环境一致
  2. 环境：reset/step 维度对、同种子可复现
  3. 网络：前向 shape、参数量
  4. 回放池：写入/采样/存盘回读
  5. 智能体：act 形状、learn 返回有限值、save/load 权重一致
  6. 环境池：多环境 reset/step（单进程降级路径）
  7. **日志契约**：真跑 TrainLogger 出一份日志，再用 analyze_training_log.py 解析它
     —— 这是最重要的一项：格式改了但脚本没跟上，只有这里能发现
  8. 端到端：3 回合训练出模型文件

用法：
    set PYTHONIOENCODING=utf-8 && python utils/smoke_test.py
    python utils/smoke_test.py --mp      # 额外验证多进程环境池（受限沙箱可能失败）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "utils"))

import numpy as np  # noqa: E402

from config import Config  # noqa: E402
from rlinfra.logger import TrainLogger  # noqa: E402
from rlinfra.paths import RunPaths  # noqa: E402

PASS, FAIL = "✅ PASS", "❌ FAIL"
_failures = []


def _tmpdir() -> str:
    """项目内临时目录。

    🔴 Windows + DSH 沙箱（2026-09-30 / 2026-10-17 两次实测）：`tempfile.mkdtemp()`
    建出的目录**对其自身进程拒写**（`PermissionError: [Errno 13]` / `WinError 5`），
    且该目录随后也删不掉 —— 会让 `TemporaryDirectory()` 在清理阶段抛异常、掩盖真实结论。
    ⇒ 一律 `os.makedirs` 自建目录（不调 tempfile），唯一性用 pid + 纳秒时间戳保证。
    """
    root = os.path.join(ROOT, ".tmp")
    os.makedirs(root, exist_ok=True)
    d = os.path.join(root, f"smoke_{os.getpid()}_{time.time_ns()}")
    os.makedirs(d, exist_ok=False)
    return d


def _rmtmp(d: str) -> None:
    """清理临时目录：沙箱可能拒绝删除，忽略错误（不能因此判失败）。"""
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"{PASS} | {name}" + (f" | {detail}" if detail else ""))
    except Exception as e:
        _failures.append(name)
        print(f"{FAIL} | {name} | {e!r}")
        traceback.print_exc(limit=3)


# ══════════════════════ 各项检查 ══════════════════════
def t_config():
    cfg = Config()
    cfg.validate()
    side = 2 * cfg.COMMON_patch + 1
    assert cfg.obs_dim == side * side * 2 + 4, "派生 obs_dim 公式对不上"
    assert cfg.eps_decay_episodes > 0
    bad = 0
    for episodes, random_ep in [(5, 5)]:
        try:
            c = Config(episodes=episodes, COMMON_random_episodes=random_ep)
            c.validate()
            bad += 1
        except AssertionError:
            pass
    assert bad == 0, "校验没拦住「预热回合 = 总轮次」这种致命值"
    return f"obs_dim={cfg.obs_dim} n_actions={cfg.n_actions}"


def t_env():
    from algos.madqn.madqn_envs import MADroneSearchEnv
    cfg = Config()
    env = MADroneSearchEnv(cfg, seed=1)
    obs = env.reset()
    assert len(obs) == cfg.agent_num, "reset 返回的观测数 ≠ agent_num"
    assert obs[0].shape == (cfg.obs_dim,), f"观测维度 {obs[0].shape} ≠ {cfg.obs_dim}"
    nobs, rew, done, info = env.step(np.zeros(cfg.agent_num, dtype=np.int64))
    assert len(rew) == cfg.agent_num and len(nobs) == cfg.agent_num
    assert set(info["reward_components"][0].keys()) <= {
        "new_cell_base", "repeat_penalty", "obstacle_penalty", "collision_penalty",
        "step_penalty", "target_bonus", "stagnation_penalty",
        "milestone_50", "milestone_75", "milestone_90"}
    # 可复现性：同种子两个环境走同样动作 → 状态完全一致
    e1, e2 = MADroneSearchEnv(cfg, seed=7), MADroneSearchEnv(cfg, seed=7)
    o1, o2 = e1.reset(), e2.reset()
    acts = np.arange(cfg.agent_num) % cfg.n_actions
    for _ in range(20):
        o1, _, _, _ = e1.step(acts)
        o2, _, _, _ = e2.step(acts)
    assert np.allclose(np.stack(o1), np.stack(o2)), "同种子两环境观测不一致（环境有隐式随机性！）"
    assert abs(e1.coverage() - e2.coverage()) < 1e-12
    return f"steps=20 cov={e1.coverage()*100:.1f}% 同种子一致"


def t_network():
    from algos.madqn.madqn_networks import MADQNQNetwork
    cfg = Config()
    net = MADQNQNetwork(cfg.obs_dim, cfg.n_actions, cfg.COMMON_hidden_dim)
    import torch
    out = net(torch.zeros(4, cfg.obs_dim))
    assert out.shape == (4, cfg.n_actions), f"输出 shape {out.shape} 不对"
    return f"params={net.n_params()}"


def t_buffer():
    from algos.madqn.madqn_replay_buffer import DroneSearchReplayBuffer
    cfg = Config(COMMON_buffer_size=64)
    buf = DroneSearchReplayBuffer(cfg.COMMON_buffer_size, cfg.obs_dim, seed=0)
    for i in range(100):     # 超过容量 → 验证环形覆盖
        buf.push(np.full(cfg.obs_dim, i, dtype=np.float32), i % cfg.n_actions, float(i),
                 np.zeros(cfg.obs_dim, dtype=np.float32), i % 3 == 0)
    assert len(buf) == 64, f"容量封顶失效：{len(buf)}"
    b = buf.sample(16)
    assert b["obs"].shape == (16, cfg.obs_dim) and b["actions"].shape == (16,)
    d = _tmpdir()
    try:
        p = os.path.join(d, "buf.npz")
        buf.save(p)
        buf2 = DroneSearchReplayBuffer.load(p)
        assert buf2.size == buf.size and np.allclose(buf2.obs[-3:], buf.obs[-3:])
    finally:
        _rmtmp(d)
    return "环形覆盖 + 存盘回读 OK"


def t_agent():
    from algos.madqn.madqn_agent import MADQNCTDEAgent
    from algos.madqn.madqn_replay_buffer import DroneSearchReplayBuffer
    cfg = Config()
    agent = MADQNCTDEAgent(cfg, cfg.obs_dim, cfg.n_actions, device="cpu")
    obs = np.random.rand(cfg.agent_num, cfg.obs_dim).astype(np.float32)
    a = agent.act(obs, net="rollout")
    assert a.shape == (cfg.agent_num,) and a.min() >= 0 and a.max() < cfg.n_actions
    buf = DroneSearchReplayBuffer(256, cfg.obs_dim)
    for _ in range(128):
        buf.push(np.random.rand(cfg.obs_dim).astype(np.float32), 0, 1.0,
                 np.random.rand(cfg.obs_dim).astype(np.float32), False)
    m = agent.learn(buf.sample(32))
    assert all(np.isfinite(list(m.values()))), f"学习指标非有限：{m}"
    d = _tmpdir()
    try:
        p = os.path.join(d, "m.pth")
        agent.save(p)
        agent2 = MADQNCTDEAgent(cfg, cfg.obs_dim, cfg.n_actions, device="cpu")
        agent2.load(p, map_location="cpu")
        for (n1, p1), (n2, p2) in zip(agent.online.state_dict().items(),
                                     agent2.online.state_dict().items()):
            assert n1 == n2 and np.allclose(p1.numpy(), p2.numpy()), f"权重 {n1} 载入后不一致"
    finally:
        _rmtmp(d)
    return f"{m['loss']:.4f} loss, save/load 一致"


def t_pool():
    from algos.madqn.madqn_envs import MADroneSearchEnv
    from algos.madqn.madqn_parallel_env import SyncEnvPool
    cfg = Config(env_count=2)
    pool = SyncEnvPool(cfg, MADroneSearchEnv, n=2, seed=0)
    obs = pool.reset()
    assert len(obs) == 2 and len(obs[0]) == cfg.agent_num
    acts = [np.zeros(cfg.agent_num, dtype=np.int64) for _ in range(2)]
    nobs, rews, dones, infos = pool.step(acts)
    assert len(nobs) == len(rews) == len(dones) == len(infos) == 2
    snap = pool.snapshot()
    assert any("网格尺寸" in k for k, _ in snap), "环境快照字段缺失"
    pool.close()
    return f"{len(snap)} 个快照字段"


def t_pool_mp():
    from algos.madqn.madqn_envs import MADroneSearchEnv
    from algos.madqn.madqn_parallel_env import AsyncWorkerEnv
    cfg = Config(env_count=2)
    pool = AsyncWorkerEnv(cfg, MADroneSearchEnv, n=2, seed=0)
    try:
        obs = pool.reset()
        acts = [np.zeros(cfg.agent_num, dtype=np.int64) for _ in range(2)]
        nobs, rews, _, _ = pool.step(acts)
        assert len(nobs) == 2 and len(rews) == 2
        assert len(pool.reset_envs([0])) == 1, "reset_envs 单环境重置失败"
    finally:
        pool.close()
    return "多进程池 reset/step/reset_envs OK"


def t_prune():
    """⑨ final 模型清理：**删文件必须先证明边界划对了**。"""
    from config import Config
    from rlinfra.paths import RunPaths
    d = _tmpdir()
    try:
        paths = RunPaths(name="prune_test", runs_root=d).ensure()
        # 造 6 个 final + 一些「不该被删」的文件
        for i in range(1, 7):
            open(paths.final_model("madqn", i), "w").close()
        untouched = [paths.best_model("madqn"), paths.final_model("madqn", 0),
                     paths.pretrain_model("madqn"), os.path.join(paths.root, "train.log"),
                     os.path.join(paths.root, "madqn_final_abc.pth"),       # 非数字序号
                     os.path.join(paths.root, "other_final_1.pth")]         # 别的算法
        for p in untouched:
            open(p, "w").close()

        removed = paths.prune_final_models("madqn", 2)
        assert len(removed) == 4, f"应删 4 个，实删 {len(removed)}"
        left = sorted(f for f in os.listdir(paths.root) if re.fullmatch(r"madqn_final_\d+\.pth", f))
        assert left == ["madqn_final_5.pth", "madqn_final_6.pth"], f"留下的不对：{left}"
        for p in untouched:
            assert os.path.exists(p), f"误删了不该删的文件：{p}"
        assert paths.prune_final_models("madqn", 0) == [], "keep=0 时应不清理"
        # best 也要在 cfg 校验里守住
        cfg = Config(save_keep_last=0)
        cfg.validate()
    finally:
        _rmtmp(d)
    return "只删本算法 final_<N>，best/final/pretrain/他算法全部幸存"


def t_log_contract(tmpdir: str):
    """最关键的一项：日志契约 ↔ 诊断脚本 双向自证。"""
    import analyze_training_log as atl
    log = os.path.join(tmpdir, "contract.log")
    lg = TrainLogger(log, algo="MADQN", echo=False)
    lg.identity("madqn", "冒烟自检")
    lg.env_banner("MADQN", [("网格尺寸 GRID_SIZE", "20"), ("智能体数量 agent_num", "2"),
                            ("智能体起始位置", "[[1, 1], [18, 18]]"),
                            ("目标点位置", "[[10, 10]]"),
                            ("固定障碍物坐标", "[[3, 5], [7, 2]]"),
                            ("初始朝向 (度)", "[0, 90]"),
                            ("覆盖率达成阈值", "0.90")])
    lg.overview("cpu", 12345.67, 2, 50000, 300)
    lg.ep_line(1, 232.25, 64.76, 357, 0.0123, 0.42, 0.0567)
    lg.ep_line(2, -156.30, 92.60, 120, 0.0055, 1.20, 0.0234)
    lg.train_line(10, 1.6372, 1.62, 0.86, 0.997, 1.0208)
    lg.reward_line({"new_cell_base": 0.0312, "repeat_penalty": -0.0783,
                    "milestone_50": 0.0040, "stagnation_penalty": -0.0012})
    lg.best_line(2, -156.3, 92.6, "/tmp/x.pth")
    lg.test_line(91.20, 100.55, 80.00, 150.5)
    lg.resources("2026-09-10 10:50:43", "2026-09-10 11:00:00", 1234.5, 512.25)
    lg.close()

    res = atl.analyze(log)
    assert "error" not in res, f"诊断脚本读不了这份日志：{res.get('error')}"
    assert len(res["ep"]) == 2, f"EP 行解析数 {len(res['ep'])} ≠ 2"
    ep1 = res["ep"][0]
    assert ep1[0] == 1 and abs(ep1[1] - 232.25) < 1e-6 and abs(ep1[2] - 64.76) < 1e-6
    assert ep1[3] == 357 and abs(ep1[5] - 0.42) < 1e-6, f"curQ 没解析出来：{ep1}"
    assert len(res["train"]) == 1 and abs(res["train"][0][4] - 0.997) < 1e-6, "TRAIN 行没解析出来"
    assert len(res["best"]) == 1 and res["best"][0][2] == 92.6, "最优模型行没解析出来"
    assert res["env"]["grid"] == "20" and res["env"]["agent"] == "2", f"环境快照解析失败：{res['env']}"
    assert abs(res["test"]["coverage"] - 91.20) < 1e-6, "测试汇总行没解析出来"
    assert abs(res["test"]["sr"] - 80.0) < 1e-6, "成功率没解析出来"
    assert res["device"] == "cpu", "运行设备没解析出来"
    assert res["episodes"] == 300, "总轮次没解析出来"
    assert res["meta"].get("cpu_time") and res["meta"].get("mem"), "资源行没解析出来"
    comps = res["reward_comp"]
    assert abs(comps["repeat_penalty"] - (-0.0783)) < 1e-9, f"奖励分量解析失败：{comps}"
    return f"EP/TRAIN/REWARD/BEST/ENV/TEST/RES 全字段可解析，{len(comps)} 个奖励分量"


def t_end2end(tmpdir: str):
    from algos.madqn.madqn_trainer import train_madqn_ctde
    cfg = Config(name="smoke_e2e", runs_root=tmpdir, episodes=3, grid_size=10,
                 agent_num=2, obstacle_num=4, COMMON_min_buffer=100,
                 COMMON_learn_per_episode=1, COMMON_batch_size=32, env_count=2,
                 COMMON_random_episodes=1, print_interval=1, save_interval=2,
                 COMMON_eval_interval=1,          # 冒烟只跑 3 回合 → 每回合都做贪心评估才存得上 best
                 init_max_train_steps=60, final_test_steps=60)
    cfg._derive(); cfg.validate()
    paths = RunPaths.from_config(cfg).ensure()
    logger = TrainLogger(paths.log, algo="MADQN", echo=False)
    try:
        out = train_madqn_ctde(cfg, paths, logger)
    finally:
        logger.close()
    assert os.path.exists(paths.best_model("madqn")), "最优模型没落盘"
    assert out["buffer_size"] > 0, "回放池没收到转移"
    import analyze_training_log as atl
    res = atl.analyze(paths.log)
    assert len(res["ep"]) == 3, f"端到端日志里 EP 行数 {len(res['ep'])} ≠ 3"
    assert res["reward_comp"], "端到端日志里没有奖励分量行"
    return f"best_cov={out['best_coverage']:.1f}% buffer={out['buffer_size']} 日志 {len(res['ep'])} 回合可解析"


# ══════════════════════ 主流程 ══════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", action="store_true", help="额外验证多进程环境池")
    args = ap.parse_args()

    print("=" * 72)
    print("RL_infra 冒烟自检")
    print("=" * 72)
    check("① 配置校验 + 派生量", t_config)
    check("② 环境 reset/step/可复现", t_env)
    check("③ 网络前向", t_network)
    check("④ 回放池", t_buffer)
    check("⑤ 智能体 act/learn/save/load", t_agent)
    check("⑥ 环境池（单进程降级路径）", t_pool)
    if args.mp:
        check("⑥b 环境池（多进程）", t_pool_mp)
    check("⑨ final 模型清理（删文件边界）", t_prune)

    # 🔴 Windows + DSH 沙箱（2026-09-30 实测）：`%TEMP%` 下**新建**的目录对 Python
    #    子进程拒写（`PermissionError: WinError 5`）；且 `tempfile.tempdir` 会被
    #    `TemporaryDirectory.__init__` 置空 → 一律走 `_tmpdir()`（显式 dir=）。
    d = _tmpdir()
    try:
        check("⑦ 日志契约 ↔ 诊断脚本", lambda: t_log_contract(d))
        check("⑧ 端到端 3 回合训练", lambda: t_end2end(d))
    finally:
        _rmtmp(d)

    print("=" * 72)
    if _failures:
        print(f"❌ 冒烟自检失败 {len(_failures)} 项：{_failures}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
