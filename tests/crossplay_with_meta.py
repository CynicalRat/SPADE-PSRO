import sys
import os

# 将项目根目录插入到 sys.path，确保可以导入 src 包
PROJECT_ROOT = os.environ.get(
    "PSRO_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re
from collections import OrderedDict
from stable_baselines3 import PPO  # 根据你的算法修改
from typing import List, Tuple
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
from src.envs.gameenv import SatelliteGameEnv
from src.envs.self_play_wrapper import RecurrentSelfPlayEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DDPG, SAC, TD3

from src.utils.payoff_table import PayoffTable
from src.utils.meta_solver import fictitious_play, maximum_entropy_nash, \
    min_support_nash, uniform_sample, naive_sp, LP_nash, \
        regularized_fictitious_play, regularized_fictitious_play_old, \
            dpp_driven_nash_, dpp_driven_nash

PROJECT_ROOT = os.environ.get(
    "PSRO_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

"""
[A] Naive Self Play 0-49
[B] Standard FP     50-99
[C] Uniform         100-149
[D] my method       150-200
[E] ppo, ddpg, td3, random  201,202,203,204
"""

strategy_dirs = {
    "NSP": PROJECT_ROOT+"/results/baselines/NSP/0717/psro_lstm/policies",
    "SFP": PROJECT_ROOT+"/results/baselines/SFP/psro_lstm/policies",
    "UNIFORM": PROJECT_ROOT+"/results/baselines/UNIFORM/0717/psro_lstm/policies",
    "my_method": PROJECT_ROOT+"/results/warmup/0312_div2/psro_lstm/policies",
    "ddpg": PROJECT_ROOT+"/results/baselines/ddpg_normalize/20250722-2203",
    "TD3": PROJECT_ROOT+"/results/baselines/TD3_normalize/20250808-1638",
    "lstm_ppo": PROJECT_ROOT+"/results/baselines/lstm_ppo_normalize/20250416-1808/test"

}

NSP = 0
SFP = 1
UNIFORM = 2
MY = 3
DDPG_ = 4
TD3_ = 5
LSTM_PPO_ = 6

lstm_ppo_policy_kwargs = dict(    
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),


def gather_all_algos(group_num=50):
    """扫描 strategy_dirs 并返回 global_id -> strategy 信息的映射表。"""
    global_index = 0
    all_strategies = []

    def _extract_num(fname):
        m = re.search(r"(\d+)", fname)
        return int(m.group(1)) if m else float("inf")

    for algo_name, dir_path in strategy_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"Warning: strategy dir not found or not a directory: {dir_path}")
            continue

        try:
            strategy_files = [f for f in os.listdir(dir_path) if f.endswith((".zip", ".pt"))]
        except Exception as e:
            print(f"Warning: cannot list dir {dir_path}: {e}")
            continue

        if not strategy_files:
            # optional: print(f"No policy files in {dir_path}")
            continue

        strategy_files = sorted(strategy_files, key=_extract_num)

        for local_id, filename in enumerate(strategy_files):
            full_path = os.path.join(dir_path, filename)
            # 构造 vecnorm 路径（更稳妥地替换后缀）
            base, ext = os.path.splitext(full_path)
            vec_norm_path = base + "_vecnorm.pkl"

            if not os.path.exists(vec_norm_path):
                print(f"Warning: vecnorm not found for {full_path} -> expected {vec_norm_path}")
                continue

            all_strategies.append({
                "algo": algo_name,
                "local_id": local_id,
                "global_id": global_index,
                "policy_path": full_path,
                "vecnorm_path": vec_norm_path
            })
            global_index += 1

            if local_id + 1 >= group_num:
                break

    print(f"共加载 {len(all_strategies)} 个策略")
    # print(all_strategies)
    id_to_strategy = {entry["global_id"]: entry for entry in all_strategies}
    return id_to_strategy, all_strategies


def make_env():
    return SatelliteGameEnv(shift_steps=6000)

# LRU cache for policy + vecnorm env wrapper
policy_cache = OrderedDict()
MAX_POLICY_CACHE_SIZE = 40  # 可根据显存/内存手动调整


def _cache_insert(global_id, value):
    if global_id in policy_cache:
        policy_cache.move_to_end(global_id)
        policy_cache[global_id] = value
        return

    policy_cache[global_id] = value
    if len(policy_cache) > MAX_POLICY_CACHE_SIZE:
        evicted_id, _ = policy_cache.popitem(last=False)
        print(f"[INFO] Policy cache exceeded size {MAX_POLICY_CACHE_SIZE}, evicted global_id={evicted_id}")


def load_policy_from_global_id(global_id: int, id_to_strategy, device):
    """Loads a policy into memory if it's not already there, with in-process caching."""
    if global_id in policy_cache:
        policy_cache.move_to_end(global_id)
        return policy_cache[global_id]

    target = id_to_strategy.get(global_id)
    if target is None:
        raise KeyError(f"global_id {global_id} not found in id_to_strategy")

    policy_path = target["policy_path"]
    env_path = target["vecnorm_path"]
    algo_class = target["algo"]

    try:
        if algo_class in ["SFP", "NSP", "UNIFORM", "my_method", "lstm_ppo"]:
            if algo_class == "lstm_ppo":
                model = RecurrentPPO.load(policy_path, device=device)
            else:
                model = RecurrentPPO.load(PROJECT_ROOT+"/results/baselines/lstm_ppo_normalize/20250416-1808/checkpoint_500000_steps.zip", device='cpu')
                model.policy.load_state_dict(torch.load(policy_path, map_location=device))

        elif algo_class == "TD3":
            model = TD3.load(policy_path, device=device)
        elif algo_class == "ddpg":
            model = DDPG.load(policy_path, device=device)
        else:
            raise ValueError(f"Unsupported algorithm class: {algo_class}")

        model.policy.eval()

        env = DummyVecEnv([make_env])
        loaded_norm_env = VecNormalize.load(env_path, env)
        loaded_norm_env.training = False
        loaded_norm_env.norm_reward = False

        _cache_insert(global_id, (model.policy, loaded_norm_env))
        return policy_cache[global_id]

    except Exception:
        import traceback
        print(f"[ERROR] Failed to load policy {global_id} ({algo_class}) at {policy_path}")
        print(f"[ERROR] Vecnorm path: {env_path}")
        traceback.print_exc()
        policy_cache[global_id] = (None, None)
        return None, None


def matchup(eval_env_instance, id_to_strategy, global_id_i, global_id_j, device, episodes_per_matchup=10):
    """评估两个策略对战的平均收益"""
    # print(f"Evaluating matchup: {global_id_i} vs {global_id_j}")

    policy_i, norm_env_for_eval_i = load_policy_from_global_id(global_id_i, id_to_strategy, device)
    policy_j, norm_env_for_eval_j = load_policy_from_global_id(global_id_j, id_to_strategy, device)

    if policy_i is None or policy_j is None or norm_env_for_eval_i is None or norm_env_for_eval_j is None:
        print(f"[WARN] Skip matchup {global_id_i} vs {global_id_j} due to missing policy")
        return 0.0, 0.0

    avg_payoffs = eval_env_instance.calculate_payoff(
        policy_i, policy_j, norm_env_for_eval_i, norm_env_for_eval_j, episodes=episodes_per_matchup
    )

    payoff_i = float(avg_payoffs.get(1, 0.0))
    payoff_j = float(avg_payoffs.get(0, 0.0))

    # print(f"Matchup result: {global_id_i} vs {global_id_j} -> {{1: {payoff_i}, 0: {payoff_j}}}")
    return payoff_i, payoff_j

def matchup_meta(eval_env_instance, algo_id_i, algo_id_j, algo_meta_1, algo_meta_2, id_to_strategy, device, episodes_per_matchup=1000):
    '''评估两个算法meta的对战表现'''
    total_payoff_i = 0.0
    total_payoff_j = 0.0
    win_i = 0
    win_j = 0

    len_i = len(algo_meta_1)
    len_j = len(algo_meta_2)

    if len_i == 0 or len_j == 0:
        raise ValueError("algo_meta_1 and algo_meta_2 must be non-empty")

    idx_i = np.random.choice(len_i, size=episodes_per_matchup, p=algo_meta_1)
    idx_j = np.random.choice(len_j, size=episodes_per_matchup, p=algo_meta_2)

    for pi, pj in zip(idx_i, idx_j):
        payoff_i, payoff_j = matchup(eval_env_instance, id_to_strategy, int(pi), int(pj), device=device, episodes_per_matchup=1)
        total_payoff_i += payoff_i
        total_payoff_j += payoff_j

        if payoff_i > payoff_j:
            win_i += 1
        elif payoff_j > payoff_i:
            win_j += 1

    # print(f"Matchup result (meta) {algo_id_i} vs {algo_id_j}: -> {total_payoff_i}, {total_payoff_j} | wins {win_i}:{win_j}")
    return total_payoff_i, total_payoff_j, win_i, win_j


def get_submatrix(full_matrix, indices):
    """
    从完整矩阵中提取子矩阵。
    """
    return full_matrix[np.ix_(indices, indices)]


def get_ne_from_submatrix(payoff_matrix, meta_solver, indices):
    sub_m = get_submatrix(payoff_matrix, indices)
    r = meta_solver(PayoffTable(sub_m))

    # 兼容多种 meta_solver 返回值格式
    # LP_nash 返回 (row_strategy, col_strategy, value)
    # naive_sp/uniform_sample/fictitious_play 返回 strategy vector
    # 可能存在 (strategy, history) 这样的结构
    if isinstance(r, (tuple, list)):
        if len(r) == 3:
            row_strat = r[0]
        elif len(r) == 2 and isinstance(r[0], (np.ndarray, list)):
            row_strat = r[0]
        else:
            row_strat = np.asarray(r).ravel()
    else:
        row_strat = np.asarray(r).ravel()

    row_strat = np.asarray(row_strat, dtype=float).ravel()

    if row_strat.size != len(indices):
        raise ValueError(
            f"row_strat length {row_strat.size} != indices length {len(indices)}"
        )

    meta_full = np.zeros(payoff_matrix.shape[0], dtype=float)
    for k, idx in enumerate(indices):
        meta_full[idx] = row_strat[k]

    return meta_full, row_strat


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from multiprocessing import Pool
    # base_env = SatelliteGameEnv()
    # env = RecurrentSelfPlayEnv(lambda: base_env)
    id_to_strategy, all_strategies = gather_all_algos(group_num=50)

    payoff_matrix = np.zeros((7, 7))
    win_matrix = np.zeros((7, 7))

    # 计算每个算法的meta-strategy
    strat_num=50
    M_all = np.load("./results/metrics/20260401_div2/payoff_final.npy")

    M1 = get_submatrix(M_all, list(range(0*strat_num, (0+1)*strat_num)))
    meta_nsp, _ = get_ne_from_submatrix(M_all, naive_sp, list(range(0*strat_num, (0+1)*strat_num)))

    M2 = get_submatrix(M_all, list(range(1*strat_num, (1+1)*strat_num)))
    meta_sfp, _ = get_ne_from_submatrix(M_all, LP_nash, list(range(1*strat_num, (1+1)*strat_num)))

    M3 = get_submatrix(M_all, list(range(2*strat_num, (2+1)*strat_num)))
    meta_uniform, _ = get_ne_from_submatrix(M_all, uniform_sample, list(range(2*strat_num, (2+1)*strat_num)))

    M4 = get_submatrix(M_all, list(range(3*strat_num, (3+1)*strat_num)))
    meta_my, _ = get_ne_from_submatrix(M_all, LP_nash, list(range(3*strat_num, (3+1)*strat_num)))  

    meta_ddpg = np.zeros_like(meta_nsp)
    meta_ddpg[-3]=1.0
    meta_td3 = np.zeros_like(meta_nsp)
    meta_td3[-2]=1.0
    meta_lstm_ppo = np.zeros_like(meta_nsp)
    meta_lstm_ppo[-1]=1.0



    # =====  尝试加载已有结果 =====
    result_dir = os.path.join(PROJECT_ROOT, "results", "metrics", "crossplay_win_meta_20260403div2")
    os.makedirs(result_dir, exist_ok=True)

    payoff_file = os.path.join(result_dir, "payoff.npy")
    win_file = os.path.join(result_dir, "win.npy")

    try:
        payoff_matrix_ = np.load(payoff_file)
        win_matrix_ = np.load(win_file)
        if payoff_matrix_.shape == payoff_matrix.shape:
            payoff_matrix = payoff_matrix_
            win_matrix = win_matrix_
            print("历史结果已加载。")
        else:
            print(f"历史结果维度不匹配，跳过加载。当前维度 {payoff_matrix.shape}，历史维度 {payoff_matrix_.shape}")
    except FileNotFoundError:
        payoff_matrix = np.zeros((7, 7))
        win_matrix = np.zeros((7, 7))
        print("未找到历史结果，重新计算。")



    def matchup_task(args):
        i, j, algo_meta_1, algo_meta_2, id_to_strategy = args
        base_env = SatelliteGameEnv()
        eval_env = RecurrentSelfPlayEnv(lambda: base_env)
        total_payoff_i, total_payoff_j, win_i, win_j = matchup_meta(
            eval_env, i, j, algo_meta_1, algo_meta_2, id_to_strategy, device="cpu", episodes_per_matchup=1000
        )
        return i, j, total_payoff_i, total_payoff_j, win_i, win_j


    tasks = [
        (i, j, meta_1, meta_2, id_to_strategy)
        for i, meta_1 in enumerate([meta_nsp, meta_sfp, meta_uniform, meta_my, meta_ddpg, meta_td3, meta_lstm_ppo])
        for j, meta_2 in enumerate([meta_nsp, meta_sfp, meta_uniform, meta_my, meta_ddpg, meta_td3, meta_lstm_ppo])
        if (i < j) and payoff_matrix[i, j]==0  # Skip self-play, duplicate matchups and done matchups
    ]

    results = Parallel(n_jobs=-1, backend="loky")(delayed(matchup_task)(task) for task in tasks)

    for i, j, pi, pj, wi, wj in results:
        payoff_matrix[i, j] = pi
        payoff_matrix[j, i] = pj
        win_matrix[i, j] = wi
        win_matrix[j, i] = wj
        # if i % 10 == 0 and j == n-1:
    np.save(payoff_file, payoff_matrix)
    np.save(win_file, win_matrix)

    print("✅ 完成meta-strategy-crossplay多进程计算。")

