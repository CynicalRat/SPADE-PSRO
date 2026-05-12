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

PROJECT_ROOT = os.environ.get(
    "PSRO_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

"""
[A] Naive Self Play 0-49
[B] Standard FP     50-99
[C] Uniform         100-149
[D] my method       150-200
[E] lstm-ppo, ddpg, td3, random  201,202,203,204
"""

strategy_dirs = {
    "NSP": PROJECT_ROOT+"/results/baselines/NSP/0717/psro_lstm/policies",
    "SFP": PROJECT_ROOT+"/results/baselines/SFP/psro_lstm/policies",
    "UNIFORM": PROJECT_ROOT+"/results/baselines/UNIFORM/0717/psro_lstm/policies",
    # "my_method": PROJECT_ROOT+"/results/warmup/0722/psro_lstm/policies",
    "my_method": PROJECT_ROOT+"/results/warmup/0120/psro_lstm/policies",
    "ddpg": PROJECT_ROOT+"/results/baselines/ddpg_normalize/20250722-2203",
    "TD3": PROJECT_ROOT+"/results/baselines/TD3_normalize/20250808-1638",
    "lstm_ppo": PROJECT_ROOT+"/results/baselines/lstm_ppo_normalize/20250416-1808/test"

}

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

            if os.path.exists(vec_norm_path):
                all_strategies.append({
                    "algo": algo_name,
                    "local_id": local_id,
                    "global_id": global_index,
                    "policy_path": full_path,
                    "vecnorm_path": vec_norm_path
                })
            else:
                # 可选择记录缺失 vecnorm 的策略或跳过
                print(f"Warning: vecnorm not found for {full_path} -> expected {vec_norm_path}")

            global_index += 1
            if local_id + 1 >= group_num:
                break

    print(f"共加载 {len(all_strategies)} 个策略")
    # print(all_strategies)
    id_to_strategy = {entry["global_id"]: entry for entry in all_strategies}
    return id_to_strategy, all_strategies




def make_env():
    return SatelliteGameEnv(shift_steps=6000)


def load_policy_from_global_id(global_id: int, id_to_strategy, device):
    """Loads a policy into memory if it's not already there."""
    try:
        target = id_to_strategy.get(global_id)
        if target is None:
            raise KeyError(f"global_id {global_id} not found in id_to_strategy")
   
        policy_path = target["policy_path"]
        env_path = target["vecnorm_path"]
        algo_class = target["algo"]

        if algo_class in ["SFP", "NSP", "UNIFORM", "my_method", "lstm_ppo"]:

            if algo_class == "lstm_ppo":
                model = RecurrentPPO.load(policy_path)
            else:
                model = RecurrentPPO.load(PROJECT_ROOT+"/results/baselines/lstm_ppo_normalize/20250416-1808/checkpoint_500000_steps.zip", device='cpu')
                model.policy.load_state_dict(torch.load(policy_path, map_location=device))

        elif algo_class in ["ddpg", "TD3"]:
            if algo_class == "TD3":
                model = TD3.load(policy_path, device=device)
            elif algo_class == "ddpg":
                model = DDPG.load(policy_path, device=device)
        model.policy.eval()
        env = DummyVecEnv([make_env])
        loaded_norm_env = VecNormalize.load(env_path, env)

        loaded_norm_env.training = False
        loaded_norm_env.norm_reward = False
        return model.policy, loaded_norm_env

    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to load policy {global_id} ({algo_class}) at {policy_path}")
        print(f"[ERROR] Vecnorm path: {env_path}")
        traceback.print_exc()
        return None


def matchup(eval_env_instance, id_to_strategy, global_id_i, global_id_j, device, episodes_per_matchup=10):
    """评估两个策略对战的平均收益"""
    print(f"Evaluating matchup: {global_id_i} vs {global_id_j}")

    policy_i, norm_env_for_eval_i = load_policy_from_global_id(global_id_i, id_to_strategy, device)
    policy_j, norm_env_for_eval_j = load_policy_from_global_id(global_id_j, id_to_strategy, device)

    avg_payoffs = eval_env_instance.calculate_payoff(
        policy_i, policy_j, norm_env_for_eval_i, norm_env_for_eval_j, episodes=episodes_per_matchup
    )
    # Update the payoff table (assuming payoff is for player 1 vs player 2)
    # PayoffTable expects {1: payoff1, 0: payoff2} or similar based on agent IDs
    # Adapt this based on your calculate_payoff return format and agent IDs
    # Assuming agent IDs are 1 and 0 as in SatelliteGameEnv
    payoff_dict = {
        1: avg_payoffs.get(1, 0), # Payoff for agent 1 (row player i)
        0: avg_payoffs.get(0, 0)  # Payoff for agent 0 (column player j)
    }
    print(f"Matchup result: {global_id_i} vs {global_id_j} -> {payoff_dict}")
    return avg_payoffs.get(1, 0), avg_payoffs.get(0, 0)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from multiprocessing import Pool
    # base_env = SatelliteGameEnv()
    # env = RecurrentSelfPlayEnv(lambda: base_env)
    id_to_strategy, all_strategies = gather_all_algos(group_num=50)
    n = len(all_strategies)
    payoff_matrix = np.zeros((n, n))

    # =====  尝试加载已有结果 =====
    partial_file = "./results/metrics/2026210/crossplay_payoff_matrix_partial.npy"
    final_file = "./results/metrics/2026210/payoff_matrix_parallel.npy"

    try:
        payoff_matrix = np.load(partial_file)
        if payoff_matrix.shape != (n, n):
            print("⚠️ 已保存矩阵尺寸不一致，重新初始化。")
            payoff_matrix = np.zeros((n, n))
        else:
            print("🔄 已加载部分结果，将继续计算未完成部分。")
    except FileNotFoundError:
        payoff_matrix = np.zeros((n, n))
        print("未找到历史结果，重新计算。")



    def matchup_task(args):
        i, j, id_to_strategy = args
        base_env = SatelliteGameEnv()
        eval_env = RecurrentSelfPlayEnv(lambda: base_env)
        pi, pj = matchup(eval_env, id_to_strategy, i, j, device="cpu", episodes_per_matchup=10)
        return i, j, pi, pj


    batch_size = 10 # 每20行一批
    for start_row in range(0, n, batch_size):
        end_row = min(start_row+batch_size, n)

        # 检查该批是否已完成（如果所有行都非零，就跳过）
        if np.all(payoff_matrix[start_row:end_row, :] != 0):
            print(f"⏭️ 跳过第 {start_row} ~ {end_row-1} 行（已完成）")
            continue


        print(f"开始计算第 {start_row} ~ {end_row-1} 行")
        tasks = [(i, j, id_to_strategy) for i in range(start_row, end_row) for j in range(i+1, n)]
        results = Parallel(n_jobs=-1, backend="loky")(delayed(matchup_task)(task) for task in tasks)
    # with Pool(processes=1) as pool:
        # results = pool.imap_unordered(matchup_task, tasks)
        
    # for task in tasks:
        # results = [matchup_task(task)]
        for i, j, a, b in results:
            payoff_matrix[i, j] = a
            payoff_matrix[j, i] = b
            # if i % 10 == 0 and j == n-1:
            np.save(partial_file, payoff_matrix)
            # print(f"进度: {i}/{n} 行完成。")
        print(f"批次 {start_row}~{end_row-1} 进度: {len(results)}/{len(tasks)} 任务完成")
    # print(payoff_matrix)
    np.save(final_file, payoff_matrix)
    print("✅ 完成 payoff matrix 多进程计算。")

    # print(payoff_matrix)

