import os
import yaml # If loading config from YAML

# Import PSRO classes and the environment
from src.algorithms.psro import BasicPSRO, LstmPSRO
from src.envs.gameenv import SatelliteGameEnv
from src.envs.self_play_wrapper import RecurrentSelfPlayEnv
from stable_baselines3.common.monitor import Monitor

from src.utils.schedules import warmup_linear_schedule, linear_schedule
# Define a function to create the environment instance
# This is passed to the PSRO class


import random
import numpy as np
import torch

# 设置全局随机种子（建议选择一个固定值，如42）
SEED = 1234

# 设置Python内置random模块的种子
random.seed(SEED)

# 设置NumPy的种子 这里设置好像会导致sample的结果一直固定 尝试在psro中弃用np.random.choice，改用numpy的新随机生成器（Generator）来采样，看看能否解决这个问题。
np.random.seed(SEED)

# 设置PyTorch的种子（包括CPU和GPU）
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # 可选：确保CUDA操作的确定性（可能会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def create_env_fn(opponent_sampler_fn=None):
    """
    Creates an instance of the SatelliteGameEnv.
    可以在这里修改基础环境的参数
    """
    base_env = SatelliteGameEnv()
    base_env.set_seed(SEED)  # 移除：SatelliteGameEnv 的 seed 属性被覆盖为整数，无法调用
    # Use RecurrentSelfPlayEnv, passing the opponent sampler
    sp_env = RecurrentSelfPlayEnv(lambda: base_env, opponent_sampler_fn)
    return Monitor(sp_env)


if __name__ == "__main__":
    print("Starting PSRO Training Process...")

    # Configure PSRO
    USE_LSTM = True
    ROOT_DIR = 'results/warmup/0413'
    lr_schedule = warmup_linear_schedule(3e-4, 5e-5)
    ORACLE_CONFIG = {
        "seed": SEED,
        "n_envs": 4,
        "learning_rate": lr_schedule, #3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef" : 0.01,

        "policy_kwargs" : dict(    
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
        "device": "cuda:3" if torch.cuda.is_available() else "cpu",
        "tensorboard_log": f'{ROOT_DIR}/psro_{"lstm" if USE_LSTM else "basic"}/logs/',
        "norm_obs": True,
        "norm_reward": False,
        "clip_obs": 15.0,
        "total_timesteps": 1_000_000, # Timesteps per BR oracle training
        "convergence_window": 50,
        "convergence_mean_change_threshold": .5, # Adjust this threshold based on expected reward scale
        "convergence_std_threshold": None, # Disable std check by default
        "win_rate_threshold":0.85,
        "verbose":1,

    }




    META_SOLVER_CONFIG = {
        "n_iterations": 10000,  # Reduced for faster meta-game solving
        "type": "dpp_driven_nash",
        "diversity_weight": 0.2,
        "chunk_size": 10,
        "alpha": 3,
    }
    

    SAVE_DIR = f'{ROOT_DIR}/psro_{"lstm" if USE_LSTM else "basic"}/policies/'
    PSRO_ITERATIONS = 50 # Number of new policies to train
    EPISODES_PER_MATCHUP = 10 # For payoff calculation

    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Ensure log directory exists
    if ORACLE_CONFIG.get("tensorboard_log"):
        os.makedirs(ORACLE_CONFIG["tensorboard_log"], exist_ok=True)


    # --- Initialize PSRO ---
    if USE_LSTM:
        psro_trainer = LstmPSRO(
            env_fn=create_env_fn,
            oracle_config=ORACLE_CONFIG,
            meta_solver_config=META_SOLVER_CONFIG,
            save_dir=SAVE_DIR,
            max_pool_size=15 # Example max policies in memory
        )
    else:
        psro_trainer = BasicPSRO(
            env_fn=create_env_fn,
            oracle_config=ORACLE_CONFIG,
            meta_solver_config=META_SOLVER_CONFIG,
            save_dir=SAVE_DIR,
            max_pool_size=15 # Example max policies in memory
        )

    # --- Run PSRO Training ---
    try:
        psro_trainer.run(
            iterations=PSRO_ITERATIONS,
            reuse_policy=True, # Start new BR training from last one
            episodes_per_matchup=EPISODES_PER_MATCHUP
        )
    except Exception as e:
        print(f"\nAn error occurred during the PSRO run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nPSRO process finished or encountered an error.")
        # Optional: Add any final cleanup or saving steps here
