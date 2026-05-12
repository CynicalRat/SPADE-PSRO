import os
import torch
import yaml # If loading config from YAML

# Import PSRO classes and the environment
from src.algorithms.psro import BasicPSRO, LstmPSRO
from src.envs.gameenv import SatelliteGameEnv
from src.envs.self_play_wrapper import RecurrentSelfPlayEnv
from stable_baselines3.common.monitor import Monitor

# Define a function to create the environment instance
# This is passed to the PSRO class

def create_env_fn(opponent_sampler_fn=None):
    """
    Creates an instance of the SatelliteGameEnv.
    可以在这里修改基础环境的参数
    """
    base_env = SatelliteGameEnv()
    # Use RecurrentSelfPlayEnv, passing the opponent sampler
    sp_env = RecurrentSelfPlayEnv(lambda: base_env, opponent_sampler_fn)
    return Monitor(sp_env)

def linear_schedule(initial_value: float, min_value: float):
    """
    返回一个 schedule(progress_remaining)：
      progress_remaining=1 时，lr = initial_value
      progress_remaining=0 时，lr = min_value
      中间线性插值
    """
    def _func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)
    return _func

def warmup_schedule(initial_value: float, min_value: float,
                           warmup_fraction: float = 0.01, 
                           warmup_start_lr: float = 1e-7):
    def _func(progress_remaining: float) -> float:
        """
        根据剩余进度计算当前学习率。
        progress_remaining 从 1.0 (开始) 变化到 0.0 (结束)。
        """
        # 计算Warm-up阶段结束时的进度点
        warmup_end_progress = 1.0 - warmup_fraction

        if progress_remaining > warmup_end_progress:
            # --- 处于Warm-up阶段 ---
            # 计算在Warm-up阶段内的进度 (从0到1)
            # 当 progress_remaining = 1.0, warmup_progress = 0
            # 当 progress_remaining = warmup_end_progress, warmup_progress = 1
            warmup_progress = (1.0 - progress_remaining) / warmup_fraction
            
            # 从 warmup_start_lr 线性插值到 initial_value
            current_lr = warmup_start_lr + warmup_progress * (initial_value - warmup_start_lr)
        else:
            # --- 处于Decay阶段 ---
            # 计算在Decay阶段内的进度 (从1到0)
            # 当 progress_remaining = warmup_end_progress, decay_progress = 1
            # 当 progress_remaining = 0.0, decay_progress = 0
            decay_progress = progress_remaining / warmup_end_progress
            
            # 从 initial_value 线性插值到 min_value
            current_lr = min_value + decay_progress * (initial_value - min_value)
            
        return current_lr

    return _func
if __name__ == "__main__":
    print("Starting PSRO Training Process...")

    # Configure PSRO
    USE_LSTM = True
    ROOT_DIR = 'results/baselines/NSP/0717'
    lr_schedule = linear_schedule(3e-4, 5e-5)
    ORACLE_CONFIG = {
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
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "tensorboard_log": f'{ROOT_DIR}/psro_{"lstm" if USE_LSTM else "basic"}/logs/',
        "norm_obs": True,
        "norm_reward": False,
        "clip_obs": 15.0,
        "total_timesteps": 1_000_000, # Timesteps per BR oracle training
        "convergence_window": 100,
        "convergence_mean_change_threshold": .5, # Adjust this threshold based on expected reward scale
        "convergence_std_threshold": None, # Disable std check by default
        "win_rate_threshold":0.8,
        "verbose":1,
    }




    META_SOLVER_CONFIG = {
        "n_iterations": 100,  # Reduced for faster meta-game solving
        "type":"last_only",
    }
    

    SAVE_DIR = f'{ROOT_DIR}/psro_{"lstm" if USE_LSTM else "basic"}/policies/'
    PSRO_ITERATIONS = 51 # Number of new policies to train
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
