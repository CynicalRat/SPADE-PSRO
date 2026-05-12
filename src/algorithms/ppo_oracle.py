import os
import numpy as np
from collections import deque
from typing import Callable, Optional, Tuple, Any

import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from src.envs.self_play_wrapper import RecurrentSelfPlayEnv


class StopTrainingOnEpisodeRewardConvergence(BaseCallback):
    """
    当最近 N 个完成的 episode 的奖励的标准差低于阈值时停止训练。
    从Monitor包装的环境中获取 episode 信息。

    :param check_freq: (int) 每多少步检查一次环境的 dones 标志。
                       不必太高，因为我们只在 episode 结束时收集数据。
                       设为 1 可以在 episode 结束后立即记录。
    :param N: (int) 计算平均奖励和标准差的 episode 数量窗口。
    :param mean_change_threshold: Mean reward change threshold to stop training.
                                  Training stops if abs(current_mean - previous_mean) < threshold.
    :param std_threshold: (float) 停止训练的奖励标准差阈值.
    :param verbose: (int) Verbosity level: 0 for no output, 1 for info messages.
    """
    def __init__(self,
                 check_freq: int,
                 N: int = 50,
                 mean_change_threshold: float = 5.0, # Threshold for change in mean reward
                 std_threshold: Optional[float] = None, # Optional std check
                 win_rate_threshold: float = 0.8,   # 如果胜率足够了可以提前终止，避免过拟合
                 verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = 1#check_freq
        self.N = N
        self.mean_change_threshold = mean_change_threshold
        self.std_threshold = std_threshold
        self.episode_rewards = deque(maxlen=self.N)
        self.episode_count = 0
        self.previous_mean_reward = None # Store the mean reward from the previous check
        self.win_window = deque(maxlen=N)
        self.win_rate_threshold = win_rate_threshold
        self.stop_count = 0

    def _on_step(self) -> bool:
        # Check only at check_freq intervals
        if self.n_calls % self.check_freq == 0:
            infos = self.locals.get("infos", [])
            newly_finished_episodes = False
            stop_training = False
            for info in infos:
                if 'episode' in info:
                    reward = info['episode']['r']
                    self.episode_rewards.append(reward)
                    self.episode_count += 1
                    newly_finished_episodes = True
                    if self.verbose > 1:
                         print(f"Call: {self.n_calls}, Episode {self.episode_count} finished. Reward: {reward:.2f}. Deque length: {len(self.episode_rewards)}")
                # 胜负统计
                if info.get('done', False) and 'winner' in info:
                    self.win_window.append(1 if info['winner'] else 0)
                    if len(self.win_window) == self.win_window.maxlen:
                        win_rate = sum(self.win_window) / len(self.win_window)
                        self.logger.record('custom/win_rate', win_rate)
                        if win_rate >= self.win_rate_threshold:
                            # self.win_window.clear() 
                            stop_training = True # 直接停止训练     
                            stop_reason = (f"Win rate ({win_rate:.3f}) is above \
                                           threshold ({self.win_rate_threshold}).")          

            # Check convergence condition only if the deque is full and new episodes finished
            if len(self.episode_rewards) >= self.N and newly_finished_episodes:
                current_mean = np.mean(self.episode_rewards)
                current_std = np.std(self.episode_rewards)

                if self.verbose > 0:
                    print(f"Num calls: {self.n_calls}, Episode count: {self.episode_count}, "
                          f"Mean reward (last {len(self.episode_rewards)} episodes): {current_mean:.3f}, "
                          f"Std: {current_std:.3f}")

                stop_reason = ""

                # --- Check 1: Mean Reward Stabilization ---
                if self.previous_mean_reward is not None:
                    mean_change = abs(current_mean - self.previous_mean_reward)
                    if self.verbose > 0:
                         print(f"Change in mean reward since last check: {mean_change:.3f}")
                    if mean_change < self.mean_change_threshold:
                        self.stop_count +=1
                        if self.stop_count >= 0.5*self.N:
                            stop_training = True
                            stop_reason = (f"Mean reward change ({mean_change:.3f}) is below "
                                        f"threshold ({self.mean_change_threshold}).")
                    else:
                        self.stop_count = 0

                # Update previous mean reward for the next check
                self.previous_mean_reward = current_mean

                # --- Check 2: Standard Deviation (Optional) ---
                if not stop_training and self.std_threshold is not None:
                    if current_std < self.std_threshold:
                        stop_training = True
                        stop_reason = (f"Standard deviation ({current_std:.3f}) is below "
                                       f"threshold ({self.std_threshold}).")

                # --- Stop Training if either condition met ---
                if stop_training:
                    if self.verbose > 0:
                        print(f"\nStopping training: {stop_reason}")
                    return False # Return False to stop training

        return True # Return True to continue training


class PPOOracle:
    def __init__(self, env_fn: Callable[[], Any], 
                 opponent_sampler_fn: Callable[[], BasePolicy], 
                 **ppo_config: Any):
        """
        Args:
            env_fn: Function to create an instance of the base game environment (e.g., SatelliteGameEnv).
            opponent_sampler_fn: Function that samples an opponent policy from the current meta-strategy.
            **ppo_config: Configuration dictionary for the PPO model. Must include 'n_envs'.
                          Example: {"n_envs": 4, "learning_rate": 3e-4, "n_steps": 2048, ...}
        """
        self.env_fn = env_fn
        self.opponent_sampler_fn = opponent_sampler_fn
        self.ppo_config = ppo_config
        self.last_policy_state_dict = None # Store state_dict instead of full policy object
        self.log_dir = self.ppo_config.get("tensorboard_log", None)
        self.n_envs = self.ppo_config.get("n_envs")
        if self.n_envs is None:
            raise ValueError("PPOOracle config must include 'n_envs'")
        
        if self.log_dir:
            # 可能需要确保目录存在
            os.makedirs(self.log_dir, exist_ok=True)

    def train(self,
              start_policy_state_dict: Optional[dict] = None,
              start_vec_normalize_path: Optional[str] = None,
              total_timesteps: int = 2_500_000
             ) -> Tuple[BasePolicy, VecNormalize]:
        """
        Trains a PPO best response policy against opponents sampled from the meta-strategy.

        Args:
            start_policy_state_dict: State dictionary of a policy to initialize training from (optional).
            start_vec_normalize_path: Path to saved VecNormalize statistics to load (optional).
            total_timesteps: Total number of training steps for the PPO agent.

        Returns:
            Tuple containing the trained policy object and the VecNormalize wrapper instance.
        """
        print(f"Starting PPO Oracle training for {total_timesteps} timesteps...")

        def make_env():
            base_env = self.env_fn()
            sp_env = RecurrentSelfPlayEnv(lambda: base_env, self.opponent_sampler_fn)
            return Monitor(sp_env)

        # Create the Dummy Vectorized Environment
        vec_env = DummyVecEnv([make_env for _ in range(self.n_envs)])

        # 加载历史 VecNormalize 统计数据
        if start_vec_normalize_path and os.path.exists(start_vec_normalize_path):
            print(f"Loading VecNormalize statistics from: {start_vec_normalize_path}")
            norm_env = VecNormalize.load(start_vec_normalize_path, vec_env)
            # Make sure to set training=True if you want stats to continue updating
            norm_env.training = True
            norm_env.norm_reward = self.ppo_config.get("norm_reward", False) # Use config or default
        else:
            print("Initializing new VecNormalize statistics.")
            norm_env = VecNormalize(vec_env,
                                    norm_obs=self.ppo_config.get("norm_obs", True),
                                    norm_reward=self.ppo_config.get("norm_reward", False), # Often False for PSRO
                                    clip_obs=self.ppo_config.get("clip_obs", 10.0),
                                    gamma=self.ppo_config.get("gamma", 0.99)) # Use gamma from config

        # Link VecNormalize back to the underlying environments for opponent normalization
        # This allows RecurrentSelfPlayEnv.normalize_obs() to work correctly
        for env_idx in range(norm_env.num_envs):
             # Access the underlying RecurrentSelfPlayEnv instance
             # Path: VecNormalize -> DummyVecEnv -> Monitor -> RecurrentSelfPlayEnv
             if hasattr(norm_env.envs[env_idx], 'env') and hasattr(norm_env.envs[env_idx].env, 'set_vec_normalizer'):
                 norm_env.envs[env_idx].env.set_vec_normalizer(norm_env)
             else:
                  print(f"Warning: Could not set VecNormalizer for env {env_idx}. Check wrapper structure.")

        # Remove n_envs from config passed to PPO, as it's handled by DummyVecEnv
        model_ppo_config = self.ppo_config.copy()
        model_ppo_config.pop("n_envs", None)
        model_ppo_config.pop("norm_obs", None)
        model_ppo_config.pop("norm_reward", None)
        model_ppo_config.pop("clip_obs", None)
        # Remove callback specific params from model config
        model_ppo_config.pop("convergence_mean_change_threshold", None)
        model_ppo_config.pop("convergence_std_threshold", None)
        model_ppo_config.pop("convergence_std_threshold", None)
        model_ppo_config.pop("win_rate_threshold", None)
        
        # 初始化模型
        model = PPO("MlpPolicy", norm_env, **model_ppo_config)
        if start_policy_state_dict:
            model.policy.load_state_dict(start_policy_state_dict)

        callback = StopTrainingOnEpisodeRewardConvergence(
            check_freq=max(1, self.ppo_config.get("n_steps", 2048) // self.n_envs), # Check roughly every rollout per env
            N=self.ppo_config.get("convergence_window", 50),
            mean_change_threshold=self.ppo_config.get("convergence_mean_change_threshold", 5.0),
            std_threshold=self.ppo_config.get("convergence_std_threshold", None), # Default to None (disabled)
            win_rate_threshold= self.ppo_config.get("win_rate_threshold", 0.8), # Default to 0.8
            verbose=1
        )


        print("Starting PPO model learning...")
        try:
            model.learn(total_timesteps=total_timesteps, callback=callback)
            print("PPO model training finished.")
        except Exception as e:
            print(f"Error during PPO training: {e}")
            # Decide how to handle errors, maybe return None or raise
            raise e
        finally:
             # Important: Close the vectorized environment
             norm_env.close()

        # 初始化模型

        # Store the state dict of the last trained policy
        self.last_policy_state_dict = model.policy.state_dict()

        # Return the trained policy and the VecNormalize wrapper
        return model.policy, norm_env

class lstmPPOOracle:
    """
    Oracle using RecurrentPPO from SB3 Contrib.
    Trains the policy on a vectorized, normalized environment with dynamic opponent sampling.
    """
    def __init__(self, env_fn: Callable[[], Any], opponent_sampler_fn: Callable[[], BasePolicy], **ppo_config: Any):
        """
        Args:
            env_fn: Function to create an instance of the base game environment.
            opponent_sampler_fn: Function that samples an opponent policy from the meta-strategy.
            **ppo_config: Configuration dictionary for the RecurrentPPO model. Must include 'n_envs'.
        """
        self.env_fn = env_fn
        self.opponent_sampler_fn = opponent_sampler_fn
        self.ppo_config = ppo_config
        self.last_policy_state_dict = None # Store state_dict
        self.log_dir = self.ppo_config.get("tensorboard_log", None)
        self.n_envs = self.ppo_config.get("n_envs")
        if self.n_envs is None:
            raise ValueError("LSTMPPOOracle config must include 'n_envs'")

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

    def train(self,
              start_policy_state_dict: Optional[dict] = None,
              start_vec_normalize_path: Optional[str] = None,
              total_timesteps: int = 2_500_000
             ) -> Tuple[BasePolicy, VecNormalize]:
        """
        Trains a RecurrentPPO best response policy.

        Args:
            start_policy_state_dict: State dictionary of a policy to initialize from (optional).
            start_vec_normalize_path: Path to saved VecNormalize statistics (optional).
            total_timesteps: Total training steps.

        Returns:
            Tuple containing the trained policy object and the VecNormalize wrapper instance.
        """
        print(f"Starting LSTM PPO Oracle training for {total_timesteps} timesteps...")

        # 传进来的 env_fn 已经是经过 RecurrentSelfPlayEnv 包装的了，直接用就行
        vec_env = DummyVecEnv([lambda: self.env_fn(self.opponent_sampler_fn) for _ in range(self.n_envs)])

        # 2. Setup or load VecNormalize
        if start_vec_normalize_path and os.path.exists(start_vec_normalize_path):
            print(f"Loading VecNormalize statistics from: {start_vec_normalize_path}")
            norm_env = VecNormalize.load(start_vec_normalize_path, vec_env)
            norm_env.training = True # Continue updating stats
            norm_env.norm_reward = self.ppo_config.get("norm_reward", False)
        else:
            print("Initializing new VecNormalize statistics.")
            norm_env = VecNormalize(vec_env,
                                    norm_obs=self.ppo_config.get("norm_obs", True),
                                    norm_reward=self.ppo_config.get("norm_reward", False),
                                    clip_obs=self.ppo_config.get("clip_obs", 10.0),
                                    gamma=self.ppo_config.get("gamma", 0.99))

        # # 3. Link VecNormalize back to the underlying environments
        # for env_idx in range(norm_env.num_envs):
        #      if hasattr(norm_env.envs[env_idx], 'env') and hasattr(norm_env.envs[env_idx].env, 'set_vec_normalizer'):
        #          norm_env.envs[env_idx].env.set_vec_normalizer(norm_env)
        #      else:
        #           print(f"Warning: Could not set VecNormalizer for env {env_idx}. Check wrapper structure.")


        # 4. Setup the RecurrentPPO model
        model_ppo_config = self.ppo_config.copy()
        model_ppo_config.pop("n_envs", None)
        model_ppo_config.pop("norm_obs", None)
        model_ppo_config.pop("norm_reward", None)
        model_ppo_config.pop("clip_obs", None)
        model_ppo_config.pop("total_timesteps",None)
        # Remove callback specific params from model config
        model_ppo_config.pop("convergence_window", None)
        model_ppo_config.pop("convergence_mean_change_threshold", None)
        model_ppo_config.pop("convergence_std_threshold", None)
        model_ppo_config.pop("win_rate_threshold", None)

        # Ensure policy_kwargs is correctly formatted if present
        if "policy_kwargs" in model_ppo_config and not isinstance(model_ppo_config["policy_kwargs"], dict):
             print(f"Warning: policy_kwargs in config is not a dict: {model_ppo_config['policy_kwargs']}. Attempting to proceed.")
             # You might need to parse it here depending on its format (e.g., if it's a string)


        model = RecurrentPPO("MlpLstmPolicy", norm_env, **model_ppo_config)

        if start_policy_state_dict:
            print("Initializing RecurrentPPO model from provided state dict.")
            model.policy.load_state_dict(start_policy_state_dict)

        # 5. Setup Callback
        callback = StopTrainingOnEpisodeRewardConvergence(
            check_freq=max(1, self.ppo_config.get("n_steps", 2048) // self.n_envs),
            N=self.ppo_config.get("convergence_window", 50),
            mean_change_threshold=self.ppo_config.get("convergence_mean_change_threshold", 5.0),
            std_threshold=self.ppo_config.get("convergence_std_threshold", None), # Default to None (disabled)
            win_rate_threshold= self.ppo_config.get("win_rate_threshold", 0.8), # Default to 0.8
            verbose=1
        )


        print("Starting RecurrentPPO model learning...")
        try:
            # Note: RecurrentPPO might need reset_num_timesteps=False if continuing training
            # Check SB3 Contrib documentation for best practices on resuming training.
            model.learn(total_timesteps=total_timesteps, callback=callback) #, reset_num_timesteps= (start_policy_state_dict is None) )
            print("RecurrentPPO model training finished.")
        except Exception as e:
            print(f"Error during RecurrentPPO training: {e}")
            raise e
        finally:
            norm_env.close()

        # Store the state dict of the last trained policy
        self.last_policy_state_dict = model.policy.state_dict()

        # Return the trained policy and the VecNormalize wrapper
        return model.policy, norm_env
