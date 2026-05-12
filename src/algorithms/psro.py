import os
import time
import numpy as np
import random
import torch
from typing import Dict, List, Tuple, Any, Callable, Optional
from collections import Counter

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_device

from src.utils.payoff_table import PayoffTable
from src.utils.meta_solver import fictitious_play, uniform_sample, naive_sp, dpp_driven_nash

# Import the updated Oracle classes and the environment wrapper
from src.algorithms.ppo_oracle import PPOOracle, lstmPPOOracle
# Assuming RecurrentSelfPlayEnv is the standard wrapper now
from src.envs.self_play_wrapper import RecurrentSelfPlayEnv




class RandomPolicy(BasePolicy):
    """A simple policy that samples random actions."""
    def __init__(self, observation_space, action_space):
        # super().__init__(observation_space, action_space) # BasePolicy __init__ might need more args
        # Minimal init for compatibility
        super().__init__(observation_space, action_space, optimizer_class=torch.optim.Adam)
        # self.device = 'cpu' # Set device explicitly

    def forward(self, obs, deterministic=False):
        # Not used for prediction, but needs to be implemented
        pass

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Sample random actions
        # Note: SB3 policies expect torch tensors, but sampling might be easier with gym space
        # This implementation might need adjustment based on how BasePolicy expects _predict
        # For simplicity, we'll handle sampling in predict directly
        raise NotImplementedError("Use predict method directly for RandomPolicy")

    def predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False,
               ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # Sample a random action from the action space
        action = self.action_space.sample()
        return action, state # Return action and None state

    # Add dummy methods if needed by other parts of SB3 or PSRO logic
    def set_training_mode(self, mode: bool) -> None:
        pass

    def state_dict(self):
        return {} # No learnable parameters

    def load_state_dict(self, state_dict):
        pass # No parameters to load

    # Add lstm_states attribute for type checking consistency if needed
    @property
    def lstm_states(self):
         return None # Indicate non-recurrent


solver_map = {

    "fictitious_play": fictitious_play,
    "uniform": uniform_sample,
    "last_only": naive_sp,
    "dpp_driven_nash": dpp_driven_nash

}

class BasePSRO:
    def __init__(self,
                 env_fn: Callable[[], Any],
                 oracle_class: type, # PPOOracle or LSTMPPOOracle
                 policy_class: type, # PPO or RecurrentPPO
                 oracle_config: Dict[str, Any],
                 meta_solver_config: Dict[str, Any],
                 save_dir: str,
                 max_pool_size: int = 10):
        """
        Args:
            env_fn: 创建基础环境实例的函数
            oracle_class: PPOOracle 或 LSTMPPOOracle
            policy_class: PPO 或 RecurrentPPO
            oracle_config: oracle训练参数
            meta_solver_config: Meta solver配置
            save_dir: 策略保存路径
            max_pool_size: 最大策略池大小
        """

        self.env_fn = env_fn
        self.policy_class = policy_class # Store policy class for instantiation
        self.oracle_config = oracle_config
        self.meta_solver_config = meta_solver_config
        self.save_dir = save_dir
        self.max_pool_size = max_pool_size
        self.device = get_device(oracle_config.get("device", "auto"))
        os.makedirs(save_dir, exist_ok=True)

        # 初始化oracle，传入对手采样函数
        self.oracle = oracle_class(
            env_fn=env_fn,
            opponent_sampler_fn=self._sample_opponent_policy,
            **oracle_config
        )
        # 初始化meta-solver
        # 使用映射表选择对应求解器
        solver_type = self.meta_solver_config.pop("type", "min_support_nash")
        print(f"init {solver_type} as meta-solver")
        self.solver = solver_map.get(solver_type, min_support_nash)


        # 初始化payoff表，策略池，环境管理
        self.payoff_table = PayoffTable()
        self.strategy_pool : Dict[str, BasePolicy] = {}
        self.saved_policy_paths: Dict[str, str] = {}
        self.vec_normalize_paths: Dict[str, str] = {}
        self.policy_names: List[str] = []
        self.policy_last_access: Dict[str, float] = {}


        # Add initial random policy if pool is empty
        if not self.policy_names:
            print("Initializing policy pool with random policy.")
            # Create a dummy env instance to get spaces
            dummy_env = self.env_fn()
            random_policy = RandomPolicy(dummy_env.observation_space, dummy_env.action_space)
            dummy_env.close()
            self._add_policy_to_pool("random", random_policy, None) # No VecNormalize for random
        # 加载已存在的策略
        self._load_existing_policies()
        self._iterations = len(self.policy_names)-1
        self._current_meta_nash = None
        
    def _get_policy_save_paths(self, policy_name: str) -> Tuple[str, str]:
        """Generates the save paths for a policy and its VecNormalize stats."""
        policy_path = os.path.join(self.save_dir, f"{policy_name}.pt")
        vec_norm_path = os.path.join(self.save_dir, f"{policy_name}_vecnorm.pkl")
        return policy_path, vec_norm_path

    def _load_existing_policies(self):
        """Loads existing policies and VecNormalize stats from the save directory."""
        print(f"Scanning for existing policies in: {self.save_dir}")
        found_policies = {}
        # Find all policy state_dict files
        for filename in os.listdir(self.save_dir):
            if filename.startswith("policy_") and filename.endswith(".pt"):
                policy_name = filename[:-3] # e.g., "policy_0"
                # Extract iteration number
                try:
                    iteration = int(policy_name.split('_')[-1])
                    policy_path = os.path.join(self.save_dir, filename)
                    vec_norm_path = os.path.join(self.save_dir, f"{policy_name}_vecnorm.pkl")

                    if os.path.exists(vec_norm_path):
                        found_policies[iteration] = (policy_name, policy_path, vec_norm_path)
                    else:
                        print(f"Warning: Found policy file {policy_path} but missing VecNormalize file {vec_norm_path}. Skipping.")
                except ValueError:
                    print(f"Warning: Could not parse iteration number from filename {filename}. Skipping.")

        # Sort by iteration number and add to the pool state
        sorted_iterations = sorted(found_policies.keys())
        for iteration in sorted_iterations:
            policy_name, policy_path, vec_norm_path = found_policies[iteration]
            if policy_name not in self.policy_names:
                 self.policy_names.append(policy_name)
                 self.saved_policy_paths[policy_name] = policy_path
                 self.vec_normalize_paths[policy_name] = vec_norm_path
                 print(f"Registered existing policy '{policy_name}' (paths only).")

            # Load the policy into memory
            policy = self._load_policy_if_needed(policy_name)
            self.oracle.last_policy_state_dict = policy.state_dict() if policy else None
            if policy is not None:
                print(f"Policy '{policy_name}' loaded into memory during initialization.")
            else:
                print(f"Warning: Could not load policy '{policy_name}' into memory during initialization.")


        print(f"Found {len(self.policy_names)} existing policy records.")
        # Policies are loaded into memory only when needed (_load_policy_if_needed)

        # for filename in os.listdir(self.save_dir):
        #     if filename.endswith("payoff.npy"):
        #         payoff_name = filename[:-4]
        #         if len(self.policy_names) == int(payoff_name.split('_')[0]):
        #             try:
        #                 payoff_path = os.path.join(self.save_dir, filename)
        #                 self.payoff_table.refresh_table(np.load(payoff_path))
        #             except ValueError:
        #                 print(f"Warning: Could not load payoff from filename {filename}. Skipping.")                 


        # Policies are loaded into memory only when needed (_load_policy_if_needed)

        # 1. 扫描并找到最新的（尺寸最大的） payoff 矩阵文件
        latest_payoff_size = -1
        latest_payoff_filename = None

        for filename in os.listdir(self.save_dir):
            if filename.endswith("payoff.npy"):
                try:
                    payoff_size = int(filename.split('_')[0])
                    if payoff_size > latest_payoff_size:
                        latest_payoff_size = payoff_size
                        latest_payoff_filename = filename
                except ValueError:
                    continue # 忽略命名不符合预期格式的文件

        # 2. 如果找到了 payoff 矩阵，则加载最大的一份
        if latest_payoff_filename is not None:
            try:
                payoff_path = os.path.join(self.save_dir, latest_payoff_filename)
                loaded_payoff = np.load(payoff_path)
                self.payoff_table.refresh_table(loaded_payoff)
                print(f"Loaded latest payoff matrix from {latest_payoff_filename} (size: {latest_payoff_size}x{latest_payoff_size}).")
                
                # 如果 payoff 矩阵小于当前策略数量，打印提示信息
                if latest_payoff_size < len(self.policy_names):
                    missing_count = len(self.policy_names) - latest_payoff_size
                    print(f"Note: Payoff matrix size ({latest_payoff_size}) is smaller than the number of policies ({len(self.policy_names)}). "
                          f"Need to compute empirical returns for {missing_count} newly added policies.")
            except Exception as e:
                print(f"Warning: Could not load payoff from filename {latest_payoff_filename}. Error: {e}. Skipping.")
        else:
            print("No existing payoff matrix found. Starting with an empty payoff table.")


    def _add_policy_to_pool(self, policy_name: str, policy: BasePolicy, norm_env: Optional[VecNormalize]):
        """Adds a policy and its VecNormalize stats to the tracking dictionaries and saves them."""
        if policy_name == "random":
             if "random" not in self.policy_names:
                 self.policy_names.append("random")
             self.strategy_pool["random"] = policy # Keep random in memory
             print("Added random policy to pool.")
             return

        # Add policy name to the list if it's new
        if policy_name not in self.policy_names:
            self.policy_names.append(policy_name)

        # Save policy state_dict and VecNormalize stats
        policy_path, vec_norm_path = self._get_policy_save_paths(policy_name)
        try:
            torch.save(policy.state_dict(), policy_path)
            self.saved_policy_paths[policy_name] = policy_path
            print(f"Policy '{policy_name}' state_dict saved to {policy_path}")



            if norm_env is not None:
                norm_env.save(vec_norm_path)
                self.vec_normalize_paths[policy_name] = vec_norm_path
                print(f"VecNormalize stats for '{policy_name}' saved to {vec_norm_path}")
            else:
                 print(f"Warning: No VecNormalize instance provided for policy '{policy_name}'. Stats not saved.")

        except Exception as e:
            print(f"Error saving policy '{policy_name}' or its VecNormalize stats: {e}")
            # Consider removing the policy entry if saving failed
            if policy_name in self.policy_names: self.policy_names.remove(policy_name)
            if policy_name in self.saved_policy_paths: del self.saved_policy_paths[policy_name]
            if policy_name in self.vec_normalize_paths: del self.vec_normalize_paths[policy_name]
            return # Stop if saving failed

        # Add the actual policy object to the in-memory pool
        self.strategy_pool[policy_name] = policy
        self._touch_policy(policy_name)

        # Manage in-memory pool size
        self._manage_memory_pool()
        self._monitor_memory_usage()   

    def _manage_memory_pool(self):
        """Keeps the number of policies in memory under control using LRU."""
        non_random_in_memory = [name for name in self.strategy_pool if name != "random"]
        num_to_remove = len(non_random_in_memory) - self.max_pool_size

        if num_to_remove > 0:
            # Remove least recently used non-random policies first
            sorted_by_access = sorted(
                non_random_in_memory,
                key=lambda name: self.policy_last_access.get(name, 0)
            )
            keys_to_remove_from_memory = sorted_by_access[:num_to_remove]

            for policy_key in keys_to_remove_from_memory:
                if policy_key in self.strategy_pool:
                    del self.strategy_pool[policy_key]
                if policy_key in self.policy_last_access:
                    del self.policy_last_access[policy_key]
                print(f"Policy '{policy_key}' unloaded from memory.")


    def _touch_policy(self, policy_name: str):
        """Mark policy as recently used for LRU cache management."""
        if policy_name and policy_name != "random":
            self.policy_last_access[policy_name] = time.time()


    def _load_policy_if_needed(self, policy_name: str) -> Optional[BasePolicy]:
        """Loads a policy into memory if it's not already there."""
        if policy_name in self.strategy_pool:
            # print(f"Policy '{policy_name}' found in memory.")
            self._touch_policy(policy_name)
            return self.strategy_pool[policy_name]

        if policy_name in self.saved_policy_paths:
            policy_path = self.saved_policy_paths[policy_name]
            print(f"Loading policy '{policy_name}' from {policy_path}...")

            try:
                # 1. Instantiate the correct model type (PPO or RecurrentPPO)
                #    We need a dummy environment and config, but won't train it.
                #    Use minimal config for instantiation.
                dummy_env = self.env_fn() # Create a temporary env instance
                # Use a minimal config, ensuring device is set
                minimal_config = {"device": self.device, "policy_kwargs": self.oracle_config.get("policy_kwargs", None)}
                if self.policy_class == RecurrentPPO:
                     # RecurrentPPO needs an environment that adheres to VecEnv interface for instantiation
                     temp_vec_env = DummyVecEnv([self.env_fn])
                     model_instance = self.policy_class("MlpLstmPolicy", temp_vec_env, **minimal_config)
                     temp_vec_env.close() # Close the temporary vec env
                else: # Standard PPO
                     model_instance = self.policy_class("MlpPolicy", dummy_env, **minimal_config)

                dummy_env.close() # Close the temporary env instance

                # 2. Load the state dictionary
                state_dict = torch.load(policy_path, map_location=self.device)
                model_instance.policy.load_state_dict(state_dict)
                loaded_policy = model_instance.policy
                loaded_policy.eval() # Set to evaluation mode

                # Add to memory pool and manage size
                self.strategy_pool[policy_name] = loaded_policy
                self._touch_policy(policy_name)
                self._manage_memory_pool()
                print(f"Policy '{policy_name}' loaded successfully.")
                return loaded_policy

            except Exception as e:
                print(f"Error loading policy '{policy_name}' from disk: {e}")
                return None
        else:
            print(f"Error: Policy '{policy_name}' not found in saved paths.")
            return None
        

    def _load_vecnormalize(self, policy_name: str, vec_env: DummyVecEnv) -> Optional[VecNormalize]:
         """Loads VecNormalize stats for a given policy name."""
         if policy_name in self.vec_normalize_paths:
              path = self.vec_normalize_paths[policy_name]
              if os.path.exists(path):
                   try:
                        print(f"Loading VecNormalize for '{policy_name}' from {path}")
                        # Load stats into a *new* VecNormalize instance wrapping the provided vec_env
                        norm_env = VecNormalize.load(path, vec_env)
                        norm_env.training = False # Set to evaluation mode
                        norm_env.norm_reward = False # Don't normalize rewards during evaluation
                        return norm_env
                   except Exception as e:
                        print(f"Error loading VecNormalize from {path}: {e}")
                        return None
              else:
                   print(f"Warning: VecNormalize file not found at {path} for policy '{policy_name}'.")
                   return None
         else:
              # print(f"No VecNormalize path found for policy '{policy_name}'.")
              return None # No stats saved for this policy (e.g., random)


    def _monitor_memory_usage(self):
        """Monitors GPU and potentially general memory usage (basic)."""
        if torch.cuda.is_available() and self.device != 'cpu':
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3) # More accurate total usage
            print(f"GPU Memory ({self.device}): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        # Can add general RAM monitoring using psutil if needed

    def solve_meta_game(self) -> np.ndarray:
        """Solves the meta-game using the current payoff table."""

        if self.payoff_table.table.shape[0] < 1:
            # Handle empty payoff table (e.g., only random policy exists)
            print("Payoff table is empty, finding out the reason...")
            if len(self.policy_names) >1:
                self.update_payoff_table(10)
                print("Payoff table updated, retrying meta-game solving...")
            else:
                print("ONLY RANDOM POLICY EXISTS, RETURNING UNIFORM DISTRIBUTION.")
                return np.ones(len(self.policy_names)) / len(self.policy_names)

        print("Solving meta-game...")
        try:
            # Ensure payoff table matches the number of policies
            if self.payoff_table.table.shape[0] != len(self.policy_names):
                # print(f"Warning: Payoff table size ({self.payoff_table.table.shape[0]}) mismatch with policy count ({len(self.policy_names)}). Recalculate payoff table.")
                # # This might indicate an issue, consider recomputing payoffs if size mismatches occur

                print(f"Notice: Payoff table size ({self.payoff_table.table.shape[0]}) mismatch with policy count ({len(self.policy_names)}). "
                    f"Computing missing matchups...")
                # 【关键修改】：去掉了 self.payoff_table = PayoffTable()，保留已加载的数据
                # 直接调用 update_payoff_table，它会利用 _idx 自动从断点处接着算

                # self.payoff_table = PayoffTable()
                self.update_payoff_table(10)


            nash_weights = self.solver(self.payoff_table, **self.meta_solver_config)



            # Use the chosen meta-solver (e.g., fictitious_play)
            # nash_weights = fictitious_play(self.payoff_table, **self.meta_solver_config)
            # nash_weights = min_support_nash(self.payoff_table, **self.meta_solver_config)

            print(f"Meta-Nash weights: {np.round(nash_weights, 3)}")
            return nash_weights
        except Exception as e:
            print(f"Error solving meta-game: {e}. Returning uniform distribution.")
            num_policies = len(self.policy_names)
            return np.ones(num_policies) / num_policies


    def _sample_opponent_policy(self) -> BasePolicy:
        """Samples an opponent policy based on the current meta-Nash equilibrium."""
        meta_nash = self._current_meta_nash if self._current_meta_nash is not None else self.solve_meta_game()

        # Ensure meta_nash length matches policy_names length
        if len(meta_nash) != len(self.policy_names):
            print(f"Warning: Meta-Nash length ({len(meta_nash)}) doesn't match policy names ({len(self.policy_names)}). Using uniform.")
            sampled_index = np.random.choice(len(self.policy_names))
        else:
            # Sample an index based on the Nash distribution
             try:
                #  sampled_index = np.random.choice(len(self.policy_names), p=meta_nash)
                 sampled_index = np.random.default_rng().choice(len(self.policy_names), p=meta_nash) # Use new numpy random generator
             except ValueError as e:
                  print(f"Error sampling from meta_nash (weights might not sum to 1 or contain negatives): {e}. Correct meta_nash sum to 1.")
                  # Fallback to uniform sampling
                  meta_nash_corrected = np.abs(meta_nash) # Ensure non-negative
                  meta_nash_corrected /= meta_nash_corrected.sum() # Ensure sums to 1
                  self._current_meta_nash = meta_nash_corrected
                  if len(meta_nash_corrected) == len(self.policy_names):
                       sampled_index = np.random.choice(len(self.policy_names), p=meta_nash_corrected)
                  else: # Fallback if length still mismatch after correction attempt
                       sampled_index = np.random.choice(len(self.policy_names))


        opponent_name = self.policy_names[sampled_index]
        print(f"Sampled opponent: '{opponent_name}' (Index: {sampled_index})")

        # Load the policy into memory if needed
        opponent_policy = self._load_policy_if_needed(opponent_name)

        if opponent_policy is None:
            print(f"Warning: Failed to load sampled opponent '{opponent_name}'. Falling back to random policy.")
            return self.strategy_pool["random"], None # Fallback to random

        opponent_env = self._load_vecnormalize(opponent_name, DummyVecEnv([self.env_fn]))   

        return opponent_policy, opponent_env


    def train_best_response(self, reuse_last: bool = True) -> Tuple[str, BasePolicy, Optional[VecNormalize]]:
        """
        Trains a new best response policy using the oracle.

        Args:
            reuse_last: Whether to initialize training from the last trained policy's weights.

        Returns:
            Tuple: (new_policy_name, trained_policy, training_vec_normalize_instance)
                   Returns None for policy and norm_env if training fails.
        """
        print("-" * 30)
        print(f"Training Best Response Policy #{self._iterations}")
        print("-" * 30)

        self._current_meta_nash = self.solve_meta_game()
        
        start_state_dict = None
        start_norm_path = None
        policy_to_resume = None # Name of the policy whose state_dict and norm_stats to use

        # Determine which policy and norm stats to potentially resume from
        if reuse_last and self.oracle.last_policy_state_dict:
             # If the oracle remembers the last state_dict, use that
             start_state_dict = self.oracle.last_policy_state_dict
             # Try to find the corresponding norm stats from the *last added* policy
             if self.policy_names and self.policy_names[-1] != "random":
                  last_policy_name = self.policy_names[-1]
                  if last_policy_name in self.vec_normalize_paths:
                       start_norm_path = self.vec_normalize_paths[last_policy_name]
                       print(f"Resuming BR training using last oracle state_dict and VecNormalize stats from '{last_policy_name}'.")
                  else:
                       print("Resuming BR training using last oracle state_dict, but no matching VecNormalize stats found. Initializing new stats.")
             else:
                  print("Resuming BR training using last oracle state_dict. Initializing new VecNormalize stats.")

        # Train using the oracle
        try:
            # The oracle now internally handles opponent sampling via _sample_opponent_policy
            trained_policy, norm_env = self.oracle.train(
                start_policy_state_dict=start_state_dict,
                start_vec_normalize_path=start_norm_path,
                total_timesteps=min(\
                    self.oracle_config.get("total_timesteps", 500_000) * (1+int(len(self.policy_names)/5)),\
                        5_000_000)  # Get timesteps from config
            )
        except Exception as e:
            print(f"Oracle training failed: {e}")
            return None, None, None # Indicate failure

        # Generate name for the new policy
        new_policy_name = f"policy_{self._iterations}"

        # Add the newly trained policy and its norm_env to the pool (this also saves them)
        self._add_policy_to_pool(new_policy_name, trained_policy, norm_env)

        return new_policy_name, trained_policy, norm_env


    def update_payoff_table(self, episodes_per_matchup: int = 5):
        """Updates the payoff table by evaluating policy matchups."""
        _idx = self.payoff_table.table.shape[0]
        num_policies = len(self.policy_names)

        if _idx == num_policies:
            print("Payoff table is already up-to-date. Skipping update.")
            return
        
        print(f"\nUpdating Payoff Table from ({_idx}x{_idx}) to ({num_policies}x{num_policies}) using {episodes_per_matchup} episodes per matchup...")


        # Expand payoff table if new policies were added
        if _idx < num_policies:
            self.payoff_table.expand(num_policies)
            print(f"Expanded payoff table to size {num_policies}x{num_policies}")


        # Create a temporary VecEnv for evaluation (can use 1 env for simplicity)
        # We need this to load VecNormalize stats into
        eval_vec_env = DummyVecEnv([self.env_fn])

        # Iterate through all unique pairs of policies (including random)
        for i in range(_idx, num_policies):
            name_i = self.policy_names[i]
            for j in range(i): # Evaluate (i, j) including self-play (i==j) if needed, or use range(i+1, num_policies)
                name_j = self.policy_names[j]
                # Skip if payoff already exists

                print(f"  Evaluating matchup: '{name_i}' vs '{name_j}'...")

                # Load policies
                policy_i = self._load_policy_if_needed(name_i)
                policy_j = self._load_policy_if_needed(name_j)

                if policy_i is None or policy_j is None:
                    print(f"  Skipping matchup ({name_i} vs {name_j}) due to policy loading error.")
                    continue

                # Load VecNormalize stats (use stats from policy_i, if available)
                # The calculate_payoff method needs a VecNormalize instance
                norm_env_for_eval_i = self._load_vecnormalize(name_i, eval_vec_env)
                norm_env_for_eval_j = self._load_vecnormalize(name_j, eval_vec_env)

                # Get the underlying evaluation environment instance from the DummyVecEnv
                # This assumes RecurrentSelfPlayEnv is used and has calculate_payoff
                eval_env_instance = eval_vec_env.envs[0]
                if not hasattr(eval_env_instance, 'calculate_payoff'):
                     # If using a different wrapper, adjust access or method name
                     if hasattr(eval_env_instance, 'env') and hasattr(eval_env_instance.env, 'calculate_payoff'):
                          eval_env_instance = eval_env_instance.env # Access wrapped env (e.g., Monitor -> RecurrentSelfPlayEnv)
                     else:
                          print(f"Error: Evaluation environment instance does not have 'calculate_payoff' method.")
                          continue # Skip matchup


                # Calculate average payoff over multiple episodes
                try:
                    # Pass the loaded VecNormalize instance for consistent normalization during eval
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
                    self.payoff_table.update(i, j, payoff_dict) # Updates (i,j) and (j,i)
                    print(f"    Payoff[{name_i}, {name_j}] = {payoff_dict[1]:.3f}") # Print payoff for player i

                except Exception as e:
                     print(f"  Error calculating payoff for '{name_i}' vs '{name_j}': {e}")


                # Clean up loaded VecNormalize instance's internal env reference if necessary
                if norm_env_for_eval_i:
                    norm_env_for_eval_i.close() # Close the temporary VecNormalize wrapper
                if norm_env_for_eval_j:
                    norm_env_for_eval_j.close()


        # Close the temporary evaluation VecEnv
        eval_vec_env.close()
        # save payoff matrix
        payoff_path = os.path.join(self.save_dir, f"{num_policies}_payoff.npy")
        np.save(payoff_path, self.payoff_table.table)
        print("Payoff table update complete.")
        print("Current Payoff Table:")
        print(np.round(self.payoff_table.table, 3))


    def run(self, iterations: int = 100, 
            reuse_policy: bool = True, 
            payoff_update_freq: int = 1, 
            episodes_per_matchup: int = 10):
        """
        Main PSRO training loop.

        Args:
            iterations: Total number of PSRO iterations (new policies to generate).
            reuse_policy: Whether to initialize new BR training from the last trained policy.
            payoff_update_freq: How often (in iterations) to update the payoff table.
            episodes_per_matchup: Number of episodes to run for each matchup in payoff calculation.
        """
        start_iteration = self._iterations # Iteration count starts from existing policies

        for i in range(start_iteration, start_iteration + iterations):
            self._iterations += 1 # Increment iteration counter (1-based for policy naming)
            print(f"\n===== PSRO Iteration {self._iterations} =====")

            # 1. Train Best Response Policy
            # The oracle internally uses _sample_opponent_policy based on the latest meta-nash
            new_policy_name, trained_policy, norm_env = self.train_best_response(reuse_last=reuse_policy)

            if new_policy_name is None:
                 print("Stopping PSRO run due to oracle training failure.")
                 break

            # 2. Update Payoff Table (periodically or every iteration)
            if self._iterations % payoff_update_freq == 0 or i == start_iteration + iterations - 1:
                 self.update_payoff_table(episodes_per_matchup=episodes_per_matchup)
            else:
                 print("Skipping payoff table update this iteration.")

            # 3. Meta-strategy (Nash equilibrium) is implicitly updated in _sample_opponent_policy
            #    when the oracle calls it for the *next* iteration's training.
            #    We can optionally solve and print it here for monitoring.
            if self._iterations % 5 == 0: # Print meta-nash periodically
                 print("\n--- Meta-Strategy Check ---")

                 print(f"Current Meta-Nash: {np.round(self._current_meta_nash, 3)}")
                 print("-" * 25)

            # Memory monitoring
            self._monitor_memory_usage()

        print("\nPSRO run finished.")



# --- Concrete PSRO Classes ---

class BasicPSRO(BasePSRO):
    """PSRO using standard PPO."""
    def __init__(self, env_fn, oracle_config, meta_solver_config, save_dir, max_pool_size=10):
        super().__init__(
            env_fn=env_fn,
            oracle_class=PPOOracle,
            policy_class=PPO, # Specify PPO model class
            oracle_config=oracle_config,
            meta_solver_config=meta_solver_config,
            save_dir=save_dir,
            max_pool_size=max_pool_size
        )
        print("Initialized BasicPSRO (using PPO Oracle).")

class LstmPSRO(BasePSRO):
     """PSRO using RecurrentPPO."""
     def __init__(self, env_fn, oracle_config, meta_solver_config, save_dir, max_pool_size=10):
          super().__init__(
               env_fn=env_fn,
               oracle_class=lstmPPOOracle,
               policy_class=RecurrentPPO, # Specify RecurrentPPO model class
               oracle_config=oracle_config,
               meta_solver_config=meta_solver_config,
               save_dir=save_dir,
               max_pool_size=max_pool_size
          )
          print("Initialized LstmPSRO (using RecurrentPPO Oracle).")