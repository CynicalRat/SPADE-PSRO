import copy
import gymnasium as gym
from .gameenv import SatelliteGameEnv,UPPER_BOUND_T, LOWER_BOUND_T, WINNING_STEP
import numpy as np


class SelfPlayWrapper(gym.Env):
    def __init__(self, env:SatelliteGameEnv):
        self.env = env

        # 添加必要属性，使其兼容 SB3 的 VecEnv 接口
        self.num_envs = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.opponent_policy = None

    def reset(self,        
            **kwargs):
        # 重置环境，返回初始观测
        obs, i = self.env.reset(**kwargs)
        obs[0:3] = np.sign(obs[0:3]) * np.log(np.abs(obs[0:3]) + 1)
        return obs, i

    def set_opponent_policy(self, policy):
        self.opponent_policy = policy

    def step(self, agent_action):
        # 从策略池中采样一个对手策略
        # opponent_policy = self.opponent_selector()
        # 根据当前环境状态获取对手观测（这里假设 env 内部能获取对手状态，具体根据你的环境调整）
        assert self.opponent_policy is not None, "opponent policy is not set"
        opponent_obs = self.get_opponent_obs()  # 需要自行实现或从 env 中获取

        opponent_action, _ = self.opponent_policy.predict(opponent_obs, deterministic=True)
        # 将双方动作组合，传入环境
        joint_action = np.vstack((agent_action, opponent_action))

        obs, r, d1, d2, i =self.env.step(joint_action)
        obs[0:3] = np.sign(obs[0:3]) * np.log(np.abs(obs[0:3]) + 1)
        return obs, r, d1, d2, i



    def get_opponent_obs(self):
        obs = self.env.obss[self.env.agent_2]
        obs[0:3] = np.sign(obs[0:3]) * np.log(np.abs(obs[0:3]) + 1)
        return obs

    def get_opponent_reward(self):
        return self.env.rewards[self.env.agent_2]
    
    # 为了兼容 VecEnv，你可能还需要实现其他方法，如 render, close 等
    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def evaluate(self, agent1, agent2):
        '''旧版payoff计算方法，舍弃不用，改用calculate_payoff'''
        obs, _ = self.reset()
        d1, d2 = False, False
        payoff= 0
        while not d1 or d2:
            opponent_obs = self.get_opponent_obs()  # 需要自行实现或从 env 中获取
            agent_action, _ = agent1.predict(obs, deterministic=True)
            opponent_action, _ = agent2.predict(opponent_obs, deterministic=True)
            # 将双方动作组合，传入环境
            joint_action = np.vstack((agent_action, opponent_action))
            obs, r, d1, d2, i =self.env.step(joint_action)
            obs[0:3] = np.sign(obs[0:3]) * np.log(np.abs(obs[0:3]) + 1)
            opp_r = self.get_opponent_obs
            payoff += r - opp_r
        return payoff
    

    def calculate_payoff(self, agent1, agent2):
        """获取收益矩阵"""
        payoffs = {}

        obs, _ = self.reset()
        self.reset()
        d1, d2 = False, False
        while not d1 or d2:
            opponent_obs = self.get_opponent_obs()  # 需要自行实现或从 env 中获取
            agent_action, _ = agent1.predict(obs, deterministic=True)
            opponent_action, _ = agent2.predict(opponent_obs, deterministic=True)
            # 将双方动作组合，传入环境
            joint_action = np.vstack((agent_action, opponent_action))
            obs, r, d1, d2, i =self.env.step(joint_action)
            obs[0:3] = np.sign(obs[0:3]) * np.log(np.abs(obs[0:3]) + 1)
        
        
        for agent_id, state in self.env.obss.items():
            distance = np.linalg.norm(state[0:3])
            angle = state[6]  # The solar angle
            distance_quality = state[10]  # Flag1: distance qualification
            angle_quality = state[11]  # Flag2: angle qualification
            
            # Calculate observation quality score (0-100)
            observation_score = 50 * distance_quality + 50 * angle_quality
            
            # Calculate position score based on how close to ideal the final position was
            # Ideal distance is the midpoint of the bounds
            ideal_distance = (UPPER_BOUND_T + LOWER_BOUND_T) / 2
            distance_score = 30 * (1 - abs(distance - ideal_distance) / ideal_distance)
            
            # Angle score - closer to 0 is better
            angle_score = 20 * (1 - abs(angle) / np.pi)  # Assuming 30 degrees is the maximum
            
            # Combined score
            payoffs[agent_id] = observation_score + distance_score + angle_score
        
        # Zero-sum adjustment - ensure payoffs sum to zero
        total = sum(payoffs.values())
        for agent_id in payoffs:
            payoffs[agent_id] -= total / 2
        
        # Bonus for "winning" by maintaining optimal observation conditions
        if self.env.winner is not None:
            win_ratio = (1.0 - self.env.timesteps / self.env.shift_steps)
            payoffs[self.env.winner] += 50 * win_ratio
            
        # Again ensure zero-sum property after win bonuses
        total = sum(payoffs.values())
        for agent_id in payoffs:
            payoffs[agent_id] -= total / 2
            
        return payoffs            
    


class RecurrentSelfPlayEnv(gym.Env):
    """
    Wrap SatelliteGameEnv for RecurrentPPO + VecNormalize + self-play per-reset opponent sampling
    """
    def __init__(self, env_fn, sample_opponent_fn=None):
        super().__init__()
        """
        Args:
            env_fn: A function that returns an instance of the base environment (e.g., SatelliteGameEnv).
            opponent_sampler_fn: A function that takes no arguments and returns an opponent policy object (e.g., a loaded SB3 policy).
                                 This function should implement the logic for sampling from the meta-strategy.
        """
        # underlying env
        self.base_env = env_fn()
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        # sampling function returns a policy object
        self.opponent_sampler = sample_opponent_fn
        # current opponent and its LSTM state
        self.current_opponent = None
        self.opp_states = None
        self.opp_episode_start = None
        self.vec_normalizer = None          # Placeholder for the VecNormalize wrapper instance.
                                            # This will be set externally after the DummyVecEnv is created.

    def set_opponent_sampler(self, sampler_fn):
        """Allows updating the opponent sampling logic (when meta-strategy changes)."""
        self.opponent_sampler = sampler_fn

    def set_vec_normalizer(self, vec_norm_wrapper):
        """Sets the VecNormalize wrapper instance."""
        self.vec_normalizer = vec_norm_wrapper

    def reset(self, **kwargs):
        # On each reset, sample a new opponent from meta-distribution
        if callable(self.opponent_sampler):
            self.current_opponent, self.vec_normalizer = self.opponent_sampler()
            # Ensure a default opponent if sampler fails or is not set initially
            if self.current_opponent is None:
                 print("Warning: Opponent sampler returned None. Falling back to random.")
                 self.current_opponent = self._get_random_policy() # Fallback      
        else:
            # Default to random policy if no sampler is provided
            print("Warning: No opponent sampler function provided. Using random opponent.")
            self.current_opponent = self._get_random_policy()      

        # Reset opponent's LSTM state
        self.opp_states = None
        self.opp_episode_start = True

        # Reset the base environment
        obs, info = self.base_env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # Normalize opponent obs later via VecNormalize wrapper
        if self.current_opponent is None:
            # Safeguard: Sample opponent if missing
            print("Warning: current_opponent is None in step. Sampling new opponent.")
            self.current_opponent = self.opponent_sampler() if callable(self.opponent_sampler) else self._get_random_policy()
            self.opp_states = None
            self.opp_episode_start = True

        opp_obs_raw = self.base_env.get_opponent_obs()
        opp_obs = self.get_normalized_obs(opp_obs_raw)

        #  Get opponent's action
        is_recurrent_opponent = hasattr(self.current_opponent, 'lstm_actor') 

        if is_recurrent_opponent:
            # predict opponent action
            opp_action, self.opp_states = self.current_opponent.predict(
                opp_obs,
                state=self.opp_states,
                episode_start=np.array([self.opp_episode_start]),
                deterministic=True
            )
        else:
            opp_action, _ = self.current_opponent.predict(
                 opp_obs,
                 deterministic=True
             )
            
        joint_action = np.vstack((action, opp_action))
        obs, reward, done, truncated, info = self.base_env.step(joint_action)
        self.opp_episode_start = bool(done or truncated)

        # Add opponent reward to info (optional)
        info['opponent_reward'] = self.base_env.rewards.get(self.base_env.agent_2, 0)

        return obs, reward, done, truncated, info

    def get_normalized_obs(self, raw_obs):
        if self.vec_normalizer \
                and hasattr(self.vec_normalizer, 'obs_rms') \
                and self.vec_normalizer.obs_rms is not None:
            mean = self.vec_normalizer.obs_rms.mean
            var = self.vec_normalizer.obs_rms.var
            eps = self.vec_normalizer.epsilon
            _obs = (raw_obs - mean) / np.sqrt(var + eps)
            return _obs
        else:
            return raw_obs
    
    def get_opponent_obs(self):
        """Gets the raw opponent observation from the base environment."""
        return self.base_env.get_opponent_obs()
    
    def _get_random_policy(self):
        """Creates a simple random policy."""
        class RandomPolicy:
            def __init__(self, action_space):
                self.action_space = action_space

            def predict(self, obs, state=None, episode_start=None, deterministic=False):
                return self.action_space.sample(), None # Return None for state

        return RandomPolicy(self.action_space)    

    def calculate_payoff(self, agent1_policy, agent2_policy, 
                         vec_normalizer_1=None, 
                         vec_normalizer_2=None,
                         episodes=10):
        """
        Calculates the average payoff between two policies over several episodes.
        Handles both standard and recurrent policies.
        Uses VecNormalize for observations if provided.

        Args:
            agent1_policy: The policy object for agent 1.
            agent2_policy: The policy object for agent 2.
            vec_normalizer(1,2): The VecNormalize instance used during training (optional but recommended).
            episodes: Number of episodes to average over.

        Returns:
            A dictionary containing the average payoffs for agent 1 and agent 2.
        """
        total_payoffs = {self.base_env.agent_1: 0.0, self.base_env.agent_2: 0.0}
        # Use a temporary VecNormalize instance for evaluation to avoid altering training stats
        temp_vec_normalizer1, temp_vec_normalizer2 = None, None

        if vec_normalizer_1 and hasattr(vec_normalizer_1, 'obs_rms') and vec_normalizer_1.obs_rms is not None:
            temp_vec_normalizer1 = copy.deepcopy(vec_normalizer_1)
            temp_vec_normalizer1.training = False # Set to eval mode
            temp_vec_normalizer1.norm_reward = False # Don't normalize reward for eval

        if vec_normalizer_2 and hasattr(vec_normalizer_2, 'obs_rms') and vec_normalizer_2.obs_rms is not None:
            temp_vec_normalizer2 = copy.deepcopy(vec_normalizer_2)
            temp_vec_normalizer2.training = False # Set to eval mode
            temp_vec_normalizer2.norm_reward = False # Don't normalize reward for eval

        for ep in range(episodes):      
            obs1_raw, _ = self.base_env.reset()

            agent1_state, agent2_state = None, None
            is_recurrent1 = hasattr(agent1_policy, 'lstm_actor')
            is_recurrent2 = hasattr(agent2_policy, 'lstm_actor')

            done, truncated = False, False
            episode_start = True

            while not (done or truncated):
                # obs1_raw = self.base_env.obss.get(self.base_env.agent_1, obs1_raw)
                obs2_raw = self.base_env.get_opponent_obs()

                # --- Normalize Observations ---
                if temp_vec_normalizer1 is not None:
                    # Ensure obs are numpy arrays before normalizing
                    predict_obs1 = temp_vec_normalizer1.normalize_obs(np.asarray(obs1_raw))
                else:
                    predict_obs1 = np.asarray(obs1_raw)
                if temp_vec_normalizer2 is not None:
                    predict_obs2 = temp_vec_normalizer2.normalize_obs(np.asarray(obs2_raw))
                else:
                    predict_obs2 = np.asarray(obs2_raw)   

                # --- Predict Actions ---
                # Agent 1
                # predict_obs1 = predict_obs1.reshape(1, -1) # Add batch dim
                if is_recurrent1:
                    action1, agent1_state = agent1_policy.predict(
                        predict_obs1,
                        state=agent1_state,
                        episode_start=np.array([episode_start]),
                        deterministic=True
                    )
                else:
                    action1, _ = agent1_policy.predict(predict_obs1, deterministic=True)

                # Agent 2
                # predict_obs2 = predict_obs2.reshape(1, -1) # Add batch dim
                if is_recurrent2:
                    action2, agent2_state = agent2_policy.predict(
                        predict_obs2,
                        state=agent2_state,
                        episode_start=np.array([episode_start]),
                        deterministic=True
                    )
                else:
                    action2, _ = agent2_policy.predict(predict_obs2, deterministic=True)
                # 将双方动作组合，传入环境
                joint_action = np.vstack((action1, action2))
                obs1_raw, _, done, truncated, info = self.base_env.step(joint_action)

                episode_start = False
            
            # --- Episode End: Calculate Payoff based on final state ---
            final_payoffs_ep = {}
            for agent_id, state in self.base_env.obss.items():
                distance = np.linalg.norm(state[0:3])
                angle = state[6]  # The solar angle
                distance_quality = state[10]  # Flag1: distance qualification
                angle_quality = state[11]  # Flag2: angle qualification
                
                # Calculate observation quality score (0-100)
                observation_score = 50 * distance_quality + 50 * angle_quality
                
                # Calculate position score based on how close to ideal the final position was
                # Ideal distance is the midpoint of the bounds
                ideal_distance = (UPPER_BOUND_T + LOWER_BOUND_T) / 2
                distance_score = 25 * (1 - abs(distance - ideal_distance) / ideal_distance)
                
                # Angle score - closer to 0 is better
                angle_score = 25 * (1 - abs(angle) / np.pi) 
                
                # Combined score
                final_payoffs_ep[agent_id] = observation_score + distance_score + angle_score
            
            # Zero-sum adjustment - ensure payoffs sum to zero
            total = sum(final_payoffs_ep.values())
            for agent_id in final_payoffs_ep:
                final_payoffs_ep[agent_id] -= total / 2
            
            # Bonus for "winning" by maintaining optimal observation conditions
            if self.base_env.winner is not None:
                win_ratio = (1.0 - self.base_env.timesteps / self.base_env.shift_steps)
                final_payoffs_ep[self.base_env.winner] += 100 * win_ratio   
                # Again ensure zero-sum property after win bonuses
                total = sum(final_payoffs_ep.values())
                for agent_id in final_payoffs_ep:
                    final_payoffs_ep[agent_id] -= total / 2
            # Accumulate payoffs
            total_payoffs[self.base_env.agent_1] += final_payoffs_ep.get(self.base_env.agent_1, 0)
            total_payoffs[self.base_env.agent_2] += final_payoffs_ep.get(self.base_env.agent_2, 0)

        # Average payoffs over episodes
        avg_payoffs = {k: v / episodes for k, v in total_payoffs.items()}
        return avg_payoffs    
    
    def calculate_payoff_versus_random(self, agent1_policy,
                         vec_normalizer_1=None, 
                         episodes=10):
        """
        Calculates the average payoff between policies and random opponents.
        Handles both standard and recurrent policies.
        Uses VecNormalize for observations if provided.

        Args:
            agent1_policy: The policy object for agent 1.
            vec_normalizer_1: The VecNormalize instance used during training (optional but recommended).
            episodes: Number of episodes to average over.

        Returns:
            A dictionary containing the average payoffs for agent 1 and agent 2.
        """
        total_payoffs = {self.base_env.agent_1: 0.0, self.base_env.agent_2: 0.0}
        # Use a temporary VecNormalize instance for evaluation to avoid altering training stats
        temp_vec_normalizer1, temp_vec_normalizer2 = None, None

        if vec_normalizer_1 and hasattr(vec_normalizer_1, 'obs_rms') and vec_normalizer_1.obs_rms is not None:
            temp_vec_normalizer1 = copy.deepcopy(vec_normalizer_1)
            temp_vec_normalizer1.training = False # Set to eval mode
            temp_vec_normalizer1.norm_reward = False # Don't normalize reward for eval

        for ep in range(episodes):      
            obs1_raw, _ = self.base_env.reset()

            agent1_state = None
            is_recurrent1 = hasattr(agent1_policy, 'lstm_actor')

            done, truncated = False, False
            episode_start = True

            while not (done or truncated):
                # obs1_raw = self.base_env.obss.get(self.base_env.agent_1, obs1_raw)

                # --- Normalize Observations ---
                if temp_vec_normalizer1 is not None:
                    # Ensure obs are numpy arrays before normalizing
                    predict_obs1 = temp_vec_normalizer1.normalize_obs(np.asarray(obs1_raw))
                else:
                    predict_obs1 = np.asarray(obs1_raw)


                # --- Predict Actions ---
                # Agent 1
                # predict_obs1 = predict_obs1.reshape(1, -1) # Add batch dim
                if is_recurrent1:
                    action1, agent1_state = agent1_policy.predict(
                        predict_obs1,
                        state=agent1_state,
                        episode_start=np.array([episode_start]),
                        deterministic=True
                    )
                else:
                    action1, _ = agent1_policy.predict(predict_obs1, deterministic=True)


                # 将双方动作组合，传入环境
                action2 = self.base_env.action_space.sample()
                joint_action = np.vstack((action1, action2))
                obs1_raw, _, done, truncated, info = self.base_env.step(joint_action)

                episode_start = False
            
            # --- Episode End: Calculate Payoff based on final state ---
            final_payoffs_ep = {}
            for agent_id, state in self.base_env.obss.items():
                distance = np.linalg.norm(state[0:3])
                angle = state[6]  # The solar angle
                distance_quality = state[10]  # Flag1: distance qualification
                angle_quality = state[11]  # Flag2: angle qualification
                
                # Calculate observation quality score (0-100)
                observation_score = 50 * distance_quality + 50 * angle_quality
                
                # Calculate position score based on how close to ideal the final position was
                # Ideal distance is the midpoint of the bounds
                ideal_distance = (UPPER_BOUND_T + LOWER_BOUND_T) / 2
                distance_score = 25 * (1 - abs(distance - ideal_distance) / ideal_distance)
                
                # Angle score - closer to 0 is better
                angle_score = 25 * (1 - abs(angle) / np.pi) 
                
                # Combined score
                final_payoffs_ep[agent_id] = observation_score + distance_score + angle_score
            
            # Zero-sum adjustment - ensure payoffs sum to zero
            total = sum(final_payoffs_ep.values())
            for agent_id in final_payoffs_ep:
                final_payoffs_ep[agent_id] -= total / 2
            
            # Bonus for "winning" by maintaining optimal observation conditions
            if self.base_env.winner is not None:
                win_ratio = (1.0 - self.base_env.timesteps / self.base_env.shift_steps)
                final_payoffs_ep[self.base_env.winner] += 100 * win_ratio   
                # Again ensure zero-sum property after win bonuses
                total = sum(final_payoffs_ep.values())
                for agent_id in final_payoffs_ep:
                    final_payoffs_ep[agent_id] -= total / 2
            # Accumulate payoffs
            total_payoffs[self.base_env.agent_1] += final_payoffs_ep.get(self.base_env.agent_1, 0)
            total_payoffs[self.base_env.agent_2] += final_payoffs_ep.get(self.base_env.agent_2, 0)

        # Average payoffs over episodes
        avg_payoffs = {k: v / episodes for k, v in total_payoffs.items()}
        return avg_payoffs    
    

    # Add render and close methods for compatibility
    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def close(self):
        self.base_env.close()