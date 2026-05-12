from datetime import datetime
import gymnasium as gym
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import torch
from typing import Any
from tensorboardX import SummaryWriter

from src.envs.gameenv import SatelliteGameEnv
DEVICE =  torch.device(f'cuda:{2}') if torch.cuda.is_available() else 'cpu'



# 自定义回调函数


class CustomCallback(BaseCallback):
    """
    自定义回调：统计每个 episode 的胜率，并在满足阈值时保存策略。
    假设环境在 info 字典中返回 "win" 字段，表示本局比赛是否获胜。
    """
    def __init__(self, verbose=0, window_size = 100, 
                 win_rate_threshold=0.95,
                 save_freq=50000,
                 save_path='./ckpt/base_ppo'):
        super().__init__(verbose)
        self.window_size = window_size
        # 用 deque 来保存最近 window_size 个 episode 的胜负结果（1 表示胜，0 表示负）
        self.win_window = deque(maxlen=window_size)
        self.current_win_rate = 0
        self.win_rate_threshold = win_rate_threshold
        self.temp = 0
        # 新增：定期保存相关参数
        self.save_freq = save_freq
        self.last_save_step = 0
        self.save_path = save_path
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        # self.writer = SummaryWriter(log_dir=save_path)


    def _on_step(self) -> bool:
        # 尝试从当前step的infos中提取比赛结果信息
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("done", False):
                self.win_window.append(1 if info.get("winner", False) else 0)

        # 如果滑动窗口中有数据，则计算胜率
        if len(self.win_window) > 0:
            win_rate = sum(self.win_window) / len(self.win_window)
            self.current_win_rate = win_rate
            if self.verbose and info.get("winner", False):
                print(f"当前滑动窗口胜率（最近 {len(self.win_window)} 场）：{win_rate:.2f}")
                # 新增：记录胜率到 TensorBoard
                # self.writer.add_scalar('win_rate', self.current_win_rate, self.num_timesteps)
                self.logger.record('custom/win_rate', self.current_win_rate)
            # 如果达到预设阈值，则保存当前策略
            if win_rate >= self.win_rate_threshold and len(self.win_window) == self.window_size:
                self.temp += 1
                self.model.save(f'{self.save_path}/{self.temp}.zip')
                self.model.env.save(f'{self.save_path}/vec_normalize_satellite_{self.temp}.pkl')
                # 保存策略后清空滑动窗口以重新统计
                self.win_window.clear()
                return False

        # 新增：检查是否需要定期保存
        current_step = self.num_timesteps
        if current_step - self.last_save_step >= self.save_freq:
            save_path = f'{self.save_path}/checkpoint_{current_step}_steps.zip'
            self.model.save(save_path)
            self.model.env.save(f'{self.save_path}/vec_normalize_satellite_{current_step}.pkl')
            if self.verbose:
                print(f"在第 {current_step} 步保存模型到: {save_path}")
            self.last_save_step = current_step

        # 继续训练
        return True





if __name__ == "__main__":
    env_ = SatelliteGameEnv()

    vec_env = DummyVecEnv([lambda: env_]) # 包装成向量化环境
    # env = LogWrapper(env_)
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, 
                       clip_obs=10.,
                       clip_reward=25,) # 防止极端值

    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    save_path = f'./results/baselines/lstm_ppo_normalize/{current_time}'
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)


    model = RecurrentPPO("MlpLstmPolicy", 
                         env, 
                         verbose=1,
                         n_steps = 4096,
                         batch_size = 512,
                         ent_coef = 0.01,
                         tensorboard_log = f'{save_path}/',
                         policy_kwargs = dict(    
                            net_arch=dict(pi=[128, 128], vf=[128, 128]),
                            ),
                         device= DEVICE,
                         )


    _callback = CustomCallback(verbose=1,
                               save_freq=500000,
                               save_path=save_path,)

    model.learn(total_timesteps=5e12, callback=_callback,
                tb_log_name='lstm_ppo')#,reset_num_timesteps=False)
    
    model.save(f"{save_path}_ppo_recurrent.zip")

    '''以下是测试模型的代码'''



    # def make_env():
    #     return SatelliteGameEnv(shift_steps=3000)

    # env = DummyVecEnv([make_env])

    # vec_env = VecNormalize.load("./results/baselines/lstm_ppo_normalize/20250416-1808/vec_normalize_satellite_500000.pkl", env)
    # # 在测试时，不需要更新归一化参数，所以设置为 False
    # vec_env.training = False
    # vec_env.norm_reward = False
    # model = RecurrentPPO.load("./results/baselines/lstm_ppo_normalize/20250416-1808/checkpoint_500000_steps.zip")
    
    
    # obs = vec_env.reset()
    # # cell and hidden state of the LSTM
    # lstm_states = None
    # num_envs = 1
    # # Episode start signals are used to reset the lstm states
    # episode_starts = np.ones((num_envs,), dtype=bool)
    # dones = False
    # while True:
    #     # action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True) 
    #     # action1 = np.zeros_like(action)
    #     # j_action = np.vstack((action,action1))
    #     # obs, rewards, dones, info = vec_env.step(j_action)

    #     oppo_obs = vec_env.env_method('get_opponent_obs')
    #     norm_oppo_obs = vec_env.normalize_obs(oppo_obs)
    #     action, lstm_states = model.predict(norm_oppo_obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    #     action1 = np.zeros_like(action)
    #     j_action = np.vstack((action1,action))
    #     o = vec_env.env_method('step',action=j_action)
    #     o = o[0]
    #     episode_starts = o[2] or o[3]

    #     print(o[4])
    #     if episode_starts:
    #         break
