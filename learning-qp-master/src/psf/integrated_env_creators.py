import numpy as np
import torch
from src.envs.linear_system import LinearSystem
from src.envs.cartpole import CartPole
from src.envs.env_creators import env_creators, sys_param
import gym
from icecream import ic
import matplotlib.pyplot as plt
from torch.distributions import Beta
import pandas as pd
import os
from datetime import datetime

class Integrated_env:
    def __init__(self, env_name, **kwargs):
        self.env_name = env_name
        self.env = env_creators[env_name](**kwargs)
        self.bs = self.env.bs
        self.device = self.env.device
        self.train_or_test = kwargs["train_or_test"]
        if env_name == "tank":
            self.u_min = self.env.u_min[0,0].item()  # 仅取值
            self.u_max = self.env.u_max[0,0].item()
        else:
            self.u_min = self.env.u_min.item()  # 仅取值
            self.u_max = self.env.u_max.item()
        self.m = self.env.m
        
        # Gym environment settings
        self.action_space = self.env.action_space
        original_shape = self.env.observation_space.shape
        new_shape = (original_shape[0] + self.m,)    # different
        # 定义新的 observation_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=new_shape,
            dtype=self.env.observation_space.dtype
        )
        # self.observation_space = self.env.observation_space
        self.state_space = self.observation_space
        self.num_states = self.env.num_states + self.env.num_actions    # different
        self.num_actions = self.env.num_actions
        
        # 可视化
        self.xs = []  # 用于记录状态历史
        self.us = []  # 用于记录动作历史
        self.uds = []  # 用于记录动作历史
        self.visualize = False
        if self.bs==1 and self.train_or_test=="test":
            self.visualize = True
            
        # 可选的noise值
        if env_name == "double_integrator":
            self.noise_values = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5]  # 可选的noise值
            self.noise = 0.3 * torch.ones((self.bs, 1), device=self.device)   # 初始默认值
        elif env_name == "cartpole":
            # self.noise_values = [0., 1., 3., 5., 6., 7., 8., 9., 10., 12., 15.]  # 可选的noise值
            self.noise_values = [0.,0., 0.5, 1., 2., 3., 5., 6., 7., 8., 10., 12., 15.]
            self.noise = 5 * torch.ones((self.bs, 1), device=self.device)   # 初始默认值
        elif env_name == "tank":
            self.noise_values = [0., 0., 0.05, 0.1, 0.3, 0.5, 1., 2., 3., 5., 8.]
            self.noise = 0.1 * torch.ones((self.bs, self.m), device=self.device)   # 初始默认值
        
        self.stats = pd.DataFrame(columns=['i', 'cumulative_deviation'])
        self.cum_deviation = torch.zeros((self.bs,), device=self.device)
        self.already_on_stats = torch.zeros((self.bs,), dtype=torch.uint8, device=self.device)   # Each worker can only contribute once to the statistics, to avoid bias towards shorter episodes
        self.run_name = self.env.run_name
        self.step_count = self.env.step_count
        self.ud = torch.zeros((self.bs, self.m), device=self.device)
        self.ud_type = torch.zeros((self.bs, 1), device=self.device, dtype = torch.long)       # 决定训练时使用的ud类型（0代表LQR+噪声，1代表bangbang控制
        self.ud_type_values = torch.zeros((self.bs, 1), device=self.device)       # 决定self.ud_type，方便按概率选择
        
        # for bangbang control
        self.bb_switch_values = [10, 20, 30, 50, 60, 80]
        self.bb_switch = 50 * torch.ones((self.bs, 1), device=self.device, dtype = torch.long)   # 初始默认值
        self.bb_amplitude_values = [-1, 1]
        self.bb_amplitude = torch.ones((self.bs, 1), device=self.device, dtype = torch.long)   # 初始默认值
    
    def select_train_ud(self):
        # 初始化 ud
        ud = torch.zeros((self.bs, self.m), device=self.device)
        
        # LQR+noise
        v = (self.noise * torch.randn((self.bs, self.m), device=self.device))
        ud_lqr = self.env.get_action_LQR(noise_level=0) + v  # 单重噪声
        ud_lqr = ud_lqr.clamp(self.env.u_min, self.env.u_max)
        
        # bangbang control
        condition_bb = (self.step_count.unsqueeze(-1) >= self.bb_switch)
        ud_bb = torch.where(condition_bb, 
                            torch.zeros_like(self.ud),  
                            self.bb_amplitude.double())
        
        # 随机ud
        random_values = torch.rand(self.bs, self.m, device=self.device)  # 形状为 (self.bs, self.m)，值在 [0, 1) 范围内
        ud_random = self.u_min + (self.u_max - self.u_min) * random_values  # 形状为 (self.bs, self.m)
        
        if self.env_name == "tank":
            # 根据 self.ud_type 的值选择不同的行为
            ud = torch.where(self.ud_type == 0, ud_lqr, 
                            torch.where(self.ud_type == 1, ud_bb, ud_random))
        else:
            ud = ud_lqr    
        return ud

    def get_ud(self, obs):
        # x, x_dot, theta, theta_dot, x_ref = self.env.obs()
        # theta = obs[..., 2]
        if self.train_or_test == "train":
            ud = self.select_train_ud()

        elif self.train_or_test == "test":    
            # bang-bang control (使用 torch.where 来向量化条件操作
            # ud = torch.where(theta >= 0.2, torch.full_like(theta, self.u_max), torch.where(theta <= -0.2, torch.full_like(theta, self.u_min), torch.zeros_like(theta)))
            # ud = torch.where((self.step_count <= 50).unsqueeze(1), torch.full_like(self.ud, -1),  torch.zeros_like(self.ud))
            
            # LQR control
            noise = 1
            v = (noise * torch.randn((self.bs, self.m), device=self.device))
            ud = self.env.get_action_LQR(noise_level = 0) + v  # 双重噪声（感觉太难了，先换成单重了。
            ud = ud.clamp(self.env.u_min, self.env.u_max)
            
        # ud = ud.squeeze(-1)
        self.ud = ud
        return ud
    
    def reset(self, *args, **kwargs):
        init_obs = self.env.reset(*args, **kwargs)
        if self.train_or_test == "train":
            init_ud = self.get_ud(init_obs)
        elif self.train_or_test == "test":
            init_ud = self.get_ud(init_obs)
            self.ud = init_ud
        # combined_obs = torch.cat((init_obs, init_ud.unsqueeze(-1)), dim=-1)
        combined_obs = torch.cat((init_obs, init_ud), dim=-1)
        return combined_obs
    
    def reset_done_envs(self, need_reset=None, randomize_seed=None):
        is_done = self.env.is_done.bool() if need_reset is None else need_reset.bool()
        
        # 为结束的环境选择一个新的ud_type
        new_ud_type_values = torch.rand(int(is_done.sum()), device=self.device)
        self.ud_type_values[is_done] = new_ud_type_values.unsqueeze(-1)
        self.ud_type[is_done] = torch.where(
            self.ud_type_values[is_done] < 0.4, 
            0, 
            torch.where(
                self.ud_type_values[is_done] < 0.8, 
                1, 
                2
            )
        )

        # 为结束的环境选择一个新的随机噪声值
        new_noise = torch.tensor(np.random.choice(self.noise_values, size=int(is_done.sum())), device=self.device)        
        # 更新self.noise张量中对应结束环境的噪声值
        self.noise[is_done] = new_noise.unsqueeze(-1)  # 确保维度匹配

        # 为结束的环境选择一个新的bangbang控制参数
        new_bb_switch = torch.tensor(np.random.choice(self.bb_switch_values, size=int(is_done.sum())), device=self.device)        
        self.bb_switch[is_done] = new_bb_switch.unsqueeze(-1)  
        new_bb_amplitude = torch.tensor(np.random.choice(self.bb_amplitude_values, size=int(is_done.sum())), device=self.device)        
        self.bb_amplitude[is_done] = new_bb_amplitude.unsqueeze(-1)

        self.cum_deviation[is_done] = 0
    
    def done(self):
        return self.env.done()
    
    def info(self):
        return self.env.info()
    
    def obs(self):
        # Get the current observation from the environment
        current_obs = self.env.obs()    # 形状是 (batch_size, observation_dim)
        # Concatenate the current observation with the control input ud
        # combined_obs = torch.cat((current_obs, self.ud.unsqueeze(-1)), dim=-1)
        combined_obs = torch.cat((current_obs, self.ud), dim=-1)
        return combined_obs
    
    # def cost(self, *args, **kwargs):
    #     return self.env.cost(*args, **kwargs)
    
    def reward(self,original_reward, action):
        if self.env_name == "double_integrator":
            coef_safety = -120.0
            coef_deviation = -50.0
            coef_survival = 10.0 
            coef_terminate = -1.
            zero_deviation_reward = 10.
            near_zero_deviation = 0
            coef_small_deviation = 0
        elif self.env_name == "cartpole":
            coef_safety = -200.0
            coef_deviation = -20.0
            coef_survival = 100.0  
            coef_terminate = -1000000.
            zero_deviation_reward = 80.
            near_zero_deviation = 0.01
            coef_small_deviation = 600
            # initial
            # coef_safety = -2000.0
            # coef_deviation = 50.0
            # coef_survival = 500.0  
            # coef_terminate = 100000.
            # zero_deviation_reward = 100.
        elif self.env_name == "tank":
            coef_safety = -1000.0
            coef_deviation = -500.0
            coef_survival = 100.0  
            coef_terminate = -1000000.
            # coef_survival = 0.0  
            # coef_terminate = -0.
            zero_deviation_reward = 500.
            near_zero_deviation = 1e-2
            coef_small_deviation = 30000
            
        # safe cost
        original_safe_cost = original_reward
        safe_cost = coef_safety * original_safe_cost
        
        # deviation cost
        # 对差值进行平方操作
        deviation = (self.ud - action).pow(2)
        # 沿着第 1 轴（dim=1）求和，得到形状为 (bs,) 的张量
        deviation = torch.sum(deviation, dim=1)
        self.cum_deviation += deviation
        
        deviation_cost = coef_deviation * deviation
        # 当 deviation 不等于 0 时，计算 deviation_cost；否则，设置为 10
        # deviation_cost = torch.where(deviation == 0, torch.full_like(deviation, zero_deviation_reward), coef_deviation * deviation)
        # deviation_cost = torch.where(deviation <= near_zero_deviation, torch.full_like(deviation, zero_deviation_reward), coef_deviation * deviation)    # 条件放宽

        # 当 deviation 小于 near_zero_deviation 时，geismall_deviation_reward, 奖励越接近 0 越大
        small_deviation_reward = torch.where(
            deviation <= near_zero_deviation,
            zero_deviation_reward - coef_small_deviation * deviation,  # 奖励函数
            torch.zeros_like(deviation)  # 默认奖励为 0
        )
        
        # survival_reward
        survival_reward = coef_survival  # 每个步骤的存活奖励

        # 添加提前terminate惩罚
        terminate_cost = coef_terminate * (self.env.is_done == 1)

        combined_reward = safe_cost + deviation_cost + small_deviation_reward + survival_reward + terminate_cost  # 注意正负号！！！

        if not self.env.quiet:
            avg_safe_cost = safe_cost.float().mean().item()
            avg_deviation_cost = deviation_cost.mean().item()
            avg_terminate_cost = coef_terminate * terminate_cost.mean().item()
            avg_small_deviation_reward = small_deviation_reward.mean().item()
            ic(avg_safe_cost)
            ic(avg_deviation_cost)
            ic(avg_terminate_cost)
            ic(avg_small_deviation_reward)
        if self.train_or_test == "train":
            return combined_reward
        elif self.train_or_test == "test":
            return original_safe_cost      # original safe cost
    
    def step(self, input_action):   
        self.reset_done_envs()
        
        # 用ud进行测试时
        # original_obs, original_reward, done, info = self.env.step(self.ud)   
        # reward = self.reward(original_reward, self.ud)

        action = input_action
        
        # 正常测试
        original_obs, original_reward, done, info = self.env.step(action)   
        reward = self.reward(original_reward, action)
        
        self.step_count = self.env.step_count
        
        if self.env.keep_stats:
            done_indices = torch.nonzero(self.env.is_done.to(dtype=torch.bool) & torch.logical_not(self.already_on_stats), as_tuple=False)
            for i in done_indices:
                self.write_episode_stats(i)
        
        # 可视化
        if self.visualize:
            self.uds.append(self.ud[0, :].cpu().numpy())   # 获取ud
            self.us.append(action[0, :].detach().cpu().numpy())
            self.xs.append(original_obs[0, :].detach().cpu().numpy())
            if done:
                self.visualize_trajectory()
        
        # get next ud
        self.get_ud(original_obs)   # update ud every step
        
        return self.obs(), reward, done, info
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
       
    def write_episode_stats(self, i):
        """
        Logs statistics of the episode for the ith environment in the batch.
        """
        self.already_on_stats[i] = 1
        cumulative_deviation = self.cum_deviation[i].item()
        ic(i)
        ic(cumulative_deviation)
        # 只在最后的时候输出一次avg_deviation
        if i==self.bs-1 :
            avg_deviation = self.cum_deviation.mean().item()
            ic(avg_deviation)
        self.stats.loc[len(self.stats)] = [i.item(),  cumulative_deviation]
        
    def dump_stats(self, filename=None):
        if filename is None:
            directory = 'test_results'
            if not os.path.exists(directory):
                os.makedirs(directory)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tag = self.run_name
            filename = os.path.join(directory, f"{tag}_{timestamp}.csv")
        original_stats = self.env.dump_stats(filename=None)
        combined_stats = pd.merge(original_stats, self.stats, on='i')
        combined_stats = combined_stats.sort_values(by='i')
        combined_stats.to_csv(filename, index=False)
        # return self.env.dump_stats(filename=None)
    
    def get_number_of_agents(self):
        return self.env.get_number_of_agents()
    
    def get_num_parallel(self):
        return self.env.get_num_parallel()

    def visualize_trajectory(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        # plt.plot(self.xs, label='States')
        state_array = np.array(self.xs)  # 将列表转换为 NumPy 数组
        
        if self.env_name == "cartpole":
            for i, state_label in enumerate(['x', 'x_dot', 'theta', 'theta_dot', 'x_ref']):
                plt.plot(state_array[:,i], label=f'{state_label}')
                # 在 y=0.5 和 y=-0.5 处绘制水平虚线
                plt.axhline(y=0.5, color='gray', linestyle='--')
                plt.axhline(y=-0.5, color='gray', linestyle='--')
        elif self.env_name == "double_integrator":
            for i, state_label in enumerate(['x', 'x_dot']):
                plt.plot(state_array[:,i], label=f'{state_label}')
                # 在 y=0.5 和 y=-0.5 处绘制水平虚线
                plt.axhline(y=0.5, color='gray', linestyle='--')
                plt.axhline(y=-0.5, color='gray', linestyle='--')
        elif self.env_name == "tank":
            for i, state_label in enumerate(['x1', 'x2', 'x3', 'x4', 'x1_ref','x2_ref', 'x3_ref', 'x4_ref']):
                plt.plot(state_array[:,i], label=f'{state_label}')
                # 在 y=20 处绘制水平虚线
                plt.axhline(y=20, color='gray', linestyle='--')
        
        plt.xlabel('Time Step')
        plt.ylabel('State Value')
        plt.title('State Over Time')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.us, label='u')
        plt.plot(self.uds, label='ud')
        plt.xlabel('Time Step')
        plt.ylabel('Control Value')
        plt.title('Control Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig("psf_trajectory.png")  # 保存图表
        plt.close()

integrated_env_creators = {
    "double_integrator": lambda **kwargs: Integrated_env("double_integrator", **kwargs),
    "tank": lambda **kwargs: Integrated_env("tank", **kwargs),
    "cartpole": lambda **kwargs: Integrated_env("cartpole", **kwargs),
}
