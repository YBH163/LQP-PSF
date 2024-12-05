import numpy as np
import torch
from src.envs.linear_system import LinearSystem
from src.envs.cartpole import CartPole
from src.envs.env_creators import env_creators, sys_param
import gym
from icecream import ic

class Integrated_env:
    def __init__(self, env_name, **kwargs):
        self.env_name = env_name
        self.env = env_creators[env_name](**kwargs)
        self.bs = self.env.bs
        self.device = self.env.device
        self.train_or_test = kwargs["train_or_test"]
        self.u_min = self.env.u_min.item()  # 仅取值
        self.u_max = self.env.u_max.item()
        
        # Gym environment settings
        self.action_space = self.env.action_space
        original_shape = self.env.observation_space.shape
        new_shape = (original_shape[0] + 1,)    # different
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
    
    def get_ud(self, obs):
        # x, x_dot, theta, theta_dot, x_ref = self.env.obs()
        # theta = obs[..., 2]
        if self.train_or_test == "train":
            # generate randomly (only work for action_dim = 1)
            # ud = self.u_min + (self.u_max - self.u_min) * torch.rand(self.env.bs, device=self.device)
            
            noise = 0.1
            v = (noise * torch.randn((self.bs, 1), device=self.device))
            ud = self.env.get_action_LQR(noise_level = 0) + v  # 双重噪声
            # ud = v
            ud = ud.clamp(self.env.u_min, self.env.u_max)
            ud = ud.squeeze(-1)
        elif self.train_or_test == "test":
            # bang-bang control
            # 当 theta 大于 0 度时，u 应该小于 0；当 theta 小于 0 度时，u 应该大于 0
            # if theta >= 0.2 :
            #     ud = self.u_min
            # elif theta <= -0.2 :
            #     ud = self.u_max
            # else:
            #     ud = 0
            
            # bang-bang control (使用 torch.where 来向量化条件操作
            # ud = torch.where(theta >= 0.2, torch.full_like(theta, self.u_max), torch.where(theta <= -0.2, torch.full_like(theta, self.u_min), torch.zeros_like(theta)))
            # LQR control
            noise = 1
            v = (noise * torch.randn((self.bs, 1), device=self.device))
            ud = self.env.get_action_LQR(noise_level = noise) + v  # 双重噪声
            # ud = v
            ud = ud.clamp(self.env.u_min, self.env.u_max)
            ud = ud.squeeze(-1)

        self.ud = ud
        return ud
    
    def reset(self, *args, **kwargs):
        init_obs = self.env.reset(*args, **kwargs)
        if self.train_or_test == "train":
            init_ud = self.get_ud(init_obs)
        elif self.train_or_test == "test":
            init_ud = torch.zeros(self.env.bs, device=self.device)     # cuz initial theta is small (0.1)
            self.ud = init_ud
        combined_obs = torch.cat((init_obs, init_ud.unsqueeze(-1)), dim=-1)
        return combined_obs
    
    def done(self):
        return self.env.done()
    
    def info(self):
        return self.env.info()
    
    def obs(self):
        # Get the current observation from the environment
        current_obs = self.env.obs()    # 形状是 (batch_size, observation_dim)
        # ud = self.get_ud()  # 形状是 (batch_size, )
        # Concatenate the current observation with the control input ud
        combined_obs = torch.cat((current_obs, self.ud.unsqueeze(-1)), dim=-1)
        return combined_obs
    
    # def cost(self, *args, **kwargs):
    #     return self.env.cost(*args, **kwargs)
    
    def reward(self,original_reward, action):
        coef_original = 0.1
        original_reward = coef_original * original_reward
        
        safe_cost = -self.env.safe_cost()
        coef_safety = 50.0
        safe_cost = coef_safety * safe_cost
        # deviation = torch.norm(self.ud - action, p=2) ** 2
        deviation = (self.ud - action.squeeze()) ** 2
        coef_deviation = 10.0
        deviation_cost = -coef_deviation * deviation
        
        # 添加存活奖励
        coef_survival = 0.0  # 存活奖励系数，可以根据需要调整
        survival_reward = coef_survival  # 每个步骤的存活奖励
        
        # 添加出界惩罚
        bound_cost = (action.squeeze() > self.u_max).float() * 1000 + (action.squeeze() < self.u_min).float() * 1000
        bound_cost = bound_cost * (bound_cost > 0).float()  # 确保只有越界的时候才为1000，否则为0

        combined_reward = original_reward + safe_cost + deviation_cost + survival_reward - bound_cost  # 注意正负号！！！
        # combined_reward = safe_cost + deviation_cost  # 注意正负号！！
        if not self.env.quiet:
            avg_safe_cost = safe_cost.float().mean().item()
            avg_deviation_cost = deviation_cost.mean().item()
            avg_bound_cost = bound_cost.mean().item()
            ic(avg_safe_cost)
            ic(avg_deviation_cost)
            ic(avg_bound_cost)
        if self.train_or_test == "train":
            return combined_reward
        elif self.train_or_test == "test":
            return safe_cost      # original safe cost
    
    def step(self, action):    
        # 用ud进行测试时
        # original_obs, original_reward, done, info = self.env.step(self.ud.unsqueeze(-1))   
        # reward = self.reward(original_reward, self.ud)
        
        # 正常测试
        original_obs, original_reward, done, info = self.env.step(action)   
        reward = self.reward(original_reward, action)
        # get next ud
        self.get_ud(original_obs)   # update ud every step
        return self.obs(), reward, done, info
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
       
    def dump_stats(self, filename=None):
        return self.env.dump_stats(filename=None)
    
    def get_number_of_agents(self):
        return self.env.get_number_of_agents()
    
    def get_num_parallel(self):
        return self.env.get_num_parallel()


integrated_env_creators = {
    "double_integrator": lambda **kwargs: Integrated_env("double_integrator", **kwargs),
    "tank": lambda **kwargs: Integrated_env("tank", **kwargs),
    "cartpole": lambda **kwargs: Integrated_env("cartpole", **kwargs),
}
