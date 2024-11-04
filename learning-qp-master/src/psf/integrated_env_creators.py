import numpy as np
import torch
from src.envs.linear_system import LinearSystem
from src.envs.cartpole import CartPole
from src.envs.env_creators import env_creators, sys_param

class Integrated_env:
    def __init__(self, env_name, **kwargs):
        self.env_name = env_name
        self.env = env_creators[env_name](**kwargs)
        self.bs = self.env.bs
        self.device = self.env.device
    
    def get_ud(self):
        # generate randomly (only work for action_dim = 1)
        ud = self.env.u_min + (self.env.u_max - self.env.u_min) * torch.rand(self.env.bs, device=self.device)
        self.ud = ud
        return ud
    
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
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
    
    def cost(self, *args, **kwargs):
        return self.env.cost(*args, **kwargs)
    
    def reward(self):
        original_reward = self.env.reward()
        deviation = torch.norm(self.ud - self.env.u, p=2) ** 2
        coef_deviation = 1.0
        deviation_cost = coef_deviation * deviation
        combined_reward = original_reward + deviation_cost
        return combined_reward
    
    def step(self, action):
        self.get_ud()   # update ud every step
        original_obs, original_reward, done, info = self.env.step(action)
        return self.obs(), self.reward(), done, info
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
       
    def dump_stats(self, filename=None):
        return self.env.dump_stats(filename=None)
    
    def get_number_of_agents(self):
        return self.env.get_number_of_agents()
    
    def get_num_parallel(self):
        return self.env.get_num_parallel()

# 使用示例
env_name = "cartpole"  # 可以是 "double_integrator", "tank", "cartpole"
kwargs = {
    "parameters": {
        "dt": 0.1,
        "m_cart": [0.7, 1.3],
        "m_pole": [0.07, 0.13],
        "l": [0.4, 0.7],
    },
    "Q": np.diag([1., 1e-4, 1., 1e-4]),
    "R": np.array([[1e-4]]),
    "noise_std": 0.5,
    "x_min": -2,
    "x_max": 2,
    "u_min": -10,
    "u_max": 10,
    "bs": 1,
    "barrier_thresh": 0.1,
    "max_steps": 500,
    "device": "cuda:0",
    "random_seed": 42,
    "quiet": False,
    "keep_stats": False,
    "reward_shaping_parameters": {},
    "run_name": "test_run",
}
integrated_env_creators = Integrated_env(env_name, **kwargs)
