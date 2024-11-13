import numpy as np
import torch
from src.envs.linear_system import LinearSystem
from src.envs.cartpole import CartPole
from src.envs.env_creators import env_creators, sys_param
import gym

class Integrated_env:
    def __init__(self, env_name, **kwargs):
        self.env_name = env_name
        self.env = env_creators[env_name](**kwargs)
        self.bs = self.env.bs
        self.device = self.env.device
        self.train_or_test = kwargs["train_or_test"]
        
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
    
    def get_ud(self):
        # generate randomly (only work for action_dim = 1)
        ud = self.env.u_min + (self.env.u_max - self.env.u_min) * torch.rand(self.env.bs, device=self.device)
        self.ud = ud
        return ud
    
    def reset(self, *args, **kwargs):
        init_obs = self.env.reset(*args, **kwargs)
        init_ud = self.get_ud()
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
    
    def cost(self, *args, **kwargs):
        return self.env.cost(*args, **kwargs)
    
    def reward(self):
        original_reward = -self.env.safe_cost()
        deviation = torch.norm(self.ud - self.env.u, p=2) ** 2
        coef_deviation = 1.0
        deviation_cost = - coef_deviation * deviation
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


integrated_env_creators = {
    "double_integrator": lambda **kwargs: Integrated_env("double_integrator", **kwargs),
    "tank": lambda **kwargs: Integrated_env("tank", **kwargs),
    "cartpole": lambda **kwargs: Integrated_env("cartpole", **kwargs),
}
