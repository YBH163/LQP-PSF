import pandas as pd
import torch
import numpy as np
from src.envs.env_creators import env_creators
from icecream import ic
import random
import os
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Controlling process noise and parametric uncertainty
noise_level = 0
parametric_uncertainty = False
# parameter_randomization_seed = 42

# Constants and options
n_sys = 4
m_sys = 2
# input_size = 5
device = "cuda:0"
bs = 1
exp_name = f"test_lqr"

# Initial position and reference position
# state0 = [-1,0]
# x_ref = [0,0]

env_name = "tank"
# 创建环境实例
env = env_creators[env_name](
    noise_level=noise_level,
    bs=bs,
    max_steps=500,
    keep_stats=True,
    run_name=exp_name,
    exp_name=exp_name,
    randomize=parametric_uncertainty,
    # quiet = True,
    # Q = np.diag([10., 0, 100., 0]),
    # R = np.array([[1]]),
    device = device,
    env_name = env_name
)

# 收集数据的函数
def collect_data(env, num_episodes, save_interval, csv_file):    
    data_all = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'terminals': []
    }
    
    for i in range(num_episodes):
        data = [{'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []} for _ in range(bs)]        
        obs = env.reset()
        done = torch.zeros(bs, dtype=torch.bool, device=device)
        # is_done = False
        # while not torch.any(done): # 只要有1个实例完成了，就退出循环
        for _ in range(env.max_steps-1):
            # K = env.get_K_LQR()
            # lqr_action = K @ obs.unsqueeze(1)
            v = (noise_level * torch.randn((bs, 1), device=device))
            action = env.get_action_LQR() + v   # LQR方法生成
            action = torch.clamp(action, -10, 10)
            # action = -10 + (20 * torch.rand((bs, 1), device=device))  # 随机生成
            env.reset_done_envs()
            obs = env.obs()
            next_obs, reward, done, _ = env.step(action)
            for j in range(bs):
                data[j]['observations'].append(obs[j].cpu().numpy())
                data[j]['actions'].append(action[j].cpu().numpy())
                data[j]['rewards'].append(reward[j].cpu().numpy())
                data[j]['next_observations'].append(next_obs[j].cpu().numpy())
                data[j]['terminals'].append(done[j].cpu().numpy())
                # if done[j]:
                    # next_obs[j] = env.reset()
            # obs = next_obs
        for k in range(bs):
            for key in data_all.keys():
                data_all[key].extend(data[k][key])
                
        # 检查是否达到保存间隔，如果是，则保存数据
        if (i+1) % save_interval == 0 or i == num_episodes - 1:
            # 将数据转换为 pandas DataFrame
            df = pd.DataFrame(data_all)

            mode = 'a'  # 始终使用追加模式
            # 检查文件是否存在，如果不存在，我们需要写入表头
            write_header = not os.path.isfile(csv_file)
            df.to_csv(csv_file, index=False, mode=mode, header=write_header)  # 只在文件首次创建时添加表头

            # 清空数组，准备下一次保存
            data_all = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': []
            }
    return data_all

def make_obs(state, x_ref):
    x = state[0]
    dx = state[1]
    raw_obs = torch.tensor(np.array([x, dx, x_ref[0], x_ref[1]]), device=device, dtype=torch.float)
    return raw_obs.unsqueeze(0)

def test_lqr(env, num_episodes):
    # 初始化性能指标的列表
    total_rewards = []
    total_lengths = []

    states = []  # 用于存储状态
    controls = []  # 用于存储控制输入

    # 测试循环
    for episode in range(num_episodes):
        # 自定义初始状态
        # t = lambda arr: torch.tensor(arr, device=device, dtype=torch.float).unsqueeze(0)
        # env.reset(t(state0[0]), t(x_ref))
        # obs = make_obs(state0, x_ref)
        
        obs = env.reset()       # 随机初始化
        
        episode_reward = torch.zeros(bs, dtype=torch.float32).to('cuda:0')
        episode_length = torch.zeros(bs, dtype=torch.int32).to('cuda:0')
        active_mask = torch.ones(bs, dtype=torch.bool).to('cuda:0')  # 初始化活跃掩码

        state_history = []  # 存储每个时间步的状态
        control_history = []  # 存储每个时间步的控制输入

        while active_mask.any():  # 只要还有活跃的episode，就继续循环
            # 选择并执行动作，只对活跃的episode
            noise_level = 0
            v = (noise_level * torch.randn((bs, 1), device=device))
            action = env.get_action_LQR(noise_level = noise_level) + v   # LQR方法生成
            # action = env.get_action_LQR(noise_level = 0) + v
            # action = (10 * torch.randn((bs, 1), device=device))
            action = torch.clamp(action, env.u_min, env.u_max)
            
            # 假设 self.eval_env.step 接受 PyTorch 张量作为输入，并且只对活跃的episode进行操作
            next_obs, reward, terminal, _ = env.step(action)

            # 更新活跃episode的奖励和长度
            episode_reward[active_mask] += reward[active_mask]
            episode_length[active_mask] += 1

            # 更新活跃掩码，停止已经结束的episode
            active_mask &= ~terminal
            
            # 记录状态和控制输入
            state_history.append(obs[0].cpu().numpy())
            control_history.append(action[0].cpu().numpy())
            
            # 转移到下一个观察，只对活跃的episode
            obs[active_mask] = next_obs[active_mask]

        # 计算平均长度
        average_episode_length = episode_length.float().mean().item()
        average_episode_reward  = episode_reward .float().mean().item()

        # 记录episode的平均奖励和平均长度（bs个取平均
        total_rewards.append(average_episode_reward)
        total_lengths.append(average_episode_length)
        
        # 存储状态和控制历史
        states.append(state_history)
        controls.append(control_history)
        
        # 打印每个episode的结果
        print(f"Episode {episode}: Reward = {episode_reward}, Length = {episode_length}")
        

    # 计算并返回平均奖励和平均长度（test_episodes个，取平均
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_lengths)
    # 打印平均奖励和平均长度
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    print(f"Average Episode Length over {num_episodes} episodes: {avg_length}")
    
    # return {"average_reward": avg_reward, "average_episode_length": avg_length}
    return {"average_reward": avg_reward, "average_episode_length": avg_length, "states": states, "controls": controls}

def plot_states_controls(states, controls):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    # plt.plot(states[0], label='States')
    
    state_array = np.array(states[0])  # 将列表转换为 NumPy 数组
    # 我们将为每个状态值绘制一条线，并设置不同的标签
    if env_name == "cartpole":
        for i, state_label in enumerate(['x', 'x_dot', 'theta', 'theta_dot', 'x_ref']):
            plt.plot(state_array[:,i], label=f'{state_label}')
    elif env_name == "double_integrator":
        for i, state_label in enumerate(['x', 'x_dot']):
            plt.plot(state_array[:,i], label=f'{state_label}')
    elif env_name == "tank":
        for i, state_label in enumerate(['x1', 'x2', 'x3', 'x4', 'x1_ref','x2_ref', 'x3_ref', 'x4_ref']):
            plt.plot(state_array[:,i], label=f'{state_label}')
    
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('State Over Time')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(controls[0], label='u')
    plt.xlabel('Time Step')
    plt.ylabel('Control Value')
    plt.title('Control Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":    
    num_episodes = 100  # 假设我们想要100个episodes的数据
    is_test = 1
    is_visualize = 1
    save_interval = 10  # 每10个episodes保存一次数据
    if is_test:
        if is_visualize:
            num_episodes = 1
            results = test_lqr(env, num_episodes)  # 测试lqr效果
            plot_states_controls(results["states"], results["controls"])  # 绘制状态和控制量
            plt.savefig("trajectory.png")  # 保存图表
            plt.close()  # 关闭图表，避免显示在屏幕上
        else:
            test_lqr(env, num_episodes)  # 测试lqr效果
    else:      
        # 保存为 CSV 文件
        csv_file = 'qube_servo_data_positive.csv'  
        dataset = collect_data(env, num_episodes,save_interval, csv_file)   # 收集数据

        # 将数据转换为 pandas DataFrame
        # df = pd.DataFrame(dataset)
        # df.to_csv(csv_file, index=False)