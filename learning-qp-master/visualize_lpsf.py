# %% Specify test case
import numpy as np

# Initial position and reference position
# state0 = [-0.1, -0.2, -0.02, -0.45]
# theta_ref = -1.3
# state0 = [1,-1]
# x_ref = [0,0]

# Controlling process noise and parametric uncertainty
noise_level = 0
parametric_uncertainty = False
parameter_randomization_seed = 42

# %% Set up test bench
from src.envs.env_creators import sys_param, env_creators
from src.psf.integrated_env_creators import integrated_env_creators
from src.modules.qp_unrolled_network import QPUnrolledNetwork
import torch
from matplotlib import pyplot as plt
from src.envs.mpc_baseline_parameters import get_mpc_baseline_parameters


# Utilities
# def make_obs(state, theta_ref):
#     theta = state[0]
#     theta_dot = state[1]
#     alpha = state[2]
#     alpha_dot = state[3]
#     raw_obs = torch.tensor(np.array([theta, theta_dot, alpha, alpha_dot, theta_ref]), device=device, dtype=torch.float)
#     return raw_obs.unsqueeze(0)

def make_obs(state, x_ref):
    x = state[0]
    dx = state[1]
    raw_obs = torch.tensor(np.array([x, dx, x_ref[0], x_ref[1]]), device=device, dtype=torch.float)
    return raw_obs.unsqueeze(0)

# 从给定的检查点文件路径加载一个模型的状态字典
def get_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    # 定义一个字符串 prefix，它表示模型状态字典中的键可能具有的前缀。这个前缀可能来源于模型被保存时的层次结构。
    prefix = "a2c_network.policy_net."
    policy_net_state_dict = {k.lstrip(prefix): v for (k, v) in model.items() if k.startswith(prefix)}
    return policy_net_state_dict

def rescale_action(action, low=-0.5, high=0.5):
    action = action.clamp(-1., 1.)
    return low + (high - low) * (action + 1) / 2

t = lambda arr: torch.tensor(arr, device=device, dtype=torch.float).unsqueeze(0)
a = lambda t: t.detach().cpu().numpy()

# Constants and options
env_name = "double_integrator"
n_sys = 2
m_sys = 1
input_size = 5      # for double_integrator 2obs+2refs+ud
n = 4
m = 30
qp_iter = 10
device = "cuda:0"
# psf_N = 1

# # Learned QP
# **get_mpc_baseline_parameters(args.env, args.mpc_baseline_N or args.imitate_mpc_N, noise_std=args.noise_level), "terminal_coef": args.mpc_terminal_cost_coef
net = QPUnrolledNetwork(device, input_size, n, m, qp_iter, None, True, True)
# exp_name = f"reproduce_qp_{n}_{m}"
exp_name = f"reproduce_psf_{n}_{m}"
if parametric_uncertainty:
    exp_name += "+rand"
checkpoint_path = f"runs/double_integrator_reproduce_qp_4_30/nn/double_integrator.pth"
policy_net_state_dict = get_state_dict(checkpoint_path)
net.load_state_dict(policy_net_state_dict,strict=False)
net.to(device)

# Environment
env = integrated_env_creators[env_name](
    train_or_test = "test",
    noise_level=noise_level,
    bs=1,
    max_steps=300,
    keep_stats=True,
    run_name=exp_name,
    exp_name=exp_name,
    randomize=parametric_uncertainty,
)

# %% Test for learned QP
xs_qp = []
us_qp = []
ud = []
done = False
obs = env.reset(randomize_seed=parameter_randomization_seed)
xs_qp.append(obs[0, :2].cpu().numpy())
# state = state0
# obs = make_obs(state, x_ref)
while not done:
    ud.append(obs[0, -1].detach().cpu().numpy())   # 获取ud
    action_all, problem_params = net(obs, return_problem_params=True)
    # u = rescale_action(action_all[:, :m_sys])
    u = action_all[:, :m_sys]
    raw_obs, reward, done_t, info = env.step(u)
    # xs_qp.append(raw_obs[0, :2].cpu().numpy())    # 获取两个状态变量
    xs_qp.append(raw_obs[0, :2].detach().cpu().numpy())
    
    us_qp.append(u[0, :].detach().cpu().numpy())
    obs = raw_obs
    done = done_t.item()

# %% Plot : Trajectory
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(xs_qp, label='States')
# 在 y=1 和 y=-1 处绘制水平虚线
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.axhline(y=-0.5, color='gray', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.title('State Over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(us_qp, label='u')
plt.plot(ud, label='ud')
plt.xlabel('Time Step')
plt.ylabel('Control Value')
plt.title('Control Over Time')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("learnedPSF_trajectory.png")  # 保存图表
plt.close()  # 关闭图表，避免显示在屏幕上
