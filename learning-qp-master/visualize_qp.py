# %% Specify test case
import numpy as np

# Initial position and reference position
state0 = [-0.1, -0.2, -0.02, -0.45]
theta_ref = -1.3

# Controlling process noise and parametric uncertainty
noise_level = 0
parametric_uncertainty = False
parameter_randomization_seed = 42

# %% Set up test bench
from src.envs.env_creators import sys_param, env_creators
from src.modules.qp_unrolled_network import QPUnrolledNetwork
import torch
from matplotlib import pyplot as plt


# Utilities
def make_obs(state, theta_ref):
    theta = state[0]
    theta_dot = state[1]
    alpha = state[2]
    alpha_dot = state[3]
    raw_obs = torch.tensor(np.array([theta, theta_dot, alpha, alpha_dot, theta_ref]), device=device, dtype=torch.float)
    return raw_obs.unsqueeze(0)

# 从给定的检查点文件路径加载一个模型的状态字典
def get_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    # 定义一个字符串 prefix，它表示模型状态字典中的键可能具有的前缀。这个前缀可能来源于模型被保存时的层次结构。
    prefix = "a2c_network.policy_net."
    policy_net_state_dict = {k.lstrip(prefix): v for (k, v) in model.items() if k.startswith(prefix)}
    return policy_net_state_dict

def rescale_action(action, low=-1., high=8.):
    action = action.clamp(-1., 1.)
    return low + (high - low) * (action + 1) / 2

t = lambda arr: torch.tensor(arr, device=device, dtype=torch.float).unsqueeze(0)
a = lambda t: t.detach().cpu().numpy()

# Constants and options
n_sys = 4
m_sys = 1
input_size = 5
n = 4
m = 24
qp_iter = 10
device = "cuda:0"

# # Learned QP
net = QPUnrolledNetwork(device, input_size, n, m, qp_iter, None, True, True)
exp_name = f"reproduce_qp_{n}_{m}"
if parametric_uncertainty:
    exp_name += "+rand"
checkpoint_path = f"experiments/QUBEServo/runs/QUBEServo_{exp_name}/nn/QUBEServo.pth"
policy_net_state_dict = get_state_dict(checkpoint_path)
net.load_state_dict(policy_net_state_dict,strict=False)
net.to(device)

# Environment
env = env_creators["QUBEServo"](
    noise_level=noise_level,
    bs=1,
    max_steps=300,
    keep_stats=True,
    run_name=exp_name,
    exp_name=exp_name,
    randomize=parametric_uncertainty,
)

# %% Test for learned QP
xs_qp = [t(state0).squeeze(0)]
us_qp = []
done = False
env.reset(t(state0[0]), t(theta_ref), randomize_seed=parameter_randomization_seed)
state = state0
obs = make_obs(state, theta_ref)
while not done:
    action_all, problem_params = net(obs, return_problem_params=True)
    u = rescale_action(action_all[:, :m_sys])
    raw_obs, reward, done_t, info = env.step(u)
    xs_qp.append(raw_obs[0, :4])
    us_qp.append(u[0, :])
    obs = raw_obs
    done = done_t.item()

# %% Plot : Trajectory
# Create a 3-row, 2-column matrix of subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

# Example to populate the subplots
# 遍历子图网格的前4个子图（左上角开始的2x2网格）
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        subscript = 2 * i + j
        #ax.plot([a(xs_mpc[k][subscript]) for k in range(len(xs_mpc))], label="MPC")
        ax.plot([a(xs_qp[k][subscript]) for k in range(len(xs_qp))], label="QP")
        # ax.plot([a(xs_mlp[k][subscript]) for k in range(len(xs_mlp))], label="MLP")
        if subscript == 0:
            ax.axhline(y=theta_ref, color='r', linestyle='--', label='Ref')
        ax.legend()
        ax.set_title(['theta', 'theta_dot', 'alpha', 'alpha_dot'][subscript])

# 绘制了子图网格中剩余的子图（第3行的第1列）
i = 2
for j in range(1):
    ax = axes[i, j]
    #ax.plot([a(us_mpc[k][j]) for k in range(len(us_mpc))], label="MPC")
    ax.plot([a(us_qp[k][j]) for k in range(len(us_qp))], label="QP")
    # ax.plot([a(us_mlp[k][j]) for k in range(len(us_mlp))], label="MLP")
    ax.legend()
    ax.set_title(f'f')

plt.tight_layout()
plt.show()

# %%
