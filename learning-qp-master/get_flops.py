'''
import numpy as np

def affine_layer_flops(input_size, output_size, has_bias, has_relu):
    """计算全连接层的浮点运算次数（乘加算作2次操作）"""
    flops = 2 * input_size * output_size    # 矩阵乘法：input_size * output_size次乘加
    if not has_bias:                        # 偏置项加法（如果无偏置需扣除）
        flops -= output_size                # 每个输出通道少1次加法
    if has_relu:                            # ReLU激活函数
        flops += output_size                # 每个输出通道1次比较操作
    return flops

def qp_flops(n_sys, n_qp, m_qp, qp_iter):
    """计算单次QP求解的FLOPS"""
    # 参数生成阶段
    get_q_flops = affine_layer_flops(2 * n_sys, n_qp, False, False)  # 生成Q矩阵（无偏置，无激活）
    get_b_flops = affine_layer_flops(n_sys, m_qp, True, False)       # 生成b向量（有偏置）
    get_mu_flops = (affine_layer_flops(n_sys, m_qp, False, False) +  # 初始对偶变量计算
                    affine_layer_flops(m_qp, m_qp, False, False) +  # 矩阵分解
                    m_qp)                                           # 向量初始化
    
    # 迭代求解阶段（每次迭代的FLOPS）
    per_iter_flops = m_qp + 2 * m_qp * (m_qp - 1) + 5 * m_qp  # 矩阵运算 + 向量操作
    
    # 总FLOPS = 参数生成 + 迭代次数×单次迭代
    return get_q_flops + get_b_flops + get_mu_flops + qp_iter * per_iter_flops

def mpc_flops(n_sys, m_sys, N, iter_counts):
    """计算MPC控制的总FLOPS（考虑不同迭代次数的情况）"""
    # MPC问题规模计算
    n_qp = m_sys * N               # QP变量数 = 控制输入维度 × 预测步长
    m_qp = 2 * (m_sys + n_sys) * N # QP约束数 = (输入+状态约束) × 步长
    
    # 统计迭代次数分布
    min_iter = np.min(iter_counts)
    max_iter = np.max(iter_counts)
    median_iter = np.median(iter_counts)
    
    # 计算不同情况下的FLOPS
    return (
        qp_flops(n_sys, n_qp, m_qp, min_iter),    # 最优情况
        qp_flops(n_sys, n_qp, m_qp, max_iter),    # 最差情况
        qp_flops(n_sys, n_qp, m_qp, median_iter)  # 典型情况
    )

def mlp_flops(input_size, output_size, hidden_layers):
    """计算MLP网络的FLOPS（包含所有隐藏层）"""
    flops = 0
    prev_size = input_size
    for size in hidden_layers:
        flops += affine_layer_flops(prev_size, size, True, True)  # 隐藏层（带偏置和ReLU）
        prev_size = size
    flops += affine_layer_flops(prev_size, output_size, True, False)  # 输出层（带偏置）
    return flops

if __name__ == "__main__":
    # 示例参数设置
    n_sys = 2       # 系统状态维度（例如倒立摆的4维状态）
    m_sys = 1       # 控制输入维度（单输入系统）
    N = 10          # MPC预测步长
    qp_iter = 10    # QP求解器迭代次数
    
    # 生成模拟的MPC迭代次数（正态分布）
    np.random.seed(42)
    mpc_iter_counts = np.random.normal(12, 2, 100).astype(int)  # 100次实验的迭代次数

    # 计算各方法FLOPS
    qp_total = qp_flops(n_sys, n_qp=4, m_qp=30, qp_iter=qp_iter)
    mpc_min, mpc_max, mpc_median = mpc_flops(n_sys, m_sys, N, mpc_iter_counts)
    mlp_total = mlp_flops(
        input_size=2*n_sys, 
        output_size=m_sys,
        hidden_layers=[256, 128, 64]  # 3个隐藏层：64和32个神经元
    )

    # 格式化输出
    print("计算复杂度对比（单步决策）:")
    print(f"QP求解器 (iter={qp_iter}): {qp_total:,} FLOPS")
    print(f"MPC控制器 (N={N}):")
    print(f"  最小FLOPS: {mpc_min:,} (iter={np.min(mpc_iter_counts)})")
    print(f"  最大FLOPS: {mpc_max:,} (iter={np.max(mpc_iter_counts)})")
    print(f"  中位数FLOPS: {mpc_median:,} (iter={np.median(mpc_iter_counts):.1f})")
    print(f"MLP网络 [64,32, 16]: {mlp_total:,} FLOPS")
'''

# %%
from glob import glob
import pandas as pd
import numpy as np
import torch

def read_csv(short_name):
    wildcard = f"{short_name}_2*"
    filename = sorted(glob(f"test_results/{wildcard}"))[-1]
    return pd.read_csv(filename, dtype={"constraint_violated": "bool"})

def read_mpc_iter_count(short_name):
    wildcard = f"{short_name}_mpc_iter_count_2*"
    filename = sorted(glob(f"test_results/{wildcard}"))[-1]
    return np.genfromtxt(filename)


def affine_layer_flops(input_size, output_size, has_bias, has_relu):
    flops = 2 * input_size * output_size
    if not has_bias:
        flops -= output_size
    if has_relu:
        flops += output_size
    return flops

def qp_flops(n_sys, n_qp, m_qp, qp_iter):
    get_q_flops = affine_layer_flops(2 * n_sys, n_qp, False, False)
    get_b_flops = affine_layer_flops(n_sys, m_qp, True, False)
    get_mu_flops = affine_layer_flops(n_sys, m_qp, False, False) + affine_layer_flops(m_qp, m_qp, False, False) + m_qp
    iter_flops = m_qp   # Adding primal-dual variables
    iter_flops += 2 * m_qp * (m_qp - 1)   # Matrix-vector multiplication
    iter_flops += 5 * m_qp   # Vector additions
    return get_q_flops + get_b_flops + get_mu_flops + qp_iter * iter_flops

def mpc_flops(n_sys, m_sys, N, iter_count_arr):
    n_qp = m_sys * N
    m_qp = 2 * (m_sys + n_sys) * N
    min_iter = np.min(iter_count_arr)
    max_iter = np.max(iter_count_arr)
    median_iter = np.median(iter_count_arr)
    min_flops = qp_flops(n_sys, n_qp, m_qp, min_iter)
    max_flops = qp_flops(n_sys, n_qp, m_qp, max_iter)
    median_flops = qp_flops(n_sys, n_qp, m_qp, median_iter)
    return min_flops, max_flops, median_flops

def mlp_flops(input_size, output_size, hidden_sizes):
    flops = 0
    prev_size = input_size
    for size in hidden_sizes:
        flops += affine_layer_flops(prev_size, size, True, True)
        prev_size = size
    flops += affine_layer_flops(prev_size, output_size, True, False)
    return flops

def count_parameters(exp_name):
    checkpoint_path = f"runs/cartpole_{exp_name}/nn/cartpole.pth"
    checkpoint = torch.load(checkpoint_path)
    total_params = 0
    for key, value in checkpoint['model'].items():
        if key.startswith("a2c_network.policy_net") or key.startswith("a2c_network.actor_mlp"):
            total_params += value.numel()
    return total_params

def get_row(short_name, method, n_sys=4, m_sys=1, n_qp=None, m_qp=None, qp_iter=10, N_mpc=None, mlp_last_size=None):
    """Output (short name, success rate, cost, penalized costs, FLOPs, learnable parameters)."""
    result_df = read_csv(short_name)
    total_episodes = len(result_df)
    penalty = 1000
    # avg_cost = result_df['cumulative_cost'].sum() / result_df['episode_length'].sum()
    # avg_cost_penalized = (result_df['cumulative_cost'].sum() + penalty * result_df["constraint_violated"].sum()) / result_df['episode_length'].sum()
    # freq_violation = result_df["constraint_violated"].sum() / result_df['episode_length'].sum()
    success_rate = 1. - result_df["constraint_violated"].sum() / total_episodes

    # Count FLOPs
    if method == "qp":
        flops = qp_flops(n_sys, n_qp, m_qp, qp_iter)
    elif method == "mpc":
        iter_count_arr = read_mpc_iter_count(short_name)
        flops = mpc_flops(n_sys, m_sys, N_mpc, iter_count_arr)
    elif method == "mlp":
        flops = mlp_flops(2 * n_sys, m_sys, [i * mlp_last_size for i in [4, 2, 1]])

    # Count learnable parameters
    if method == "mpc":
        num_param = 0
    else:
        num_param = count_parameters(short_name)

    return short_name, success_rate, flops, num_param

# %%
rows = [
    # get_row("reproduce_mpc_2_0", "mpc", N_mpc=2),
    # get_row("reproduce_mpc_2_1", "mpc", N_mpc=2),
    # get_row("reproduce_mpc_2_10", "mpc", N_mpc=2),
    # get_row("reproduce_mpc_2_100", "mpc", N_mpc=2),
    # get_row("reproduce_mpc_4_0", "mpc", N_mpc=4),
    # get_row("reproduce_mpc_4_1", "mpc", N_mpc=4),
    # get_row("reproduce_mpc_4_10", "mpc", N_mpc=4),
    # get_row("reproduce_mpc_4_100", "mpc", N_mpc=4),
    # get_row("reproduce_mpc_8_0", "mpc", N_mpc=8),
    # get_row("reproduce_mpc_8_1", "mpc", N_mpc=8),
    # get_row("reproduce_mpc_8_10", "mpc", N_mpc=8),
    # get_row("reproduce_mpc_8_100", "mpc", N_mpc=8),
    # get_row("reproduce_mpc_16_0", "mpc", N_mpc=16),
    # get_row("reproduce_mpc_16_1", "mpc", N_mpc=16),
    # get_row("reproduce_mpc_16_10", "mpc", N_mpc=16),
    # get_row("reproduce_mpc_16_100", "mpc", N_mpc=16),

    get_row("mpc_10t", "mpc", N_mpc=10),

    # get_row("reproduce_mlp_8", "mlp", mlp_last_size=8),
    # get_row("reproduce_mlp_16", "mlp", mlp_last_size=16),
    # get_row("reproduce_mlp_32", "mlp", mlp_last_size=32),
    # get_row("reproduce_mlp_64", "mlp", mlp_last_size=64),
    
    # get_row("reproduce_qp_4_24", "qp", n_qp=4, m_qp=24),
    # get_row("reproduce_qp_8_48", "qp", n_qp=8, m_qp=48),
    # get_row("reproduce_qp_16_96", "qp", n_qp=16, m_qp=96),
]

df_result = pd.DataFrame(rows, columns=["name", "success_rate", "flops", "num_param"])
df_result.to_csv("test_results/reproduce_table.csv", index=False)
print(df_result)
