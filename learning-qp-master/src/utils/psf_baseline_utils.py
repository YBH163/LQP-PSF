import torch
from .torch_utils import make_psd, bmv, kron
import numpy as np
import cvxpy as cp
from scipy.linalg import kron as sp_kron
from numpy.linalg import matrix_power as np_matrix_power
from scipy.linalg import block_diag
import do_mpc
from ..envs.mpc_baseline_parameters import get_mpc_baseline_parameters
import time
import warnings


def generate_random_problem(bs, n, m, device):
    P_params = -1 + 2 * torch.rand((bs, n * (n + 1) // 2), device=device)
    q = -1 + 2 * torch.rand((bs, n), device=device)
    H_params = -1 + 2 * torch.rand((bs, m * n), device=device)
    x0 = -1 + 2 * torch.rand((bs, n), device=device)
    P = make_psd(P_params)
    H = H_params.view(-1, m, n)
    b = bmv(H, x0)
    return q, b, P, H


def psf2qp(n_sys, m_sys, N, A, B, x_min, x_max, u_min, u_max, x0, x_ref, normalize=False, Qf=None, F = None, g = None):
    """
    Converts Predictive Safety Filter (PSF) problem parameters into Quadratic Programming (QP) form.

    Parameters:
    - n_sys (int): Dimension of the state space.
    - m_sys (int): Dimension of the input space.
    - N (int): Prediction horizon.
    - A (torch.Tensor): State transition matrix, shape (n_sys, n_sys).
    - B (torch.Tensor): Control input matrix, shape (n_sys, m_sys).
    - x_min (float): Lower state bounds.
    - x_max (float): Upper state bounds.
    - u_min (float): Lower control bounds.
    - u_max (float): Upper control bounds.
    - x0 (torch.Tensor): Initial state, shape (batch_size, n_sys).
    - x_ref (torch.Tensor): Reference state, shape (batch_size, n_sys).
    - normalize (bool): Whether to normalize the control actions. If set to True, the solution of the QP problem will be rescaled actions within range [-1, 1].
    - Qf (torch.Tensor, optional): Terminal state cost matrix, shape (n_sys, n_sys).

    Returns:
    - n (int): Number of decision variables.
    - m (int): Number of constraints.
    - P (torch.Tensor): QP cost matrix, shape (n, n).
    - q (torch.Tensor): QP cost vector, shape (batch_size, n).
    - H (torch.Tensor): Constraint matrix, shape (m, n).
    - b (torch.Tensor): Constraint bounds, shape (batch_size, m).

    The converted QP problem is in form:
        minimize    (1/2)x'Px + q'x
        subject to  Hx + b >= 0,

    Notes:
    - The function assumes that A, B, Q, R are single matrices, and x0 and x_ref are in batch.
    - All tensors are expected to be on the same device.
    """
    bs = x0.shape[0]
    device = x0.device
    
    # 获取 x 的最后一个数据，并增加新的维度，生成形状为 (batch_size, 1) 的张量
    ud = x0[:, -1].unsqueeze(-1)    
    # 使用索引去掉最后一列
    x0 = x0[:, :-1]

    Ax0 = torch.cat([bmv((torch.linalg.matrix_power(A, k + 1)).unsqueeze(0), x0) for k in range(N)], 1)   # (bs, N * n_sys)
    m_original = 2 * (n_sys + m_sys) * N   # number of constraints
    n_original = m_sys * N                 # number of decision variables
    if (F != None) and (g != None):
        xN_constraints = g.shape[0]
        m = m_original + xN_constraints
        n = n_original
    else:
        m = m_original
        n = n_original
        
    # 将（4,1）的numpy array变成（bs，4）的torch tensor
    x_min = torch.from_numpy(x_min).view(-1).repeat(bs, 1).to(device)
    x_max = torch.from_numpy(x_max).view(-1).repeat(bs, 1).to(device)
    # 将形状为 (bs, 4) 的向量重复 N 次，变成形状为 (bs, 4N) 的向量
    x_min_repeated = x_min.repeat(1, N)     # repeat(1, N) 表示在第一个维度（bs）上重复1次，在第二个维度（4）上重复N次
    x_max_repeated = x_max.repeat(1, N)
    
    # b = torch.cat([
    #     Ax0 - x_min_repeated,
    #     x_max_repeated - Ax0,
    #     -u_min * torch.ones((bs, n_original), device=device),
    #     u_max * torch.ones((bs, n_original), device=device),
    # ], 1)
    b = torch.cat([
        x_max_repeated - Ax0,
        Ax0 - x_min_repeated,  
        u_max * torch.ones((bs, n_original), device=device),
        -u_min * torch.ones((bs, n_original), device=device),
    ], 1)
    
    if (F != None) and (g != None):
        g_tensor = torch.from_numpy(g).to(device)  # 注意负号！
        # 然后，重复 g_tensor 以匹配 b 的批次大小
        g_repeated = g_tensor.repeat(b.shape[0], 1)
        ANx0 = bmv((torch.linalg.matrix_power(A, N)).unsqueeze(0), x0)
        
        # 首先，将 F 转换为 PyTorch 张量
        F_tensor = torch.from_numpy(F).float().to(device)
        # 第一个参数是批次大小，第二个参数是1，意味着在列方向上不重复
        F_repeated = F_tensor.unsqueeze(0).repeat(b.shape[0], 1, 1)
        # 最后，沿着列的方向（dim=1）追加 g_repeated 到 b
        b = torch.cat([b, (F_repeated@(ANx0.unsqueeze(-1))).squeeze(-1)-g_repeated], dim=1)
    # b = b.float()

    XU = torch.zeros((N, n_sys, N, m_sys), device=device)
    for k in range(N):
        for j in range(k + 1):
            XU[k, :, j, :] = (torch.linalg.matrix_power(A, k - j) @ B)
    XU = XU.flatten(0, 1).flatten(1, 2)   # (N * n_MPC, N * m_MPC)
    
    # H = torch.cat([XU, -XU, torch.eye(n_original, device=device), -torch.eye(n_original, device=device)], 0)  # (m, n)
    H = torch.cat([-XU, XU, -torch.eye(n_original, device=device), torch.eye(n_original, device=device)], 0)  # (m, n)
 
    if (F != None) and (g != None):
        # FAB = (torch.cat([(F@ torch.linalg.matrix_power(A, N-1-k) @B for k in range(N)) ], 1))
        # 使用 list() 将生成器转换为列表
        FAB_list = [F_tensor @ torch.linalg.matrix_power(A, N-1-k) @ B for k in range(N)]
        # 现在 FAB_list 是一个张量列表，可以安全地传递给 torch.cat
        FAB = torch.cat(FAB_list, dim=1)
        
        H = torch.cat([H,FAB],0)
    
    # 创建一个形状为 (batch_size, n_qp) 的全零张量
    q_vector = torch.zeros((bs, n), device=device)
    # 将 last_data_unsqueezed 赋值给 result_vector 的第一个元素
    q_vector[:, 0] = -ud.squeeze(-1)  # 赋值并去除多余的维度
    q = q_vector
    
    eps = 1e-3     # a very small number
    # 创建一个对角线元素全为 eps 的对角阵
    P = torch.diag(torch.full((n,), eps))
    # 将第一行第一列的元素设置为 1
    P[0, 0] = 1        # only work for action_dim = 1
    P = P.to(device)
    
    if normalize:
        # u = alpha * u_normalized + beta
        alpha = (u_max - u_min) / 2 * torch.ones((m_sys,), device=device)   # (m_MPC,)
        beta = (u_max + u_min) / 2 * torch.ones((m_sys,), device=device)    # (m_MPC,)
        Alpha = torch.diag_embed(alpha.repeat(N))  # (n, n)
        Beta = beta.repeat(N)  # (n,)
        P_nom = Alpha @ P @ Alpha    # (n,)
        q_nom = bmv(Alpha.unsqueeze(0), q + bmv(P, Beta).unsqueeze(0))    # (bs, n)
        H_nom = H @ Alpha    # (m, n)
        b_nom = (H @ Beta).unsqueeze(0) + b    # (bs, m)
        P, q, H, b = P_nom, q_nom, H_nom, b_nom

    return n, m, P, q, H, b

