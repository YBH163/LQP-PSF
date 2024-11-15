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


def psf2qp(n_sys, m_sys, N, A, B, x_min, x_max, u_min, u_max, x0, x_ref, normalize=False, Qf=None):
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
    ud = x0[:, -1].unsqueeze(-1)    # x0 or x?
    # 使用索引去掉最后一列
    x0 = x0[:, :-1]

    Ax0 = torch.cat([bmv((torch.linalg.matrix_power(A, k + 1)).unsqueeze(0), x0) for k in range(N)], 1)   # (bs, N * n_sys)
    m = 2 * (n_sys + m_sys) * N   # number of constraints
    n = m_sys * N                 # number of decision variables

    b = torch.cat([
        Ax0 - x_min,
        x_max - Ax0,
        -u_min * torch.ones((bs, n), device=device),
        u_max * torch.ones((bs, n), device=device),
    ], 1)

    XU = torch.zeros((N, n_sys, N, m_sys), device=device)
    for k in range(N):
        for j in range(k + 1):
            XU[k, :, j, :] = (torch.linalg.matrix_power(A, k - j) @ B)
    XU = XU.flatten(0, 1).flatten(1, 2)   # (N * n_MPC, N * m_MPC)


    # q = -2 * XU.t().unsqueeze(0) @ Q_kron.unsqueeze(0) @ (kron(torch.ones((bs, N, 1), device=device), x_ref.unsqueeze(-1)) - Ax0.unsqueeze(-1))   # (bs, N * m_MPC, 1)
    # q = q.squeeze(-1)  # (bs, N * m_MPC) = (bs, n)
    
    # 创建一个形状为 (batch_size, n_qp) 的全零张量
    q_vector = torch.zeros((bs, n), device=device)
    # 将 last_data_unsqueezed 赋值给 result_vector 的第一个元素
    q_vector[:, 0] = -ud.squeeze(-1)  # 赋值并去除多余的维度
    q = q_vector
    
    # P = 2 * XU.t() @ Q_kron @ XU + 2 * kron(torch.eye(N, device=device), R)  # (n, n)
    
    eps = 1e-6     # a very small number
    # 创建一个对角线元素全为 eps 的对角阵
    P = torch.diag(torch.full((n,), eps))
    # 将第一行第一列的元素设置为 1
    P[0, 0] = 1        # only work for action_dim = 1
    P = P.to(device)
    
    H = torch.cat([XU, -XU, torch.eye(n, device=device), -torch.eye(n, device=device)], 0)  # (m, n)

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

