import torch
import functools
import scipy
import numpy as np
from src.utils.osqp_utils import osqp_oracle
from src.utils.np_batch_op import np_batch_op

n_qp = 4
m_qp = 30
device = "cuda:0"
force_feasible = True
feasible_lambda = 10

eps = 1e-6     # a very small number
# 创建一个对角线元素全为 eps 的对角阵
P = torch.diag(torch.full((n_qp,), eps))
# 将第一行第一列的元素设置为 1
P[0, 0] = 1        # only work for action_dim = 1
P = P.to(device)
if force_feasible:
    zeros_n = torch.zeros((n_qp, 1), device=device)
    I = torch.eye(1, device=device)
    tilde_P = torch.cat([
        torch.cat([P, zeros_n], dim=1),
        torch.cat([zeros_n.transpose(0, 1), feasible_lambda * I], dim=1)
    ], dim=0)
    P = tilde_P

q = torch.tensor([[-6.7464,  0.0000,  0.0000, 0.0000]], device='cuda:0')
if force_feasible:
    zeros_1 = torch.zeros((1, 1), device=device)
    # \tilde{q} = [q; 0]
    tilde_q = torch.cat([q, zeros_1], dim=1)
    q = tilde_q

# 定义 H 和 b
H = torch.tensor([[[-0.3325,  2.3814, -0.3990,  1.0669,  1.0000],
         [-1.2009,  3.3538, -2.0721,  0.0742,  1.0000],
         [-0.6007,  3.1400, -1.1665,  1.2686,  1.0000],
         [-0.2172,  2.8690, -0.6277,  1.7452,  1.0000],
         [-0.1728,  1.8317, -3.9059, -0.2060,  1.0000],
         [-0.7044, -0.3224, -0.5362, -2.9028,  1.0000],
         [-0.9618, -0.2953,  0.0808, -3.1649,  1.0000],
         [-1.6348,  2.9826, -2.2683, -1.2421,  1.0000],
         [ 0.3761, -0.7957, -0.5315, -1.2920,  1.0000],
         [-0.5896,  2.2938, -0.2078,  0.4069,  1.0000],
         [-1.5102,  1.7262,  1.0269, -0.2878,  1.0000],
         [-0.1371,  0.8116,  0.6735, -0.3764,  1.0000],
         [-0.2303,  0.6536,  0.9516, -0.6965,  1.0000],
         [-0.2200,  0.6624,  0.9229, -0.6747,  1.0000],
         [ 0.1553,  1.2058,  0.2312,  0.6507,  1.0000],
         [-0.4041,  0.1419, -3.2367, -1.8866,  1.0000],
         [ 0.4768, -1.5457,  0.2094, -1.8393,  1.0000],
         [-0.7060,  3.6499, -1.2777,  1.6804,  1.0000],
         [ 0.2355,  0.2349, -3.2933, -1.0629,  1.0000],
         [-0.7637,  1.9064, -3.6160, -1.2266,  1.0000],
         [-1.7651,  0.7980,  0.8537, -3.2557,  1.0000],
         [-1.5070,  1.2525,  0.6516, -1.0954,  1.0000],
         [ 0.0060, -1.8561,  0.2896, -3.2094,  1.0000],
         [ 0.1599,  0.3121, -0.6179, -0.6492,  1.0000],
         [-0.4655,  3.4750, -1.1824,  1.9181,  1.0000],
         [-1.0315,  3.2563, -3.8242, -0.1974,  1.0000],
         [-0.4377, -0.2574, -0.2755, -2.1793,  1.0000],
         [-0.4080,  0.6244, -3.7736, -1.6979,  1.0000],
         [-1.1875,  3.4297,  0.0086,  0.3307,  1.0000],
         [-0.4932,  2.0894,  0.1507,  0.4161,  1.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
if not force_feasible:
    H = H[:, :-1, :-1]
H = H.squeeze(0)

# b = torch.tensor([[ 22.7906, -22.2197,  -7.6467,  27.8491,  32.6169,   6.4014, -12.5474,
#           -3.9420,  17.8911,  -9.6213,  28.1120,  11.5298,  12.5685,  12.4370,
#            5.4271, -58.4826,  43.9484,  24.3436,  19.4929,  13.4744, -42.8695,
#          -26.0789,  11.4214,  14.5985,   8.3701,   9.2256, -10.2844,  33.5245,
#          -25.3994,  -9.0650,   0.0000]], device='cuda:0')
b = torch.tensor([[ 22.7906, -22.2197,  -7.6467,  27.8491,  32.6169,   6.4014, -12.5474,
          -3.9420,  17.8911,  -9.6213,  28.1120,  11.5298,  12.5685,  12.4370,
           5.4271, -58.4826,  43.9484,  24.3436,  19.4929,  13.4744, -42.8695,
         -26.0789,  11.4214,  14.5985,   8.3701,   9.2256, -10.2844,  33.5245,
         -25.3994,  -9.0650]], device='cuda:0')
if force_feasible:
    zeros_1 = torch.zeros((1, 1), device=device)
    tilde_b = torch.cat([b, zeros_1], dim=1)
    b = tilde_b

# Conversions between torch and np
t = lambda a: torch.tensor(a, device=device)
f = lambda t: t.detach().cpu().numpy()
f_sparse = lambda t: scipy.sparse.csc_matrix(t.cpu().numpy())

osqp_oracle_with_iter_count = functools.partial(osqp_oracle, return_iter_count=True)
if q.shape[0] > 1:
    sol_np, iter_counts = np_batch_op(osqp_oracle_with_iter_count, f(q), f(b), f_sparse(P), f_sparse(H))
    sol = t(sol_np)
else:
    sol_np, iter_count = osqp_oracle_with_iter_count(f(q[0, :]), f(b[0, :]), f_sparse(P), f_sparse(H))
    sol = t(sol_np).unsqueeze(0)
    iter_counts = np.array([iter_count])

print(sol)
print(iter_counts)