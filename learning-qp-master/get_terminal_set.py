import pandas as pd
import torch
import numpy as np
from src.envs.env_creators import env_creators
from icecream import ic
import random
import os

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class Terminal_set:
    def __init__(self, Hx, Hu, K, Ak, h):
        # self.dt = 0.01
        self.Hx = Hx
        self.Hu = Hu
        self.K = K
        self.Ak = Ak
        self.H = np.block([[Hu, np.zeros((Hu.shape[0], Hx.shape[1]))],
                          [np.zeros((Hx.shape[0], Hu.shape[1])), Hx]])
        self.Nc = self.H.shape[0]
        self.Nx = Ak.shape[1]
        self.h = h
        self.K_aug = np.vstack((-K, np.eye(self.Nx)))
        self.maxiter = 200
        self.Xf = self.terminal_set_cal()
        self.Xf_nr = self.remove_redundancy()
        # self.test_input_inbound(0.15)

    def terminal_set_cal(self):
        Ainf = np.zeros([0, self.Nx])
        binf = np.zeros([0, 1])
        Ft = np.eye(self.Nx)
        self.C = self.H@self.K_aug

        for t in range(self.maxiter):
            Ainf = np.vstack((Ainf, self.C@Ft))
            binf = np.vstack((binf, self.h))

            Ft = self.Ak@Ft
            fobj = self.C@Ft
            violation = False
            for i in range(self.Nc):
                val, x = self.solve_linprog(fobj[i, :], Ainf, binf)
                if val > self.h[i]:
                    violation = True
                    break
            if not violation:
                return [Ainf, binf]

    def solve_linprog(self, obj, Ainf, binf):
        x = cp.Variable((self.Nx, 1))
        objective = cp.Maximize(obj@x)
        constraints = [Ainf@x <= binf]
        linear_program = cp.Problem(objective, constraints)

        # result = linear_program.solve(verbose=False)
        result = linear_program.solve(solver=cp.SCS, verbose=True)
        return result, x.value

    def remove_redundancy(self):
        A_inf, Binf = self.Xf
        Ainf_nr, binf_nr = A_inf.copy(), Binf.copy()
        i = 0
        while i < Ainf_nr.shape[0]:
            obj = Ainf_nr[i, :]
            binf_temp = binf_nr.copy()
            binf_temp[i] += 1
            val, x = self.solve_linprog(obj, Ainf_nr, binf_temp)
            if val < binf_nr[i] or val == binf_nr[i]:
                Ainf_nr = np.delete(Ainf_nr, i, 0)
                binf_nr = np.delete(binf_nr, i, 0)
            else:
                i += 1
        return [Ainf_nr, binf_nr]

    def get_vertices(self):
        # 假设 self.Xf_nr 是去除冗余后的终端集，其中包含 Ainf 和 binf
        A_inf, b_inf = self.Xf_nr
        vertices = []
        for i in range(A_inf.shape[0]):
            obj = A_inf[i, :]
            _, x = self.solve_linprog(obj, A_inf, b_inf)
            vertices.append(x)
        return np.array(vertices)
    
    def visualize_terminal_set(self, vertices):
        # 提取 theta 和 theta_dot 用于可视化
        theta = vertices[:, 2]
        theta_dot = vertices[:, 3]
        # 使用 column_stack 将它们组合为一个 (n, 2) 的数组
        combined = np.column_stack((theta, theta_dot))
        
        # 创建一个多边形对象
        poly = Polygon(combined, closed=True , fill=True, edgecolor='red', facecolor='blue', linewidth=2, alpha=0.5)

        # 创建图形和轴
        fig, ax = plt.subplots()

        # 将多边形添加到轴上
        ax.add_patch(poly)

        # 设置坐标轴范围以包含所有顶点
        ax.set_xlim(-0.004, 0.014)
        ax.set_ylim(-0.02, 0.008)

        # 设置坐标轴标签和标题
        ax.set_xlabel('theta')
        ax.set_ylabel('theta_dot')
        ax.set_title('Terminal Set in theta-theta_dot Plane')

        # 显示网格
        ax.grid(True)

        # 显示图形
        plt.show()

        # 创建一个 patch 集合
        # patches = PatchCollection([poly], match_original=True)

        # 假设 patches 是一个 PatchCollection 对象
        # 假设您想要添加第一个 Patch 到轴上
        # first_patch = patches[0]  # 获取 PatchCollection 中的第一个 Patch 对象

        # 创建一个散点图
        # plt.scatter(theta, theta_dot, c='b', marker='o')
        
        # 假设 patches 是一个 PatchCollection 对象
        # for patch in patches:  # 遍历 PatchCollection 中的每个 Patch 对象
        #     plt.gca().add_patch(patch, facecolor='none', edgecolor='r', alpha=0.5)  # 将每个 Patch 对象添加到轴上

        

        # 添加多边形到散点图
        # plt.gca().add_patch(patches)
        # 添加 Patch 到轴上
        # plt.gca().add_patch(first_patch)

        # 1. 绘制多边形的边界
        # plt.gca().add_patch(patches, facecolor='none', edgecolor='r')

        # 2. 填充多边形的内部
        # plt.gca().add_patch(patches, facecolor='r', edgecolor='r', alpha=0.5)

        # plt.gca().set_xlabel('theta')
        # plt.gca().set_ylabel('theta_dot')
        # plt.gca().set_title('Terminal Set in theta-theta_dot Plane with Polygon')
        # plt.gca().grid(True)
        # plt.gca().axis('equal')  # 保持纵横比
        # plt.gca().show()
        
        
        # 绘制散点图
        # plt.scatter(theta, theta_dot, c='b', marker='o')
        # plt.xlabel('theta')
        # plt.ylabel('theta_dot')
        # plt.title('Terminal Set in theta-theta_dot Plane')
        # plt.grid(True)
        # plt.show()
        
    def test_input_inbound(self,u_limit):
        A_inf,b_inf=self.Xf_nr
        violation =False
        for i in range(4):
            x = cp.Variable((12, 1))
            u = cp.Variable((4, 1))
            cost=0
            constr = []
            constr.append(A_inf@x[:,0] <= b_inf.squeeze())
            constr.append(u[:, 0]==-self.K@x[:,0])
            cost=u[i, 0]
            problem = cp.Problem(cp.Maximize(cost), constr)
            problem.solve()
            print('Input u',i,'<',problem.value)
            if problem.value >u_limit:
                violation =True
        if violation ==False:
            print('Input inbound')
        
    def get_boundary_points(self, num_points=100):
        A_inf, b_inf = self.Xf_nr
        points = []
        directions = np.array([[np.cos(2 * np.pi * i / num_points), np.sin(2 * np.pi * i / num_points)] for i in range(num_points)])
        for direction in directions:
            obj = direction.reshape(-1, 1)
            _, x = self.solve_linprog(obj, A_inf, b_inf)
            points.append(x.flatten())
        return np.array(points)

    def visualize_terminal_set_directly(self, num_points=100):
        boundary_points = self.get_boundary_points(num_points)
        theta = boundary_points[:, 2]
        theta_dot = boundary_points[:, 3]
        plt.figure()
        plt.plot(theta, theta_dot, 'r-')  # 使用红色线条绘制边界
        plt.xlabel('theta')
        plt.ylabel('theta_dot')
        plt.title('Terminal Set in theta-theta_dot Plane')
        plt.grid(True)
        plt.show()
        

if __name__ == "__main__":    
    
    # seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Controlling process noise and parametric uncertainty
    noise_level = 0.5
    parametric_uncertainty = False
    # parameter_randomization_seed = 42

    # Constants and options
    n_sys = 4
    m_sys = 1
    input_size = 5
    device = "cuda:0"
    bs = 100
    exp_name = f"test_lqr"

    # 创建环境实例
    env = env_creators["cartpole"](
        noise_level=noise_level,
        bs=bs,
        max_steps=100,
        keep_stats=True,
        run_name=exp_name,
        exp_name=exp_name,
        randomize=parametric_uncertainty,
        quiet = True,
        Q = np.diag([10., 1e-4, 100., 1e-4]),
        R = np.array([[1]]),
        device = device
    )
    
    # 初始化 Terminal_set 类
    terminal_set = Terminal_set(env.Hx, env.Hu, env.K, env.Ak, env.h)
    
    # 可视化终端集
    # vertices = terminal_set.get_vertices()
    # terminal_set.visualize_terminal_set(vertices)
    
    terminal_set.visualize_terminal_set_directly()
    plt.savefig('terminal_set.png')
    plt.close()
    