import torch
import torch.nn as nn
import numpy as np
import random
import gym
import pandas as pd
import os
from datetime import datetime
from ..utils.torch_utils import bmv, bqf, bsolve, conditional_fork_rng, get_rng
from icecream import ic
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from ..utils.sets import compute_MCI
from scipy.spatial import ConvexHull , Delaunay

class LinearSystem():
    def __init__(
        self, A, B, Q, R, sqrt_W,
        x_min, x_max, u_min, u_max, bs, barrier_thresh,
        max_steps,
        u_eq_min=None, u_eq_max=None,
        skip_to_steady_state=False,
        stabilization_only=False,
        device="cuda:0",
        random_seed=None,
        quiet=False,
        keep_stats=False,
        reward_shaping_parameters={},
        run_name="",
        initial_generator=None,
        ref_generator=None,
        randomizer=None,
        **kwargs
    ):
        '''
        Initializes the LinearSystem environment with given parameters.

        Parameters:
            A (ndarray): System dynamics matrix. Perturbed by randomizer if specified.
            B (ndarray): Input matrix. Perturbed by randomizer if specified.
            Q (ndarray): State cost matrix.
            R (ndarray): Control input cost matrix.
            sqrt_W (ndarray): Square root of the process noise covariance matrix.
            x_min (ndarray): Lower bound for each state variable.
            x_max (ndarray): Upper bound for each state variable.
            u_min (ndarray): Lower bound for each control input.
            u_max (ndarray): Upper bound for each control input.
            bs (int): Batch size for parallel environment execution.
            barrier_thresh (float): Threshold for state constraint barriers.
            max_steps (int): Maximum number of steps in an episode.
            u_eq_min (ndarray, optional): Lower bound for equilibrium control input.
            u_eq_max (ndarray, optional): Upper bound for equilibrium control input.
            skip_to_steady_state (bool, optional): Debug option to skip to steady state -- changes the problem to a static problem of optimizing u; side effects include changing max_steps to 1, and ignoring the process noise.
            stabilization_only (bool, optional): Option to exclude reference and only do stabilization around the origin. When True, the observation will be the current state; when False, the observation will be current state & reference.
            device (str, optional): Computational device ("cpu" or "cuda").
            random_seed (int, optional): Random seed for reproducibility.
            quiet (bool, optional): Suppresses debug prints when set to True.
            keep_stats (bool, optional): Whether to maintain statistics of episodes.
            reward_shaping_parameters (dict, optional): Parameters for reward shaping.
            run_name (str, optional): Name tag for the run, useful for logging.
            initial_generator (function, optional): Function that generates initial states.
            ref_generator (function, optional): Function that generates reference states.
            randomizer (function, optional): Function that randomizes the system dynamics (returns \Delta A, \Delta B).
        '''
        # Random seed and random number generators for different components
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        # Random number generator for initial states and references
        self.rng_initial = get_rng(device, random_seed)
        # Random number generator for process noise
        self.rng_process = get_rng(device, random_seed)
        # Random number generator for randomization of A, B matrices
        self.rng_dynamics = get_rng(device, random_seed)

        # Environment definitions
        self.device = device
        self.n = A.shape[0]
        self.m = B.shape[1]
        t = lambda a: torch.tensor(a, dtype=torch.float, device=device).unsqueeze(0)
        self.A = t(A)
        self.B = t(B)
        self.Q = t(Q)
        self.R = t(R)
        self.A0 = self.A
        self.B0 = self.B
        
        self.dt = 0.1
        # 定义连续时间系统的参数
        self.A_continuous = A
        self.B_continuous = B
        C_continuous = np.array([[1, 0]])  # 虽然不需要，但函数需要这个参数
        D_continuous = np.array([[0]])  # 虽然不需要，但函数需要这个参数
        # 使用 cont2discrete 函数将连续时间系统转换为离散时间系统
        # 只提取 A_discrete 和 B_discrete
        self.A_discrete, self.B_discrete, _, _, _ = cont2discrete(
            (self.A_continuous, self.B_continuous, C_continuous, D_continuous), self.dt
        )
        self.A = t(self.A_discrete)
        self.B = t(self.B_discrete)
        self.A0 = self.A
        self.B0 = self.B
        
        self.randomizer = randomizer
        self.reward_shaping_parameters = reward_shaping_parameters
        if randomizer is not None:
            # Copy the nominal A, B, and repeat A, B along the batch dimension to allow randomization later
            self.A = self.A.repeat(bs, 1, 1)
            self.B = self.B.repeat(bs, 1, 1)
        self.sqrt_W = t(sqrt_W)
        self.x_min = t(x_min)
        self.x_max = t(x_max)
        self.u_min = t(u_min)
        self.u_max = t(u_max)
        self.u_eq_min = t(u_eq_min) if u_eq_min is not None else self.u_min
        self.u_eq_max = t(u_eq_max) if u_eq_max is not None else self.u_max
        self.bs = bs
        self.barrier_thresh = barrier_thresh    # 安全边界或屏障阈值
        self.max_steps = max_steps
        self.stabilization_only = stabilization_only
        self.num_states = self.n if stabilization_only else 2 * self.n
        self.num_actions = self.m
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        self.action_space = gym.spaces.Box(low=u_min, high=u_max, shape=(self.num_actions,))
        self.state_space = self.observation_space
        self.x = 0.5 * (self.x_max + self.x_min) * torch.ones((bs, self.n), device=device)
        self.x0 = 0.5 * (self.x_max + self.x_min) * torch.ones((bs, self.n), device=device)
        self.u = torch.zeros((bs, self.m), device=device)
        self.x_ref = 0.5 * (self.x_max + self.x_min) * torch.ones((bs, self.n), device=device)
        # Keep record of the first noise vector in each trajectory, as an identifier of the instantiation of the randomness
        self.w0 = torch.zeros_like(self.x)
        self.is_done = torch.zeros((bs,), dtype=torch.uint8, device=device)
        self.step_count = torch.zeros((bs,), dtype=torch.long, device=device)
        self.cum_cost = torch.zeros((bs,), dtype=torch.float, device=device)
        self.run_name = run_name
        self.keep_stats = keep_stats
        self.already_on_stats = torch.zeros((bs,), dtype=torch.uint8, device=device)   # Each worker can only contribute once to the statistics, to avoid bias towards shorter episodes
        self.stats = pd.DataFrame(columns=['i', 'x0', 'x_ref', 'A', 'B', 'w0', 'episode_length', 'cumulative_cost', 'constraint_violated'])
        self.quiet = quiet
                
        if skip_to_steady_state:
            self.max_steps = 1
            self.skip_to_steady_state = True
        else:
            self.skip_to_steady_state = False

        self.info_dict = {}

        self.initial_generator = initial_generator
        self.ref_generator = ref_generator
        
        # for LQR control        
        # self.P = solve_continuous_are(A, B, Q, R) 
        # self.K = (np.linalg.solve(R, B.T)) @ self.P 
        self.P = solve_discrete_are(self.A_discrete, self.B_discrete, Q, R) 
        self.K = (np.linalg.solve(R,  self.B_discrete.T)) @ self.P 
        
        # for terminal set calculation
        # 构造 Hx 和 h
        self.Hx = np.block([
            [np.eye(self.n)],
            [-np.eye(self.n)],# 状态变量的上界和下界
        ])
        # hx = np.concatenate([self.x_max, self.x_dot_max, self.theta_max, self.theta_dot_max, -self.x_min, -self.x_dot_min, -self.theta_min, -self.theta_dot_min])
        self.hx = np.concatenate([
            x_max,
            -x_min
        ]).reshape((4, 1))

        # 构造 Hu 和 h
        self.Hu = np.array([[1], [-1]])  # 控制输入的上界和下界
        self.hu = np.array([u_max, -u_min])
        # 构造 h，将状态和控制输入的上界合并
        self.h = np.concatenate([self.hx, self.hu]).reshape(-1, 1)

        # 计算闭环系统的状态转移矩阵 Ak
        self.Ak = self.A_discrete - np.dot(self.B_discrete, self.K)
        
        # 计算MCI
        self.mci_vertices = compute_MCI(self.A_discrete, self.B_discrete, x_min, x_max, u_min, u_max, iterations=10)
    
    def get_action_LQR(self, noise_level = None):        
        # 将K转为torch tensor类型
        K_tensor = torch.from_numpy(self.K).to(self.device, dtype=torch.float32)
        # LQR控制律
        if noise_level is None:
            action = K_tensor @ ((self.x_ref-self.x).T)
            # 得到的是1*bs的，还需要转置一下成为bs*1的才行
            action_transposed = action.t()  # (bs,1)
        else:
            # 生成一个与K_tensor形状相同，均值为0，标准差为noise_level的高斯噪声
            noise = torch.randn_like(K_tensor) * noise_level
            # 将噪声添加到K_tensor上
            K_tensor_noisy = K_tensor + noise
            # 使用带有噪声的K_tensor_noisy
            action = K_tensor_noisy @ ((self.x_ref - self.x).T)
            # 得到的是1*bs的，还需要转置一下成为bs*1的才行
            action_transposed = action.t()  # (bs,1)
        return action_transposed

    def obs(self):
        """
        Returns the current observation, which is a concatenation of the current state and reference state.
        """
        if not self.stabilization_only:
            return torch.cat([self.x, self.x_ref], -1)
        else:
            return self.x

    def cost(self, x, u):
        """
        Computes the cost based on the state deviation from the reference and control effort.
        """
        return bqf(x, self.Q) + bqf(u, self.R)

    def reward(self):
        """
        Computes the reward based on the current state, control input, and various coefficients.
        """
        cost = self.cost(self.x - self.x_ref, self.u)
        rew_main = -cost
        rew_state_bar = torch.sum(torch.log(((self.x_max - self.x) / self.barrier_thresh).clamp(1e-8, 1.)) + torch.log(((self.x - self.x_min) / self.barrier_thresh).clamp(1e-8, 1.)), dim=-1)
        rew_done = -1.0 * (self.is_done == 1)

        # Reward shaping for address steady-state error: c1 * exp(-c2 * (cost - c3))
        c1 = self.reward_shaping_parameters.get("steady_c1", 0.)
        c2 = self.reward_shaping_parameters.get("steady_c2", 1.)
        c3 = self.reward_shaping_parameters.get("steady_c3", 0.)
        rew_steady = c1 * torch.exp(-c2 * (cost - c3))

        coef_const = 0.
        coef_main = 1.
        coef_steady = 1.
        coef_bar = 0.
        coef_done = 100000.

        rew_total = coef_const + coef_main * rew_main + coef_steady * rew_steady + coef_bar * rew_state_bar + coef_done * rew_done

        self.info_dict["actual_costs"] = cost + coef_done * (self.is_done == 1)

        if not self.quiet:
            avg_rew_main, avg_rew_state_bar, avg_rew_done, avg_rew_steady, avg_rew_total = coef_main * rew_main.mean().item(), coef_bar * rew_state_bar.mean().item(), coef_done * rew_done.mean().item(), rew_steady.mean().item(), rew_total.mean().item()
            ic(avg_rew_main, avg_rew_done, avg_rew_steady, avg_rew_total)
        return rew_total

    def safe_cost(self):
        # 检查每个状态变量是否越界，返回一个(bs, 2)形状的布尔张量
        boundary_check = ((self.x < self.x_min) | (self.x > self.x_max))
        # 将布尔张量转换为整数张量，任何状态变量越界则为1，否则为0
        safe_cost = boundary_check.any(dim=-1).int()
        self.info_dict["safe_cost"] = safe_cost
        return safe_cost

    def done(self):
        """
        Checks whether the episode has terminated for each environment in the batch.
        """
        return self.is_done.bool()

    def info(self):
        """
        Returns additional information.
        """
        self.info_dict["already_on_stats"] = self.already_on_stats
        return self.info_dict

    def get_number_of_agents(self):
        """
        Returns the number of agents in the environment, which is 1 in this case.
        """
        return 1

    def get_num_parallel(self):
        """
        Returns the batch size for parallel environment execution.
        """
        return self.bs

    def generate_ref(self, size):
        """
        Generates a reference state based on the control input bounds and nominal system dynamics.
        """
        if not self.stabilization_only:
            if self.ref_generator is not None:
                return self.ref_generator(size, self.device, self.rng_initial)
            else:
                # 随机生成x_ref，dx_ref = 0
                # x_ref = self.x_min + (self.x_max - self.x_min) * torch.rand((size, self.n), generator=self.rng_initial, device=self.device)
                # x_ref[:, -1] = 0  # 将最后一维的所有元素设置为0 (稳态时速度为0)
                # x_ref = x_ref.clamp(self.x_min + self.barrier_thresh, self.x_max - self.barrier_thresh)
                # 稳定到原点
                x_ref = torch.zeros((size, self.n), device=self.device)
                
                # Fall back to default reference generation
                # u_ref = self.u_eq_min + (self.u_eq_max - self.u_eq_min) * torch.rand((size, self.m), generator=self.rng_initial, device=self.device)
                # epsilon = 1e-6
                # A_reg = torch.eye(self.n, device=self.device).unsqueeze(0) - self.A0 + epsilon * torch.eye(self.n, device=self.device).unsqueeze(0)
                # x_ref = bsolve(A_reg, bmv(self.B0, u_ref))
                # # x_ref = bsolve(torch.eye(self.n, device=self.device).unsqueeze(0) - self.A0, bmv(self.B0, u_ref))
                # x_ref += self.barrier_thresh * torch.randn((size, self.n), generator=self.rng_initial, device=self.device)
                # x_ref = x_ref.clamp(self.x_min + self.barrier_thresh, self.x_max - self.barrier_thresh)
        else:
            return torch.zeros((size, self.n), device=self.device)
        return x_ref
    
    def is_point_in_hull(self, point):
        # 确保点和凸包的顶点都在 CPU 上
        # point_np = point.cpu().numpy()
        hull_points_np = self.mci_vertices

        # 创建 ConvexHull 对象
        hull = ConvexHull(hull_points_np)

        # 检查点是否在凸包内
        return hull._in_hull(point)
    
    # def is_point_in_hull(self, point):
    #     # 使用 ConvexHull 检查点是否在凸包内
    #     hull = ConvexHull(self.mci_vertices)
    #     return hull.points_in_hull(point)

    def generate_random_point_in_hull(self):
        while True:
            # 随机选择一个三角形
            simplex_indices = np.random.choice(len(self.mci_vertices), size=3, replace=False)
            simplex = self.mci_vertices[simplex_indices]

            # 生成该三角形内的随机点
            u = np.random.rand(3)
            u /= u.sum()  # 确保 u 是一个概率分布
            random_point = np.dot(u, simplex)

            # 检查点是否在凸包内
            # if self.is_point_in_hull(random_point):
            #     return random_point
            return random_point

    def generate_initial(self, size):
        """
        Generates an initial state based on the state bounds.
        """
        if self.initial_generator is not None:
            return self.initial_generator(size, self.device, self.rng_initial)
        else:
            # x0 = self.x_min + self.barrier_thresh + (self.x_max - self.x_min - 2 * self.barrier_thresh) * torch.rand((size, self.n), generator=self.rng_initial, device=self.device)
            initial_states_list = []

            for _ in range(size):
                new_point = self.generate_random_point_in_hull()
                initial_states_list.append(new_point)

            # 将 NumPy 数组列表转换为 PyTorch 张量
            x0 = torch.tensor(initial_states_list,dtype=torch.float32, device=self.device).reshape(size, 2)

            # 初始化在MCI内
            # hull = ConvexHull(self.mci_vertices)
            # points = self.mci_vertices[hull.vertices]
            # delaunay = Delaunay(points)
            
            # 在 generate_initial 方法中调用 generate_random_point_in_hull 时，确保使用 .cpu().numpy() 转换
            # x0 = torch.tensor([self.generate_random_point_in_hull().cpu().numpy() for _ in range(size)], device=self.device)
            '''
            # 生成凸包内的随机点
            def generate_random_point_in_hull():
                while True:
                    # 随机选择一个三角形
                    # simplex_indices = self.rng_initial.integers(0, len(hull.simplices), size=3)
                    simplex_indices = torch.randint(0, len(hull.simplices), size=(3,), generator=self.rng_initial, device=self.device)
                    simplex = points[simplex_indices]
                    
                    # 生成该三角形内的随机点
                    u = self.rng_initial.random(size=3)
                    u /= u.sum()  # 确保 u 是一个概率分布
                    random_point = np.dot(u, simplex)
                    if hull.points_in_hull(random_point):
                        return random_point
            x0 = torch.tensor([generate_random_point_in_hull() for _ in range(size,)], device=self.device)
            '''
            
            return x0

    def reset_done_envs(self, need_reset=None, x=None, x_ref=None, randomize_seed=None):
        """
        Resets the environments that are marked as 'done', reinitializing their states and references.

        Parameters:
        - need_reset (torch.Tensor, optional): A boolean tensor indicating which environments need to be reset.
                                            If None, the function will automatically determine this based on self.is_done.
        - x (torch.Tensor, optional): Initial state tensor for the environments that need to be reset.
                                    If None, random initial states are generated within defined bounds.
        - x_ref (torch.Tensor, optional): Reference state tensor for the environments that need to be reset.
                                        If None, references are generated via self.generate_ref().
        - randomize_seed (int, optional): Seed for random number generation when randomizing system matrices A and B.
                                        If None, no seeding is applied.

        Side Effects:
        - Modifies self.step_count, self.cum_cost, self.x_ref, self.x0, self.x, self.is_done, self.A, and self.B for the
        environments that are reset.

        Notes:
        - The function expects self.is_done, self.x_min, self.x_max, self.barrier_thresh, self.n, self.m, self.device,
        self.randomizer, self.A0, and self.B0 to be pre-defined.
        - Utilizes the conditional_fork_rng context manager for conditional seeding.
        """
        is_done = self.is_done.bool() if need_reset is None else need_reset
        size = torch.sum(is_done)
        self.step_count[is_done] = 0
        self.cum_cost[is_done] = 0
        self.x_ref[is_done, :] = self.generate_ref(size) if x_ref is None else x_ref
        self.x0[is_done, :] = self.generate_initial(size) if x is None else x
        self.x[is_done, :] = self.x0[is_done, :]
        self.is_done[is_done] = 0
        if self.randomizer is not None:
            if randomize_seed is not None:
                # Seed for randomization of dynamics is specified in function call; use it directly
                with torch.random.fork_rng():
                    torch.manual_seed(randomize_seed)
                    Delta_A, Delta_B = self.randomizer(size, self.device, None)
            else:
                # No seed specified; use predefined random number generator for randomization of dynamics
                Delta_A, Delta_B = self.randomizer(size, self.device, self.rng_dynamics)
            self.A[is_done, :, :] = self.A0 + Delta_A
            self.B[is_done, :, :] = self.B0 + Delta_B


    def reset(self, x=None, x_ref=None, randomize_seed=None):
        """
        Resets the environment, reinitializing the states and references.
        """
        self.reset_done_envs(torch.ones(self.bs, dtype=torch.bool, device=self.device), x, x_ref, randomize_seed)
        return self.obs()

    def check_in_bound(self):
        """
        Checks whether the current state is within the predefined bounds.
        """
        return ((self.x_min <= self.x) & (self.x <= self.x_max)).all(dim=-1)

    def write_episode_stats(self, i):
        """
        Logs statistics of the episode for the ith environment in the batch.
        """
        self.already_on_stats[i] = 1
        x0 = self.x0[i, :].cpu().numpy()
        x_ref = self.x_ref[i, :].cpu().numpy()

        # Get the A and B matrices and flatten for the ith environment
        index = 0 if self.randomizer is None else i
        A = self.A[index, :, :].cpu().numpy().flatten()
        B = self.B[index, :, :].cpu().numpy().flatten()

        w0 = self.w0[i, :].cpu().numpy()

        episode_length = self.step_count[i].item()
        cumulative_cost = self.cum_cost[i].item()
        constraint_violated = (self.is_done[i] == 1).item()
        self.stats.loc[len(self.stats)] = [i.item(), x0, x_ref, A, B, w0, episode_length, cumulative_cost, constraint_violated]

    def dump_stats(self, filename=None):
        """
        Writes the accumulated statistics to a CSV file.
        """
        if filename is None:
            directory = 'test_results'
            if not os.path.exists(directory):
                os.makedirs(directory)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tag = self.run_name
            filename = os.path.join(directory, f"{tag}_{timestamp}.csv")
        self.stats = self.stats.sort_values(by='i')
        self.stats.to_csv(filename, index=False)

    def step(self, u):
        """
        Executes one step in the environment based on the given control input.
        """
        self.reset_done_envs()
        u = u.clamp(self.u_min, self.u_max)
        self.u = u
        # self.cum_cost += self.cost(self.x - self.x_ref, u)
        w = bmv(self.sqrt_W, torch.randn((self.bs, self.n), generator=self.rng_process, device=self.device))
        w = w.float()
        self.w0[self.step_count == 0, :] = w[self.step_count == 0, :]
        if not self.skip_to_steady_state:
            self.x = bmv(self.A, self.x) + bmv(self.B, u) + w
        else:
            self.x = bsolve(torch.eye(self.n, device=self.device).unsqueeze(0) - self.A, bmv(self.B, u))
        self.step_count += 1
        self.is_done[self.step_count >= self.max_steps] = 2  # 2 for timeout
        self.is_done[torch.logical_not(self.check_in_bound()).nonzero()] = 1   # 1 for failure
        if self.keep_stats:
            done_indices = torch.nonzero(self.is_done.to(dtype=torch.bool) & torch.logical_not(self.already_on_stats), as_tuple=False)
            for i in done_indices:
                self.write_episode_stats(i)
        return self.obs(), -self.safe_cost(), self.done(), self.info()

    def render(self, **kwargs):
        """
        Prints the current state, reference state, and control input for debugging purposes.
        """
        ic(self.x, self.x_ref, self.u)
        avg_cost = (self.cum_cost / self.step_count).cpu().numpy()
        ic(avg_cost)
