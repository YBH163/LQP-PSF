from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 
from torch import nn
import numpy as np
import gym

class A2CAgent(a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = (self.has_central_value and self.use_experimental_cv) \
                            or (not self.has_phasic_policy_gradients and not self.has_central_value) 
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        # 从 input_dict 中获取旧的值预测、旧的动作对数概率、优势函数、旧的均值和标准差、回报、动作、观察值等
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)    # 对观察值进行预处理
        gt_batch = input_dict.get("ground_truths", None)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        # 创建一个包含训练所需信息的 batch_dict
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        # 如果模型使用RNN，获取RNN掩码、状态、序列长度和完成标志
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            batch_dict['dones'] = input_dict['dones']
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision): # 使用 torch.cuda.amp.autocast 进行混合精度训练，减少内存使用并提高性能
            # 前向传播，得到新的动作对数概率、值估计、熵、均值、标准差等
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            prediction = res_dict.get("prediction", None)
            autonomous_losses = res_dict.get("autonomous_losses", None)

            # 计算演员损失（a_loss），这是策略梯度损失，衡量新旧动作对数概率的差异
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                # 计算评论家损失（c_loss），这是值函数损失，衡量预测值与回报的差异
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                # 计算边界损失（b_loss），用于限制动作的输出范围
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            if prediction is not None:
                # 如果存在预测任务，计算监督损失（s_loss），这是预测值与真实值的差异
                s_loss = (prediction - gt_batch) ** 2

            loss_terms = [a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)]
            if prediction is not None:
                loss_terms += [s_loss]
            if autonomous_losses is not None:
                if prediction is None:
                    loss_terms += [torch.tensor(0.)]  # placeholder for supervised loss
                loss_terms += list(autonomous_losses.values())
            losses, sum_mask = torch_ext.apply_masks(loss_terms, rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            if prediction is not None:
                s_loss = losses[4]
            if autonomous_losses is not None:
                for (i, k) in enumerate(autonomous_losses):
                    autonomous_losses[k] = losses[5 + i]

            # Ignore normal actor loss if it is prescribed in autonomous_losses that only imitation loss should be used
            if autonomous_losses is not None and 'imitation_only' in autonomous_losses:
                coef_a_loss = 0.0
            else:
                coef_a_loss = 1.0
            # 将所有损失项组合成总损失，包括演员损失、评论家损失、熵损失和边界损失
            loss = coef_a_loss * a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            if prediction is not None:
                loss += s_loss
            if autonomous_losses is not None:
                for l in autonomous_losses.values():
                    loss += l
            
            # 如果使用多GPU，使用 self.optimizer.zero_grad() 清零梯度；否则，手动将模型参数的梯度设置为None
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        # 使用 self.scaler.scale(loss).backward() 进行反向传播，计算梯度。这里 self.scaler 可能是一个梯度缩放器，用于混合精度训练中的梯度缩放
        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        # 进行梯度裁剪和参数更新
        self.trancate_gradients_and_step()

        with torch.no_grad():
            # 使用 torch_ext.policy_kl 计算新旧策略之间的KL散度，用于监控策略更新的幅度
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        # 将损失、KL散度、学习率、梯度裁剪系数等存储在 self.train_result 中
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        if prediction is not None:
            self.train_result += (s_loss,)
        if autonomous_losses is not None:
            if prediction is None:
                self.train_result += (torch.tensor(0.),)  # placeholder for supervised loss
            self.train_result += (autonomous_losses,)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
