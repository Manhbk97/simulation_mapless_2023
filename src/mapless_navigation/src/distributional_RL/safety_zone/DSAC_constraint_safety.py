#!/usr/bin/env python3
import rospy
import numpy as np
# import gtimer as gt
# np.random.bit_generator = np.random._bit_generator
import matplotlib.pyplot as plt
import random
import time
import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from networks import FlattenMlp, QuantileMlp
from collections import deque
from collections import namedtuple
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal
import pytorch_util as ptu
from utils import LinearSchedule
from risk import distortion_de, distortion_fn
from eval_util import create_stats_ordered_dict
import wandb
# from wandb.keras import WandbCallback
# Initialize your W&B project allowing it to sync with TensorBoard
config = dict (
    discount=0.99,
    reward_scale=1.0,
    alpha=0.3,
    policy_lr=3e-4,
    zf_lr=3e-4,
    tau_type='iqn',
    fp_lr=1e-5,
    num_quantiles=32,
    risk_type='neutral',
    risk_param=0.0,
    risk_param_final=None,
    risk_schedule_timesteps=1,
    optimizer_class=optim.Adam,
    soft_target_tau=5e-4,
    target_update_period=1,
    clip_norm=1.0,
    use_automatic_entropy_tuning=True,
  lr_alpha =  0.001,
  input_nn = 24,
  batch_size = 256,
  init_alpha  = 1, 
  environment = "indoor_nana_complex_env",
  tau  = 0.0005,
  buffer_limit   = 300000 , 
  simulation_speed = 3.0,
  End_time_step_peisode = 1300, 
  start_learning = 5000, 
  angular_velocity = "[-1.0, 1.0]" ,
  linear_velocity = "[0.0, 1.0]", 
  Actor= "layer_norm(108_256)_layernorm(256_256)_layernorm(64_108)_2(64_2)",
  min_range= 0.4,
  min_front_range= 0.6,
  min_range_reward = 1.0,
  target_size = 0.2,
  time_end_eps = 2000,
  lidar_specification="120 degree", Hz = 10,
  distance_range = "0.4-3.0",
)
wandb.init(name='DSAC_1213_safety', project="safety_nana_robot",config=config)
import GPUtil
import psutil
from threading import Thread
import time


from env_21_safety_zone import Env
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from collections import OrderedDict
from std_srvs.srv import Empty

from collections import namedtuple
import collections, random


class ReplayBuffer():
    def __init__(self, buffer_limit, DEVICE):
        self.buffer = deque(maxlen=buffer_limit)
        self.dev = DEVICE

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst,Ct_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done, Ct = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            Ct_lst.append(Ct)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])
        # print("S_batch", (s_lst))
        s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.dev)
        a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.dev)
        r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.dev)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.dev)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.dev)
        Ct_batch = torch.tensor(Ct_lst, dtype=torch.float).to(self.dev)

        # r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, Ct_batch

    def size(self):
        return len(self.buffer)
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,num_safe, actor_lr, DEVICE):
        super(PolicyNetwork, self).__init__()


        ### lan1 24 input 
        # self.fc_1 = nn.Linear(state_dim, 256)
        # self.norm1 = nn.LayerNorm(256)
        # self.fc_2 = nn.Linear(256, 256)
        # self.norm2 = nn.LayerNorm(256)
        # self.fc_3 = nn.Linear(256, 64)
        # self.fc_mu = nn.Linear(64, action_dim)
        # self.fc_std = nn.Linear(64, action_dim)
        #108 input
        self.fc_1 = nn.Linear(state_dim + num_safe, 512)
        self.norm1 = nn.LayerNorm(512)
        self.fc_2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.fc_3 = nn.Linear(256, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)
        self.dropout = nn.Dropout(0.25)


        self.lr = actor_lr
        self.dev = DEVICE

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = torch.FloatTensor([1.0, 1.0]).to(self.dev)
        self.min_action = torch.FloatTensor([-1.0, 0]).to(self.dev)
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0
        
        self.to(self.dev)

        # self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, ct):
        # x = F.relu(self.fc_1(x))
        # x = F.relu(self.fc_2(x))
        # x = F.leaky_relu(self.fc_3(x))
        if x.dim()==1:
            x = torch.cat((x,ct), dim = 0)
        else: 
            x = torch.cat((x,ct), dim = 1)
        # print("concatenate x", x.dim())
        # x = self.dropout(x)
        x= F.relu(self.norm1(self.fc_1(x)))
        x = self.dropout(x)
        x= F.relu(self.norm2(self.fc_2(x)))
        x = self.dropout(x)
        x= F.relu(self.fc_3(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state, Ct):
        mean, log_std = self.forward(state, Ct)
        # wandb.log({"angular_mean": mean[0].mean()}) 
        # wandb.log({"velocity_mean": mean[1].mean()}) 
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)  ## (-1,1)
        action = self.action_scale * y_t + self.action_bias
        #print("action:" + str(action))
        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)
        # print("log_prob",log_prob)      

        # print("log_prob1",log_prob.shape)
        std = torch.exp(log_std)
        # log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, mean, log_std, log_prob, std
             

def softmax(x):
    return F.softmax(x, dim=-1)

class SAC_Agent:
    def __init__(
            self,
            fp=None,
            target_fp=None,
            discount=0.99,
            reward_scale=1.0,
            alpha=0.3,
            policy_lr=3e-4,
            zf_lr=3e-4,
            tau_type='iqn',
            fp_lr=1e-5,
            num_quantiles=32,
            risk_type='neutral',
            risk_param=0.00,
            risk_param_final=None,
            risk_schedule_timesteps=1,
            optimizer_class=optim.Adam,
            soft_target_tau=5e-4,
            target_update_period=1,
            clip_norm=1.0,
            use_automatic_entropy_tuning=True,
            target_entropy=-2,
    ):
        super().__init__()
        self.state_dim      = 24
        self.action_dim     = 2       # 
        self.num_quantiles  = num_quantiles
        self.num_safe       = 1 
        self.layer_size     = 256
        self.lr_pi          = 0.0003  # 0.0001
        # self.lr_q           = 0.0003
        self.batch_size     = 256   # 200
        self.buffer_limit   = 300000
        self.tau            = 0.005   # for soft-update of Q using Q-target
        self.init_alpha     = 1    #8
        self.target_entropy = -2     #-self.action_dim  # == -2
        self.discount       = discount
        self.risk_type = risk_type
        self.clip_norm = clip_norm
        
        self.lr_alpha       = 0.001  # 0.001
        self.DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory         = ReplayBuffer(self.buffer_limit, self.DEVICE)
        print("device used : ", self.DEVICE)
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.tau_type = tau_type
        self.num_quantiles = num_quantiles        
        self._n_train_steps_total = 0
        # self.target_entropy = -2 # torch.tensor([-2]).to(self.DEVICE)    #-self.action_dim  # == -2
        # self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        # self.log_alpha.requires_grad = True
        # self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        self.zf_criterion = self.quantile_regression_loss
        

        self.policy  = PolicyNetwork(self.state_dim, self.action_dim, self.num_safe, self.lr_pi, self.DEVICE)
        self.target_policy = PolicyNetwork(self.state_dim, self.action_dim,self.num_safe, self.lr_pi, self.DEVICE)
        # wandb.watch(self.PI)
        # self.policy.load_state_dict(torch.load("/home/manh/catkin_ws/src/mapless_navigation/src/dsac/model/DSAC_nana_0714/sacDSAC_nana_0714_neutral_iqn_108_input_tau_EP1585.pt"))
        # self.target_policy.load_state_dict(torch.load("/home/manh/catkin_ws/src/mapless_navigation/src/dsac/model/DSAC_nana_0714/sacDSAC_nana_0714_neutral_iqn_108_input_tau_EP1585.pt"))

        self.layer_size = 256 
        self.zf1 = QuantileMlp(
                input_size=self.state_dim  + self.action_dim + self.num_safe,
                output_size=1,
                num_quantiles=self.num_quantiles,
                hidden_sizes=[self.layer_size, self.layer_size],
        )
        self.target_zf1 = QuantileMlp(
                input_size=self.state_dim  + self.action_dim + self.num_safe,
                output_size=1,
                num_quantiles=self.num_quantiles,
                hidden_sizes=[self.layer_size, self.layer_size],
        )
        self.target_zf2 = QuantileMlp(
                input_size=self.state_dim  + self.action_dim + self.num_safe,
                output_size=1,
                num_quantiles=self.num_quantiles,
                hidden_sizes=[self.layer_size, self.layer_size],
        )   
        self.zf2 = QuantileMlp(
                input_size = self.state_dim  + self.action_dim + self.num_safe,
                output_size=1,
                num_quantiles = self.num_quantiles,
                hidden_sizes=[self.layer_size, self.layer_size],
        ) 
        target_fp = FlattenMlp(
            input_size=self.state_dim + self.action_dim,
            output_size=num_quantiles,
            hidden_sizes=[self.layer_size // 2 , self.layer_size // 2],
            output_activation=softmax,
        )
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        # self.target_entropy = -np.prod(2).item()  # heuristic value from Tuomas
        # self.log_alpha = ptu.zeros(1, requires_grad=True)
        # self.log_alpha.requires_grad = True
        # self.alpha_optimizer = optimizer_class(
        #     [self.log_alpha],
        #     lr=policy_lr,
        # )
        # self.policy_optimizer = optimizer_class(
        #     self.policy.parameters(),
        #     lr=policy_lr,
        # )
        self.policy_optimizer=optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.zf1_optimizer = optimizer_class(
            self.zf1.parameters(),
            lr=zf_lr,
        )
        self.zf2_optimizer = optimizer_class(
            self.zf2.parameters(),
            lr=zf_lr,
        )

        self.fp = fp
        self.target_fp = target_fp
        if self.tau_type == 'fqf':
            self.fp_optimizer = optimizer_class(
                self.fp.parameters(),
                lr=fp_lr,
            )

        self.risk_schedule = LinearSchedule(risk_schedule_timesteps, risk_param,
                                            risk_param if risk_param_final is None else risk_param_final)
        
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

            # wandb.watch(self.Q1)
        # self.policy.load_state_dict(torch.load("/home/manh/catkin_ws/src/mapless_navigation/src/distributional_RL/dsac/model/21_input_DSAC_nana_rastech_env_0808_auxiliary/sac21_input_DSAC_nana_rastech_env_0808_auxiliary_EP1725.pt"))  # 90%%
        # self.target_zf1.load_state_dict(torch.load("/home/manh/catkin_ws/src/mapless_navigation/src/distributional_RL/dsac/model/21_input_DSAC_nana_rastech_env_0808_auxiliary/Q1target21_input_DSAC_nana_rastech_env_0808_auxiliary.pt"))
        # self.zf1.load_state_dict(torch.load("/home/manh/catkin_ws/src/mapless_navigation/src/distributional_RL/dsac/model/21_input_DSAC_nana_rastech_env_0808_auxiliary/Q121_input_DSAC_nana_rastech_env_0808_auxiliary.pt"))
        # self.zf2.load_state_dict(torch.load("/home/manh/catkin_ws/src/mapless_navigation/src/distributional_RL/dsac/model/21_input_DSAC_nana_rastech_env_0808_auxiliary/Q221_input_DSAC_nana_rastech_env_0808_auxiliary.pt"))
        # self.target_zf2.load_state_dict(torch.load("/home/manh/catkin_ws/src/mapless_navigation/src/distributional_RL/dsac/model/21_input_DSAC_nana_rastech_env_0808_auxiliary/Q2target21_input_DSAC_nana_rastech_env_0808_auxiliary.pt"))



        
    def quantile_regression_loss(self, input, target, tau, weight):
        """
        input: (N, T)
        target: (N, T)
        tau: (N, T)
        """
        input = input.unsqueeze(-1)
        target = target.detach().unsqueeze(-2)
        tau = tau.detach().unsqueeze(-1)
        weight = weight.detach().unsqueeze(-2)
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
        sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
        # print("tau", tau) #[256, 64, 1]
        # print("sign", sign) # [256, 64, 64]
        # print("weight", weight) # [256, 1, 64]
        rho = torch.abs(tau - sign) * L * weight
        return rho.sum(dim=-1).mean()

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau.to(self.DEVICE), tau_hat.to(self.DEVICE), presum_tau.to(self.DEVICE)




    def choose_action(self, s, ct):
        with torch.no_grad():
            action, _ , _ , log_prob, std = self.policy.sample(s.to(self.DEVICE),ct.to(self.DEVICE) )
        return action, log_prob, std



    def train_agent(self,ep, collision):
        mini_batch = self.memory.sample(self.batch_size)
        obs, actions, rewards, next_obs, terminals, Ct = mini_batch

        # rewards = batch['rewards']
        # terminals = batch['terminals']
        # obs = batch['observations']
        # actions = batch['actions']
        # next_obs = batch['next_observations']

        """
        Update Alpha
        """
        # print("dimension obs and Ct", obs.size(), Ct.size())
        new_actions, policy_mean, policy_log_std, log_pi, _ = self.policy.sample(obs, Ct)
        # print("log_pi", log_pi)
        log_pi = log_pi.unsqueeze(1)
        log_pi = log_pi.permute(0, 2, 1)
        # print("log_pi", log_pi.shape)    #(256,2,1)
        # print("self.log_alpha", self.log_alpha)
        # print("self.target_entropy", self.target_entropy)
        self.log_alpha = self.log_alpha.to(self.DEVICE)
        
        if self.use_automatic_entropy_tuning:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            alpha = self.log_alpha.exp()
            # print("log_alpha", self.log_alpha)
            # print("update alpha", alpha)
        else:
            alpha_loss = 0
            alpha = self.alpha
        """
        Update ZF
        """
        with torch.no_grad():
            
            new_next_actions, _, _, new_log_pi, _ = self.target_policy.sample(next_obs, Ct)
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs, new_next_actions, fp=self.target_fp)
            # print("next_obs", next_obs)
            # print("next_obs", new_next_actions)
            # print("next tau hat ", next_tau_hat.shape)
            next_obs = torch.cat((next_obs, Ct),1)
            target_z1_values = self.target_zf1(next_obs, new_next_actions, next_tau_hat)
            target_z2_values = self.target_zf2(next_obs, new_next_actions, next_tau_hat)
            # print("target_z2_values", target_z2_values.shape)
            # print("alpha * new_log_pi", (alpha * new_log_pi).shape)
            target_z = torch.min(target_z1_values, target_z2_values)
            # target_z = torch.add(target_z1_values, target_z2_values)
            target_z_values = torch.stack([(target_z[:,:,i] - alpha * new_log_pi) for i in range(0,32)])
            target_z_values = target_z_values.permute(1, 2, 0)
            # print("target_z_values", target_z_values.shape)  #(m,N_action,N_quantiles) 
            target_z_values = torch.reshape(target_z_values, (target_z_values.shape[0], -1)) #(m,N_action x N_quantiles) 
            # for i in range(0,32):
            #     target_z_values_i = torch.min(target_z1_values, target_z2_values)[:,:,i] - alpha * new_log_pi
            # z_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_z_values
            z_target = self.reward_scale * rewards + terminals * self.discount * target_z_values + 0.05*Ct
        tau, tau_hat, presum_tau = self.get_tau(obs, actions, fp=self.fp)
        # print("tau", tau.shape)
        obs = torch.cat((obs, Ct),1)
        z1_pred = self.zf1(obs, actions, tau_hat)
        z1_pred = torch.reshape(z1_pred, (z1_pred.shape[0], -1))
        z2_pred = self.zf2(obs, actions, tau_hat)
        z2_pred = torch.reshape(z2_pred, (z2_pred.shape[0], -1))
        # tau_hat = torch.stack([(tau_hat) for i in range(0,2)])
        tau_hat = torch.cat((tau,tau), 1)
        next_presum_tau = torch.cat((next_presum_tau,next_presum_tau),1)
        zf1_loss = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
        zf2_loss = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)

        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        self.zf1_optimizer.step()
        
        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        self.zf2_optimizer.step()

        """
        Update FP
        """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 0.5 * (2 * self.zf1(obs, actions, tau[:, :-1]) - z1_pred[:, :-1] - z1_pred[:, 1:] +
                                2 * self.zf2(obs, actions, tau[:, :-1]) - z2_pred[:, :-1] - z2_pred[:, 1:])
                dWdtau /= dWdtau.shape[0]  # (N, T-1)


            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()

        """
        Update Policy
        """
        risk_param = self.risk_schedule(self._n_train_steps_total)
        # obs = torch.cat((obs, Ct),1)
        if self.risk_type == 'VaR':
            tau_ = ptu.ones_like(rewards) * risk_param
            q1_new_actions = self.zf1(obs, new_actions, tau_)
            q2_new_actions = self.zf2(obs, new_actions, tau_)
        else:
            with torch.no_grad():
                new_tau, new_tau_hat, new_presum_tau = self.get_tau(obs, new_actions, fp=self.fp)
            # print("new_presum_tau", new_presum_tau)
            # print("new_tau_hat", new_tau_hat)
            z1_new_actions = self.zf1(obs, new_actions, new_tau_hat)
            z2_new_actions = self.zf2(obs, new_actions, new_tau_hat)
            new_presum_tau = new_presum_tau.unsqueeze(1)
            if self.risk_type in ['neutral', 'std']:
                q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=1, keepdims=True)
                if self.risk_type == 'std':
                    q1_std = new_presum_tau * (z1_new_actions - q1_new_actions).pow(2)
                    q2_std = new_presum_tau * (z2_new_actions - q2_new_actions).pow(2)
                    q1_new_actions -= risk_param * q1_std.sum(dim=1, keepdims=True).sqrt()
                    q2_new_actions -= risk_param * q2_std.sum(dim=1, keepdims=True).sqrt()
            else:
                with torch.no_grad():
                    risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
        # Sum_Ct = torch.sum(Ct, dim=1, keepdims=True)
        # q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        print("collision rate", collision)
        # if collision >= 0.05:
        q_new_actions = torch.add(q1_new_actions, q2_new_actions)
        # else:
        # q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        wandb.log({"q_new_actions": q_new_actions.mean()})  
        # print("anpha", alpha)
        wandb.log({"anpha": alpha})  
        # print("q_new_actions", q_new_actions.size())
        # print("Ct", Ct.size())
        Ct = Ct.unsqueeze(1)
        print("Ct", Ct.size())
        # print("log_pi", log_pi.size())
        # print("log_pi", log_pi)

        policy_loss = (alpha  * log_pi - q_new_actions).mean()
        # if policy_loss < 0.05 and policy_loss > -0.5:
        #     print("loss zero ")
        #     policy_loss = policy_loss - 0.5
        # gt.stamp('preback_policy', unique=False)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
        self.policy_optimizer.step()
        # gt.stamp('backward_policy', unique=False)
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf1, self.target_zf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf2, self.target_zf2, self.soft_target_tau)
            if self.tau_type == 'fqf':
                ptu.soft_update_from_to(self.fp, self.target_fp, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        wandb.log({"ZF1 Loss": zf1_loss})  
        wandb.log({"ZF2 Loss": zf2_loss})
        wandb.log({"Policy Loss": policy_loss})  
        wandb.log({"Alpha Loss": alpha_loss}) 

                
        self._n_train_steps_total += 1
        print("train_steps_total", self._n_train_steps_total)



class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            # print("CPU percentage: ", psutil.cpu_percent())
            # print('CPU virtual_memory used:', psutil.virtual_memory()[2], "\n")
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        



if __name__ == '__main__':
    rospy.init_node('mobile_robot_sac')
    
    GPU_CPU_monitor = Monitor(60)
    
    date = 'DSAC_1213_safety'
    save_dir = "/home/manh/catkin_ws/src/mapless_navigation/src/distributional_RL/dsac/model/" + date 
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir += "/" 
    
    writer = SummaryWriter('SAC_log/'+date)
    
    env = Env()
    agent = SAC_Agent()
    
    EPISODE = 2000000
    MAX_STEP_SIZE = 3000
    episode = 0
    sim_rate = rospy.Rate(10)   

    score = 0.0
    print_once = True
    scores_, episodes_, average_ , = [], [], [] # used in matplotlib plots
    collision_rate_,collision_ = [],[]
    state = env.reset()
    Ct = np.array([0.0])
    score, done = 0.0 , False 
    collision = 0.0
    for EP in range(EPISODE):   
        # print("state", state.shape)
        
        Ct = torch.FloatTensor(Ct)
        print("Ct", Ct)
        action, log_prob, std = agent.choose_action(torch.FloatTensor(state),Ct )
        # print("action:" + str(action))
        action = action.detach().cpu().numpy()
        wandb.log({"angular_std": std[0].mean()}) 
        wandb.log({"velocity_std": std[1].mean()}) 
        # print("action", action)
        # print("action", type(action))
        #action = [a[0]*1.5 , (a[1]+1)/4]
        
        state_prime, reward, done, arrival, Ct = env.step(action)
        #print("done episode:" + str(done))
        wandb.log({"linear_velocity": action[1]})  
        wandb.log({"angular_velocity": action[0]})  
        agent.memory.put((state, action, reward, state_prime, done, Ct))
        score += reward
        state = state_prime
        #print("state:" + str(state)) (4,1)
        if arrival==False:
            collision = 1
        else:
            collision = 0

        if done:
            episode += 1
            scores_.append(score)
            
            episodes_.append(episode)
            average_.append(sum(scores_[-50:]) / len(scores_[-50:]))  
            collision_.append(collision)
            collision_rate_.append(sum(collision_[-50:]) / len(collision_[-50:]))  
            score_collision = sum(collision_[-20:]) / len(collision_[-20:])
            writer.add_scalar("Score", score, EP) 
            print("EP:{}, Score:{:.1f}".format(EP, score), "\n")
            wandb.log({"episode": episode})
            wandb.log({"Scores": score})
            wandb.log({"Average": average_[-1]}) 
            wandb.log({"collision rate": collision_rate_[-1]}) 
            if episode % 5 == 0  and episode > 100: 
                model = save_dir + "sac"+date+"_EP"+str(episode)+".pt"
                critic_target1 = save_dir + "Q1target"+date+".pt"
                critic_target2 = save_dir + "Q2target"+date+".pt"
                critic1 = save_dir + "Q1"+date+".pt"
                critic2 = save_dir + "Q2"+date+".pt"
                torch.save(agent.policy.state_dict(), model)
                if episode % 10 ==0:
                    torch.save(agent.target_zf1.state_dict(), critic_target1)
                    torch.save(agent.target_zf2.state_dict(), critic_target2)
                    torch.save(agent.zf1.state_dict(), critic2)
                    torch.save(agent.zf2.state_dict(), critic1)
            state, done, score = env.reset() , False, 0
        # print("memory",agent.memory.size())
        if agent.memory.size()>agent.batch_size:
            if print_once: 
                print("start_learning")
                print_once = False
            agent.train_agent(EP, score_collision)
            
        if agent.memory.size()<=10000:
            # sim_rate.sleep()   
            time.sleep(0.002)      
    
        








    
    
    
    
    
    
    
    
    
    
    
    
    

