import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.dqn import DQNAgent

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, sf_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 13 * 13

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.compress = nn.Sequential(
            nn.Linear(self.repr_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, sf_dim),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.reshape(h.shape[0], -1)
        h = self.compress(h)
        return h
 
class CriticSF(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, hidden_dim, sf_dim):
        super().__init__()

        self.obs_type = obs_type
        self.action_dim = action_dim

        def make_q():
            q_layers = []
            q_layers += [nn.Linear(obs_dim, hidden_dim), nn.ReLU(inplace=True)]
            if obs_type == "pixels":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            q_layers += [nn.Linear(hidden_dim, action_dim*sf_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, task):
        """
        Compute Q-values for all actions given observations.

        Args:
            obs: Tensor of shape [B, obs_dim] or [1, obs_dim]
            action: Tensor of shape [num_actions]

        Returns:
            q1: Q-values from Q1 network (shape: [B, num_actions])
            q2: Q-values from Q2 network (shape: [B, num_actions])
        """
        batch_size = obs.shape[0]
        device = obs.device
        if isinstance(task, np.ndarray) or isinstance(task, list):
            task = torch.as_tensor(task, device=device) #[B, sf_dim]
        
        # Compute Q-values
        q1 = self.Q1(obs) #[B, A*sf_dim]
        q2 = self.Q2(obs) #[B, A*sf_dim]
        q1 = q1.view(q1.shape[0], self.action_dim, -1) #[B, A, sf_dim]
        q2 = q2.view(q2.shape[0], self.action_dim, -1) #[B, A, sf_dim]

        q1 = torch.einsum('bij,bij->bi', task.unsqueeze(1), q1).view(batch_size, self.action_dim) #[B, A]
        q2 = torch.einsum('bij,bij->bi', task.unsqueeze(1), q2).view(batch_size, self.action_dim) #[B, A]

        return q1, q2

class SFTraditionalAgent(DQNAgent):
    def __init__(
        self,
        update_task_every_step,
        sf_dim,
        num_init_steps,
        lr_task,
        normalize_basis_features,
        normalize_task_params,
        **kwargs
    ):

        self.sf_dim = sf_dim
        self.update_task_every_step = update_task_every_step
        self.num_init_steps = num_init_steps
        self.normalize_basis_features = normalize_basis_features
        self.normalize_task_params = normalize_task_params

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        super().__init__(**kwargs)

        if self.obs_type == "pixels":
            self.aug = nn.Identity()
            self.encoder = Encoder(self.obs_shape, self.feature_dim, self.sf_dim).to(
                self.device
            )
            # self.obs_dim = self.sf_dim + self.meta_dim
            self.obs_dim = self.sf_dim
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = self.obs_shape[0] + self.meta_dim
            self.encoder_opt = None


        # overwrite critic with critic sf
        self.critic = CriticSF(
            self.obs_type,
            self.sf_dim,
            self.action_dim,
            self.hidden_dim,
            self.sf_dim,
        ).to(self.device)

        self.critic_target = CriticSF(
            self.obs_type,
            self.sf_dim,
            self.action_dim,
            self.hidden_dim,
            self.sf_dim,
        ).to(self.device)


        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.task_params = torch.randn(self.sf_dim).to(device=self.device)
        
        # set solved_meta to the value of the task_params
        with torch.no_grad():
            self.solved_meta = OrderedDict()
            self.solved_meta["task"] = self.task_params.detach().cpu().numpy()
        self.train()
        self.critic_target.train()
        # self.featureNet.train()

    def get_meta_specs(self):
        """
        Meta dimension always follows the successor feature dimension.
        """
        return (specs.Array((self.sf_dim,), np.float32, "task"),)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        task = torch.randn(self.sf_dim)
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta["task"] = task
        return meta

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, next_obs, dones, task = replay_iter.sample()
        discount = torch.ones_like(reward) * (1.0 - dones.float())

        full_obs, full_reward = replay_iter.get_full_observation_reward()

        task = self.task_params.unsqueeze(0).repeat(task.shape[0],1)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            full_next_obs = self.aug_and_encode(full_obs)
            if self.normalize_basis_features:
                next_basis_features = F.normalize(full_next_obs, p=2, dim=-1)
            else:
                next_basis_features = next_obs

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update meta
        if step % self.update_task_every_step == 0:
            metrics.update(
                self.regress_meta_grad_descent(
                    full_reward, next_basis_features
                )
            )

        # update critic
        metrics.update(
            self.update_critic(
                obs, action, reward, discount, next_obs, task.detach(), step
            )
        )

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def regress_meta_grad_descent(
        self, reward, next_basis_features=None
    ):
        metrics = dict()

        # set solved_meta to the value of the task_params
        with torch.no_grad():
            self.task_params = torch.FloatTensor(np.linalg.lstsq(next_basis_features.detach().cpu().numpy(), reward.detach().cpu().numpy(), rcond=None)[0]).to(self.device).squeeze()
            meta = self.task_params.detach().cpu().numpy()
            # normalize solved meta using l2 norm
            # meta = meta / np.linalg.norm(meta)
            self.solved_meta = OrderedDict()
            self.solved_meta["task"] = meta

        return metrics
    
    def update_critic(self, obs, action, reward, discount, next_obs, task, step):
        """diff is critic takes task as input"""
        metrics = dict()

        with torch.no_grad():
            # Step 1: Get Q-values for all actions from current (online) critic for next_obs
            # next_h = self.featureNet(next_obs, [])[0]
            next_Q1, next_Q2 = self.critic(next_obs, task)  # shape: [B, A]
            # Step 2: Double Q-learning action selection
            next_Q = torch.min(next_Q1, next_Q2)  # shape: [B, A]
            next_action = torch.argmax(next_Q, dim=1, keepdim=True)  # shape: [B, 1]

            target_Q1, target_Q2 = self.critic_target(next_obs, task)
            target_Q1 = target_Q1.gather(1, next_action).squeeze(1)  # shape: [B]
            target_Q2 = target_Q2.gather(1, next_action).squeeze(1)  # shape: [B]
            target_V = torch.min(target_Q1, target_Q2)  # shape: [B]

            # Step 4: Compute target Q-value using Bellman backup
            target_Q = reward.squeeze() + discount.squeeze() * target_V  # shape: [B]

        # Step 5: Get current Q-values for the actions taken in obs
        # h = self.featureNet(obs, action)[0]
        current_Q1, current_Q2 = self.critic(obs, task)
        current_Q1 = current_Q1.gather(1, action.to(torch.int64)).squeeze(1)  # shape: [B]
        current_Q2 = current_Q2.gather(1, action.to(torch.int64)).squeeze(1)  # shape: [B]

        # Step 6: Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = current_Q1.mean().item()
            metrics["critic_q2"] = current_Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        # self.feature_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        self.critic_opt.step()
        # self.feature_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    
    @torch.no_grad()
    def solved_meta(self):
        return self.solved_meta

    @torch.no_grad()
    def num_params(self):
        all_params = (
            + list(self.critic.parameters())
            # + list(self.featureNet.parameters())
            + list(self.encoder.parameters())
        )

        num_parameters = sum([params.numel() for params in all_params])
        num_parameters += self.meta_dim

        return num_parameters
