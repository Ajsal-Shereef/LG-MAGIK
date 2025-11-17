from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from architectures.common_utils import get_train_transform_cnn, get_train_transform_mlp

import utils

from absl import logging


class Encoder(nn.Module):
    def __init__(self, obs_shape):
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

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type
        self.action_dim = action_dim

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.critic_trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.critic_trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            trunk_dim = hidden_dim
        def make_q():
            q_layers = []
            q_layers += [nn.Linear(trunk_dim, hidden_dim), nn.ReLU(inplace=True)]

            if obs_type == "pixels":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]

            q_layers += [nn.Linear(hidden_dim, 1)]

            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, meta=None, conditioner=None):
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

        if self.obs_type == "pixels":
            # Extract features for each observation
            h = self.critic_trunk(obs)  # [B, hidden_dim]
            h = h.unsqueeze(1).repeat(1, self.action_dim, 1)  # [B, A, hidden_dim]
            h = h.view(-1, h.shape[-1])  # [B*A, hidden_dim]

            # Create one-hot encoded actions for all actions
            actions = torch.eye(self.action_dim)[torch.arange(self.action_dim)].repeat(batch_size, 1, 1).to(device)  # [B, A, A]
            actions = actions.view(-1, self.action_dim)  # [B*A, A]

            h = torch.cat([h, actions], dim=-1)  # [B*A, hidden_dim + A]
        else:
            # For vector obs type, need to tile obs for each action
            obs_expanded = obs.unsqueeze(1).repeat(1, self.action_dim, 1)  # [B, A, obs_dim]
            obs_expanded = obs_expanded.view(-1, obs.shape[-1])  # [B*A, obs_dim]

            # One-hot encode all actions
            actions = torch.eye(self.action_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, A, A]
            actions = actions.view(-1, self.action_dim)  # [B*A, A]

            inpt = torch.cat([obs_expanded, actions], dim=-1)  # [B*A, obs_dim + A]
            h = self.critic_trunk(inpt)  # [B*A, hidden_dim]

        # Compute Q-values
        q1 = self.Q1(h).view(batch_size, self.action_dim)
        q2 = self.Q2(h).view(batch_size, self.action_dim)

        return q1, q2

class DQNAgent:
    def __init__(
        self,
        obs_type,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        batch_size,
        stddev_clip,
        init_critic,
        use_wandb,
        update_encoder,
        meta_dim=0,
        **kwargs
    ):
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = OrderedDict()
        self.update_encoder = update_encoder
        self.meta_dim = meta_dim
        self.batch_size = batch_size
        self.action_conditioner = None

        self.solved_meta["task"] = []

        # models
        if obs_type == "pixels":
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim
        self.critic = Critic(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)

        self.critic_target = Critic(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        if obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        meta  = OrderedDict()
        meta["task"] = []
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode, epsilon=0):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        h = self.encoder(obs)
        Q1, Q2 = self.critic(h, np.array([meta]), self.action_conditioner)
        Q = torch.min(Q1, Q2)
        action = torch.argmax(Q.squeeze(), dim=-1).unsqueeze(0)
        if (step < self.num_expl_steps and not eval_mode) or np.random.rand() < epsilon:
            action = torch.randint(self.action_dim, (1,), device=self.device)

        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            # Step 1: Get Q-values for all actions from current (online) critic for next_obs
            next_Q1, next_Q2 = self.critic(next_obs)  # shape: [B, A]
            # Step 2: Double Q-learning action selection
            next_Q = torch.min(next_Q1, next_Q2)  # shape: [B, A]
            next_action = torch.argmax(next_Q, dim=1, keepdim=True)  # shape: [B, 1]

            # Step 3: Use target critic to get Q-values at next_action
            target_Q1, target_Q2 = self.critic_target(next_obs)  # shape: [B, A]
            target_Q1 = target_Q1.gather(1, next_action).squeeze(1)  # shape: [B]
            target_Q2 = target_Q2.gather(1, next_action).squeeze(1)  # shape: [B]
            target_V = torch.min(target_Q1, target_Q2)  # shape: [B]

            # Step 4: Compute target Q-value using Bellman backup
            target_Q = reward.squeeze() + discount.squeeze() * target_V  # shape: [B]

        # Step 5: Get current Q-values for the actions taken in obs
        current_Q1, current_Q2 = self.critic(obs)  # shape: [B, A]
        current_Q1 = current_Q1.gather(1, action.to(torch.int64)).squeeze(1)  # shape: [B]
        current_Q2 = current_Q2.gather(1, action.to(torch.int64)).squeeze(1)  # shape: [B]

        # Step 6: Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optional logging
        if self.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = current_Q1.mean().item()
            metrics["critic_q2"] = current_Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # Step 7: Optimize
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()
        # import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, next_obs, dones, _ = replay_iter.sample()
        discount = torch.ones_like(reward) * (1.0 - dones.float())

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    @torch.no_grad()
    def solved_meta(self):
        return self.solved_meta

    @torch.no_grad()
    def num_params(self):
        all_params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        return sum(p.numel() for p in all_params)
