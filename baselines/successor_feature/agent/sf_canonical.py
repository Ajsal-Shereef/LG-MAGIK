import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.sf_simple import SFSimpleAgent


class CriticSF(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, hidden_dim, sf_dim):
        super().__init__()

        self.obs_type = obs_type

        # a small difference compared to aps is that aps uses an additional state_feat_net
        # whereas we directly use the trunk to get the features. Therefore, to get the
        # basis features dim to match the sf_dim, we include an additional linear layer.

        self.obs_type = obs_type
        self.action_dim = action_dim

        def make_q():
            q_layers = []
            q_layers += [nn.Linear(obs_dim, hidden_dim), nn.ReLU(inplace=True)]
            if obs_type == "pixels":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            q_layers += [nn.Linear(hidden_dim, sf_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, h, task, conditioner=None):
        """
        Compute Q-values for all actions given observations.

        Args:
            obs: Tensor of shape [B, obs_dim] or [1, obs_dim]
            action: Tensor of shape [num_actions]

        Returns:
            q1: Q-values from Q1 network (shape: [B, num_actions])
            q2: Q-values from Q2 network (shape: [B, num_actions])
        """
        batch_size = h.shape[0]
        device = h.device

        if isinstance(task, np.ndarray) or isinstance(task, list):
            task = torch.as_tensor(task, device=device) #[B, sf_dim]
        
        if self.obs_type == "pixels":
            # Extract features for each observation
            h = h.unsqueeze(1).repeat(1, self.action_dim, 1)  # [B, A, hidden_dim]
            h = h.view(-1, h.shape[-1])  # [B*A, hidden_dim]

            # Create one-hot encoded actions for all actions
            actions = torch.eye(self.action_dim)[torch.arange(self.action_dim)].repeat(batch_size, 1, 1).to(device)  # [B, A, A]
            actions = actions.view(-1, self.action_dim)  # [B*A, A]

            h = conditioner(h, actions)  # [B*A, hidden_dim + A]
        else:
            # For vector obs type, need to tile obs for each action
            obs_expanded = h.unsqueeze(1).repeat(1, self.action_dim, 1)  # [B, A, obs_dim]
            obs_expanded = obs_expanded.view(-1, h.shape[-1])  # [B*A, obs_dim]

            # One-hot encode all actions
            actions = torch.eye(self.action_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, A, A]
            actions = actions.view(-1, self.action_dim)  # [B*A, A]

            inpt = torch.cat([obs_expanded, actions], dim=-1)  # [B*A, obs_dim + A]
            h = self.critic_trunk(inpt)  # [B*A, hidden_dim]

        # Compute Q-values
        sf1 = self.Q1(h).view(batch_size, self.action_dim, -1)  # [B, A, sf_dim]
        sf2 = self.Q2(h).view(batch_size, self.action_dim, -1)  # [B, A, sf_dim]

        # task = task.unsqueeze(1).repeat(1, self.action_dim, 1).to(device)
        q1 = torch.einsum('bij,bij->bi', task.unsqueeze(1), sf1).view(batch_size, self.action_dim) #[B, A]
        q2 = torch.einsum('bij,bij->bi', task.unsqueeze(1), sf2).view(batch_size, self.action_dim) #[B, A]


        return q1, q2, sf1, sf2
    
class Actionconditioner(nn.Module):
    def __init__(self, feature_dim, conditioning_dim):
        """
        FiLM layer applies feature-wise affine transformations using conditioning input.

        Args:
            feature_dim (int): Number of features to apply FiLM over (e.g., channels in CNN).
            conditioning_dim (int): Dimensionality of the input used to generate FiLM parameters.
        """
        super(Actionconditioner, self).__init__()
        self.film_generator = nn.Linear(conditioning_dim, 2 * feature_dim)

    def forward(self, x, conditioning_input):
        """
        Args:
            x (Tensor): Input features of shape [B, C, H, W] or [B, C] (for MLPs).
            conditioning_input (Tensor): Conditioning input of shape [B, conditioning_dim].

        Returns:
            Tensor: FiLM-modulated features.
        """
        if conditioning_input.shape[-1] != self.film_generator.in_features:
            conditioning_input = torch.eye(self.film_generator.in_features)[conditioning_input.squeeze().cpu().to(torch.int64)].to(conditioning_input.device)
        gamma_beta = self.film_generator(conditioning_input)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        
        # Reshape for broadcasting
        if x.dim() == 4:  # [B, C, H, W]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class SFCanonicalAgent(SFSimpleAgent):
    def __init__(
        self,
        update_task_every_step,
        sf_dim,
        num_init_steps,
        lr_task,
        normalize_basis_features,
        **kwargs
    ):
        self.sf_dim = sf_dim
        self.update_task_every_step = update_task_every_step
        self.num_init_steps = num_init_steps
        self.lr_task = lr_task

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        # create actor and critic
        super().__init__(
            update_task_every_step,
            sf_dim,
            num_init_steps,
            lr_task,
            normalize_basis_features,
            **kwargs
        )
        self.action_conditioner = Actionconditioner(self.sf_dim, self.action_dim).to(self.device)
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
        self.action_conditioner_opt = torch.optim.Adam(self.action_conditioner.parameters(), lr=self.lr)

        self.train()
        self.critic_target.train()

    def update_critic(
        self, obs, action, reward, discount, next_obs, task, step, basis_features
    ):
        """
        critic here is the sf-td loss. We learn both the basis features and the successor features at the same
        time.
        """
        metrics = dict()

        with torch.no_grad():
            # Step 1: Get Q-values for all actions from current (online) critic for next_obs
            next_Q1, next_Q2, _, _ = self.critic(next_obs, task, self.action_conditioner)  # shape: [B, A]
            # Step 2: Double Q-learning action selection
            next_Q = torch.min(next_Q1, next_Q2)  # shape: [B, A]
            next_action = torch.argmax(next_Q, dim=1, keepdim=True)  # shape: [B, 1]
            target_Q1, target_Q2, target_sf1, target_sf2 = self.critic_target(next_obs, task, self.action_conditioner)
            actions_expanded = next_action.unsqueeze(-1).expand(-1, -1, target_sf1.size(-1))
            target_sf1 = torch.gather(target_sf1, dim=1, index=actions_expanded)  # shape: [B]
            target_sf2 = torch.gather(target_sf2, dim=1, index=actions_expanded)  # shape: [B]

            # compute the l2 norm of the target sf
            target_sf1_norm = torch.norm(target_sf1, p=2, dim=-1)
            target_sf2_norm = torch.norm(target_sf2, p=2, dim=-1)
            sf_norm_compare = target_sf1_norm < target_sf2_norm

            target_sf = []

            # for rows in sf_norm_compare if true then target_sf1 else target_sf2. The result is a tensor
            # of shape (batch_size, sf_dim)
            for idx, row in enumerate(sf_norm_compare):
                if row:
                    target_sf.append(target_sf1[idx, :])
                else:
                    target_sf.append(target_sf2[idx, :])

            target_sf = torch.stack(target_sf)
            # conditioned_basis_features = self.action_conditioner(obs, action)
            target = basis_features + (discount * target_sf.squeeze())

        Q1, Q2, SF1, SF2 = self.critic(obs, task, self.action_conditioner)
        actions_expanded = action.to(torch.int64).unsqueeze(-1).expand(-1, -1, SF1.size(-1))
        SF1 = torch.gather(SF1, dim=1, index=actions_expanded)  # shape: [B]
        SF2 = torch.gather(SF2, dim=1, index=actions_expanded)  # shape: [B]
        critic_loss = F.mse_loss(SF1.squeeze(), target) + F.mse_loss(SF2.squeeze(), target)

        if self.use_tb or self.use_wandb:
            metrics["critic_target_sf"] = target_sf.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_sf1"] = SF1.mean().item()
            metrics["critic_sf2"] = SF2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            self.action_conditioner_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
            self.action_conditioner_opt.step()
        return metrics

    def act(self, obs, meta, step, eval_mode, epsilon=0):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        Q1, Q2, _, _ = self.critic(h, np.array([meta]), self.action_conditioner)
        Q = torch.min(Q1, Q2)
        action = torch.argmax(Q.squeeze(), dim=-1).unsqueeze(0)
        if (step < self.num_expl_steps and not eval_mode) or np.random.rand() < epsilon:
            action = torch.randint(self.action_dim, (1,), device=self.device)

        return action.cpu().numpy()[0]

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, next_obs, dones, task = replay_iter.sample()
        discount = torch.ones_like(reward) * (1.0 - dones.float())

        task = self.task_params.unsqueeze(0).repeat(task.shape[0],1)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.normalize_basis_features:
            conditioned_basis_features = self.action_conditioner(obs, action)
            basis_features = F.normalize(conditioned_basis_features, p=2, dim=-1)
        else:
            basis_features = self.action_conditioner(obs, action)
            # basis_features = obs

        next_obs = next_obs.detach()

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # normalize task
        task_normalized = F.normalize(task, p=2, dim=-1)

        # extend observations with normalized task
        # obs = torch.cat([obs, task_normalized], dim=1)
        # next_obs = torch.cat([next_obs, task_normalized], dim=1)

        # update critic which includes the sf-td loss, reward prediction loss and the critic loss
        metrics.update(
            self.update_critic(
                obs,
                action,
                reward,
                discount,
                next_obs,
                task.detach(),
                step,
                basis_features,
            )
        )

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
    
    def get_phi(self, obs, action):
        obs = self.aug_and_encode(obs)
        obs_condition = self.action_conditioner(obs, action)
        return F.normalize(obs_condition, p=2, dim=-1)