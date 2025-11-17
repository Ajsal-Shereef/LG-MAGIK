import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from PIL import Image
from collections import OrderedDict

import utils

from baselines.successor_feature.agent.dqn import DQNAgent

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, sf_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        
        # Get individual dimensions
        C, H, W = obs_shape[0], obs_shape[1], obs_shape[2]
        
        self.convnet = nn.Sequential(
            nn.Conv2d(C, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            # Create a dummy input tensor in the format PyTorch
            # Conv2d expects: (1, C, H, W)
            dummy_input = torch.zeros(1, C, H, W)
            
            # Pass it through the convolutional network
            dummy_output = self.convnet(dummy_input)
            
            # Calculate the flattened output size (C' * H' * W')
            self.repr_dim = dummy_output.flatten(1).shape[1]

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

# class FeatureNet(nn.Module):
#     def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, sf_dim):
#         super().__init__()

#         self.obs_type = obs_type

#         # a small difference compared to aps is that aps uses an additional state_feat_net
#         # whereas we directly use the trunk to get the features. Therefore, to get the
#         # basis features dim to match the sf_dim, we include an additional linear layer.

#         if obs_type == "pixels":
#             # for pixels actions will be added after trunk
#             self.trunk = nn.Sequential(
#                 nn.Linear(obs_dim, feature_dim),
#                 nn.LayerNorm(feature_dim),
#                 nn.Tanh(),
#                 nn.Linear(feature_dim, sf_dim),
#             )
#         else:
#             # for states actions come in the beginning
#             self.trunk = nn.Sequential(
#                 nn.Linear(obs_dim + action_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim, sf_dim),
#             )

#         self.apply(utils.weight_init)

#     def forward(self, obs, action):
#         inpt = obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
#         h = self.trunk(inpt)
#         basis_features = F.normalize(h, p=2, dim=-1)
#         return h, basis_features

# class Actionconditioner(nn.Module):
#     def __init__(self, feature_dim, conditioning_dim):
#         """
#         FiLM layer applies feature-wise affine transformations using conditioning input.

#         Args:
#             feature_dim (int): Number of features to apply FiLM over (e.g., channels in CNN).
#             conditioning_dim (int): Dimensionality of the input used to generate FiLM parameters.
#         """
#         super(Actionconditioner, self).__init__()
#         self.film_generator = nn.Linear(conditioning_dim, 2 * feature_dim)

#     def forward(self, x, conditioning_input):
#         """
#         Args:
#             x (Tensor): Input features of shape [B, C, H, W] or [B, C] (for MLPs).
#             conditioning_input (Tensor): Conditioning input of shape [B, conditioning_dim].

#         Returns:
#             Tensor: FiLM-modulated features.
#         """
#         if conditioning_input.shape[-1] != self.film_generator.in_features:
#             conditioning_input = torch.eye(self.film_generator.in_features)[conditioning_input.squeeze().cpu().to(torch.int64)].to(conditioning_input.device)
#         gamma_beta = self.film_generator(conditioning_input)
#         gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        
#         # Reshape for broadcasting
#         if x.dim() == 4:  # [B, C, H, W]
#             gamma = gamma.unsqueeze(-1).unsqueeze(-1)
#             beta = beta.unsqueeze(-1).unsqueeze(-1)
#         return gamma * x + beta
    
class CriticSF(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, hidden_dim, sf_dim):
        super().__init__()

        self.obs_type = obs_type
        self.action_dim = action_dim

        def make_q():
            q_layers = []
            q_layers += [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(inplace=True)]
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
            # h = conditioner(h, actions)  # [B*A, hidden_dim + A]
            h = torch.cat([h, actions], dim=-1)
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
        q1 = self.Q1(h)
        q2 = self.Q2(h)

        task = task.unsqueeze(1).repeat(1, self.action_dim, 1).to(device)
        task = task.view(-1, task.shape[-1])
        q1 = torch.einsum("bi,bi->b", task, q1).view(batch_size, self.action_dim)
        q2 = torch.einsum("bi,bi->b", task, q2).view(batch_size, self.action_dim)

        return q1, q2

class SFSimpleAgent(DQNAgent):
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
        # self.update_encoder = update_encoder
        self.lr_task = lr_task
        self.normalize_basis_features = normalize_basis_features
        self.normalize_task_params = normalize_task_params

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        # create actor and critic
        super().__init__(**kwargs)

        if self.obs_type == "pixels":
            self.aug = nn.Identity()
            # self.action_conditioner = Actionconditioner(self.sf_dim, self.action_dim).to(self.device)
            self.encoder = Encoder(self.obs_shape, self.feature_dim, self.sf_dim).to(
                self.device
            )
            # self.obs_dim = self.sf_dim + self.meta_dim
            self.obs_dim = self.sf_dim
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
            # self.action_conditioner_opt = torch.optim.Adam(self.action_conditioner.parameters(), lr=self.lr)
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = self.obs_shape[0] + self.meta_dim
            self.encoder_opt = None

        # self.featureNet = FeatureNet(
        #     self.obs_type,
        #     self.obs_dim,
        #     self.action_dim,
        #     self.feature_dim,
        #     self.hidden_dim,
        #     self.sf_dim,
        # ).to(self.device)

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
        # self.feature_opt = torch.optim.Adam(self.featureNet.parameters(), lr=self.lr)

        self.task_params = nn.Parameter(
            torch.randn(self.sf_dim, requires_grad=True, device=self.device)
        )

        self.task_opt = torch.optim.Adam([self.task_params], lr=self.lr_task)

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

        task = self.task_params.unsqueeze(0).repeat(task.shape[0],1)
        if self.normalize_task_params:
            # normalize task
            task_normalized = F.normalize(task, p=2, dim=-1)
        else:
            task_normalized = task

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            # obs_condition = self.action_conditioner(obs, action)
            if self.normalize_basis_features:
                next_basis_features = F.normalize(next_obs, p=2, dim=-1)
            else:
                next_basis_features = next_obs

        if self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with normalized task
        # obs = torch.cat([obs, task_normalized], dim=1)
        # next_obs = torch.cat([next_obs, task_normalized], dim=1)

        # update meta
        if step % self.update_task_every_step == 0:
            metrics.update(
                self.regress_meta_grad_descent(
                    next_obs, task, reward, step, next_basis_features
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

    def get_phi(self, obs, action):
        obs = self.aug_and_encode(obs)
        # obs_condition = self.action_conditioner(obs, action)
        return F.normalize(obs, p=2, dim=-1)

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def regress_meta_grad_descent(
        self, next_obs, task, reward, step, next_basis_features=None
    ):
        metrics = dict()

        if self.obs_type == "pixels":
            predicted_reward = torch.einsum(
                "bi,i->b", next_basis_features, self.task_params
            ).reshape(-1, 1)

        else:
            with torch.no_grad():
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                next_rep = self.featureNet(next_obs, next_action)[-1]
            predicted_reward = torch.einsum(
                "bi,i->b", next_rep, self.task_params
            ).reshape(-1, 1)

        reward_prediction_loss = F.mse_loss(predicted_reward, reward)

        self.task_opt.zero_grad(set_to_none=True)
        reward_prediction_loss.backward()
        self.task_opt.step()

        # set solved_meta to the value of the task_params
        with torch.no_grad():
            meta = self.task_params.detach().cpu().numpy()
            # normalize solved meta using l2 norm
            # meta = meta / np.linalg.norm(meta)
            self.solved_meta = OrderedDict()
            self.solved_meta["task"] = meta

        if self.use_wandb:
            metrics["reward_prediction_loss"] = reward_prediction_loss.item()
            metrics["task_grad_norm"] = self.task_params.grad.norm().item()

        return metrics
    
    def update_critic(self, obs, action, reward, discount, next_obs, task, step):
        """diff is critic takes task as input"""
        metrics = dict()

        with torch.no_grad():
            # Step 1: Get Q-values for all actions from current (online) critic for next_obs
            # next_h = self.featureNet(next_obs, [])[0]
            next_Q1, next_Q2 = self.critic(next_obs, task, self.action_conditioner)  # shape: [B, A]
            # Step 2: Double Q-learning action selection
            next_Q = torch.min(next_Q1, next_Q2)  # shape: [B, A]
            next_action = torch.argmax(next_Q, dim=1, keepdim=True)  # shape: [B, 1]

            target_Q1, target_Q2 = self.critic_target(next_obs, task, self.action_conditioner)
            target_Q1 = target_Q1.gather(1, next_action).squeeze(1)  # shape: [B]
            target_Q2 = target_Q2.gather(1, next_action).squeeze(1)  # shape: [B]
            target_V = torch.min(target_Q1, target_Q2)  # shape: [B]

            # Step 4: Compute target Q-value using Bellman backup
            target_Q = reward.squeeze() + discount.squeeze() * target_V  # shape: [B]

        # Step 5: Get current Q-values for the actions taken in obs
        # h = self.featureNet(obs, action)[0]
        current_Q1, current_Q2 = self.critic(obs, task, self.action_conditioner)
        current_Q1 = current_Q1.gather(1, action.to(torch.int64)).squeeze(1)  # shape: [B]
        current_Q2 = current_Q2.gather(1, action.to(torch.int64)).squeeze(1)  # shape: [B]

        # Step 6: Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = current_Q1.mean().item()
            metrics["critic_q2"] = current_Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            # self.action_conditioner_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        # self.feature_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        self.critic_opt.step()
        # self.feature_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
            # self.action_conditioner_opt.step()
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
