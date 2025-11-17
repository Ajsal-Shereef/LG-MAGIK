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


class SFLaplacianAgent(SFSimpleAgent):
    """
    Learning Successor Features with Laplacian (also known as orthogonality) regularization on the basis features.
    """
    def __init__(
        self, update_task_every_step, sf_dim, num_init_steps, lr_task, normalize_basis_features, normalize_task_params, **kwargs
    ):
        # create actor and critic
        super().__init__(
            update_task_every_step, sf_dim, num_init_steps, lr_task, normalize_basis_features, normalize_task_params, **kwargs
        )

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
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        self.critic_opt.step()

        return metrics

    def update_laplacian(self, basis_features, next_basis_features):
        metrics = dict()

        # compute laplacian loss
        laplacian_loss = (basis_features - next_basis_features).pow(2).mean()
        Cov = torch.matmul(basis_features, next_basis_features.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        total_loss = laplacian_loss + orth_loss

        if self.use_tb or self.use_wandb:
            metrics["laplacian_loss"] = laplacian_loss.item()
            metrics["orth_loss"] = orth_loss.item()
            metrics["orth_loss_diag"] = orth_loss_diag.item()
            metrics["orth_loss_offdiag"] = orth_loss_offdiag.item()
            metrics["total_laplacian_orth_loss"] = total_loss.item()

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)

        # self.feature_opt.zero_grad(set_to_none=True)

        total_loss.backward()

        # self.feature_opt.step()

        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, next_obs, dones, task = replay_iter.sample()
        discount = torch.ones_like(reward) * (1.0 - dones.float())

        task = self.task_params.unsqueeze(0).repeat(task.shape[0],1)

        # augment and encode
        obs = self.aug_and_encode(obs)

        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.normalize_basis_features:
            basis_features = F.normalize(obs, p=2, dim=-1)
            next_basis_features = F.normalize(next_obs, p=2, dim=-1)
        else:
            basis_features = obs
            next_basis_features = next_obs

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        if self.normalize_task_params:
            # normalize task
            task_normalized = F.normalize(task, p=2, dim=-1)

        else:
            task_normalized = task

        # extend observations with normalized task
        # obs = torch.cat([obs, task_normalized], dim=1)
        # next_obs = torch.cat([next_obs, task_normalized], dim=1)

        # update meta
        if step % self.update_task_every_step == 0:
            metrics.update(
                self.regress_meta_grad_descent(
                    next_obs, task, reward, step, next_basis_features.detach()
                )
            )

        # update laplacian
        metrics.update(self.update_laplacian(basis_features, next_basis_features))

        # update critic
        metrics.update(
            self.update_critic(
                obs.detach(), action, reward, discount, next_obs.detach(), task.detach(), step
            )
        )

        # update actor
        # metrics.update(self.update_actor(obs.detach(), task.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
