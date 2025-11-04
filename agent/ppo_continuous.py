import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_

# --- Import the Base Discrete PPO class and its critics ---
from agent.ppo_descrete import PPO
from architectures.cnn import Conv2d_MLP_Model
from architectures.mlp import MLP, Linear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiagGaussianDistribution:
    """
    Represents a tanh-squashed Normal distribution.
    a = tanh(raw_action)
    """
    def __init__(self, mean, std):
        self.normal = Normal(mean, std)

    def sample(self):
        action = self.normal.rsample()  # rsample for reparametrization trick
        return action

    def log_prob(self, action):
        log_prob = self.normal.log_prob(action)
        return log_prob.sum(dim=-1)

    def mean(self):
        return self.normal.mean

    def entropy(self):
        # Approximate entropy (not exact under tanh)
        return self.normal.entropy().sum(dim=-1)

# ============================================================
# --- Continuous CNN Actor ---
# ============================================================
class ContinuousCNNActor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_channels = kwargs["input_dim"]
        channels = kwargs["channels"]
        kernel_sizes = kwargs["kernel_sizes"]
        strides = kwargs["strides"]
        paddings = kwargs["paddings"]
        cnn_activ = kwargs["cnn_activ"]
        use_maxpool = kwargs["use_maxpool"]
        fc_input_size = kwargs["fc_input_size"]
        feature_dim = kwargs["fc_output"]
        fc_hidden_sizes = kwargs["fc_hidden_sizes"]
        fc_hidden_activation = kwargs["fc_hidden_activation"]
        dropout_prob = kwargs["dropout_prob"]
        norm = kwargs["norm"]
        self.action_dim = kwargs["action_dim"]
        self.log_std_bounds = kwargs.get("log_std_bounds", [-20, 2])

        self.net = Conv2d_MLP_Model(
            input_channels=input_channels,
            fc_input_size=fc_input_size,
            fc_output_size=feature_dim,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=cnn_activ,
            use_maxpool=use_maxpool,
            fc_hidden_sizes=fc_hidden_sizes,
            fc_hidden_activation=fc_hidden_activation,
            fc_output_activation="identity",
            dropout_prob=dropout_prob,
            norm=norm
        )
        self.mu_layer = nn.Linear(feature_dim, self.action_dim)
        self.log_std_layer = nn.Linear(feature_dim, self.action_dim)

    def forward(self, x):
        h = self.net(x)[0]
        mu = self.mu_layer(h)
        raw_log_std = self.log_std_layer(h)
        min_val, max_val = self.log_std_bounds
        log_std = min_val + 0.5 * (torch.tanh(raw_log_std) + 1.0) * (max_val - min_val)
        std = torch.exp(log_std)
        return DiagGaussianDistribution(mu, std)  # return tanh-squashed Normal


# ============================================================
# --- Continuous MLP Actor ---
# ============================================================
class ContinuousMLPActor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_dim = kwargs["input_dim"]
        self.action_dim = kwargs["action_dim"]
        feature_dim = kwargs["shared_fc_out_dim"]
        hidden_sizes = kwargs["shared_fc_hidden_sizes"]
        activation = kwargs["shared_fc_hidden_activation"]
        self.log_std_bounds = kwargs["log_std_bounds"]
        self.net = MLP(input_dim, feature_dim, hidden_sizes, activation, norm = "none")
        self.mu_layer = Linear(feature_dim, self.action_dim, norm = "none")
        self.log_std_layer = Linear(feature_dim, self.action_dim, norm = "none")

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_layer(h)
        raw_log_std = self.log_std_layer(h)
        min_val, max_val = self.log_std_bounds
        log_std = min_val + 0.5 * (torch.tanh(raw_log_std) + 1.0) * (max_val - min_val)
        std = torch.exp(log_std)
        return DiagGaussianDistribution(mu, std)


# ============================================================
# --- PPO Continuous Agent ---
# ============================================================
class PPOContinuous(PPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions_dim = kwargs["action_dim"]
        if self.observation_mode == 'image':
            self.actor = ContinuousCNNActor(**kwargs).to(device)
        else:
            self.actor = ContinuousMLPActor(**kwargs).to(device)

    def get_action(self, state, steps=0, deterministic=False):
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()

            if state.dim() == 1:
                state = state.unsqueeze(0)

            dist = self.actor(state)

            if deterministic:
                # Deterministic action: mean of the Gaussian
                z = dist.mean()
                action = z
            else:
                # Stochastic action: sample
                action = dist.sample()

            # Compute log prob 
            self.action_logprob = dist.log_prob(action)

            self.actor.train()
            self.critic.train()

        self.state = state
        self.raw_action = action.cpu().numpy().flatten()
        clipped_action = np.clip(self.raw_action, np.array([-1]*self.actions_dim), np.array([1]*self.actions_dim))
        return clipped_action

    def learn(self, timestep):
        if len(self.buffer) < self.update_timestep:
            return {}

        # --- 1. Compute returns & advantages (fixed after this) ---
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32).to(device)
        old_values = torch.stack(self.buffer.values).to(device)

        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0

        last_value = self.buffer.last_value

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_values = last_value if t == len(rewards) - 1 else old_values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - old_values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + old_values[t]

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 2. Prepare old tensors (fixed) ---
        old_states = torch.stack(self.buffer.states).to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(device)
        old_actions = torch.tensor(np.asarray(self.buffer.actions), dtype=torch.float32).to(device)

        do_early_stop = False
        approx_kl = None
        
        # --- 3. Optimization loop ---
        for epoch in range(self.epochs):
            if do_early_stop:
                break
            indices = np.arange(len(rewards))
            np.random.shuffle(indices)
            for start in range(0, len(rewards), self.batch_size):
                mb_indices = indices[start:start + self.batch_size]
                mb_states = old_states[mb_indices]
                mb_actions = old_actions[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                # Moving the model to eval mode to make it consistent with the data collected earlier
                dist = self.actor(mb_states)
                new_values = self.critic(mb_states).squeeze()
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                # PPO losses
                logratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(logratio)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(new_values, mb_returns)
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                if self.clip_grad_param:
                    clip_grad_norm_(self.actor.parameters(), self.clip_grad_param)
                    clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # --- KL early stopping check (approx KL between old and new policy) ---
                # Use the "mean" approx_kl = mean(old_logprob - new_logprob)
                with torch.no_grad():
                    # compute approximate KL for this minibatch
                    mb_approx_kl = (mb_old_logprobs - new_logprobs).mean().item()
                    approx_kl = mb_approx_kl if approx_kl is None else approx_kl  # store last seen or first seen
                    if self.target_kl is not None and mb_approx_kl > self.target_kl:
                        # set flag to break outer loops
                        do_early_stop = True
                        # optionally print/log stopping reason
                        # print(f"Early stopping at epoch {epoch}, batch start {start}. Approx KL: {mb_approx_kl:.5f} > target {self.target_kl}")
                        break

        # --- 4. Clear buffer ---
        self.buffer.clear()

        metric = {
            "Actor Loss": actor_loss.item() if 'actor_loss' in locals() else None,
            "Critic Loss": critic_loss.item() if 'critic_loss' in locals() else None,
            "Approx KL": approx_kl
        }
        return metric
    
    def add_transition_to_buffer(self, transition):
        """Add a single transition to the rollout buffer."""
        state, action, reward, next_state, terminated, truncated = transition
        self.buffer.states.append(self.state.squeeze().cpu())
        self.buffer.actions.append(self.raw_action)
        self.buffer.logprobs.append(self.action_logprob.squeeze().detach().cpu())
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(terminated or truncated)
        self.buffer.values.append(self.critic(self.train_transform(state).unsqueeze(0)).squeeze().detach().cpu())
        if terminated or truncated:
            self.buffer.last_value = self.critic(self.train_transform(next_state).unsqueeze(0)).squeeze().detach().cpu()