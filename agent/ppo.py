import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_norm_
from architectures.cnn import Conv2d_MLP_Model
from architectures.common_utils import save_gif

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    """Buffer for storing trajectories for on-policy learning."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]

    def __len__(self):
        return len(self.states)

class CNNActor(nn.Module):
    """Placeholder for a CNN-based Actor (Policy) Network."""
    def __init__(self, **kwargs):
        super(CNNActor, self).__init__()
        # conv2d layer arguments
        input_channels = kwargs["input_dim"]
        # conv2d optional arguments
        channels = kwargs["channels"]
        kernel_sizes = kwargs["kernel_sizes"]
        strides = kwargs["strides"]
        paddings = kwargs["paddings"]
        cnn_activ = kwargs["cnn_activ"]
        use_maxpool = kwargs["use_maxpool"]
        # fc layer arguments
        fc_input_size = kwargs["fc_input_size"]
        feature_dim = kwargs["fc_output"]
        # fc layer optional arguments
        fc_hidden_sizes = kwargs["fc_hidden_sizes"]
        fc_hidden_activation = kwargs["fc_hidden_activation"]
        dropout_prob = kwargs["dropout_prob"]
        norm = kwargs["norm"]
        # Final net
        self.action_dim = kwargs["action_dim"]

        self.net = Conv2d_MLP_Model(# conv2d layer arguments
                                    input_channels = input_channels,
                                    # fc layer arguments
                                    fc_input_size = fc_input_size,
                                    fc_output_size = feature_dim,
                                    # conv2d optional arguments
                                    channels=channels,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    paddings=paddings,
                                    nonlinearity=cnn_activ,
                                    use_maxpool=use_maxpool,
                                    # fc layer optional arguments
                                    fc_hidden_sizes=fc_hidden_sizes,
                                    fc_hidden_activation=fc_hidden_activation,
                                    fc_output_activation="identity",
                                    dropout_prob = dropout_prob,
                                    norm = norm
                                    )
        self.q_layer = nn.Linear(feature_dim, self.action_dim)

    def forward(self, x):
        h = self.net(x)[0]
        action_logits = self.q_layer(h)
        dist = Categorical(logits=action_logits.squeeze())
        return dist

class CNNCritic(nn.Module):
    """Critic (Q-Network) Model for SAC."""

    def __init__(self, **kwargs):
        """Initialize parameters and build model."""
        super(CNNCritic, self).__init__()
        # conv2d layer arguments
        input_channels = kwargs["input_dim"]
        # conv2d optional arguments
        channels = kwargs["channels"]
        kernel_sizes = kwargs["kernel_sizes"]
        strides = kwargs["strides"]
        paddings = kwargs["paddings"]
        cnn_activ = kwargs["cnn_activ"]
        use_maxpool = kwargs["use_maxpool"]
        # fc layer arguments
        fc_input_size = kwargs["fc_input_size"]
        feature_dim = kwargs["fc_output"]
        # fc layer optional arguments
        fc_hidden_sizes = kwargs["fc_hidden_sizes"]
        fc_hidden_activation = kwargs["fc_hidden_activation"]
        dropout_prob = kwargs["dropout_prob"]
        norm = kwargs["norm"]
        
        self.net = Conv2d_MLP_Model(# conv2d layer arguments
                                    input_channels = input_channels,
                                    # fc layer arguments
                                    fc_input_size = fc_input_size,
                                    fc_output_size = feature_dim,
                                    # conv2d optional arguments
                                    channels=channels,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    paddings=paddings,
                                    nonlinearity=cnn_activ,
                                    use_maxpool=use_maxpool,
                                    # fc layer optional arguments
                                    fc_hidden_sizes=fc_hidden_sizes,
                                    fc_hidden_activation=fc_hidden_activation,
                                    fc_output_activation="identity",
                                    dropout_prob = dropout_prob,
                                    norm = norm
                                    )
        self.q_layer = nn.Linear(feature_dim, 1)

    def forward(self, state):
        h = self.net(state)[0]
        q = self.q_layer(h)
        return q

# --- Main PPO Agent Class ---

class PPO(nn.Module):
    """Interacts with and learns from the environment using PPO."""

    def __init__(self, **kwargs):
        """Initialize an Agent object."""
        super(PPO, self).__init__()

        self.action_size = kwargs["action_dim"]
        
        # PPO uses an actor-critic architecture
        self.actor = CNNActor(**kwargs).to(device)
        self.critic = CNNCritic(**kwargs).to(device)

    def initialise_buffer(self, config):
        """Initialise the on-policy rollout buffer."""
        self.buffer = RolloutBuffer()
        self.update_timestep = config.update_timestep
        self.batch_size = config.batch_size

    def set_training_params(self, config, train_transform):
        """Set PPO-specific training hyperparameters."""
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.epochs = config.epochs
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.train_transform = train_transform

    def set_optimizer(self, cfg):
        """Set optimizers for actor and critic networks."""
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay
        )
        self.clip_grad_param = cfg.clip_grad_param

    def get_action(self, state, steps=0, deterministic=False):
        """
        Selects an action from the policy, and returns action, log_prob, and value.
        """
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()
            
            dist = self.actor(state)
            self.value = self.critic(state).squeeze()
            
            if deterministic:
                action = dist.probs.argmax(dim=-1, keepdim=True)
            else:
                action = dist.sample()
            
            self.action_logprob = dist.log_prob(action)
            
            self.actor.train()
            self.critic.train()
        self.state = state
        return action.item()

    def learn(self, timestep):
        """
        PPO update rule. This is called after a rollout is collected.
        """
        if len(self.buffer) < self.update_timestep:
            return {}

        # --- 1. Compute advantages and returns ---
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32).to(device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32).to(device)
        
        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = 0 # No next state if terminal
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]

        # --- 2. Convert old data to tensors ---
        old_states = torch.stack(self.buffer.states)
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.int64).to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 3. Optimize policy and value network for K epochs ---
        for _ in range(self.epochs):
            # Create minibatches
            indices = np.arange(len(rewards))
            np.random.shuffle(indices)
            for start in range(0, len(rewards), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]

                # Get new logprobs, values, and entropy
                dist = self.actor(old_states[mb_indices])
                new_values = self.critic(old_states[mb_indices]).squeeze()
                new_logprobs = dist.log_prob(old_actions[mb_indices])
                entropy = dist.entropy()

                # --- Critic Loss (Value Function Loss) ---
                critic_loss = F.mse_loss(new_values, returns[mb_indices])

                # --- Actor Loss (Policy Loss) ---
                logratio = new_logprobs - old_logprobs[mb_indices]
                ratio = torch.exp(logratio)
                
                mb_advantages = advantages[mb_indices]
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Total Loss ---
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()

                # --- Update ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.actor.parameters(), self.clip_grad_param)
                clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # --- 4. Clear buffer for next rollout ---
        self.buffer.clear()

        metric = {
            "Actor Loss": actor_loss.item(),
            "Critic Loss": critic_loss.item(),
        }
        return metric
    

    def soft_update(self, local_model, target_model):
        """Not used in this PPO implementation."""
        pass

    def do_post_episode_processing(self, steps_done):
        """Not used in this PPO implementation."""
        pass

    def add_transition_to_buffer(self, transition):
        """Add a single transition to the rollout buffer."""
        state, action, reward, next_state, terminated, truncated = transition
        self.buffer.states.append(self.state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(self.action_logprob.cpu())
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(terminated or truncated)
        self.buffer.values.append(self.value.cpu())

    def do_post_task_processing(self):
        """Clear buffer after a task is complete."""
        self.buffer.clear()

    def test(self, env, fps, dump_dir, test_episodes=10):
        """Test the agent in the environment with a deterministic policy."""
        for episode in range(test_episodes):
            frame_array_full = []
            state, info = env.reset()
            frame_array_full.append(env.unwrapped.get_frame())
            cumulative_reward = 0
            done = False
            
            while not done:
                # Use deterministic=True for evaluation
                action = self.get_action(self.train_transform(state), deterministic=True)
                next_state, reward, truncated, terminated, _ = env.step(action)
                frame_array_full.append(env.unwrapped.get_frame())
                done = truncated or terminated
                cumulative_reward += reward
                state = next_state
                
            save_gif(frame_array_full, episode, dump_dir, fps=fps, save_name="full")

    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path + "PPO.tar", map_location=device, weights_only=True)
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])
        print(f"[INFO] Loaded the PPO model from {path}")

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        
        params = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        
        checkpoint_path = os.path.join(dump_dir, f"{save_name}.tar")
        torch.save(params, checkpoint_path)
        print(f"[INFO] PPO model saved to: {checkpoint_path}")