import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from architectures.mlp import MLP
from architectures.stochastic import SquashedNormal
from torch.nn.utils import clip_grad_norm_
from architectures.cnn import Conv2d_MLP_Model
from agent.agent_utils.buffer import ReplayBuffer
from architectures.common_utils import save_gif, zip_strict
from agent.agent_utils.networks import CNNActor, MLPActor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNNSACCritic(nn.Module):
    """Critic (Q-Network) Model for SAC."""

    def __init__(self, **kwargs):
        """Initialize parameters and build model."""
        super(CNNSACCritic, self).__init__()
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
        action_dim = kwargs["action_dim"]
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
        self.q_layer = nn.Linear(feature_dim + action_dim, 1)

    def forward(self, state, action):
        h = self.net(state)[0]
        h = torch.cat([h, action], dim=-1)
        q = self.q_layer(h)
        return q
    
class MLPSACCritic(nn.Module):
    """Critic (Q-Network) Model for SAC."""

    def __init__(self, **kwargs):
        """Initialize parameters and build model."""
        super(MLPSACCritic, self).__init__()
        
        self.q_layer = MLP(kwargs["input_dim"] + kwargs["action_dim"], 1, kwargs["shared_fc_hidden_sizes"], kwargs["shared_fc_hidden_activation"])

    def forward(self, state, action):
        h = torch.cat([state, action], dim=-1)
        q = self.q_layer(h)
        return q

class SAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self, **kwargs):
        """Initialize an Agent object.
        """
        super(SAC, self).__init__()
        self.observation_mode = kwargs["observation_mode"]
        if self.observation_mode == 'image':
            # Actor with fc_output_actor
            actor_kwargs = kwargs.copy()
            actor_kwargs["fc_output"] = actor_kwargs.pop("fc_output_actor", 64)
            self.actor = CNNActor(**actor_kwargs).to(device)

            # Critics with fc_output_critic
            critic_kwargs = kwargs.copy()
            critic_kwargs["fc_output"] = critic_kwargs.pop("fc_output_critic", 64)
            self.critic1 = CNNSACCritic(**critic_kwargs).to(device)
            self.critic2 = CNNSACCritic(**critic_kwargs).to(device)
            self.critic_target1 = CNNSACCritic(**critic_kwargs).to(device)
            self.critic_target2 = CNNSACCritic(**critic_kwargs).to(device)
            
        else:
            self.actor = MLPActor(kwargs).to(device)
            self.critic1 = MLPSACCritic(**kwargs).to(device)
            self.critic2 = MLPSACCritic(**kwargs).to(device)
            self.critic_target1 = MLPSACCritic(**kwargs).to(device)
            self.critic_target2 = MLPSACCritic(**kwargs).to(device)
            
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        for target_model in [self.critic_target1, self.critic_target2]:
            for params in target_model.parameters():
                params.requires_grad = False
        
        self.action_dim = kwargs["action_dim"]
        self.is_training = True

    def initialise_buffer(self, config):
        #Buffer for storing the experience
        self.batch_size = config.batch_size
        self.buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device, train_transform=self.train_transform)
            
    def set_training_params(self, config, train_transform):
        self.hard_update = config.hard_update
        self.gamma = config.gamma
        if not self.hard_update:
            self.tau = config.tau
        else:
            self.target_update_frequency = config.target_update_frequency
        self.initial_random_samples = config.initial_random_samples
        self.learn_after = config.learn_after
        self.train_transform = train_transform
        
    def set_optimizer(self, cfg):
        self.actor_optimizer = optim.Adam(
                                        self.actor.parameters(),
                                        lr=cfg.actor_lr,
                                        betas=tuple(cfg.betas),
                                        weight_decay=cfg.weight_decay,
                                        eps=cfg.eps,
                                    )
        self.critic_optimizer = optim.Adam(
                                        list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                        lr=cfg.critic_lr,
                                        betas=tuple(cfg.betas),
                                        weight_decay=cfg.weight_decay,
                                        eps=cfg.eps,
                                    )
        # self.clip_grad_param = cfg.clip_grad_param
        if cfg.alpha == 'auto':
            self.learnable_alpha = True
            # Target entropy is -|A|
            self.target_entropy = -float(self.action_dim)
            # We optimize log_alpha instead of alpha directly for stability
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.actor_lr, betas=tuple(cfg.betas))
        else:
            self.learnable_alpha = False
            self.alpha = cfg.alpha
        
    def get_action(self, state, steps=0):
        if steps < self.initial_random_samples:
            return np.random.uniform(-1.0, 1.0, self.action_dim)
        
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        else:
            # Add batch dimension if not present
            if state.dim() == 3 and self.observation_mode == 'image':
                 state = state.unsqueeze(0).to(device)
            elif state.dim() == 1:
                 state = state.unsqueeze(0).to(device)
            else:
                 state = state.to(device)
            
        original_mode = self.actor.training
        try:
            self.actor.eval()
            with torch.no_grad():
                mu, std = self.actor(state)
        finally:
            self.actor.train(original_mode) # Restore original mode
        
        if not self.is_training:
            # For testing, we take the deterministic action (mode of the distribution)
            action = torch.tanh(mu)
        else:
            # For training, we sample from the distribution.
            dist = SquashedNormal(mu, std)
            action = dist.sample() # .sample() ALREADY returns the squashed action
            
        return action.squeeze(0).cpu().numpy()
    
    def learn(self, timstep):
        """
        SAC update rule
        """
        
        if len(self.buffer) < self.batch_size or timstep < self.learn_after:
            return {}
        
        states, actions, rewards, next_states, truncated, terminated = self.buffer.sample()
        done = truncated + terminated
        
        # ---------------------------- update critics ---------------------------- #
        # Get next-state actions and Q targets
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            Q_target_next1 = self.critic_target1(next_states, next_actions)
            Q_target_next2 = self.critic_target2(next_states, next_actions)
            min_Q_target_next = torch.min(Q_target_next1, Q_target_next2)
            Q_targets = rewards + (self.gamma * (1 - done) * (min_Q_target_next - self.alpha * next_log_pi))

        # Get current Q estimates
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        
        # Compute critic loss (using MSE is more standard for SAC, but smooth_l1 is also fine)
        critic_loss = (F.mse_loss(current_Q1, Q_targets) + F.mse_loss(current_Q2, Q_targets))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), self.clip_grad_param)
        self.critic_optimizer.step()
        
        # -------------------------- update actor -------------------------- #
        new_actions, log_pi, _ = self.actor.sample(states)
        min_Q_pi = torch.min(
            self.critic1(states, new_actions),
            self.critic2(states, new_actions)
        )
        
        # --- Detach alpha if it is learnable ---
        alpha = self.alpha.detach() if self.learnable_alpha else self.alpha
        actor_loss = (alpha * log_pi - min_Q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # clip_grad_norm_(self.actor.parameters(), self.clip_grad_param)
        self.actor_optimizer.step()

        # --- Add update step for learnable alpha ---
        if self.learnable_alpha:
            # We want to minimize: -log_alpha * (log_pi + target_entropy)
            # This moves log_alpha in the direction that makes log_pi closer to target_entropy
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha value for the next iteration
            self.alpha = self.log_alpha.exp()

        # ----------------------- update target networks ----------------------- #
        if not self.hard_update:
            self.soft_update(self.critic1, self.critic_target1)
            self.soft_update(self.critic2, self.critic_target2)
        else:
            if timstep % self.target_update_frequency == 0:
                self.critic_target1.load_state_dict(self.critic1.state_dict())
                self.critic_target2.load_state_dict(self.critic2.state_dict())
        
        metric = {"Critic loss": critic_loss.item(), 
                  "Actor loss": actor_loss.item()}

        # --- Add alpha metrics for logging ---
        if self.learnable_alpha:
            metric["Alpha loss"] = alpha_loss.item()
            metric["Alpha"] = self.alpha.item()
        
        return metric
    
    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        with torch.no_grad():
            for param, target_param in zip_strict(local_model.parameters(), target_model.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
            
    def do_post_episode_processing(self, steps_done):
        # No epsilon decay for SAC
        pass
                       
    def add_transition_to_buffer(self, transition):
        self.buffer.add(transition)
                 
    def do_post_task_processing(self):
        self.buffer.clear()
        
    def test(self, env, fps, dump_dir, test_episodes=10):
        """Test the agent in the environment."""
        is_training = self.is_training
        self.is_training = False
        for episode in range(test_episodes):
            frame_array_partial = []
            frame_array_full = []
            state, info = env.reset()
            frame_array_partial.append(state)
            frame_array_full.append(env.unwrapped.get_frame())
            cumulative_reward = 0
            done = False
            while not done:
                action = self.get_action(self.train_transform(state), self.initial_random_samples+1)
                next_state, reward, truncated, terminated, _ = env.step(action)
                frame_array_partial.append(next_state)
                frame_array_full.append(env.unwrapped.get_frame())
                done = truncated + terminated
                cumulative_reward += reward
                state = next_state
            # write_video(frame_array, episode, dump_dir, frameSize=(env.unwrapped.get_frame().shape[1], env.unwrapped.get_frame().shape[0]))
            save_gif(frame_array_partial, episode, dump_dir, fps=fps, save_name= " partial")
            save_gif(frame_array_full, episode, dump_dir, fps=fps, save_name= " full")
        self.is_training = is_training
        
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path + "SAC.tar", map_location=device, weights_only=True)
        self.critic1.load_state_dict(params["critic1"])
        self.critic2.load_state_dict(params["critic2"])
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.actor.load_state_dict(params["actor"])
        self.critic_optimizer.load_state_dict(params["critic_optim"])
        self.actor_optimizer.load_state_dict(params["actor_optim"])
        # --- Load alpha parameters if they exist ---
        if self.learnable_alpha and "alpha_optim" in params:
            self.log_alpha.data.copy_(params["log_alpha"])
            self.alpha_optimizer.load_state_dict(params["alpha_optim"])
            self.alpha = self.log_alpha.exp()
        print("[INFO] loaded the DQN model", path)

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        params = {
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_optim" : self.critic_optimizer.state_dict(),
                "actor_optim" : self.actor_optimizer.state_dict(),
                }
        if self.learnable_alpha:
            params["log_alpha"] = self.log_alpha
            params["alpha_optim"] = self.alpha_optimizer.state_dict()
        save_dir = dump_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        checkpoint_path = save_dir + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] SAC model saved to: ", checkpoint_path)