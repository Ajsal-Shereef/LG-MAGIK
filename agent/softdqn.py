import os
import copy
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from torch.nn.utils import clip_grad_norm_
from agent.agent_utils.networks import CNNCritic
from agent.agent_utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from architectures.common_utils import save_gif, zip_strict, get_train_transform_cnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SoftDQN(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self, **kwargs):
        """Initialize an Agent object.
        """
        super(SoftDQN, self).__init__()
        
        self.critic = CNNCritic(**kwargs).to(device)
        self.critic_target = CNNCritic(**kwargs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for params in self.critic_target.parameters():
            params.requires_grad = False
        
        self.action_size = kwargs["action_dim"]

    def initialise_buffer(self, config):
        #Buffer for storing the experience
        self.use_per = config.use_per
        self.batch_size = config.batch_size
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device, train_transform=get_train_transform_cnn)
        else:
            self.buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device, train_transform=get_train_transform_cnn)
            
    def set_training_params(self, config):
        self.hard_update = config.hard_update
        self.gamma = config.gamma
        self.alpha = config.alpha
        if not self.hard_update:
            self.tau = config.tau
        else:
            self.target_update_freequency = config.target_update_frequency
        self.initial_random_samples = config.initial_random_samples
        self.learn_after = config.learn_after
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.epsilon = self.epsilon_start
        
    def set_optimizer(self, cfg):
        self.optimizer = optim.AdamW(
                                        self.critic.parameters(),
                                        lr=cfg.lr,
                                        betas=tuple(cfg.betas),
                                        weight_decay=cfg.weight_decay,
                                        eps=cfg.eps,
                                    )
        self.clip_grad_param = cfg.clip_grad_param
        
    def get_action(self, state, steps=0):
        if random.random() < self.epsilon or steps < self.initial_random_samples:
            return random.randrange(self.action_size)
        else:
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                self.critic.eval()
                q = self.critic(state)[0]
                self.critic.train()
            if self.epsilon < 1e-6:
                return q.argmax().item()
            else:
                probs = F.softmax(q / self.alpha, dim=0)
                return torch.multinomial(probs, 1).item()
    
    def learn(self, timestep):
        """
        Soft DQN update rule
        """
        
        if len(self.buffer) < self.batch_size or timestep < self.learn_after:
            return {}
        
        if self.use_per:
            states, actions, rewards, next_states, truncated, terminated, idxs, is_weights = self.buffer.sample()
        else:
            states, actions, rewards, next_states, truncated, terminated = self.buffer.sample()
        done = truncated + terminated
        #Compute losses--------------------------------------------------
        # ---------------------------- Critic ---------------------------- #
        # Get predicted next-state Q values from target model
        with torch.no_grad():
            Q_target_next = self.critic_target(next_states)[0]
            
            # Compute V targets for current states (y_i)
            V_target_next = self.alpha * torch.logsumexp(Q_target_next / self.alpha, dim=1, keepdim=True)
            Q_targets = rewards + (self.gamma * (1 - done) * V_target_next) 

        # Compute critic loss
        q = self.critic(states)[0]
        action_q_values = q.gather(1, actions.long())
        if self.use_per:
            critic_loss = (is_weights.unsqueeze(1) * F.smooth_l1_loss(action_q_values, Q_targets, reduction="none")).mean()
        else:
            critic_loss = F.smooth_l1_loss(action_q_values, Q_targets, reduction="none").mean()

        self.optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
        self.optimizer.step()
        
        if self.use_per:
            td_errors = (Q_targets - action_q_values).detach()
            #Update the priorities
            self.buffer.update_priorities(idxs, td_errors.squeeze())
        
        # ----------------------- update target networks ----------------------- #
        if not self.hard_update:
            self.soft_update(self.critic, self.critic_target)
        else:
            if timestep % self.target_update_freequency == 0:
                self.critic_target.load_state_dict(self.critic.state_dict())
        
        metric = {"Critic loss": critic_loss.item(), 
                  "epsilon" : self.epsilon}
        
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
        # Update epsilon at the end of each episode
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * steps_done / self.epsilon_decay)
                       
    def add_transition_to_buffer(self, transition):
        self.buffer.add(transition)
                 
    def do_post_task_processing(self):
        self.buffer.clear()
        
    def test(self, env, fps, dump_dir, test_episodes=10):
        """Test the agent in the environment."""
        epsilon = self.epsilon
        self.epsilon = 0
        train_transform = get_train_transform_cnn()
        for episode in range(test_episodes):
            frame_array_partial = []
            frame_array_full = []
            state, info = env.reset()
            frame_array_partial.append(state)
            frame_array_full.append(env.unwrapped.get_frame())
            cumulative_reward = 0
            done = False
            while not done:
                action = self.get_action(train_transform(state), self.initial_random_samples+1)
                next_state, reward, truncated, terminated, _ = env.step(action)
                frame_array_partial.append(next_state)
                frame_array_full.append(env.unwrapped.get_frame())
                done = truncated + terminated
                cumulative_reward += reward
                state = next_state
            # write_video(frame_array, episode, dump_dir, frameSize=(env.unwrapped.get_frame().shape[1], env.unwrapped.get_frame().shape[0]))
            save_gif(frame_array_partial, episode, dump_dir, fps=fps, save_name= " partial")
            save_gif(frame_array_full, episode, dump_dir, fps=fps, save_name= " full")
        self.epsilon = epsilon
        
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path + "SoftDQN.tar", map_location=device, weights_only=True)
        self.critic.load_state_dict(params["critic"])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.optimizer.load_state_dict(params["critic_optim"])
        print("[INFO] loaded the SoftDQN model", path)

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        params = {
                "critic": self.critic.state_dict(),
                "critic_optim" : self.optimizer.state_dict(),
                }
        save_dir = dump_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        checkpoint_path = save_dir + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] SoftDQN model saved to: ", checkpoint_path)