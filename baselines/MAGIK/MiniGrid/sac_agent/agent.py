import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sac_agent.buffer import ReplayBuffer
from sac_agent.networks import Critic, Actor
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                input_dim,
                action_size,
                fc_hidden_size,
                device,
                buffer_size,
                batch_size = 64
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SAC, self).__init__()
        self.input_dim = input_dim
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        fc_hidden_size = fc_hidden_size
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.7], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # Actor Network 
        self.actor_local = Actor(input_dim, action_size, fc_hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(input_dim, action_size, fc_hidden_size).to(device)
        self.critic2 = Critic(input_dim, action_size, fc_hidden_size).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(input_dim, action_size, fc_hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(input_dim, action_size, fc_hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
        
        #Buffer for storing the experience
        self.buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
        
        #Action_space
        self.action_space = torch.tensor(range(self.action_size)).to(device)

        print("[INFO] Device : ", device)
        
    def get_action(self, state):
        """Returns actions for given state as per current policy."""
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        #Action is similar to tensor([x])
        if len(action)>1:
            return action
        else:
            return action.item()

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self, experiences, gamma):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, _ = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp().cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        action_q_values1 = q1.gather(1, actions.long())
        action_q_values2 = q2.gather(1, actions.long())
        
        
        critic1_loss = 0.5 * F.mse_loss(action_q_values1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(action_q_values2, Q_targets)
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
   
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path, map_location=device, weights_only=True)
        self.actor_local.load_state_dict(params["actor"])
        self.critic1.load_state_dict(params["critic1"])
        self.critic2.load_state_dict(params["critic2"])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.alpha_optimizer.load_state_dict(params["alpha_optim"])
        self.actor_optimizer.load_state_dict(params["actor_optim"])
        self.critic1_optimizer.load_state_dict(params["critic1_optim"])
        self.critic2_optimizer.load_state_dict(params["critic2_optim"])
        self.log_alpha = params["log_alpha"]
        print("[INFO] loaded the SAC model", path)

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        params = {
                "actor": self.actor_local.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "alpha_optim" : self.alpha_optimizer.state_dict(),
                "actor_optim" : self.actor_optimizer.state_dict(),
                "critic1_optim" : self.critic1_optimizer.state_dict(),
                "critic2_optim" : self.critic2_optimizer.state_dict(),
                "log_alpha" : self.log_alpha
                }
        save_dir = dump_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # if self.influenced:
        #     if self.args.is_influence_reward:
        #         suffix = '{}_Influenced_with_influence_reward_with_weight_{}'.format(self.args.demon_prob, self.args.influence_reward_weight)
        #     else:
        #         suffix = '{}_Influenced_without_influence_reward'.format(self.args.demon_prob)
        # else:
        #     suffix = "No_influence"
        # if self.args.is_predicting_state:
        #     suffix += '_Prediciting_state'
        checkpoint_path = save_dir + '/' + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] Model saved to: ", checkpoint_path)