"""
Code modified from https://github.com/RobvanGastel/rnn-sac/tree/main
"""

import copy
import time
import wandb
import itertools

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from sac_agent.core import ActorCritic, count_vars
from sac_agent.traj_buffer import EpisodicBuffer



class SAC:
    def __init__(self, env,
                 save_freq=1, gamma=0.99, lr=1e-4, ac_kwargs=dict(),
                 polyak=0.995, steps_per_epoch=20, epochs=1, batch_size=16,
                 replay_size=int(1e6), time_step=50, hidden_size=256,
                 start_steps=2000, update_after=1000, update_every=50,
                 exploration_sampling=False, clip_ratio=1.0,
                 number_of_trajectories=20000, use_alpha_annealing=True):

        self.env = env
        self.save_freq = save_freq

        self.max_ep_len = 20

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Meta-learning parameters
        self.epochs = epochs

        # Number of steps per trial
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs
        self.number_of_trajectories = number_of_trajectories

        self.total_traj = 0
        self.current_test_epoch = 0
        self.current_epoch = 0
        self.steps_per_epoch = self.number_of_trajectories * self.max_ep_len
        replay_size = self.steps_per_epoch

        # Meta-testing
        # Increase global steps for the next trial
        self.global_test_steps = 0
        self.global_steps = 0

        # Updating the network parameters
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak
        self.clip_ratio = clip_ratio
        self.update_counter = 0
        self.update_multiplier = 20
        self.start_steps = start_steps
        self.update_every = update_every
        self.update_after = update_after

        self.batch_size = batch_size
        self.time_step = time_step
        self.hidden_size = hidden_size

        # The online and target networks
        self.ac = ActorCritic(env.observation_space,
                              env.action_space, self.device,
                              **ac_kwargs).to(self.device)
        self.ac_targ = copy.deepcopy(self.ac)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        obs_dim = (env.observation_space['image'].shape[-1], env.observation_space['image'].shape[0], env.observation_space['image'].shape[1])
        act_dim = env.action_space.n

        # replay buffer to stay close to the idea of updating
        # on the whole trajectories for meta-learning purposes
        self.buffer = EpisodicBuffer(
            obs_dim, act_dim, replay_size, self.hidden_size, self.device,
            use_sac=True, use_exploration_sampling=exploration_sampling)

        # Optimize entropy exploration-exploitation parameter
        self.use_alpha_annealing = use_alpha_annealing
        if use_alpha_annealing:
            self.entropy_target = 0.98 * (-np.log(1 / self.env.action_space.n))
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(),
                                        self.ac.q2.parameters())

        self.pi_params = itertools.chain(self.ac.pi.parameters(),
                                         self.ac.memory.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi_params, lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

    def compute_critic_loss(self, batch):
        obs, act, rew = batch['obs'], batch['act'], batch['rew']
        next_obs, done = batch['obs2'], batch['done']

        # RL^2 variables
        h_out = batch['hid_out'].view(1, 1, self.hidden_size)

        rew = rew.view(-1, 1)

        # Current online network q values
        q1 = self.ac.q1(obs)
        q2 = self.ac.q2(obs)

        q1 = q1.gather(1, act.view(-1,1).long())
        q2 = q2.gather(1, act.view(-1,1).long())

        # Target actions come from *current* policy
        with torch.no_grad():
            memory_emb_pi, _ = self.ac.memory(
                next_obs, act, rew, h_out, training=True)
            _, a2, logp_a2 = self.ac.pi.sample(memory_emb_pi)

            # Target Q-values
            q1_targ = self.ac_targ.q1(next_obs)
            q2_targ = self.ac_targ.q2(next_obs)
            q_targ = torch.min(q1_targ, q2_targ)

            # To map R^|A| -> R
            next_q = (a2 * (q_targ - self.alpha * logp_a2)
                      ).sum(dim=1, keepdim=True)

        assert rew.shape == next_q.shape
        backup = rew + self.gamma * (1 - done) * next_q

        # MSE loss against Bellman backup
        loss_q1 = (q1 - backup).pow(2).mean()
        loss_q2 = (q2 - backup).pow(2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        mean_q1 = q1.detach().mean().item()
        mean_q2 = q1.detach().mean().item()
        q_info = dict(Q1Vals=mean_q1, Q2Vals=mean_q2)

        return loss_q, q_info

    def compute_policy_loss(self, batch):
        obs = batch['obs']

        # RL^2 variables
        prev_act, prev_rew = batch['prev_act'], batch['prev_rew'].view(-1, 1)
        h_in = batch['hid'].view(1, 1, self.hidden_size)

        memory_emb, _ = self.ac.memory(
            obs, prev_act, prev_rew, h_in, training=True)
        _, pi, logp_pi = self.ac.pi.sample(memory_emb)

        with torch.no_grad():
            q1_pi = self.ac.q1(obs)
            q2_pi = self.ac.q2(obs)
            q_pi = torch.min(q1_pi, q2_pi)

        # Expectation of entropy
        entropy = -torch.sum(pi * logp_pi, dim=1, keepdim=True)

        # Expectations of Q
        q = torch.sum(q_pi * pi, dim=1, keepdim=True)

        # Entropy-regularized policy loss
        loss_pi = (- q - self.alpha * entropy).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy(),
                       entropy=entropy.cpu().detach().numpy())

        return loss_pi, logp_pi, pi_info

    def update(self):
        batch = self.buffer.get(self.batch_size)

        for episode in batch:
            # Optimize Q-networks
            # First run one gradient descent step for Q1 and Q2
            loss_q, q_info = self.compute_critic_loss(episode)

            self.q_optimizer.zero_grad()
            loss_q.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.clip_ratio)
            self.q_optimizer.step()

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Optimize the Policy
            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi, logp_pi, pi_info = self.compute_policy_loss(episode)

            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.clip_ratio)
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Optimize the alpha
            # Entropy values
            if self.use_alpha_annealing:
                alpha_loss = -(self.log_alpha * (logp_pi.detach() +
                                                 self.entropy_target)).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0)
                self.alpha = torch.tensor(0.2)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update
                # target params, as opposed to "mul" and "add", which would
                # make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return loss_q, loss_pi, alpha_loss, self.alpha

    def get_action(self, obs, prev_act, prev_rew, hid_in,
                   greedy=True):
        # obs shape: [1, obs_dim]
        obs = torch.as_tensor(
            obs, dtype=torch.float32).to(self.device).unsqueeze(0)

        # Don't unsqueeze for one-hot encoding
        # act shape: [1]
        prev_act = torch.as_tensor(
            [prev_act], dtype=torch.float32).to(self.device)

        # rew shape: [1, 1]
        prev_rew = torch.as_tensor(
            [prev_rew], dtype=torch.float32).to(self.device).unsqueeze(0)

        return self.ac.act(obs, prev_act, prev_rew, hid_in) if greedy \
            else self.ac.explore(obs, prev_act, prev_rew, hid_in)

    def test_agent(self, test_env, num_test_episodes=10):
        self.test_env = test_env
        h = torch.zeros([1, 1, self.hidden_size]).to(self.device)
        a2, r2 = 0, 0

        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                a, h = self.get_action(
                    o, a2, r2, h, greedy=True)

                o2, r, d, _ = self.test_env.step(a)

                o = o2
                r2 = r
                a2 = a

                ep_ret += r
                ep_len += 1

                self.global_test_steps += 1

    def train_agent(self, env):
        # Prepare for interaction with environment
        self.env = env
        start_time = time.time()

        # RL^2 variables
        previous_action, previous_reward = 0, 0

        # for _ in range(self.epochs):
        #     # Inbetween trials reset the hidden weights
        h_in = torch.zeros([1, 1, self.hidden_size]).to(self.device)
        h_out = torch.zeros([1, 1, self.hidden_size]).to(self.device)

        # To sample k trajectories
        for tr in range(self.number_of_trajectories):
            done, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
            state, info = self.env.reset()
            state = np.transpose(state['image']/255, (2,0,1))
            while not(done):
                # Keeping track of current hidden states
                h_in = h_out
                # Until start_steps have elapsed, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if self.global_steps > self.start_steps:
                    action, h_out = self.get_action(
                        state, previous_action, previous_reward, h_in, greedy=False)
                else:
                    action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                ep_ret += reward
                ep_len += 1
                done = terminated + truncated
                self.buffer.store(state, np.transpose(next_state['image']/255, (2,0,1)),
                    action, reward, done,
                    previous_action, previous_reward,
                    (h_in.cpu().numpy(), h_out.cpu().numpy()))
                # Super critical, easy to overlook step: make sure to
                # update most recent observation!
                state = np.transpose(next_state['image']/255, (2,0,1))
                # Set previous action and reward
                previous_reward = reward
                previous_action = action
                # End of trajectory handling
                if done:
                    self.buffer.finish_path()
                # Increase global steps for the next trial
                self.global_steps += 1
                # Update handling
                if self.global_steps > self.start_steps:
                    loss_q, loss_pi, alpha_loss, alpha = self.update()
            if self.global_steps > self.start_steps:
                wandb.log({"Reward": ep_ret,
                           "Episode steps" : ep_len,
                           "Steps": self.global_steps,
                           "Policy Loss": loss_pi,
                           "Alpha Loss": alpha_loss,
                           "Alpha": alpha,
                           "Episode": tr})
        # self.current_epoch += 1