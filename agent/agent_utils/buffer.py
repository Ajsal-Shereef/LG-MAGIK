import os
import pickle
import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "truncated", "terminated"])
    
    def add(self, transition):
        """Add a new experience to memory."""
        state, action, reward, next_state, truncated, terminated = transition
        e = self.experience(state, action, reward, next_state, truncated, terminated)
        self.memory.append(e)

    def dump_data(self, dir):
        states = [exp.next_state for exp in self.memory]
        data_path = os.path.join(dir, 'data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(states, f)
        print(f"Collected {len(states)} transitions and saved to {data_path}")
        data_path = os.path.join(dir, 'class_prob.pkl')

    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        if isinstance(experiences[0].state, np.ndarray):
            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        else:
            states = torch.stack([e.state for e in experiences if e is not None]).float().squeeze().to(self.device)
            next_states = torch.stack([e.next_state for e in experiences if e is not None]).squeeze().float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        truncated = torch.from_numpy(np.vstack([e.truncated for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        terminated = torch.from_numpy(np.vstack([e.terminated for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, truncated, terminated)
    
    def dump_data(self, dir):
        states = [exp.next_state for exp in self.memory]
        class_probs = [exp.class_prob for exp in self.memory]
        data_path = os.path.join(dir, 'data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(states, f)
        print(f"Collected {len(states)} transitions and saved to {data_path}")
        data_path = os.path.join(dir, 'class_prob.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(class_probs, f)
        print(f"Collected {len(class_probs)} transitions and saved to {data_path}")

    def get_full_observation_reward(self):
        states = torch.from_numpy(np.stack([e.next_state for e in self.memory if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in self.memory if e is not None])).float().to(self.device)
        return (states, rewards)
    
    def clear(self):
        """Clear all experiences from memory."""
        self.memory.clear()
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class SumTree:
    """Binary tree for efficient sampling by priority."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        idx = 0
        while idx < self.capacity - 1:  # not a leaf
            left = 2 * idx + 1
            right = left + 1
            if v <= self.tree[left]:
                idx = left
            else:
                v -= self.tree[left]
                idx = right
        dataIdx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[dataIdx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.eps = 1e-5  # small value to avoid zero priority

    def add(self, transition, priority=1.0):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = priority
        self.tree.add(max_priority, transition)

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total / self.batch_size
        priorities = []
        
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        
        for i in range(self.batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get_leaf(s)
            batch.append(data)
            priorities.append(p)
            idxs.append(idx)
        
        sampling_probabilities = np.array(priorities) / self.tree.total
        is_weights = np.power(self.tree.size * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        
        # Convert to tensors
        states = torch.stack([e[0] for e in batch]).to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in batch])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in batch])).float().to(self.device)
        next_states = torch.stack([e[3] for e in batch]).to(self.device)
        truncated = torch.from_numpy(np.vstack([e[4] for e in batch]).astype(np.uint8)).float().to(self.device)
        terminated = torch.from_numpy(np.vstack([e[5] for e in batch]).astype(np.uint8)).float().to(self.device)
        
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
        
        return (states, actions, rewards, next_states, truncated, terminated, idxs, is_weights)

    def update_priorities(self, idxs, td_errors):
        td_errors = td_errors.detach().cpu().numpy()
        for idx, error in zip(idxs, td_errors):
            p = (np.abs(error) + self.eps) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.size