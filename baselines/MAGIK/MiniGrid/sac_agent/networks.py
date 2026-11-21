import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from architectures.cnn import CNNLayer, CNN
from architectures.mlp import Linear, MLP

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_dim, action_size, fc_hidden_size=576):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        conv1 = CNNLayer(input_dim, 32, 8, 2)
        conv2 = CNNLayer(32, 64, 8, 2)
        conv3 = CNNLayer(64, 64, 3)
        conv_feature = Linear(fc_hidden_size, action_size)
        self.net = CNN([conv1, conv2, conv3], conv_feature)
        # self.fc1 = nn.Linear(32, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, action_size)
        # self.net = MLP(input_dim, action_size, [128 ,64])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        # feature = self.feature_encoder(state)
        # x = F.relu(self.fc1(feature[0]))
        # x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.net(state)[0])
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        self.eval()
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        self.train()
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_dim, action_size, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        conv1 = CNNLayer(input_dim, 32, 8, 2)
        conv2 = CNNLayer(32, 64, 8, 2)
        conv3 = CNNLayer(64, 64, 3)
        conv_feature = Linear(hidden_size, action_size)
        self.net = CNN([conv1, conv2, conv3], conv_feature)
        # self.fc1 = nn.Linear(32, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, action_size)
        # self.net = MLP(input_dim, action_size, [128 ,64])
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # self.net.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # feature = self.feature_encoder(state)
        # x = F.relu(self.fc1(feature[0]))
        # x = F.relu(self.fc2(x))
        return self.net(state)[0]