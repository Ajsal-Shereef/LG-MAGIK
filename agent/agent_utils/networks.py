import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from architectures.cnn import CNNLayer, Conv2d_MLP_Model
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
        self.net = MLP(input_dim, action_size, fc_hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        action_probs = self.softmax(self.net(state))
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
    
    def reset_parameters(self):
        self.net.weight.data.uniform_(-3e-3, 3e-3)
    
    def get_det_action(self, state):
        self.eval()
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        self.train()
        return action.detach().cpu()


class CNNCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, **kwargs):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(CNNCritic, self).__init__()
        
        # conv2d layer arguments
        input_channels = kwargs["input_dim"]
        # conv2d optional arguments
        channels=kwargs["channels"]
        kernel_sizes=kwargs["kernel_sizes"]
        strides=kwargs["strides"]
        paddings=kwargs["paddings"]
        cnn_activ=kwargs["cnn_activ"]
        use_maxpool=kwargs["use_maxpool"]
        # fc layer arguments
        fc_input_size = kwargs["fc_input_size"]
        fc_output_size = kwargs["action_dim"]
        # fc layer optional arguments
        fc_hidden_sizes=kwargs["fc_hidden_sizes"]
        fc_hidden_activation=kwargs["fc_hidden_activation"]
        dropout_prob = kwargs["dropout_prob"]
        norm = kwargs["norm"]
        
        
        self.net = Conv2d_MLP_Model(# conv2d layer arguments
                                        input_channels = input_channels,
                                        # fc layer arguments
                                        fc_input_size = fc_input_size,
                                        fc_output_size = fc_output_size,
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

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        return self.net(state)
    