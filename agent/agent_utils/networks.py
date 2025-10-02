import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from architectures.cnn import CNNLayer, Conv2d_MLP_Model
from architectures.mlp import Linear, MLP
from torch.distributions import Normal

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CNNActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, **kwargs):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(CNNActor, self).__init__()
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
        fc_output_size = kwargs["fc_output"]
        action_dim = kwargs["action_dim"]
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
        self.mu_layer = nn.Linear(fc_output_size, action_dim)
        self.log_std_layer = nn.Linear(fc_output_size, action_dim)

    def forward(self, state):
        h = self.net(state)[0]
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), -20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        x_t = dist.rsample()          # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = dist.log_prob(x_t).sum(dim=-1) - torch.log(1 - y_t.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, mu.tanh()

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
    