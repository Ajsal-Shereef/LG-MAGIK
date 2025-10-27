import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from architectures.cnn import CNNLayer, Conv2d_MLP_Model
from architectures.mlp import Linear, MLP
from architectures.stochastic import SquashedNormal

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class MLPActor(nn.Module):
    """
    MLP-based Actor (Policy) Model for SAC.
    This network maps a state vector to the parameters of a squashed Gaussian
    distribution for continuous actions.
    """
    def __init__(self, config):
        """
        Initialize parameters and build the model.

        """
        super(MLPActor, self).__init__()

        self.input_dim = config["input_dim"]
        self.action_dim = config["action_dim"]
        self.log_std_bounds = config.get("log_std_bounds", [-5, 2])

        # Create the MLP layers for the shared network body
        self.shared_mlp = MLP(self.input_dim, config["shared_fc_out_dim"], config["shared_fc_hidden_sizes"], config["shared_fc_hidden_activation"])
        
        # Layer for the mean of the action distribution
        self.mu_layer = Linear(config["shared_fc_out_dim"], self.action_dim)
        
        # Layer for the log standard deviation of the action distribution
        self.log_std_layer = Linear(config["shared_fc_out_dim"], self.action_dim)

    def forward(self, state):
        """
        Performs the forward pass to get the distribution parameters.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The mean of the action distribution.
            torch.Tensor: The standard deviation of the action distribution.
        """
        x = state
        x = self.shared_mlp(x)
        
        # Calculate mu
        mu = self.mu_layer(x)
        
        # Calculate log_std and clamp it for stability
        log_std = self.log_std_layer(x)
        
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        
        # Convert log_std to std
        std = log_std.exp()
        
        return mu, std

    def sample(self, state):
        # Get mean and standard deviation from the policy network
        mean, std = self.forward(state)
        
        # Create the distribution
        dist = SquashedNormal(mean, std)
        
        # Sample an action using the reparameterization trick
        x_t = dist.rsample()  # Pre-squashed action u
        
        # Apply the tanh squashing function to get the final action
        y_t = torch.tanh(x_t) # Final action a
        
        # Calculate the log-probability
        # log_prob = dist.log_prob(x_t)
        log_prob = dist.log_prob(x_t)
        
        # The log_prob from a multivariate normal is per-dimension, so sum it up
        log_prob = log_prob.sum(dim=-1)
        
        return y_t, log_prob, mean # Return squashed action, corrected log_prob, and mean

class CNNActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, **kwargs):
        """Initialize parameters and build model.
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
        return action, log_prob, torch.tanh(mu)

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
    