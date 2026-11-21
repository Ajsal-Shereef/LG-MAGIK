import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.mlp import Linear
from architectures.film import FiLMLayer
from architectures.mlp import MLP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ContentEncoderMLP(nn.Module):
    """
    Abstracts the features from the data
    """
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(ContentEncoderMLP, self).__init__()
        self.model = MLP(input_dim, out_dim, hidden_dim)
        # self.residual_net = ResidualMLP(out_dim)

    def forward(self, x):
        # x = self.attention(x)
        x = self.model(x)
        # return self.residual_net(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        identity = x
        out = self.bn1(self.linear1(x))
        out = F.relu(out)
        out = self.bn2(self.linear2(out))
        out += identity
        out = F.relu(out)
        return out

class ContentEncoderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2):
        super(ContentEncoderMLP, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_dim = output_dim
        if output_dim is not None:
            self.final_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.final_layer = None

    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.res_blocks:
            x = block(x)
        if self.final_layer is not None:
            x = self.final_layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, cond_dim):
        """
         Args:
             in_dim (int): Input dimension.
             hidden_dims (list[int]): List of hidden layer dimensions.
             out_dim (int): Output dimension.
             cond_dim (int): Dimension of the conditioning vector for FiLM.
         """
        super(Decoder, self).__init__()
        self.film1 = FiLMLayer(in_dim, cond_dim)
        self.fc1 = Linear(in_dim, hidden_dims[0], dropout_prob=0.00)
        self.film2 = FiLMLayer(hidden_dims[0], cond_dim)
        self.fc2 = Linear(hidden_dims[0], hidden_dims[1], dropout_prob=0.00)
        self.film3 = FiLMLayer(hidden_dims[1], cond_dim)
        self.out = Linear(hidden_dims[1], out_dim)
        self.intermediate_activation = nn.LeakyReLU()

    def forward(self, z, c):
        film1_out = self.film1(z, c)
        fc1_out = self.fc1(film1_out)
        fc1_out = self.intermediate_activation(fc1_out)
        film2_out = self.film2(fc1_out, c)
        fc2_out = self.fc2(film2_out)
        fc3_out = self.intermediate_activation(fc2_out)
        film3_out = self.film3(fc3_out, c)
        out = self.out(film3_out)
        # out[11] = F.tanh(out[11])
        return out
        # return self.activation(out)