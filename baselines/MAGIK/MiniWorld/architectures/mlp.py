import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal

try:
    from .common_utils import identity
    from .math_utils import TanhNormal
except ImportError:
    from architectures.common_utils import identity
    from architectures.math_utils import TanhNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_layer_uniform(layer, init_w=3e-3, init_b=0.1):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_b, init_b)
    return layer


def init_layer_orthogonal(layer, gain=None):
    if gain is None:
        gain = float(np.sqrt(2.0))
    elif torch.is_tensor(gain):
        gain = float(gain.item())
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class Linear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        post_activation=identity,
        dropout_prob=0,
    ):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        nn.init.orthogonal_(self.linear_layer.weight)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.post_activation = post_activation

    def forward(self, x):
        x = self.linear_layer(x)
        if x.size(0) > 1 or not self.training:
            x = self.batch_norm(x)
        x = self.dropout_layer(x)
        x = self.post_activation(x)
        return x


class MLP(nn.Module):
    """
    Multilayer perceptron matching original MAGIK implementation.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=F.relu,
        output_activation=identity,
        linear_layer=nn.Linear,
        use_output_layer=True,
        n_category=-1,
        init_fn=init_layer_orthogonal,
        dropout_prob=0,
        stabilise_factor=0,
    ):
        super(MLP, self).__init__()
        self.hidden_sizes = list(hidden_sizes)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category
        self.stabilise_factor = stabilise_factor

        in_size = self.input_size
        self.hidden_layers = nn.Sequential()
        for i, next_size in enumerate(self.hidden_sizes):
            fc = Linear(
                in_size,
                int(next_size),
                post_activation=self.hidden_activation,
                dropout_prob=dropout_prob,
            )
            in_size = int(next_size)
            self.hidden_layers.add_module("hidden_fc_{}".format(i), fc)

        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, self.output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x) + self.stabilise_factor
        return x
