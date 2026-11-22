import torch
import numpy as np
import torch.nn as nn

try:
    from ..mlp import MLP
    from ..cnn import ResBlocks, Conv2dBlock
except ImportError:
    from architectures.mlp import MLP
    from architectures.cnn import ResBlocks, Conv2dBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FiLM(nn.Module):
    """
    FiLM layer: learns per-channel scale and shift from a condition vector.
    """

    def __init__(self, num_features, cond_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, h_dim, fc_hidden, fc_input_dim):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

        self.mlp = MLP(int(dim * np.prod(fc_input_dim)), int(h_dim), list(fc_hidden))

    def forward(self, x):
        x = self.model(x)
        return self.mlp(x.reshape(x.shape[0], -1))


class Decoder(nn.Module):
    def __init__(
        self,
        n_upsample,
        n_res,
        dim,
        output_dim,
        z_dim,
        y_dim,
        res_norm="adain",
        activ="relu",
        pad_type="zero",
        fc_input_dim=[10, 10],
    ):
        super(Decoder, self).__init__()
        self.dim = int(dim)
        self.fc_input = list(fc_input_dim)
        # fc layers
        self.mlp = MLP(int(z_dim), int(dim * np.prod(fc_input_dim)), [128, 512, 2048])
        # residual blocks
        self.res_layers = ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        # FiLM layer initialisation
        self.film1 = FiLM(dim, y_dim)
        # upsampling blocks
        self.upsample_layers = nn.Sequential()
        for i in range(n_upsample):
            self.upsample_layers.add_module("UpSampling_{}".format(i), nn.Upsample(scale_factor=2))
            self.upsample_layers.add_module(
                "Conv2dBlock_{}".format(i),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm="ln", activation=activ, pad_type=pad_type),
            )
            dim //= 2
        # use reflection padding in the last conv layer
        self.upsample_layers.add_module(
            "Conv2dBlock_{}".format(n_upsample),
            Conv2dBlock(dim, output_dim, 7, 1, 3, norm="none", activation="none", pad_type=pad_type),
        )
        # Final FiLM layer
        self.flim2 = FiLM(output_dim, y_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x, y):
        x = self.mlp(x)
        x = x.view(x.shape[0], self.dim, self.fc_input[0], self.fc_input[1])
        x = self.res_layers(x)
        x = self.film1(x, y)
        x = self.upsample_layers(x)
        x = self.flim2(x, y)
        return self.output_activation(x)
