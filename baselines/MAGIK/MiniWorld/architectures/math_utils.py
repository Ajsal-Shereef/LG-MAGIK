import torch
from torch.distributions import Distribution, Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = (
            self.normal_mean
            + self.normal_std
            * Normal(
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size()),
            )
            .sample()
            .to(device)
        )
        if return_pretanh_value:
            return torch.tanh(z), z
        return torch.tanh(z)
