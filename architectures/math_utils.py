import numpy as np
import torch
from torch.distributions import Independent, Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TanhNormal:
    """
    Represents a tanh-squashed Normal distribution.
    a = tanh(raw_action)
    """
    def __init__(self, mean, std):
        self.normal = Normal(mean, std)

    def sample(self):
        z = self.normal.rsample()  # rsample for reparametrization trick
        action = torch.tanh(z)
        return action, z

    def log_prob(self, action, pre_tanh_value=None):
        if pre_tanh_value is None:
            # Invert tanh safely
            clipped = action.clamp(-1 + 1e-6, 1 - 1e-6)
            pre_tanh_value = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        log_prob = self.normal.log_prob(pre_tanh_value)
        # Correction term for tanh squashing
        log_prob -= torch.log(1 - torch.tanh(pre_tanh_value).pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def mean(self):
        return torch.tanh(self.normal.mean)

    def entropy(self):
        # Approximate entropy (not exact under tanh)
        return self.normal.entropy().sum(dim=-1)


def normal_log_density(means, stds, actions):
    dist = Independent(Normal(means, stds), 1)
    return dist.log_prob(actions)


# noinspection PyPep8Naming
def gaussian_log_pdf(params, x):
    mean, log_diag_std = params
    N, d = mean.shape
    cov = np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=1)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    return log_probs


def categorical_log_pdf(params, x, one_hot=True):
    if not one_hot:
        raise NotImplementedError()
    probs = params[0]
    return np.log(np.max(probs * x, axis=1))

