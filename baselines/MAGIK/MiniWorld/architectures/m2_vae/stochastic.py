import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size(), device=mu.device, dtype=mu.dtype), requires_grad=False)
        std = log_var.mul(0.5).exp_()
        if self.training:
            return mu.addcmul(std, epsilon)
        else:
            return mu


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))
        return self.reparametrize(mu, log_var), mu, log_var


class GaussianMerge(GaussianSample):
    """
    Precision weighted merging of two Gaussian
    distributions.
    """

    def __init__(self, in_features, out_features):
        super(GaussianMerge, self).__init__(in_features, out_features)

    def forward(self, z, mu1, log_var1):
        mu2 = self.mu(z)
        log_var2 = F.softplus(self.log_var(z))
        precision1, precision2 = (1 / torch.exp(log_var1), 1 / torch.exp(log_var2))
        mu = ((mu1 * precision1) + (mu2 * precision2)) / (precision1 + precision2)
        var = 1 / (precision1 + precision2)
        log_var = torch.log(var + 1e-8)
        return self.reparametrize(mu, log_var), mu, log_var


class GumbelSoftmax(Stochastic):
    """
    Layer that represents a sample from a categorical
    distribution using the Gumbel-Softmax trick.
    """

    def __init__(self, in_features, out_features, n_distributions):
        super(GumbelSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_distributions = n_distributions
        self.logits = nn.Linear(in_features, n_distributions * out_features)

    def forward(self, x, tau=1.0):
        logits = self.logits(x).view(-1, self.n_distributions)
        softmax = F.softmax(logits, dim=-1)
        sample = self.reparametrize(logits, tau).view(-1, self.n_distributions, self.out_features)
        sample = torch.mean(sample, dim=1)
        return sample, softmax

    def reparametrize(self, logits, tau=1.0):
        epsilon = Variable(torch.rand(logits.size(), device=logits.device, dtype=logits.dtype), requires_grad=False)
        gumbel = -torch.log(-torch.log(epsilon + 1e-8) + 1e-8)
        y = F.softmax((logits + gumbel) / tau, dim=1)
        return y
