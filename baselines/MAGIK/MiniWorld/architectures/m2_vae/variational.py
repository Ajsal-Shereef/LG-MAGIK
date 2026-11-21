from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .distributions import log_standard_categorical
except ImportError:
    from architectures.m2_vae.distributions import log_standard_categorical


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max_val), dim=dim, keepdim=True) + 1e-8) + max_val


def enumerate_discrete(x, y_dim):
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])
    if x.is_cuda:
        generated = generated.cuda()
    return generated.float()


class ImportanceWeightedSampler(object):
    def __init__(self, mc=1, iw=1):
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t
        return self.t


class SVI(nn.Module):
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)

    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None):
        is_labelled = False if y is None else True
        xs, ys = (x, y)

        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)

        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        reconstruction, z, z_mu, z_log_var = self.model(xs, ys)
        likelihood = -self.likelihood(reconstruction, xs)
        prior = -log_standard_categorical(ys)
        elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        L = self.sampler(elbo)

        if is_labelled:
            return torch.mean(L)

        logits = self.model.classify(x)
        L = L.view_as(logits.t()).t()
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)
        U = L + H
        return torch.mean(U)
