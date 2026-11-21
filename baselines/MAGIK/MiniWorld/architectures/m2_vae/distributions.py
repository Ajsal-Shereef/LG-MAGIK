import math
import torch
import torch.nn.functional as F


def log_standard_gaussian(x):
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    log_pdf = -0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu) ** 2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_standard_categorical(p):
    prior = F.softmax(torch.ones_like(p), dim=-1)
    prior.requires_grad = False
    cross_entropy = -p * torch.log(prior + 1e-8)
    return cross_entropy
