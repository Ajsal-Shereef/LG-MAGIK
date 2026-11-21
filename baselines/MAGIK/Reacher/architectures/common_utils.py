import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Compute the focal loss for multi-class classification.
    
    Args:
      inputs (Tensor): Raw model outputs (logits) of shape [B, num_classes].
      targets (Tensor): Ground truth class indices of shape [B].
      alpha (float or Tensor): Weighting factor for classes. Can be a scalar or a tensor of shape [num_classes].
      gamma (float): Focusing parameter.
      reduction (str): 'none' | 'mean' | 'sum'
      
    Returns:
      Tensor: The focal loss.
    """
    # Compute log-softmax
    logpt = F.log_softmax(inputs, dim=1)
    pt = torch.exp(logpt)
    
    # Gather the log probabilities for the true class.
    logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # If alpha is a tensor, gather the per-class weights.
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(inputs.device)
        at = alpha.gather(0, targets)
    else:
        at = alpha
    
    loss = -at * (1 - pt) ** gamma * logpt
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def identity(x, dim=0):
    """
    Return input without any change.
    x: torch.Tensor
    :return: torch.Tensor
    """
    return x


def normalize_01(x, dim=1):
    """
    Normalize the elements in x to [0, 1]
    x: torch.Tensor, the shape should be (batch size, flatten vector)
    :return: torch.Tensor
    """

    min_x = torch.min(x, dim=1, keepdim=True)[0]
    max_x = torch.max(x, dim=1, keepdim=True)[0]

    delta = max_x - min_x
    zero_idxs = (delta[:, 0] == 0)
    delta[zero_idxs, :] = 1.0

    x = x - min_x
    x = x/delta
    x[zero_idxs, :] = 0

    return x


def soft_update(local, target, tau):
    """
    Soft-update: target = tau*local + (1-tau)*target.
    local: nn.Module
    target: nn.Module
    tau: float
    """
    for t_param, l_param in zip(target.parameters(), local.parameters()):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


def hard_update(local, target):
    """
    Hard update: target <- local.
    local: nn.Module
    target: nn.Module
    """
    target.load_state_dict(local.state_dict())


def set_random_seed(seed, env):
    """
    Set random seed
    seed: int
    env: gym.Env
    """
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_one_hot(labels, c):
    """
    Converts an integer label to a one-hot Variable.
    labels (torch.Tensor): list of labels to be converted to one-hot variable
    c (int): number of possible labels
    """
    y = torch.eye(c).to(device)
    labels = labels.type(torch.LongTensor)
    return y[labels]


def one_hot_to_discrete_action(action, is_softmax=False):
    """
    convert the discrete action representation to one-hot representation
    action: in the format of a vector [one-hot-selection]
    """
    flatten_action = action.flatten()
    if not is_softmax:
        return np.argmax(flatten_action)
    else:
        return np.random.choice(flatten_action.shape[0], size=1, p=softmax(flatten_action)).item()


def discrete_action_to_one_hot(action_id, action_dim):
    """
    return one-hot representation of the action in the format of np.ndarray
    """
    action = np.array([0 for _ in range(action_dim)]).astype(np.float)
    action[action_id] = 1.0
    # in the format of one-hot-vector
    return action


