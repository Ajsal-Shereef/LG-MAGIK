import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, conditioning_dim):
        super(FiLMLayer, self).__init__()
        # Linear layers to generate gamma and beta
        self.gamma_fc = nn.Linear(conditioning_dim, feature_dim)
        self.beta_fc = nn.Linear(conditioning_dim, feature_dim)

    def forward(self, features, conditioning):
        # Compute gamma and beta from the conditioning input
        gamma = self.gamma_fc(conditioning)
        beta = self.beta_fc(conditioning)
        
        # # Reshape gamma and beta for broadcasting
        # gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        # beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # Apply FiLM modulation
        return gamma * features + beta
