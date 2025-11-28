import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from architectures.common_utils import get_normalisation_2d

class PatchDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator.
    Input: image tensor in range [-1, 1], shape [B, C, H, W]
    Output: patch logits [B, 1, H', W']
    """
    def __init__(self, in_channels=3, base_channels=64, n_layers=3, norm='in'):
        super().__init__()
        layers = []

        # first conv (no normalization)
        layers += [
            spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        curr_dim = base_channels
        for i in range(1, n_layers):
            next_dim = min(curr_dim * 2, 512)
            conv = spectral_norm(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1))
            norm_layer = get_normalisation_2d(norm, next_dim)
            layers += [conv, norm_layer, nn.LeakyReLU(0.2, inplace=True)]
            curr_dim = next_dim

        # final patch output
        layers += [spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1))]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# === MINE: Mutual Information Neural Estimator ===
class MINECritic(nn.Module):
    def __init__(self, z_dim, t_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + t_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, t):
        zt = torch.cat([z, t], dim=-1)
        return self.net(zt)
    
# === The Core VQ-VAE Bottleneck ===
class VectorQuantizer(nn.Module):
    """
    The core VQ-VAE layer. This layer takes a tensor of continuous features and
    maps them to a discrete set of learned embedding vectors (the "codebook").
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, latents):
        # latents shape: [Batch, Channels, Height, Width]
        # CRITICAL: The input `latents` Channels dimension MUST equal self.embedding_dim
        if latents.shape[1] != self.embedding_dim:
            raise ValueError(f"Input latent channel dimension ({latents.shape[1]}) "
                             f"must match embedding dimension ({self.embedding_dim})")

        # Reshape latents for distance calculation: [B, C, H, W] -> [B, H, W, C]
        latents_reshaped = latents.permute(0, 2, 3, 1).contiguous()
        # Flatten to a list of vectors: [B*H*W, C]
        latents_flat = latents_reshaped.view(-1, self.embedding_dim)

        # --- Quantization ---
        distances = (torch.sum(latents_flat**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(latents_flat, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)
        quantized_latents_flat = self.embedding(encoding_indices)
        quantized_latents = quantized_latents_flat.view(latents_reshaped.shape)

        # --- Loss Calculation ---
        e_loss = F.mse_loss(quantized_latents.detach(), latents_reshaped)
        q_loss = F.mse_loss(quantized_latents, latents_reshaped.detach())
        vq_loss = q_loss + self.commitment_cost * e_loss

        # --- Straight-Through Estimator ---
        quantized_latents = latents_reshaped + (quantized_latents - latents_reshaped).detach()
        
        # --- FIX: Reshape back to [B, C, H, W] for the decoder ---
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()
        
        # --- Perplexity Calculation (no change needed here) ---
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "vq_loss": vq_loss,
            "quantized": quantized_latents,
            "perplexity": perplexity,
            "encoding_indices": encoding_indices.view(latents.shape[0], latents.shape[2], latents.shape[3])
        }
