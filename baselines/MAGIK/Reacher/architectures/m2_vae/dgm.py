import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.distributions import Gamma
from architectures.mlp import MLP, Linear
from architectures.common_utils import focal_loss
from architectures.m2_vae.stochastic import GaussianSample
from helper_functions.utils import sample_gumbel_softmax
# from architectures.m2_vae.distributions import log_standard_categorical
from architectures.m2_vae.vae_utils import ContentEncoderMLP, Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepGenerativeModel(nn.Module):
    def __init__(self, params, class_weights):
        """
        Semi-supervised VAE disentagling class-agnostic and class related features
        """
        super(DeepGenerativeModel, self).__init__()
        input_dim = params['input_dim']
        hidden_dim = params['hidden_dim']
        decoder_hidden_dims = params['decoder_hidden_dims']
        self.input_dim = params['input_dim']
        self.h_dim = params['h_dim']
        self.z_dim = params['z_dim']
        self.y_dim = params['y_dim']
        classifier_hidden = params['classifier_hidden_dim']
        self.z_encoder = ContentEncoderMLP(input_dim, hidden_dim, self.h_dim)
        self.c_encoder = ContentEncoderMLP(input_dim, hidden_dim, self.h_dim)
        self.z_latent = GaussianSample(self.h_dim, self.z_dim)
        self.decoder = Decoder(self.z_dim, decoder_hidden_dims, input_dim, self.y_dim)
        self.classifier = Linear(self.h_dim, self.y_dim, dropout_prob=0.0)

        self.class_weights_tensor = class_weights
        self.normalised_class_weights_tensor = self.class_weights_tensor / self.class_weights_tensor.sum()
        self.label_loss_function = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

        # self.label_loss_function = nn.CrossEntropyLoss()

        self.label_weight, self.recon_weight = params['label_weight'], params['recon_weight']
        self.hsic_weight = params['hsic_weight']
        self.aux_loss_weight = params['aux_loss_weight']

        self.reconstruction_loss_per_feature = []

    @staticmethod
    def reparameterize_dirichlet(alpha):
        gamma_samples = Gamma(concentration=alpha, rate=torch.ones_like(alpha).to(device)).rsample()
        c = gamma_samples / torch.sum(gamma_samples, dim=1, keepdim=True)
        return c

    def forward(self, x, y=None, temp=0.01):
        # Add label and data and generate latent variable
        z_h = self.z_encoder(x)
        c_h = self.c_encoder(x)
        z_latent, z_mu, z_log_var = self.z_latent(z_h)
        c_logits = self.classify(c_h)

        # tau = torch.min(torch.exp(self.log_tau), torch.tensor(temp).to(device))

        c = sample_gumbel_softmax(c_logits, tau=temp)
        # c_residual = self.c_residual(c_h)
        # Reconstruct data point from latent data and label
        if y is not None:
            c = y
        # c = torch.cat([c,c_residual], dim=-1)
        if self.training:
            x_reconstructed = self.decoder(z_latent, c)
        else:
            x_reconstructed = self.decoder(z_mu, c)

        return x_reconstructed, z_latent, z_mu, z_log_var, c_logits, c
    
    
    def classify(self, h):
        c_logits = self.classifier(h)
        return c_logits
    
    def generate(self, x_input: torch.Tensor, c_cond: torch.Tensor, use_mean_z: bool = True) -> torch.Tensor:
        """
        Generates data conditionally based on the input image and the provided soft label

        Args:
            x_input: Input data (B, F)
            c_cond: Conditioning soft labels (B, num_classes)
            use_mean_z: If True, uses the mean latent vector mu; otherwise, samples z

        Returns:
            x_generated: Generated data (B, F)
        """
        z_h = self.z_encoder(x_input)
        # c_h = self.c_encoder(x_input)
        z, mu, logvar = self.z_latent(z_h)
        # residual = self.residual_connection(c_h)
        if use_mean_z:
            z = mu  # Use mean of q(z|x)
        x_generated = self.decoder(z, c_cond)
        return x_generated

    
    def kl_divergence_z(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def kl_divergence_c(self, logits):
        q_c = F.softmax(logits, dim=-1)                 # [batch_size, num_classes]
        log_q_c = F.log_softmax(logits, dim=-1)         # [batch_size, num_classes]
        K = logits.size(-1)                             # Number of classes
        # Compute log uniform prior probability
        log_uniform_prob = -torch.log(torch.tensor(K, dtype=torch.float, device=logits.device))
        kl_per_sample = torch.sum(q_c * (log_q_c - log_uniform_prob), dim=1)    # [batch_size]
        kl = kl_per_sample.mean()
        return kl

    def rbf_kernel(self, x, sigma=None):
        # x: (batch_size, dim)
        pairwise_dist = torch.cdist(x, x) ** 2  # (batch_size, batch_size)

        if sigma is None:
            # Median heuristic for sigma
            sigma = torch.sqrt(0.5 * torch.median(pairwise_dist))
            sigma = sigma.clamp(min=1e-4)  # Avoid σ ≈ 0

        return torch.exp(-pairwise_dist / (2 * sigma**2 + 1e-8))
    
    def linear_kernel(self, x):
        # For categorical variables (Gumbel-Softmax outputs)
        return x @ x.T

    def hsic(self, z, c, sigma_z=None):
        # z: (batch_size, z_dim), continuous
        # c: (batch_size, c_dim), categorical probabilities from Gumbel-Softmax

        batch_size = z.size(0)
        H = torch.eye(batch_size) - (1.0 / batch_size) * torch.ones(batch_size, batch_size)
        H = H.to(z.device)

        z = (z - z.mean(0)) / (z.std(0) + 1e-8)  # Normalize z
        c = (c - c.mean(0)) / (c.std(0) + 1e-8)  # Normalize c

        # Compute kernel matrices
        K_z = self.rbf_kernel(z, sigma_z)
        K_c = self.linear_kernel(c)  # Treat c as continuous probabilities

        # Center kernel matrices
        K_z = H @ K_z @ H
        K_c = H @ K_c @ H

        # HSIC value
        return torch.trace(K_z @ K_c) / ((batch_size - 1) ** 2 + 1e-8)
    
    def L(self, x, x_recon, mu_z, logvar_z, y, logits_c, x_c, kl_weight, epoch):
        #Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='none')
        self.reconstruction_loss_per_feature.append(torch.mean(recon_loss, dim=0))
        recon_loss = torch.mean(recon_loss.sum(dim=-1))
        # KL divergence for z
        kl_z = self.kl_divergence_z(mu_z, logvar_z)
        # KL divergence for c
        kl_c = self.kl_divergence_c(logits_c)
        label_loss = self.label_loss_function(logits_c, torch.argmax(y, axis=-1))

        #Hilbert-Schmidit Independence criterion
        hsic_loss = self.hsic(mu_z, x_c)
        
        total_loss = self.recon_weight*recon_loss + kl_weight*kl_z + kl_weight*kl_c + self.label_weight*label_loss + self.hsic_weight*hsic_loss
        if torch.isnan(total_loss):
            raise ValueError ("Loss is nan")
        return total_loss, label_loss
    
    def U(self, x, x_recon, mu_z, logvar_z, logits_c, x_c, kl_weight, epoch):
        #Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='none')
        recon_loss = torch.mean(recon_loss.sum(dim=-1))
        # KL divergence for z
        kl_z = self.kl_divergence_z(mu_z, logvar_z)
        # KL divergence for c
        kl_c = self.kl_divergence_c(logits_c)

        #Hilbert-Schmidit Independence criterion
        hsic_loss = self.hsic(mu_z, x_c)

        total_loss = self.recon_weight*recon_loss + kl_weight*kl_z + kl_weight*kl_c + self.hsic_weight*hsic_loss 
        if torch.isnan(total_loss):
            raise ValueError ("Loss is nan")
        return total_loss, recon_loss, kl_z, kl_c, hsic_loss
 
    def save(self, save_path, epochs=0):
        torch.save(self.state_dict(), f'{save_path}/model.pt')
        print(f"[INFO] model saved after {epochs} at, {save_path}/model.pt")
        with open(f'{save_path}/specs.json', 'w') as f:
            f.write('''{
            "input_dim": %d,
            "num_goals": %d,
            "h_dim": %d,
            "latent_dim": %d
            }''' % (self.input_dim, self.y_dim, self.h_dim, self.z_dim))

    def load(self, model_dir):
        params = torch.load(model_dir, map_location=device, weights_only=True)
        self.load_state_dict(params, strict=False)
        # self.load_state_dict(params)
        print("[INFO] VAE model loaded from: ", model_dir)  