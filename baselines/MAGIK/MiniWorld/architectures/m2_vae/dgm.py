import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..mlp import MLP
    from .stochastic import GaussianSample
    from .vae_utils import ContentEncoder, Decoder
except ImportError:
    from architectures.mlp import MLP
    from architectures.m2_vae.stochastic import GaussianSample
    from architectures.m2_vae.vae_utils import ContentEncoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel_softmax(c_logit, training=True, tau=0.05):
    if training:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(c_logit) + 1e-10) + 1e-10)
        y = F.softmax((c_logit + gumbel_noise) / tau, dim=-1)
        return y
    else:
        return F.softmax(c_logit / tau, dim=-1)


class DeepGenerativeModel(nn.Module):
    def __init__(self, params, class_weights=None):
        """
        Semi-supervised M2-VAE model with a continuous latent variable z and a discrete label variable c.
        Adapted from original MAGIK (baselines/MAGIK/MiniGrid) for MiniWorld (80x80x3 observations).
        """
        super(DeepGenerativeModel, self).__init__()
        dim = int(params["dim"])
        n_downsample = int(params["n_downsample"])
        n_res = int(params["n_res"])
        activ = str(params["activ"])
        pad_type = str(params["pad_type"])
        self.input_dim = int(params["input_dim"])
        self.h_dim = int(params["h_dim"])
        self.z_dim = int(params["z_dim"])
        self.y_dim = int(params["y_dim"])
        classifier_hidden = list(params["classifier_hidden_dim"])
        fc_hidden = list(params["feature_encoder_hidden"])
        conv_out_dim = list(params["feature_encoder_conv_out_dim"])

        self.z_encoder = ContentEncoder(
            n_downsample,
            n_res,
            self.input_dim,
            dim,
            "in",
            activ,
            pad_type=pad_type,
            h_dim=self.h_dim,
            fc_hidden=fc_hidden,
            fc_input_dim=conv_out_dim,
        )
        self.c_encoder = ContentEncoder(
            n_downsample,
            n_res,
            self.input_dim,
            dim,
            "in",
            activ,
            pad_type=pad_type,
            h_dim=self.h_dim,
            fc_hidden=fc_hidden,
            fc_input_dim=conv_out_dim,
        )
        self.z_latent = GaussianSample(self.h_dim, self.z_dim)
        self.decoder = Decoder(
            n_downsample,
            n_res,
            self.z_encoder.output_dim,
            self.input_dim,
            self.z_dim,
            self.y_dim,
            res_norm="in",
            activ=activ,
            pad_type=pad_type,
            fc_input_dim=conv_out_dim,
        )
        self.classifier = MLP(self.h_dim, self.y_dim, classifier_hidden)

        self.register_buffer("c_prior_mu", torch.full((self.z_dim,), 1.0 / self.z_dim))
        self.register_buffer("c_prior_logvar", torch.zeros(self.z_dim))

        self.reconstruction_loss_function = nn.BCELoss(reduction="none")
        if class_weights is not None:
            self.label_loss_function = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.label_loss_function = nn.CrossEntropyLoss()

        self.label_weight = float(params["label_weight"])
        self.recon_weight = float(params["recon_weight"])
        self.hsic_weight = float(params.get("hsic_weight", 0.10))

    def forward(self, x, y=None):
        z_h = self.z_encoder(x)
        c_h = self.c_encoder(x)
        z_latent, z_mu, z_log_var = self.z_latent(z_h)
        c_logits = self.classify(c_h)
        c = sample_gumbel_softmax(c_logits, training=self.training)
        if y is not None:
            c = y
        if self.training:
            x_reconstructed = self.decoder(z_latent, c)
        else:
            x_reconstructed = self.decoder(z_mu, c)

        return x_reconstructed, z_latent, z_mu, z_log_var, c_logits, c

    def get_latent(self, state):
        z_h = self.z_encoder(state)
        _, z_mu, _ = self.z_latent(z_h)
        c_h = self.c_encoder(state)
        c_logits = self.classify(c_h)
        c = sample_gumbel_softmax(c_logits, training=self.training)
        return z_mu, c

    def classify(self, h):
        c_logits = self.classifier(h)
        return c_logits

    def predict_class(self, x: torch.Tensor) -> int:
        """
        Predicts the discrete class index (0..y_dim-1) for an input image tensor x (B, C, H, W) or (C, H, W).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            c_h = self.c_encoder(x)
            c_logits = self.classify(c_h)
            probs = F.softmax(c_logits, dim=-1)
            return int(torch.argmax(probs, dim=-1).item())

    def get_reconstruction(self, x):
        out = self(x)
        return out[0]

    def generate(self, x_input: torch.Tensor, c_cond: torch.Tensor, use_mean_z: bool = True) -> torch.Tensor:
        """
        Generates images conditionally based on the input image and the provided class label vector.

        Args:
            x_input: Input images (B, 3, H, W) in [0, 1]
            c_cond: Conditioning class labels (B, num_classes)
            use_mean_z: If True, uses the mean latent vector mu; otherwise, samples z

        Returns:
            x_generated: Generated images (B, 3, H, W) in [0, 1]
        """
        z_h = self.z_encoder(x_input)
        z, mu, logvar = self.z_latent(z_h)
        if use_mean_z:
            z = mu
        x_generated = self.decoder(z, c_cond)
        return x_generated

    def kl_divergence_z(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def kl_divergence_c(self, logits):
        q_c = F.softmax(logits, dim=-1)
        log_q_c = F.log_softmax(logits, dim=-1)
        K = logits.size(-1)
        log_uniform_prob = -torch.log(torch.tensor(K, dtype=torch.float, device=logits.device))
        kl_per_sample = torch.sum(q_c * (log_q_c - log_uniform_prob), dim=1)
        kl = kl_per_sample.mean()
        return kl

    def rbf_kernel(self, x, sigma=None):
        pairwise_dist = torch.cdist(x, x) ** 2
        if sigma is None:
            sigma = torch.sqrt(0.5 * torch.median(pairwise_dist))
            sigma = sigma.clamp(min=1e-4)
        return torch.exp(-pairwise_dist / (2 * sigma ** 2 + 1e-8))

    def linear_kernel(self, x):
        return x @ x.T

    def hsic(self, z, c, sigma_z=None):
        batch_size = z.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device)
        H = torch.eye(batch_size, device=z.device) - (1.0 / batch_size) * torch.ones(batch_size, batch_size, device=z.device)
        K_z = self.rbf_kernel(z, sigma_z)
        K_c = self.linear_kernel(c)
        K_z = H @ K_z @ H
        K_c = H @ K_c @ H
        return torch.trace(K_z @ K_c) / ((batch_size - 1) ** 2 + 1e-8)

    def L(self, x, x_recon, mu_z, logvar_z, y, logits_c, x_c, ws_weight, kl_weight):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="none")
        recon_loss = torch.mean(recon_loss.sum(dim=-1))
        # KL divergence for z
        kl_z = self.kl_divergence_z(mu_z, logvar_z)
        # KL divergence for c
        kl_c = self.kl_divergence_c(logits_c)
        label_loss = self.label_loss_function(logits_c, y)

        # Hilbert-Schmidt Independence Criterion
        hsic_loss = self.hsic(mu_z, x_c)

        total_loss = (
            self.recon_weight * recon_loss
            + kl_weight * kl_z
            + kl_weight * kl_c
            + self.label_weight * label_loss
            + self.hsic_weight * hsic_loss
        )
        if torch.isnan(total_loss):
            raise ValueError("Loss is nan")
        return total_loss, label_loss

    def U(self, x, x_recon, mu_z, logvar_z, logits_c, x_c, ws_weight, kl_weight):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="none")
        recon_loss = torch.mean(recon_loss.sum(dim=[-1, -2, -3]))
        # KL divergence for z
        kl_z = self.kl_divergence_z(mu_z, logvar_z)
        # KL divergence for c
        kl_c = self.kl_divergence_c(logits_c)

        # Hilbert-Schmidt Independence Criterion
        hsic_loss = self.hsic(mu_z, x_c)

        total_loss = (
            self.recon_weight * recon_loss
            + kl_weight * kl_z
            + kl_weight * kl_c
            + self.hsic_weight * hsic_loss
        )
        if torch.isnan(total_loss):
            raise ValueError("Loss is nan")
        return total_loss, recon_loss, kl_z, kl_c, hsic_loss

    def save(self, save_path, epochs=0):
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, "model.pt")
        torch.save(self.state_dict(), model_file)
        print(f"[INFO] model saved after {epochs} epochs at: {model_file}")
        specs_file = os.path.join(save_path, "specs.json")
        with open(specs_file, "w") as f:
            json.dump(
                {
                    "input_dim": self.input_dim,
                    "num_goals": self.y_dim,
                    "h_dim": self.h_dim,
                    "latent_dim": self.z_dim,
                },
                f,
                indent=2,
            )

    def load(self, model_dir):
        target_device = next(self.parameters()).device if list(self.parameters()) else device
        if os.path.isdir(model_dir):
            model_dir = os.path.join(model_dir, "model.pt")
        params = torch.load(model_dir, map_location=target_device, weights_only=True)
        if isinstance(params, dict) and "state_dict" in params:
            params = params["state_dict"]
        self.load_state_dict(params)
        print("[INFO] MAGIK M2-VAE model loaded from:", model_dir)
