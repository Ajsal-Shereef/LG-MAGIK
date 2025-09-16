import torch
import torch.nn as nn
import numpy as np

from typing import Dict
import torch.nn.functional as F
from architectures.mlp import MLP
from architectures.common_utils import grad_reverse
from architectures.cnn import CNNEncoder, CNNTextConditionedDecoder
from architectures.stochastic import GaussianSample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextConditionedVAE(nn.Module):
    def __init__(self, **kwargs):
        super(TextConditionedVAE, self).__init__()
        n_downsample = kwargs["n_downsample"]
        encoder_n_res = kwargs["encoder_n_res"]
        input_dim = kwargs["input_dim"]
        dim = kwargs["dim"]
        norm = kwargs["norm"]
        activ = kwargs["activ"]
        pad_type = kwargs["pad_type"]
        h_dim = kwargs["h_dim"]
        z_dim = kwargs["z_dim"]
        fc_hidden = kwargs["fc_hidden"]
        fc_input_dim = kwargs["fc_input_dim"]
        self.latent_dim = z_dim
        output_dim = kwargs["output_dim"]
        clip_model = kwargs["text_encoder"]
        
        self.encoder = CNNEncoder(
            n_downsample, encoder_n_res, input_dim, dim, norm, activ,
            pad_type=pad_type, h_dim=h_dim, fc_hidden=fc_hidden, fc_input_dim=fc_input_dim
        )
        
        self.bottleneck = GaussianSample(h_dim, z_dim)
        
        self.decoder = CNNTextConditionedDecoder(n_downsample, self.encoder.output_dim , output_dim, z_dim, clip_model, fc_input_dim=fc_input_dim)
        self.caption_descriminator = MLP(z_dim, self.decoder.text_dim, kwargs["discriminator_fc_hidden"])
        
    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of the Text conditioned VAE.

        Args:
            x (torch.Tensor): The input batch of images, text.

        Returns:
            Dict: A dictionary containing
                the reconstructed images and the posterior distribution.
        """
        images = x["pixel_values"]
        text_tockens = x["input_ids"]
        # Encode the input to get the posterior distribution
        hidden = self.encoder(images)
        
        # Posterior and sampled latent
        sampler = self.bottleneck(hidden)
 
        # Decode the latents to reconstruct the image
        # .sample is used to get the mean of the decoded output distribution
        if self.training:
            reconstructed_x = self.decoder(sampler.latent, text_tockens)
        else:
            reconstructed_x = self.decoder(sampler.mean, text_tockens)
        
        return {
            "reconstructed_x": reconstructed_x, 
            "posterior": sampler,
            "latent" : sampler.latent
        }
        
    def loss_function(
        self,
        original_x: torch.Tensor,
        forward_output: Dict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the Text conditioned VAE loss components.

        Args:
            original_x (torch.Tensor): The original input images.
            forward_output (Dict): The output from the forward pass.
            training_configs (dict): The weight for the KL divergence term.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing total loss,
                reconstruction loss, and KL divergence loss.
        """
        original_x = original_x["pixel_values"]
        reconstructed_x = forward_output["reconstructed_x"]
        posterior = forward_output["posterior"]

        # Reconstruction Loss (MSE)
        # We flatten the image dimensions and sum over them, then take the mean over the batch.
        recon_loss = F.mse_loss(reconstructed_x, original_x, reduction="none")
        recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1).mean()

        # KL Divergence Loss
        # This is the standard formula for KL divergence between the posterior and a standard normal prior.
        kl_loss = -0.5 * torch.sum(1 + posterior.log_variance - posterior.mean.pow(2) - posterior.log_variance.exp(), dim=-1).mean()
        
        # Adversarial loss
        z_grl=grad_reverse(posterior.latent, kwargs["adv_lambda"])
        pred=self.caption_descriminator(z_grl)
        disc_loss=F.mse_loss(pred, self.decoder.text_feats.mean(1).detach())

        # Total Loss
        total_loss = recon_loss + kwargs["kl_weight"] * kl_loss + kwargs["adv_lambda"]*disc_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "advesarial_loss" : disc_loss,
        }
        
    def generate(self, img_size, num_samples: int, device: torch.device, *args) -> torch.Tensor:
        """
        Generates new images by sampling from the latent space, conditioning each
        latent sample on every provided text prompt.
        
        Args:
            img_size (int) : image_size
            num_samples (int): The number of latent vectors to sample.
            device (torch.device): The device to generate the images on (e.g., 'cuda').
            *args: A tuple of text prompts, e.g., ("a photo of a cat", "a drawing of a dog").

        Returns:
            torch.Tensor: A tensor of generated images with shape 
                          (num_samples * len(args), channels, height, width).
        """
        # Ensure there are prompts to generate from
        if not args:
            raise ValueError("The generate function requires at least one text prompt in *args.")
            
        prompts = args[0] if isinstance(args[0], (list, tuple)) else args
        num_prompts = len(prompts)
        
        # Determine the device from the model's parameters
        device = next(self.parameters()).device
        num_samples = num_samples//num_prompts
        # 1. Sample initial random noise
        z = torch.randn(
            num_samples, 
            self.latent_dim
        ).to(device)
        
        # 2. Tokenize all text prompts
        tokenised_text = self.decoder.tokenizer(
            prompts, 
            max_length=self.decoder.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        # Move tokens to the correct device
        input_ids = tokenised_text["input_ids"].to(device)

        # --- TENSOR MANIPULATION TO MATCH BATCH SIZES ---
        # Original shapes: z -> (num_samples, z_dim), text_embeddings -> (num_prompts, text_dim)
        
        # 3. Expand z: Repeat each z vector `num_prompts` times
        # Shape becomes: (num_samples * num_prompts, z_dim)
        z_expanded = z.repeat_interleave(num_prompts, dim=0)

        # 4. Expand text_embeddings: Repeat the whole block of embeddings `num_samples` times
        # Shape becomes: (num_samples * num_prompts, text_dim)
        text_input_ids_expanded = input_ids.repeat(num_samples, 1)
        
        # Now, z_expanded and text_embeddings_expanded both have a batch size of (num_samples * num_prompts)

        # 5. Decode the paired latents and embeddings to generate images
        generated_images = self.decoder(z_expanded, text_input_ids_expanded)
        
        return generated_images
        