
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from typing import Dictf
from architectures.common_utils import VGGLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(AutoencoderKL):
    """
    A Variational Autoencoder class that inherits from diffusers.AutoencoderKL.
    
    This class encapsulates the forward pass and loss calculation, making the
    training loop cleaner and allowing for easy extension with new methods.
    """
    def __init__(self, *args, **kwargs):
        # The __init__ method accepts all arguments that AutoencoderKL accepts
        # and passes them directly to the parent class constructor.
        # This makes it fully compatible with Hydra's instantiation.
        super().__init__(*args, **kwargs)
        self.vgg_loss = VGGLoss(device)

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of the VAE.

        Args:
            x (torch.Tensor): The input batch of images.

        Returns:
            Dict: A dictionary containing
                the reconstructed images and the posterior distribution.
        """
        x = x["pixel_values"]
        # Encode the input to get the posterior distribution
        posterior = self.encode(x).latent_dist
        
        # Sample from the posterior distribution to get the latents
        latents = posterior.sample()
        
        # Decode the latents to reconstruct the image
        # Sample is used to get the mean of the decoded output distribution
        if self.training:
            reconstructed_x = self.decode(latents).sample
        else:
            reconstructed_x = self.decode(posterior).sample
        
        return {
            "reconstructed_x": reconstructed_x, 
            "posterior": posterior,
            "latent" : latents
        }
        
    def set_optimizers(self, parms):
        self.vae_optim = optim.AdamW(self.parameters(),
                                lr=parms.lr,
                                betas=tuple(parms.betas),
                                weight_decay=parms.weight_decay,
                                eps=parms.eps,
                               )

    def loss_function(
        self,
        original_x: torch.Tensor,
        forward_output: Dict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the VAE loss components.

        Args:
            original_x (torch.Tensor): The original input images.
            forward_output (Dict): The output from the forward pass.
            kl_weight (float): The weight for the KL divergence term.

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
        
        perceptual_loss = self.vgg_loss(original_x, reconstructed_x)

        # KL Divergence Loss
        # This is the standard formula for KL divergence between the posterior and a standard normal prior.
        kl_loss = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp(), dim=[1, 2, 3]).mean()

        # Total Loss
        total_loss = recon_loss + perceptual_loss + kwargs["kl_weight"] * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
        
    def optimize(self, losses, accelerator):
        self.vae_optim.zero_grad()
        accelerator.backward(losses["total_loss"])
        self.vae_optim.step()

    def generate(self, sampler, num_samples: int, device: torch.device, *args) -> torch.Tensor:
        """
        Generates new images by sampling from the latent space.
        
        This is an example of an additional method you can add for more controllability.

        Args:
            num_samples (int): The number of images to generate.
            device (torch.device): The device to generate the images on (e.g., 'cuda').

        Returns:
            torch.Tensor: A tensor of generated images.
        """
        return None
        # The latent space dimensions are determined by the model's architecture.
        # It's (batch_size, latent_channels, height/downsample_factor, width/downsample_factor)
        # latent_height = img_size // (2*(len(self.config.block_out_channels)-1))
        # latent_width = latent_height

        # # Sample random noise from a standard normal distribution
        # z = torch.randn(
        #     num_samples, 
        #     self.config.latent_channels, 
        #     latent_height, 
        #     latent_width
        # ).to(device)
        
        # # Decode the random noise to generate images
        # generated_images = self.decode(z).sample
        
        # generated_images = generated_images*0.5 + 0.5
        
        # return generated_images
        
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path, map_location=device)
        self.load_state_dict(params["network"])
        print("[INFO] loaded the Text Conditioned VAE model", path)

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        params = {
                "network": self.state_dict(),
                }
        save_dir = dump_dir
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = save_dir + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] Text Conditioned VAE model saved to: ", checkpoint_path)