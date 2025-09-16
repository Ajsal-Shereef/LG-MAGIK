import torch
import torch.nn as nn
import numpy as np

from typing import Dict
import torch.nn.functional as F
from architectures.mlp import MLP
from torchvision.utils import make_grid
from architectures.common_utils import grad_reverse
from architectures.cnn import CNNEncoder, CNNTextConditionedDecoder
from architectures.stochastic import GaussianSampleSpatial

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
        output_dim = kwargs["output_dim"]
        clip_model = kwargs["text_encoder"]
        encoder_final_dim = kwargs["encoder_final_dim"]
        
        self.encoder = CNNEncoder(
            n_downsample, encoder_n_res, input_dim, dim, norm, activ, pad_type=pad_type)
        
        self.bottleneck = GaussianSampleSpatial(self.encoder.output_dim, kwargs["latent_channel"])
        
        self.decoder = CNNTextConditionedDecoder(n_downsample, self.encoder.output_dim, output_dim, clip_model, kwargs["latent_channel"])
        self.caption_descriminator = MLP(kwargs["latent_channel"]*np.prod(encoder_final_dim), self.decoder.tokenizer.model_max_length*self.decoder.text_dim, kwargs["discriminator_fc_hidden"])
        
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
        text_tokens = x["input_ids"]
        # Encode the input to get the posterior distribution
        hidden = self.encoder(images)
        
        # Posterior and sampled latent
        sampler = self.bottleneck(hidden)
 
        # Decode the latents to reconstruct the image
        # .sample is used to get the mean of the decoded output distribution
        if self.training:
            reconstructed_x = self.decoder(sampler.latent, text_tokens)
        else:
            reconstructed_x = self.decoder(sampler.mean, text_tokens)
        
        return {
            "x" : images,
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
        kl_loss = -0.5 * torch.sum(1 + posterior.log_variance - posterior.mean.pow(2) - posterior.log_variance.exp(), dim=[1,2,3]).mean()
        
        # Adversarial loss
        z_grl=grad_reverse(posterior.latent, kwargs["adv_lambda"])
        pred=self.caption_descriminator(z_grl.view(z_grl.shape[0], -1))
        tgt_n = F.normalize(self.decoder.text_feats.detach(), dim=-1)
        pred_n = F.normalize(pred.view(tgt_n.shape[0], tgt_n.shape[1], -1), dim=-1)
        disc_loss = 1 - (pred_n * tgt_n).sum(dim=[1,2]).mean()
        
        # Total Loss
        total_loss = recon_loss + kwargs["kl_weight"] * kl_loss + kwargs["adv_weight"]*disc_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "advesarial_loss" : disc_loss,
        }
        
    def generate(self, output: dict, num_samples: int, device, *prompts) -> torch.Tensor:
        """
        Generates a comparison grid to visualize disentanglement.

        For each of the `num_samples`, it creates a row containing:
        1. The original input image.
        2. The reconstruction of that image.
        3. New images generated by pairing the same latent code with each provided prompt.

        Args:
            batch (dict): The original input batch, containing "pixel_values".
            output (dict): The output from the model's forward pass.
            num_samples (int): The number of latent vectors (and thus rows) to use.
            *prompts: A tuple of text prompts, e.g., ("a photo of a cat", "a drawing of a dog").

        Returns:
            torch.Tensor: A single image grid of size (num_samples, 2 + num_prompts).
        """
        # --- 1. SETUP ---
        if not prompts:
            raise ValueError("The generate function requires at least one text prompt.")

        # Handle if prompts are passed as a list/tuple inside the first argument
        prompts = prompts[0] if isinstance(prompts[0], (list, tuple)) else prompts
        num_prompts = len(prompts)

        # --- 2. GATHER BASE IMAGES AND LATENTS ---
        # Get the latents, original images, and reconstructions from the first `num_samples` of the batch.
        # We use .mean for generation as it's the most representative point in the latent distribution.
        latents = output["posterior"].mean[:num_samples]

        # Normalize images from [-1, 1] to [0, 1] for visualization
        original_images = (output["x"][:num_samples].detach() * 0.5 + 0.5).clamp(0, 1)
        reconstructions = (output["reconstructed_x"][:num_samples].detach() * 0.5 + 0.5).clamp(0, 1)

        # --- 3. GENERATE PROMPT-DRIVEN IMAGES ---
        # Tokenize and encode the text prompts to get embeddings
        tokenised_text = self.decoder.tokenizer(
            prompts, max_length=self.decoder.tokenizer.model_max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        input_ids = tokenised_text["input_ids"].to(device)

        # Expand latents and embeddings to create all (z, prompt) pairs
        z_expanded = latents.repeat_interleave(num_prompts, dim=0)
        text_embeddings_expanded = input_ids.repeat(num_samples, 1)

        # Decode the pairs to generate new images
        generated_images = self.decoder(z_expanded, text_embeddings_expanded)
        # Normalize generated images for visualization
        generated_images = (generated_images.detach() * 0.5 + 0.5).clamp(0, 1)

        # --- 4. ASSEMBLE THE GRID ---
        # Create a list to hold all images in the correct order for the grid
        grid_images = []
        for i in range(num_samples):
            # For each sample, add the original image and its reconstruction
            grid_images.append(original_images[i])
            grid_images.append(reconstructions[i])

            # Then, add all the images generated using this sample's latent code
            start_idx = i * num_prompts
            end_idx = start_idx + num_prompts
            grid_images.extend(list(generated_images[start_idx:end_idx]))

        # Stack the list of image tensors into a single batch
        final_image_tensor = torch.stack(grid_images)

        # Create the grid with (2 + num_prompts) columns
        grid = make_grid(final_image_tensor, nrow=(2 + num_prompts))

        return grid
        