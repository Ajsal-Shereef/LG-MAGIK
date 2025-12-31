import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from typing import Optional, Dict

class DiffusionImaginationModel(nn.Module):
    def __init__(self, vae_model, unet_config: Optional[Dict] = None):
        super().__init__()
        
        # 1. Helper VAE (Frozen)
        # We assume vae_model is already loaded and passed here.
        self.vae = vae_model
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 2. U-Net
        # If no config provided, load a default or small config
        # For this project, we might want to start with a standard small config if not loading pretrained
        if unet_config is None:
             # Default small config for 64x64 or 128x128 latents
             # Get latent channel from VAE config if available
            latent_channels = 4
            if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'latent_channels'):
                latent_channels = self.vae.config.latent_channels
            elif hasattr(self.vae, 'latent_channel'):
                latent_channels = self.vae.latent_channel
                
            self.unet = UNet2DConditionModel(
                sample_size=16,
                in_channels=latent_channels,
                out_channels=latent_channels,
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                cross_attention_dim=512, # CLIP text embedding dim
                attention_head_dim=8,
            )
        else:
            self.unet = UNet2DConditionModel(**unet_config)

        # 3. Scheduler
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.inference_scheduler = DDIMScheduler(num_train_timesteps=1000)

    def forward(self, x, text_embeddings):
        """
        Training forward pass.
        x: Input images [B, C, H, W]
        text_embeddings: [B, SeqLen, Dim]
        """
        # 1. Encode to Latent
        with torch.no_grad():
            # Standard VAE encode
            posterior = self.vae.encode(x).latent_dist
            latents = posterior.sample()
            
            # Optional: Scale latents if VAE expects it (standard SD uses 0.18215)
            # We will use 0.18215 factor for consistency with typical usage
            # latents = latents * 0.18215
            
        # 2. Sample Noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # 3. Sample Timesteps
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        
        # 4. Add Noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Predict Noise
        # UNet forward
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        
        return model_pred, noise

    @torch.no_grad()
    def imagine(self, state_image, target_text_embeddings, strength=0.8, num_inference_steps=50, guidance_scale=7.5):
        """
        SDEdit / Img2Img pipeline.
        
        state_image: [B, C, H, W]
        target_text_embeddings: [B, SeqLen, Dim]
        strength: 0.0 to 1.0. Higher = more destruction of original image.
        """
        self.inference_scheduler.set_timesteps(num_inference_steps)
        
        # 1. Encode Source
        # Standard VAE encode
        posterior = self.vae.encode(state_image).latent_dist
        init_latents = posterior.mean # Use mean for deterministic starting point
        
        # init_latents = init_latents * 0.18215
        
        # 2. Add Noise (Partial)
        # Calculate start timestep
        init_timestep = int(num_inference_steps * strength)
        timesteps = self.inference_scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps], device=state_image.device)
        
        noise = torch.randn_like(init_latents)
        latents = self.inference_scheduler.add_noise(init_latents, noise, timesteps)
        
        # 3. Denoise Loop
        t_start = max(num_inference_steps - init_timestep, 0)
        
        for i, t in enumerate(self.inference_scheduler.timesteps[t_start:]):
            # Expand for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.inference_scheduler.scale_model_input(latent_model_input, t)
            
            # Duplicate text embeddings for CFG (simulated, ideally use uncond)
            text_embeddings_input = torch.cat([target_text_embeddings] * 2)
            
            # Predict noise
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_input).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Decode
        # latents = latents / 0.18215
        return latents
