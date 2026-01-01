import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from typing import Optional, Dict

class DiffusionImaginationModel(nn.Module):
    def __init__(self, vae_model, unet_config: Optional[Dict] = None, num_train_timesteps=1000, scaling_factor=1.0):
        super().__init__()
        
        self.scaling_factor = scaling_factor
        
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
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.inference_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

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
            
            # Scale latents to unit variance
            latents = latents * self.scaling_factor
            
        # 2. Sample Noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # 3. Sample Timesteps
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        
        # 4. Add Noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Predict Noise
        # Pad from 10x10 to 16x16 for U-Net stability
        # pad (left, right, top, bottom) => (0, 6, 0, 6) to go 10->16
        noisy_latents_padded = F.pad(noisy_latents, (0, 6, 0, 6))
        
        # UNet forward on 16x16
        pred_padded = self.unet(noisy_latents_padded, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Crop back to 10x10
        # [B, C, 16, 16] -> [B, C, 10, 10]
        model_pred = pred_padded[:, :, :10, :10]
        
        return model_pred, noise

    @torch.no_grad()
    def imagine(self, state_image, target_text_embeddings, uncond_text_embeddings=None, strength=0.8, num_inference_steps=50, guidance_scale=7.5):
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
        
        init_latents = init_latents * self.scaling_factor
        
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
            # Pad latents 10x10 -> 16x16 for U-Net
            latents_padded = F.pad(latents, (0, 6, 0, 6))
            
            # Expand for classifier free guidance (on padded latents)
            latent_model_input = torch.cat([latents_padded] * 2)
            latent_model_input = self.inference_scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare text embeddings for CFG
            if uncond_text_embeddings is None:
                # Fallback to duplicating (no guidance) if not provided
                uncond_text_embeddings = target_text_embeddings
            
            text_embeddings_input = torch.cat([uncond_text_embeddings, target_text_embeddings])
            
            # Predict noise
            noise_pred_padded = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_input).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred_padded.chunk(2)
            noise_pred_padded = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Crop predicted noise to 10x10 to step the scheduler?
            # Scheduler expects noise of same shape as latents.
            noise_pred = noise_pred_padded[:, :, :10, :10]
            
            # Step
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample
            


        # 4. Decode
        latents = latents / self.scaling_factor
        return latents
