
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
            self.unet = UNet2DConditionModel(
                sample_size=16,
                in_channels=self.vae.latent_channel if hasattr(self.vae, 'latent_channel') else 4,
                out_channels=self.vae.latent_channel if hasattr(self.vae, 'latent_channel') else 4,
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
            hidden = self.vae.encoder(x)
            sampler = self.vae.bottleneck(hidden)
            latents = sampler.latent * 0.18215 
            
        # 2. Sample Noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # 3. Sample Timesteps
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        
        # 4. Add Noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Predict Noise
        # Pad from 5x5 to 8x8 for U-Net stability
        # pad (left, right, top, bottom) => (0, 3, 0, 3) to go 5->8
        noisy_latents_padded = F.pad(noisy_latents, (0, 3, 0, 3))
        
        # UNet forward on 8x8
        pred_padded = self.unet(noisy_latents_padded, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Crop back to 5x5
        # [B, C, 8, 8] -> [B, C, 5, 5]
        model_pred = pred_padded[:, :, :5, :5]
        
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
        hidden = self.vae.encoder(state_image)
        sampler = self.vae.bottleneck(hidden)
        init_latents = sampler.mean * 0.18215 # Use mean for deterministic starting point
        
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
            # Pad latents 5x5 -> 8x8 for U-Net
            latents_padded = F.pad(latents, (0, 3, 0, 3))
            
            # Expand for classifier free guidance (on padded latents)
            latent_model_input = torch.cat([latents_padded] * 2)
            latent_model_input = self.inference_scheduler.scale_model_input(latent_model_input, t)
            
            # Duplicate text embeddings for CFG (simulated, ideally use uncond)
            text_embeddings_input = torch.cat([target_text_embeddings] * 2)
            
            # Predict noise
            noise_pred_padded = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_input).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred_padded.chunk(2)
            noise_pred_padded = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Crop predicted noise to 5x5 to step the scheduler?
            # Scheduler expects noise of same shape as latents.
            noise_pred = noise_pred_padded[:, :, :5, :5]
            
            # Step
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Decode
        latents = latents / 0.18215
        return latents
