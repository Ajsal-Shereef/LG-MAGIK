import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from tqdm.auto import tqdm
from hydra.utils import instantiate
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from architectures.common_utils import get_dataloader

@hydra.main(version_base=None, config_path="../config", config_name="train_diffusion")
def main(cfg: DictConfig):
    accelerator = Accelerator()
    device = accelerator.device
    
    # 1. Load Data
    print("Loading dataset...")
    # Reduce batch size for safety or keep as is
    cfg.models.data.batch_size = 32
    dataloader = get_dataloader(cfg) 
    
    # 2. Initialize VAE
    print(f"Loading VAE from {cfg.models.vae_path}")
    vae = instantiate(cfg.models.vae.model)
    vae.load_params(cfg.models.vae_path)
    vae.eval()
    vae.to(device)
    vae.requires_grad_(False)
    
    print("Collecting latents...")
    all_latents = []
    num_batches = 50 # Run enough batches to get a good estimate (e.g. 32*50 = 1600 images)
    
    max_batches = min(len(dataloader), num_batches)
    
    for i, batch in enumerate(tqdm(dataloader, total=max_batches)):
        if i >= max_batches:
            break
            
        images = batch["pixel_values"].to(device)
        
        with torch.no_grad():
            # Encode
            # Check if VAE wrapper or direct model. Assuming wrapper based on train_diffusion.py
            # But train_diffusion checks for 'encode' vs 'forward'.
            # Let's try standard encode pattern first.
            if hasattr(vae, 'encode'):
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample()
            else:
                # Fallback if it's the wrapper that takes dict and returns dict
                 mini_batch = {"pixel_values": images}
                 out = vae(mini_batch)
                 # We need latents. The wrapper might not expose them directly in forward.
                 # Let's inspect the wrapper structure if this fails, but usually 
                 # we can access the underlying autoencoder.
                 # If vae is the wrapper, valid access might be vae.encode(images) if implemented.
                 pass
        
        all_latents.append(latents.cpu())
        
    all_latents = torch.cat(all_latents, dim=0)
    print(f"Collected latents shape: {all_latents.shape}")
    
    # Calculate Global Stats
    global_std = all_latents.std()
    global_mean = all_latents.mean()
    
    print(f"Global Latent Mean: {global_mean.item()}")
    print(f"Global Latent Std: {global_std.item()}")
    
    # Calculate Per-Channel Stats
    # Shape: [N, C, H, W] -> Std across [0, 2, 3]
    channel_means = all_latents.mean(dim=(0, 2, 3))
    channel_stds = all_latents.std(dim=(0, 2, 3))
    
    print("\nPer-Channel Statistics:")
    for c in range(len(channel_means)):
        print(f"  Channel {c}: Mean={channel_means[c].item():.4f}, Std={channel_stds[c].item():.4f}")
        
    print(f"\nRecommended Scaling Factor (Global): {1.0 / global_std.item()}")

if __name__ == "__main__":
    main()
