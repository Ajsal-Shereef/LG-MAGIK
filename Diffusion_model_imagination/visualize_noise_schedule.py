import torch
import hydra
import sys
import os
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate
from diffusers import DDPMScheduler
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from architectures.common_utils import get_dataloader, calculate_vae_scaling_factor

@hydra.main(version_base=None, config_path="../config", config_name="train_diffusion")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading dataset...")
    cfg.models.data.batch_size = 4 
    dataloader = get_dataloader(cfg)
    batch = next(iter(dataloader))
    
    # Pick the fourth image (index 3) to get a different one
    original_image = batch["pixel_values"][3:4].to(device) # [1, 3, 80, 80]
    
    # Save original image
    save_image((original_image * 0.5 + 0.5).clamp(0, 1), "original_image.png")
    print("Saved original_image.png")
    
    # 2. Scheduler
    num_train_timesteps = cfg.models.training.num_train_timesteps
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    
    # 3. Timesteps to visualize
    timesteps_to_viz = [0, 350, 400, 450, 500, 550, 600]
    noisy_images_list = []
    
    print("Generating noisy samples...")
    # Use fixed noise
    noise = torch.randn_like(original_image)
    
    for t_idx in timesteps_to_viz:
        # Create timestep tensor
        t = torch.tensor([t_idx], device=device).long()
        
        # Add noise
        if t_idx == 0:
            noisy_image = original_image
        else:
            noisy_image = noise_scheduler.add_noise(original_image, noise, t)
            
        # Denormalize
        noisy_image = (noisy_image * 0.5 + 0.5).clamp(0, 1)
        noisy_images_list.append(noisy_image.cpu())
        
    # 4. Save Grid
    # Shape: [N_steps, 3, 80, 80]
    grid_tensor = torch.cat(noisy_images_list, dim=0)
    save_path = "pixel_noise_schedule_viz.png"
    save_image(grid_tensor, save_path, nrow=len(timesteps_to_viz), padding=2)
    
    print(f"Saved visualization to {save_path}")
    print("Timesteps:", timesteps_to_viz)

if __name__ == "__main__":
    main()
