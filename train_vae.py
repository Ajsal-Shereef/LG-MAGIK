import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from diffusers import AutoencoderKL
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision.utils import make_grid
from architectures.common_utils import create_dump_directory

from architectures.common_utils import get_dataloader

def train(cfg: DictConfig) -> None:
    """
    Main training function for the Variational Autoencoder (VAE).

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    # --- 1. Initialization and Setup ---
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    # Check for the logging flag in the config. Defaults to True if not present.
    log_values_and_images = cfg.training.get("log_values_and_images", True)
    
    # Conditionally set the logger based on the flag
    log_with = cfg.accelerator.log_with if log_values_and_images else None

    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.accelerator.project_dir, 
        logging_dir=cfg.accelerator.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        mixed_precision=cfg.accelerator.mixed_precision,
        log_with=log_with, # Use the conditional logger
        project_config=accelerator_project_config,
    )
    
    # Conditionally initialize trackers
    if accelerator.is_main_process and log_values_and_images:
        accelerator.init_trackers(cfg.training.experiment_name, config=OmegaConf.to_container(cfg, resolve=True))


    # --- 2. Load Data ---
    accelerator.print("Loading dataset...")
    dataloader = get_dataloader(cfg)

    # --- 3. Define Model ---
    accelerator.print("Initializing VAE model...")
    vae = AutoencoderKL(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        down_block_types=tuple(cfg.model.down_block_types),
        up_block_types=tuple(cfg.model.up_block_types),
        block_out_channels=tuple(cfg.model.block_out_channels),
        latent_channels=cfg.model.latent_channels,
        layers_per_block=cfg.model.layers_per_block,
        norm_num_groups=cfg.model.norm_num_groups,
    )

    # --- 4. Define Optimizer ---
    optimizer = optim.AdamW(
        vae.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
    )

    # --- 5. Prepare for Distributed Training ---
    vae, optimizer, dataloader = accelerator.prepare(vae, optimizer, dataloader)
    
    # --- 6. Training Loop ---
    accelerator.print("Starting VAE training loop...")
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        vae.train()
        total_loss, total_recon_loss, total_kl_loss = 0, 0, 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(vae):
                images = batch["pixel_values"]
                
                # Forward pass
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample()
                reconstructed = vae.decode(latents).sample
                
                # Calculate losses
                recon_loss = F.mse_loss(reconstructed, images, reduction="none")
                recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1).mean()

                kl_loss = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp(), dim=[1, 2, 3]).mean()
                
                loss = recon_loss + cfg.training.kl_weight * kl_loss
                
                # Backward pass and optimization
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # Reduce / average losses across processes reliably
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                reduced_recon_loss = accelerator.reduce(recon_loss.detach(), reduction="mean").item()
                reduced_kl_loss = accelerator.reduce(kl_loss.detach(), reduction="mean").item()

                total_loss += reduced_loss
                total_recon_loss += reduced_recon_loss
                total_kl_loss += reduced_kl_loss

                # Logging
                if accelerator.is_main_process and log_values_and_images:
                    accelerator.log({
                        "loss": reduced_loss,
                        "recon_loss": reduced_recon_loss,
                        "kl_loss": reduced_kl_loss,
                        "epoch": epoch,
                        "step": global_step,
                    }, step=global_step)

                    # Log images at regular intervals based on config
                    if global_step > 0 and global_step % cfg.training.log_media_interval == 0:
                        num_images_to_log = min(images.shape[0], 8)
                        
                        img_to_log = (images[:num_images_to_log].detach() * 0.5 + 0.5).clamp(0, 1)
                        recon_to_log = (reconstructed[:num_images_to_log].detach() * 0.5 + 0.5).clamp(0, 1)

                        # Create a single grid for comparison
                        comparison_tensor = torch.cat([img_to_log, recon_to_log])
                        comparison_grid = make_grid(comparison_tensor, nrow=num_images_to_log)
                        
                        tracker = accelerator.get_tracker("wandb")
                        tracker.log({
                            "Original vs. Reconstructed": wandb.Image(comparison_grid)
                        }, step=global_step)
                
                global_step += 1

        # Print epoch summary
        avg_epoch_loss = total_loss / len(dataloader)
        avg_epoch_recon_loss = total_recon_loss / len(dataloader)
        avg_epoch_kl_loss = total_kl_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | Loss: {avg_epoch_loss:.4f} | Recon Loss: {avg_epoch_recon_loss:.4f} | KL Loss: {avg_epoch_kl_loss:.4f}")

    accelerator.wait_for_everyone()

    # --- 7. Save the trained model ---
    if accelerator.is_main_process:
        unwrapped_vae = accelerator.unwrap_model(vae)
        # Save the model to a 'vae' subdirectory for easy integration
        # with Stable Diffusion pipelines.
        pipeline_save_path = cfg.training.output_dir
        unwrapped_vae.save_pretrained(pipeline_save_path)
        accelerator.print(f"VAE model saved for pipeline integration at: {pipeline_save_path}")

    # Conditionally end training
    if log_values_and_images:
        accelerator.end_training()

@hydra.main(version_base=None, config_path="config", config_name="train_vae")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()