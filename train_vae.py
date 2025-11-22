import os
import hydra
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from hydra.utils import instantiate
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision.utils import make_grid
from architectures.common_utils import get_dataloader, create_dump_directory

def train(args: DictConfig) -> None:
    """
    Main training function for the Variational Autoencoder (VAE).

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    cfg = args.models
    # Creating the directory to save the model weights and configs. Placed at the top to generate different dir name before seeding
    save_dir = create_dump_directory(os.path.join(args.save_path, cfg.model_name, args.env.name))
    # --- 1. Initialization and Setup ---
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    # Check for the logging flag in the config. Defaults to True if not present.
    log_values_and_images = cfg.training.get("log_values_and_images", True)
    
    # Conditionally set the logger based on the flag. 
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
        tracker_config = {log_with : {"name":f"{cfg.model_name}_{args.env.name}"}}
        accelerator.init_trackers(cfg.training.experiment_name, config=OmegaConf.to_container(args, resolve=True), init_kwargs=tracker_config)

    # --- 2. Load Data ---
    accelerator.print("Loading dataset...")
    dataloader = get_dataloader(args)
    accelerator.print("Save dir: ", save_dir)
    config_path = os.path.join(save_dir, "config.yaml")
    OmegaConf.save(config=args, f=config_path)
    
    # --- 3. Define Model ---
    accelerator.print("Initializing VAE model...")
    vae = instantiate(cfg.model)
   
    # --- 5. Prepare for Distributed Training ---
    vae, dataloader = accelerator.prepare(vae, dataloader)
   
    # --- 4. Define Optimizer within model ---
    vae.set_optimizers(cfg.optimizer)
    
    # --- 6. Training Loop ---
    accelerator.print("Starting VAE training loop...")
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        vae.train()
        # Use a defaultdict to dynamically store running totals for any loss component
        epoch_losses = defaultdict(float)
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(vae):
            
                # Forward pass
                output = vae(batch)
                
                # Calculate losses
                losses = vae.loss_function(batch, output, **cfg.training)
                
                # Backward pass and optimization
                vae.optimize(losses, accelerator)
                
                # Reduce all loss components across processes and convert to scalar
                reduced_losses = {
                    key: accelerator.reduce(value.detach(), reduction="mean").item()
                    for key, value in losses.items()
                }

                # Update the running totals for the epoch
                for key, value in reduced_losses.items():
                    epoch_losses[key] += value

                # --- DYNAMIC LOGGING ---
                if accelerator.is_main_process and log_values_and_images:
                    # Create the log payload, including dynamic losses and static values
                    log_payload = {
                        **reduced_losses,
                        "epoch": epoch,
                        "step": global_step,
                    }
                    accelerator.log(log_payload, step=global_step)

                    # Log images at regular intervals based on config
                    if global_step > 0 and global_step % cfg.training.log_media_interval == 0:
                        num_images_to_log = min(batch["pixel_values"].shape[0], 8)
                        
                        img_to_log = (batch["pixel_values"][:num_images_to_log].detach() * 0.5 + 0.5).clamp(0, 1)
                        recon_to_log = (output["reconstructed_x"][:num_images_to_log].detach() * 0.5 + 0.5).clamp(0, 1)

                        # Create a single grid for comparison
                        if not cfg.model.get("use_weighted_recon", False):
                            comparison_tensor = torch.cat([img_to_log, recon_to_log])
                        else:
                            text_aligned_to_log = (output["text_aligned_reconstructed_x"][:num_images_to_log].detach() * 0.5 + 0.5).clamp(0, 1)
                            text_agnostic_to_log = (output["text_agnostic_reconstructed_x"][:num_images_to_log].detach() * 0.5 + 0.5).clamp(0, 1)
                            comparison_tensor = torch.cat([img_to_log, recon_to_log, text_aligned_to_log, text_agnostic_to_log])
                            
                        comparison_grid = make_grid(comparison_tensor, nrow=num_images_to_log)
                        
                        tracker = accelerator.get_tracker("wandb")
                        tracker.log({
                            "Original vs. Reconstructed": wandb.Image(comparison_grid)
                        }, step=global_step)
                        
                        #Generate sample images
                        if global_step % cfg.training.generate_interval == 0:
                            validation_prompts = cfg.training.get("validation_prompts", [])
                            if validation_prompts:
                                generated_images = vae.generate(output, cfg.training.num_images_to_generate, accelerator.device, *validation_prompts)
                                if args.models.model.observation_mode == "image":
                                    tracker.log({"Generated": wandb.Image(generated_images)}, step=global_step)
                global_step += 1
        if epoch % cfg.training.save_weight_freequency == 0:
            vae.save(f"{save_dir}/", save_name=f"{cfg.project_name}")       
        

        # Print epoch summary
        avg_epoch_losses = {key: value / len(dataloader) for key, value in epoch_losses.items()}
        # Create a dynamic string for printing the epoch summary
        loss_summary_str = " | ".join([f"{key}: {value:.4f}" for key, value in avg_epoch_losses.items()])
        accelerator.print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | {loss_summary_str}")

    accelerator.wait_for_everyone()

    # --- 7. Save the trained model ---
    if accelerator.is_main_process:
        unwrapped_vae = accelerator.unwrap_model(vae)
        pipeline_save_path = f"{save_dir}/{cfg.model_name}"
        unwrapped_vae.save(f"{save_dir}/", save_name=f"{cfg.model_name}")
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