import os
import random
import hydra
import torch
import numpy as np
import torch.nn as nn
import wandb
import torch.nn.functional as F
from hydra.utils import instantiate
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision.utils import make_grid
from utils import seed_everything
from architectures.common_utils import get_dataloader

# --- Helper: KL Annealing ---
def get_kl_weight(step, total_steps, cfg_anneal, max_kl_weight):
    if not cfg_anneal.get("enable", False):
        return max_kl_weight
    
    start = cfg_anneal.get("start", 0.0)
    stop = cfg_anneal.get("stop", max_kl_weight)
    n_cycles = cfg_anneal.get("n_cycles", 1)
    ratio = cfg_anneal.get("ratio", 0.5)
    
    cycle_len = total_steps // n_cycles
    cycle_step = step % cycle_len
    
    if cycle_step < cycle_len * ratio:
        # Linear increase
        return start + (stop - start) * (cycle_step / (cycle_len * ratio))
    else:
        # Constant at stop value
        return stop

def train(args: DictConfig) -> None:
    """
    Main training function for the Variational Autoencoder (VAE).

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    cfg = args.models
    # --- 1. Initialization and Setup ---
    # Check if text discriminator ablation is enabled
    use_text_discriminator = cfg.model.get("use_text_discriminator", args.get("use_text_discriminator", True))
    if not use_text_discriminator:
        cfg.model_name = f"{cfg.model_name}_no_text_disc"

    seed = getattr(args, "seed", None)
    if seed is None and hasattr(cfg, "training") and cfg.training is not None:
        seed = cfg.training.get("seed", None)
    if seed is None and hasattr(cfg, "seed"):
        seed = cfg.get("seed", None)

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        print(f"[INFO] Seed was None. Sampled random seed: {seed}")
    else:
        seed = int(seed)
        print(f"[INFO] Using provided random seed: {seed}")

    if hasattr(cfg, "training") and cfg.training is not None:
        cfg.training.seed = seed
    if hasattr(cfg, "seed"):
        cfg.seed = seed
    if hasattr(args, "seed"):
        args.seed = seed

    seed_everything(seed)

    # Creating the directory to save the model weights and configs: model_weights/{env_name}/{model_name}/seed_{seed}
    seed_name = f"seed_{seed}" if not str(seed).startswith("seed_") else str(seed)
    save_dir = os.path.join(args.save_path, args.env.name, cfg.model_name, seed_name)
    os.makedirs(save_dir, exist_ok=True)

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
    set_seed(seed, device_specific=True)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    wandb_project = cfg.training.get("experiment_name", cfg.get("project_name", "LG_MAGIK_VAE_TRAINING"))
    run_name = f"{cfg.model_name}_{args.env.name}_{seed_name}"

    if not use_text_discriminator:
        accelerator.print(f"[ABLATION] Training WITHOUT text discriminator! Model: {cfg.model_name}, Project: {wandb_project}, Run name: {run_name}")
    
    # Conditionally initialize trackers
    if accelerator.is_main_process and log_values_and_images:
        tracker_config = {log_with: {"name": run_name}}
        accelerator.print(f"[TRACKER] Initializing tracker -> Project: '{wandb_project}', Run name: '{run_name}'")
        accelerator.init_trackers(wandb_project, config=OmegaConf.to_container(args, resolve=True), init_kwargs=tracker_config)

    # --- 2. Load Data ---
    accelerator.print("Loading dataset...")
    dataloader = get_dataloader(args)
    accelerator.print("Save dir: ", save_dir)
    config_path = os.path.join(save_dir, "config.yaml")
    OmegaConf.save(config=args, f=config_path)
    
    # --- 3. Define Model ---
    accelerator.print("Initializing VAE model...")
    vae = instantiate(cfg.model)
    if cfg.training.get("is_model_fine_tune", False):
        vae.load_params(cfg.test.model_dir)
   
    # --- 5. Prepare for Distributed Training ---
    vae, dataloader = accelerator.prepare(vae, dataloader)
   
    # --- 4. Define Optimizer within model ---
    # Auto-calculate total_steps for OneCycleLR
    if cfg.optimizer.get("scheduler") and cfg.optimizer.scheduler.get("type") == "one_cycle":
        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * cfg.training.num_epochs
        cfg.optimizer.scheduler.total_steps = total_steps
        accelerator.print(f"Auto-configured OneCycleLR total_steps to {total_steps} ({steps_per_epoch} steps/epoch * {cfg.training.num_epochs} epochs)")

    vae.set_optimizers(cfg.optimizer)
    
    # --- Prepare Optimizers/Schedulers with Accelerator ---
    _vae = accelerator.unwrap_model(vae)
    _vae.vae_optim = accelerator.prepare(_vae.vae_optim)
    if _vae.caption_disc_optim is not None:
        _vae.caption_disc_optim = accelerator.prepare(_vae.caption_disc_optim)
    if _vae.scheduler is not None:
        _vae.scheduler = accelerator.prepare(_vae.scheduler)
    if _vae.caption_disc_scheduler is not None:
        _vae.caption_disc_scheduler = accelerator.prepare(_vae.caption_disc_scheduler)
    if _vae.disc_optim is not None:
        _vae.disc_optim = accelerator.prepare(_vae.disc_optim)
        if _vae.disc_scheduler is not None:
            _vae.disc_scheduler = accelerator.prepare(_vae.disc_scheduler)
    
    # --- 6. Training Loop ---
    accelerator.print("Starting VAE training loop...")
    global_step = 0
    total_steps = len(dataloader) * cfg.training.num_epochs
    
    for epoch in range(cfg.training.num_epochs):
        vae.train()
        # Use a defaultdict to dynamically store running totals for any loss component
        epoch_losses = defaultdict(float)
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(vae):
            
                # Forward pass
                output = vae(batch)
                
                # Calculate KL weight
                current_kl_weight = get_kl_weight(
                    global_step, 
                    total_steps, 
                    cfg.training.get("kl_annealing", {}), 
                    cfg.training.get("kl_weight", 1.0)
                )
                
                # Prepare kwargs for loss function
                loss_kwargs = OmegaConf.to_container(cfg.training, resolve=True)
                loss_kwargs["kl_weight"] = current_kl_weight
                
                # Calculate losses
                losses = vae.loss_function(batch, output, **loss_kwargs)
                
                # --- Optimization Steps ---
                
                # 1. Update Discriminator (Every Step)
                vae.optimize_discriminator(losses, accelerator)
                
                # 2. Update Generator (Every critic_updates steps)
                critic_updates = cfg.training.get("critic_updates", 5)
                if global_step % critic_updates == 0:
                    vae.optimize_generator(losses, accelerator, forward_output=output, **loss_kwargs)
                
                # Step the scheduler (Moved to per-batch for OneCycleLR)
                vae.step_schedulers()

                if accelerator.is_main_process and log_values_and_images:
                    # Reduce all loss components across processes only when logging to WandB
                    reduced_losses = {
                        key: accelerator.reduce(value.detach(), reduction="mean").item()
                        for key, value in losses.items()
                    }
                    for key, value in reduced_losses.items():
                        epoch_losses[key] += value

                    log_payload = {
                        **reduced_losses,
                        **vae.get_lr(),
                        "epoch": epoch,
                        "step": global_step,
                        "kl_weight": current_kl_weight,
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
                        
                        # Generate sample images
                        if global_step % cfg.training.generate_interval == 0:
                            validation_prompts = cfg.training.get("validation_prompts", [])
                            if validation_prompts:
                                was_training = vae.training
                                vae.eval()
                                
                                with torch.no_grad():
                                    gen_output = vae(batch)
                                    generated_images = vae.generate(gen_output, cfg.training.num_images_to_generate, accelerator.device, *validation_prompts)
                                
                                if was_training:
                                    vae.train()

                                if args.models.model.observation_mode == "image":
                                    tracker.log({"Generated": wandb.Image(generated_images)}, step=global_step)
                else:
                    for key, value in losses.items():
                        epoch_losses[key] += value.detach()

                global_step += 1
        if epoch % cfg.training.save_weight_freequency == 0:
            vae.save(f"{save_dir}/", save_name=f"{cfg.project_name}")       

        # Print epoch summary
        avg_epoch_losses = {
            key: (value.item() if torch.is_tensor(value) else value) / len(dataloader)
            for key, value in epoch_losses.items()
        }
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