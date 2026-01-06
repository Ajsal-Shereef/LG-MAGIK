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
from torch.optim.swa_utils import AveragedModel

# --- Helper: EMA Update Function ---
def get_ema_avg_fn(decay=0.999):
    def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
        return decay * averaged_model_parameter + (1 - decay) * model_parameter
    return ema_avg

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
    
    # --- EMA Setup ---
    use_ema = cfg.model.get("use_ema", False)
    ema_model = None
    if use_ema:
        accelerator.print("Initializing EMA model...")
        ema_avg_fn = get_ema_avg_fn(decay=cfg.model.get("ema_decay", 0.999))
        ema_model = AveragedModel(vae, avg_fn=ema_avg_fn)
        # We don't prepare EMA model with accelerator as it's just a shadow copy
    
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
                    
                    # Gradient Clipping (Only for generator/VAE params)
                    if cfg.model.get("max_grad_norm", None):
                        accelerator.clip_grad_norm_(vae.parameters(), cfg.model.max_grad_norm)
                    
                    # Update EMA (Only when generator updates)
                    if use_ema:
                        ema_model.update_parameters(vae)
                
                # Reduce all loss components across processes and convert to scalar
                reduced_losses = {
                    key: accelerator.reduce(value.detach(), reduction="mean").item()
                    for key, value in losses.items()
                }

                # Update the running totals for the epoch
                for key, value in reduced_losses.items():
                    epoch_losses[key] += value

                # Step the scheduler (Moved to per-batch for OneCycleLR)
                vae.step_schedulers()

                # --- DYNAMIC LOGGING ---
                if accelerator.is_main_process and log_values_and_images:
                    # Create the log payload, including dynamic losses and static values
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
                        # Use EMA model for generation if enabled
                        eval_model = ema_model if use_ema else vae
                        # Need to put EMA model in eval mode and maybe move to device if not handled
                        # AveragedModel keeps params on same device as source usually, but let's be safe
                        
                        num_images_to_log = min(batch["pixel_values"].shape[0], 8)
                        
                        # For reconstruction logging, we can just use the current batch output (from main model)
                        # or run a forward pass with EMA model. Let's stick to main model for training progress,
                        # and maybe use EMA for generation.
                        
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
                                # Use eval_model (EMA or regular) for generation
                                # Note: AveragedModel wraps the module, so we access it via .module if needed, 
                                # but it also forwards calls. However, `generate` is a custom method on VAE.
                                # AveragedModel doesn't automatically forward custom methods unless we subclass.
                                # So we need to call eval_model.module.generate if it's an AveragedModel.
                                generator = eval_model.module if use_ema else eval_model
                                
                                # Ensure generator is in eval mode
                                was_training = generator.training
                                generator.eval()
                                
                                with torch.no_grad():
                                    # We need 'output' for the latents. If using EMA, we should probably re-run forward
                                    # to get consistent latents, or just use the batch.
                                    # The `generate` method takes `output` dict.
                                    # Let's re-run forward with generator to get consistent state
                                    gen_output = generator(batch)
                                    
                                    generated_images = generator.generate(gen_output, cfg.training.num_images_to_generate, accelerator.device, *validation_prompts)
                                
                                if was_training:
                                    generator.train()

                                if args.models.model.observation_mode == "image":
                                    tracker.log({"Generated": wandb.Image(generated_images)}, step=global_step)
                global_step += 1
        if epoch % cfg.training.save_weight_freequency == 0:
            vae.save(f"{save_dir}/", save_name=f"{cfg.project_name}")       
            if use_ema:
                # Save EMA model too
                # Access underlying module
                ema_model.module.save(f"{save_dir}/", save_name=f"{cfg.project_name}_ema")

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
        
        if use_ema:
            # Save EMA model
            # AveragedModel -> module -> save
            ema_model.module.save(f"{save_dir}/", save_name=f"{cfg.model_name}_ema")
            accelerator.print(f"EMA VAE model saved at: {save_dir}/{cfg.model_name}_ema")

    # Conditionally end training
    if log_values_and_images:
        accelerator.end_training()

@hydra.main(version_base=None, config_path="config", config_name="train_vae")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()