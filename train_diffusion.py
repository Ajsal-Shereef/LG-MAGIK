import os
import hydra
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.optimization import get_scheduler  # Added for LR scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from hydra.utils import instantiate
from torchvision.utils import make_grid
from architectures.common_utils import get_dataloader  # This now imports the unified dataloader

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    for the given timesteps using the provided noise scheduler.

    Args:
        noise_scheduler (`NoiseScheduler`):
            An object containing the noise schedule parameters, specifically `alphas_cumprod`, which is used to compute
            the SNR values.
        timesteps (`torch.Tensor`):
            A tensor of timesteps for which the SNR is computed.

    Returns:
        `torch.Tensor`: A tensor containing the computed SNR values for each timestep.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def train(cfg: DictConfig) -> None:
    """Main training function for the Diffusion model (UNet)."""
    # --- 1. Initialization and Setup ---
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.accelerator.project_dir,
        logging_dir=cfg.accelerator.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        mixed_precision=cfg.accelerator.mixed_precision,
        log_with=cfg.accelerator.log_with if cfg.training.log_values_and_images else None,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process and cfg.training.log_values_and_images:
        accelerator.init_trackers(cfg.training.experiment_name, config=OmegaConf.to_container(cfg, resolve=True))

    # --- 2. Load Models and Tokenizer (all frozen except UNet) ---
    accelerator.print("Loading models and tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.text_encoder_path)
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.text_encoder_path)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae_path)

    # Freeze VAE and Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # --- 3. Load Data ---
    accelerator.print("Loading dataset...")
    # The dataloader now only loads raw images and text captions
    dataloader = get_dataloader(cfg, tokenizer)

    # --- 4. Define Diffusion Model (UNet) and Scheduler ---
    accelerator.print("Initializing UNet and Noise Scheduler...")
    unet = instantiate(cfg.model.unet)
    noise_scheduler = instantiate(cfg.noise_scheduler)

    # --- 5. Define Optimizer ---
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
    )

    # --- 6. Prepare for Distributed Training ---
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # --- Added: Define LR Scheduler with Warmup ---
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = cfg.training.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    # --- 7. Training Loop ---
    accelerator.print("Starting Diffusion training loop...")
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        unet.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample() * cfg.model.vae_scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                
                # --- Loss with optional SNR gamma weighting ---
                if cfg.training.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, cfg.training.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                # Backward pass
                accelerator.backward(loss)
                if not cfg.training.max_grad_norm:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad()

                # --- Logging and Validation ---
                avg_loss = accelerator.gather(loss.repeat(cfg.data.batch_size)).mean().item()
                total_loss += avg_loss
                
                if accelerator.is_main_process:
                    if cfg.training.log_values_and_images and global_step % cfg.training.log_metrics_interval == 0:
                        accelerator.log({"train_loss": avg_loss, "learning_rate" : lr_scheduler.get_last_lr()[0]}, step=global_step)
                    
                    if global_step > 0 and global_step % cfg.training.log_media_interval == 0 and cfg.training.log_values_and_images:
                        log_validation(vae, text_encoder, tokenizer, unet, cfg, accelerator, global_step)

                global_step += 1
        
        avg_epoch_loss = total_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | Average Loss: {avg_epoch_loss:.4f}")

    # --- 8. Save the Final Model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        pipeline_save_path = os.path.join(cfg.training.output_dir, "unet")
        unwrapped_unet.save_pretrained(pipeline_save_path)
        accelerator.print(f"Final UNet model saved to {pipeline_save_path}")

    accelerator.end_training()

def log_validation(vae, text_encoder, tokenizer, unet, cfg, accelerator, step):
    """Generates and logs validation images for text-to-image generation."""
    accelerator.print("Generating validation images...")

    # Create text-to-image pipeline with the components
    pipeline = StableDiffusionPipeline(
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        scheduler=instantiate(cfg.noise_scheduler),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed) if cfg.seed else None

    with torch.cuda.amp.autocast():
        images = pipeline(
            prompt=list(cfg.validation.prompts),
            num_inference_steps=cfg.validation.num_inference_steps,
            guidance_scale=cfg.validation.guidance_scale,
            generator=generator,
            height=112,  # Match your training resolution
            width=112,
        ).images

    # Log images to wandb
    tracker = accelerator.get_tracker("wandb")
    tracker.log(
        {"validation_images": [wandb.Image(img, caption=prompt) for img, prompt in zip(images, cfg.validation.prompts)]},
        step=step
    )
    accelerator.print("Validation images logged.")
    del pipeline
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="config", config_name="train_diffusion")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()