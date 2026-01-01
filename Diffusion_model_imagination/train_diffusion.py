import os
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import DictConfig, OmegaConf
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
import wandb

import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from architectures.common_utils import get_dataloader, create_dump_directory, calculate_vae_scaling_factor
from torchvision.utils import make_grid
from hydra.utils import instantiate
from Diffusion_model_imagination.models.diffusion import DiffusionImaginationModel
from transformers import CLIPTokenizer, CLIPTextModel

@hydra.main(version_base=None, config_path="../config", config_name="train_diffusion")
def main(cfg: DictConfig):
    # 1. Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.models.accelerator.project_dir, 
        logging_dir=cfg.models.accelerator.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.models.accelerator.gradient_accumulation_steps,
        mixed_precision=cfg.models.accelerator.mixed_precision,
        log_with=cfg.models.accelerator.log_with,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.models.training.experiment_name, config=OmegaConf.to_container(cfg, resolve=True))
        
    set_seed(cfg.models.training.seed)
    
    # 2. Load Data
    accelerator.print("Loading dataset...")
    dataloader = get_dataloader(cfg) # Using existing dataloader
    
    # 3. Initialize VAE (Frozen)
    accelerator.print(f"Loading VAE from {cfg.models.vae_path}")
    vae = instantiate(cfg.models.vae.model)
    vae.load_params(cfg.models.vae_path)
    vae.eval()
    vae.requires_grad_(False)
    
    # Calculate Scaling Factor (Main Process only, then broadcast? Or just calculate on all if fast enough)
    # Using 500 samples (user asked for 5000, but let's stick to what's reasonable interactively unless forced. 
    # User asked for "5000 images". Okay.)
    # Note: If multi-gpu, this might be redundant work on each proc, but it's safe.
    vae_scaling_factor = 1.0
    # Actually, we should probably just run this on main process and broadcast, OR just run on all. 
    # Running on all ensures everyone has same value if deterministic. 
    # But let's just run it. 
    accelerator.print("Calculating VAE scaling factor...")
    # Create a temporary dataloader for this? Or reuse `dataloader`.
    # `calculate_vae_scaling_factor` handles iterator.
    vae_scaling_factor = calculate_vae_scaling_factor(vae, dataloader, num_images=5000, device=accelerator.device)
    accelerator.print(f"Using VAE Scaling Factor: {vae_scaling_factor}")

    # Load Text Encoder and Tokenizer explicitly
    accelerator.print(f"Loading Text Encoder from {cfg.models.data.text_encoder_path}")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.models.data.text_encoder_path)
    text_encoder = CLIPTextModel.from_pretrained(cfg.models.data.text_encoder_path)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    
    # 4. Initialize Diffusion Model
    accelerator.print("Initializing Diffusion Model...")
    
    # Get latent channel from VAE
    latent_channel = 4
    if hasattr(vae, 'config') and hasattr(vae.config, 'latent_channels'):
        latent_channel = vae.config.latent_channels
    elif hasattr(vae, 'latent_channel'):
        latent_channel = vae.latent_channel
    
    accelerator.print(f"Detected VAE latent channel: {latent_channel}")
    
    # Construct config for 10x10 latents (padded to 16x16)
    # We pad to 16x16 for U-Net stability (16->8->4 through 2 downsamples).
    unet_config = {
        "sample_size": 16, 
        "in_channels": latent_channel,
        "out_channels": latent_channel,
        "layers_per_block": 1,
        "block_out_channels": (128, 256), # Only 2 levels
        "down_block_types": (
            "CrossAttnDownBlock2D", # 16 -> 8
            "DownBlock2D",          # 8 -> 4
        ),
        "up_block_types": (
            "UpBlock2D",            # 4 -> 8
            "CrossAttnUpBlock2D",   # 8 -> 16
        ),
        "cross_attention_dim": text_encoder.config.hidden_size, # Standard CLIP embedding dim
        "attention_head_dim": 8,
    }

    # Instantiate custom wrapper
    model = DiffusionImaginationModel(
        vae, 
        unet_config=unet_config,
        num_train_timesteps=cfg.models.training.num_train_timesteps,
        scaling_factor=vae_scaling_factor
    )
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.unet.parameters(),
        lr=cfg.models.optimizer.lr,
        betas=(cfg.models.optimizer.beta1, cfg.models.optimizer.beta2),
        weight_decay=cfg.models.optimizer.weight_decay,
        eps=cfg.models.optimizer.eps,
    )
    
    # 5.1 Scheduler
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        cfg.models.optimizer.scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=cfg.models.optimizer.scheduler.warmup_steps,
        num_training_steps=cfg.models.training.num_epochs * len(dataloader),
    )
    
    # 6. Prepare
    model.unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(model.unet, optimizer, dataloader, lr_scheduler)
    # VAE and Text Encoder are kept on device but not prepared for training
    model.vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # 7. Training Loop
    num_epochs = cfg.models.training.num_epochs
    global_step = 0
    
    # Create save directory
    save_dir = create_dump_directory(os.path.join(cfg.save_path, cfg.models.model_name, cfg.env.name))
    
    # Track top-k checkpoints
    saved_checkpoints = []
    
    for epoch in range(num_epochs):
        model.unet.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model.unet):
                # Unpack batch
                images = batch["pixel_values"]
                input_ids = batch["input_ids"] # Text tokens for caption
                attention_mask = batch["attention_masks"]

                # Unconditional Dropout for Classifier Free Guidance
                if hasattr(cfg.models.training, 'uncond_prob') and cfg.models.training.uncond_prob > 0:
                    random_prob = torch.rand(input_ids.shape[0], device=input_ids.device)
                    # Create mask where we drop the text
                    drop_mask = random_prob < cfg.models.training.uncond_prob
                    
                    if drop_mask.any():
                        # We need to replace these with empty string tokens.
                        # Let's create a reference empty batch
                        with torch.no_grad():
                             empty_tokens = tokenizer(
                                 [""] * drop_mask.sum().item(),
                                 max_length=cfg.models.model.max_sequence_length,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt"
                             )
                             # Replace in batch
                             # Ensure device compatibility
                             empty_ids = empty_tokens.input_ids.to(input_ids.device)
                             empty_mask = empty_tokens.attention_mask.to(attention_mask.device)
                             
                             # Match dtypes
                             empty_ids = empty_ids.to(dtype=input_ids.dtype)
                             empty_mask = empty_mask.to(dtype=attention_mask.dtype)
                             
                             input_ids[drop_mask] = empty_ids
                             attention_mask[drop_mask] = empty_mask
                
                with torch.no_grad():
                    # Get Text Embeddings using standalone Text Encoder
                    encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]
                
                # Forward pass (VAE encoding happens inside model forward)
                model_pred, noise = model(images, encoder_hidden_states)
                
                # Loss
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logs
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                   accelerator.log({
                       "loss": loss.item(),
                       "lr": lr_scheduler.get_last_lr()[0],
                       "epoch": epoch,
                   }, step=global_step)
                   
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            # Save checkpoint
            if global_step % cfg.models.training.save_interval == 0 and accelerator.is_main_process:
                 checkpoint_name = f"checkpoint-{global_step}"
                 save_path = os.path.join(save_dir, checkpoint_name)
                 os.makedirs(save_path, exist_ok=True)
                 unwrapped_unet = accelerator.unwrap_model(model.unet)
                 unwrapped_unet.save_pretrained(save_path)
                 
                 saved_checkpoints.append(save_path)
                 if len(saved_checkpoints) > cfg.models.training.save_top_k:
                     oldest_ckpt = saved_checkpoints.pop(0)
                     import shutil
                     if os.path.exists(oldest_ckpt):
                         shutil.rmtree(oldest_ckpt)
                         accelerator.print(f"Removed old checkpoint: {oldest_ckpt}")

            # Log Generation
            if global_step % cfg.models.training.log_media_interval == 0 and accelerator.is_main_process:
                model.unet.eval()
                with torch.no_grad():
                     # Use the current batch for visualization to see reconstruction vs imagination
                     # Grid: 10 rows (images), 7 cols (1 recon + 6 imaginations)
                     
                     validation_prompts = cfg.env.get("validation_prompts", [])
                     num_vis = 10
                     num_variations = len(validation_prompts)
                     
                     # Get batch slice
                     vis_images = batch["pixel_values"][:num_vis].to(accelerator.device)
                     # vis_input_ids = batch["input_ids"][:num_vis].to(accelerator.device)
                     # vis_attn_mask = batch["attention_masks"][:num_vis].to(accelerator.device)
                     current_bsz = vis_images.shape[0]
                     
                     if current_bsz == 0:
                         accelerator.print("Batch empty, skipping visualization.")
                     else:
                         # 1. Reconstruction (Column 1)
                         # Use VAE forward pass (handles encoding -> decoding)
                         # Standard VAE forward does not take dict anymore if we use AutoencoderKL style
                         # But let's check VAE.py wrapper. It takes "x".
                         # If we use VAE.py wrapper:
                         # It expects dict with "pixel_values"
                         mini_batch = {
                             "pixel_values": vis_images,
                         }
                         # VAE.forward returns dict.
                         vae_out = model.vae(mini_batch)
                         recon_images = vae_out["reconstructed_x"]
                         
                         # Clamp and Normalize for display [0,1]
                         recon_images = (recon_images * 0.5 + 0.5).clamp(0, 1)
                         
                         # 2. Imagination (Columns 2-7)
                         imagination_cols = []
                         for i in range(num_variations):
                             prompt = str(validation_prompts[i])
                             
                             # Tokenize prompt (repeated for batch size)
                             tokens = tokenizer(
                                 [prompt], 
                                 max_length=cfg.models.model.max_sequence_length, 
                                 padding="max_length", 
                                 truncation=True, 
                                 return_tensors="pt"
                             )
                             
                             # Repeat for batch size [1, Seq] -> [B, Seq]
                             p_input_ids = tokens.input_ids.to(accelerator.device).repeat(current_bsz, 1)
                             p_attn_mask = tokens.attention_mask.to(accelerator.device).repeat(current_bsz, 1)
                             
                             # Get embedding
                             txt_embeds = text_encoder(p_input_ids, attention_mask=p_attn_mask)[0]

                             # Unconditional embeddings (empty string)
                             uncond_tokens = tokenizer(
                                 [""], 
                                 max_length=cfg.models.model.max_sequence_length, 
                                 padding="max_length", 
                                 truncation=True, 
                                 return_tensors="pt"
                             )
                             u_input_ids = uncond_tokens.input_ids.to(accelerator.device).repeat(current_bsz, 1)
                             u_attn_mask = uncond_tokens.attention_mask.to(accelerator.device).repeat(current_bsz, 1)
                             uncond_embeds = text_encoder(u_input_ids, attention_mask=u_attn_mask)[0]

                             # Img2Img Generation
                             # Use vis_images as state_image
                             # Strength < 1.0 to preserve structure (e.g. 0.75)
                             img_latents = model.imagine(
                                 state_image=vis_images,
                                 target_text_embeddings=txt_embeds,
                                 uncond_text_embeddings=uncond_embeds,
                                 strength=cfg.models.training.get("imagination_strength", 0.75), 
                                 num_inference_steps=cfg.models.training.num_inference_steps
                             )
                             
                             # Decode (Normal VAE decode - no text conditioning)
                             if hasattr(model.vae, 'decode'):
                                 # Standard AutoencoderKL decode
                                 gen_images = model.vae.decode(img_latents).sample
                             else:
                                 # Fallback? VAE.py should inherit decode.
                                 gen_images = model.vae.decoder(img_latents) # If it was TextConditioned, but we are switching.
                             
                             gen_images = (gen_images * 0.5 + 0.5).clamp(0, 1)
                             imagination_cols.append(gen_images)
                         
                         # 3. Assemble Grid
                         rows = []
                         for i in range(current_bsz):
                             orig_img = (vis_images[i] * 0.5 + 0.5).clamp(0, 1)
                             row_imgs = [orig_img, recon_images[i]] # Col 1: Orig, Col 2: Recon
                             for col in imagination_cols:
                                 row_imgs.append(col[i]) # Col 3-8
                             
                             # Concat horizontally
                             rows.append(torch.cat(row_imgs, dim=2)) # Dim 2 is Width (C,H,W)
                             
                         # Concat vertically
                         full_grid = torch.cat(rows, dim=1) # Dim 1 is Height
                         
                         # Log
                         accelerator.log({"Validation Generation (Orig + Recon + Imagination)": wandb.Image(full_grid)}, step=global_step)
                         
                model.unet.train()

    accelerator.end_training()

if __name__ == "__main__":
    main()
