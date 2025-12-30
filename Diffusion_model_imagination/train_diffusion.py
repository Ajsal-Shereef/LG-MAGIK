
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

from architectures.common_utils import get_dataloader, create_dump_directory
from torchvision.utils import make_grid
from hydra.utils import instantiate
from Diffusion_model_imagination.models.diffusion import DiffusionImaginationModel

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
    # We need to load the config for the VAE to instantiate it correctly
    vae_cfg_path = os.path.join(os.path.dirname(cfg.models.vae_path), "config.yaml")
    vae_args = OmegaConf.load(vae_cfg_path)
    vae = instantiate(vae_args.models.model)
    vae.load_params(cfg.models.vae_path)
    vae.eval()
    vae.requires_grad_(False)
    
    # 4. Initialize Diffusion Model
    accelerator.print("Initializing Diffusion Model...")
    
    # Get latent channel from VAE args
    latent_channel = vae_args.models.model.get("latent_channel", 4)
    accelerator.print(f"Detected VAE latent channel: {latent_channel}")
    
    # Construct config for very small 5x5 latents
    # Since 5x5 is small and odd, deep downsampling (like 3 layers) will fail or mismtach (5->2->1).
    # We will use a shallow U-Net with just 1 downsample layer (5->2) or maybe 2 (5->2->1).
    # Ideally we'd pad to 8x8, but let's try a minimal config first.
    # Construct config for very small 5x5 latents (padded to 8x8)
    # Since we pad to 8x8, we can use a slightly deeper U-Net if desired, or keep it shallow.
    # 8 -> 4 -> 2 is safe.
    unet_config = {
        "sample_size": 8, 
        "in_channels": latent_channel,
        "out_channels": latent_channel,
        "layers_per_block": 1,
        "block_out_channels": (128, 256), # Only 2 levels
        "down_block_types": (
            "CrossAttnDownBlock2D", # 8 -> 4
            "DownBlock2D",          # 4 -> 2
        ),
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "cross_attention_dim": 512,
        "attention_head_dim": 8,
    }

    # Instantiate custom wrapper
    model = DiffusionImaginationModel(vae, unet_config=unet_config)
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.unet.parameters(),
        lr=cfg.models.optimizer.lr,
        betas=(cfg.models.optimizer.beta1, cfg.models.optimizer.beta2),
        weight_decay=cfg.models.optimizer.weight_decay,
        eps=cfg.models.optimizer.eps,
    )
    
    # 6. Prepare
    model.unet, optimizer, dataloader = accelerator.prepare(model.unet, optimizer, dataloader)
    # VAE is kept on device but not prepared for training
    model.vae.to(accelerator.device)
    
    # 7. Training Loop
    num_epochs = cfg.models.training.num_epochs
    global_step = 0
    
    for epoch in range(num_epochs):
        model.unet.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model.unet):
                # Unpack batch
                images = batch["pixel_values"]
                input_ids = batch["input_ids"] # Text tokens for caption
                
                # Get Text Embeddings using VAE's text encoder (CLIP)
                # The VAE decoder has the tokenizer and text_encoder usually, or we access it via model.vae.decoder.text_encoder
                # TextConditionedDecoder in VAE/text_conditioned_vae.py has a 'clip_model'.
                # We need to verify how to get embeddings. 
                # model.vae.decoder.text_encoder(input_ids, attention_mask=...)
                
                with torch.no_grad():
                    # Encode Image -> Latent
                    hidden = model.vae.encoder(images)
                    sampler = model.vae.bottleneck(hidden)
                    latents = sampler.latent
                    
                    # Encode Text
                    attention_mask = batch["attention_masks"]
                    # We utilize the VAE's text encoder to get embeddings
                    # The VAE decoder usually holds the text encoder.
                    encoder_hidden_states = model.vae.decoder.text_encoder(input_ids, attention_mask=attention_mask)[0]
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                
                # Add noise
                noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict
                noise_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                
                # Loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logs
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                   accelerator.log({"loss": loss.item()}, step=global_step)
                   
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            # Save checkpoint
            if global_step % cfg.models.training.save_interval == 0 and accelerator.is_main_process:
                 save_path = os.path.join(cfg.save_path, f"checkpoint-{global_step}")
                 os.makedirs(save_path, exist_ok=True)
                 unwrapped_unet = accelerator.unwrap_model(model.unet)
                 unwrapped_unet.save_pretrained(save_path)

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
                     vis_input_ids = batch["input_ids"][:num_vis].to(accelerator.device)
                     vis_attn_mask = batch["attention_masks"][:num_vis].to(accelerator.device)
                     current_bsz = vis_images.shape[0]
                     
                     if current_bsz == 0:
                         accelerator.print("Batch empty, skipping visualization.")
                     else:
                         # 1. Reconstruction (Column 1)
                         # Use VAE forward pass (handles encoding -> bottleneck -> decoding)
                         # We need to construct a mini-batch dict for the VAE forward method
                         mini_batch = {
                             "pixel_values": vis_images,
                             "input_ids": vis_input_ids,
                             "attention_masks": vis_attn_mask
                         }
                         vae_out = model.vae(mini_batch)
                         recon_images = vae_out["reconstructed_x"]
                         
                         # Clamp and Normalize for display [0,1]
                         recon_images = (recon_images * 0.5 + 0.5).clamp(0, 1)
                         
                         # 2. Imagination (Columns 2-7)
                         # Use Validation Prompts for variations
                         # We need to Tokenize the validation prompts first
                         # Access tokenizer from VAE decoder
                         tokenizer = model.vae.decoder.tokenizer
                         
                         val_input_ids_all, val_attn_mask_all = [], []
                         if num_variations > 0:
                             # Batch tokenize all validation prompts
                             # But we iterate one by one
                             pass # We'll do it inside the loop for simplicity or pre-process

                         imagination_cols = []
                         for i in range(num_variations):
                             prompt = str(validation_prompts[i])
                             
                             # Tokenize prompt (repeated for batch size)
                             # We can use the helper tokenize_captions from common_utils ONLY IF we import it, 
                             # or just use tokenizer directly. tokenizer is available.
                             
                             # Tokenize SINGLE prompt
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
                             out = model.vae.decoder.text_encoder(p_input_ids, attention_mask=p_attn_mask, return_dict=True)
                             if hasattr(out, 'last_hidden_state'):
                                 txt_embeds = out.last_hidden_state
                             else:
                                 txt_embeds = out[0]

                             # Img2Img Generation
                             # Use vis_images as state_image
                             # Strength < 1.0 to preserve structure (e.g. 0.75)
                             img_latents = model.imagine(
                                 state_image=vis_images,
                                 target_text_embeddings=txt_embeds,
                                 strength=0.75, 
                                 num_inference_steps=50
                             )
                             
                             # Decode (Conditioned on the NEW prompt)
                             gen_images = model.vae.decoder(img_latents, p_input_ids, p_attn_mask)
                             gen_images = (gen_images * 0.5 + 0.5).clamp(0, 1)
                             imagination_cols.append(gen_images)
                         
                         # 3. Assemble Grid
                         # Rows: [Recon, Gen1, Gen2, ..., Gen6]
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
