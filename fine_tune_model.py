# train_instruct_lora_full.py
"""
Full Instruct-style LoRA fine-tuning script for Stable Diffusion (v1.5 style).
Sensible defaults included for a single multi-GB GPU (A100/H100/H200). Uses accelerate.

Example run:
accelerate launch --mixed_precision=fp16 train_instruct_lora_full.py \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --train_data_dir ./paired_dataset/train \
  --val_data_dir ./paired_dataset/val \
  --output_dir ./lora_out \
  --use_wandb \
  --wandb_project "minigrid-lora" \
  --resolution 256 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 5000
"""

import os
import math
import pickle
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm
import json
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

# Optional logging
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# ---------------------------
# Tiny LoRA wrapper for Linear
# ---------------------------
class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: int = 16):
        super().__init__()
        self.orig = orig_linear
        in_dim = orig_linear.in_features
        out_dim = orig_linear.out_features
        self.r = r
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_dim)))
            self.lora_B = nn.Parameter(torch.zeros((out_dim, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0

        # freeze original
        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.orig(x)
        if self.r > 0:
            delta = (x @ self.lora_A.t())  # batch x r
            delta = delta @ self.lora_B.t()  # batch x out_dim
            return base + delta * self.scaling
        else:
            return base

def patch_unet_with_lora(unet: UNet2DConditionModel, r=4, alpha=16, target_modules=None):
    if target_modules is None:
        # typical names in UNet attention / MLPs
        target_modules = ["to_q", "to_k", "to_v", "to_out", "proj_out", "fc1", "fc2"]

    patched = []
    for name, module in list(unet.named_modules()):
        if isinstance(module, nn.Linear):
            if any(tm in name for tm in target_modules):
                parent_path = name.rsplit(".", 1)[0] if "." in name else ""
                parent = unet
                if parent_path:
                    for attr in parent_path.split("."):
                        parent = getattr(parent, attr)
                attr_name = name.split(".")[-1]
                orig = getattr(parent, attr_name)
                wrapper = LoRALinear(orig, r=r, alpha=alpha)
                setattr(parent, attr_name, wrapper)
                patched.append(name)
    return patched

class PairedEditDataset(Dataset):
    """
    Loads dataset from a pickle file.
    The pickle file contains a list of dictionaries, each with keys:
    'frame', 'change_description', 'changed_frame', and others.
    Extracts 'frame' (source), 'change_description' (instruction), and 'changed_frame' (target).
    """
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.samples = []

        # Load pickle file
        if not self.data_path.exists():
            raise ValueError(f"Pickle file not found at {self.data_path}")
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract relevant keys from each dictionary
        for idx, item in enumerate(data):
            if all(key in item for key in ["frame", "change_description", "changed_frame"]):
                self.samples.append({
                    "source": item["frame"],
                    "instruction": item["change_description"],
                    "target": item["changed_frame"],
                    "id": idx
                })
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Ensure images are in PIL Image format
        src = sample["source"] if isinstance(sample["source"], Image.Image) else Image.fromarray(sample["source"]).convert("RGB")
        tar = sample["target"] if isinstance(sample["target"], Image.Image) else Image.fromarray(sample["target"]).convert("RGB")
        ins = sample["instruction"]
        return {"source": src, "target": tar, "instruction": ins, "id": idx}

def collate_fn(samples):
    return samples

# ---------------------------
# Utils
# ---------------------------
def load_image_to_tensor(img: Image.Image, device, resolution=256):
    img = img.resize((resolution, resolution), resample=Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = 2.0 * arr - 1.0  # [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

# ---------------------------
# Main training
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="HF repo or local path with v1.5-style layout (vae/, unet/, text_encoder/, tokenizer/)")
    parser.add_argument("--train_data_dir", type=str, default="data/MiniGrid/training/paired.pkl")
    parser.add_argument("--val_data_dir", type=str, default="data/MiniGrid/validation/paired.pkl")
    parser.add_argument("--output_dir", type=str, default="./lora_out")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="lg_magik")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--device_map", type=str, default=None)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else None)
    device = accelerator.device

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models (this may take a while)...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)

    # Freeze core weights
    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False

    print("Patching UNet with LoRA...")
    patched = patch_unet_with_lora(unet, r=args.lora_r, alpha=args.lora_alpha)
    print(f"Patched {len(patched)} modules (examples: {patched[:6]})")

    # collect trainable params
    trainable = [p for n, p in unet.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable LoRA params: {total_trainable:,}")

    # Scheduler for diffusion timesteps
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # datasets / loaders
    train_dataset = PairedEditDataset(args.train_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = None
    if args.val_data_dir:
        val_dataset = PairedEditDataset(args.val_data_dir)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW([p for p in trainable], lr=args.learning_rate)

    # accelerator prepare
    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

    # logging
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args))

    global_step = 0
    pbar = tqdm(total=args.max_train_steps, desc="Steps")

    # helper to encode text
    def encode_text(prompts):
        inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = text_encoder(**inputs)
        return out.last_hidden_state

    # VAE scaling factor if available
    try:
        scaling_factor = vae.config.scaling_factor
    except Exception:
        scaling_factor = 0.18215

    # training loop
    print("Starting training loop...")
    while global_step < args.max_train_steps:
        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            bs = len(batch)
            # prepare images and prompts
            sources = [sample["source"] for sample in batch]
            targets = [sample["target"] for sample in batch]
            prompts = [sample["instruction"] for sample in batch]

            # encode text
            text_embeddings = encode_text(prompts)

            # encode images via VAE
            with torch.no_grad():
                src_t = torch.cat([load_image_to_tensor(img, device, resolution=args.resolution) for img in sources], dim=0)
                tar_t = torch.cat([load_image_to_tensor(img, device, resolution=args.resolution) for img in targets], dim=0)

                # keep images in float32 for VAE
                src_t = src_t.to(dtype=torch.float32)
                tar_t = tar_t.to(dtype=torch.float32)

                with torch.no_grad():
                    src_latents = vae.encode(src_t).latent_dist.sample().to(device, dtype=torch.float32)
                    tar_latents = vae.encode(tar_t).latent_dist.sample().to(device, dtype=torch.float32)

                # scale for diffusion
                src_latents = src_latents * scaling_factor
                tar_latents = tar_latents * scaling_factor

            # sample noise / timesteps
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=device).long()
            noise = torch.randn_like(tar_latents)
            noisy_latents = noise_scheduler.add_noise(tar_latents, noise, timesteps)

            # Model input: concat noisy target latents and source latents along channel dimension
            # Note: shapes must match [B, C, H, W]
            model_input = model_input = noisy_latents
            
            # Forward pass through UNet (only LoRA parameters require grad)
            model_pred = unet(model_input, timesteps, encoder_hidden_states=text_embeddings).sample

            loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # logging
            if accelerator.is_main_process:
                step_loss = loss.item() * args.gradient_accumulation_steps
                # tb_writer.add_scalar("train/loss", step_loss, global_step)
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({"train/loss": step_loss, "step": global_step})
                if global_step % args.log_every == 0:
                    tqdm.write(f"[Step {global_step}] loss: {step_loss:.6f}")

            global_step += 1
            pbar.update(1)

            # validation visualization
            if accelerator.is_main_process and val_loader and (global_step % args.log_every == 0 or global_step == 1):
                num_val_samples = 10  # how many validation samples to log
                val_iter = iter(val_loader)
                samples = []
            
                for _ in range(num_val_samples):
                    try:
                        val_sample = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_sample = next(val_iter)
            
                    vsrc = load_image_to_tensor(val_sample[0]["source"], device, resolution=args.resolution)
                    vtar = load_image_to_tensor(val_sample[0]["target"], device, resolution=args.resolution)
                    vinstr = val_sample[0]["instruction"]

                    # keep images in float32 for VAE
                    vsrc = vsrc.to(dtype=torch.float32)
                    vtar = vtar.to(dtype=torch.float32)
                    
                    # encode source
                    with torch.no_grad():
                        src_latents = vae.encode(vsrc * 2 - 1).latent_dist.sample() * 0.18215
            
                    # encode instruction
                    inputs = tokenizer([vinstr], padding="max_length", truncation=True, return_tensors="pt").to(accelerator.device)
                    text_embeds = text_encoder(inputs.input_ids)[0]
            
                    # add noise & run diffusion
                    noise = torch.randn_like(src_latents)
                    latents = src_latents.clone()
                    for t in torch.linspace(noise_scheduler.num_train_timesteps-1, 0, 30, dtype=torch.long).to(accelerator.device):
                        t_batch = torch.tensor([t], device=accelerator.device, dtype=torch.long)
                        with torch.no_grad():
                            noise_pred = unet(latents, t_batch, encoder_hidden_states=text_embeds).sample
                        latents = noise_scheduler.step(noise_pred, t_batch, latents).prev_sample
            
                    # decode generated image
                    with torch.no_grad():
                        gen_image = vae.decode(1 / 0.18215 * latents).sample
                        gen_image = (gen_image / 2 + 0.5).clamp(0, 1)
            
                    # convert to PIL for logging
                    src_pil = val_sample[0]["source"]
                    tgt_pil = val_sample[0]["target"]
                    gen_pil = transforms.ToPILImage()(gen_image[0].cpu())
                    samples.append((src_pil, tgt_pil, gen_pil, vinstr))
            
                # build grid row-wise: [src | tgt | gen]
                def pil_grid_triplets(triplets, sep_width=5, row_sep_height=10):
                    """
                    Build a grid with three rows:
                        - Row 1: all source images
                        - Row 2: all target images
                        - Row 3: all generated images
                    Thin white line between images in a row.
                    Thicker white line between rows.
                    No text labels.
                    """
                    n = len(triplets)               

                    # assume all images have the same size
                    img_w, img_h = triplets[0][0].size              

                    # compute total width and height
                    total_w = n * img_w + (n-1) * sep_width
                    total_h = 3 * img_h + 2 * row_sep_height                

                    new_im = Image.new("RGB", (total_w, total_h), color=(0, 0, 0))              

                    for row_idx, row_images in enumerate(zip(*[(s, t, g) for s, t, g, _ in triplets])):
                        y_offset = row_idx * (img_h + row_sep_height)
                        for col_idx, im in enumerate(row_images):
                            x_offset = col_idx * (img_w + sep_width)
                            new_im.paste(im, (x_offset, y_offset))              

                    return new_im
            
                gridimg = pil_grid_triplets(samples)
            
                # log to TensorBoard
                tb_writer.add_image("val/source_target_generated", np.array(gridimg).transpose(2, 0, 1), global_step)
            
                # log to W&B
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "val/source_target_generated": wandb.Image(
                                gridimg, 
                                caption="\n".join([f"{i+1}: {s[3]}" for i, s in enumerate(samples)])
                            ),
                            "step": global_step
                        }
                    )

            # checkpointing LoRA params
            if accelerator.is_main_process and args.checkpointing_steps and (global_step % args.checkpointing_steps == 0):
                out_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(out_dir, exist_ok=True)
                state = {n: p.detach().cpu().clone() for n, p in unet.named_parameters() if p.requires_grad}
                torch.save(state, os.path.join(out_dir, "lora_unet.pt"))
                print(f"Saved LoRA params to {out_dir}")

            if global_step >= args.max_train_steps:
                break

    # final save
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        state = {n: p.detach().cpu().clone() for n, p in unet.named_parameters() if p.requires_grad}
        torch.save(state, os.path.join(final_dir, "lora_unet.pt"))
        print(f"Training complete. Saved final LoRA to {final_dir}")
        tb_writer.close()
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

if __name__ == "__main__":
    main()
