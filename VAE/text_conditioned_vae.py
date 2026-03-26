import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import Dict
from itertools import chain
import torch.optim as optim
from architectures.mlp import MLP
from torchvision.utils import make_grid
from architectures.common_utils import grad_reverse
from architectures.vae_utils import PatchDiscriminator, MINECritic
from architectures.cnn import CNNEncoder, CNNTextConditionedDecoder
from architectures.mlp import MLPEncoder, MLPTextConditionedDecoder
from architectures.stochastic import GaussianSampleSpatial, GaussianSample
from architectures.common_utils import tokenize_captions, get_train_transform_cnn, get_train_transform_mlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Modified TextConditionedVAE ===
class TextConditionedVAE(nn.Module):
    def __init__(self, **kwargs):
        super(TextConditionedVAE, self).__init__()
        
        # ---- Base configs ----
        self.observation_model = kwargs["observation_mode"]
        self.use_image_discriminator = kwargs.get("use_image_discriminator", False)
        self.use_caption_discriminator = kwargs.get("use_caption_discriminator", False)
        self.use_mine = kwargs.get("use_mine", False)
        disc_params = kwargs.get("disc_params", {})

        # ---- Encoder/Decoder ----
        if self.observation_model == "image":
            n_downsample = kwargs["n_downsample"]
            encoder_n_res = kwargs["encoder_n_res"]
            input_dim = kwargs["input_dim"]
            dim = kwargs["dim"]
            norm = kwargs["norm"]
            activ = kwargs["activ"]
            pad_type = kwargs["pad_type"]
            clip_model = kwargs["text_encoder"]
            encoder_final_dim = kwargs["encoder_final_dim"]
            latent_channel = kwargs["latent_channel"]
            discriminator_fc_hidden  = kwargs["discriminator_fc_hidden"]
            use_coord_conv = kwargs.get("use_coord_conv", True)
            self.is_perceptual_loss = kwargs["is_perceptual_loss"]
            self.max_sequence_length = kwargs.get("max_sequence_length", None)
            
            self.encoder = CNNEncoder(n_downsample, encoder_n_res, input_dim, dim, norm, activ, pad_type=pad_type)
            self.bottleneck = GaussianSampleSpatial(self.encoder.output_dim, latent_channel)
            self.decoder = CNNTextConditionedDecoder(n_downsample, self.encoder.output_dim, input_dim, clip_model, latent_channel, use_coord_conv=use_coord_conv)
            
            # Use max_sequence_length if provided, else fall back to tokenizer default
            # IMPORTANT: Clamp to model_max_length to avoid errors with models like CLIP (max 77)
            if self.max_sequence_length is not None:
                token_len = min(self.max_sequence_length, self.decoder.tokenizer.model_max_length)
            else:
                token_len = self.decoder.tokenizer.model_max_length
            
            self.caption_discriminator = MLP(latent_channel * np.prod(encoder_final_dim),
                                             self.decoder.text_dim,
                                             discriminator_fc_hidden)
            if self.use_image_discriminator:
                in_ch = kwargs.get("disc_in_channels", input_dim)
                base = disc_params.get("base_channels", 64)
                n_layers = disc_params.get("n_layers", 3)
                norm = disc_params.get("norm", "in")
                self.image_discriminator = PatchDiscriminator(in_channels=in_ch, base_channels=base,
                                                              n_layers=n_layers, norm=norm).to(device)
            else:
                self.image_discriminator = None
                
            self.train_transform = get_train_transform_cnn()
            if self.is_perceptual_loss:
                from architectures.common_utils import VGGLoss
                self.vgg_loss = VGGLoss(device)
        else:
            self.is_perceptual_loss = None
            input_dim = kwargs["input_dim"]
            encoder_out_dim = kwargs["encoder_output_dim"]
            hidden_dims = kwargs["encoder_hidden_dims"]
            num_resblocks = kwargs["num_resblocks"]
            clip_model = kwargs["text_encoder"]
            norm = kwargs["norm"]
            activ = kwargs["activ"]
            pad_type = kwargs["pad_type"]
            dropout = kwargs["dropout"]
            latent_dim = kwargs["latent_dim"]
            discriminator_fc_hidden  = kwargs["discriminator_fc_hidden"]
            decoder_hidden_dims = kwargs["decoder_hidden_dims"]

            self.encoder = MLPEncoder(input_dim, hidden_dims, encoder_out_dim, num_resblocks, norm, activ, dropout)
            self.bottleneck = GaussianSample(encoder_out_dim, latent_dim)
            self.decoder = MLPTextConditionedDecoder(latent_dim, input_dim, decoder_hidden_dims, clip_model)
            self.caption_discriminator = MLP(latent_dim,
                                             self.decoder.text_dim,
                                             discriminator_fc_hidden)
            self.image_discriminator = None
            self.train_transform = get_train_transform_mlp()

        # ---- Optional MINE module ----
        if self.use_mine:
            self.mine_critic = MINECritic(latent_dim if self.observation_model != "image" else latent_channel * np.prod(encoder_final_dim),
                                          self.decoder.text_dim)
        else:
            self.mine_critic = None
            
    def set_optimizers(self, parms):
        # VAE parameters (Encoder, Bottleneck, Decoder, Caption Discriminator)
        vae_params = list(self.encoder.parameters()) + list(self.bottleneck.parameters()) + list(self.decoder.parameters()) + \
                          list(self.caption_discriminator.parameters())
        
        self.vae_optim = optim.AdamW(vae_params,
                                lr=parms.lr,
                                betas=tuple(parms.betas),
                                weight_decay=parms.weight_decay,
                                eps=parms.eps,
                               )
        
        # Image Discriminator parameters (Separate Optimizer)
        if self.image_discriminator:
            self.disc_optim = optim.AdamW(self.image_discriminator.parameters(),
                                    lr=parms.lr, # Using same LR for now, could be separate
                                    betas=tuple(parms.betas),
                                    weight_decay=parms.weight_decay,
                                    eps=parms.eps,
                                   )
        else:
            self.disc_optim = None

        if self.use_mine:
            self.mine_optim = optim.AdamW(self.mine_critic.parameters(),
                                lr=parms.lr_mine,
                                betas=tuple(parms.betas),
                                weight_decay=parms.weight_decay,
                                eps=parms.eps,
                               )
        
        # Initialize schedulers
        self.scheduler = self._get_scheduler(self.vae_optim, parms.scheduler)
        if self.disc_optim:
            self.disc_scheduler = self._get_scheduler(self.disc_optim, parms.scheduler)
        else:
            self.disc_scheduler = None
            
        if self.use_mine:
            self.mine_scheduler = self._get_scheduler(self.mine_optim, parms.scheduler)
        else:
            self.mine_scheduler = None

    def _get_scheduler(self, optimizer, scheduler_params):
        if scheduler_params.type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=scheduler_params.T_max, 
                eta_min=scheduler_params.eta_min
            )
        elif scheduler_params.type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_params.step_size,
                gamma=scheduler_params.gamma
            )
        elif scheduler_params.type == "one_cycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_params.max_lr,
                total_steps=scheduler_params.total_steps,
                pct_start=scheduler_params.pct_start,
                div_factor=scheduler_params.div_factor,
                final_div_factor=scheduler_params.final_div_factor,
                anneal_strategy=scheduler_params.get("anneal_strategy", "cos")
            )
        else:
            return None

    def step_schedulers(self):
        if self.scheduler:
            self.scheduler.step()
        if self.disc_scheduler:
            self.disc_scheduler.step()
        if self.mine_scheduler:
            self.mine_scheduler.step()

    def get_lr(self):
        lrs = {"lr": self.vae_optim.param_groups[0]["lr"]}
        if self.disc_optim:
             lrs["lr_disc"] = self.disc_optim.param_groups[0]["lr"]
        if self.use_mine:
            lrs["lr_mine"] = self.mine_optim.param_groups[0]["lr"]
        return lrs
    
    def forward(self, x):
        """
        x: dict containing pixel_values, input_ids, attention_masks
        """
        images = x["pixel_values"]
        text_tokens = x["input_ids"]
        attention_mask = x["attention_masks"]
        
        hidden = self.encoder(images)
        sampler = self.bottleneck(hidden)
        latent = sampler.latent

        # Store text features from the decoder for later use (e.g., caption discriminator)
        reconstructed_x, text_feats = self.decoder(latent, text_tokens, attention_mask, return_text_feats=True)

        return {
                "x": images,
                "reconstructed_x": reconstructed_x,
                "posterior": sampler,
                "latent": latent,
                "text_feats": text_feats # Add text features to the output
                }

    def loss_function(self, batch, forward_output, **kwargs):
        # Retrieve attention masks from the original batch for pooling
        attention_mask = batch.get("attention_masks", None)
        
        original_x = batch["pixel_values"]
        reconstructed_x = forward_output["reconstructed_x"]
        posterior = forward_output["posterior"]
        text_feats = forward_output["text_feats"] # Retrieve text features

        recon_loss = F.mse_loss(reconstructed_x, original_x, reduction="none")

        if self.observation_model == "image":
            recon_loss = recon_loss.sum(dim=[1,2,3]).mean()
            kl_loss = -0.5 * torch.sum(1 + posterior.log_variance - posterior.mean.pow(2) - posterior.log_variance.exp(), dim=[1,2,3]).mean()
        else:
            recon_loss = recon_loss.sum(dim=-1).mean()
            kl_loss = -0.5 * torch.sum(1 + posterior.log_variance - posterior.mean.pow(2) - posterior.log_variance.exp(), dim=-1).mean()

        if self.is_perceptual_loss:
            perceptual_loss = self.vgg_loss(original_x, reconstructed_x)
        else:
            perceptual_loss = torch.tensor(0.0, device=original_x.device)
        
        # === Adversarial disentanglement loss ===
        if self.use_caption_discriminator:
            z_grl = grad_reverse(posterior.latent, kwargs["adv_lambda"])
            pred = self.caption_discriminator(z_grl.view(z_grl.shape[0], -1))
            
            # --- Pooled Embedding Logic ---
            if attention_mask is not None:
                # text_feats: [B, T, D]
                # attention_mask: [B, T] -> [B, T, 1]
                mask_expanded = attention_mask.unsqueeze(-1).to(text_feats.device).float()
                
                # Masked Sum
                sum_embeddings = torch.sum(text_feats.detach() * mask_expanded, dim=1)
                # Sum of mask (valid token count)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                
                pooled_text = sum_embeddings / sum_mask # [B, D]
            else:
                # Fallback if no mask provided (shouldn't happen with correct dataloader)
                pooled_text = text_feats.detach().mean(dim=1)

            # Normalize
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(pooled_text, dim=-1)

            # Contrastive Loss on POOLED embeddings
            logits = (pred_n @ tgt_n.T) / kwargs.get("temperature", 0.07)
            labels = torch.arange(logits.shape[0], device=logits.device)
            disc_loss = F.cross_entropy(logits, labels)

            vae_loss = recon_loss + perceptual_loss + kwargs["kl_weight"] * kl_loss + kwargs["adv_weight"] * disc_loss
        else:
            disc_loss = torch.tensor(0.0, device=original_x.device)
            vae_loss = recon_loss + perceptual_loss + kwargs["kl_weight"] * kl_loss
        
        # Store core VAE loss (without image GAN loss) for generator optimization
        vae_loss_core = vae_loss
        
        # === Patch discriminator loss (Refactored for GAN) ===
        img_disc_loss = torch.tensor(0.0, device=original_x.device)
        gen_adv_loss = torch.tensor(0.0, device=original_x.device)

        if self.image_discriminator is not None:
            # 1. Discriminator Loss: Hinge Loss
            # We detach reconstructed_x so gradients don't flow to generator during D update
            D_real = self.image_discriminator(original_x)
            D_fake_detached = self.image_discriminator(reconstructed_x.detach())
            
            loss_real = torch.mean(F.relu(1.0 - D_real))
            loss_fake = torch.mean(F.relu(1.0 + D_fake_detached))
            img_disc_loss = 0.5 * (loss_real + loss_fake)
            
            # 2. Generator Loss: Hinge Loss
            # We use the attached reconstructed_x here
            D_fake = self.image_discriminator(reconstructed_x)
            gen_adv_loss = -D_fake.mean()
            
            # Add generator adversarial loss to total VAE loss (for logging/legacy)
            vae_loss = vae_loss + kwargs.get("img_adv_weight", kwargs["adv_weight"]) * gen_adv_loss

        # === MINE loss ===
        if self.use_mine and self.mine_critic is not None:
            z_flat = posterior.latent.view(posterior.latent.size(0), -1)
            t_flat = text_feats.mean(dim=1) # Use retrieved text_feats
            
            # 1. Unbiased MI estimate for VAE Generator (minimize MI)
            joint = self.mine_critic(z_flat, t_flat).mean()
            shuffled_t = t_flat[torch.randperm(t_flat.size(0))]
            marginal_scores = self.mine_critic(z_flat, shuffled_t)
            
            marginal_log = torch.logsumexp(marginal_scores, dim=0) - math.log(max(marginal_scores.size(0), 1))
            
            mi_estimate = joint - marginal_log
            mi_for_vae = joint - marginal_log.detach()
            
            vae_loss = vae_loss + kwargs.get("mine_weight", 0.1) * mi_for_vae
            vae_loss_core = vae_loss_core + kwargs.get("mine_weight", 0.1) * mi_for_vae
            
            # 2. Re-compute for Critic on DETACHED inputs (maximize MI)
            z_detached = z_flat.detach()
            t_detached = t_flat.detach()
            
            joint_detached = self.mine_critic(z_detached, t_detached).mean()
            shuffled_t_detached = t_detached[torch.randperm(t_detached.size(0))]
            marginal_scores_detached = self.mine_critic(z_detached, shuffled_t_detached)
            
            marginal_log_detached = torch.logsumexp(marginal_scores_detached, dim=0) - math.log(max(marginal_scores_detached.size(0), 1))
            
            critic_loss = -(joint_detached - marginal_log_detached)
            mine_loss = mi_estimate  # The MI estimate for logging
        else:
            mine_loss = torch.tensor(0.0, device=original_x.device)
            critic_loss = torch.tensor(0.0, device=original_x.device)

        return {
            "vae_loss": vae_loss,
            "vae_loss_core": vae_loss_core,
            "recon_loss": recon_loss,
            "perceptual_loss" : perceptual_loss,
            "kl_loss": kl_loss,
            "adversarial_loss": disc_loss,
            "critic_loss": critic_loss,
            "mine_loss" : mine_loss,
            "img_disc_loss" : img_disc_loss,
            "gen_adv_loss": gen_adv_loss
        }
        
    def optimize_generator(self, losses, accelerator, forward_output=None, **kwargs):
        self.vae_optim.zero_grad()
        if self.use_mine:
            self.mine_optim.zero_grad()
            
        # Use vae_loss_core and re-compute GAN loss if needed
        total_loss = losses["vae_loss_core"]
        
        if self.image_discriminator and forward_output is not None:
            # Re-compute generator adversarial loss with updated discriminator
            reconstructed_x = forward_output["reconstructed_x"]
            D_fake = self.image_discriminator(reconstructed_x)
            gen_adv_loss = -D_fake.mean()
            total_loss = total_loss + kwargs.get("img_adv_weight", kwargs.get("adv_weight", 1.0)) * gen_adv_loss
        elif self.image_discriminator:
             # Fallback if forward_output not provided (shouldn't happen in new loop)
             # This might fail if D was updated inplace
             total_loss = losses["vae_loss"]

        accelerator.backward(total_loss)
        
        if self.use_mine:
            self.mine_optim.zero_grad()  # Clear unwanted grads from vae_loss on critic params
            accelerator.backward(losses["critic_loss"])
            
        self.vae_optim.step()
        if self.use_mine:
            self.mine_optim.step()

    def optimize_discriminator(self, losses, accelerator):
        if self.image_discriminator:
            self.disc_optim.zero_grad()
            accelerator.backward(losses["img_disc_loss"])
            self.disc_optim.step()

    def optimize(self, losses, accelerator):
        # Legacy method if needed, but we should use specific ones
        self.optimize_generator(losses, accelerator)
        self.optimize_discriminator(losses, accelerator)
        
    def imagine(self, state, description):
        train_transforms = self.train_transform
        if self.observation_model == "image":
            images = [Image.fromarray(image).convert("RGB") for image in np.expand_dims(state, axis=0)]
            states_tensors = [train_transforms(image) for image in images]
        else:
            states_tensors = [train_transforms(data) for data in np.expand_dims(state, axis=0)]
        state_tensor = torch.stack(states_tensors)
        
        tokeniser = self.decoder.tokenizer
        captions_tokenised, attention_mask = tokenize_captions(tokeniser, [description], max_length=self.max_sequence_length)
        captions_tokenised = captions_tokenised.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            hidden = self.encoder(state_tensor.to(device))
            sampler = self.bottleneck(hidden)
            mean = sampler.mean # Sampled latent
            
            # Decode conditional
            reconstructed_x, _ = self.decoder(mean, captions_tokenised, attention_mask, return_text_feats=True)
            
            # Prepare output dict similar to forward
            out = {
                "reconstructed_x": reconstructed_x,
                "mean": mean
            }

        if self.observation_model == "image":
            imagined_state = ((out["reconstructed_x"].squeeze().detach().cpu().numpy()*0.5+0.5).transpose(1,2,0)* 255).astype(np.uint8)
        else:
            imagined_state = ((out["reconstructed_x"].squeeze().detach().cpu().numpy()))
        return out["reconstructed_x"], imagined_state
        
    def test(self, data, changed_captions, save_dir):
        # --- Imports needed for drawing on the image ---
        from PIL import Image, ImageDraw, ImageFont
        import torchvision.transforms as transforms
        from torchvision.utils import make_grid
        
        states = []
        descriptions = []
        for item in data:
            states.append(Image.fromarray(item["frame"]))
            descriptions.append(item["description"])

        # Normalising the datapoint to match the training datapoint
        train_transforms = self.train_transform
        images = [image.convert("RGB") for image in states]
        states_tensors = [train_transforms(image) for image in images]
        
        tokeniser = self.decoder.tokenizer
        captions_tokenised, attention_mask = tokenize_captions(tokeniser, descriptions, max_length=self.max_sequence_length)
        captions_tokenised = captions_tokenised.to(device)
        attention_mask = attention_mask.to(device)
        changed_captions_list = list(chain.from_iterable(
            [item if isinstance(item, list) else [item] for sublist in changed_captions for item in (sublist if isinstance(sublist, list) else [sublist])]
        ))
        changed_captions_tokenised_list, changed_caption_attention_mask = tokenize_captions(tokeniser, changed_captions_list, max_length=self.max_sequence_length)
        changed_captions_tokenised_list = changed_captions_tokenised_list.to(device)
        changed_caption_attention_mask = changed_caption_attention_mask.to(device)
        changed_captions_tokenised = [changed_captions_tokenised_list[i:i+len(changed_captions)] for i in range(0, len(changed_captions_tokenised_list), len(changed_captions))]
        changed_caption_attention_mask = [changed_caption_attention_mask[i:i+len(changed_captions)] for i in range(0, len(changed_caption_attention_mask), len(changed_captions))]
        
        # This list will hold all images for the grid in the new order
        grid_images = []

        with torch.no_grad():
            hidden = self.encoder(torch.stack(states_tensors).to(device))
            sampler = self.bottleneck(hidden)
            latents = sampler.mean

            for i in range(len(states_tensors)):
                # 1. ADD ORIGINAL IMAGE
                # The original, ground-truth image tensor
                grid_images.append(states_tensors[i].cpu())

                latent_z = latents[i].unsqueeze(0)
                
                # 2. ADD RECONSTRUCTION
                original_tokens = captions_tokenised[i].unsqueeze(0)
                original_mask = attention_mask[i].unsqueeze(0)
                recon_original, _ = self.decoder(latent_z, original_tokens, original_mask, return_text_feats=True)
                grid_images.append(recon_original.squeeze(0).cpu())

                # 3. ADD GENERATED IMAGES
                changed_tokens = changed_captions_tokenised[i]
                changed_captions_mask = changed_caption_attention_mask[i]
                latent_z_expanded = latent_z.repeat(len(changed_tokens), 1, 1, 1)
                recons_changed, _ = self.decoder(latent_z_expanded, changed_tokens, changed_captions_mask, return_text_feats=True)
                grid_images.extend(list(recons_changed.cpu()))

        # --- 4. CREATE GRID, ADD NUMBERS, AND SAVE ---
        final_image_tensor = torch.stack(grid_images)
        final_image_tensor = (final_image_tensor * 0.5 + 0.5).clamp(0, 1)

        # The grid now has 1 (Original) + 1 (Recon) + 10 (Generated) = 12 columns
        num_columns = 1 + 1 + len(changed_captions)
        num_rows = len(states)
        
        grid_tensor = make_grid(final_image_tensor, nrow=num_columns, padding=2)

        # Convert tensor grid to a PIL Image
        grid_pil = transforms.ToPILImage()(grid_tensor)

        # Define margins and new canvas size
        top_margin, left_margin = 30, 30
        new_width = grid_pil.width + left_margin
        new_height = grid_pil.height + top_margin
        
        final_image = Image.new('RGB', (new_width, new_height), 'white')
        final_image.paste(grid_pil, (left_margin, top_margin))

        draw = ImageDraw.Draw(final_image)
        try:
            font = ImageFont.load_default(size=16)
        except AttributeError:
            font = ImageFont.load_default()

        # Get individual image tile dimensions (including padding)
        tile_width = states[0].width + 2
        tile_height = states[0].height + 2

        # Draw column numbers (0-9), starting from the 3rd column
        num_generated_images = len(changed_captions)
        for j in range(num_generated_images):
            # The number 'j' (0-9) corresponds to the (j+2)-th column in the grid
            column_index_in_grid = j + 2 
            x = left_margin + (column_index_in_grid * tile_width) + (tile_width // 2)
            y = 10
            draw.text((x, y), str(j), fill="black", font=font, anchor="mt")

        # Draw row numbers (i)
        for i in range(num_rows):
            x = 15
            y = top_margin + (i * tile_height) + (tile_height // 2)
            draw.text((x, y), str(i), fill="black", font=font, anchor="lm")
            
        # Save the final image with the new layout and numbering
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'reconstruction_grid_final.png')
        final_image.save(save_path)
        print(f"[INFO] Saved final reconstruction grid to {save_path}")
        
    def generate(self, output: dict, num_samples: int, device, *prompts) -> torch.Tensor:
        """
        Generates a comparison grid to visualize disentanglement.

        For each of the `num_samples`, it creates a row containing:
        1. The original input image.
        2. The reconstruction of that image.
        3. New images generated by pairing the same latent code with each provided prompt.

        Args:
            batch (dict): The original input batch, containing "pixel_values".
            output (dict): The output from the model's forward pass.
            num_samples (int): The number of latent vectors (and thus rows) to use.
            *prompts: A tuple of text prompts, e.g., ("a photo of a cat", "a drawing of a dog").

        Returns:
            torch.Tensor: A single image grid of size (num_samples, 2 + num_prompts).
        """
        # --- 1. SETUP ---
        if not prompts:
            raise ValueError("The generate function requires at least one text prompt.")

        # Handle if prompts are passed as a list/tuple inside the first argument
        prompts = prompts[0] if isinstance(prompts[0], (list, tuple)) else prompts
        num_prompts = len(prompts)

        # --- 2. GATHER BASE IMAGES AND LATENTS ---
        # Get the latents, original images, and reconstructions from the first `num_samples` of the batch.
        # We use .mean for generation as it's the most representative point in the latent distribution.
        latents = output["posterior"].mean[:num_samples]

        # Normalize images from [-1, 1] to [0, 1] for visualization
        original_images = (output["x"][:num_samples].detach() * 0.5 + 0.5).clamp(0, 1)
        reconstructions = (output["reconstructed_x"][:num_samples].detach() * 0.5 + 0.5).clamp(0, 1)

        # --- 3. GENERATE PROMPT-DRIVEN IMAGES ---
        # Tokenize and encode the text prompts to get embeddings
        if self.max_sequence_length is not None:
            length = min(self.max_sequence_length, self.decoder.tokenizer.model_max_length)
        else:
            length = self.decoder.tokenizer.model_max_length
        tokenised_text = self.decoder.tokenizer(
            prompts, max_length=length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        input_ids = tokenised_text["input_ids"].to(device)
        attention_mask = tokenised_text["attention_mask"].to(device)

        # Expand latents and embeddings to create all (z, prompt) pairs
        z_expanded = latents.repeat_interleave(num_prompts, dim=0)
        text_embeddings_expanded = input_ids.repeat(num_samples, 1)
        attention_mask_expanded = attention_mask.repeat(num_samples, 1)

        # Decode the pairs to generate new images
        generated_images, _ = self.decoder(z_expanded, text_embeddings_expanded, attention_mask_expanded.float(), return_text_feats=True)
        
        # Normalize generated images for visualization
        if self.observation_model == "image":
            generated_images = (generated_images.detach() * 0.5 + 0.5).clamp(0, 1)
        else:
            return None
        # --- 4. ASSEMBLE THE GRID ---
        # Create a list to hold all images in the correct order for the grid
        grid_images = []
        for i in range(num_samples):
            # For each sample, add the original image and its reconstruction
            grid_images.append(original_images[i])
            grid_images.append(reconstructions[i])

            # Then, add all the images generated using this sample's latent code
            start_idx = i * num_prompts
            end_idx = start_idx + num_prompts
            grid_images.extend(list(generated_images[start_idx:end_idx]))

        # Stack the list of image tensors into a single batch
        final_image_tensor = torch.stack(grid_images)

        # Create the grid with (2 + num_prompts) columns
        grid = make_grid(final_image_tensor, nrow=(2 + num_prompts))

        return grid

    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path, map_location=device)
        self.encoder.load_state_dict(params["encoder"])
        self.bottleneck.load_state_dict(params["bottleneck"])
        self.decoder.load_state_dict(params["decoder"])
        try:
            self.caption_discriminator.load_state_dict(params["caption_discriminator"])
        except RuntimeError as e:
             print(f"[NOTE] Skipping caption_discriminator loading due to mismatch (acceptable for inference): {e}")
        if self.use_image_discriminator:
            self.image_discriminator.load_state_dict(params["image_discriminator"])
        print("[INFO] loaded the Text Conditioned VAE model", path)

    def save(self, dump_dir, save_name):
        """Save model and optimizer parameters."""
        params = {
                "encoder": self.encoder.state_dict(),
                "bottleneck" : self.bottleneck.state_dict(),
                "decoder" : self.decoder.state_dict(),
                "caption_discriminator" : self.caption_discriminator.state_dict()
                }
        if self.use_image_discriminator:
            params["image_discriminator"] = self.image_discriminator.state_dict()
        save_dir = dump_dir
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = save_dir + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] Text Conditioned VAE model saved to: ", checkpoint_path)
        