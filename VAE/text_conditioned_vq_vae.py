
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from itertools import chain
from architectures.mlp import MLP
from torchvision.utils import make_grid
from architectures.common_utils import grad_reverse
from architectures.cnn import CNNEncoder, CNNTextConditionedDecoder
from architectures.vae_utils import PatchDiscriminator, VectorQuantizer
from architectures.common_utils import tokenize_captions, get_train_transform_cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === TextConditionedVQVAE (follows your original VAE structure) ===
class TextConditionedVQVAE(nn.Module):
    def __init__(self, **kwargs):
        super(TextConditionedVQVAE, self).__init__()
        
        # ---- Base configs ----
        self.observation_model = kwargs["observation_mode"]
        self.use_image_discriminator = kwargs.get("use_image_discriminator", False)
        # All other base configs from your VAE can be added here if needed

        # ---- Encoder/Decoder ----
        if self.observation_model == "image":
            # --- Network params ---
            n_downsample = kwargs["n_downsample"]
            encoder_n_res = kwargs["encoder_n_res"]
            input_dim = kwargs["input_dim"]
            dim = kwargs["dim"]
            norm = kwargs["norm"]
            activ = kwargs["activ"]
            pad_type = kwargs["pad_type"]
            output_dim = kwargs["output_dim"]
            clip_model = kwargs["text_encoder"]
            discriminator_fc_hidden = kwargs["discriminator_fc_hidden"]
            encoder_final_dim = kwargs["encoder_final_dim"]
            self.is_perceptual_loss = kwargs["is_perceptual_loss"]
            
            # --- VQ-VAE specific params ---
            embedding_dim = dim*(2**n_downsample) # D
            num_embeddings = kwargs["vqvae_params"]["num_embeddings"] # K
            commitment_cost = kwargs["vqvae_params"]["commitment_cost"] # Beta

            self.encoder = CNNEncoder(n_downsample, encoder_n_res, input_dim, dim, norm, activ, pad_type=pad_type)
            # Replace GaussianSampleSpatial with VectorQuantizer
            self.bottleneck = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            # Decoder now takes embedding_dim as input channel
            self.decoder = CNNTextConditionedDecoder(n_downsample, embedding_dim, output_dim, clip_model, embedding_dim)
            
            # Caption Discriminator input dim needs to be embedding_dim * H * W
            self.caption_discriminator = MLP(embedding_dim * np.prod(encoder_final_dim),
                                             self.decoder.tokenizer.model_max_length * self.decoder.text_dim,
                                             discriminator_fc_hidden)
            
            self.train_transform = get_train_transform_cnn()

            if self.use_image_discriminator:
                disc_params = kwargs.get("disc_params", {})
                in_ch = kwargs.get("disc_in_channels", input_dim)
                self.image_discriminator = PatchDiscriminator(in_channels=in_ch, **disc_params).to(device)
            else:
                self.image_discriminator = None
            if self.is_perceptual_loss:
                from architectures.common_utils import VGGLoss
                self.vgg_loss = VGGLoss(device)
        else:
            raise NotImplementedError("VQ-VAE is only implemented for image observation model.")

    def set_optimizers(self, parms):
        # The optimizer setup remains the same, as `self.bottleneck.parameters()`
        # will correctly grab the learnable embedding weights from the codebook.
        vae_params = list(self.encoder.parameters()) + list(self.bottleneck.parameters()) + \
                     list(self.decoder.parameters()) + list(self.caption_discriminator.parameters())
        if self.image_discriminator:
            vae_params += list(self.image_discriminator.parameters())
            
        self.vae_optim = optim.AdamW(vae_params,
                                   lr=parms.lr,
                                   betas=tuple(parms.betas),
                                   weight_decay=parms.weight_decay,
                                   eps=parms.eps)

    def forward(self, x):
        images = x["pixel_values"]
        text_tokens = x["input_ids"]
        attention_mask = x["attention_masks"]
        
        hidden = self.encoder(images)
        bottleneck_out = self.bottleneck(hidden)
        quantized_latent = bottleneck_out["quantized"]
        
        # The rest of the forward pass is the same, just using the quantized latent
        reconstructed_x = self.decoder(quantized_latent, text_tokens, attention_mask)

        return {
            "x": images,
            "reconstructed_x": reconstructed_x,
            "vq_loss": bottleneck_out["vq_loss"],
            "perplexity": bottleneck_out["perplexity"],
            "latent": quantized_latent # Using "latent" key for consistency
        }

    def loss_function(self, original_x, forward_output, **kwargs):
        original_x = original_x["pixel_values"]
        reconstructed_x = forward_output["reconstructed_x"]
        vq_loss = forward_output["vq_loss"]
        
        # --- Reconstruction Loss ---
        recon_loss = F.mse_loss(reconstructed_x, original_x, reduction="none")
        
        if self.observation_model == "image":
            recon_loss = recon_loss.sum(dim=[1,2,3]).mean()
        else:
            raise NotImplementedError("VQ-VAE is only implemented for image observation model.")
        
        if self.is_perceptual_loss:
            perceptual_loss = self.vgg_loss(original_x, reconstructed_x)
        else:
            perceptual_loss = torch.tensor(0.0, device=original_x.device)
            
        # --- Adversarial Disentanglement Loss ---
        quantized_latent = forward_output["latent"]
        z_grl = grad_reverse(quantized_latent, kwargs["adv_lambda"])
        # ... (rest of the caption discriminator loss code is identical to your VAE)
        pred = self.caption_discriminator(z_grl.view(z_grl.shape[0], -1))
        pred = pred.view_as(self.decoder.text_feats)
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(self.decoder.text_feats.detach(), dim=-1)
        B, T, Fd = pred_n.shape
        pred_flat = pred_n.reshape(B*T, Fd)
        tgt_flat = tgt_n.reshape(B*T, Fd)
        logits = (pred_flat @ tgt_flat.T) / kwargs.get("temperature", 0.07)
        labels = torch.arange(B*T, device=logits.device)
        disc_loss = F.cross_entropy(logits, labels)
        
        # --- Total VQ-VAE Loss ---
        # The main loss is recon_loss + vq_loss
        total_loss = recon_loss + perceptual_loss + vq_loss + kwargs["adv_weight"] * disc_loss

        # --- Patch Discriminator Loss ---
        img_disc_loss = torch.tensor(0.0, device=original_x.device)
        if self.image_discriminator is not None:
            D_real = self.image_discriminator(original_x)
            reconstructed_grl = grad_reverse(reconstructed_x, kwargs.get("img_adv_lambda", kwargs["adv_lambda"]))
            D_fake = self.image_discriminator(reconstructed_grl)
            loss_D_real = F.relu(1.0 - D_real).mean()
            loss_D_fake = F.relu(1.0 + D_fake).mean()
            img_disc_loss = loss_D_real + loss_D_fake
            total_loss += kwargs.get("img_adv_weight", kwargs["adv_weight"]) * img_disc_loss

        return {
            "vae_loss": total_loss, # Using "vae_loss" key for consistency with your training loop
            "recon_loss": recon_loss,
            "perceptual_loss" : perceptual_loss,
            "vq_loss": vq_loss,
            "adversarial_loss": disc_loss,
            "img_disc_loss": img_disc_loss,
            "perplexity": forward_output["perplexity"]
        }

    def optimize(self, losses, accelerator):
        self.vae_optim.zero_grad()
        accelerator.backward(losses["vae_loss"])
        self.vae_optim.step()
        
    def imagine(self, state, description):
        train_transforms = self.train_transform
        images = [Image.fromarray(image).convert("RGB") for image in np.expand_dims(state, axis=0)]
        states_tensors = [train_transforms(image) for image in images]
        state_tensor = torch.stack(states_tensors)
        
        tokeniser = self.decoder.tokenizer
        captions_tokenised, attention_mask = tokenize_captions(tokeniser, [description])
        captions_tokenised = captions_tokenised.to(device)
        attention_mask = attention_mask.to(device)
        out = self({"pixel_values" : state_tensor.to(device),
                      "input_ids" : captions_tokenised,
                      "attention_masks" : attention_mask})
        imagined_state = ((out["reconstructed_x"].squeeze().detach().cpu().numpy()*0.5+0.5).transpose(1,2,0)* 255).astype(np.uint8)
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

        train_transforms = self.train_transform
        images = [image.convert("RGB") for image in states]
        states_tensors = [train_transforms(image) for image in images]
        
        tokeniser = self.decoder.tokenizer
        captions_tokenised, attention_mask = tokenize_captions(tokeniser, descriptions)
        captions_tokenised = captions_tokenised.to(device)
        attention_mask = attention_mask.to(device)
        changed_captions_list = list(chain.from_iterable(
            [item if isinstance(item, list) else [item] for sublist in changed_captions for item in (sublist if isinstance(sublist, list) else [sublist])]
        ))
        changed_captions_tokenised_list, changed_caption_attention_mask = tokenize_captions(tokeniser, changed_captions_list)
        changed_captions_tokenised_list = changed_captions_tokenised_list.to(device)
        changed_caption_attention_mask = changed_caption_attention_mask.to(device)
        changed_captions_tokenised = [changed_captions_tokenised_list[i:i+len(changed_captions)] for i in range(0, len(changed_captions_tokenised_list), len(changed_captions))]
        changed_caption_attention_mask = [changed_caption_attention_mask[i:i+len(changed_captions)] for i in range(0, len(changed_caption_attention_mask), len(changed_captions))]
        
        grid_images = []

        with torch.no_grad():
            hidden = self.encoder(torch.stack(states_tensors).to(device))
            
            # --- Get quantized latents from the VQ bottleneck ---
            bottleneck_out = self.bottleneck(hidden)
            latents = bottleneck_out["quantized"] # Instead of sampler.mean

            for i in range(len(states_tensors)):
                # 1. ADD ORIGINAL IMAGE
                grid_images.append(states_tensors[i].cpu())

                latent_z = latents[i].unsqueeze(0)
                
                # 2. ADD RECONSTRUCTION
                original_tokens = captions_tokenised[i].unsqueeze(0)
                original_mask = attention_mask[i].unsqueeze(0)
                recon_original = self.decoder(latent_z, original_tokens, original_mask)
                grid_images.append(recon_original.squeeze(0).cpu())

                # 3. ADD GENERATED IMAGES
                changed_tokens = changed_captions_tokenised[i]
                changed_captions_mask = changed_caption_attention_mask[i]
                latent_z_expanded = latent_z.repeat(len(changed_tokens), 1, 1, 1)
                recons_changed = self.decoder(latent_z_expanded, changed_tokens, changed_captions_mask)
                grid_images.extend(list(recons_changed.cpu()))

        # --- 4. CREATE GRID, ADD NUMBERS, AND SAVE ---
        final_image_tensor = torch.stack(grid_images)
        final_image_tensor = (final_image_tensor * 0.5 + 0.5).clamp(0, 1)
        num_columns = 1 + 1 + len(changed_captions)
        grid_tensor = make_grid(final_image_tensor, nrow=num_columns, padding=2)
        grid_pil = transforms.ToPILImage()(grid_tensor)
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
        tile_width = states[0].width + 2
        tile_height = states[0].height + 2
        num_generated_images = len(changed_captions)
        for j in range(num_generated_images):
            column_index_in_grid = j + 2
            x = left_margin + (column_index_in_grid * tile_width) + (tile_width // 2)
            y = 10
            draw.text((x, y), str(j), fill="black", font=font, anchor="mt")
        num_rows = len(states)
        for i in range(num_rows):
            x = 15
            y = top_margin + (i * tile_height) + (tile_height // 2)
            draw.text((x, y), str(i), fill="black", font=font, anchor="lm")
            
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'reconstruction_grid_final.png')
        final_image.save(save_path)
        print(f"[INFO] Saved final reconstruction grid to {save_path}")

    def generate(self, output: dict, num_samples: int, device, *prompts) -> torch.Tensor:
        # --- 1. SETUP ---
        if not prompts:
            raise ValueError("The generate function requires at least one text prompt.")
        prompts = prompts[0] if isinstance(prompts[0], (list, tuple)) else prompts
        num_prompts = len(prompts)

        # --- 2. GATHER BASE IMAGES AND LATENTS ---
        
        # --- Get quantized latents from the forward output dictionary ---
        # Instead of `output["posterior"].mean`, we use the "latent" key which holds the quantized tensor.
        latents = output["latent"][:num_samples]

        original_images = (output["x"][:num_samples].detach() * 0.5 + 0.5).clamp(0, 1)
        reconstructions = (output["reconstructed_x"][:num_samples].detach() * 0.5 + 0.5).clamp(0, 1)

        # --- 3. GENERATE PROMPT-DRIVEN IMAGES (This part is identical) ---
        tokenised_text = self.decoder.tokenizer(
            prompts, max_length=self.decoder.tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = tokenised_text["input_ids"].to(device)
        attention_mask = tokenised_text["attention_mask"].to(device)

        z_expanded = latents.repeat_interleave(num_prompts, dim=0)
        text_embeddings_expanded = input_ids.repeat(num_samples, 1)
        attention_mask_expanded = attention_mask.repeat(num_samples, 1)

        generated_images = self.decoder(z_expanded, text_embeddings_expanded, attention_mask_expanded.float())
        if self.observation_model == "image":
            generated_images = (generated_images.detach() * 0.5 + 0.5).clamp(0, 1)
        else:
            return None

        # --- 4. ASSEMBLE THE GRID (This part is identical) ---
        grid_images = []
        for i in range(num_samples):
            grid_images.append(original_images[i])
            grid_images.append(reconstructions[i])
            start_idx = i * num_prompts
            end_idx = start_idx + num_prompts
            grid_images.extend(list(generated_images[start_idx:end_idx]))

        final_image_tensor = torch.stack(grid_images)
        grid = make_grid(final_image_tensor, nrow=(2 + num_prompts))
        return grid
    
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path, map_location=device)
        self.encoder.load_state_dict(params["encoder"])
        self.bottleneck.load_state_dict(params["bottleneck"])
        self.decoder.load_state_dict(params["decoder"])
        self.caption_discriminator.load_state_dict(params["caption_discriminator"])
        if self.use_image_discriminator:
            self.image_discriminator.load_state_dict(params["image_discriminator"])
        print("[INFO] loaded the Text Conditioned VQ-VAE model", path)

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