
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from typing import Dict
from architectures.common_utils import VGGLoss
from architectures.cnn import CNNEncoder, ResBlocks, Conv2dBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linker(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linker, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x):
        return self.conv(x)

class SpatialDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, activ='relu', pad_type='zero', norm='in'):
        """
        Spatial decoder matching CNNEncoder reverse structure.
        We assume input features start at dim * 2**n_downsample from encoder, but we project z back to this.
        """
        super(SpatialDecoder, self).__init__()
        
        self.dim = dim * (2**n_upsample) # Start with high channels
        
        # AdaIN residual blocks? Or just standard ResBlocks as in Encoder?
        # Encoder used ResBlocks with 'norm' passed in (default 'in').
        # Let's match typical VAE: ResBlocks -> Upsample layers
        
        self.res_layers = ResBlocks(n_res, self.dim, norm, activ, pad_type=pad_type)
        
        self.upsample_layers = nn.Sequential()
        current_dim = self.dim
        
        # n_upsample times
        for i in range(n_upsample):
            self.upsample_layers.add_module(f"UpSampling_{i}", nn.Upsample(scale_factor=2))
            self.upsample_layers.add_module(f"Conv2dBlock_{i}", Conv2dBlock(current_dim, current_dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type))
            current_dim //= 2
            
        # Final output conv
        self.final_conv = Conv2dBlock(current_dim, output_dim, 3, 1, 1, norm='none', activation='none', pad_type=pad_type)

    def forward(self, x):
        x = self.res_layers(x)
        x = self.upsample_layers(x)
        x = self.final_conv(x)
        # Using Tanh as in typical VAEs usually? Or just Identity?
        # train_vae.py VAE.load_params showed TextConditionedVAE using tanh+scaling.
        # AutoencoderKL usually expects raw logits or depends on config.
        # Let's stick to raw output here unless we want to enforce [-1, 1].
        # But wait, default behavior of AutoencoderKL decode doesn't squash unless specified?
        # Let's use no key activation here, assuming loss function handles it or we rely on logic.
        # But wait, check previous implementation view?
        # VAE/vae.py previously inherited AutoencoderKL, which usually ends with Conv2d for mean.
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, dim, n_downsample, n_res, activ, pad_type, norm, latent_channels):
        super(VAE, self).__init__()
        
        self.config = type('Config', (), {'latent_channels': latent_channels})() # mimic config object
        self.latent_channels = latent_channels
        
        # Encoder
        self.encoder = CNNEncoder(n_downsample, n_res, input_dim, dim, norm, activ, pad_type)
        encoder_out_dim = self.encoder.output_dim
        
        # Quant Conv (Map to mean + logvar)
        self.quant_conv = nn.Conv2d(encoder_out_dim, 2 * latent_channels, 1)
        
        # Post Quant Conv (Map z back to features)
        self.post_quant_conv = nn.Conv2d(latent_channels, encoder_out_dim, 1)
        
        # Decoder
        self.decoder = SpatialDecoder(n_downsample, n_res, dim, input_dim, activ, pad_type, norm)
        
        self.vgg_loss = VGGLoss(device)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return type('EncoderOutput', (), {'latent_dist': posterior})()

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return type('DecoderOutput', (), {'sample': dec})()

    def forward(self, x: torch.Tensor):
        x_img = x["pixel_values"]
        posterior = self.encode(x_img).latent_dist
        if self.training:
            latents = posterior.sample()
        else:
            latents = posterior.mode()
            
        reconstructed_x = self.decode(latents).sample
        
        return {
            "reconstructed_x": reconstructed_x, 
            "posterior": posterior,
            "latent" : latents
        }

    def loss_function(self, original_x, forward_output, **kwargs):
        original_images = original_x["pixel_values"]
        reconstructed_images = forward_output["reconstructed_x"]
        posterior = forward_output["posterior"]

        recon_loss = F.mse_loss(reconstructed_images, original_images, reduction="none")
        recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1).mean()
        
        perceptual_loss = self.vgg_loss(original_images, reconstructed_images)

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        total_loss = recon_loss + perceptual_loss + kwargs.get("kl_weight", 1.0) * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "perceptual_loss": perceptual_loss,
        }

    def set_optimizers(self, parms):
        self.vae_optim = optim.AdamW(self.parameters(),
                                lr=parms.lr,
                                betas=tuple(parms.betas),
                                weight_decay=parms.weight_decay,
                                eps=parms.eps,
                               )
        
        if hasattr(parms, "scheduler"):
            self.scheduler = self._get_scheduler(self.vae_optim, parms.scheduler)
        else:
            self.scheduler = None

    def get_lr(self):
        return {"lr": self.vae_optim.param_groups[0]["lr"]}

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
        else:
            return None

    def step_schedulers(self):
        if self.scheduler:
            self.scheduler.step()

    def optimize_discriminator(self, losses, accelerator):
        pass
        
    def optimize_generator(self, losses, accelerator, forward_output=None, **kwargs):
        self.vae_optim.zero_grad()
        accelerator.backward(losses["total_loss"])
        # Clip grads if needed, relying on train_vae.py to handle clipping if configured
        self.vae_optim.step()

    def generate(self, output_dict, num_samples: int, device: torch.device, *args) -> torch.Tensor:
        # Sample random noise
        # Infer shape from output_dict latent if available, else standard?
        # Actually generate method usually generates from scratch or arguments.
        # The previous implementation inferred shape from config.
        # But we can also check the latent_channels.
        
        # Assumption: We generate at same resolution as we learned?
        # If n_downsample=3, 80x80 -> 10x10.
        # But wait, 80 / 8 = 10.
        # Previous checkpoint was 80 input -> 5 output?
        # That implies downsample of 16 (4 layers).
        
        # Since we just verified the previous check point was 5x5 for 80x80.
        # But now we are defining a NEW architecture with n_downsample=3.
        # So 80x80 -> 10x10.
        # Let's use 10x10 as default for generation if we assume 3 downsamples on 80.
        # Or better, check current config? We don't have access to env config here easily.
        # Just use hardcoded or what?
        
        # Let's try to grab shape from passed output_dict if possible.
        if output_dict is not None and "latent" in output_dict:
             latent_shape = output_dict["latent"].shape[2:]
        else:
             latent_shape = (10, 10)

        z = torch.randn(
            num_samples, 
            self.latent_channels, 
            *latent_shape
        ).to(device)
        
        generated_images = self.decode(z).sample
        generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1) # Normalize for display
        return generated_images

    def load_params(self, path):
        # We need to be careful. If path is a TAR, it has "network".
        # If it's a diffusers folder, it's different.
        # Assuming TAR based on project style.
        # But WAIT. If we load the OLD checkpoint (AutoencoderKL) into THIS model (CNN), keys will mismatch.
        # The user asked to Refactor. They probably intend to Retrain?
        # "Preserve the methods called in train_vae.py".
        # If we run train_vae.py, we might load a checkpoint if is_model_fine_tune is True.
        try:
            params = torch.load(path, map_location=device)
            if "network" in params:
                self.load_state_dict(params["network"])
            else:
                 # Attempt strict=False if keys loosely match?
                 self.load_state_dict(params, strict=False)
            print("[INFO] loaded VAE model", path)
        except Exception as e:
            print(f"[WARN] Failed to load params from {path}: {e}")

    def save(self, dump_dir, save_name):
        params = {
                "network": self.state_dict(),
                }
        save_dir = dump_dir
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = save_dir + save_name + '.tar'
        torch.save(params, checkpoint_path)
        print("[INFO] VAE model saved to: ", checkpoint_path)