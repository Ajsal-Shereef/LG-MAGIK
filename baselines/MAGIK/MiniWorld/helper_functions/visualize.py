import os
import math
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image


def my_save_image(grid, filename, resize=False):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if resize:
        im = im.resize((im.size[0] // 3, im.size[1] // 3), Image.Resampling.LANCZOS)
    im.save(filename)
    print("[INFO] Visualisation saved at:", filename)


class Visualizer:
    def __init__(self, model, device, root="visualizations/m2_vae/"):
        self.device = device
        self.model = model
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def reconstructions(self, data, size=(8, 8), label=None):
        data = data.to(self.device).float()
        with torch.no_grad():
            x_reconstructed = self.model.get_reconstruction(data)
            # Flip to Class 1 (blue box, source reward object)
            label_flipped = (
                torch.tensor([0, 1, 0, 0], device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(data.shape[0], 1)
            )
            recon_data_flipped = self.model.generate(data, label_flipped)

        num_images = min(size[0] * size[1] // 2, data.size(0))
        originals = data[:num_images].cpu()
        reconstructions = x_reconstructed[:num_images].cpu()
        reconstructions_flipped = recon_data_flipped[:num_images].cpu()

        comparison = torch.empty((2 * num_images,) + originals.size()[1:])
        comparison[0::2] = originals
        comparison[1::2] = reconstructions

        comparison_flipped = torch.empty((2 * num_images,) + originals.size()[1:])
        comparison_flipped[0::2] = originals
        comparison_flipped[1::2] = reconstructions_flipped

        save_image(comparison.data, os.path.join(self.root, "recon.png"), nrow=size[0], pad_value=0.3)
        save_image(comparison_flipped.data, os.path.join(self.root, "recon_flipped.png"), nrow=size[0], pad_value=0.3)
        print("[INFO] Reconstruction plots saved at", self.root)

    def swap(self, images, device):
        count = len(images)
        image_shape = images.size()[1:]
        with torch.no_grad():
            z_mu, c = self.model.get_latent(images.to(device).float())
            z = z_mu
            u = c
            z_dim = z_mu[0].shape[0]
            latents = []
            for j in range(count):
                for i in range(count):
                    latents.append(torch.cat([z[j], u[i]]).unsqueeze(0))
            latents_tensor = torch.stack(latents).squeeze(1)
            z_tensor = latents_tensor[:, :z_dim]
            u_tensor = latents_tensor[:, z_dim:]
            all_images = torch.zeros(count + 1, count + 1, *image_shape)
            all_images[0, 1:] = images.cpu()
            all_images[1:, 0] = images.cpu()
            all_images[1:, 1:] = self._decode_latents(z_tensor, u_tensor).view(count, count, *image_shape)
            grid = make_grid(all_images.view(-1, *image_shape), nrow=count + 1, pad_value=0)
            my_save_image(grid, os.path.join(self.root, "swap.png"), resize=False)

    def _decode_latents(self, z_dim, c_dim):
        with torch.no_grad():
            return self.model.decoder(z_dim, c_dim).cpu()
