import os
import math
import torch
import numpy as np
from PIL import Image
try:
    from torchvision.utils import make_grid, save_image
except ImportError:
    make_grid, save_image = None, None


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


MAGIK_DEFAULT_CLASS_NAMES = {
    0: "green_ball_only",
    1: "blue_box_only",
    2: "both_objects",
    3: "no_object",
}

MAGIK_CLASS_COLORS = {
    0: "#2ca02c",  # Green
    1: "#1f77b4",  # Blue
    2: "#d62728",  # Red
    3: "#7f7f7f",  # Gray
}


def draw_random_labeled_samples(
    images,
    labels,
    num_samples: int = 32,
    save_path: str = "visualizations/sample_32_images.png",
    class_names: dict = None,
    mode: str = "random",
    seed: int = None,
    nrows: int = 4,
    ncols: int = 8,
    figsize: tuple = (20, 11),
    show: bool = False,
):
    """
    Draws a grid of `num_samples` (default 32) random images with their corresponding labels.

    Args:
        images: Array-like / Tensor of shape (N, 3, H, W) or (N, H, W, 3).
        labels: Array-like / Tensor of shape (N, C) (one-hot) or (N,) (class indices or string labels).
        num_samples: Number of samples to visualize (default: 32).
        save_path: Path to save the output plot figure.
        class_names: Dict mapping class index -> human-readable string.
        mode: 'random' for uniform random sampling across the whole dataset,
              'balanced' to sample equally from all present classes.
        seed: Optional random seed for reproducible sample selection.
        nrows: Grid rows (default: 4).
        ncols: Grid columns (default: 8).
        figsize: Matplotlib figure size (width, height) in inches.
        show: Whether to call plt.show().

    Returns:
        save_path (str): The file path where the figure was saved.
        selected_indices (np.ndarray): The dataset indices of the visualized samples.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = MAGIK_DEFAULT_CLASS_NAMES

    if seed is not None:
        np.random.seed(seed)

    total_samples = len(images)
    num_samples = min(num_samples, total_samples)

    # Resolve labels to 1D integer indices or string representations
    if torch.is_tensor(labels):
        labels_arr = labels.detach().cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    if labels_arr.ndim == 2:
        class_indices = np.argmax(labels_arr, axis=1)
    else:
        class_indices = labels_arr

    # Select sample indices
    if mode == "balanced" and np.issubdtype(class_indices.dtype, np.integer):
        unique_classes = np.unique(class_indices)
        samples_per_cls = max(1, num_samples // len(unique_classes))
        selected_indices = []
        for c in unique_classes:
            c_indices = np.where(class_indices == c)[0]
            if len(c_indices) > 0:
                chosen = np.random.choice(c_indices, size=min(samples_per_cls, len(c_indices)), replace=False)
                selected_indices.extend(chosen.tolist())
        # If needed, fill remaining slots
        if len(selected_indices) < num_samples:
            remaining = np.setdiff1d(np.arange(total_samples), selected_indices)
            extra = np.random.choice(remaining, size=num_samples - len(selected_indices), replace=False)
            selected_indices.extend(extra.tolist())
        selected_indices = np.array(selected_indices[:num_samples])
        np.random.shuffle(selected_indices)
    else:
        selected_indices = np.random.choice(total_samples, size=num_samples, replace=False)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        img = images[idx]

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        else:
            img = np.asarray(img)

        # Convert CHW -> HWC if needed
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))

        # Normalize pixel values for matplotlib imshow
        if img.dtype != np.uint8 and img.max() > 1.0:
            img = np.clip(img / 255.0, 0.0, 1.0)
        elif img.dtype == np.uint8:
            img = np.clip(img, 0, 255)

        raw_label = class_indices[idx]
        if isinstance(raw_label, (int, np.integer)):
            cls_idx = int(raw_label)
            cls_name = class_names.get(cls_idx, f"Class {cls_idx}")
            title_text = f"[{cls_idx}] {cls_name}"
            color = MAGIK_CLASS_COLORS.get(cls_idx, "#333333")
        else:
            title_text = str(raw_label)
            color = "#333333"

        ax.imshow(img)
        ax.set_title(f"#{idx}\n{title_text}", fontsize=9, fontweight="bold", color=color, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

        # Colored border matching the class
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)

    # Hide any unused subplots
    for j in range(len(selected_indices), len(axes)):
        axes[j].axis("off")

    mode_str = "Balanced" if mode == "balanced" else "Uniform Random"
    plt.suptitle(
        f"Visualisation of {num_samples} Samples ({mode_str}) with MAGIK Class Labels",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Visualisation of {num_samples} samples saved at: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return save_path, selected_indices

