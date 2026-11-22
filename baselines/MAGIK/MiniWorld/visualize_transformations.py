#!/usr/bin/env python3
"""
Visualizes M2-VAE object counterfactual generation on images sampled from data/MiniWorld/agent.
Creates a 5-column visualization:
  - Column 1: Reconstruction (model.get_reconstruction(x))
  - Column 2: Transformed to Class 0 (green_ball_only: [1, 0, 0, 0])
  - Column 3: Transformed to Class 1 (blue_box_only:   [0, 1, 0, 0])
  - Column 4: Transformed to Class 2 (both_objects:    [0, 0, 1, 0])
  - Column 5: Transformed to Class 3 (no_object/empty: [0, 0, 0, 1])

Also saves an optional 6-column version with the original input image as Column 0.
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from baselines.MAGIK.MiniWorld.architectures.m2_vae.dgm import DeepGenerativeModel
from baselines.MAGIK.MiniWorld.helper_functions.utils import (
    get_magik_class,
    MAGIK_CLASS_NAMES,
)

CLASS_LABELS_5COL = [
    "Reconstruction",
    "→ Class 0: Green Ball",
    "→ Class 1: Blue Box",
    "→ Class 2: Both Objects",
    "→ Class 3: No Object (Empty)",
]

CLASS_COLORS = {
    0: "#2ca02c",  # Green
    1: "#1f77b4",  # Blue
    2: "#d62728",  # Red / Both
    3: "#7f7f7f",  # Gray / Empty
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample images and visualize M2-VAE counterfactual transformations across 5 columns."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/MiniWorld/agent/images",
        help="Directory of agent images (default: data/MiniWorld/agent/images)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model_weights/MiniWorld/MAGIK/m2_vae/seed_42/model.pt",
        help="Path to trained M2-VAE model.pt (default: seed_42)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="baselines/MAGIK/MiniWorld/config/MiniWorld.yaml",
        help="Path to MiniWorld yaml config",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of image rows to visualize (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output-5col",
        type=str,
        default="visualizations/magik_5col_transformations.png",
        help="Output path for 5-column figure",
    )
    parser.add_argument(
        "--output-6col",
        type=str,
        default="visualizations/magik_6col_with_original.png",
        help="Output path for 6-column figure including original image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load configuration & M2-VAE model
    print(f"[INFO] Loading model from: {args.model_path}")
    cfg = OmegaConf.load(args.config_path)
    model = DeepGenerativeModel(cfg.Network).to(device)
    model.load(args.model_path)
    model.to(device)
    model.eval()

    # Find candidate image files
    img_dir = os.path.abspath(args.image_dir)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    valid_exts = (".png", ".jpg", ".jpeg")
    print(f"[INFO] Scanning candidate images from {img_dir}...")
    all_filenames = []
    with os.scandir(img_dir) as it:
        for entry in it:
            if entry.name.lower().endswith(valid_exts):
                all_filenames.append(entry.name)
                if len(all_filenames) >= 15000:
                    break

    if not all_filenames:
        raise RuntimeError(f"No valid images found in {img_dir}")

    # Group candidate images by their ground-truth class to ensure diverse rows
    print("[INFO] Classifying candidates for stratified row sampling...")
    random.shuffle(all_filenames)
    class_buckets = {0: [], 1: [], 2: [], 3: []}
    
    for fname in all_filenames:
        fpath = os.path.join(img_dir, fname)
        try:
            with Image.open(fpath) as pil_img:
                arr_hwc = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            cls_vec = get_magik_class(arr_hwc, min_pixel_threshold=20)
            cls_idx = int(np.argmax(cls_vec))
            class_buckets[cls_idx].append((fname, arr_hwc))
            # Stop once each class has sufficient candidates
            if all(len(b) >= max(3, args.num_samples // 4 + 1) for b in class_buckets.values()):
                break
        except Exception:
            continue

    # Pick samples balanced across classes
    samples_per_class = max(1, args.num_samples // 4)
    selected_items = []
    for c_idx in (0, 1, 2, 3):
        bucket = class_buckets[c_idx]
        chosen = bucket[:samples_per_class] if len(bucket) >= samples_per_class else bucket
        selected_items.extend(chosen)

    # If any slots remain to reach num_samples, fill from remaining
    if len(selected_items) < args.num_samples:
        remaining = [item for b in class_buckets.values() for item in b if item not in selected_items]
        needed = args.num_samples - len(selected_items)
        selected_items.extend(remaining[:needed])

    selected_items = selected_items[:args.num_samples]
    print(f"[INFO] Selected {len(selected_items)} diverse image rows for visualization.")

    # Target class condition tensors: [0..3]
    target_c_tensors = []
    for c in range(4):
        vec = torch.zeros((1, 4), device=device, dtype=torch.float32)
        vec[0, c] = 1.0
        target_c_tensors.append(vec)

    # Compute reconstructions and 4 counterfactual transformations
    rows_data = []  # list of dicts with original, recon, and gen[0..3]
    with torch.no_grad():
        for fname, arr_hwc in selected_items:
            cls_vec = get_magik_class(arr_hwc, min_pixel_threshold=20)
            orig_cls = int(np.argmax(cls_vec))
            orig_cls_name = MAGIK_CLASS_NAMES.get(orig_cls, f"Class {orig_cls}")

            # Tensor shape: (1, 3, 80, 80), float [0, 1]
            x = torch.from_numpy(np.transpose(arr_hwc.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)

            # 1. Reconstruction
            recon_tensor = model.get_reconstruction(x)
            recon_hwc = np.clip(np.transpose(recon_tensor.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)

            # 2..5 Counterfactual generations
            gens = []
            for target_c in target_c_tensors:
                gen_tensor = model.generate(x, target_c)
                gen_hwc = np.clip(np.transpose(gen_tensor.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
                gens.append(gen_hwc)

            rows_data.append({
                "fname": fname,
                "orig_img": arr_hwc,
                "orig_cls": orig_cls,
                "orig_cls_name": orig_cls_name,
                "recon_img": recon_hwc,
                "gens": gens,
            })

    # =========================================================================
    # 1. Generate 5-Column Figure (Exact Request)
    #    Col 1: Reconstruction
    #    Col 2: Class 0 (green ball)
    #    Col 3: Class 1 (blue box)
    #    Col 4: Class 2 (both objects)
    #    Col 5: Class 3 (no object)
    # =========================================================================
    n_rows = len(rows_data)
    fig5, axes5 = plt.subplots(n_rows, 5, figsize=(15, 3.0 * n_rows))

    if n_rows == 1:
        axes5 = np.expand_dims(axes5, axis=0)

    for r_idx, rdata in enumerate(rows_data):
        # Column 0: Reconstruction
        ax_rec = axes5[r_idx, 0]
        ax_rec.imshow(rdata["recon_img"])
        orig_color = CLASS_COLORS.get(rdata["orig_cls"], "#333333")
        ax_rec.set_ylabel(
            f"Row {r_idx + 1}\n{rdata['fname']}\nOrig: {rdata['orig_cls_name']}",
            fontsize=9,
            fontweight="bold",
            color=orig_color,
            rotation=0,
            labelpad=55,
            va="center",
        )
        if r_idx == 0:
            ax_rec.set_title("Column 1\nReconstruction", fontsize=11, fontweight="bold", pad=8)
        ax_rec.set_xticks([])
        ax_rec.set_yticks([])
        for spine in ax_rec.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(1.5)

        # Columns 1..4: Target Object Transformations
        for c_idx in range(4):
            ax = axes5[r_idx, c_idx + 1]
            ax.imshow(rdata["gens"][c_idx])
            col_color = CLASS_COLORS.get(c_idx, "#333333")
            if r_idx == 0:
                ax.set_title(
                    f"Column {c_idx + 2}\n{CLASS_LABELS_5COL[c_idx + 1]}",
                    fontsize=10,
                    fontweight="bold",
                    color=col_color,
                    pad=8,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(col_color)
                spine.set_linewidth(2.0)

    plt.suptitle(
        "MiniWorld MAGIK M2-VAE: Reconstruction & Object Counterfactual Transformations (5 Columns)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_5col)), exist_ok=True)
    plt.savefig(args.output_5col, dpi=160, bbox_inches="tight")
    plt.close(fig5)
    print(f"[SUCCESS] 5-Column visualization saved at: {args.output_5col}")

    # =========================================================================
    # 2. Generate 6-Column Figure (With Original Image alongside Reconstruction)
    # =========================================================================
    fig6, axes6 = plt.subplots(n_rows, 6, figsize=(18, 3.0 * n_rows))
    if n_rows == 1:
        axes6 = np.expand_dims(axes6, axis=0)

    for r_idx, rdata in enumerate(rows_data):
        # Col 0: Original
        ax_orig = axes6[r_idx, 0]
        ax_orig.imshow(rdata["orig_img"])
        orig_color = CLASS_COLORS.get(rdata["orig_cls"], "#333333")
        ax_orig.set_ylabel(
            f"Row {r_idx + 1}\n{rdata['fname']}\n{rdata['orig_cls_name']}",
            fontsize=9,
            fontweight="bold",
            color=orig_color,
            rotation=0,
            labelpad=55,
            va="center",
        )
        if r_idx == 0:
            ax_orig.set_title("Input Original", fontsize=11, fontweight="bold", pad=8)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        for spine in ax_orig.spines.values():
            spine.set_edgecolor(orig_color)
            spine.set_linewidth(2.5)

        # Col 1: Reconstruction
        ax_rec = axes6[r_idx, 1]
        ax_rec.imshow(rdata["recon_img"])
        if r_idx == 0:
            ax_rec.set_title("Reconstruction", fontsize=11, fontweight="bold", pad=8)
        ax_rec.set_xticks([])
        ax_rec.set_yticks([])

        # Cols 2..5: Target classes
        for c_idx in range(4):
            ax = axes6[r_idx, c_idx + 2]
            ax.imshow(rdata["gens"][c_idx])
            col_color = CLASS_COLORS.get(c_idx, "#333333")
            if r_idx == 0:
                ax.set_title(
                    CLASS_LABELS_5COL[c_idx + 1],
                    fontsize=10,
                    fontweight="bold",
                    color=col_color,
                    pad=8,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(col_color)
                spine.set_linewidth(2.0)

    plt.suptitle(
        "MiniWorld MAGIK M2-VAE: Input Original, Reconstruction & Counterfactual Object Generation",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_6col)), exist_ok=True)
    plt.savefig(args.output_6col, dpi=160, bbox_inches="tight")
    plt.close(fig6)
    print(f"[SUCCESS] 6-Column visualization saved at: {args.output_6col}")


if __name__ == "__main__":
    main()
