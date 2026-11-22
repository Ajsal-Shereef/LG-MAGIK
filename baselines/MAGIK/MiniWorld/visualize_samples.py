#!/usr/bin/env python3
"""
CLI script to visualize 32 (or N) random images and their corresponding MAGIK labels.
Can sample either uniformly at random across the dataset, or in a balanced manner across classes.
"""
import os
import sys
import argparse

MINIWORLD_MAGIK_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MINIWORLD_MAGIK_DIR, "../../.."))
if MINIWORLD_MAGIK_DIR not in sys.path:
    sys.path.insert(0, MINIWORLD_MAGIK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helper_functions.utils import (
    load_miniworld_magik_dataset,
    MAGIK_CLASS_NAMES,
)
from helper_functions.visualize import draw_random_labeled_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize random samples and their corresponding MAGIK labels."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="data",
        help="Base path to data folder (default: data)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MiniWorld",
        help="Environment name (default: MiniWorld)",
    )
    parser.add_argument(
        "--data-sub-dir",
        type=str,
        default="MAGIK",
        help="Subdirectory for MAGIK dataset pickle files (default: MAGIK)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of images to visualize (default: 32)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "balanced"],
        help="Sampling mode: 'random' for uniform sampling, 'balanced' for equal samples per class (default: random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=4,
        help="Number of grid rows (default: 4)",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=8,
        help="Number of grid columns (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/sample_32_images.png",
        help="Output image filepath (default: visualizations/sample_32_images.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Loading dataset for {args.env} from {args.datapath}...")
    datasets, class_probs = load_miniworld_magik_dataset(
        datapath=args.datapath,
        env_name=args.env,
        data_sub_dir=args.data_sub_dir,
        collect_if_missing=False,
    )
    print(f"[INFO] Loaded {len(datasets)} images and labels.")

    save_path, indices = draw_random_labeled_samples(
        images=datasets,
        labels=class_probs,
        num_samples=args.num_samples,
        save_path=args.output,
        class_names=MAGIK_CLASS_NAMES,
        mode=args.mode,
        seed=args.seed,
        nrows=args.nrows,
        ncols=args.ncols,
    )
    print(f"[SUCCESS] Visualized {len(indices)} samples saved to: {save_path}")


if __name__ == "__main__":
    main()
