import os
import sys
import pickle
import argparse
import numpy as np

MINIWORLD_MAGIK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(MINIWORLD_MAGIK_DIR, "../../.."))
if MINIWORLD_MAGIK_DIR not in sys.path:
    sys.path.insert(0, MINIWORLD_MAGIK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helper_functions.utils import (
    get_magik_class,
    load_miniworld_magik_dataset,
    resolve_project_path,
    MAGIK_CLASS_NAMES,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare or collect MiniWorld transition data and 4-class labels for MAGIK M2-VAE."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MiniWorld",
        help="Name of the MiniWorld environment (default: MiniWorld)",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=150000,
        help="Number of transitions to collect if dataset is not already present (default: 150000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode during collection (default: 50)",
    )
    parser.add_argument(
        "--data-sub-dir",
        type=str,
        default="MAGIK",
        help="Subdirectory for MAGIK dataset pickle files (default: MAGIK)",
    )
    parser.add_argument(
        "--force-collect",
        action="store_true",
        help="Force fresh data collection from PickObjectEnv even if PRISM images exist",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    env_cfg = {
        "name": args.env,
        "task_mode": "source",
        "source_task_mode": "source",
        "size": 10,
        "max_steps": args.max_steps,
        "obs_width": 80,
        "obs_height": 80,
        "render_mode": "none",
        "verbose": False,
    }

    if args.force_collect:
        base_data_dir = resolve_project_path(os.path.join(args.datapath, args.env))
        pkl_dir = os.path.join(base_data_dir, args.data_sub_dir)
        for fname in ("data.pkl", "class_prob.pkl"):
            fpath = os.path.join(pkl_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

    datasets, class_probs = load_miniworld_magik_dataset(
        datapath=args.datapath,
        env_name=args.env,
        data_sub_dir=args.data_sub_dir,
        collect_if_missing=True,
        env_cfg=env_cfg,
        num_samples=args.samples,
        cache_pkl=True,
    )

    cls_indices = np.argmax(class_probs, axis=1)
    print(f"\n[INFO] Total MiniWorld samples ready for MAGIK M2-VAE: {len(datasets)}")
    for idx, cname in MAGIK_CLASS_NAMES.items():
        cnt = int(np.sum(cls_indices == idx))
        pct = (cnt / len(cls_indices) * 100.0) if len(cls_indices) > 0 else 0.0
        print(f"  Class {idx} ({cname:16s}): {cnt:7d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
