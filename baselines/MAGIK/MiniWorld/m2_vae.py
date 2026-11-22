import os
import sys

MINIWORLD_MAGIK_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MINIWORLD_MAGIK_DIR, "../../.."))
if MINIWORLD_MAGIK_DIR not in sys.path:
    sys.path.insert(0, MINIWORLD_MAGIK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydra
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from itertools import cycle
from torch.autograd import Variable
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from helper_functions.utils import (
    TwoListDataset,
    load_miniworld_magik_dataset,
    resolve_project_path,
    MAGIK_CLASS_NAMES,
)
from architectures.m2_vae.dgm import DeepGenerativeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="config", config_name="MiniWorld")
def main(args: DictConfig) -> None:
    seed = int(args.General.get("seed", 42))
    seed_all(seed)

    if torch.cuda.is_available():
        use_benchmark = bool(args.Network.get("cudnn_benchmark", True))
        torch.backends.cudnn.benchmark = use_benchmark
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print(f"[INFO] cuDNN benchmark: {torch.backends.cudnn.benchmark}, TF32 precision: high")

    use_wandb = bool(args.General.get("use_wandb", False))
    if use_wandb:
        import wandb
        wandb.init(
            project=f"MAGIK_{args.General.env}",
            name=f"M2_VAE_{args.General.env}_seed_{seed}",
            config=OmegaConf.to_container(args, resolve=True),
        )

    # Load dataset and 4-way one-hot MAGIK labels
    datasets, class_probs = load_miniworld_magik_dataset(
        datapath=args.General.datapath,
        env_name=args.General.env,
        data_sub_dir=args.General.get("data_sub_dir", "MAGIK"),
        collect_if_missing=bool(args.General.get("collect_if_missing", False)),
        env_cfg=OmegaConf.to_container(args.General, resolve=True),
        num_samples=int(args.General.get("num_collect_samples", 150000)),
        cache_pkl=True,
    )
    print("[INFO] Total data:", len(datasets))

    cls_indices = np.argmax(class_probs, axis=1)
    for idx, cname in MAGIK_CLASS_NAMES.items():
        cnt = int(np.sum(cls_indices == idx))
        print(f"  Class {idx} ({cname}): {cnt}")

    try:
        unlabelled_data, labelled_data, unused_labels, labels = train_test_split(
            datasets,
            class_probs,
            test_size=float(args.General.human_label_ratio),
            random_state=seed,
            stratify=cls_indices,
        )
    except ValueError:
        unlabelled_data, labelled_data, unused_labels, labels = train_test_split(
            datasets,
            class_probs,
            test_size=float(args.General.human_label_ratio),
            random_state=seed,
        )

    print("[INFO] Number of labelled data:", len(labelled_data))
    print("[INFO] Labelled data ratio:", args.General.human_label_ratio)

    save_dir_cfg = args.Test.get(
        "save_dir",
        f"model_weights/{args.General.env}/MAGIK/m2_vae/seed_{seed}",
    )
    model_dir = resolve_project_path(save_dir_cfg)
    os.makedirs(model_dir, exist_ok=True)
    OmegaConf.save(config=args, f=os.path.join(model_dir, "config.yaml"))
    print("[INFO] Model dump dir:", model_dir)

    # Optional balanced class weights to handle class imbalance across the 4 MiniWorld view classes
    class_weights_tensor = None
    if args.Network.get("use_class_weights", True):
        y_indices = np.argmax(labels, axis=1)
        present_classes = np.unique(y_indices)
        if len(present_classes) == int(args.Network.y_dim):
            cw = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(int(args.Network.y_dim)),
                y=y_indices,
            )
            class_weights_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
            print("[INFO] Balanced class weights:", cw)

    batch_size = int(args.Network.batch_size)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        total_cpus = int(slurm_cpus)
    else:
        try:
            total_cpus = len(os.sched_getaffinity(0))
        except Exception:
            total_cpus = os.cpu_count() or 8

    # Set PyTorch intra-op threads to match requested CPU allocation
    torch.set_num_threads(max(1, total_cpus))

    cfg_workers = args.Network.get("num_workers", "auto")
    if str(cfg_workers).lower() in ("auto", "none"):
        # Two simultaneous DataLoaders (unlabelled and labelled):
        # allocate half to each so total workers match the requested CPUs
        num_workers = max(1, total_cpus // 2)
    else:
        num_workers = int(cfg_workers)

    pin_memory = bool(args.Network.get("pin_memory", True))
    print(
        f"[INFO] CPUs allocated: {total_cpus} (PyTorch threads: {torch.get_num_threads()}). "
        f"DataLoader workers per loader: {num_workers} ({num_workers * 2} total across labelled/unlabelled)."
    )

    unlabelled_ds = TwoListDataset(unlabelled_data, unused_labels)
    labelled_ds = TwoListDataset(labelled_data, labels)

    drop_last_u = len(unlabelled_ds) > batch_size
    drop_last_l = len(labelled_ds) > batch_size

    unlabelled_data_loader = DataLoader(
        unlabelled_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_u,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    labelled_data_loader = DataLoader(
        labelled_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_l,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model = DeepGenerativeModel(args.Network, class_weights=class_weights_tensor).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.Network.lr_model), betas=(0.9, 0.999)
    )

    epoch_bar = tqdm(range(int(args.Network.epochs)), desc="Training Progress", unit="epoch")
    num_batches = max(1, len(unlabelled_data_loader))

    for epoch in epoch_bar:
        model.train()
        total_loss, accuracy_labeled, accuracy_unlabeled, hsic_losses = 0.0, 0.0, 0.0, 0.0
        classification_losses, reconstruction_errors, z_kl_losses, u_kl_losses = 0.0, 0.0, 0.0, 0.0
        L_losses, U_losses = 0.0, 0.0

        for (unlabelled, unused_lbls), (labelled, lbls) in zip(
            unlabelled_data_loader, cycle(labelled_data_loader)
        ):
            # Asynchronous GPU transfer with pinned memory
            if labelled.dtype == torch.uint8:
                x = labelled.to(device, non_blocking=True).float().div_(255.0)
            else:
                x = labelled.to(device, non_blocking=True).float()
                if x.max() > 1.0:
                    x = x / 255.0
            y = lbls.to(device, non_blocking=True).float()

            if unlabelled.dtype == torch.uint8:
                u = unlabelled.to(device, non_blocking=True).float().div_(255.0)
            else:
                u = unlabelled.to(device, non_blocking=True).float()
                if u.max() > 1.0:
                    u = u / 255.0
            uy = unused_lbls.to(device, non_blocking=True).float()

            x_reconstructed, x_z, x_z_mu, x_z_log_var, x_c_logits, x_c = model(x, y)
            kl_divergence_weight = float(args.Network.kl_divergence_weight)
            wessestein_weight = 1.0

            total_loss_L, cls_loss = model.L(
                x,
                x_reconstructed,
                x_z_mu,
                x_z_log_var,
                y,
                x_c_logits,
                x_c,
                wessestein_weight,
                kl_weight=kl_divergence_weight,
            )

            u_reconstructed, u_z, u_z_mu, u_z_log_var, u_c_logits, u_c = model(u)
            total_loss_U, recon_loss, kl_z, kl_c, hsic_loss = model.U(
                u,
                u_reconstructed,
                u_z_mu,
                u_z_log_var,
                u_c_logits,
                u_c,
                wessestein_weight,
                kl_weight=kl_divergence_weight,
            )

            J_alpha = total_loss_L + total_loss_U
            optimizer.zero_grad()
            J_alpha.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += J_alpha.item()
            classification_losses += cls_loss.item()
            reconstruction_errors += recon_loss.item()
            z_kl_losses += kl_z.item()
            u_kl_losses += kl_c.item()
            L_losses += total_loss_L.item()
            U_losses += total_loss_U.item()
            hsic_losses += hsic_loss.item()

            acc_l = torch.mean(
                (torch.argmax(F.softmax(x_c_logits, dim=-1), dim=1) == torch.argmax(y, dim=1)).float()
            )
            acc_u = torch.mean(
                (torch.argmax(F.softmax(u_c_logits, dim=-1), dim=1) == torch.argmax(uy, dim=1)).float()
            )
            accuracy_labeled += acc_l.item()
            accuracy_unlabeled += acc_u.item()

        if epoch % int(args.Network.save_freequency) == 0:
            model.save(model_dir, epoch)

        metrics = {
            "Loss": total_loss / num_batches,
            "Classification_loss": classification_losses / num_batches,
            "Reconstruction_error": reconstruction_errors / num_batches,
            "Z_kl_loss": z_kl_losses / num_batches,
            "U_kl_loss": u_kl_losses / num_batches,
            "L loss": L_losses / num_batches,
            "U loss": U_losses / num_batches,
            "HSIC Loss": hsic_losses / num_batches,
            "Accuracy": accuracy_labeled / num_batches,
            "Unlabelled Accuracy": accuracy_unlabeled / num_batches,
            "Current Learning rate": optimizer.param_groups[0]["lr"],
            "KL divergence weight": kl_divergence_weight,
        }
        if use_wandb:
            import wandb
            wandb.log(metrics, step=epoch)

        epoch_bar.set_description(
            f"Acc L: {metrics['Accuracy']:.4f} | Acc U: {metrics['Unlabelled Accuracy']:.4f} | Loss: {metrics['Loss']:.4f}"
        )

    model.save(model_dir, int(args.Network.epochs))
    model.eval()
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
