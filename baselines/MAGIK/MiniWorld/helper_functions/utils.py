import os
import json
import yaml
import torch
import pickle
import string
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

MAGIK_CLASS_NAMES = {
    0: "green_ball_only",   # Target 3 reward object (Source distractor)
    1: "blue_box_only",     # Source reward object (Target 3 distractor)
    2: "both_objects",      # Both green ball and blue box visible
    3: "no_object",         # Neither object visible (empty room view)
}


def resolve_project_path(path: str) -> str:
    """
    Resolves a path relative to current working directory or PROJECT_ROOT.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    # Default to PROJECT_ROOT for output paths when running from subdirectories
    if os.path.abspath(os.getcwd()) != PROJECT_ROOT:
        return candidate
    return path


def get_magik_class(obs: np.ndarray, caption: str = None) -> np.ndarray:
    """
    Computes the 4-way one-hot MAGIK class label for MiniWorld (Source & Case 3):
      - Class 0 ([1, 0, 0, 0]): Only green ball is visible (Case 3 target / Source distractor)
      - Class 1 ([0, 1, 0, 0]): Only blue box is visible (Source target / Case 3 distractor)
      - Class 2 ([0, 0, 1, 0]): Both green ball and blue box are visible
      - Class 3 ([0, 0, 0, 1]): No objects are visible in the current view

    Uses the exact RGB color bounds defined in `PickObjectEnv.get_class` (`env/MiniWorld.py`).
    """
    if obs is not None:
        obs_img = np.asarray(obs)
        # Convert CHW -> HWC if needed
        if obs_img.ndim == 3 and obs_img.shape[0] == 3 and obs_img.shape[-1] != 3:
            obs_img = np.transpose(obs_img, (1, 2, 0))
        # Convert float [0, 1] -> uint8 [0, 255] if needed
        if obs_img.dtype != np.uint8:
            if obs_img.max() <= 1.0:
                obs_img = np.clip(obs_img * 255.0, 0, 255).astype(np.uint8)
            else:
                obs_img = np.clip(obs_img, 0, 255).astype(np.uint8)

        green_lower, green_upper = np.array([0, 120, 0]), np.array([20, 255, 20])
        blue_lower, blue_upper = np.array([0, 0, 120]), np.array([20, 20, 255])

        has_green = int(np.sum(np.all((obs_img >= green_lower) & (obs_img <= green_upper), axis=-1))) > 2
        has_blue = int(np.sum(np.all((obs_img >= blue_lower) & (obs_img <= blue_upper), axis=-1))) > 2
    elif caption is not None:
        cap_low = caption.lower()
        has_green = "green ball" in cap_low
        has_blue = "blue box" in cap_low
    else:
        has_green = False
        has_blue = False

    if has_green and has_blue:
        return np.array([0, 0, 1, 0], dtype=np.float32)
    elif has_green:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    elif has_blue:
        return np.array([0, 1, 0, 0], dtype=np.float32)
    else:
        return np.array([0, 0, 0, 1], dtype=np.float32)


def get_magik_transformations(
    cls_idx: int,
    policy_mode: str = "single_objects",
    distractor_mode: str = "empty",
):
    """
    Returns (need_transformation, transformations) for MAGIK inference on MiniWorld Case 3.
    Mirrors `baselines/MAGIK/MiniGrid/test_imagination_net.py`:
      - cls_idx == 0 (green ball only -> Case 3 target):
          transform to [0, 1, 0, 0] (blue box only -> Source reward object)
      - cls_idx == 1 (blue box only -> old Source target, now Case 3 distractor):
          transform to [0, 0, 0, 1] (empty room view, or [1, 0, 0, 0] if distractor_mode == 'distractor')
      - cls_idx == 2 (both green ball and blue box visible):
          two-stage transformation [[1, 0, 0, 0], [0, 1, 0, 0]]:
          first isolate green ball ([1, 0, 0, 0]), then transform green ball into blue box ([0, 1, 0, 0])
      - cls_idx == 3 (no object visible):
          no transformation needed
    """
    if policy_mode == "all_objects":
        if cls_idx == 0:
            return True, [torch.tensor([0, 1, 0, 0], dtype=torch.float32)]
        return False, []

    if cls_idx == 0:
        return True, [torch.tensor([0, 1, 0, 0], dtype=torch.float32)]
    elif cls_idx == 1:
        if distractor_mode == "distractor":
            return True, [torch.tensor([1, 0, 0, 0], dtype=torch.float32)]
        return True, [torch.tensor([0, 0, 0, 1], dtype=torch.float32)]
    elif cls_idx == 2:
        return True, [
            torch.tensor([1, 0, 0, 0], dtype=torch.float32),
            torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        ]
    else:
        return False, []


def magik_imagine_state(
    model,
    state_uint8: np.ndarray,
    device: torch.device,
    policy_mode: str = "single_objects",
    distractor_mode: str = "empty",
):
    """
    Runs MAGIK observation classification and imagination on a single MiniWorld observation (H, W, 3) uint8.

    Returns:
        imagined_state_uint8 (np.ndarray): (H, W, 3) uint8 imagined observation for the source policy
        cls_idx (int): Predicted class index (0..3)
        need_transformation (bool): Whether imagination transformation was applied
    """
    x = torch.tensor(
        np.transpose(state_uint8.astype(np.float32) / 255.0, (2, 0, 1)),
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        c_h = model.c_encoder(x.unsqueeze(0))
        c_logits = model.classify(c_h)
        cls_idx = int(torch.argmax(F.softmax(c_logits, dim=-1), dim=-1).item())

        need_transformation, transformations = get_magik_transformations(
            cls_idx, policy_mode=policy_mode, distractor_mode=distractor_mode
        )
        if need_transformation:
            imagined_tensor = x.unsqueeze(0)
            for transformation in transformations:
                imagined_tensor = model.generate(
                    imagined_tensor,
                    transformation.to(device).float().unsqueeze(0),
                )
            imagined_np = imagined_tensor.squeeze(0).detach().cpu().numpy()
            imagined_state_uint8 = np.clip(
                np.transpose(imagined_np, (1, 2, 0)) * 255.0, 0, 255
            ).astype(np.uint8)
        else:
            imagined_state_uint8 = state_uint8

    return imagined_state_uint8, cls_idx, need_transformation


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_dump_directory(path):
    path = resolve_project_path(path)
    rand_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    dump_dir = os.path.join(path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_{}".format(rand_str))
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir


def sample_gumbel_softmax(c_logit, training=True, tau=0.05):
    if training:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(c_logit) + 1e-10) + 1e-10)
        y = F.softmax((c_logit + gumbel_noise) / tau, dim=-1)
        return y
    else:
        return F.softmax(c_logit / tau, dim=-1)


def get_data(datapath):
    datapath = resolve_project_path(datapath)
    return np.load(datapath, allow_pickle=True)


def anneal_coefficient(epoch, max_epoch, start_value, end_value, buffer_timestep, is_increase=True):
    if epoch < buffer_timestep:
        return start_value
    adjusted_epoch = epoch - buffer_timestep
    annealing_duration = max(1, max_epoch - buffer_timestep)
    if is_increase:
        return start_value + (end_value - start_value) * (adjusted_epoch / annealing_duration)
    else:
        return start_value - (start_value - end_value) * (adjusted_epoch / annealing_duration)


def _to_chw_float_tensor(img):
    if isinstance(img, np.ndarray):
        arr = img
    elif torch.is_tensor(img):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] != 3:
        arr = np.transpose(arr, (2, 0, 1))

    is_uint8 = (arr.dtype == np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if is_uint8 or t.max() > 1.0:
        t = t / 255.0
    return t


class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _to_chw_float_tensor(self.data[idx]), idx


class TwoListDataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2

    def __len__(self):
        return len(self.list1)

    def __getitem__(self, idx):
        x = _to_chw_float_tensor(self.list1[idx])
        y = self.list2[idx]
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        elif not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = y.float()
        return x, y


def load_miniworld_magik_dataset(
    datapath: str = "data",
    env_name: str = "MiniWorld",
    collect_if_missing: bool = False,
    env_cfg: dict = None,
    num_samples: int = 150000,
    cache_pkl: bool = True,
):
    """
    Loads (datasets, class_probs) for training MAGIK's M2-VAE on MiniWorld.
    Supports:
      1. Pre-saved MAGIK pickle files: `data/{env_name}/Random/data.pkl` and `class_prob.pkl`
      2. Existing PRISM image datasets: `data/{env_name}/agent/images`, `data/{env_name}/images`,
         or `data/{env_name}/Random/vae/domain_2/images` (automatically computes the 4-way one-hot
         MAGIK class labels for each image and caches `.pkl` files)
      3. On-the-fly collection from `PickObjectEnv` (`env/MiniWorld.py` in `source` task_mode)
         if `collect_if_missing=True`
    """
    base_data_dir = resolve_project_path(os.path.join(datapath, env_name))
    pkl_dir = os.path.join(base_data_dir, "Random")
    data_pkl_path = os.path.join(pkl_dir, "data.pkl")
    cls_pkl_path = os.path.join(pkl_dir, "class_prob.pkl")

    # 1. Check for existing MAGIK pickle files
    if os.path.isfile(data_pkl_path) and os.path.isfile(cls_pkl_path):
        print(f"[INFO] Loading MAGIK dataset from {data_pkl_path} and {cls_pkl_path}...")
        datasets = get_data(data_pkl_path)
        class_probs = get_data(cls_pkl_path)
        return datasets, class_probs

    # 2. Check for existing PRISM image folders
    candidate_dirs = [
        os.path.join(base_data_dir, "agent"),
        base_data_dir,
        os.path.join(base_data_dir, "Random", "vae", "domain_2"),
    ]
    found_img_dir = None
    for cdir in candidate_dirs:
        img_sub = os.path.join(cdir, "images")
        if os.path.isdir(img_sub) and any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(img_sub)):
            found_img_dir = img_sub
            break
        elif os.path.isdir(cdir) and any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(cdir)):
            found_img_dir = cdir
            break

    if found_img_dir is not None:
        print(f"[INFO] Building MAGIK dataset and 4-class labels from PRISM image directory: {found_img_dir}")
        valid_exts = (".png", ".jpg", ".jpeg")
        filenames = sorted(f for f in os.listdir(found_img_dir) if f.lower().endswith(valid_exts))
        datasets = []
        class_probs = []
        for fname in tqdm(filenames, desc="Loading MiniWorld images & extracting MAGIK classes"):
            fpath = os.path.join(found_img_dir, fname)
            try:
                with Image.open(fpath) as img:
                    arr_hwc = np.array(img.convert("RGB"), dtype=np.uint8)
                cls_vec = get_magik_class(arr_hwc)
                # Store as uint8 CHW (3, 80, 80) to keep memory footprint compact (~2.8GB for 150k images)
                arr_chw = np.transpose(arr_hwc, (2, 0, 1))
                datasets.append(arr_chw)
                class_probs.append(cls_vec)
            except Exception as e:
                print(f"[WARNING] Skipping corrupted image {fpath}: {e}")

        datasets = np.asarray(datasets, dtype=np.uint8)
        class_probs = np.asarray(class_probs, dtype=np.float32)

        if cache_pkl and len(datasets) > 0:
            try:
                os.makedirs(pkl_dir, exist_ok=True)
                with open(data_pkl_path, "wb") as f:
                    pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(cls_pkl_path, "wb") as f:
                    pickle.dump(class_probs, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[INFO] Cached MAGIK dataset ({len(datasets)} samples) to {pkl_dir}")
            except Exception as e:
                print(f"[WARNING] Could not cache .pkl files: {e}")

        return datasets, class_probs

    # 3. Collect from environment if requested
    if collect_if_missing:
        print(f"[INFO] Dataset not found on disk. Collecting {num_samples} samples from MiniWorld source environment...")
        import sys
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from env.MiniWorld import PickObjectEnv

        cfg = dict(env_cfg) if env_cfg is not None else {}
        cfg.setdefault("name", "MiniWorld")
        cfg["task_mode"] = cfg.get("source_task_mode", "source")
        cfg.setdefault("size", 10)
        cfg.setdefault("max_steps", 50)
        cfg.setdefault("obs_width", 80)
        cfg.setdefault("obs_height", 80)
        cfg.setdefault("render_mode", "none")
        cfg.setdefault("verbose", False)

        env = PickObjectEnv(cfg)
        datasets = []
        class_probs = []
        state, _ = env.reset()
        for _ in tqdm(range(num_samples), desc="Collecting MiniWorld transitions"):
            action = env.action_space.sample()
            next_state, _, terminated, truncated, _ = env.step(action)
            cls_vec = get_magik_class(next_state)
            datasets.append(np.transpose(next_state.astype(np.uint8), (2, 0, 1)))
            class_probs.append(cls_vec)
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state
        env.close()

        datasets = np.asarray(datasets, dtype=np.uint8)
        class_probs = np.asarray(class_probs, dtype=np.float32)

        if cache_pkl:
            os.makedirs(pkl_dir, exist_ok=True)
            with open(data_pkl_path, "wb") as f:
                pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cls_pkl_path, "wb") as f:
                pickle.dump(class_probs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[INFO] Saved collected MAGIK dataset ({len(datasets)} samples) to {pkl_dir}")

        return datasets, class_probs

    raise FileNotFoundError(
        f"Could not find MiniWorld dataset in '{base_data_dir}'. "
        f"Checked: '{data_pkl_path}', '{os.path.join(base_data_dir, 'agent', 'images')}', "
        f"and '{os.path.join(base_data_dir, 'images')}'. "
        f"Run `python baselines/MAGIK/MiniWorld/helper_functions/collect_vae_training_data.py` "
        f"or pass `General.collect_if_missing=true`."
    )


def write_video(frames, episode, dump_dir, frameSize=(160, 80)):
    import cv2
    os.makedirs(dump_dir, exist_ok=True)
    video_path = os.path.join(dump_dir, f"{episode}.avi")
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2.5, frameSize, isColor=True)
    for img in frames:
        if img.shape[1] != frameSize[0] or img.shape[0] != frameSize[1]:
            img = cv2.resize(img, frameSize)
        video.write(img)
    video.release()
