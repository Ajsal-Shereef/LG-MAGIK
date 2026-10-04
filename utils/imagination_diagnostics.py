import os
import re
import cv2
import json
import torch
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Any, Tuple, Optional


# ==============================================================================
# 1. GRID-BASED IMAGE DIFFERENCE ANALYSIS (VAE)
# ==============================================================================

ENV_GRID_SPECS = {
    "MiniGridRelational": {
        "expected_shape": (128, 128),
        "grid_rows": 8,
        "grid_cols": 8,
        "cell_h": 16,
        "cell_w": 16,
        "total_cells": 64
    },
    "SimplePickup": {
        "expected_shape": (56, 56),
        "grid_rows": 7,
        "grid_cols": 7,
        "cell_h": 8,
        "cell_w": 8,
        "total_cells": 49
    },
    "MiniWorld": {
        "expected_shape": (80, 80),
        "grid_rows": 8,
        "grid_cols": 8,
        "cell_h": 10,
        "cell_w": 10,
        "total_cells": 64
    },
    "PickEnv": {
        "expected_shape": (128, 128),
        "grid_rows": 8,
        "grid_cols": 8,
        "cell_h": 16,
        "cell_w": 16,
        "total_cells": 64
    }
}


def get_env_grid_spec(env_name: str, img_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Returns the exact grid partition specifications for the given environment.
    MiniGridRelational: 128x128 full observation, 8x8 grid -> 16x16 px per cell.
    SimplePickup: 56x56 partial view (agent_view_size 7) -> 7x7 grid, 8x8 px per cell.
    MiniWorld: 80x80 observation -> 8x8 grid, 10x10 px per cell.
    """
    if env_name:
        for key, spec in ENV_GRID_SPECS.items():
            if key.lower() in env_name.lower():
                return spec

    # Default fallback: 8x8 grid based on input dimensions
    h, w = img_shape[:2]
    cell_h = max(h // 8, 1)
    cell_w = max(w // 8, 1)
    return {
        "expected_shape": (h, w),
        "grid_rows": 8,
        "grid_cols": 8,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "total_cells": 64
    }


def compute_grid_differences(
    original_np: np.ndarray,
    imagined_np: np.ndarray,
    env_name: str = "MiniWorld",
    cell_diff_threshold: float = 15.0,
    fail_threshold: Optional[int] = None
) -> Dict[str, Any]:
    """
    Divides the original and imagined images into environment-specific grid cells.
    Computes the Mean Absolute Error (MAE) per cell.
    A grid cell is marked as significantly differing if MAE >= cell_diff_threshold (default 15.0 on [0, 255]).
    If differing_grid_count >= fail_threshold (default 10 for MiniWorld, 5 for other envs), the VAE is marked as failed.
    """
    if fail_threshold is not None:
        eff_fail_threshold = fail_threshold
    elif env_name and "MiniWorld" in env_name:
        eff_fail_threshold = 10
    else:
        eff_fail_threshold = 5

    spec = get_env_grid_spec(env_name, original_np.shape)
    exp_h, exp_w = spec["expected_shape"]
    grid_rows = spec["grid_rows"]
    grid_cols = spec["grid_cols"]
    cell_h = spec["cell_h"]
    cell_w = spec["cell_w"]
    total_cells = spec["total_cells"]

    # Ensure RGB images
    orig = original_np.copy()
    imag = imagined_np.copy()

    # Scale to [0, 255] if normalized in [0, 1]
    if orig.max() <= 1.01 and orig.dtype != np.uint8:
        orig = (orig * 255.0).clip(0, 255)
    if imag.max() <= 1.01 and imag.dtype != np.uint8:
        imag = (imag * 255.0).clip(0, 255)

    # Resize to exact expected observation dimensions if needed
    if orig.shape[0] != exp_h or orig.shape[1] != exp_w:
        orig = cv2.resize(orig.astype(np.uint8), (exp_w, exp_h), interpolation=cv2.INTER_AREA)
    if imag.shape[0] != exp_h or imag.shape[1] != exp_w:
        imag = cv2.resize(imag.astype(np.uint8), (exp_w, exp_h), interpolation=cv2.INTER_AREA)

    orig_f = orig.astype(np.float32)
    imag_f = imag.astype(np.float32)

    diff_matrix = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    differing_cells = []

    for r in range(grid_rows):
        y1 = r * cell_h
        y2 = min((r + 1) * cell_h, exp_h)
        for c in range(grid_cols):
            x1 = c * cell_w
            x2 = min((c + 1) * cell_w, exp_w)

            cell_orig = orig_f[y1:y2, x1:x2]
            cell_imag = imag_f[y1:y2, x1:x2]

            cell_diff = float(np.mean(np.abs(cell_orig - cell_imag)))
            diff_matrix[r, c] = cell_diff

            if cell_diff >= cell_diff_threshold:
                differing_cells.append({
                    "row": r,
                    "col": c,
                    "diff": round(cell_diff, 2)
                })

    differing_grid_count = len(differing_cells)
    is_grid_failed = bool(differing_grid_count >= eff_fail_threshold)

    return {
        "differing_grid_count": differing_grid_count,
        "fail_threshold": eff_fail_threshold,
        "cell_diff_threshold": cell_diff_threshold,
        "is_grid_failed": is_grid_failed,
        "grid_shape": [grid_rows, grid_cols],
        "cell_size": [cell_h, cell_w],
        "total_cells": total_cells,
        "max_grid_diff": round(float(np.max(diff_matrix)), 2) if total_cells > 0 else 0.0,
        "mean_grid_diff": round(float(np.mean(diff_matrix)), 2) if total_cells > 0 else 0.0,
        "differing_cells": differing_cells,
        "grid_diff_matrix": diff_matrix.round(2).tolist()
    }


# ==============================================================================
# 2. LATENT DIFFERENCE ANALYSIS (VAE)
# ==============================================================================

def compute_latent_difference(
    vision_model,
    original_tensor: Optional[torch.Tensor] = None,
    imagined_tensor: Optional[torch.Tensor] = None,
    original_np: Optional[np.ndarray] = None,
    imagined_np: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    latent_l2_threshold: Optional[float] = None,
    env_name: str = "MiniWorld"
) -> Dict[str, Any]:
    """
    Encodes original and imagined inputs through the VAE encoder to obtain their latent representations.
    Calculates normalized L2 distance between the latent means.
    Marks is_latent_failed = True if normalized L2 distance exceeds threshold (default 0.75).
    """
    if latent_l2_threshold is not None:
        l2_thresh = latent_l2_threshold
    else:
        l2_thresh = 0.75

    if vision_model is None:
        return {
            "latent_l2": None,
            "latent_cosine": None,
            "latent_l2_threshold": l2_thresh,
            "is_latent_failed": False,
            "is_latent_consistent": True,
            "note": "Latent difference not evaluated (vision model missing)"
        }

    try:
        with torch.no_grad():
            orig_t = original_tensor
            imag_t = imagined_tensor

            if orig_t is None and original_np is not None:
                if hasattr(vision_model, "train_transform"):
                    orig_t = vision_model.train_transform(original_np).unsqueeze(0)
                else:
                    arr = (original_np.astype(np.float32) / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)
                    orig_t = torch.from_numpy(arr).unsqueeze(0)

            if imag_t is None and imagined_np is not None:
                if hasattr(vision_model, "train_transform"):
                    imag_t = vision_model.train_transform(imagined_np).unsqueeze(0)
                else:
                    arr = (imagined_np.astype(np.float32) / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)
                    imag_t = torch.from_numpy(arr).unsqueeze(0)

            if orig_t is None or imag_t is None:
                return {
                    "latent_l2": None,
                    "latent_cosine": None,
                    "latent_l2_threshold": l2_thresh,
                    "is_latent_failed": False,
                    "is_latent_consistent": True,
                    "note": "Latent difference not evaluated (tensors unavailable)"
                }

            if device is not None:
                orig_t = orig_t.to(device)
                imag_t = imag_t.to(device)

            if orig_t.dim() == 3:
                orig_t = orig_t.unsqueeze(0)
            if imag_t.dim() == 3:
                imag_t = imag_t.unsqueeze(0)

            # 1. Encode original
            hidden_orig = vision_model.encoder(orig_t)
            if getattr(vision_model, "latent_type", "spatial") == "vector":
                sampler_orig = vision_model.bottleneck(hidden_orig.flatten(1))
            else:
                sampler_orig = vision_model.bottleneck(hidden_orig)
            mean_orig = sampler_orig.mean.flatten(1)

            # 2. Encode imagined
            hidden_recon = vision_model.encoder(imag_t)
            if getattr(vision_model, "latent_type", "spatial") == "vector":
                sampler_recon = vision_model.bottleneck(hidden_recon.flatten(1))
            else:
                sampler_recon = vision_model.bottleneck(hidden_recon)
            mean_recon = sampler_recon.mean.flatten(1)

            # 3. Normalized L2 distance & Cosine similarity
            l2_dist = float(torch.norm(mean_orig - mean_recon, p=2).item()) / np.sqrt(mean_orig.numel())
            cos_sim = float(torch.nn.functional.cosine_similarity(mean_orig, mean_recon).mean().item())

            is_failed = bool(l2_dist >= l2_thresh)

            return {
                "latent_l2": round(l2_dist, 4),
                "latent_cosine": round(cos_sim, 4),
                "latent_l2_threshold": l2_thresh,
                "is_latent_failed": is_failed,
                "is_latent_consistent": not is_failed
            }
    except Exception as e:
        return {
            "latent_l2": None,
            "latent_cosine": None,
            "latent_l2_threshold": l2_thresh,
            "is_latent_failed": False,
            "is_latent_consistent": True,
            "error": str(e)
        }


# Maintain backward compatibility alias
compute_cycle_consistency = compute_latent_difference


# ==============================================================================
# 2B. CONNECTED COMPONENT DIFFERENCE ANALYSIS (SPATIAL LOCALIZATION)
# ==============================================================================

def compute_connected_component_analysis(
    original_np: np.ndarray,
    imagined_np: np.ndarray,
    diff_threshold: float = 38.0,
    min_ratio: float = 0.35,
    max_total_occupancy: float = 0.65,
    min_diff_pixels: int = 120
) -> Dict[str, Any]:
    """
    Binarizes the absolute difference between the original and imagined observation.
    Uses OpenCV connected components with morphological filtering to verify that visual
    changes are concentrated in a dominant localized object blob (e.g., duckie -> box)
    rather than scattered noisy artifacts or global scene obliteration.

    Returns:
      - is_failed (bool): True if changes are scattered noise or room geometry destroyed.
      - failure_type (str): Specific error code.
      - reason (str): Human-readable diagnosis.
      - max_blob_area (int): Area in pixels of the largest connected difference blob.
      - total_diff_area (int): Total changed pixels exceeding diff_threshold.
      - concentration_ratio (float): max_blob_area / total_diff_area (healthy: >= 0.35).
      - occupancy_ratio (float): max_blob_area / total_frame_pixels.
      - total_occupancy (float): total_diff_area / total_frame_pixels.
      - num_blobs (int): Number of distinct foreground blobs.
      - diff_threshold (float): Difference threshold used.
      - binary_mask (np.ndarray): Cleaned binary difference mask (uint8, 0 or 255).
    """
    orig = original_np.copy()
    imag = imagined_np.copy()
    if orig.max() <= 1.01 and orig.dtype != np.uint8:
        orig = (orig * 255.0).clip(0, 255)
    if imag.max() <= 1.01 and imag.dtype != np.uint8:
        imag = (imag * 255.0).clip(0, 255)
    orig = orig.astype(np.uint8)
    imag = imag.astype(np.uint8)

    target_h = max(orig.shape[0], 200)
    target_w = max(orig.shape[1], 200)
    if orig.shape[0] < target_h or orig.shape[1] < target_w:
        orig_disp = cv2.resize(orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        imag_disp = cv2.resize(imag, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        orig_disp = orig
        imag_disp = imag
        target_h, target_w = orig.shape[:2]

    diff = np.max(np.abs(orig_disp.astype(np.float32) - imag_disp.astype(np.float32)), axis=2)
    binary = (diff >= diff_threshold).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    total_frame_pixels = target_h * target_w
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    import io, base64
    from PIL import Image as PILImage
    mask_pil = PILImage.fromarray(clean_mask)
    buf = io.BytesIO()
    mask_pil.save(buf, format="PNG")
    mask_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    if num_labels <= 1:
        return {
            "is_failed": False,
            "failure_type": None,
            "reason": "No significant differences between original and imagined frames.",
            "max_blob_area": 0,
            "total_diff_area": 0,
            "concentration_ratio": 1.0,
            "occupancy_ratio": 0.0,
            "total_occupancy": 0.0,
            "num_blobs": 0,
            "diff_threshold": diff_threshold,
            "binary_mask": clean_mask,
            "mask_base64": mask_base64
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(np.max(areas))
    total_diff_area = int(np.sum(areas))
    concentration_ratio = float(max_area) / float(total_diff_area) if total_diff_area > 0 else 1.0
    occupancy_ratio = float(max_area) / float(total_frame_pixels)
    total_occupancy = float(total_diff_area) / float(total_frame_pixels)

    if total_diff_area < min_diff_pixels:
        is_failed = False
        failure_type = None
        reason = f"Minimal differences detected ({total_diff_area} px < {min_diff_pixels} px threshold)."
    elif total_occupancy > max_total_occupancy:
        is_failed = True
        failure_type = "VAE_GLOBAL_DISTORTION_FAILURE"
        reason = (
            f"Global distortion failure: {total_occupancy:.1%} of frame altered "
            f"(threshold <= {max_total_occupancy:.1%}; room geometry collapsed)."
        )
    elif concentration_ratio < min_ratio and total_diff_area > (total_frame_pixels * 0.12):
        is_failed = True
        failure_type = "VAE_SCATTERED_ARTIFACTS_FAILURE"
        reason = (
            f"Scattered artifacts failure: Max connected blob ratio is {concentration_ratio:.2f} "
            f"(threshold >= {min_ratio:.2f}); {total_diff_area} altered pixels scattered across {len(areas)} blobs."
        )
    else:
        is_failed = False
        failure_type = None
        reason = (
            f"Localized object transformation verified: dominant blob ratio {concentration_ratio:.2f} >= {min_ratio:.2f} "
            f"covering {occupancy_ratio:.1%} of frame ({len(areas)} blobs total)."
        )

    return {
        "is_failed": is_failed,
        "failure_type": failure_type,
        "reason": reason,
        "max_blob_area": max_area,
        "total_diff_area": total_diff_area,
        "concentration_ratio": round(concentration_ratio, 4),
        "occupancy_ratio": round(occupancy_ratio, 4),
        "total_occupancy": round(total_occupancy, 4),
        "num_blobs": len(areas),
        "diff_threshold": diff_threshold,
        "binary_mask": clean_mask,
        "mask_base64": mask_base64
    }


# ==============================================================================
# 2C. SEMANTIC PRESENCE & SOURCE SUPPRESSION VERIFICATION
# ==============================================================================

def extract_semantic_targets_from_caption(caption: str) -> Dict[str, Any]:
    """
    Parses object descriptions, target colors, and spatial sectors from captions.
    Isolates spatial sectors per object clause so multiple objects in a caption
    (e.g., green ball to the left, blue box to the right) are correctly attributed.
    """
    if not caption:
        return {"targets": [], "target_object": None, "target_color": None, "is_empty": False, "sector": None}

    cap_lower = caption.lower()
    is_empty = any(w in cap_lower for w in ["completely empty", "no objects present", "entirely vacant", "no object"])
    if is_empty:
        return {"targets": [], "target_object": None, "target_color": None, "is_empty": True, "sector": None}

    color_object_patterns = [
        ("blue box", "blue"),
        ("blue cube", "blue"),
        ("yellow duckie", "yellow"),
        ("duckie", "yellow"),
        ("green ball", "green"),
        ("green sphere", "green"),
        ("red medkit", "red"),
        ("red ball", "red"),
        ("red box", "red"),
        ("purple box", "purple"),
        ("blue", "blue"),
        ("yellow", "yellow"),
        ("green", "green"),
        ("red", "red"),
        ("purple", "purple"),
    ]

    # Split into clauses/sentences to isolate per-object directions without breaking decimal numbers (e.g. 34.5, 4.3)
    clauses = re.split(r'(?<!\d)\.+(?!\d)|[\n;]+|\band\b', caption)
    targets = []
    seen_objects = set()

    for c in clauses:
        c_low = c.lower().strip()
        if not c_low:
            continue
        for obj_name, col in color_object_patterns:
            if obj_name in c_low and obj_name not in seen_objects:
                seen_objects.add(obj_name)
                sector = None
                if "left" in c_low:
                    sector = "left"
                elif "right" in c_low:
                    sector = "right"
                elif any(k in c_low for k in ["center", "forward", "ahead", "directly"]):
                    sector = "center"
                dist_m = re.search(r'distance\s*(?:of|is|:)?\s*([\d\.]+)', c_low)
                dist = float(dist_m.group(1)) if dist_m else None
                targets.append({
                    "target_object": obj_name,
                    "target_color": col,
                    "sector": sector,
                    "distance": dist,
                    "clause": c_low
                })
                break

    # Fallback if clause splitting didn't catch an object
    if not targets:
        for obj_name, col in color_object_patterns:
            if obj_name in cap_lower and obj_name not in seen_objects:
                seen_objects.add(obj_name)
                idx = cap_lower.find(obj_name)
                window = cap_lower[max(0, idx - 40):min(len(cap_lower), idx + len(obj_name) + 60)]
                sector = None
                if "left" in window:
                    sector = "left"
                elif "right" in window:
                    sector = "right"
                elif any(k in window for k in ["center", "forward", "ahead", "directly"]):
                    sector = "center"
                dist_m = re.search(r'distance\s*(?:of|is|:)?\s*([\d\.]+)', window)
                dist = float(dist_m.group(1)) if dist_m else None
                targets.append({
                    "target_object": obj_name,
                    "target_color": col,
                    "sector": sector,
                    "distance": dist,
                    "clause": window
                })
                break

    primary_target = targets[0] if targets else {"target_object": None, "target_color": None, "sector": None}

    return {
        "targets": targets,
        "target_object": primary_target.get("target_object"),
        "target_color": primary_target.get("target_color"),
        "sector": primary_target.get("sector"),
        "distance": primary_target.get("distance"),
        "is_empty": is_empty
    }



def get_color_pixel_count(
    img_np: np.ndarray,
    color_name: str,
    env_name: str = "MiniWorld",
    sector: Optional[str] = None
) -> Tuple[int, float, np.ndarray]:
    """
    Returns (pixel_count, sector_fraction, mask).
    For MiniWorld, blue detection excludes the top 20% sky rows to prevent sky false positives.
    """
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return 0, 1.0, np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

    hsv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, w = img_np.shape[:2]

    if color_name == "blue":
        # Saturated blue object
        mask = cv2.inRange(hsv, (95, 60, 50), (135, 255, 255))
        if "MiniWorld" in env_name:
            sky_cutoff = int(h * 0.20)
            mask[:sky_cutoff, :] = 0
    elif color_name == "yellow":
        mask = cv2.inRange(hsv, (15, 70, 70), (38, 255, 255))
    elif color_name == "green":
        if "MiniWorld" in env_name:
            # Strict HSV range for pure green entity (OpenGL [0, 1, 0], H~60, S>=185, V>=110)
            # Distinguishes pure green ball from olive/yellowish grass floor (H~41, S<=180, V<=150)
            mask = cv2.inRange(hsv, (50, 185, 110), (75, 255, 255))
        else:
            mask = cv2.inRange(hsv, (40, 100, 80), (85, 255, 255))
    elif color_name == "red":
        mask1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 60), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif color_name == "purple":
        mask = cv2.inRange(hsv, (125, 50, 50), (165, 255, 255))
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    pixel_count = int(np.sum(mask > 0))
    sector_fraction = 1.0

    if sector and pixel_count > 0:
        ys, xs = np.where(mask > 0)
        if sector == "left":
            sector_fraction = float(np.sum(xs < int(w * 0.65))) / pixel_count
        elif sector == "right":
            sector_fraction = float(np.sum(xs > int(w * 0.35))) / pixel_count
        elif sector == "center":
            sector_fraction = float(np.sum((xs >= int(w * 0.20)) & (xs <= int(w * 0.80)))) / pixel_count

    return pixel_count, sector_fraction, mask


def evaluate_semantic_presence(
    imagined_np: np.ndarray,
    caption: str,
    original_np: Optional[np.ndarray] = None,
    env_name: str = "MiniWorld",
    min_target_pixels: int = 80,
    max_residual_source_pixels: int = 50,
    min_sector_fraction: float = 0.50
) -> Dict[str, Any]:
    """
    Verifies semantic fidelity of the VAE generation against the text caption:
    1. Target presence: If caption specifies an object (e.g. 'blue box'), verifies that
       sufficient target-colored pixels exist in the imagined image.
    2. Spatial sector alignment: If caption indicates 'left' / 'right' / 'center', verifies
       that the target pixels are situated in that sector.
    3. Source suppression: If source object was in original_np (e.g. yellow duckie), verifies
       it was removed/painted over.
    4. Empty scene check: If caption specifies an empty room, verifies no foreground objects exist.
    """
    parsed = extract_semantic_targets_from_caption(caption)
    is_empty = parsed["is_empty"]
    targets = parsed.get("targets", [])
    if not targets and parsed.get("target_object"):
        targets = [{
            "target_object": parsed["target_object"],
            "target_color": parsed["target_color"],
            "sector": parsed["sector"],
            "distance": parsed.get("distance")
        }]

    # Source color verification in original observation
    source_pixels_orig = 0
    source_pixels_recon = 0
    source_col = None
    target_colors = {t["target_color"] for t in targets}
    if original_np is not None:
        if "blue" in target_colors and "MiniWorld" in env_name:
            for candidate_col in ["yellow", "green"]:
                cnt, _, _ = get_color_pixel_count(original_np, candidate_col, env_name)
                if cnt > 0:
                    source_col = candidate_col
                    source_pixels_orig = cnt
                    source_pixels_recon, _, _ = get_color_pixel_count(imagined_np, candidate_col, env_name)
                    break
        elif "purple" in target_colors and "SimplePickup" in env_name:
            source_col = "green"
            source_pixels_orig, _, _ = get_color_pixel_count(original_np, source_col, env_name)
            source_pixels_recon, _, _ = get_color_pixel_count(imagined_np, source_col, env_name)

    target_evals = []
    failing_target_eval = None

    for t in targets:
        t_obj = t["target_object"]
        t_col = t["target_color"]
        t_sec = t["sector"]
        t_dist = t.get("distance")
        t_pixels, t_sec_frac, _ = get_color_pixel_count(imagined_np, t_col, env_name, t_sec)

        # Compute adaptive minimum pixel threshold based on distance and source object scale
        eff_min_pixels = min_target_pixels

        # 1. Scale down requirement for distant objects when distance is present in caption
        if t_dist is not None:
            if t_dist >= 4.0:
                eff_min_pixels = min(eff_min_pixels, 20)
            elif t_dist >= 3.0:
                eff_min_pixels = min(eff_min_pixels, 35)
            elif t_dist >= 2.0:
                eff_min_pixels = min(eff_min_pixels, 55)

        # 2. Scale requirement relative to the source object if source object is present in original image.
        # If the source object occupied few pixels (due to distance or boundary clipping),
        # the translated target object will naturally occupy fewer pixels as well.
        if source_col and original_np is not None and source_pixels_orig < min_target_pixels:
            if source_pixels_orig > 0:
                eff_min_pixels = min(eff_min_pixels, max(15, int(source_pixels_orig * 0.6)))
            elif t_dist is not None and t_dist >= 3.5:
                eff_min_pixels = min(eff_min_pixels, 15)

        t_res = {
            "target_object": t_obj,
            "target_color": t_col,
            "sector": t_sec,
            "distance": t_dist,
            "target_pixels": t_pixels,
            "min_required_pixels": eff_min_pixels,
            "sector_fraction": round(t_sec_frac, 3),
            "is_failed": False,
            "failure_type": None,
            "reason": ""
        }

        if t_pixels < eff_min_pixels:
            t_res["is_failed"] = True
            t_res["failure_type"] = "VAE_SEMANTIC_MISSING_OBJECT"
            sec_spec = f" in '{t_sec}' sector" if t_sec else ""
            t_res["reason"] = (
                f"Semantic presence failure: Caption describes '{t_obj}'{sec_spec}, but only {t_pixels} "
                f"{t_col} pixels found in imagined image (required >= {eff_min_pixels} px, adaptive for distance/scale)."
            )
        elif t_sec and t_sec_frac < min_sector_fraction and t_pixels >= eff_min_pixels:
            t_res["is_failed"] = True
            t_res["failure_type"] = "VAE_SEMANTIC_SECTOR_MISMATCH"
            t_res["reason"] = (
                f"Semantic sector failure: Caption places '{t_obj}' in '{t_sec}' sector, "
                f"but only {t_sec_frac:.1%} of {t_col} pixels are in that sector (threshold >= {min_sector_fraction:.0%})."
            )
        else:
            sec_str = f" in '{t_sec}' sector ({t_sec_frac:.1%})" if t_sec else ""
            t_res["reason"] = f"'{t_obj}' confirmed with {t_pixels} px (required >= {eff_min_pixels} px){sec_str}"

        target_evals.append(t_res)
        if t_res["is_failed"] and failing_target_eval is None:
            failing_target_eval = t_res

    is_failed = False
    failure_type = None
    reason = "Semantic presence verified."

    if failing_target_eval is not None:
        is_failed = True
        failure_type = failing_target_eval["failure_type"]
        reason = failing_target_eval["reason"]
        primary_target_obj = failing_target_eval["target_object"]
        primary_target_col = failing_target_eval["target_color"]
        primary_target_pixels = failing_target_eval["target_pixels"]
        primary_min_req_pixels = failing_target_eval.get("min_required_pixels", min_target_pixels)
        primary_sector = failing_target_eval["sector"]
        primary_sector_frac = failing_target_eval["sector_fraction"]
    elif source_col and source_pixels_orig >= 100 and source_pixels_recon > max_residual_source_pixels:
        is_failed = True
        failure_type = "VAE_SEMANTIC_SOURCE_RESIDUAL"
        reason = (
            f"Source suppression failure: Original '{source_col}' object was not replaced; "
            f"{source_pixels_recon} residual pixels remain in imagined image (threshold <= {max_residual_source_pixels} px)."
        )
        primary_target_obj = targets[0]["target_object"] if targets else None
        primary_target_col = targets[0]["target_color"] if targets else None
        primary_target_pixels = target_evals[0]["target_pixels"] if target_evals else 0
        primary_min_req_pixels = target_evals[0].get("min_required_pixels", min_target_pixels) if target_evals else min_target_pixels
        primary_sector = target_evals[0]["sector"] if target_evals else None
        primary_sector_frac = target_evals[0]["sector_fraction"] if target_evals else 1.0
    elif is_empty:
        # Check all possible foreground objects for hallucinations
        fg_checks = [("blue", "blue box"), ("yellow", "yellow duckie"), ("red", "red medkit"), ("purple", "purple box")]
        for fg_col, fg_name in fg_checks:
            fg_px, _, _ = get_color_pixel_count(imagined_np, fg_col, env_name)
            if fg_px >= min_target_pixels:
                is_failed = True
                failure_type = "VAE_SEMANTIC_HALLUCINATED_OBJECT"
                reason = f"Empty room hallucination: Caption specifies empty room, but {fg_px} {fg_col} pixels ({fg_name}) are present."
                primary_target_obj = fg_name
                primary_target_col = fg_col
                primary_target_pixels = fg_px
                primary_min_req_pixels = min_target_pixels
                primary_sector = None
                primary_sector_frac = 1.0
                break
        if not is_failed:
            reason = "Semantic verified: Empty scene confirmed."
            primary_target_obj, primary_target_col, primary_target_pixels, primary_min_req_pixels, primary_sector, primary_sector_frac = None, None, 0, min_target_pixels, None, 1.0
    elif target_evals:
        reason = "Semantic verified: " + "; ".join([t["reason"] for t in target_evals])
        primary_target_obj = target_evals[0]["target_object"]
        primary_target_col = target_evals[0]["target_color"]
        primary_target_pixels = target_evals[0]["target_pixels"]
        primary_min_req_pixels = target_evals[0].get("min_required_pixels", min_target_pixels)
        primary_sector = target_evals[0]["sector"]
        primary_sector_frac = target_evals[0]["sector_fraction"]
    else:
        primary_target_obj, primary_target_col, primary_target_pixels, primary_min_req_pixels, primary_sector, primary_sector_frac = None, None, 0, min_target_pixels, None, 1.0

    return {
        "is_failed": is_failed,
        "failure_type": failure_type,
        "reason": reason,
        "target_object": primary_target_obj,
        "target_color": primary_target_col,
        "target_pixels": primary_target_pixels,
        "min_required_pixels": primary_min_req_pixels,
        "sector": primary_sector,
        "sector_fraction": round(primary_sector_frac, 3) if primary_sector_frac is not None else 1.0,
        "source_object_removed": bool(source_pixels_recon <= max_residual_source_pixels) if source_col else True,
        "residual_source_pixels": source_pixels_recon if source_col else 0,
        "all_targets": target_evals
    }


# ==============================================================================
# 3. COMPREHENSIVE VAE ERROR EVALUATOR (DEGENERACY & CONNECTED COMPONENTS)
# ==============================================================================

def evaluate_vae_quality(
    original_np: np.ndarray,
    imagined_np: np.ndarray,
    vision_model=None,
    original_tensor: Optional[torch.Tensor] = None,
    imagined_tensor: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    env_name: str = "MiniWorld",
    grid_fail_threshold: Optional[int] = None,
    cell_diff_threshold: float = 15.0,
    latent_l2_threshold: Optional[float] = None,
    diff_threshold: float = 38.0,
    min_ratio: float = 0.35,
    max_total_occupancy: float = 0.65,
    min_laplacian_ratio: float = 0.50,
    min_orig_blur_var: float = 25.0,
    caption: str = ""
) -> Dict[str, Any]:
    """
    Evaluates VAE generation quality based on:
      1. Numerical sanity (NaN / Inf)
      2. Mode collapse and edge blurriness (Std Dev < 3.0, dynamic range < 15, or Laplacian variance < 20.0)
      3. Connected component spatial localization (Dominant blob ratio >= min_ratio, total occupancy <= max_total_occupancy)
      4. Semantic presence & source suppression (Required target color pixels present, source object suppressed)
      5. Latent difference telemetry
    """
    metrics: Dict[str, Any] = {}

    # 1. Numerical checks
    if np.isnan(imagined_np).any() or np.isinf(imagined_np).any():
        return {
            "is_valid": False,
            "error_type": "VAE_NAN_INF",
            "failed_component": "NUMERICAL_SANITY",
            "reason": "Imagined image contains NaN or Inf values.",
            "metrics": metrics
        }

    # For MiniGrid-based environments (SimplePickup, MiniGrid, MiniGridRelational):
    # Use Grid Difference Telemetry ONLY, bypassing continuous 3D heuristics
    is_minigrid_env = any(k in env_name.lower() for k in ["minigrid", "simplepickup"])
    if is_minigrid_env:
        eff_grid_fail_threshold = grid_fail_threshold if grid_fail_threshold is not None else 5
        grid_res = compute_grid_differences(
            original_np=original_np,
            imagined_np=imagined_np,
            env_name=env_name,
            cell_diff_threshold=cell_diff_threshold,
            fail_threshold=eff_grid_fail_threshold
        )
        metrics["grid_analysis"] = grid_res
        metrics["differing_grid_count"] = grid_res["differing_grid_count"]
        metrics["grid_fail_threshold"] = grid_res["fail_threshold"]
        metrics["is_grid_failed"] = grid_res["is_grid_failed"]

        if grid_res["is_grid_failed"]:
            return {
                "is_valid": False,
                "error_type": "VAE_GRID_DIFFERENCE_FAILURE",
                "failed_component": "GRID_DIFFERENCE_TELEMETRY",
                "reason": (
                    f"Grid difference failure: {grid_res['differing_grid_count']} cells differ "
                    f"(threshold < {grid_res['fail_threshold']})."
                ),
                "metrics": metrics
            }
        else:
            return {
                "is_valid": True,
                "error_type": None,
                "failed_component": None,
                "reason": f"Grid difference verified: only {grid_res['differing_grid_count']} cells differ (< {grid_res['fail_threshold']}).",
                "metrics": metrics
            }

    # 2. Dynamic range, variance, and blur checks
    img_std = float(imagined_np.std())
    metrics["imagined_std"] = round(img_std, 2)
    dyn_range = float(imagined_np.max() - imagined_np.min())
    metrics["dynamic_range"] = round(dyn_range, 2)

    if imagined_np.ndim == 3 and imagined_np.shape[2] == 3:
        imag_gray = cv2.cvtColor(imagined_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        imag_gray = imagined_np.astype(np.uint8)
    blur_var = float(cv2.Laplacian(imag_gray, cv2.CV_64F).var())
    metrics["blur_var"] = round(blur_var, 2)

    if imagined_np.ndim >= 2 and (img_std < 3.0 or dyn_range < 15.0):
        return {
            "is_valid": False,
            "error_type": "VAE_MODE_COLLAPSE",
            "failed_component": "IMAGE_DEGENERACY",
            "reason": f"Mode collapse / washed-out canvas (std={img_std:.2f}, range={dyn_range:.1f}).",
            "metrics": metrics
        }

    # Relative Laplacian drop check: compare edge variance against original observation
    if original_np is not None:
        if original_np.ndim == 3 and original_np.shape[2] == 3:
            orig_gray = cv2.cvtColor(original_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original_np.astype(np.uint8)
        orig_blur_var = float(cv2.Laplacian(orig_gray, cv2.CV_64F).var())
        metrics["orig_blur_var"] = round(orig_blur_var, 2)
        lap_ratio = blur_var / max(orig_blur_var, 1e-5)
        metrics["laplacian_ratio"] = round(lap_ratio, 3)

        if orig_blur_var >= min_orig_blur_var and lap_ratio < min_laplacian_ratio:
            return {
                "is_valid": False,
                "error_type": "VAE_BLUR_COLLAPSE",
                "failed_component": "IMAGE_DEGENERACY",
                "reason": (
                    f"Severe edge loss / relative blur drop: Laplacian variance dropped from "
                    f"{orig_blur_var:.1f} to {blur_var:.1f} (ratio={lap_ratio:.1%} < {min_laplacian_ratio:.0%})."
                ),
                "metrics": metrics
            }

    # Fallback absolute blur check for cases where original is unavailable or degenerate
    if blur_var < 20.0 and img_std < 10.0:
        return {
            "is_valid": False,
            "error_type": "VAE_BLUR_COLLAPSE",
            "failed_component": "IMAGE_DEGENERACY",
            "reason": f"Severe blur / edge loss (Laplacian variance={blur_var:.1f} < 20.0).",
            "metrics": metrics
        }

    # 3. Connected Component Difference Analysis (Spatial localization)
    cc_res = compute_connected_component_analysis(
        original_np=original_np,
        imagined_np=imagined_np,
        diff_threshold=diff_threshold,
        min_ratio=min_ratio,
        max_total_occupancy=max_total_occupancy
    )
    metrics["connected_components"] = cc_res
    metrics["binary_mask"] = cc_res.get("binary_mask")
    # 4. Semantic Presence & Source Suppression Verification
    sem_res = evaluate_semantic_presence(
        imagined_np=imagined_np,
        caption=caption,
        original_np=original_np,
        env_name=env_name
    )
    metrics["semantic_presence"] = sem_res

    # 5. Grid difference telemetry (recorded for monitoring, not used for failure blocking)
    eff_grid_fail_threshold = grid_fail_threshold if grid_fail_threshold is not None else (10 if "MiniWorld" in env_name else 5)
    grid_res = compute_grid_differences(
        original_np=original_np,
        imagined_np=imagined_np,
        env_name=env_name,
        cell_diff_threshold=cell_diff_threshold,
        fail_threshold=eff_grid_fail_threshold
    )
    metrics["grid_analysis"] = grid_res
    metrics["differing_grid_count"] = grid_res["differing_grid_count"]
    metrics["grid_fail_threshold"] = grid_res["fail_threshold"]
    metrics["is_grid_failed"] = grid_res["is_grid_failed"]

    # 6. Latent difference analysis (recorded for monitoring)
    latent_res = compute_latent_difference(
        vision_model=vision_model,
        original_tensor=original_tensor,
        imagined_tensor=imagined_tensor,
        original_np=original_np,
        imagined_np=imagined_np,
        device=device,
        latent_l2_threshold=latent_l2_threshold,
        env_name=env_name
    )
    metrics["latent_analysis"] = latent_res
    metrics["latent_l2"] = latent_res.get("latent_l2")
    metrics["latent_cosine"] = latent_res.get("latent_cosine")
    metrics["latent_l2_threshold"] = latent_res.get("latent_l2_threshold")
    metrics["is_latent_failed"] = latent_res.get("is_latent_failed", False)
    metrics["cycle_l2"] = metrics["latent_l2"]
    metrics["cycle_cosine"] = metrics["latent_cosine"]
    metrics["cycle_l2_threshold"] = metrics["latent_l2_threshold"]
    metrics["is_cycle_consistent"] = not metrics["is_latent_failed"]

    # --- Error Threshold Evaluation ---
    if sem_res.get("is_failed"):
        return {
            "is_valid": False,
            "error_type": sem_res["failure_type"],
            "failed_component": "SEMANTIC_PRESENCE",
            "reason": sem_res["reason"],
            "metrics": metrics
        }

    if cc_res["is_failed"]:
        return {
            "is_valid": False,
            "error_type": cc_res["failure_type"],
            "failed_component": "CONNECTED_COMPONENT_ANALYSIS",
            "reason": cc_res["reason"],
            "metrics": metrics
        }

    return {
        "is_valid": True,
        "error_type": None,
        "failed_component": None,
        "reason": f"{cc_res['reason']} | {sem_res['reason']}",
        "metrics": metrics
    }


# ==============================================================================
# 4. RULE-BASED LLM MAPPING EVALUATOR
# ==============================================================================

def _extract_coords(text: str, entity_name: str) -> Optional[Tuple[int, int]]:
    """
    Extracts (x, y) coordinates for a given entity pattern from a text string.
    Example: 'The yellow box is at (4, 3)' -> (4, 3)
    """
    pattern = rf"{re.escape(entity_name)}\s+(?:is\s+)?at\s+\((\d+),\s*(\d+)\)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def evaluate_llm_mapping(
    env_name: str,
    task_mode: str,
    input_description: str,
    llm_reply_json: Any,
    raw_reply: Optional[str] = None,
    grid_size: int = 8
) -> Dict[str, Any]:
    """
    Validates whether the LLM output correctly remapped the scene according to:
      1. Structural JSON validity & non-empty content
      2. Object visibility: If an object is visible, imagine must be True and description must not be empty
      3. Environment and target-specific rule verification:
         - SimplePickup:
             target1: purple box -> red ball
             target2: purple box -> red ball and blue wall -> grey wall
             target3: green ball -> red ball, red ball -> green ball
         - MiniWorld:
             target1: duckie -> box
             target2: duckie -> box and wood/brick_wall -> grass/concrete
             target3: ball -> box, box -> ball
         - PickEnv:
             target: heavy circle -> heavy square, light square -> light circle
         - MiniGridRelational:
             target1: red ball -> blue ball, yellow box -> green target (adjacent)
             target2: red ball -> blue ball, yellow box -> green target (symmetric opposite)
             target3: red ball -> blue ball, yellow box -> green target (Manhattan-2 nearest)
             target4: red ball -> blue ball, yellow box -> green target (sym north-2)
             target5: sequential pairs (red->blue & yellow->green, purple->blue & grey->green)
    """
    # 1. Structural extraction of remapped description (No imagine flag check)
    if isinstance(llm_reply_json, dict):
        out_desc = str(llm_reply_json.get("description", "")).strip()
    elif isinstance(llm_reply_json, str):
        out_desc = llm_reply_json.strip()
    else:
        return {
            "is_valid": False,
            "error_type": "LLM_SYNTAX_ERROR",
            "reason": f"LLM reply could not be parsed as a dict or text (got {type(llm_reply_json)})."
        }

    in_text = (input_description or "").strip()
    out_lower = out_desc.lower()
    in_lower = in_text.lower()

    if not out_desc:
        return {
            "is_valid": False,
            "error_type": "LLM_EMPTY_DESCRIPTION",
            "reason": "LLM produced an empty or missing description."
        }

    # ==========================================================================
    # 2. DOMAIN-SPECIFIC RULE VERIFICATION
    # ==========================================================================

    # --- SimplePickup ---
    if env_name == "SimplePickup" or "simplepickup" in env_name.lower():
        if task_mode == "target1":
            # Rule: purple box -> red ball
            if "purple box" in in_lower:
                if "purple box" in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "SimplePickup target1: 'purple box' must not remain in the imagined description."
                    }
                if "red ball" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "SimplePickup target1: 'purple box' must be remapped to 'red ball'."
                    }

        elif task_mode == "target2":
            # Rule: purple box -> red ball AND blue wall -> grey wall
            if "purple box" in in_lower:
                if "purple box" in out_lower or "red ball" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "SimplePickup target2: 'purple box' must be remapped to 'red ball'."
                    }
            if "blue wall" in in_lower or "blue" in in_lower and "wall" in in_lower:
                if "blue wall" in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "SimplePickup target2: 'blue wall' must not remain in the imagined description."
                    }
                if "grey wall" not in out_lower and "gray wall" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "SimplePickup target2: 'blue wall' must be remapped to 'grey wall'."
                    }

        elif task_mode == "target3":
            # Rule: green ball -> red ball, red ball -> green ball
            if "green ball" in in_lower and "red ball" not in in_lower:
                if "green ball" in out_lower or "red ball" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "SimplePickup target3: 'green ball' must be remapped to 'red ball'."
                    }
            elif "green ball" in in_lower and "red ball" in in_lower:
                # Both present: target green ball must become red ball, original red ball must become green or be removed
                coords_green = _extract_coords(in_text, "green ball")
                coords_red_out = _extract_coords(out_desc, "red ball")
                if coords_green and coords_red_out and coords_green != coords_red_out:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": f"SimplePickup target3: target 'green ball' at {coords_green} was not remapped to 'red ball' at the same coordinates (got {coords_red_out})."
                    }

    # --- MiniWorld ---
    elif env_name.startswith("MiniWorld"):
        if task_mode == "target1":
            # Rule: duckie -> box
            if "duckie" in in_lower:
                if "duckie" in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "MiniWorld target1: 'duckie' must not remain in the imagined description."
                    }
                if "box" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "MiniWorld target1: 'duckie' must be remapped to 'box'."
                    }

        elif task_mode == "target2":
            # Rule: duckie -> box AND wood/brick_wall -> grass/concrete
            if "duckie" in in_lower:
                if "duckie" in out_lower or "box" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "MiniWorld target2: 'duckie' must be remapped to 'box'."
                    }
            has_wood_or_brick = "wood" in in_lower or "brick" in in_lower
            if has_wood_or_brick:
                if "wood" in out_lower or "brick" in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "MiniWorld target2: 'wood floor' / 'brick wall' must be transformed to grass/concrete/gray."
                    }
                if not ("grass" in out_lower or "concrete" in out_lower or "gray" in out_lower or "grey" in out_lower):
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "MiniWorld target2: environment textures must be mapped to grass floor and gray/concrete walls."
                    }

        elif task_mode == "target3":
            # Rule: ball -> box, box -> ball
            if "ball" in in_lower and "box" not in in_lower:
                if "ball" in out_lower or "box" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "MiniWorld target3: target 'ball' must be remapped to 'box'."
                    }

    # --- PickEnv ---
    elif env_name == "PickEnv" or "pickenv" in env_name.lower():
        # Rule: heavy circle -> heavy square, light square -> light circle
        if "heavy circle" in in_lower or ("heavy" in in_lower and "circle" in in_lower):
            if "heavy square" not in out_lower:
                return {
                    "is_valid": False,
                    "error_type": "LLM_RULE_VIOLATION",
                    "reason": "PickEnv: 'heavy circle' must be remapped to 'heavy square'."
                }
        if "light square" in in_lower or ("light" in in_lower and "square" in in_lower):
            if "light circle" not in out_lower:
                return {
                    "is_valid": False,
                    "error_type": "LLM_RULE_VIOLATION",
                    "reason": "PickEnv: 'light square' must be remapped to 'light circle'."
                }

    # --- MiniGridRelational ---
    elif env_name == "MiniGridRelational" or "relational" in env_name.lower():
        if task_mode in ["target1", "target2", "target3", "target4"]:
            # Pick mapping rule: red ball -> blue ball
            if "red ball" in in_lower:
                if "red ball" in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": f"MiniGridRelational {task_mode}: 'red ball' must not remain in the imagined description."
                    }
                if "blue ball" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": f"MiniGridRelational {task_mode}: 'red ball' must be remapped to 'blue ball'."
                    }

            # Drop mapping rule: yellow box must NOT remain
            if "yellow box" in in_lower:
                if "yellow box" in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": f"MiniGridRelational {task_mode}: 'yellow box' must not remain in output; it must be mapped to 'green target'."
                    }
                if "green target" not in out_lower:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": f"MiniGridRelational {task_mode}: must introduce 'green target' for drop goal."
                    }

                # Coordinate verification if both landmark and green target coordinates can be parsed
                box_pos = _extract_coords(in_text, "yellow box")
                tgt_pos = _extract_coords(out_desc, "green target")

                if box_pos and tgt_pos:
                    bx, by = box_pos
                    tx, ty = tgt_pos

                    if task_mode == "target1":
                        # Cardinal neighbors (Manhattan distance == 1)
                        if abs(tx - bx) + abs(ty - by) != 1:
                            return {
                                "is_valid": False,
                                "error_type": "LLM_RULE_VIOLATION",
                                "reason": f"MiniGridRelational target1: 'green target' at ({tx}, {ty}) is not immediately adjacent to yellow box at ({bx}, {by})."
                            }

                    elif task_mode == "target2":
                        # Symmetric opposite: (W-1-bx, H-1-by)
                        exp_x = (grid_size - 1) - bx
                        exp_y = (grid_size - 1) - by
                        if (tx, ty) != (exp_x, exp_y):
                            return {
                                "is_valid": False,
                                "error_type": "LLM_RULE_VIOLATION",
                                "reason": f"MiniGridRelational target2: 'green target' at ({tx}, {ty}) does not match symmetric opposite ({exp_x}, {exp_y}) of yellow box ({bx}, {by})."
                            }

                    elif task_mode == "target3":
                        # Manhattan distance 2
                        if abs(tx - bx) + abs(ty - by) != 2:
                            return {
                                "is_valid": False,
                                "error_type": "LLM_RULE_VIOLATION",
                                "reason": f"MiniGridRelational target3: 'green target' at ({tx}, {ty}) is not at Manhattan distance 2 from yellow box ({bx}, {by})."
                            }

                    elif task_mode == "target4":
                        # (W-1-bx, H-1-(by-2))
                        exp_x = (grid_size - 1) - bx
                        exp_y = (grid_size - 1) - (by - 2)
                        if (tx, ty) != (exp_x, exp_y):
                            return {
                                "is_valid": False,
                                "error_type": "LLM_RULE_VIOLATION",
                                "reason": f"MiniGridRelational target4: 'green target' at ({tx}, {ty}) does not match formula ({exp_x}, {exp_y}) for yellow box ({bx}, {by})."
                            }

        elif task_mode == "target5":
            # Two pairs: red ball -> yellow target, purple ball -> grey target
            # Remapped subtask should map the active ball to blue ball and matching target to green target
            if "red ball" in in_lower and "yellow target" in in_lower:
                if "blue ball" in out_lower and "green target" in out_lower:
                    pass
                elif "purple ball" in in_lower and "grey target" in in_lower and "blue ball" in out_lower and "green target" in out_lower:
                    pass
                else:
                    return {
                        "is_valid": False,
                        "error_type": "LLM_RULE_VIOLATION",
                        "reason": "MiniGridRelational target5: Active pair must be remapped to 'blue ball' and 'green target'."
                    }

    return {
        "is_valid": True,
        "error_type": None,
        "reason": f"LLM mapping passed all rule-based criteria for {env_name} ({task_mode})."
    }


# ==============================================================================
# 5. PIE CHART GENERATOR (ACROSS SEEDS)
# ==============================================================================

def generate_pie_chart(
    diagnostics_summary: Dict[str, Any],
    output_path: str
) -> str:
    """
    Renders and saves a clean, publication-ready pie chart of error breakdown:
      - LLM Mapping Failure
      - VAE Artifact Failure
      - Policy Execution Failure
    """
    total_eps = diagnostics_summary.get("total_episodes", 0)
    successes = diagnostics_summary.get("success_count", 0)
    fails = diagnostics_summary.get("fail_count", 0)
    breakdown = diagnostics_summary.get("failure_breakdown", {})

    llm_errs = breakdown.get("LLM_MAPPING_FAILURE", 0)
    vae_errs = breakdown.get("VAE_ARTIFACT_FAILURE", 0)
    pol_errs = breakdown.get("POLICY_NAVIGATION_FAILURE", breakdown.get("POLICY_EXECUTION_FAILURE", 0))

    env_name = diagnostics_summary.get("env_name", "Environment")
    task_mode = diagnostics_summary.get("task_mode", "target")

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))

    labels = []
    sizes = []
    colors = []
    explode = []

    palette = {
        "LLM Mapping Error": "#ff6b6b",
        "VAE Artifact Error": "#ffa94d",
        "Policy Execution Error": "#4dabf7",
        "Success": "#51cf66"
    }

    if llm_errs > 0:
        labels.append(f"LLM Mapping Error\n({llm_errs})")
        sizes.append(llm_errs)
        colors.append(palette["LLM Mapping Error"])
        explode.append(0.05)

    if vae_errs > 0:
        labels.append(f"VAE Artifact Error\n({vae_errs})")
        sizes.append(vae_errs)
        colors.append(palette["VAE Artifact Error"])
        explode.append(0.05)

    if pol_errs > 0:
        labels.append(f"Policy Execution Error\n({pol_errs})")
        sizes.append(pol_errs)
        colors.append(palette["Policy Execution Error"])
        explode.append(0.05)

    if successes > 0:
        labels.append(f"Success\n({successes})")
        sizes.append(successes)
        colors.append(palette["Success"])
        explode.append(0.0)

    if not sizes:
        sizes = [1]
        labels = ["No Episodes"]
        colors = ["#adb5bd"]
        explode = [0]

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops=dict(color="#ffffff", fontsize=11, weight="bold")
    )

    for text in texts:
        text.set_color("#212529")
        text.set_fontsize(11)

    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontsize(10)

    ax.set_title(
        f"Imagination Error Analysis: {env_name} ({task_mode})\nTotal Episodes: {total_eps} | Success: {successes} | Failures: {fails}",
        fontsize=13,
        weight="bold",
        pad=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================================================================
# 9. ERROR SAMPLE LOGGING & SIDE-BY-SIDE VISUALIZATION
# ==============================================================================

def log_llm_error_sample(
    env_name: str,
    task_mode: str,
    seed: int,
    episode: int,
    step: int,
    target_description: str,
    mapped_description: str,
    raw_reply: str,
    reason: str,
    called_model: Optional[str] = None,
    output_dir: str = "Results/diagnostics/error_samples"
) -> Dict[str, Any]:
    """
    Logs an LLM mapping error with target description, mapped description, reason, and called LLM model.
    Appends to {output_dir}/llm_errors.jsonl and prints to console.
    """
    os.makedirs(output_dir, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_name": env_name,
        "task_mode": task_mode,
        "seed": seed,
        "episode": episode,
        "step": step,
        "called_model": called_model,
        "target_description": target_description,
        "mapped_description": mapped_description,
        "raw_reply": raw_reply,
        "reason": reason
    }
    log_file = os.path.join(output_dir, "llm_errors.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\n[DIAGNOSTIC LLM ERROR] {env_name} ({task_mode}) seed {seed} ep {episode} step {step}", flush=True)
    if called_model:
        print(f"  Called LLM:         {called_model}", flush=True)
    print(f"  Target Description: \"{target_description}\"", flush=True)
    print(f"  Mapped Description: \"{mapped_description}\"", flush=True)
    print(f"  Error Reason:       {reason}\n", flush=True)
    return record


def save_vae_error_sample(
    original_np: np.ndarray,
    imagined_np: np.ndarray,
    caption: str,
    reason: str,
    error_type: str,
    env_name: str,
    task_mode: str,
    seed: int,
    episode: int,
    step: int,
    failed_component: Optional[str] = None,
    binary_mask: Optional[np.ndarray] = None,
    output_dir: str = "Results/diagnostics/error_samples"
) -> str:
    """
    Saves:
      1. Combined side-by-side PNG (original + imagined) WITHOUT caption text, in output_dir.
      2. Individual original image, in output_dir/individual/original/.
      3. Individual reconstructed image, in output_dir/individual/reconstructed/.
      4. Binary difference mask from connected component analysis, in output_dir/individual/masks/.
    Logs metadata with failed_component to {output_dir}/vae_errors.jsonl and console.
    """
    os.makedirs(output_dir, exist_ok=True)
    indiv_dir = os.path.join(output_dir, "individual")
    orig_dir = os.path.join(indiv_dir, "original")
    recon_dir = os.path.join(indiv_dir, "reconstructed")
    mask_dir = os.path.join(indiv_dir, "masks")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    orig = original_np.copy()
    imag = imagined_np.copy()
    if orig.max() <= 1.01 and orig.dtype != np.uint8:
        orig = (orig * 255.0).clip(0, 255)
    if imag.max() <= 1.01 and imag.dtype != np.uint8:
        imag = (imag * 255.0).clip(0, 255)
    orig = orig.astype(np.uint8)
    imag = imag.astype(np.uint8)

    # Upscale to at least 200px height for clear visual inspection
    target_h = max(orig.shape[0], 200)
    target_w = max(orig.shape[1], 200)
    if orig.shape[0] < target_h or orig.shape[1] < target_w:
        orig_disp = cv2.resize(orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        imag_disp = cv2.resize(imag, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        orig_disp = orig
        imag_disp = imag
        target_h, target_w = orig.shape[:2]

    # Binary mask extraction
    if binary_mask is not None:
        mask_disp = binary_mask.copy()
        if mask_disp.shape[:2] != (target_h, target_w):
            mask_disp = cv2.resize(mask_disp, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        diff = np.max(np.abs(orig_disp.astype(np.float32) - imag_disp.astype(np.float32)), axis=2)
        binary = (diff >= 30.0).astype(np.uint8) * 255
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_disp = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        mask_disp = cv2.morphologyEx(mask_disp, cv2.MORPH_CLOSE, kernel_close)

    # 1. Combined image WITHOUT caption text / headers / footers
    divider_w = 8
    total_w = target_w * 2 + divider_w
    composite = np.zeros((target_h, total_w, 3), dtype=np.uint8)
    composite[:, :target_w] = orig_disp
    composite[:, target_w + divider_w:] = imag_disp
    composite[:, target_w:target_w + divider_w] = (60, 60, 80)

    core_tag = f"{env_name}_{task_mode}_seed_{seed}_ep{episode}_step{step}"
    sample_filename = f"vae_{core_tag}.png"
    sample_path = os.path.join(output_dir, sample_filename)
    Image.fromarray(composite).save(sample_path)

    # 2. Individual images
    orig_path = os.path.join(orig_dir, f"original_{core_tag}.png")
    recon_path = os.path.join(recon_dir, f"reconstructed_{core_tag}.png")
    mask_path = os.path.join(mask_dir, f"mask_{core_tag}.png")

    Image.fromarray(orig_disp).save(orig_path)
    Image.fromarray(imag_disp).save(recon_path)
    Image.fromarray(mask_disp).save(mask_path)

    # Determine failed component if not explicitly provided
    if failed_component:
        eff_component = failed_component
    elif "COLLAPSE" in error_type or "BLUR" in error_type:
        eff_component = "IMAGE_DEGENERACY"
    elif "DISTORTION" in error_type or "ARTIFACT" in error_type:
        eff_component = "CONNECTED_COMPONENT_ANALYSIS"
    elif "GRID" in error_type:
        eff_component = "GRID_DIFFERENCE_TELEMETRY"
    elif "NAN" in error_type:
        eff_component = "NUMERICAL_SANITY"
    else:
        eff_component = "VAE_ANALYSIS"

    rec = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_name": env_name,
        "task_mode": task_mode,
        "seed": seed,
        "episode": episode,
        "step": step,
        "caption": caption,
        "failed_component": eff_component,
        "error_type": error_type,
        "reason": reason,
        "sample_image": sample_path,
        "original_image": orig_path,
        "reconstructed_image": recon_path,
        "binary_mask_image": mask_path
    }
    with open(os.path.join(output_dir, "vae_errors.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"\n[DIAGNOSTIC VAE ERROR] {env_name} ({task_mode}) seed {seed} ep {episode} step {step}", flush=True)
    print(f"  Failed Component:   {eff_component}", flush=True)
    print(f"  Error Type:         {error_type}", flush=True)
    print(f"  Error Reason:       {reason}", flush=True)
    print(f"  Caption:            \"{caption}\"", flush=True)
    print(f"  Combined (No Text): {sample_path}", flush=True)
    print(f"  Individual Images:  {orig_path}, {recon_path}", flush=True)
    print(f"  Binary Mask:        {mask_path}\n", flush=True)
    return sample_path
