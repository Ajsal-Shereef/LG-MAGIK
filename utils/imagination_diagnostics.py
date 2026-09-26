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
    Marks is_latent_failed = True if normalized L2 distance exceeds threshold.
    """
    if latent_l2_threshold is not None:
        l2_thresh = latent_l2_threshold
    elif env_name and "MiniWorld" in env_name:
        l2_thresh = 0.23
    else:
        l2_thresh = 0.25

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
# 3. COMPREHENSIVE VAE ERROR EVALUATOR (GRID & LATENT DIFFERENCE)
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
    latent_l2_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates VAE generation quality based on:
      1. Numerical sanity (NaN / Inf)
      2. Mode collapse (Std Dev < 3.0 or dynamic range < 15)
      3. Environment grid difference:
         Divides image into grid cells (16px for MiniGridRelational, 8px for SimplePickup, 10px for MiniWorld).
         If >= fail threshold (10 for MiniWorld, 5 for others) grid cells differ significantly, VAE fails (VAE_GRID_DIFFERENCE_FAILURE).
      4. Latent difference:
         Encodes original and imagined images to latent space.
         If latent L2 distance >= threshold, VAE fails (VAE_LATENT_DIFFERENCE_FAILURE).
    """
    metrics: Dict[str, Any] = {}

    if grid_fail_threshold is not None:
        eff_grid_fail_threshold = grid_fail_threshold
    elif env_name and "MiniWorld" in env_name:
        eff_grid_fail_threshold = 10
    else:
        eff_grid_fail_threshold = 5

    # 1. Numerical checks
    if np.isnan(imagined_np).any() or np.isinf(imagined_np).any():
        return {
            "is_valid": False,
            "error_type": "VAE_NAN_INF",
            "reason": "Imagined image contains NaN or Inf values.",
            "metrics": metrics
        }

    # 2. Dynamic range and variance (basic sanity)
    img_std = float(imagined_np.std())
    metrics["imagined_std"] = round(img_std, 2)
    dyn_range = float(imagined_np.max() - imagined_np.min())
    metrics["dynamic_range"] = round(dyn_range, 2)

    # 3. Grid difference analysis
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
    metrics["max_grid_diff"] = grid_res["max_grid_diff"]
    metrics["mean_grid_diff"] = grid_res["mean_grid_diff"]
    metrics["grid_shape"] = grid_res["grid_shape"]
    metrics["cell_size"] = grid_res["cell_size"]

    # 4. Latent difference analysis
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

    # Retain cycle aliases for frontend compatibility if needed
    metrics["cycle_l2"] = metrics["latent_l2"]
    metrics["cycle_cosine"] = metrics["latent_cosine"]
    metrics["cycle_l2_threshold"] = metrics["latent_l2_threshold"]
    metrics["is_cycle_consistent"] = not metrics["is_latent_failed"]

    # --- Error Threshold Evaluations (in priority order) ---
    if imagined_np.ndim >= 2 and (img_std < 3.0 or dyn_range < 15.0):
        return {
            "is_valid": False,
            "error_type": "VAE_MODE_COLLAPSE",
            "reason": f"Mode collapse / washed-out canvas (std={img_std:.2f}, range={dyn_range:.1f}).",
            "metrics": metrics
        }

    if grid_res["is_grid_failed"]:
        return {
            "is_valid": False,
            "error_type": "VAE_GRID_DIFFERENCE_FAILURE",
            "reason": (
                f"Grid difference failure: {grid_res['differing_grid_count']} grid cells significantly differ "
                f"(failure threshold is >= {eff_grid_fail_threshold} cells; cell MAE threshold is {cell_diff_threshold})."
            ),
            "metrics": metrics
        }

    if latent_res.get("is_latent_failed"):
        l2_thresh = latent_res.get("latent_l2_threshold", 0.23 if (env_name and "MiniWorld" in env_name) else 0.25)
        return {
            "is_valid": False,
            "error_type": "VAE_LATENT_DIFFERENCE_FAILURE",
            "reason": (
                f"Latent difference failure: Latent L2 distance is {latent_res['latent_l2']} "
                f"(threshold >= {l2_thresh})."
            ),
            "metrics": metrics
        }

    return {
        "is_valid": True,
        "error_type": None,
        "reason": (
            f"Imagined observation passed: only {grid_res['differing_grid_count']} grid cells differ "
            f"(< {eff_grid_fail_threshold}) and latent L2 distance is {latent_res['latent_l2']} (< {latent_res['latent_l2_threshold']})."
        ),
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
    output_dir: str = "Results/diagnostics/error_samples"
) -> str:
    """
    Saves a labeled side-by-side PNG image showing the original observation,
    the imagined VAE generation, the caption used, and the error reason.
    Appends metadata to {output_dir}/vae_errors.jsonl and returns the PNG path.
    """
    os.makedirs(output_dir, exist_ok=True)

    orig = original_np.copy()
    imag = imagined_np.copy()
    if orig.max() <= 1.01 and orig.dtype != np.uint8:
        orig = (orig * 255.0).clip(0, 255)
    if imag.max() <= 1.01 and imag.dtype != np.uint8:
        imag = (imag * 255.0).clip(0, 255)
    orig = orig.astype(np.uint8)
    imag = imag.astype(np.uint8)

    # Upscale to at least 200px height for clear inspection
    target_h = max(orig.shape[0], 200)
    target_w = max(orig.shape[1], 200)
    if orig.shape[0] < target_h or orig.shape[1] < target_w:
        orig_disp = cv2.resize(orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        imag_disp = cv2.resize(imag, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        orig_disp = orig
        imag_disp = imag
        target_h, target_w = orig.shape[:2]

    header_h = 32
    footer_h = 68
    divider_w = 8
    total_w = target_w * 2 + divider_w
    total_h = target_h + header_h + footer_h

    composite = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    composite[:] = (24, 24, 37)  # Dark theme

    # Place observations
    composite[header_h:header_h + target_h, :target_w] = orig_disp
    composite[header_h:header_h + target_h, target_w + divider_w:] = imag_disp
    composite[header_h:header_h + target_h, target_w:target_w + divider_w] = (60, 60, 80)

    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(composite)
    draw = ImageDraw.Draw(pil_img)

    # Header labels
    draw.text((12, 8), "Original Observation", fill=(140, 200, 255))
    draw.text((target_w + divider_w + 12, 8), "Imagined Observation (VAE)", fill=(255, 180, 180))

    # Footer text
    foot_y = header_h + target_h + 8
    draw.text((12, foot_y), f"Env: {env_name} | Target: {task_mode} | Seed: {seed} | Ep: {episode} | Step: {step}", fill=(180, 180, 200))
    cap_text = f"Caption: \"{caption}\""
    if len(cap_text) > 100:
        cap_text = cap_text[:97] + "..."
    draw.text((12, foot_y + 18), cap_text, fill=(240, 240, 240))

    err_text = f"Error [{error_type}]: {reason}"
    if len(err_text) > 100:
        err_text = err_text[:97] + "..."
    draw.text((12, foot_y + 38), err_text, fill=(255, 110, 110))

    sample_filename = f"vae_{env_name}_{task_mode}_seed_{seed}_ep{episode}_step{step}.png"
    sample_path = os.path.join(output_dir, sample_filename)
    pil_img.save(sample_path)

    rec = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_name": env_name,
        "task_mode": task_mode,
        "seed": seed,
        "episode": episode,
        "step": step,
        "caption": caption,
        "error_type": error_type,
        "reason": reason,
        "sample_image": sample_path
    }
    with open(os.path.join(output_dir, "vae_errors.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"\n[DIAGNOSTIC VAE ERROR] {env_name} ({task_mode}) seed {seed} ep {episode} step {step}", flush=True)
    print(f"  Caption: \"{caption}\"", flush=True)
    print(f"  Error Type: {error_type}", flush=True)
    print(f"  Error Reason: {reason}", flush=True)
    print(f"  Sample Saved: {sample_path}\n", flush=True)
    return sample_path
