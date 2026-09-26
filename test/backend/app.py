import sys
import os
import io
import json
import re
import time
from typing import Optional
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import instantiate
import base64
import uvicorn

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, "config/.env"))
    load_dotenv()
except ImportError:
    pass

from architectures.common_utils import get_train_transform_cnn, preprocess_llm_output, query_llm
from utils.imagination_diagnostics import (
    evaluate_llm_mapping,
    evaluate_vae_quality,
    compute_grid_differences,
    compute_latent_difference
)

app = FastAPI(title="LG-MAGIK Lens with Imagination Diagnostics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = None
config_args = None
llm_pipeline = None

# Environment metadata for LLM prompt construction
ENV_METADATA = {
    "SimplePickup": {
        "description": "This environment is a gridworld where an agent navigates a grid with objects (balls, boxes) and walls.",
        "source_mission": "Pick the red ball and avoid green ball from the room with grey wall",
        "target_missions": {
            "target1": "Pick up the purple box.",
            "target2": "Pick up the purple box.",
            "target3": "Pick up the green ball."
        },
        "default_caption": "Agent is at (3, 4) facing right. The purple box is at (2, 2). The green ball is at (5, 5)."
    },
    "MiniWorld": {
        "description": "A continuous 3D room with objects like duckies, boxes, balls, with colored walls and floor.",
        "source_mission": "Pick the blue box and avoid green ball from the room with grass floor and concrete wall",
        "target_missions": {
            "target1": "Pick up the duckie.",
            "target2": "Pick up the duckie.",
            "target3": "Pick up the ball."
        },
        "default_caption": "A duckie is visible at 3.1 units and 5.5 degrees to the left, and a ball is visible at 2.0 units and 24.5 degrees to the right, both located on a green grass floor surrounded by a grey wall under a blue sky."
    },
    "PickEnv": {
        "description": "A continuous 2D environment with circular and square objects of different weights (heavy, light).",
        "source_mission": "Pick the light circle or heavy square without breaking it by applying required force.",
        "target_missions": {
            "target": "Pick up the heavy square."
        },
        "default_caption": "The heavy circle is at (-0.2, 0.4) and the light square is at (0.3, -0.1)."
    },
    "MiniGridRelational": {
        "description": "A relational gridworld pick-and-place environment.",
        "source_mission": "Pick up the blue ball and drop it on the green target.",
        "target_missions": {
            "target1": "Pick up the red ball and drop it strctly on a cell immediately adjacent to the yellow box.",
            "target2": "Pick up the red ball and drop it at the symmetric opposite of the yellow box.",
            "target3": "Pick up the red ball and drop it on a cell that is exactly 2 Manhattan-distance away from the yellow box, choosing the one nearest to the agent not colluding with the agent location (choose only one if multiple).",
            "target4": "Pick up the red ball and drop it at the symmetric opposite of the cell two cells above the yellow box.",
            "target5": "Pick up the red ball and drop it on the yellow target and pick up the purple ball and drop it on the grey target. The two pairs may be completed in any order."
        },
        "default_caption": "Agent is at (3, 4) facing right. The red ball is at (2, 2). The yellow box is at (4, 4)."
    }
}


def extract_agent_source_mission(env_name: str, config_args=None) -> str:
    """
    Extracts 'what agent knows' (source task mission) from the agent training config file.
    Follows test-imagination config (config/test_imagination.yaml) -> dqn_model_dir -> config.yaml -> env.mission
    """
    # 1. Primary approach: extract from dqn_model_dir in config_args (matches test_imagination.py)
    if config_args:
        dqn_model_dir = getattr(config_args, "dqn_model_dir", None)
        agent_name = getattr(config_args, "agent_name", "DQN")
        if dqn_model_dir:
            path_str = str(dqn_model_dir).replace("{agent_name}", str(agent_name))
            if not os.path.isabs(path_str):
                path_str = os.path.join(PROJECT_ROOT, path_str)

            agent_dir = os.path.dirname(path_str) if (path_str.endswith(".zip") or os.path.isfile(path_str)) else path_str
            agent_cfg_path = os.path.join(agent_dir, "config.yaml")
            if os.path.exists(agent_cfg_path):
                try:
                    agent_cfg = OmegaConf.load(agent_cfg_path)
                    mission = agent_cfg.get("env", {}).get("mission") or agent_cfg.get("mission")
                    cfg_env_name = agent_cfg.get("env", {}).get("name")
                    if mission and (not cfg_env_name or cfg_env_name == env_name or env_name == "MiniWorld"):
                        return str(mission).strip()
                except Exception as e:
                    print(f"[WARN] Failed to load agent config from {agent_cfg_path}: {e}")

    # 2. Secondary fallback: find agent training config under model_weights/{env_name}/
    for algo in ["PPO", "DQN", "SAC"]:
        for seed in ["seed_123", "seed_42", "seed_789", "seed_456", "seed_1024"]:
            candidate = os.path.join(PROJECT_ROOT, "model_weights", env_name, algo, seed, "config.yaml")
            if os.path.exists(candidate):
                try:
                    agent_cfg = OmegaConf.load(candidate)
                    mission = agent_cfg.get("env", {}).get("mission") or agent_cfg.get("mission")
                    if mission:
                        return str(mission).strip()
                except Exception:
                    pass

    # 3. Final fallback: use ENV_METADATA default
    return ENV_METADATA.get(env_name, {}).get("source_mission", "Solve the source task.")


_ENV_INFO_CACHE = {}

def get_target_env_info(env_name: str, task_mode: str) -> tuple[str, str]:
    """
    Initialises the environment with task_mode from the dropdown and extracts:
    - env.mission (matching test_imagination.py:L192)
    - env.env_description (matching test_imagination.py:L55 / L66)
    """
    cache_key = (env_name, task_mode)
    if cache_key in _ENV_INFO_CACHE:
        return _ENV_INFO_CACHE[cache_key]

    mission = None
    env_description = None

    try:
        if env_name.startswith("MiniWorld"):
            from env.MiniWorld import PickObjectEnv
            env_cfg = {
                "name": env_name,
                "task_mode": task_mode,
                "size": 10,
                "obs_width": 80,
                "obs_height": 80,
                "render_mode": "none",
                "max_steps": 100
            }
            env = PickObjectEnv(env_cfg)
            mission = getattr(env, "mission", getattr(getattr(env, "unwrapped", None), "mission", None))
            raw_desc = getattr(env, "env_description", getattr(getattr(env, "unwrapped", None), "env_description", None))
            env_description = raw_desc() if callable(raw_desc) else raw_desc

        elif env_name == "SimplePickup":
            from env.SimplePickup import SimplePickup
            cfg_path = os.path.join(PROJECT_ROOT, "config/env/SimplePickup.yaml")
            if os.path.exists(cfg_path):
                env_cfg = OmegaConf.load(cfg_path)
            else:
                env_cfg = OmegaConf.create({
                    "name": "SimplePickup", "size": 9, "tile_size": 8,
                    "max_steps": 50, "agent_view_size": 7, "render_mode": "none", "highlight": False
                })
            env_cfg.task_mode = task_mode
            env = SimplePickup(env_cfg)
            mission = getattr(env, "mission", getattr(getattr(env, "unwrapped", None), "mission", None))
            raw_desc = getattr(env, "env_description", getattr(getattr(env, "unwrapped", None), "env_description", None))
            env_description = raw_desc() if callable(raw_desc) else raw_desc

        elif env_name == "MiniGridRelational":
            from env.MiniGridRelational import RelationalPickPlaceEnv
            cfg_path = os.path.join(PROJECT_ROOT, "config/env/MiniGridRelational.yaml")
            if os.path.exists(cfg_path):
                env_cfg = OmegaConf.load(cfg_path)
            else:
                env_cfg = OmegaConf.create({
                    "name": "MiniGridRelational", "size": 8, "tile_size": 8,
                    "max_steps": 30, "render_mode": "none"
                })
            env_cfg.task_mode = task_mode
            env = RelationalPickPlaceEnv(env_cfg)
            mission = getattr(env, "mission", getattr(getattr(env, "unwrapped", None), "mission", None))
            raw_desc = getattr(env, "env_description", getattr(getattr(env, "unwrapped", None), "env_description", None))
            env_description = raw_desc() if callable(raw_desc) else raw_desc

        elif env_name == "PickEnv":
            from env.PickEnv import PickEnv
            cfg_path = os.path.join(PROJECT_ROOT, "config/env/PickEnv.yaml")
            if os.path.exists(cfg_path):
                env_cfg = OmegaConf.load(cfg_path)
            else:
                env_cfg = OmegaConf.create({
                    "name": "PickEnv", "width": 128, "height": 128,
                    "max_steps": 100
                })
            env_cfg.task_mode = task_mode
            env_cfg.mode = "target" if task_mode == "target" else "train"
            env = PickEnv(env_cfg)
            mission = getattr(env, "mission", getattr(getattr(env, "unwrapped", None), "mission", None))
            raw_desc = getattr(env, "env_description", getattr(getattr(env, "unwrapped", None), "env_description", None))
            env_description = raw_desc() if callable(raw_desc) else raw_desc

    except Exception as e:
        print(f"[WARN] Failed to initialize env '{env_name}' with task_mode '{task_mode}': {e}")

    if not mission:
        mission = ENV_METADATA.get(env_name, {}).get("target_missions", {}).get(task_mode, f"Solve task {task_mode}.")
    if not env_description:
        env_description = ENV_METADATA.get(env_name, {}).get("description", "A reinforcement learning environment.")

    result = (str(mission).strip(), str(env_description).strip())
    _ENV_INFO_CACHE[cache_key] = result
    return result


def get_target_mission_from_env(env_name: str, task_mode: str) -> str:
    return get_target_env_info(env_name, task_mode)[0]


current_env_name = None
current_vae_path = None


def find_vae_model_for_env(env_name: str) -> Optional[str]:
    """Finds the best matching VAE model checkpoint for a given environment."""
    # 1. If config_args has a vae_model_dir that matches env_name
    if config_args:
        cfg_vae = getattr(config_args, "vae_model_dir", None)
        if cfg_vae and f"model_weights/{env_name}/" in str(cfg_vae):
            full_p = os.path.join(PROJECT_ROOT, str(cfg_vae)) if not os.path.isabs(str(cfg_vae)) else str(cfg_vae)
            if os.path.exists(full_p):
                return full_p

    # 2. Check prioritized subdirectories for env_name under model_weights
    env_dir = os.path.join(PROJECT_ROOT, "model_weights", env_name)
    if not os.path.exists(env_dir):
        return None

    preferred_variants = [
        "VAE_1D_no_text_disc",
        "VAE_no_text_disc",
        "VAE_1D",
        "VAE"
    ]
    preferred_seeds = ["seed_123", "seed_42", "seed_456", "seed_789", "seed_1024"]

    for variant in preferred_variants:
        for seed in preferred_seeds:
            candidate_dir = os.path.join(env_dir, variant, seed)
            if os.path.isdir(candidate_dir):
                for fn in [f"{variant}.tar", "LG_MAGIK_VAE_TRAINING.tar", "VAE.tar"]:
                    cp = os.path.join(candidate_dir, fn)
                    if os.path.exists(cp) and os.path.exists(os.path.join(candidate_dir, "config.yaml")):
                        return cp

    # 3. Fallback: recursively search for any .tar with config.yaml in same directory
    for root, dirs, files in os.walk(env_dir):
        if "config.yaml" in files:
            for f in files:
                if f.endswith(".tar"):
                    return os.path.join(root, f)

    return None


def load_vision_model_for_env(env_name: str) -> bool:
    """Reloads the VAE vision model matching the environment chosen from the dropdown."""
    global vision_model, current_env_name, current_vae_path, device

    if current_env_name == env_name and vision_model is not None:
        return True

    vision_model_path = find_vae_model_for_env(env_name)
    if not vision_model_path:
        print(f"[WARN] No VAE model weights found for environment: {env_name}")
        return False

    print(f"[INFO] Reloading VAE model for {env_name} from: {vision_model_path}")
    model_dir = os.path.dirname(vision_model_path)
    model_config_path = os.path.join(model_dir, "config.yaml")

    if not os.path.exists(model_config_path):
        print(f"[WARN] Config file not found in {model_config_path}")
        return False

    try:
        vision_model_args = OmegaConf.load(model_config_path)
        cfg = vision_model_args.models

        # Free previous model from GPU memory
        if vision_model is not None:
            try:
                del vision_model
                vision_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        target_device = device
        if target_device.type == "cuda":
            try:
                free_mem, total_mem = torch.cuda.mem_get_info()
                if free_mem < 512 * 1024 * 1024:
                    print(f"[WARN] Low GPU memory ({free_mem / 1024**2:.1f} MiB). Falling back to CPU for VAE.")
                    target_device = torch.device("cpu")
            except Exception:
                pass

        try:
            new_model = instantiate(cfg.model)
            new_model.load_params(vision_model_path)
            new_model.to(target_device)
            new_model.eval()
        except Exception as load_err:
            if target_device.type == "cuda":
                print(f"[WARN] Failed to load on CUDA ({load_err}). Retrying on CPU...")
                target_device = torch.device("cpu")
                new_model = instantiate(cfg.model)
                new_model.load_params(vision_model_path)
                new_model.to(target_device)
                new_model.eval()
            else:
                raise load_err

        vision_model = new_model
        current_env_name = env_name
        current_vae_path = vision_model_path
        print(f"[INFO] Vision Model for '{env_name}' reloaded successfully on {target_device}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load vision model for {env_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.on_event("startup")
async def startup_event():
    global config_args
    try:
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Initialize hydra with config path relative to this file
        with initialize(version_base=None, config_path="../../config"):
            args = compose(config_name="test_imagination")
            config_args = args

        # Load initial VAE for default environment (MiniWorld)
        default_env = getattr(getattr(config_args, "env", None), "name", "MiniWorld") or "MiniWorld"
        load_vision_model_for_env(default_env)

    except Exception as e:
        print(f"[ERROR] Startup initialization error: {e}")
        import traceback
        traceback.print_exc()


def encode_image_base64(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return "data:image/png;base64," + base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')


def create_latent_grid(mean_tensor, ref_tensor=None):
    mean = mean_tensor.squeeze(0)
    ref = ref_tensor.squeeze(0) if ref_tensor is not None else mean

    num_channels = mean.shape[0]
    cols = 4
    rows = (num_channels + cols - 1) // cols

    channels_np = mean.cpu().numpy()
    ref_np = ref.cpu().numpy()
    padding = 1

    if mean.dim() == 1:
        h, w = 16, 16
        grid_w = cols * w + (cols + 1) * padding
        grid_h = rows * h + (rows + 1) * padding
        grid_img = Image.new('L', (grid_w, grid_h), color=255)

        ref_min = float(ref_np.min())
        ref_max = float(ref_np.max())
        diff = max(ref_max - ref_min, 1e-5)
        for i in range(num_channels):
            val = int(np.clip((channels_np[i] - ref_min) / diff * 255, 0, 255))
            ch_img = Image.fromarray(np.full((h, w), val, dtype=np.uint8))
            r = i // cols
            c = i % cols
            x_pos = padding + c * (w + padding)
            y_pos = padding + r * (h + padding)
            grid_img.paste(ch_img, (x_pos, y_pos))
    else:
        h, w = channels_np.shape[1], channels_np.shape[2]
        grid_w = cols * w + (cols + 1) * padding
        grid_h = rows * h + (rows + 1) * padding
        grid_img = Image.new('L', (grid_w, grid_h), color=255)

        for i in range(num_channels):
            ch_data = channels_np[i]
            ch_min = ref_np[i].min()
            ch_max = ref_np[i].max()
            if ch_max - ch_min > 1e-5:
                ch_norm = (ch_data - ch_min) / (ch_max - ch_min) * 255
                ch_norm = np.clip(ch_norm, 0, 255)
            else:
                ch_norm = np.zeros_like(ch_data)

            ch_img = Image.fromarray(ch_norm.astype(np.uint8))
            r = i // cols
            c = i % cols
            x_pos = padding + c * (w + padding)
            y_pos = padding + r * (h + padding)
            grid_img.paste(ch_img, (x_pos, y_pos))

    if grid_img.width < 512:
        scale = 512 / grid_img.width
        new_size = (int(grid_img.width * scale), int(grid_img.height * scale))
        grid_img = grid_img.resize(new_size, Image.NEAREST)
    return grid_img


# ==============================================================================
# ENDPOINT: /remap_caption (LLM INVOCATION & ERROR ANALYSIS)
# ==============================================================================

@app.post("/remap_caption")
async def remap_caption(
    caption: str = Form(...),
    env_name: str = Form("MiniWorld"),
    task_mode: str = Form("target1"),
    query_mode: Optional[str] = Form(None),
    querry_mode: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None)
):
    """
    Invokes the LLM using the imagination system prompt to transform the input description.
    Once generated, automatically runs the rule-based LLM Error Analyser.
    """
    global config_args, llm_pipeline

    raw_source_mission = extract_agent_source_mission(env_name, config_args)
    clean_source_mission = raw_source_mission.rstrip(".")

    # Dynamically extract target task mission and environment description from env
    target_mission, env_description = get_target_env_info(env_name, task_mode)

    # Exact replication of test_imagination.py:L219-L224:
    user_prompt = (
        f"Environment description : {env_description}\n"
        f"Target task : {target_mission}\n"
        f"What agent knows : {clean_source_mission}.\n"
        f"Input description: {caption.strip()}"
    )

    system_prompt = getattr(config_args, "system_prompt", "") if config_args else ""
    if not system_prompt:
        cfg_path = os.path.join(PROJECT_ROOT, "config/test_imagination.yaml")
        if os.path.exists(cfg_path):
            try:
                raw_cfg = OmegaConf.load(cfg_path)
                system_prompt = str(raw_cfg.get("system_prompt", ""))
            except Exception:
                pass

    active_query_mode = query_mode or querry_mode
    q_mode = (active_query_mode.strip() if active_query_mode and active_query_mode.strip() else None) or (getattr(config_args, "querry_mode", "nvidia") if config_args else "nvidia")
    clean_model = llm_model.strip() if llm_model and llm_model.strip() else None
    model_name = clean_model or (getattr(config_args, "llm_model", "nvidia/nemotron-3-super-120b-a12b") if config_args else "nvidia/nemotron-3-super-120b-a12b")
    alt_model = getattr(config_args, "alternate_llm_model", None) if config_args else None

    if q_mode == "nvidia":
        api_key = os.getenv("NVIDIA_API") or os.getenv("NVIDIA_API_KEY")
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
    raw_reply = None
    reasoning_text = None
    llm_reply_json = None
    error_msg = None

    try:
        if q_mode == "openrouter" and not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is not set on the server.")
        if q_mode == "nvidia" and not api_key:
            raise RuntimeError("NVIDIA_API environment variable is not set on the server.")

        if q_mode == "huggingface" and llm_pipeline is not None and getattr(llm_pipeline, "_model_name", None) == model_name:
            pipe_input = llm_pipeline
        else:
            pipe_input = model_name

        start_time = time.time()
        raw_reply, raw_reasoning = query_llm(
            system=system_prompt,
            prompt=user_prompt,
            api_key=api_key,
            pipeline=pipe_input,
            alternative_pipe=alt_model,
            mode=q_mode
        )
        llm_response_time = round(time.time() - start_time, 3)
        llm_reply_json = preprocess_llm_output(raw_reply)

        # Extract reasoning content from query_llm metadata or raw response
        if isinstance(raw_reasoning, str) and raw_reasoning.strip():
            reasoning_text = raw_reasoning.strip()
        elif isinstance(raw_reasoning, dict):
            reasoning_text = raw_reasoning.get("reasoning") or raw_reasoning.get("analysis")

        # Fallback: check if reasoning was produced within thinking tags or before the JSON object in raw_reply
        if not reasoning_text and raw_reply:
            think_match = re.search(r"<think>(.*?)</think>", raw_reply, flags=re.DOTALL)
            if think_match:
                reasoning_text = think_match.group(1).strip()
            else:
                channel_match = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)", raw_reply, flags=re.DOTALL)
                if channel_match:
                    reasoning_text = channel_match.group(1).strip()
                else:
                    first_brace = raw_reply.find("{")
                    if first_brace > 0:
                        pre_json = raw_reply[:first_brace].strip()
                        pre_json = re.sub(r"^```(?:json)?", "", pre_json, flags=re.MULTILINE).strip()
                        if pre_json:
                            reasoning_text = pre_json

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        llm_response_time = round(time.time() - start_time, 3) if 'start_time' in locals() else None
        print(f"[WARN] LLM invocation failed ({e}). Returning structured fallback.")
        llm_reply_json = {
            "imagine": False,
            "description": caption,
            "error": error_msg
        }
        raw_reply = f"[ERROR: {error_msg}]"

    # Run LLM Error Analyser
    llm_analysis = evaluate_llm_mapping(
        env_name=env_name,
        task_mode=task_mode,
        input_description=caption,
        llm_reply_json=llm_reply_json,
        raw_reply=raw_reply
    )

    return JSONResponse(content={
        "status": "success" if not error_msg else "llm_failed",
        "env_name": env_name,
        "task_mode": task_mode,
        "target_mission": target_mission,
        "env_description": env_description,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "response_time": llm_response_time,
        "raw_reply": raw_reply,
        "reasoning": reasoning_text,
        "reply_json": llm_reply_json,
        "remapped_caption": llm_reply_json.get("description", "") if isinstance(llm_reply_json, dict) else "",
        "imagine": llm_reply_json.get("imagine", False) if isinstance(llm_reply_json, dict) else False,
        "llm_analysis": llm_analysis
    })


# ==============================================================================
# ENDPOINT: /imagine (WITH VAE SMUDGE & CYCLE CONSISTENCY ANALYSIS)
# ==============================================================================

@app.post("/imagine")
async def imagine(
    file: UploadFile = File(...),
    caption: str = Form(...),
    mode: str = Form("imagination"),
    channel_scales: str = Form(None),
    env_name: str = Form("MiniWorld")
):
    global vision_model, current_env_name
    if env_name and env_name != current_env_name:
        load_vision_model_for_env(env_name)

    if not vision_model:
        raise HTTPException(status_code=500, detail=f"Vision model for environment '{env_name}' not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        response_data = {}

        model_device = next(vision_model.parameters()).device

        if mode == "latent":
            # --- Latent Visualization & Manipulation Logic ---
            transform = vision_model.train_transform
            state_tensor = transform(image_np).unsqueeze(0).to(model_device)

            with torch.no_grad():
                hidden = vision_model.encoder(state_tensor)
                if getattr(vision_model, "latent_type", "spatial") == "vector":
                    sampler = vision_model.bottleneck(hidden.flatten(1))
                else:
                    sampler = vision_model.bottleneck(hidden)
                mean = sampler.mean

                num_latent_channels = mean.shape[1]
                response_data["latent_channels"] = num_latent_channels

                grid_img = create_latent_grid(mean)
                response_data["original_latent"] = encode_image_base64(grid_img)

                if channel_scales:
                    scales = json.loads(channel_scales)
                    if len(scales) < num_latent_channels:
                        scales = list(scales) + [1.0] * (num_latent_channels - len(scales))
                    else:
                        scales = list(scales)[:num_latent_channels]
                else:
                    scales = [1.0] * num_latent_channels

                modified_mean = mean.clone()
                for i, s in enumerate(scales):
                    if modified_mean.dim() == 4:
                        modified_mean[:, i, :, :] *= float(s)
                    else:
                        modified_mean[:, i] *= float(s)

                if hasattr(vision_model, "decoder") and hasattr(vision_model.decoder, "tokenizer"):
                    tokeniser = vision_model.decoder.tokenizer
                    from architectures.common_utils import tokenize_captions
                    captions_tokenised, attention_mask = tokenize_captions(
                        tokeniser, [caption if caption else ""], max_length=vision_model.max_sequence_length
                    )
                    captions_tokenised = captions_tokenised.to(model_device)
                    attention_mask = attention_mask.to(model_device)
                    reconstructed_x, _ = vision_model.decoder(modified_mean, captions_tokenised, attention_mask, return_text_feats=True)
                else:
                    reconstructed_x = vision_model.decode(modified_mean).sample

                imagined_numpy = ((reconstructed_x.squeeze(0).detach().cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                res_image = Image.fromarray(imagined_numpy)
                response_data["reconstruction"] = encode_image_base64(res_image)

                mod_grid = create_latent_grid(modified_mean, ref_tensor=mean)
                response_data["modified_latent"] = encode_image_base64(mod_grid)

                # Evaluate VAE quality & smudge on the latent-modified reconstruction
                imagined_tensor = transform(res_image).unsqueeze(0).to(model_device)
                vae_analysis = evaluate_vae_quality(
                    original_np=image_np,
                    imagined_np=imagined_numpy,
                    vision_model=vision_model,
                    original_tensor=state_tensor,
                    imagined_tensor=imagined_tensor,
                    device=model_device,
                    env_name=env_name
                )
                response_data["vae_analysis"] = vae_analysis

        else:
            # --- Standard Imagination Logic with Optional Latent Channel Scaling ---
            transform = vision_model.train_transform
            state_tensor = transform(image_np).unsqueeze(0).to(model_device)

            with torch.no_grad():
                hidden = vision_model.encoder(state_tensor)
                if getattr(vision_model, "latent_type", "spatial") == "vector":
                    sampler = vision_model.bottleneck(hidden.flatten(1))
                else:
                    sampler = vision_model.bottleneck(hidden)
                mean_original = sampler.mean
                num_latent_channels = mean_original.shape[1]
                response_data["latent_channels"] = num_latent_channels

                # If channel scales are provided for testing smudging by sliding latent:
                if channel_scales:
                    scales = json.loads(channel_scales)
                    if len(scales) < num_latent_channels:
                        scales = list(scales) + [1.0] * (num_latent_channels - len(scales))
                    else:
                        scales = list(scales)[:num_latent_channels]
                    target_mean = mean_original.clone()
                    for i, s in enumerate(scales):
                        if target_mean.dim() == 4:
                            target_mean[:, i, :, :] *= float(s)
                        else:
                            target_mean[:, i] *= float(s)
                else:
                    target_mean = mean_original

                if hasattr(vision_model, "decoder") and hasattr(vision_model.decoder, "tokenizer"):
                    tokeniser = vision_model.decoder.tokenizer
                    from architectures.common_utils import tokenize_captions
                    captions_tokenised, attention_mask = tokenize_captions(
                        tokeniser, [caption if caption else ""], max_length=vision_model.max_sequence_length
                    )
                    captions_tokenised = captions_tokenised.to(model_device)
                    attention_mask = attention_mask.to(model_device)
                    reconstructed_x, _ = vision_model.decoder(target_mean, captions_tokenised, attention_mask, return_text_feats=True)
                else:
                    reconstructed_x = vision_model.decode(target_mean).sample

                imagined_numpy = ((reconstructed_x.squeeze(0).detach().cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                res_image = Image.fromarray(imagined_numpy)
                response_data["result"] = encode_image_base64(res_image)

                imagined_pil = Image.fromarray(imagined_numpy).convert("RGB")
                imagined_tensor = transform(imagined_pil).unsqueeze(0).to(model_device)
                hidden_recon = vision_model.encoder(imagined_tensor)
                if getattr(vision_model, "latent_type", "spatial") == "vector":
                    sampler_recon = vision_model.bottleneck(hidden_recon.flatten(1))
                else:
                    sampler_recon = vision_model.bottleneck(hidden_recon)
                mean_recon = sampler_recon.mean

                grid_original = create_latent_grid(mean_original)
                grid_recon = create_latent_grid(mean_recon, ref_tensor=mean_original)

                response_data["original_latent"] = encode_image_base64(grid_original)
                response_data["reconstructed_latent"] = encode_image_base64(grid_recon)

            # Evaluate VAE quality (Smudge & Cycle Consistency)
            vae_analysis = evaluate_vae_quality(
                original_np=image_np,
                imagined_np=imagined_numpy,
                vision_model=vision_model,
                original_tensor=state_tensor,
                imagined_tensor=imagined_tensor,
                device=model_device,
                env_name=env_name
            )
            response_data["vae_analysis"] = vae_analysis

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINT: /diagnostics/vae (STANDALONE VAE ANALYSIS)
# ==============================================================================

@app.post("/diagnostics/vae")
async def diagnostics_vae(
    original_file: UploadFile = File(...),
    imagined_file: UploadFile = File(...),
    env_name: str = Form("MiniWorld")
):
    """
    Evaluates smudge detection, degeneracy, and cycle consistency between two uploaded images.
    """
    global vision_model, current_env_name
    if env_name and env_name != current_env_name:
        load_vision_model_for_env(env_name)
    try:
        orig_bytes = await original_file.read()
        imag_bytes = await imagined_file.read()

        orig_img = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
        imag_img = Image.open(io.BytesIO(imag_bytes)).convert("RGB")

        orig_np = np.array(orig_img)
        imag_np = np.array(imag_img)

        state_tensor = None
        imagined_tensor = None
        if vision_model is not None:
            transform = vision_model.train_transform
            state_tensor = transform(orig_np).unsqueeze(0).to(device)
            imagined_tensor = transform(imag_np).unsqueeze(0).to(device)

        analysis = evaluate_vae_quality(
            original_np=orig_np,
            imagined_np=imag_np,
            vision_model=vision_model,
            original_tensor=state_tensor,
            imagined_tensor=imagined_tensor,
            device=device,
            env_name=env_name
        )
        return JSONResponse(content=analysis)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata/environments")
async def get_environments():
    """Returns available environments, target modes, and sample captions."""
    metadata_copy = {}
    for env, data in ENV_METADATA.items():
        data_copy = dict(data)
        data_copy["source_mission"] = extract_agent_source_mission(env, config_args)
        target_dict = {}
        for tm in data.get("target_missions", {}):
            target_dict[tm] = get_target_mission_from_env(env, tm)
        data_copy["target_missions"] = target_dict
        metadata_copy[env] = data_copy
    return JSONResponse(content=metadata_copy)


@app.post("/load_env_model")
def load_env_model(env_name: str = Form(...)):
    """Reloads the VAE vision model matching the environment chosen from the dropdown."""
    success = load_vision_model_for_env(env_name)
    latent_channels = 8
    if vision_model and hasattr(vision_model, "bottleneck"):
        latent_channels = getattr(
            vision_model.bottleneck, "out_features",
            getattr(vision_model.bottleneck, "out_channels", getattr(vision_model.bottleneck, "latent_dim", 8))
        )
    return JSONResponse(content={
        "status": "success" if success else "failed",
        "env_name": env_name,
        "current_env": current_env_name,
        "model_path": current_vae_path,
        "latent_channels": latent_channels
    })


@app.get("/health")
async def health_check():
    latent_channels = 8
    if vision_model and hasattr(vision_model, "bottleneck"):
        latent_channels = getattr(
            vision_model.bottleneck, "out_features",
            getattr(vision_model.bottleneck, "out_channels", getattr(vision_model.bottleneck, "latent_dim", 8))
        )
    return {
        "status": "ready" if vision_model is not None else "not_loaded",
        "current_env": current_env_name,
        "current_vae_path": current_vae_path,
        "device": str(device),
        "latent_channels": latent_channels,
        "querry_mode": getattr(config_args, "querry_mode", "nvidia") if config_args else "nvidia",
        "llm_model": getattr(config_args, "llm_model", "nvidia/nemotron-3-super-120b-a12b") if config_args else "nvidia/nemotron-3-super-120b-a12b"
    }


@app.head("/")
async def head_index():
    return Response(status_code=200)


@app.get("/")
async def read_index():
    index_file = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    with open(index_file, "r") as f:
        html_content = f.read()
    return Response(content=html_content, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
