import os
import re
import json
import time
import copy
import hydra
import torch
import logging
import numpy as np
from PIL import Image
from collections import OrderedDict
from dotenv import load_dotenv
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate.utils import ProjectConfiguration
from architectures.common_utils import save_gif, preprocess_llm_output, initialize_llm_hf_pipeline, query_llm, post_process_caption
from utils.captioner import encode_image, query_llm as query_llm_vision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="config", config_name="test_imagination")
def main(args: DictConfig) -> None:
    # Suppress HTTP request logs from httpx, httpcore, and urllib3
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    evaluated_env_name = args.env.name
    task_mode = args.env.get("task_mode", "unknown")

    if args.env.name ==  "SimplePickup":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.SimplePickup import SimplePickup
        env = SimplePickup(args.env)
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        env_name = env.unwrapped.env_name
        env_description = env.unwrapped.env_description
    elif args.env.name ==  "PickEnv":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.PickEnv import PickEnv
        env = PickEnv(args.env)
        env_name = "PickEnv"
        env_description = env.env_description
    elif args.env.name.startswith("MiniWorld"):
        if args.mode == "transfer":
            args.env.verbose = True
        from env.MiniWorld import PickObjectEnv
        env = PickObjectEnv(args.env)
        env_name = env.env_name
        env_description = env.env_description
    elif args.env.name == "MiniGridRelational":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.MiniGridRelational import RelationalPickPlaceEnv
        env = RelationalPickPlaceEnv(args.env)
        from minigrid.wrappers import RGBImgObsWrapper
        env = RGBImgObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        env_name = env.unwrapped.env_name
        env_description = env.unwrapped.env_description
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "task_mode"):
        task_mode = env.unwrapped.task_mode
    elif hasattr(env, "task_mode"):
        task_mode = env.task_mode

    print("[INFO] Agent name: ", args.agent_name)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Environment seed configuration (increments per episode)
    base_seed = args.get("seed", 0)
    if base_seed is not None:
        print(f"[INFO] Environment seeding enabled with base_seed: {base_seed}")
    
    #Make the agent
    if args.agent_name == "SAC":
        from stable_baselines3 import SAC
        agent = SAC.load(args.dqn_model_dir)
    elif args.agent_name == "PPO":
        from stable_baselines3 import PPO
        agent = PPO.load(args.dqn_model_dir)
    elif args.agent_name == "DQN":
        from stable_baselines3 import DQN
        agent = DQN.load(args.dqn_model_dir)
        
    agent_model_dir = args.dqn_model_dir
    if os.path.exists(os.path.dirname(agent_model_dir) + "/config.yaml"):
        agent_model_args =  OmegaConf.load(os.path.dirname(agent_model_dir) + "/config.yaml")
        args.env = agent_model_args.env
    else:
        raise FileNotFoundError(f"Config file not found in {os.path.dirname(agent_model_dir)}/config.yaml")
    
    # Get data trasnformer
    if args.env.get("observation_mode", "image") == "image":
        from architectures.common_utils import get_train_transform_cnn
        train_transforms = get_train_transform_cnn() 
    else:
        from architectures.common_utils import get_train_transform_mlp
        train_transforms = get_train_transform_mlp()  
    
    # Make the vision model
    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.models.accelerator.project_dir,  
        logging_dir=args.models.accelerator.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.models.accelerator.gradient_accumulation_steps,
        mixed_precision=args.models.accelerator.mixed_precision,
        log_with=None, # Use the conditional logger
        project_config=accelerator_project_config,
    )
    if args.mode == "transfer":
        #Load the vision models
        vision_model_path = args.vae_model_dir
        vison_model_dir = os.path.dirname(vision_model_path)
        if os.path.exists(vison_model_dir + "/config.yaml"):
            vision_model_args =  OmegaConf.load(vison_model_dir + "/config.yaml")
            cfg = vision_model_args.models
        else:
            raise FileNotFoundError(f"Config file not found in {vison_model_dir}/config.yaml")
        accelerator.print("Initializing VAE model...")
        vision_model = instantiate(cfg.model)
        vision_model.load_params(vision_model_path)

        # Prepare agent for inference
        vision_model = accelerator.prepare(vision_model)
        
        #Setting the VAE model to eval mode
        vision_model.eval()
        
        system_prompt = args.get("system_prompt", "")
        # Load the .env file if available
        if load_dotenv is not None and os.path.exists("config/.env"):
            load_dotenv(dotenv_path="config/.env")
    
        mapping_strategy = args.get("mapping_strategy", "llm")
        caption_retriever = None
        if mapping_strategy == "retrieval":
            from baselines.retrieval_baseline import CaptionRetriever
            train_dir = cfg.data.train_dir
            text_encoder_path = cfg.data.get("text_encoder_path", "openai/clip-vit-base-patch32")
            caption_col = cfg.data.get("caption_column", "text")
            unwrapped_vm = accelerator.unwrap_model(vision_model)
            decoder = getattr(unwrapped_vm, "decoder", None)
            tok = getattr(decoder, "tokenizer", None)
            enc = getattr(decoder, "text_encoder", None)
            caption_retriever = CaptionRetriever(
                train_dir=train_dir,
                text_encoder_path=text_encoder_path,
                tokenizer=tok,
                text_encoder=enc,
                device=device,
                caption_column=caption_col,
                env_name=args.env.name
            )
            api_key = None
            pipe = None
            alternative_pipe = None
        elif args.querry_mode == "openrouter":
            # Access the API key
            api_key = os.getenv('OPENROUTER_API_KEY')
            pipe = args.llm_model
            alternative_pipe = args.alternate_llm_model
        elif args.querry_mode == "huggingface":
            api_key = None
            pipe = initialize_llm_hf_pipeline(args.llm_model)
            alternative_pipe = None
        elif args.querry_mode == "google":
            # Access the API key
            api_key = os.getenv('GOOGLE_API_KEY')
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            pipe = genai.GenerativeModel(args.llm_model)
            alternative_pipe = None
        elif args.querry_mode == "nvidia":
            api_key = os.getenv('NVIDIA_API')
            pipe = args.llm_model
            alternative_pipe = args.alternate_llm_model
        

    # Get the mission
    mission = env.unwrapped.mission
  
    # Rolling in-memory LLM response cache (LRU bounded to max_cache_size)
    max_cache_size = args.get("max_cache_size", 10000)
    llm_cache = OrderedDict()
    cache_hits = 0
    cache_misses = 0

    scores = []
    running_average_score = 0.0
    diagnostic_records = []
    episode_records_map = {}
    prior_perf = {}

    env_name_str = args.env.name
    # Error analysis scopes:
    # 1. SimplePickup, MiniWorld, MiniGridRelational: Both LLM and VAE error analysis
    # 2. PickEnv: Only LLM error analysis
    # 3. MiniWorldNoisy: Neither analysis
    do_llm_analysis = env_name_str in ("SimplePickup", "MiniWorld", "MiniGridRelational", "PickEnv")
    if env_name_str in ("SimplePickup", "MiniWorld"):
        do_vae_analysis = task_mode in ("target1", "target3")
    elif env_name_str == "MiniGridRelational":
        do_vae_analysis = True
    else:
        do_vae_analysis = False

    performance_md_file = args.get("performance_md_file", "Results/agent_performance.md")
    completed_episodes = {}
    if performance_md_file:
        from utils.update_performance_md import (
            get_completed_episodes_from_cache,
            append_or_update_metric,
            compute_performance_delta,
            merge_performances
        )
        completed_episodes = get_completed_episodes_from_cache(
            md_file_path=performance_md_file,
            env_name=evaluated_env_name,
            task_mode=task_mode,
            seed=base_seed,
            agent_name=args.agent_name
        )

    # If all requested episodes are already done, skip running
    if len(completed_episodes) >= args.num_episode:
        print(f"[RESUME] All {args.num_episode} episodes for {evaluated_env_name} ({task_mode}) seed {base_seed} already completed in cache. Skipping execution.", flush=True)
        return

    # Pre-populate already completed episodes
    for ep_idx in sorted(completed_episodes.keys()):
        if ep_idx < args.num_episode:
            ep_data = completed_episodes[ep_idx]
            scores.append(ep_data.get("score", 0.0))
            diagnostic_records.append(ep_data)
            episode_records_map[f"episode_{ep_idx}"] = ep_data
            if "performance" in ep_data:
                prior_perf = merge_performances(prior_perf, ep_data["performance"])

    if completed_episodes:
        running_average_score = float(np.mean(scores)) if scores else 0.0
        print(f"[RESUME] Resuming {evaluated_env_name} ({task_mode}) seed {base_seed} from episode {len(scores)}/{args.num_episode} (already completed: {sorted(completed_episodes.keys())}, running avg score: {running_average_score:.4f})", flush=True)

    def get_current_env_metric():
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_performance_metric"):
            m = env.unwrapped.get_performance_metric()
        elif hasattr(env, "get_performance_metric"):
            m = env.get_performance_metric()
        else:
            m = {}
        return copy.deepcopy(m) if m else {}

    for episode in range(args.num_episode):
        if episode in completed_episodes:
            continue
        frame_array_partial = []
        frame_array_full = []
        episode_seed = (base_seed + episode) if base_seed is not None else None
        print(f"----------- Starting Episode {episode}/{args.num_episode} (seed: {episode_seed}) ----------------", flush=True)
        env_metric_before_ep = get_current_env_metric()
        state, info = env.reset(seed=episode_seed)
        episode_step = 1
        frame_array_full.append(env.unwrapped.get_frame())
        cumulative_reward = 0
        done = False
        ep_imagination_steps = 0
        ep_llm_errors = 0
        ep_vae_errors = 0
        ep_llm_times = []
        while not done:
            if args.mode == "transfer":
                first_user_prompt = (
                                f"Environment description : {env_description}\n"
                                f"Target task : {mission}\n"
                                f"What agent knows : {args.env.mission}.\n"
                                f"Input description: {info['description']}"
                            )
                if args.env.name == "MiniWorldNoisy":
                    # Capture the frame
                    frame = state
                    # Encode the frame
                    base64_image = encode_image(frame)
                    
                    sensor_data = env.unwrapped.get_sensor_data() if hasattr(env.unwrapped, "get_sensor_data") else None
                    prompt_text = args.get("caption_user_prompt", "Describe this image for a text-to-image training dataset.")
                    if sensor_data:
                        prompt_text += f"\nSensor Data (Ground Truth): {sensor_data}, Incorporate this sensor data into the description."

                    vision_prompt = [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                    
                    try:
                        vllm_api_key = api_key if api_key is not None else os.getenv('OPENROUTER_API_KEY')
                        vllm_mode = args.get("vllm_mode")
                        if vllm_mode is None:
                            if args.get("vllm_model") == "google/gemma-4-12B-it" or args.querry_mode == "huggingface":
                                vllm_mode = "huggingface"
                            elif args.vllm_model and "/" in str(args.vllm_model):
                                vllm_mode = "openrouter"
                            else:
                                vllm_mode = args.querry_mode

                        if vllm_mode == "huggingface" and pipe is not None and (args.get("vllm_model") == args.llm_model or not args.get("vllm_model")):
                            vllm_pipeline = pipe
                        else:
                            vllm_pipeline = args.get("vllm_model", "google/gemma-4-12B-it")

                        caption = query_llm_vision(
                            system=args.caption_system_prompt,
                            prompt=vision_prompt,
                            api_key=vllm_api_key,
                            mode=vllm_mode,
                            pipeline=vllm_pipeline,
                            alternative_pipe=args.alternative_vllm,
                            temperature=0.1
                        )
                        # Post-processing
                        caption = post_process_caption(caption)

                        info['description'] = caption
                        # Reconstruct user prompt with new description
                        first_user_prompt = (
                                f"Environment description : {env_description}\n"
                                f"Target task : {mission}\n"
                                f"What agent knows : {args.env.mission}.\n"
                                f"Input description: {info['description']}"
                            )
                        # print(f"[INFO] Updated description via LLM: {caption}")
                    except Exception as e:
                        print(f"[ERROR] LLM Captioning failed: {e}")

                if mapping_strategy == "retrieval":
                    target_caption = info['description']
                    matched_caption, sim_score = caption_retriever.retrieve(target_caption)
                    changed_state, imagined_state = vision_model.imagine(state, matched_caption)
                else:
                    called_model = None
                    is_target2_bg_transfer = (task_mode == "target2" and args.env.name in ("SimplePickup", "MiniWorld"))
                    is_empty_view = ("No other objects can be seen." in info['description'] or "No objects are visible in the current view." in info['description'])

                    if is_empty_view and not is_target2_bg_transfer:
                        llm_reply = info['description']
                    elif first_user_prompt in llm_cache:
                        cache_hits += 1
                        llm_cache.move_to_end(first_user_prompt)
                        cached_entry = llm_cache[first_user_prompt]
                        llm_reply = cached_entry.get("reply", "")
                        reasoning = cached_entry.get("reasoning", None)
                        called_model = cached_entry.get("model", pipe)
                    else:
                        cache_misses += 1
                        _t_start = time.time()
                        llm_reply, reasoning = query_llm(system_prompt, first_user_prompt, api_key, pipe, alternative_pipe, args.querry_mode)
                        _llm_dt = time.time() - _t_start
                        ep_llm_times.append(_llm_dt)
                        called_model = pipe
                        if isinstance(reasoning, dict) and reasoning.get("model"):
                            called_model = reasoning["model"]
                        llm_cache[first_user_prompt] = {
                            "reply": llm_reply,
                            "reasoning": reasoning,
                            "response_time": _llm_dt,
                            "model": called_model
                        }
                        llm_cache.move_to_end(first_user_prompt)
                        if len(llm_cache) > max_cache_size:
                            llm_cache.popitem(last=False)

                    is_imagination_step = (not is_empty_view) or is_target2_bg_transfer
                    if is_imagination_step:
                        ep_imagination_steps += 1

                    llm_reply_json = preprocess_llm_output(llm_reply)
                    step_llm_failed = False
                    if is_imagination_step and do_llm_analysis:
                        from utils.imagination_diagnostics import evaluate_llm_mapping
                        llm_eval = evaluate_llm_mapping(
                            env_name=args.env.name,
                            task_mode=task_mode,
                            input_description=info.get('description', ''),
                            llm_reply_json=llm_reply_json,
                            raw_reply=llm_reply
                        )
                        if not llm_eval["is_valid"]:
                            step_llm_failed = True
                            ep_llm_errors += 1
                            from utils.imagination_diagnostics import log_llm_error_sample
                            log_llm_error_sample(
                                env_name=args.env.name,
                                task_mode=task_mode,
                                seed=base_seed,
                                episode=episode,
                                step=episode_step,
                                called_model=called_model,
                                target_description=info.get('description', ''),
                                mapped_description=llm_reply_json.get('description', ''),
                                raw_reply=llm_reply,
                                reason=llm_eval.get('reason', '')
                            )

                    if llm_reply_json.get("imagine", False):
                        changed_state, imagined_state = vision_model.imagine(state, llm_reply_json.get("description", ""))
                        # Only evaluate and attribute VAE error if enabled and LLM did NOT fail at this timestep
                        if do_vae_analysis and not step_llm_failed and isinstance(imagined_state, np.ndarray):
                            from utils.imagination_diagnostics import evaluate_vae_quality
                            vae_res = evaluate_vae_quality(
                                original_np=state,
                                imagined_np=imagined_state,
                                vision_model=vision_model,
                                env_name=args.env.name
                            )
                            if not vae_res["is_valid"]:
                                ep_vae_errors += 1
                                from utils.imagination_diagnostics import save_vae_error_sample
                                save_vae_error_sample(
                                    original_np=state,
                                    imagined_np=imagined_state,
                                    caption=llm_reply_json.get("description", ""),
                                    reason=vae_res.get("reason", ""),
                                    error_type=vae_res.get("error_type", ""),
                                    env_name=args.env.name,
                                    task_mode=task_mode,
                                    seed=base_seed,
                                    episode=episode,
                                    step=episode_step
                                )
                    else:
                        changed_state, imagined_state = train_transforms(state), state
            else:
                changed_state, imagined_state = train_transforms(state), state
            action = agent.predict(imagined_state, deterministic=False)[0]
            next_state, reward, truncated, terminated, info = env.step(action)
            frame_array_partial.append(np.hstack((state, imagined_state)))
            frame_array_full.append(env.unwrapped.get_frame())
            done = truncated + terminated
            cumulative_reward += reward
            state = next_state
            # print(f"Episode step done: {episode_step}")
            episode_step += 1

        scores.append(cumulative_reward)
        running_average_score = float(np.mean(scores))

        # write_video(frame_array, episode, dump_dir, frameSize=(env.unwrapped.get_frame().shape[1], env.unwrapped.get_frame().shape[0]))
        if args.mode == "transfer":
            save_dir = f"result/{args.agent_name}/{args.env.name}/{env_name}/transfer"
        else:
            save_dir = f"result/{args.agent_name}/{args.env.name}/{env_name}/source"
        save_gif(frame_array_partial, episode, save_dir, fps=args.env.fps, save_name= " partial")
        save_gif(frame_array_full, episode, save_dir, fps=args.env.fps, save_name= " full")
        total_lookups = cache_hits + cache_misses
        hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0.0
        print(f"----------- Episode done:  {episode}/{args.num_episode} | Score: {cumulative_reward} | Running Average Score: {running_average_score:.4f} | LLM Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate) ----------------", flush=True)

        is_success = (cumulative_reward > 5.0) if args.env.name.startswith("MiniWorld") else (cumulative_reward > 0.0)
        if is_success:
            failure_cause = "SUCCESS"
        else:
            if do_llm_analysis and ep_llm_errors > 0:
                failure_cause = "LLM_FAILURE"
            elif do_vae_analysis and ep_vae_errors > 0:
                failure_cause = "VAE_FAILURE"
            elif not do_llm_analysis and not do_vae_analysis:
                failure_cause = "NOT_EVALUATED"
            else:
                failure_cause = "DQN_POLICY_FAILURE"

        env_metric_after_ep = get_current_env_metric()
        from utils.update_performance_md import compute_performance_delta, merge_performances
        ep_perf_delta = compute_performance_delta(env_metric_after_ep, env_metric_before_ep)
        prior_perf = merge_performances(prior_perf, ep_perf_delta)

        avg_llm_time = round(float(np.mean(ep_llm_times)), 3) if ep_llm_times else 0.0
        ep_diag = {
            "episode": episode,
            "seed": episode_seed,
            "score": float(cumulative_reward),
            "running_average_score": running_average_score,
            "is_success": bool(is_success),
            "total_steps": episode_step,
            "imagination_steps": ep_imagination_steps,
            "llm_errors": ep_llm_errors,
            "vae_errors": ep_vae_errors,
            "avg_llm_response_time": avg_llm_time,
            "failure_cause": failure_cause,
            "performance": ep_perf_delta
        }
        diagnostic_records.append(ep_diag)
        episode_records_map[f"episode_{episode}"] = ep_diag

        if not is_success:
            print(f"[DIAGNOSTIC] Episode {episode}/{args.num_episode} FAILED -> Attributed Cause: {failure_cause} (LLM errors: {ep_llm_errors}/{ep_imagination_steps} steps, VAE errors: {ep_vae_errors}/{ep_imagination_steps} steps, Avg LLM Time: {avg_llm_time}s)", flush=True)
        else:
            print(f"[DIAGNOSTIC] Episode {episode}/{args.num_episode} SUCCESS (Score: {cumulative_reward:.2f}, Avg LLM Time: {avg_llm_time}s)", flush=True)

        # Compute running diagnostics summary metrics across all diagnostic_records
        total_eps = len(diagnostic_records)
        successes = sum(1 for d in diagnostic_records if d.get("is_success"))
        fails = total_eps - successes
        cause_counts = {
            "LLM_FAILURE": sum(1 for d in diagnostic_records if d.get("failure_cause") == "LLM_FAILURE"),
            "VAE_FAILURE": sum(1 for d in diagnostic_records if d.get("failure_cause") == "VAE_FAILURE"),
            "DQN_POLICY_FAILURE": sum(1 for d in diagnostic_records if d.get("failure_cause") in ("DQN_POLICY_FAILURE", "POLICY_NAVIGATION_FAILURE"))
        }
        total_imag_steps = sum(d.get("imagination_steps", 0) for d in diagnostic_records)
        total_llm_errs = sum(d.get("llm_errors", 0) for d in diagnostic_records)
        total_vae_errs = sum(d.get("vae_errors", 0) for d in diagnostic_records)
        llm_err_pct = (total_llm_errs / total_imag_steps * 100) if total_imag_steps > 0 else 0.0
        vae_err_pct = (total_vae_errs / total_imag_steps * 100) if total_imag_steps > 0 else 0.0

        running_perf = copy.deepcopy(prior_perf)
        running_perf["running_average_score"] = running_average_score
        running_perf["success_rate"] = round((successes / total_eps * 100) if total_eps > 0 else 0.0, 1)
        running_perf["total_imagination_steps"] = total_imag_steps
        running_perf["llm_error_steps"] = total_llm_errs if do_llm_analysis else None
        running_perf["vae_error_steps"] = total_vae_errs if do_vae_analysis else None
        running_perf["llm_error_rate_pct"] = round(llm_err_pct, 2) if do_llm_analysis else None
        running_perf["vae_error_rate_pct"] = round(vae_err_pct, 2) if do_vae_analysis else None
        running_perf["failure_breakdown"] = cause_counts if (do_llm_analysis or do_vae_analysis) else None

        # Real-time persistence of each completed episode and updated running performance
        if performance_md_file:
            try:
                from utils.update_performance_md import append_or_update_metric
                append_or_update_metric(
                    md_file_path=performance_md_file,
                    env_name=evaluated_env_name,
                    task_mode=task_mode,
                    seed=base_seed,
                    agent_name=args.agent_name,
                    performance=running_perf,
                    engine_name=args.get("llm_model", "google/gemma-4-12B-it"),
                    num_episodes=args.num_episode,
                    episode_records=episode_records_map
                )
            except Exception as e:
                print(f"[WARNING] Failed to update episode progress in cache: {e}")

    agent_performance = running_perf
    print("Agent performance" , agent_performance, flush=True)

    total_lookups = cache_hits + cache_misses
    hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0.0
    print(f"[INFO] Final in-memory LLM Cache summary: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate). Total unique cached entries: {len(llm_cache)} (max: {max_cache_size})", flush=True)

    # Save diagnostic failure analysis summary
    diag_dir = "Results/diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    diag_file = os.path.join(diag_dir, f"{evaluated_env_name}_{task_mode}_seed_{base_seed}.json")
    diag_summary = {
        "env_name": evaluated_env_name,
        "task_mode": task_mode,
        "seed": base_seed,
        "agent_name": args.agent_name,
        "total_episodes": total_eps,
        "success_count": successes,
        "fail_count": fails,
        "analysis_scope": {
            "do_llm_analysis": do_llm_analysis,
            "do_vae_analysis": do_vae_analysis
        },
        "timestep_metrics": {
            "total_imagination_steps": total_imag_steps,
            "llm_error_steps": total_llm_errs if do_llm_analysis else None,
            "llm_error_rate_pct": round(llm_err_pct, 2) if do_llm_analysis else None,
            "vae_error_steps": total_vae_errs if do_vae_analysis else None,
            "vae_error_rate_pct": round(vae_err_pct, 2) if do_vae_analysis else None
        },
        "failure_breakdown": cause_counts if (do_llm_analysis or do_vae_analysis) else None,
        "episodes": diagnostic_records
    }
    with open(diag_file, "w") as f:
        json.dump(diag_summary, f, indent=2)

    print("\n================ DIAGNOSTIC FAILURE ANALYSIS SUMMARY ================", flush=True)
    print(f"Scenario: {evaluated_env_name} ({task_mode}) | Seed: {base_seed} | Agent: {args.agent_name}", flush=True)
    print(f"Total Episodes: {total_eps} | Successes: {successes} | Failures: {fails}", flush=True)
    if do_llm_analysis or do_vae_analysis:
        llm_str = f"{total_llm_errs} ({llm_err_pct:.1f}%)" if do_llm_analysis else "N/A (Disabled)"
        vae_str = f"{total_vae_errs} ({vae_err_pct:.1f}%)" if do_vae_analysis else "N/A (Disabled)"
        print(f"Timestep Imagination Steps: {total_imag_steps} | LLM Errors: {llm_str} | VAE Errors: {vae_str}", flush=True)
        if fails > 0:
            if do_llm_analysis:
                print(f"  • LLM Failures:        {cause_counts['LLM_FAILURE']} ({(cause_counts['LLM_FAILURE']/fails)*100:.1f}%)", flush=True)
            if do_vae_analysis:
                print(f"  • VAE Failures:        {cause_counts['VAE_FAILURE']} ({(cause_counts['VAE_FAILURE']/fails)*100:.1f}%)", flush=True)
            print(f"  • DQN Policy Failures: {cause_counts['DQN_POLICY_FAILURE']} ({(cause_counts['DQN_POLICY_FAILURE']/fails)*100:.1f}%)", flush=True)
    else:
        print("[INFO] Error analysis disabled for this environment (MiniWorldNoisy).", flush=True)
    print(f"[INFO] Diagnostics saved to {diag_file}", flush=True)
    print("=====================================================================\n", flush=True)
    
if __name__ == "__main__":
    main()