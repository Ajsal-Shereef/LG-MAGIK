import os
import re
import sys
import json
import time
import copy
import logging
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from architectures.common_utils import query_llm, initialize_llm_hf_pipeline
from utils.update_performance_md import (
    get_completed_episodes_from_cache,
    append_or_update_metric,
    merge_performances,
    compute_performance_delta
)

def parse_action_response(llm_reply: str, env_name: str, env_actions=None):
    """
    Parses the action from the LLM reply.
    Handles both discrete actions and continuous 3D control vectors for PickEnv.
    """
    clean_text = llm_reply.strip()
    # Strip special turn tokens and markdown code fences if present
    clean_text = re.sub(r"<turn\|>|<end_of_turn>|<\|end\|>|<\|return\|>", "", clean_text).strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE)
    clean_text = clean_text.strip()
    
    # Try parsing json directly or via regex
    action_val = None
    try:
        data = json.loads(clean_text)
        action_val = data.get("action")
    except Exception:
        # Search for "action": ... pattern
        match = re.search(r'"action"\s*:\s*([^,\}\]]+)', clean_text)
        if match:
            raw_val = match.group(1).strip().strip('"\'')
            action_val = raw_val
        else:
            # Search for list pattern like [0.1, 0.5, -1.0]
            list_match = re.search(r'\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]', clean_text)
            if list_match:
                try:
                    action_val = [float(x.strip()) for x in list_match.group(1).split(",")]
                except Exception:
                    pass

    # Environment-specific conversion
    if env_name == "PickEnv":
        # Continuous action: [steer, distance, force]
        default_pick_action = np.array([0.0, 0.5, -1.0], dtype=np.float32)
        if isinstance(action_val, (list, tuple)) and len(action_val) >= 3:
            try:
                steer = float(np.clip(float(action_val[0]), -1.0, 1.0))
                dist = float(np.clip(float(action_val[1]), -1.0, 1.0))
                force = float(np.clip(float(action_val[2]), -1.0, 1.0))
                return np.array([steer, dist, force], dtype=np.float32)
            except Exception:
                return default_pick_action
        elif isinstance(action_val, dict):
            try:
                steer = float(np.clip(float(action_val.get("steer", 0.0)), -1.0, 1.0))
                dist = float(np.clip(float(action_val.get("distance", 0.5)), -1.0, 1.0))
                force = float(np.clip(float(action_val.get("force", -1.0)), -1.0, 1.0))
                return np.array([steer, dist, force], dtype=np.float32)
            except Exception:
                return default_pick_action
        return default_pick_action

    # Discrete environments
    act_str = str(action_val).strip().lower().replace(" ", "_") if action_val is not None else ""
    
    # Check if direct integer was provided
    if act_str.isdigit():
        return int(act_str)

    if env_name == "SimplePickup":
        discrete_map = {
            "turn_left": 0, "left": 0, "rotate_left": 0,
            "turn_right": 1, "right": 1, "rotate_right": 1,
            "move_forward": 2, "forward": 2, "move": 2, "step_forward": 2,
            "pickup": 3, "pick_up": 3, "pick": 3
        }
        return discrete_map.get(act_str, 2)  # default: move_forward

    elif env_name.startswith("MiniWorld"):
        pickup_idx = 3
        if env_actions is not None and hasattr(env_actions, "pickup"):
            pickup_idx = env_actions.pickup.value
        discrete_map = {
            "turn_left": 0, "left": 0, "rotate_left": 0,
            "turn_right": 1, "right": 1, "rotate_right": 1,
            "move_forward": 2, "forward": 2, "move": 2, "step_forward": 2,
            "pickup": pickup_idx, "pick_up": pickup_idx, "pick": pickup_idx
        }
        return discrete_map.get(act_str, 2)  # default: move_forward

    elif env_name == "MiniGridRelational":
        discrete_map = {
            "turn_left": 0, "left": 0, "rotate_left": 0,
            "turn_right": 1, "right": 1, "rotate_right": 1,
            "move_forward": 2, "forward": 2, "move": 2, "step_forward": 2,
            "pickup": 3, "pick_up": 3, "pick": 3,
            "drop": 4, "put_down": 4, "place": 4
        }
        return discrete_map.get(act_str, 2)  # default: move_forward

    return 0

def get_env_llm_actor_description(env):
    """
    Extracts the dedicated llm_actor_env_description from the environment.
    """
    unwrapped = getattr(env, "unwrapped", env)
    for target in (unwrapped, env):
        if hasattr(target, "llm_actor_env_description"):
            val = getattr(target, "llm_actor_env_description")
            return val() if callable(val) else val
        if hasattr(target, "_get_llm_actor_env_description"):
            val = getattr(target, "_get_llm_actor_env_description")
            return val() if callable(val) else val
    # Fallback to standard env_description
    for target in (unwrapped, env):
        if hasattr(target, "env_description"):
            val = getattr(target, "env_description")
            return val() if callable(val) else val
        if hasattr(target, "_get_environment_description"):
            val = getattr(target, "_get_environment_description")
            return val() if callable(val) else val
    return "Complete the assigned mission."

@hydra.main(version_base=None, config_path="../config", config_name="test_imagination")
def main(args: DictConfig) -> None:
    # Suppress verbose HTTP logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    evaluated_env_name = args.env.name
    task_mode = args.env.get("task_mode", "unknown")

    # Load environment
    if args.env.name == "SimplePickup":
        args.env.verbose = True
        from env.SimplePickup import SimplePickup
        env = SimplePickup(args.env)
        from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        env = ImgObsWrapper(env)
    elif args.env.name == "PickEnv":
        args.env.verbose = True
        from env.PickEnv import PickEnv
        env = PickEnv(args.env)
    elif args.env.name.startswith("MiniWorld"):
        args.env.verbose = True
        from env.MiniWorld import PickObjectEnv
        env = PickObjectEnv(args.env)
    elif args.env.name == "MiniGridRelational":
        args.env.verbose = True
        from env.MiniGridRelational import RelationalPickPlaceEnv
        env = RelationalPickPlaceEnv(args.env)
        from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
        env = RGBImgObsWrapper(env, tile_size=args.env.tile_size)
        env = ImgObsWrapper(env)
    else:
        raise NotImplementedError(f"Environment {args.env.name} is not supported.")

    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "task_mode"):
        task_mode = env.unwrapped.task_mode
    elif hasattr(env, "task_mode"):
        task_mode = env.task_mode

    # Configure reasoning model: strictly local Gemma 4 model
    llm_model = args.get("llm_model", "google/gemma-4-12B-it")
    agent_display_name = f"LLM_Actor_{args.get('agent_name', 'Agent')}"
    base_seed = args.get("seed", 42)
    num_episodes = args.get("num_episode", 10)
    performance_md_file = args.get("performance_md_file", "Results/baselines/llm_actor_gemma.md")
    if performance_md_file == "Results/agent_performance.md":
        performance_md_file = "Results/baselines/llm_actor_gemma.md"
    cache_file = args.get("cache_file", "Results/baselines/llm_actor_cache_gemma.json")

    print("================================================================================")
    print(f"[LLM ACTOR GEMMA 4 BASELINE] Starting Evaluation")
    print(f"Env:        {evaluated_env_name} ({task_mode})")
    print(f"Seed:       {base_seed}")
    print(f"Episodes:   {num_episodes}")
    print(f"Model:      {llm_model} (Local HuggingFace Pipeline)")
    print(f"Report:     {performance_md_file}")
    print(f"Cache:      {cache_file}")
    print("================================================================================")

    # Resume check
    completed_episodes = {}
    if performance_md_file:
        completed_episodes = get_completed_episodes_from_cache(
            md_file_path=performance_md_file,
            env_name=evaluated_env_name,
            task_mode=task_mode,
            seed=base_seed,
            agent_name=agent_display_name,
            cache_file_path=cache_file
        )

    if len(completed_episodes) >= num_episodes:
        print(f"[RESUME] All {num_episodes} episodes for {evaluated_env_name} ({task_mode}) seed {base_seed} already completed. Skipping.")
        return

    # Initialize local HuggingFace Gemma pipeline
    pipe = initialize_llm_hf_pipeline(llm_model)

    env_desc = get_env_llm_actor_description(env)

    scores = []
    episode_records_map = {}
    prior_perf = {}

    for ep_idx in sorted(completed_episodes.keys()):
        if ep_idx < num_episodes:
            ep_data = completed_episodes[ep_idx]
            scores.append(ep_data.get("score", 0.0))
            episode_records_map[f"episode_{ep_idx}"] = ep_data
            if "performance" in ep_data:
                prior_perf = merge_performances(prior_perf, ep_data["performance"])

    def get_current_env_metric():
        unwrapped = getattr(env, "unwrapped", env)
        if hasattr(unwrapped, "get_performance_metric"):
            m = unwrapped.get_performance_metric()
        elif hasattr(env, "get_performance_metric"):
            m = env.get_performance_metric()
        else:
            m = {}
        return copy.deepcopy(m) if m else {}

    # Run episodes
    for episode in range(num_episodes):
        if episode in completed_episodes:
            continue

        episode_seed = (base_seed + episode) if base_seed is not None else None
        print(f"\n----------- Starting Episode {episode}/{num_episodes} (seed: {episode_seed}) ----------------", flush=True)
        
        env_metric_before_ep = get_current_env_metric()
        state, info = env.reset(seed=episode_seed)
        
        # Get target mission
        mission = getattr(env.unwrapped, "mission", getattr(env, "mission", args.env.get("mission", "")))
        
        cumulative_reward = 0.0
        done = False
        episode_step = 0
        ep_prompt_tokens = 0
        ep_completion_tokens = 0
        ep_total_tokens = 0
        ep_latencies = []

        unwrapped = getattr(env, "unwrapped", env)
        env_actions = getattr(unwrapped, "actions", getattr(env, "actions", None))

        while not done:
            episode_step += 1
            obs_desc = info.get("description", "No description available.")

            # Unified memoryless prompt across all environments
            prompt_content = (
                f"Environment description:\n{env_desc}\n\n"
                f"Target task: {mission}\n"
                f"Current observation: {obs_desc}\n\n"
                "Based on the environment description and current observation, choose the single best action to achieve the target task.\n"
                "Output your decision strictly as a JSON object:\n"
                "{\n"
                '  "reasoning": "<concise spatial analysis and plan>",\n'
                '  "action": <action>\n'
                "}"
            )

            # Query local Gemma model only (no fallbacks)
            t0 = time.time()
            llm_reply, reasoning = query_llm(
                system="You are an autonomous decision-making agent acting directly in the environment. Output only valid JSON.",
                prompt=prompt_content,
                api_key=None,
                pipeline=pipe,
                alternative_pipe=None,
                mode="huggingface",
                secondary_api_model=None,
                max_tokens=int(args.get("max_tokens", 6000))
            )
            dt = time.time() - t0
            ep_latencies.append(dt)

            # Token tracking
            if isinstance(reasoning, dict) and reasoning.get("usage"):
                usage = reasoning["usage"]
                p_tok = usage.get("prompt_tokens", 0)
                c_tok = usage.get("completion_tokens", 0)
                ep_prompt_tokens += p_tok
                ep_completion_tokens += c_tok
                ep_total_tokens += (p_tok + c_tok)

            # Parse and execute action
            action = parse_action_response(llm_reply, evaluated_env_name, env_actions)
            
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"[STEP ERROR] Failed step with action {action}: {e}", flush=True)
                break

            cumulative_reward += float(reward)
            done = bool(terminated or truncated)
            reasoning_snippet = ""
            try:
                raw_json = re.sub(r"<turn\|>|<end_of_turn>|<\|end\|>|<\|return\|>", "", llm_reply.strip()).strip()
                raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json, flags=re.MULTILINE)
                raw_json = re.sub(r"\s*```$", "", raw_json, flags=re.MULTILINE).strip()
                data = json.loads(raw_json)
                reasoning_snippet = data.get("reasoning", "")
            except Exception:
                pass

            if reasoning_snippet:
                snip_disp = str(reasoning_snippet).strip().replace("\n", " ")
                if len(snip_disp) > 65:
                    snip_disp = snip_disp[:62] + "..."
                print(f"  [Step {episode_step}] Action: {action} | Reasoning: {snip_disp!r} | Reward: {reward} | Done: {done} | Latency: {dt:.2f}s", flush=True)
            else:
                clean_reply = llm_reply.strip().replace("\n", " ")
                if len(clean_reply) > 80:
                    clean_reply = clean_reply[:77] + "..."
                print(f"  [Step {episode_step}] Action: {action} (raw: {clean_reply!r}) | Reward: {reward} | Done: {done} | Latency: {dt:.2f}s", flush=True)

        scores.append(cumulative_reward)
        running_average_score = float(np.mean(scores))
        
        env_metric_after_ep = get_current_env_metric()
        ep_metric_delta = compute_performance_delta(env_metric_after_ep, env_metric_before_ep)
        prior_perf = merge_performances(prior_perf, ep_metric_delta)

        # Store episode record
        mean_lat = float(np.mean(ep_latencies)) if ep_latencies else 0.0
        ep_record = {
            "episode": episode,
            "score": cumulative_reward,
            "timesteps": episode_step,
            "performance": ep_metric_delta,
            "token_usage": {
                "prompt_tokens": ep_prompt_tokens,
                "completion_tokens": ep_completion_tokens,
                "total_tokens": ep_total_tokens,
            },
            "latency": mean_lat
        }
        episode_records_map[f"episode_{episode}"] = ep_record

        # Compute running totals for performance md
        running_perf = copy.deepcopy(prior_perf)
        running_perf["running_average_score"] = running_average_score
        running_perf["total_tokens"] = sum(
            rec.get("token_usage", {}).get("total_tokens", 0) for rec in episode_records_map.values()
        )
        all_lats = [rec.get("latency", 0.0) for rec in episode_records_map.values() if rec.get("latency", 0.0) > 0]
        mean_lat_val = round(float(np.mean(all_lats)), 2) if all_lats else 0.0
        running_perf["mean_net_latency"] = mean_lat_val
        running_perf["avg_llm_net_latency"] = mean_lat_val
        running_perf["avg_llm_response_time"] = mean_lat_val

        print(f"----------- Episode {episode}/{num_episodes} Done | Score: {cumulative_reward:.4f} | Running Avg: {running_average_score:.4f} | Steps: {episode_step} | Latency: {mean_lat:.2f}s | Tokens: {ep_total_tokens} -----------", flush=True)

        # Real-time persistence into cache and markdown
        if performance_md_file:
            try:
                append_or_update_metric(
                    md_file_path=performance_md_file,
                    env_name=evaluated_env_name,
                    task_mode=task_mode,
                    seed=base_seed,
                    agent_name=agent_display_name,
                    performance=running_perf,
                    engine_name=llm_model,
                    num_episodes=num_episodes,
                    episode_records=episode_records_map,
                    cache_file_path=cache_file
                )
            except Exception as e:
                print(f"[WARNING] Failed to persist progress to markdown: {e}", flush=True)

    print(f"\n[LLM ACTOR GEMMA 4 BASELINE] Successfully finished all {num_episodes} episodes for {evaluated_env_name} ({task_mode}). Final Score: {running_average_score:.4f}\n", flush=True)

if __name__ == "__main__":
    main()
