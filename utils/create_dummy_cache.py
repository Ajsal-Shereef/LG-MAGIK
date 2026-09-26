import os
import json
from datetime import datetime
from utils.update_performance_md import append_or_update_metric

def generate_dummy_cache(output_md_path: str):
    """
    Generates a realistic dummy cache file using append_or_update_metric
    to demonstrate the exact structure after running test_imagination.
    """
    cache_path = os.path.splitext(output_md_path)[0] + "_cache.json"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # -------------------------------------------------------------
    # Scenario 1: SimplePickup (Target 1, Seed 42) - Fully Completed (10 episodes)
    # -------------------------------------------------------------
    episodes_simple = {}
    scores_simple = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0] # 9 successes, 1 failure
    running_scores_simple = []
    
    for i, sc in enumerate(scores_simple):
        running_scores_simple.append(sc)
        is_succ = (sc == 1.0)
        episodes_simple[f"episode_{i}"] = {
            "episode": i,
            "seed": 42 + i,
            "score": float(sc),
            "running_average_score": round(sum(running_scores_simple) / len(running_scores_simple), 4),
            "is_success": is_succ,
            "total_steps": 14 if is_succ else 40,
            "imagination_steps": 12 if is_succ else 35,
            "llm_errors": 0 if is_succ else 1,
            "vae_errors": 0,
            "avg_llm_response_time": 0.38,
            "failure_cause": "SUCCESS" if is_succ else "LLM_FAILURE",
            "performance": {
                "rewarding_objects": {"purple box": 1 if is_succ else 0},
                "non_rewarding_objects": {"green ball": 0}
            }
        }

    perf_simple = {
        "rewarding_objects": {"purple box": 9},
        "non_rewarding_objects": {"green ball": 0},
        "running_average_score": 0.9,
        "success_rate": 90.0,
        "total_imagination_steps": 143,
        "llm_error_steps": 1,
        "vae_error_steps": 0,
        "llm_error_rate_pct": 0.7,
        "vae_error_rate_pct": 0.0,
        "failure_breakdown": {
            "LLM_FAILURE": 1,
            "VAE_FAILURE": 0,
            "DQN_POLICY_FAILURE": 0
        }
    }

    append_or_update_metric(
        md_file_path=output_md_path,
        env_name="SimplePickup",
        task_mode="target1",
        seed=42,
        agent_name="DQN",
        performance=perf_simple,
        engine_name="google/gemma-4-12B-it",
        num_episodes=10,
        episode_records=episodes_simple
    )

    # -------------------------------------------------------------
    # Scenario 2: MiniWorld (Target 1, Seed 42) - Fully Completed (10 episodes)
    # -------------------------------------------------------------
    episodes_miniworld = {}
    scores_mw = [12.74, 13.23, 12.35, 13.08, 13.94, 12.95, 11.57, 13.35, 12.42, 10.14]
    running_scores_mw = []

    for i, sc in enumerate(scores_mw):
        running_scores_mw.append(sc)
        is_succ = (sc > 5.0)
        episodes_miniworld[f"episode_{i}"] = {
            "episode": i,
            "seed": 42 + i,
            "score": float(sc),
            "running_average_score": round(sum(running_scores_mw) / len(running_scores_mw), 4),
            "is_success": is_succ,
            "total_steps": 65,
            "imagination_steps": 65,
            "llm_errors": 0,
            "vae_errors": 0,
            "avg_llm_response_time": 0.42,
            "failure_cause": "SUCCESS",
            "performance": {
                "rewarding_objects": {"duckie": 1},
                "non_rewarding_objects": {"ball": 0}
            }
        }

    perf_miniworld = {
        "rewarding_objects": {"duckie": 10},
        "non_rewarding_objects": {"ball": 0},
        "running_average_score": round(sum(scores_mw) / len(scores_mw), 4),
        "success_rate": 100.0,
        "total_imagination_steps": 650,
        "llm_error_steps": 0,
        "vae_error_steps": 0,
        "llm_error_rate_pct": 0.0,
        "vae_error_rate_pct": 0.0,
        "failure_breakdown": {
            "LLM_FAILURE": 0,
            "VAE_FAILURE": 0,
            "DQN_POLICY_FAILURE": 0
        }
    }

    append_or_update_metric(
        md_file_path=output_md_path,
        env_name="MiniWorld",
        task_mode="target1",
        seed=42,
        agent_name="PPO",
        performance=perf_miniworld,
        engine_name="google/gemma-4-12B-it",
        num_episodes=10,
        episode_records=episodes_miniworld
    )

    # -------------------------------------------------------------
    # Scenario 3: PickEnv (Target, Seed 42) - Partially Completed (3 episodes in progress)
    # Demonstrating how the cache looks when interrupted / ready to resume!
    # -------------------------------------------------------------
    episodes_pick = {}
    scores_pick = [1.0, 0.0, 1.0] # 3 episodes done so far
    running_scores_pick = []

    for i, sc in enumerate(scores_pick):
        running_scores_pick.append(sc)
        is_succ = (sc == 1.0)
        episodes_pick[f"episode_{i}"] = {
            "episode": i,
            "seed": 42 + i,
            "score": float(sc),
            "running_average_score": round(sum(running_scores_pick) / len(running_scores_pick), 4),
            "is_success": is_succ,
            "total_steps": 25,
            "imagination_steps": 25,
            "llm_errors": 0 if is_succ else 1,
            "vae_errors": None, # PickEnv skips VAE analysis
            "avg_llm_response_time": 0.35,
            "failure_cause": "SUCCESS" if is_succ else "LLM_FAILURE",
            "performance": {
                "picked": 1 if is_succ else 0,
                "brocken": 0 if is_succ else 1
            }
        }

    perf_pick = {
        "picked": 2,
        "brocken": 1,
        "running_average_score": 0.6667,
        "success_rate": 66.7,
        "total_imagination_steps": 75,
        "llm_error_steps": 1,
        "vae_error_steps": None,
        "llm_error_rate_pct": 1.33,
        "vae_error_rate_pct": None,
        "failure_breakdown": {
            "LLM_FAILURE": 1,
            "VAE_FAILURE": 0,
            "DQN_POLICY_FAILURE": 0
        }
    }

    append_or_update_metric(
        md_file_path=output_md_path,
        env_name="PickEnv",
        task_mode="target",
        seed=42,
        agent_name="SAC",
        performance=perf_pick,
        engine_name="google/gemma-4-12B-it",
        num_episodes=10,
        episode_records=episodes_pick
    )

    print(f"Generated dummy cache at: {cache_path}")

if __name__ == "__main__":
    generate_dummy_cache("Results/agent_performance_cache_run.md")
    # Also create copy as Results/agent_performance_cache_run.json
    cache_json = "Results/agent_performance_cache_run.json"
    print(f"Ready: {cache_json}")
