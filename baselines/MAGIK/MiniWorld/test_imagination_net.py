import os
import sys

MINIWORLD_MAGIK_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MINIWORLD_MAGIK_DIR, "../../.."))
while sys.path and os.path.abspath(sys.path[0]) == MINIWORLD_MAGIK_DIR:
    sys.path.pop(0)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MINIWORLD_MAGIK_DIR not in sys.path:
    sys.path.append(MINIWORLD_MAGIK_DIR)

import copy
import glob
import json
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from baselines.MAGIK.MiniWorld.helper_functions.utils import (
    Dataset,
    get_magik_class,
    magik_imagine_state,
    load_miniworld_magik_dataset,
    resolve_project_path,
    MAGIK_CLASS_NAMES,
)
from baselines.MAGIK.MiniWorld.helper_functions.visualize import Visualizer
from baselines.MAGIK.MiniWorld.architectures.m2_vae.dgm import DeepGenerativeModel
from utils.update_performance_md import (
    get_completed_episodes_from_cache,
    append_or_update_metric,
    compute_performance_delta,
    merge_performances,
)
from architectures.common_utils import save_gif


def _resolve_vae_checkpoint(vae_ckpt_cfg: str, env_name: str, seed: int) -> str:
    """
    Resolves the MAGIK M2-VAE checkpoint path, searching common fallback locations if needed.
    """
    primary = resolve_project_path(vae_ckpt_cfg)
    if os.path.isfile(primary):
        return primary
    if os.path.isdir(primary) and os.path.isfile(os.path.join(primary, "model.pt")):
        return os.path.join(primary, "model.pt")

    search_patterns = [
        resolve_project_path(f"model_weights/{env_name}/MAGIK/m2_vae/seed_{seed}/model.pt"),
        resolve_project_path(f"model_weights/{env_name}/MAGIK/m2_vae/*/model.pt"),
        resolve_project_path("models/m2_vae/*/model.pt"),
    ]
    for pattern in search_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return primary


def _load_sb3_agent(agent_name: str, checkpoint_path: str):
    ckpt = resolve_project_path(checkpoint_path)
    name_upper = str(agent_name).upper()
    if "SAC" in name_upper:
        from stable_baselines3 import SAC
        return SAC.load(ckpt)
    elif "PPO" in name_upper:
        from stable_baselines3 import PPO
        return PPO.load(ckpt)
    elif "DQN" in name_upper:
        from stable_baselines3 import DQN
        return DQN.load(ckpt)
    else:
        raise ValueError(f"Unsupported RL agent_name: {agent_name}. Expected PPO, DQN, or SAC.")


@hydra.main(version_base=None, config_path="config", config_name="MiniWorld")
def main(args: DictConfig) -> None:
    evaluated_env_name = str(args.General.get("env", "MiniWorld"))
    task_mode = str(args.General.get("task_mode", "target3"))
    is_agent = bool(args.Test.get("is_agent", False))
    policy_mode = str(args.Test.get("policy_mode", "single_objects"))
    distractor_mode = str(args.Test.get("distractor_mode", "empty"))
    num_test_env = int(args.Test.get("num_episode", 10))
    base_seed = int(args.Test.get("seed", args.General.get("seed", 42)))
    agent_name = str(args.Test.get("agent_name", "PPO"))
    deterministic = bool(args.Test.get("deterministic", False))

    print("*******************************************************************")
    if is_agent:
        print(f"Selected testing mode is: DIRECT {agent_name} AGENT (No Imagination)")
    else:
        print(f"Selected testing mode is: MAGIK IMAGINATION ({policy_mode}, distractor={distractor_mode})")
    print(f"Environment: {evaluated_env_name} | Task Mode: {task_mode} (Case 3) | Seed: {base_seed}")
    print("*******************************************************************")

    # Initialize MiniWorld Case 3 environment
    from env.MiniWorld import PickObjectEnv
    env_cfg = OmegaConf.to_container(args.General, resolve=True)
    env_cfg["name"] = evaluated_env_name
    env_cfg["task_mode"] = task_mode
    env = PickObjectEnv(env_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    # Load pre-trained source RL policy (PPO trained on MiniWorld source task: pick blue box, avoid green ball)
    agent_ckpt = resolve_project_path(args.Test.agent_checkpoint)
    print(f"[INFO] Loading pre-trained source {agent_name} agent from: {agent_ckpt}")
    agent = _load_sb3_agent(agent_name, agent_ckpt)

    # Load trained MAGIK M2-VAE model
    model = None
    if not is_agent:
        vae_ckpt = _resolve_vae_checkpoint(
            args.Test.vae_checkpoint, evaluated_env_name, int(args.General.get("seed", base_seed))
        )
        model = DeepGenerativeModel(args.Network).to(device)
        model.load(vae_ckpt)
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    agent_display_name = f"Direct_{agent_name}" if is_agent else f"MAGIK_{agent_name}"
    performance_md_file = resolve_project_path(
        args.Test.get("performance_md_file", "Results/baselines/magik.md")
    )
    cache_file = resolve_project_path(
        args.Test.get("cache_file", "Results/baselines/magik_cache.json")
    )

    completed_episodes = {}
    if performance_md_file:
        completed_episodes = get_completed_episodes_from_cache(
            md_file_path=performance_md_file,
            env_name=evaluated_env_name,
            task_mode=task_mode,
            seed=base_seed,
            agent_name=agent_display_name,
            cache_file_path=cache_file,
        )

    if len(completed_episodes) >= num_test_env:
        print(
            f"[RESUME] All {num_test_env} episodes for {evaluated_env_name} ({task_mode}) "
            f"seed {base_seed} already completed in {cache_file}. Skipping execution.",
            flush=True,
        )
        env.close()
        return

    scores = []
    diagnostic_records = []
    episode_records_map = {}
    prior_perf = {}

    for ep_idx in sorted(completed_episodes.keys()):
        if ep_idx < num_test_env:
            ep_data = completed_episodes[ep_idx]
            scores.append(ep_data.get("score", 0.0))
            diagnostic_records.append(ep_data)
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

    total_cls_correct = 0
    total_cls_evaluated = 0

    for episode in range(num_test_env):
        if episode in completed_episodes:
            continue

        frame_array_partial = []
        frame_array_full = []
        episode_seed = (base_seed + episode) if base_seed is not None else None
        print(
            f"----------- Starting Episode {episode}/{num_test_env} (seed: {episode_seed}) ----------------",
            flush=True,
        )

        env_metric_before_ep = get_current_env_metric()
        state, info = env.reset(seed=episode_seed)
        frame_array_full.append(env.unwrapped.get_frame())

        cumulative_reward = 0.0
        episode_step = 0
        done = False
        ep_imagination_steps = 0
        ep_cls_errors = 0

        while not done:
            episode_step += 1
            gt_cls_vec = get_magik_class(state)
            gt_cls_idx = int(np.argmax(gt_cls_vec))

            if is_agent:
                imagined_state = state
                pred_cls_idx = gt_cls_idx
                did_transform = False
            else:
                imagined_state, pred_cls_idx, did_transform = magik_imagine_state(
                    model=model,
                    state_uint8=state,
                    device=device,
                    policy_mode=policy_mode,
                    distractor_mode=distractor_mode,
                )
                total_cls_evaluated += 1
                if pred_cls_idx == gt_cls_idx:
                    total_cls_correct += 1
                else:
                    ep_cls_errors += 1

                if did_transform or gt_cls_idx != 3:
                    ep_imagination_steps += 1

            action, _ = agent.predict(imagined_state, deterministic=deterministic)
            next_state, reward, terminated, truncated, info = env.step(action)

            frame_array_partial.append(np.hstack((state, imagined_state)))
            frame_array_full.append(env.unwrapped.get_frame())

            cumulative_reward += float(reward)
            done = bool(terminated or truncated)
            state = next_state

        scores.append(cumulative_reward)
        running_average_score = float(np.mean(scores))

        result_root = resolve_project_path(args.Test.get("result_dir", "result"))
        mode_subdir = "source" if is_agent else "transfer"
        save_dir = os.path.join(result_root, agent_display_name, evaluated_env_name, task_mode, mode_subdir)
        fps = float(args.General.get("fps", 2.5))
        try:
            save_gif(frame_array_partial, episode, save_dir, fps=fps, save_name=" partial")
            save_gif(frame_array_full, episode, save_dir, fps=fps, save_name=" full")
        except Exception as e:
            print(f"[WARNING] Could not save rollout GIFs: {e}", flush=True)

        env_metric_after_ep = get_current_env_metric()
        ep_perf_delta = compute_performance_delta(env_metric_after_ep, env_metric_before_ep)
        prior_perf = merge_performances(prior_perf, ep_perf_delta)

        is_success = cumulative_reward > 5.0
        if is_success:
            failure_cause = "SUCCESS"
        elif ep_cls_errors > 0:
            failure_cause = "VAE_FAILURE"
        else:
            failure_cause = "DQN_POLICY_FAILURE"

        ep_diag = {
            "episode": episode,
            "seed": episode_seed,
            "score": float(cumulative_reward),
            "running_average_score": running_average_score,
            "is_success": bool(is_success),
            "total_steps": episode_step,
            "imagination_steps": ep_imagination_steps,
            "llm_errors": 0,
            "vae_errors": ep_cls_errors,
            "failure_cause": failure_cause,
            "performance": ep_perf_delta,
        }
        diagnostic_records.append(ep_diag)
        episode_records_map[f"episode_{episode}"] = ep_diag

        rew_picked = ep_perf_delta.get("rewarding_objects", {})
        non_rew_picked = ep_perf_delta.get("non_rewarding_objects", {})
        print(
            f"----------- Episode {episode}/{num_test_env} Done | Score: {cumulative_reward:.4f} | "
            f"Running Avg: {running_average_score:.4f} | Target Picked: {rew_picked} | "
            f"Distractor Picked: {non_rew_picked} | Classifier Errors: {ep_cls_errors}/{episode_step} -----------",
            flush=True,
        )

        # Compute running summary metrics
        total_eps = len(diagnostic_records)
        successes = sum(1 for d in diagnostic_records if d.get("is_success"))
        cause_counts = {
            "LLM_FAILURE": 0,
            "VAE_FAILURE": sum(1 for d in diagnostic_records if d.get("failure_cause") == "VAE_FAILURE"),
            "DQN_POLICY_FAILURE": sum(
                1 for d in diagnostic_records if d.get("failure_cause") == "DQN_POLICY_FAILURE"
            ),
        }
        total_imag_steps = sum(d.get("imagination_steps", 0) for d in diagnostic_records)
        total_vae_errs = sum(d.get("vae_errors", 0) for d in diagnostic_records)
        vae_err_pct = (total_vae_errs / total_imag_steps * 100.0) if total_imag_steps > 0 else 0.0

        running_perf = copy.deepcopy(prior_perf)
        running_perf["running_average_score"] = running_average_score
        running_perf["success_rate"] = round((successes / total_eps * 100.0) if total_eps > 0 else 0.0, 1)
        running_perf["total_imagination_steps"] = total_imag_steps
        running_perf["llm_error_steps"] = 0
        running_perf["vae_error_steps"] = total_vae_errs
        running_perf["llm_error_rate_pct"] = 0.0
        running_perf["vae_error_rate_pct"] = round(vae_err_pct, 2)
        running_perf["failure_breakdown"] = cause_counts

        if performance_md_file:
            try:
                append_or_update_metric(
                    md_file_path=performance_md_file,
                    env_name=evaluated_env_name,
                    task_mode=task_mode,
                    seed=base_seed,
                    agent_name=agent_display_name,
                    performance=running_perf,
                    engine_name="MAGIK (M2-VAE)",
                    num_episodes=num_test_env,
                    episode_records=episode_records_map,
                    cache_file_path=cache_file,
                )
            except Exception as e:
                print(f"[WARNING] Failed to update episode progress in {performance_md_file}: {e}", flush=True)

    # Final summary printout
    rew_summary = prior_perf.get("rewarding_objects", {})
    non_rew_summary = prior_perf.get("non_rewarding_objects", {})
    final_avg_score = float(np.mean(scores)) if scores else 0.0
    cls_acc = (total_cls_correct / total_cls_evaluated * 100.0) if total_cls_evaluated > 0 else 0.0

    print("\n================ MAGIK BASELINE EVALUATION SUMMARY ================", flush=True)
    print(f"Environment:               {evaluated_env_name} ({task_mode} - Case 3)", flush=True)
    print(f"Seed:                      {base_seed}", flush=True)
    print(f"Agent:                     {agent_display_name}", flush=True)
    print(f"Rewarding Picked (ball):   {rew_summary} (out of {num_test_env})", flush=True)
    print(f"Distractor Picked (box):   {non_rew_summary} (out of {num_test_env})", flush=True)
    print(f"Mean Episode Score:        {final_avg_score:.4f}", flush=True)
    if total_cls_evaluated > 0:
        print(f"M2-VAE Classifier Acc:     {cls_acc:.2f}% ({total_cls_correct}/{total_cls_evaluated} steps)", flush=True)
    print(f"Performance Report:        {performance_md_file}", flush=True)
    print("===================================================================\n", flush=True)

    # Save diagnostic JSON
    diag_dir = resolve_project_path("Results/diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    diag_file = os.path.join(diag_dir, f"MAGIK_{evaluated_env_name}_{task_mode}_seed_{base_seed}.json")
    with open(diag_file, "w") as f:
        json.dump(
            {
                "env_name": evaluated_env_name,
                "task_mode": task_mode,
                "seed": base_seed,
                "agent_name": agent_display_name,
                "total_episodes": len(diagnostic_records),
                "mean_score": final_avg_score,
                "classifier_accuracy_pct": round(cls_acc, 2),
                "performance": prior_perf,
                "episodes": diagnostic_records,
            },
            f,
            indent=2,
        )

    # Optional latent swap visualization (matching original test_imagination_net.py)
    if not is_agent and model is not None and args.Test.get("run_Swap_viz", False):
        try:
            datasets, _ = load_miniworld_magik_dataset(
                datapath=args.General.datapath,
                env_name=evaluated_env_name,
                collect_if_missing=False,
                cache_pkl=False,
            )
            viz_dir = resolve_project_path("visualizations/m2_vae")
            viz = Visualizer(model, device, root=f"{viz_dir}/")
            data = Dataset(datasets)
            data_loader = DataLoader(data, batch_size=min(10, len(data)), shuffle=True)
            for batch in data_loader:
                viz.swap(images=batch[0], device=device)
                break
        except Exception as e:
            print(f"[INFO] Skipping optional latent swap visualization: {e}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
