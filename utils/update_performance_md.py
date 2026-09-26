import os
import json
from datetime import datetime
import numpy as np

def format_metric_value(val):
    if isinstance(val, dict):
        # E.g. {"purple box": 8} -> "purple box: 8" or sum if multiple
        items = [f"{k}: {v}" for k, v in val.items()]
        return ", ".join(items) if items else "0"
    elif isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def extract_summary_val(val):
    if isinstance(val, dict):
        return sum(v for v in val.values() if isinstance(v, (int, float)))
    elif isinstance(val, (int, float)):
        return val
    return 0

def format_failure_breakdown(fb, env_name=None):
    if env_name == "MiniWorldNoisy" or fb is None:
        return "-"
    if not isinstance(fb, dict):
        return "-"
    llm = fb.get("LLM_FAILURE", fb.get("LLM_MAPPING_FAILURE", 0))
    vae = fb.get("VAE_FAILURE", fb.get("VAE_ARTIFACT_FAILURE", 0))
    pol = fb.get("DQN_POLICY_FAILURE", fb.get("POLICY_NAVIGATION_FAILURE", 0))
    if env_name == "PickEnv":
        return f"{llm} / - / {pol}"
    return f"{llm} / {vae} / {pol}"

def format_err_pct(val, is_disabled=False):
    if is_disabled or val is None or val == "-":
        return "-"
    if isinstance(val, (int, float)):
        return f"{val:.1f}%"
    return str(val)

def compute_performance_delta(after_perf, before_perf):
    """
    Computes delta of metrics earned in a single episode.
    """
    import copy
    if not after_perf:
        return {}
    if not before_perf:
        return copy.deepcopy(after_perf)
    delta = {}
    for k, v in after_perf.items():
        if isinstance(v, dict):
            delta[k] = {}
            before_sub = before_perf.get(k, {})
            if isinstance(before_sub, dict):
                for sub_k, sub_v in v.items():
                    delta[k][sub_k] = sub_v - before_sub.get(sub_k, 0)
            else:
                delta[k] = copy.deepcopy(v)
        elif isinstance(v, (int, float)):
            delta[k] = v - before_perf.get(k, 0)
        else:
            delta[k] = v
    return delta

def merge_performances(base_perf, delta_perf):
    """
    Merges delta metrics into base performance accumulator.
    """
    import copy
    if not delta_perf:
        return copy.deepcopy(base_perf) if base_perf else {}
    if not base_perf:
        return copy.deepcopy(delta_perf)
    merged = copy.deepcopy(base_perf)
    for k, v in delta_perf.items():
        if isinstance(v, dict):
            if k not in merged or not isinstance(merged[k], dict):
                merged[k] = {}
            for sub_k, sub_v in v.items():
                merged[k][sub_k] = merged[k].get(sub_k, 0) + sub_v
        elif isinstance(v, (int, float)):
            merged[k] = merged.get(k, 0) + v
        else:
            merged[k] = v
    return merged

def get_completed_episodes_from_cache(
    md_file_path: str,
    env_name: str,
    task_mode: str,
    seed: int,
    agent_name: str
) -> dict:
    """
    Returns a dict mapping episode_idx (int) -> episode_data (dict)
    for already recorded episodes in the cache file.
    """
    cache_file = os.path.splitext(md_file_path)[0] + "_cache.json"
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r") as f:
            records = json.load(f)
        for r in records:
            if (
                r.get("env_name") == env_name
                and r.get("task_mode") == task_mode
                and r.get("seed") == seed
                and r.get("agent_name") == agent_name
            ):
                ep_dict = {}
                for k, v in r.items():
                    if k.startswith("episode_"):
                        try:
                            ep_idx = int(k.split("_")[1])
                            ep_dict[ep_idx] = v
                        except (ValueError, IndexError):
                            pass
                return ep_dict
    except Exception:
        pass
    return {}

def append_or_update_metric(
    md_file_path: str,
    env_name: str,
    task_mode: str,
    seed: int,
    agent_name: str,
    performance: dict,
    engine_name: str = "google/gemma-4-12B-it",
    num_episodes: int = 10,
    episode_records: dict = None
):
    """
    Appends or updates a record in the markdown report and its companion cache.
    Preserves all existing episode_{i} keys and merges new ones.
    Re-renders the markdown file with clean tables.
    """
    os.makedirs(os.path.dirname(os.path.abspath(md_file_path)), exist_ok=True)
    cache_file = os.path.splitext(md_file_path)[0] + "_cache.json"

    # Load cache
    records = []
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Update or insert record
    target_idx = -1
    for i, r in enumerate(records):
        if (
            r.get("env_name") == env_name
            and r.get("task_mode") == task_mode
            and r.get("seed") == seed
            and r.get("agent_name") == agent_name
        ):
            target_idx = i
            break

    target_record = records[target_idx] if target_idx >= 0 else {}
    target_record["env_name"] = env_name
    target_record["task_mode"] = task_mode
    target_record["seed"] = seed
    target_record["agent_name"] = agent_name
    target_record["performance"] = performance
    target_record["engine_name"] = engine_name
    target_record["num_episodes"] = num_episodes
    target_record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if episode_records:
        target_record.update(episode_records)

    if target_idx >= 0:
        records[target_idx] = target_record
    else:
        records.append(target_record)

    with open(cache_file, "w") as f:
        json.dump(records, f, indent=2)

    # Re-render markdown
    render_markdown_report(md_file_path, records, engine_name, num_episodes)


def render_markdown_report(md_file_path: str, records: list, engine_name: str, num_episodes: int):
    # Sort order for environments
    env_order = ["SimplePickup", "MiniWorld", "MiniWorldNoisy", "PickEnv", "MiniGridRelational"]
    env_display_names = {
        "SimplePickup": "1. MiniGrid (SimplePickup)",
        "MiniWorld": "2. MiniWorld",
        "MiniWorldNoisy": "3. MiniWorldNoisy",
        "PickEnv": "4. PickEnv",
        "MiniGridRelational": "5. MiniGridRelational"
    }

    # Group records by environment
    grouped = {}
    for r in records:
        e = r["env_name"]
        grouped.setdefault(e, []).append(r)

    lines = []
    lines.append("# Agent Performance Evaluation Summary")
    lines.append("")
    lines.append(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Reasoning Engine:** `{engine_name}`")
    lines.append(f"- **Episodes per Scenario:** `{num_episodes}`")
    lines.append(f"- **Total Completed Runs:** `{len(records)}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Master Overview Table
    lines.append("## Overall Performance Summary (Aggregated Across Seeds)")
    lines.append("")
    lines.append("| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) |")
    lines.append("|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|")

    # Order keys in overview
    all_envs = sorted(list(grouped.keys()), key=lambda x: env_order.index(x) if x in env_order else 99)

    for env_k in all_envs:
        env_recs = grouped[env_k]
        # Group by task_mode
        by_target = {}
        for r in env_recs:
            by_target.setdefault(r["task_mode"], []).append(r)

        for target, t_recs in sorted(by_target.items()):
            scores = [r["performance"].get("running_average_score", 0.0) for r in t_recs]
            seeds_done = sorted([r["seed"] for r in t_recs])
            seeds_str = ", ".join(str(s) for s in seeds_done)
            mean_score = np.mean(scores) if scores else 0.0
            std_score = np.std(scores) if len(scores) > 1 else 0.0

            # Step error rates
            llm_rates = [r["performance"].get("llm_error_rate_pct") for r in t_recs if r["performance"].get("llm_error_rate_pct") is not None]
            vae_rates = [r["performance"].get("vae_error_rate_pct") for r in t_recs if r["performance"].get("vae_error_rate_pct") is not None]

            if env_k == "MiniWorldNoisy":
                mean_llm_err = "-"
                mean_vae_err = "-"
                tot_fails_str = "-"
            elif env_k == "PickEnv":
                mean_llm_err = f"{np.mean(llm_rates):.1f}%" if llm_rates else "-"
                mean_vae_err = "-"
                tot_llm = sum(r["performance"].get("failure_breakdown", {}).get("LLM_FAILURE", 0) for r in t_recs if isinstance(r["performance"].get("failure_breakdown"), dict))
                tot_pol = sum(r["performance"].get("failure_breakdown", {}).get("DQN_POLICY_FAILURE", 0) for r in t_recs if isinstance(r["performance"].get("failure_breakdown"), dict))
                tot_fails_str = f"{tot_llm} / - / {tot_pol}"
            else:
                mean_llm_err = f"{np.mean(llm_rates):.1f}%" if llm_rates else "-"
                mean_vae_err = f"{np.mean(vae_rates):.1f}%" if vae_rates else "-"
                tot_llm = sum(r["performance"].get("failure_breakdown", {}).get("LLM_FAILURE", 0) for r in t_recs if isinstance(r["performance"].get("failure_breakdown"), dict))
                tot_vae = sum(r["performance"].get("failure_breakdown", {}).get("VAE_FAILURE", 0) for r in t_recs if isinstance(r["performance"].get("failure_breakdown"), dict))
                tot_pol = sum(r["performance"].get("failure_breakdown", {}).get("DQN_POLICY_FAILURE", 0) for r in t_recs if isinstance(r["performance"].get("failure_breakdown"), dict))
                tot_fails_str = f"{tot_llm} / {tot_vae} / {tot_pol}"

            # Primary metric depending on env
            primary_vals = []
            primary_name = "Success"
            for r in t_recs:
                perf = r["performance"]
                if "rewarding_objects" in perf:
                    primary_vals.append(extract_summary_val(perf["rewarding_objects"]))
                    primary_name = "Reward Objects Picked"
                elif "picked" in perf:
                    primary_vals.append(perf["picked"])
                    primary_name = "Picked"
                elif "successful_drop" in perf:
                    primary_vals.append(perf["successful_drop"])
                    primary_name = "Successful Drop"
                else:
                    primary_vals.append(0)

            mean_primary = np.mean(primary_vals) if primary_vals else 0.0
            lines.append(
                f"| {env_k} | `{target}` | {seeds_str} | **{mean_score:.4f} ± {std_score:.4f}** | {mean_primary:.2f} ({primary_name}) | {mean_llm_err} | {mean_vae_err} | {tot_fails_str} |"
            )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed Per-Environment Tables
    for env_k in all_envs:
        display_title = env_display_names.get(env_k, env_k)
        env_recs = grouped[env_k]
        # Sort by target, then seed
        env_recs = sorted(env_recs, key=lambda x: (str(x["task_mode"]), int(x["seed"])))

        lines.append(f"## {display_title}")
        lines.append("")

        if env_k in ("SimplePickup", "MiniWorld", "MiniWorldNoisy"):
            lines.append("| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Completed At |")
            lines.append("|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|")
            for r in env_recs:
                perf = r["performance"]
                rew = format_metric_value(perf.get("rewarding_objects", {}))
                non_rew = format_metric_value(perf.get("non_rewarding_objects", {}))
                score = perf.get("running_average_score", 0.0)
                if env_k == "MiniWorldNoisy":
                    llm_err = "-"
                    vae_err = "-"
                    fb_str = "-"
                else:
                    llm_err = format_err_pct(perf.get("llm_error_rate_pct"))
                    vae_err = format_err_pct(perf.get("vae_error_rate_pct"))
                    fb_str = format_failure_breakdown(perf.get("failure_breakdown"), env_name=env_k)
                lines.append(
                    f"| `{r['task_mode']}` | {r['seed']} | {r['agent_name']} | {rew} | {non_rew} | **{score:.4f}** | {llm_err} | {vae_err} | {fb_str} | {r['timestamp']} |"
                )
        elif env_k == "PickEnv":
            lines.append("| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Completed At |")
            lines.append("|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|")
            for r in env_recs:
                perf = r["performance"]
                picked = perf.get("picked", 0)
                broken = perf.get("brocken", 0)
                score = perf.get("running_average_score", 0.0)
                llm_err = format_err_pct(perf.get("llm_error_rate_pct"))
                vae_err = "-"
                fb_str = format_failure_breakdown(perf.get("failure_breakdown"), env_name="PickEnv")
                lines.append(
                    f"| `{r['task_mode']}` | {r['seed']} | {r['agent_name']} | {picked} | {broken} | **{score:.4f}** | {llm_err} | {vae_err} | {fb_str} | {r['timestamp']} |"
                )
        elif env_k == "MiniGridRelational":
            lines.append("| Target | Seed | Agent | Successful Pick | Successful Drop | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Completed At |")
            lines.append("|--------|------|-------|-----------------|-----------------|-------------------|----------------|----------------|---------------------------|--------------|")
            for r in env_recs:
                perf = r["performance"]
                pick = perf.get("successful_pick", 0)
                drop = perf.get("successful_drop", 0)
                score = perf.get("running_average_score", 0.0)
                llm_err = format_err_pct(perf.get("llm_error_rate_pct"))
                vae_err = format_err_pct(perf.get("vae_error_rate_pct"))
                fb_str = format_failure_breakdown(perf.get("failure_breakdown"), env_name="MiniGridRelational")
                lines.append(
                    f"| `{r['task_mode']}` | {r['seed']} | {r['agent_name']} | {pick} | {drop} | **{score:.4f}** | {llm_err} | {vae_err} | {fb_str} | {r['timestamp']} |"
                )
        else:
            # Generic fallback table with all keys
            keys = set()
            for r in env_recs:
                keys.update(r["performance"].keys())
            header_keys = sorted(list(keys))
            headers = ["Target", "Seed", "Agent"] + header_keys + ["Completed At"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in env_recs:
                row = [f"`{r['task_mode']}`", str(r["seed"]), str(r["agent_name"])]
                for k in header_keys:
                    row.append(format_metric_value(r["performance"].get(k, "")))
                row.append(r["timestamp"])
                lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    content = "\n".join(lines) + "\n"
    with open(md_file_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update Markdown Performance Table")
    parser.add_argument("--md_path", type=str, default="Results/agent_performance.md")
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--task_mode", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--agent_name", type=str, required=True)
    parser.add_argument("--performance_json", type=str, required=True)
    parser.add_argument("--engine_name", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--num_episodes", type=int, default=10)

    args = parser.parse_args()
    perf = json.loads(args.performance_json)
    append_or_update_metric(
        md_file_path=args.md_path,
        env_name=args.env_name,
        task_mode=args.task_mode,
        seed=args.seed,
        agent_name=args.agent_name,
        performance=perf,
        engine_name=args.engine_name,
        num_episodes=args.num_episodes
    )
