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

def append_or_update_metric(
    md_file_path: str,
    env_name: str,
    task_mode: str,
    seed: int,
    agent_name: str,
    performance: dict,
    engine_name: str = "google/gemma-4-12B-it",
    num_episodes: int = 10
):
    """
    Appends or updates a record in the markdown report and its companion cache.
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
    new_record = {
        "env_name": env_name,
        "task_mode": task_mode,
        "seed": seed,
        "agent_name": agent_name,
        "performance": performance,
        "engine_name": engine_name,
        "num_episodes": num_episodes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    found = False
    for i, r in enumerate(records):
        if (
            r.get("env_name") == env_name
            and r.get("task_mode") == task_mode
            and r.get("seed") == seed
            and r.get("agent_name") == agent_name
        ):
            records[i] = new_record
            found = True
            break

    if not found:
        records.append(new_record)

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
    lines.append("| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) |")
    lines.append("|-------------|--------|-----------------|-------------------|------------------------|")

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
                f"| {env_k} | `{target}` | {seeds_str} | **{mean_score:.4f} ± {std_score:.4f}** | {mean_primary:.2f} ({primary_name}) |"
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
            lines.append("| Target | Seed | Agent | Rewarding Objects Picked | Non-Rewarding Objects Picked | Running Avg Score | Completed At |")
            lines.append("|--------|------|-------|--------------------------|------------------------------|-------------------|--------------|")
            for r in env_recs:
                perf = r["performance"]
                rew = format_metric_value(perf.get("rewarding_objects", {}))
                non_rew = format_metric_value(perf.get("non_rewarding_objects", {}))
                score = perf.get("running_average_score", 0.0)
                lines.append(
                    f"| `{r['task_mode']}` | {r['seed']} | {r['agent_name']} | {rew} | {non_rew} | **{score:.4f}** | {r['timestamp']} |"
                )
        elif env_k == "PickEnv":
            lines.append("| Target | Seed | Agent | Picked | Broken | Running Avg Score | Completed At |")
            lines.append("|--------|------|-------|--------|--------|-------------------|--------------|")
            for r in env_recs:
                perf = r["performance"]
                picked = perf.get("picked", 0)
                broken = perf.get("brocken", 0)
                score = perf.get("running_average_score", 0.0)
                lines.append(
                    f"| `{r['task_mode']}` | {r['seed']} | {r['agent_name']} | {picked} | {broken} | **{score:.4f}** | {r['timestamp']} |"
                )
        elif env_k == "MiniGridRelational":
            lines.append("| Target | Seed | Agent | Successful Pick | Successful Drop | Running Avg Score | Completed At |")
            lines.append("|--------|------|-------|-----------------|-----------------|-------------------|--------------|")
            for r in env_recs:
                perf = r["performance"]
                pick = perf.get("successful_pick", 0)
                drop = perf.get("successful_drop", 0)
                score = perf.get("running_average_score", 0.0)
                lines.append(
                    f"| `{r['task_mode']}` | {r['seed']} | {r['agent_name']} | {pick} | {drop} | **{score:.4f}** | {r['timestamp']} |"
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
