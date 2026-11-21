# Agent Performance Evaluation Summary

**Last Updated:** 2026-10-08 09:05:44
- **Reasoning Engine:** `google/gemma-4-12B-it`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `12`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 71,165 | 5.46s |
| SimplePickup | `target2` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 71,355 | 5.35s |
| SimplePickup | `target3` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 73,620 | 6.12s |
| MiniWorld | `target1` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 168,496 | 7.02s |
| MiniWorld | `target2` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 154,492 | 5.56s |
| MiniWorld | `target3` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 160,358 | 6.24s |
| PickEnv | `target` | 789 | **-9.0315 ± 0.0000** | 0.00 (Picked) | - | - | 0 / - / 0 | 130,540 | 9.37s |
| MiniGridRelational | `target1` | 789 | **0.0000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 123,134 | 12.60s |
| MiniGridRelational | `target2` | 789 | **0.0000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 131,312 | 14.38s |
| MiniGridRelational | `target3` | 789 | **0.0000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 152,062 | 18.16s |
| MiniGridRelational | `target4` | 789 | **0.0000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 146,201 | 17.77s |
| MiniGridRelational | `target5` | 789 | **1.2000 ± 0.0000** | 1.00 (Successful Drop) | - | - | 0 / 0 / 0 | 144,875 | 15.60s |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_PPO | purple box: 0 | green ball: 0 | **0.0000** | - | - | - | 71,165 | 5.46s | 2026-10-07 22:59:57 |
| `target2` | 789 | LLM_Actor_PPO | purple box: 0 | green ball: 0 | **0.0000** | - | - | - | 71,355 | 5.35s | 2026-10-07 23:23:00 |
| `target3` | 789 | LLM_Actor_PPO | green ball: 0 | red ball: 0 | **0.0000** | - | - | - | 73,620 | 6.12s | 2026-10-07 23:49:09 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_PPO | duckie: 0 | ball: 0 | **0.0000** | - | - | - | 168,496 | 7.02s | 2026-10-08 00:48:23 |
| `target2` | 789 | LLM_Actor_PPO | duckie: 0 | ball: 0 | **0.0000** | - | - | - | 154,492 | 5.56s | 2026-10-08 01:35:28 |
| `target3` | 789 | LLM_Actor_PPO | ball: 0 | box: 0 | **0.0000** | - | - | - | 160,358 | 6.24s | 2026-10-08 02:28:17 |

## 4. PickEnv

| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target` | 789 | LLM_Actor_PPO | 0 | 8 | **-9.0315** | - | - | - | 130,540 | 9.37s | 2026-10-07 22:36:29 |

## 5. MiniGridRelational

| Target | Seed | Agent | Successful Pick | Successful Drop | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|-----------------|-----------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_PPO | 0 | 0 | **0.0000** | - | - | - | 123,134 | 12.60s | 2026-10-08 03:32:14 |
| `target2` | 789 | LLM_Actor_PPO | 0 | 0 | **0.0000** | - | - | - | 131,312 | 14.38s | 2026-10-08 04:45:08 |
| `target3` | 789 | LLM_Actor_PPO | 0 | 0 | **0.0000** | - | - | - | 152,062 | 18.16s | 2026-10-08 06:16:55 |
| `target4` | 789 | LLM_Actor_PPO | 0 | 0 | **0.0000** | - | - | - | 146,201 | 17.77s | 2026-10-08 07:46:44 |
| `target5` | 789 | LLM_Actor_PPO | 2 | 1 | **1.2000** | - | - | - | 144,875 | 15.60s | 2026-10-08 09:05:44 |

