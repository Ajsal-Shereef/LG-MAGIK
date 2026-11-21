# Agent Performance Evaluation Summary

**Last Updated:** 2026-10-07 19:04:44
- **Reasoning Engine:** `google/gemma-4-12B-it`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `13`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 789, 1000 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 70,578 | 4.09s |
| SimplePickup | `target2` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 128,132 | 4.07s |
| SimplePickup | `target3` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 128,137 | 4.13s |
| MiniWorld | `target1` | 789 | **0.0750 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 504,648 | 4.88s |
| MiniWorld | `target2` | 789 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 511,195 | 5.28s |
| MiniWorld | `target3` | 789 | **0.0750 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 499,617 | 4.54s |
| PickEnv | `target` | 789 | **33.6683 ± 0.0000** | 10.00 (Picked) | - | - | 0 / - / 0 | 62,067 | 10.58s |
| MiniGridRelational | `target1` | 789 | **0.0000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 174,702 | 8.21s |
| MiniGridRelational | `target2` | 789 | **0.0000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 180,890 | 9.33s |
| MiniGridRelational | `target3` | 789 | **0.1000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 206,172 | 12.27s |
| MiniGridRelational | `target4` | 789 | **0.2000 ± 0.0000** | 0.00 (Successful Drop) | - | - | 0 / 0 / 0 | 199,040 | 12.17s |
| MiniGridRelational | `target5` | 789 | **1.2000 ± 0.0000** | 1.00 (Successful Drop) | - | - | 0 / 0 / 0 | 184,748 | 9.33s |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_DQN | purple box: 0 | green ball: 0 | **0.0000** | - | - | - | 128,469 | 4.20s | 2026-10-06 17:43:27 |
| `target1` | 1000 | LLM_Actor_DQN | purple box: 0 | green ball: 0 | **0.0000** | - | - | - | 12,688 | 3.98s | 2026-10-06 17:25:49 |
| `target2` | 789 | LLM_Actor_DQN | purple box: 0 | green ball: 0 | **0.0000** | - | - | - | 128,132 | 4.07s | 2026-10-06 18:00:54 |
| `target3` | 789 | LLM_Actor_DQN | green ball: 0 | red ball: 0 | **0.0000** | - | - | - | 128,137 | 4.13s | 2026-10-06 18:18:57 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_DQN | duckie: 0 | ball: 0 | **0.0750** | - | - | - | 504,648 | 4.88s | 2026-10-06 19:40:52 |
| `target2` | 789 | LLM_Actor_DQN | duckie: 0 | ball: 0 | **0.0000** | - | - | - | 511,195 | 5.28s | 2026-10-06 21:09:43 |
| `target3` | 789 | LLM_Actor_DQN | ball: 0 | box: 0 | **0.0750** | - | - | - | 499,617 | 4.54s | 2026-10-06 22:26:07 |

## 4. PickEnv

| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target` | 789 | LLM_Actor_PPO | 10 | 0 | **33.6683** | - | - | - | 62,067 | 10.58s | 2026-10-07 19:04:44 |

## 5. MiniGridRelational

| Target | Seed | Agent | Successful Pick | Successful Drop | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|-----------------|-----------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_DQN | 0 | 0 | **0.0000** | - | - | - | 174,702 | 8.21s | 2026-10-06 23:08:01 |
| `target2` | 789 | LLM_Actor_DQN | 0 | 0 | **0.0000** | - | - | - | 180,890 | 9.33s | 2026-10-06 23:55:11 |
| `target3` | 789 | LLM_Actor_DQN | 1 | 0 | **0.1000** | - | - | - | 206,172 | 12.27s | 2026-10-07 00:57:05 |
| `target4` | 789 | LLM_Actor_DQN | 2 | 0 | **0.2000** | - | - | - | 199,040 | 12.17s | 2026-10-07 01:58:37 |
| `target5` | 789 | LLM_Actor_DQN | 2 | 1 | **1.2000** | - | - | - | 184,748 | 9.33s | 2026-10-07 02:43:53 |

