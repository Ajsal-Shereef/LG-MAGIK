# Agent Performance Evaluation Summary

**Last Updated:** 2026-10-10 15:51:05
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `12`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 789 | **0.5000 ± 0.0000** | 5.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 271,189 | 86.42s |
| SimplePickup | `target2` | 789 | **0.4000 ± 0.0000** | 4.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 365,479 | 84.16s |
| SimplePickup | `target3` | 789 | **0.5000 ± 0.0000** | 5.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 412,572 | 66.75s |
| MiniWorld | `target1` | 789 | **0.1499 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 461,628 | 37.66s |
| MiniWorld | `target2` | 789 | **0.1648 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 599,691 | 56.45s |
| MiniWorld | `target3` | 789 | **0.1647 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 567,684 | 64.30s |
| PickEnv | `target` | 789 | **21.2175 ± 0.0000** | 6.00 (Picked) | - | - | 0 / - / 0 | 445,884 | 70.95s |
| MiniGridRelational | `target1` | 789 | **1.4000 ± 0.0000** | 1.00 (Successful Drop) | - | - | 0 / 0 / 0 | 473,565 | 57.06s |
| MiniGridRelational | `target2` | 789 | **1.4000 ± 0.0000** | 1.00 (Successful Drop) | - | - | 0 / 0 / 0 | 588,410 | 90.10s |
| MiniGridRelational | `target3` | 789 | **2.3000 ± 0.0000** | 2.00 (Successful Drop) | - | - | 0 / 0 / 0 | 756,002 | 121.74s |
| MiniGridRelational | `target4` | 789 | **4.5000 ± 0.0000** | 4.00 (Successful Drop) | - | - | 0 / 0 / 0 | 615,723 | 69.68s |
| MiniGridRelational | `target5` | 789 | **1.4000 ± 0.0000** | 1.00 (Successful Drop) | - | - | 0 / 0 / 0 | 456,128 | 40.00s |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_PPO | purple box: 5 | green ball: 0 | **0.5000** | - | - | - | 271,189 | 86.42s | 2026-10-08 01:59:11 |
| `target2` | 789 | LLM_Actor_PPO | purple box: 4 | green ball: 0 | **0.4000** | - | - | - | 365,479 | 84.16s | 2026-10-08 06:58:57 |
| `target3` | 789 | LLM_Actor_PPO | green ball: 5 | red ball: 0 | **0.5000** | - | - | - | 412,572 | 66.75s | 2026-10-08 10:38:30 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_PPO | duckie: 0 | ball: 0 | **0.1499** | - | - | - | 461,628 | 37.66s | 2026-10-08 15:52:54 |
| `target2` | 789 | LLM_Actor_PPO | duckie: 0 | ball: 0 | **0.1648** | - | - | - | 599,691 | 56.45s | 2026-10-08 23:44:01 |
| `target3` | 789 | LLM_Actor_PPO | ball: 0 | box: 0 | **0.1647** | - | - | - | 567,684 | 64.30s | 2026-10-09 08:40:25 |

## 4. PickEnv

| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target` | 789 | LLM_Actor_PPO | 6 | 4 | **21.2175** | - | - | - | 445,884 | 70.95s | 2026-10-09 12:29:39 |

## 5. MiniGridRelational

| Target | Seed | Agent | Successful Pick | Successful Drop | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|-----------------|-----------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_PPO | 4 | 1 | **1.4000** | - | - | - | 473,565 | 57.06s | 2026-10-09 16:46:15 |
| `target2` | 789 | LLM_Actor_PPO | 4 | 1 | **1.4000** | - | - | - | 588,410 | 90.10s | 2026-10-09 23:34:37 |
| `target3` | 789 | LLM_Actor_PPO | 3 | 2 | **2.3000** | - | - | - | 756,002 | 121.74s | 2026-10-10 08:20:22 |
| `target4` | 789 | LLM_Actor_PPO | 5 | 4 | **4.5000** | - | - | - | 615,723 | 69.68s | 2026-10-10 12:42:57 |
| `target5` | 789 | LLM_Actor_PPO | 4 | 1 | **1.4000** | - | - | - | 456,128 | 40.00s | 2026-10-10 15:51:05 |

