# Agent Performance Evaluation Summary

**Last Updated:** 2026-09-30 04:08:31
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `35`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 42, 123, 456, 789, 1024 | **0.3000 ± 0.0632** | 3.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 35 | - | - |
| SimplePickup | `target2` | 42, 123, 456, 789, 1024 | **0.1000 ± 0.0632** | 1.00 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 45 | - | - |
| SimplePickup | `target3` | 42, 123, 456, 789, 1024 | **0.1400 ± 0.1020** | 1.40 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 43 | - | - |
| MiniWorld | `target1` | 42, 123, 456, 789, 1024 | **9.0282 ± 2.9980** | 7.60 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 15 | - | - |
| MiniWorld | `target2` | 42, 123, 456, 789, 1024 | **10.2664 ± 2.9611** | 8.20 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 11 | - | - |
| MiniWorld | `target3` | 42, 123, 456, 789, 1024 | **1.1133 ± 1.7197** | 1.20 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 44 | - | - |
| PickEnv | `target` | 42, 123, 456, 789, 1024 | **45.5449 ± 6.8469** | 9.60 (Picked) | 0.0% | - | 0 / - / 1 | - | - |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Finetuned20k_DQN | purple box: 3 | green ball: 0 | **0.3000** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-09-30 03:55:11 |
| `target1` | 123 | Finetuned20k_DQN | purple box: 3 | green ball: 0 | **0.3000** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-09-30 03:55:26 |
| `target1` | 456 | Finetuned20k_DQN | purple box: 4 | green ball: 0 | **0.4000** | 0.0% | 0.0% | 0 / 0 / 6 | - | - | 2026-09-30 03:55:40 |
| `target1` | 789 | Finetuned20k_DQN | purple box: 2 | green ball: 0 | **0.2000** | 0.0% | 0.0% | 0 / 0 / 8 | - | - | 2026-09-30 03:55:56 |
| `target1` | 1024 | Finetuned20k_DQN | purple box: 3 | green ball: 0 | **0.3000** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-09-30 03:56:10 |
| `target2` | 42 | Finetuned20k_DQN | purple box: 1 | green ball: 0 | **0.1000** | 0.0% | - | 0 / 0 / 9 | - | - | 2026-09-30 03:56:26 |
| `target2` | 123 | Finetuned20k_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 03:56:41 |
| `target2` | 456 | Finetuned20k_DQN | purple box: 1 | green ball: 0 | **0.1000** | 0.0% | - | 0 / 0 / 9 | - | - | 2026-09-30 03:56:56 |
| `target2` | 789 | Finetuned20k_DQN | purple box: 2 | green ball: 0 | **0.2000** | 0.0% | - | 0 / 0 / 8 | - | - | 2026-09-30 03:57:11 |
| `target2` | 1024 | Finetuned20k_DQN | purple box: 1 | green ball: 0 | **0.1000** | 0.0% | - | 0 / 0 / 9 | - | - | 2026-09-30 03:57:26 |
| `target3` | 42 | Finetuned20k_DQN | green ball: 1 | red ball: 0 | **0.1000** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-09-30 03:57:41 |
| `target3` | 123 | Finetuned20k_DQN | green ball: 2 | red ball: 0 | **0.2000** | 0.0% | 0.0% | 0 / 0 / 8 | - | - | 2026-09-30 03:57:56 |
| `target3` | 456 | Finetuned20k_DQN | green ball: 1 | red ball: 0 | **0.1000** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-09-30 03:58:11 |
| `target3` | 789 | Finetuned20k_DQN | green ball: 3 | red ball: 0 | **0.3000** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-09-30 03:58:25 |
| `target3` | 1024 | Finetuned20k_DQN | green ball: 0 | red ball: 0 | **0.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:58:41 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Finetuned20k_PPO | duckie: 8 | ball: 0 | **10.3365** | 0.0% | 0.0% | 0 / 0 / 2 | - | - | 2026-09-30 03:59:09 |
| `target1` | 123 | Finetuned20k_PPO | duckie: 4 | ball: 1 | **4.3531** | 0.0% | 0.0% | 0 / 0 / 6 | - | - | 2026-09-30 03:59:42 |
| `target1` | 456 | Finetuned20k_PPO | duckie: 7 | ball: 2 | **6.7285** | 0.0% | 0.0% | 0 / 0 / 5 | - | - | 2026-09-30 04:00:15 |
| `target1` | 789 | Finetuned20k_PPO | duckie: 10 | ball: 1 | **11.9786** | 0.0% | 0.0% | 0 / 0 / 1 | - | - | 2026-09-30 04:00:56 |
| `target1` | 1024 | Finetuned20k_PPO | duckie: 9 | ball: 0 | **11.7443** | 0.0% | 0.0% | 0 / 0 / 1 | - | - | 2026-09-30 04:01:43 |
| `target2` | 42 | Finetuned20k_PPO | duckie: 10 | ball: 1 | **11.7487** | 0.0% | - | 0 / 0 / 1 | - | - | 2026-09-30 04:02:26 |
| `target2` | 123 | Finetuned20k_PPO | duckie: 6 | ball: 0 | **8.7686** | 0.0% | - | 0 / 0 / 4 | - | - | 2026-09-30 04:03:10 |
| `target2` | 456 | Finetuned20k_PPO | duckie: 5 | ball: 1 | **5.1599** | 0.0% | - | 0 / 0 / 6 | - | - | 2026-09-30 04:03:43 |
| `target2` | 789 | Finetuned20k_PPO | duckie: 10 | ball: 0 | **13.1490** | 0.0% | - | 0 / 0 / 0 | - | - | 2026-09-30 04:04:10 |
| `target2` | 1024 | Finetuned20k_PPO | duckie: 10 | ball: 0 | **12.5059** | 0.0% | - | 0 / 0 / 0 | - | - | 2026-09-30 04:04:35 |
| `target3` | 42 | Finetuned20k_PPO | ball: 1 | box: 1 | **0.6564** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-09-30 04:05:09 |
| `target3` | 123 | Finetuned20k_PPO | ball: 3 | box: 0 | **3.3609** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-09-30 04:05:41 |
| `target3` | 456 | Finetuned20k_PPO | ball: 0 | box: 2 | **-1.8478** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 04:06:15 |
| `target3` | 789 | Finetuned20k_PPO | ball: 1 | box: 0 | **1.8907** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-09-30 04:06:49 |
| `target3` | 1024 | Finetuned20k_PPO | ball: 1 | box: 0 | **1.5062** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-09-30 04:07:24 |

## 4. PickEnv

| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target` | 42 | Finetuned20k_SAC | 9 | 1 | **46.3635** | 0.0% | - | 0 / - / 1 | - | - | 2026-09-30 04:07:39 |
| `target` | 123 | Finetuned20k_SAC | 10 | 0 | **43.1797** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:07:52 |
| `target` | 456 | Finetuned20k_SAC | 9 | 1 | **39.3580** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:08:05 |
| `target` | 789 | Finetuned20k_SAC | 10 | 0 | **40.4688** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:08:18 |
| `target` | 1024 | Finetuned20k_SAC | 10 | 0 | **58.3548** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:08:31 |

