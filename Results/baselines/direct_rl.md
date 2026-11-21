# Agent Performance Evaluation Summary

**Last Updated:** 2026-09-30 03:54:55
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `35`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 42, 123, 456, 789, 1024 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 50 | - | - |
| SimplePickup | `target2` | 42, 123, 456, 789, 1024 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 50 | - | - |
| SimplePickup | `target3` | 42, 123, 456, 789, 1024 | **-0.9000 ± 0.0894** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 50 | - | - |
| MiniWorld | `target1` | 42, 123, 456, 789, 1024 | **0.4748 ± 1.4359** | 0.60 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 47 | - | - |
| MiniWorld | `target2` | 42, 123, 456, 789, 1024 | **3.8607 ± 3.1012** | 2.80 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 37 | - | - |
| MiniWorld | `target3` | 42, 123, 456, 789, 1024 | **-8.7328 ± 0.6286** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 50 | - | - |
| PickEnv | `target` | 42, 123, 456, 789, 1024 | **18.3867 ± 6.1817** | 3.20 (Picked) | 0.0% | - | 0 / - / 13 | - | - |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:41:57 |
| `target1` | 123 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:42:12 |
| `target1` | 456 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:42:27 |
| `target1` | 789 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:42:42 |
| `target1` | 1024 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:42:57 |
| `target2` | 42 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 03:43:12 |
| `target2` | 123 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 03:43:27 |
| `target2` | 456 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 03:43:42 |
| `target2` | 789 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 03:43:58 |
| `target2` | 1024 | Direct_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 03:44:14 |
| `target3` | 42 | Direct_DQN | green ball: 0 | red ball: 9 | **-0.9000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:44:28 |
| `target3` | 123 | Direct_DQN | green ball: 0 | red ball: 8 | **-0.8000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:44:43 |
| `target3` | 456 | Direct_DQN | green ball: 0 | red ball: 10 | **-1.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:44:58 |
| `target3` | 789 | Direct_DQN | green ball: 0 | red ball: 10 | **-1.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:45:11 |
| `target3` | 1024 | Direct_DQN | green ball: 0 | red ball: 8 | **-0.8000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:45:24 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Direct_PPO | duckie: 0 | ball: 1 | **-0.8269** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:45:58 |
| `target1` | 123 | Direct_PPO | duckie: 0 | ball: 1 | **-1.3527** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:46:31 |
| `target1` | 456 | Direct_PPO | duckie: 2 | ball: 1 | **0.9673** | 0.0% | 0.0% | 0 / 0 / 8 | - | - | 2026-09-30 03:47:04 |
| `target1` | 789 | Direct_PPO | duckie: 1 | ball: 0 | **2.6753** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-09-30 03:47:37 |
| `target1` | 1024 | Direct_PPO | duckie: 0 | ball: 0 | **0.9109** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:48:11 |
| `target2` | 42 | Direct_PPO | duckie: 4 | ball: 1 | **4.3255** | 0.0% | - | 0 / 0 / 7 | - | - | 2026-09-30 03:48:40 |
| `target2` | 123 | Direct_PPO | duckie: 0 | ball: 0 | **0.2196** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 03:49:14 |
| `target2` | 456 | Direct_PPO | duckie: 1 | ball: 0 | **1.0741** | 0.0% | - | 0 / 0 / 9 | - | - | 2026-09-30 03:49:48 |
| `target2` | 789 | Direct_PPO | duckie: 6 | ball: 0 | **8.9716** | 0.0% | - | 0 / 0 / 4 | - | - | 2026-09-30 03:50:16 |
| `target2` | 1024 | Direct_PPO | duckie: 3 | ball: 0 | **4.7130** | 0.0% | - | 0 / 0 / 7 | - | - | 2026-09-30 03:50:47 |
| `target3` | 42 | Direct_PPO | ball: 0 | box: 9 | **-8.3551** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:51:21 |
| `target3` | 123 | Direct_PPO | ball: 0 | box: 9 | **-9.1606** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:51:54 |
| `target3` | 456 | Direct_PPO | ball: 0 | box: 9 | **-8.4523** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:52:29 |
| `target3` | 789 | Direct_PPO | ball: 0 | box: 10 | **-9.7269** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:53:04 |
| `target3` | 1024 | Direct_PPO | ball: 0 | box: 9 | **-7.9692** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 03:53:44 |

## 4. PickEnv

| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target` | 42 | Direct_SAC | 3 | 4 | **19.6906** | 0.0% | - | 0 / - / 3 | - | - | 2026-09-30 03:53:59 |
| `target` | 123 | Direct_SAC | 5 | 5 | **27.1863** | 0.0% | - | 0 / - / 1 | - | - | 2026-09-30 03:54:14 |
| `target` | 456 | Direct_SAC | 5 | 3 | **21.6759** | 0.0% | - | 0 / - / 2 | - | - | 2026-09-30 03:54:28 |
| `target` | 789 | Direct_SAC | 3 | 2 | **14.0811** | 0.0% | - | 0 / - / 4 | - | - | 2026-09-30 03:54:41 |
| `target` | 1024 | Direct_SAC | 0 | 4 | **9.2996** | 0.0% | - | 0 / - / 3 | - | - | 2026-09-30 03:54:55 |

