# Agent Performance Evaluation Summary

**Last Updated:** 2026-09-30 04:19:49
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `35`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 42, 123, 456, 789, 1024 | **0.5800 ± 0.1166** | 5.80 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 21 | - | - |
| SimplePickup | `target2` | 42, 123, 456, 789, 1024 | **0.4200 ± 0.1166** | 4.20 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 29 | - | - |
| SimplePickup | `target3` | 42, 123, 456, 789, 1024 | **0.2400 ± 0.1020** | 2.40 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 38 | - | - |
| MiniWorld | `target1` | 42, 123, 456, 789, 1024 | **11.8035 ± 1.0433** | 9.60 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 4 | - | - |
| MiniWorld | `target2` | 42, 123, 456, 789, 1024 | **11.8618 ± 1.6092** | 9.40 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 4 | - | - |
| MiniWorld | `target3` | 42, 123, 456, 789, 1024 | **7.5175 ± 2.3031** | 6.40 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 19 | - | - |
| PickEnv | `target` | 42, 123, 456, 789, 1024 | **47.1948 ± 6.7810** | 9.80 (Picked) | 0.0% | - | 0 / - / 0 | - | - |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Finetuned50k_DQN | purple box: 7 | green ball: 0 | **0.7000** | 0.0% | 0.0% | 0 / 0 / 3 | - | - | 2026-09-30 04:08:46 |
| `target1` | 123 | Finetuned50k_DQN | purple box: 4 | green ball: 0 | **0.4000** | 0.0% | 0.0% | 0 / 0 / 6 | - | - | 2026-09-30 04:09:01 |
| `target1` | 456 | Finetuned50k_DQN | purple box: 7 | green ball: 0 | **0.7000** | 0.0% | 0.0% | 0 / 0 / 3 | - | - | 2026-09-30 04:09:15 |
| `target1` | 789 | Finetuned50k_DQN | purple box: 5 | green ball: 0 | **0.5000** | 0.0% | 0.0% | 0 / 0 / 5 | - | - | 2026-09-30 04:09:31 |
| `target1` | 1024 | Finetuned50k_DQN | purple box: 6 | green ball: 0 | **0.6000** | 0.0% | 0.0% | 0 / 0 / 4 | - | - | 2026-09-30 04:09:45 |
| `target2` | 42 | Finetuned50k_DQN | purple box: 4 | green ball: 0 | **0.4000** | 0.0% | - | 0 / 0 / 6 | - | - | 2026-09-30 04:10:00 |
| `target2` | 123 | Finetuned50k_DQN | purple box: 2 | green ball: 0 | **0.2000** | 0.0% | - | 0 / 0 / 8 | - | - | 2026-09-30 04:10:16 |
| `target2` | 456 | Finetuned50k_DQN | purple box: 5 | green ball: 0 | **0.5000** | 0.0% | - | 0 / 0 / 5 | - | - | 2026-09-30 04:10:30 |
| `target2` | 789 | Finetuned50k_DQN | purple box: 5 | green ball: 0 | **0.5000** | 0.0% | - | 0 / 0 / 5 | - | - | 2026-09-30 04:10:45 |
| `target2` | 1024 | Finetuned50k_DQN | purple box: 5 | green ball: 0 | **0.5000** | 0.0% | - | 0 / 0 / 5 | - | - | 2026-09-30 04:11:01 |
| `target3` | 42 | Finetuned50k_DQN | green ball: 1 | red ball: 0 | **0.1000** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-09-30 04:11:16 |
| `target3` | 123 | Finetuned50k_DQN | green ball: 4 | red ball: 0 | **0.4000** | 0.0% | 0.0% | 0 / 0 / 6 | - | - | 2026-09-30 04:11:30 |
| `target3` | 456 | Finetuned50k_DQN | green ball: 2 | red ball: 0 | **0.2000** | 0.0% | 0.0% | 0 / 0 / 8 | - | - | 2026-09-30 04:11:45 |
| `target3` | 789 | Finetuned50k_DQN | green ball: 3 | red ball: 0 | **0.3000** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-09-30 04:12:01 |
| `target3` | 1024 | Finetuned50k_DQN | green ball: 2 | red ball: 0 | **0.2000** | 0.0% | 0.0% | 0 / 0 / 8 | - | - | 2026-09-30 04:12:16 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Finetuned50k_PPO | duckie: 10 | ball: 1 | **11.3861** | 0.0% | 0.0% | 0 / 0 / 1 | - | - | 2026-09-30 04:12:39 |
| `target1` | 123 | Finetuned50k_PPO | duckie: 10 | ball: 0 | **13.2131** | 0.0% | 0.0% | 0 / 0 / 0 | - | - | 2026-09-30 04:13:05 |
| `target1` | 456 | Finetuned50k_PPO | duckie: 9 | ball: 1 | **10.1838** | 0.0% | 0.0% | 0 / 0 / 2 | - | - | 2026-09-30 04:13:30 |
| `target1` | 789 | Finetuned50k_PPO | duckie: 9 | ball: 0 | **11.6373** | 0.0% | 0.0% | 0 / 0 / 1 | - | - | 2026-09-30 04:13:56 |
| `target1` | 1024 | Finetuned50k_PPO | duckie: 10 | ball: 0 | **12.5974** | 0.0% | 0.0% | 0 / 0 / 0 | - | - | 2026-09-30 04:14:19 |
| `target2` | 42 | Finetuned50k_PPO | duckie: 10 | ball: 0 | **12.6615** | 0.0% | - | 0 / 0 / 0 | - | - | 2026-09-30 04:14:44 |
| `target2` | 123 | Finetuned50k_PPO | duckie: 10 | ball: 0 | **13.2729** | 0.0% | - | 0 / 0 / 0 | - | - | 2026-09-30 04:15:09 |
| `target2` | 456 | Finetuned50k_PPO | duckie: 8 | ball: 1 | **8.7425** | 0.0% | - | 0 / 0 / 3 | - | - | 2026-09-30 04:15:37 |
| `target2` | 789 | Finetuned50k_PPO | duckie: 9 | ball: 0 | **12.0211** | 0.0% | - | 0 / 0 / 1 | - | - | 2026-09-30 04:16:03 |
| `target2` | 1024 | Finetuned50k_PPO | duckie: 10 | ball: 0 | **12.6112** | 0.0% | - | 0 / 0 / 0 | - | - | 2026-09-30 04:16:27 |
| `target3` | 42 | Finetuned50k_PPO | ball: 6 | box: 1 | **6.6432** | 0.0% | 0.0% | 0 / 0 / 4 | - | - | 2026-09-30 04:16:55 |
| `target3` | 123 | Finetuned50k_PPO | ball: 7 | box: 0 | **8.6042** | 0.0% | 0.0% | 0 / 0 / 3 | - | - | 2026-09-30 04:17:21 |
| `target3` | 456 | Finetuned50k_PPO | ball: 4 | box: 1 | **3.7924** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-09-30 04:17:51 |
| `target3` | 789 | Finetuned50k_PPO | ball: 9 | box: 0 | **10.7790** | 0.0% | 0.0% | 0 / 0 / 1 | - | - | 2026-09-30 04:18:17 |
| `target3` | 1024 | Finetuned50k_PPO | ball: 6 | box: 0 | **7.7689** | 0.0% | 0.0% | 0 / 0 / 4 | - | - | 2026-09-30 04:18:46 |

## 4. PickEnv

| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target` | 42 | Finetuned50k_SAC | 10 | 0 | **49.3153** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:18:59 |
| `target` | 123 | Finetuned50k_SAC | 10 | 0 | **43.8024** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:19:12 |
| `target` | 456 | Finetuned50k_SAC | 9 | 1 | **41.3892** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:19:24 |
| `target` | 789 | Finetuned50k_SAC | 10 | 0 | **41.9259** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:19:37 |
| `target` | 1024 | Finetuned50k_SAC | 10 | 0 | **59.5413** | 0.0% | - | 0 / - / 0 | - | - | 2026-09-30 04:19:49 |

