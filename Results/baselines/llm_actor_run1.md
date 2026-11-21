# Agent Performance Evaluation Summary

**Last Updated:** 2026-10-07 21:18:52
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `4`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 789 | **0.7000 ± 0.0000** | 7.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 251,950 | 79.89s |
| SimplePickup | `target2` | 789 | **0.5000 ± 0.0000** | 5.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 309,052 | 48.93s |
| SimplePickup | `target3` | 789 | **0.6000 ± 0.0000** | 6.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 295,270 | 78.08s |
| MiniWorld | `target1` | 789 | **0.1490 ± 0.0000** | 0.00 (Reward Objects Picked) | - | - | 0 / 0 / 0 | 53,023 | 50.00s |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_DQN | purple box: 7 | green ball: 0 | **0.7000** | - | - | - | 251,950 | 79.89s | 2026-10-07 14:25:24 |
| `target2` | 789 | LLM_Actor_DQN | purple box: 5 | green ball: 0 | **0.5000** | - | - | - | 309,052 | 48.93s | 2026-10-07 17:04:18 |
| `target3` | 789 | LLM_Actor_PPO | green ball: 6 | red ball: 0 | **0.6000** | - | - | - | 295,270 | 78.08s | 2026-10-07 20:36:50 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 789 | LLM_Actor_PPO | duckie: 0 | ball: 0 | **0.1490** | - | - | - | 53,023 | 50.00s | 2026-10-07 21:18:52 |

