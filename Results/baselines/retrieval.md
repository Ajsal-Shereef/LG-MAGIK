# Agent Performance Evaluation Summary

**Last Updated:** 2026-09-30 04:25:38
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `7`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| SimplePickup | `target1` | 42 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 10 | - | - |
| SimplePickup | `target2` | 42 | **0.0000 ± 0.0000** | 0.00 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 10 | - | - |
| SimplePickup | `target3` | 42 | **-0.9000 ± 0.0000** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 10 | - | - |
| MiniWorld | `target1` | 42 | **0.8802 ± 0.0000** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 10 | - | - |
| MiniWorld | `target2` | 42 | **-0.2188 ± 0.0000** | 1.00 (Reward Objects Picked) | 0.0% | - | 0 / 0 / 9 | - | - |
| MiniWorld | `target3` | 42 | **-8.2121 ± 0.0000** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 10 | - | - |
| PickEnv | `target` | 42 | **9.9923 ± 0.0000** | 2.00 (Picked) | 0.0% | - | 0 / - / 5 | - | - |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Retrieval_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 04:20:17 |
| `target2` | 42 | Retrieval_DQN | purple box: 0 | green ball: 0 | **0.0000** | 0.0% | - | 0 / 0 / 10 | - | - | 2026-09-30 04:20:44 |
| `target3` | 42 | Retrieval_DQN | green ball: 0 | red ball: 9 | **-0.9000** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 04:21:07 |

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | Retrieval_PPO | duckie: 0 | ball: 0 | **0.8802** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 04:22:37 |
| `target2` | 42 | Retrieval_PPO | duckie: 1 | ball: 3 | **-0.2188** | 0.0% | - | 0 / 0 / 9 | - | - | 2026-09-30 04:23:40 |
| `target3` | 42 | Retrieval_PPO | ball: 0 | box: 9 | **-8.2121** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-09-30 04:24:49 |

## 4. PickEnv

| Target | Seed | Agent | Picked | Broken | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|--------|--------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target` | 42 | Retrieval_SAC | 2 | 5 | **9.9923** | 0.0% | - | 0 / - / 5 | - | - | 2026-09-30 04:25:38 |

