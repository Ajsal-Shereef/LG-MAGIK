# Agent Performance Evaluation Summary

**Last Updated:** 2026-11-24 02:36:06
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `5`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| MiniWorld | `target3` | 42, 123, 456, 789, 1024 | **0.1510 ± 0.4646** | 0.20 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 49 | - | - |

---

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target3` | 42 | Finetuned20k_PPO_multimodal | ball: 0 | box: 0 | **0.1927** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:34:19 |
| `target3` | 123 | Finetuned20k_PPO_multimodal | ball: 0 | box: 0 | **-0.1642** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:34:46 |
| `target3` | 456 | Finetuned20k_PPO_multimodal | ball: 0 | box: 1 | **-0.2868** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:35:13 |
| `target3` | 789 | Finetuned20k_PPO_multimodal | ball: 1 | box: 0 | **1.0233** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-11-24 02:35:40 |
| `target3` | 1024 | Finetuned20k_PPO_multimodal | ball: 0 | box: 0 | **-0.0099** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:36:06 |

