# Agent Performance Evaluation Summary

**Last Updated:** 2026-11-24 02:38:17
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `5`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| MiniWorld | `target3` | 42, 123, 456, 789, 1024 | **1.0841 ± 0.8584** | 1.40 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 44 | - | - |

---

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target3` | 42 | Finetuned50k_PPO_multimodal | ball: 1 | box: 0 | **1.1704** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-11-24 02:36:33 |
| `target3` | 123 | Finetuned50k_PPO_multimodal | ball: 1 | box: 0 | **1.3774** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-11-24 02:36:59 |
| `target3` | 456 | Finetuned50k_PPO_multimodal | ball: 3 | box: 1 | **2.4611** | 0.0% | 0.0% | 0 / 0 / 7 | - | - | 2026-11-24 02:37:24 |
| `target3` | 789 | Finetuned50k_PPO_multimodal | ball: 2 | box: 2 | **0.4815** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-11-24 02:37:50 |
| `target3` | 1024 | Finetuned50k_PPO_multimodal | ball: 0 | box: 0 | **-0.0697** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:38:17 |

