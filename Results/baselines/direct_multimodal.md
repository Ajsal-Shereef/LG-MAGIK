# Agent Performance Evaluation Summary

**Last Updated:** 2026-11-24 02:33:53
- **Reasoning Engine:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `5`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| MiniWorld | `target3` | 42, 123, 456, 789, 1024 | **-6.0309 ± 1.5153** | 0.00 (Reward Objects Picked) | 0.0% | 0.0% | 0 / 0 / 50 | - | - |

---

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target3` | 42 | Direct_PPO_multimodal | ball: 0 | box: 5 | **-4.5812** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:32:06 |
| `target3` | 123 | Direct_PPO_multimodal | ball: 0 | box: 4 | **-4.0854** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:32:33 |
| `target3` | 456 | Direct_PPO_multimodal | ball: 0 | box: 8 | **-7.9369** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:32:59 |
| `target3` | 789 | Direct_PPO_multimodal | ball: 0 | box: 6 | **-6.1227** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:33:26 |
| `target3` | 1024 | Direct_PPO_multimodal | ball: 0 | box: 8 | **-7.4285** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-24 02:33:53 |

