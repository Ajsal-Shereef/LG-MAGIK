# Agent Performance Evaluation Summary

**Last Updated:** 2026-11-22 12:04:04
- **Reasoning Engine:** `MAGIK (M2-VAE)`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `5`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| MiniWorld | `target3` | 42, 123, 456, 789, 1024 | **0.9062 ± 0.9939** | 0.60 (Reward Objects Picked) | 0.0% | 0.7% | 0 / 6 / 41 | - | - |

---

## 2. MiniWorld

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target3` | 42 | MAGIK_PPO | ball: 0 | box: 0 | **1.9775** | 0.0% | 0.2% | 0 / 1 / 9 | - | - | 2026-11-22 11:59:22 |
| `target3` | 123 | MAGIK_PPO | ball: 1 | box: 3 | **-0.2397** | 0.0% | 0.0% | 0 / 0 / 9 | - | - | 2026-11-22 12:00:23 |
| `target3` | 456 | MAGIK_PPO | ball: 0 | box: 0 | **1.1070** | 0.0% | 0.0% | 0 / 0 / 10 | - | - | 2026-11-22 12:01:36 |
| `target3` | 789 | MAGIK_PPO | ball: 1 | box: 1 | **1.9455** | 0.0% | 0.3% | 0 / 1 / 8 | - | - | 2026-11-22 12:02:50 |
| `target3` | 1024 | MAGIK_PPO | ball: 1 | box: 3 | **-0.2592** | 0.0% | 2.9% | 0 / 4 / 5 | - | - | 2026-11-22 12:04:04 |

