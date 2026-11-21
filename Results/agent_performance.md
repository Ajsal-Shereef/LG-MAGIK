# Agent Performance Evaluation Summary

**Last Updated:** 2026-10-12 16:01:11
- **Reasoning Engine:** `google/gemma-4-12B-it`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `15`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) | Step LLM Err % | Step VAE Err % | Episode Failures (LLM / VAE / Policy) | Tokens (Total) | Net Latency |
|-------------|--------|-----------------|-------------------|------------------------|----------------|----------------|----------------------------------------|----------------|-------------|
| MiniWorldNoisy | `target1` | 42, 123, 456, 789, 1024 | **8.5490 ± 3.6052** | 8.20 (Reward Objects Picked) | - | - | - | 454,240 | 12.31s |
| MiniWorldNoisy | `target2` | 42, 123, 456, 789, 1024 | **11.4086 ± 0.8214** | 9.20 (Reward Objects Picked) | - | - | - | 408,257 | 12.00s |
| MiniWorldNoisy | `target3` | 42, 123, 456, 789, 1024 | **7.1160 ± 3.4845** | 5.40 (Reward Objects Picked) | - | - | - | 383,734 | 12.68s |

---

## 3. MiniWorldNoisy

| Target | Seed | Agent | Rewarding Picked | Non-Rewarding Picked | Running Avg Score | Step LLM Err % | Step VAE Err % | Failures (LLM/VAE/Policy) | Total Tokens | Net Latency | Completed At |
|--------|------|-------|------------------|----------------------|-------------------|----------------|----------------|---------------------------|--------------|-------------|--------------|
| `target1` | 42 | PPO | duckie: 10 | ball: 1 | **11.6282** | - | - | - | 364,525 | 12.88s | 2026-10-11 22:24:25 |
| `target1` | 123 | PPO | duckie: 8 | ball: 3 | **8.5209** | - | - | - | 555,343 | 12.63s | 2026-10-12 00:02:10 |
| `target1` | 456 | PPO | duckie: 5 | ball: 5 | **2.0966** | - | - | - | 489,351 | 12.40s | 2026-10-12 01:35:16 |
| `target1` | 789 | PPO | duckie: 10 | ball: 1 | **12.2604** | - | - | - | 446,369 | 11.75s | 2026-10-12 02:46:33 |
| `target1` | 1024 | PPO | duckie: 8 | ball: 2 | **8.2389** | - | - | - | 415,614 | 11.90s | 2026-10-12 03:58:49 |
| `target2` | 42 | PPO | duckie: 9 | ball: 1 | **10.3457** | - | - | - | 373,290 | 11.88s | 2026-10-12 05:00:28 |
| `target2` | 123 | PPO | duckie: 9 | ball: 0 | **11.8605** | - | - | - | 444,604 | 12.04s | 2026-10-12 06:14:31 |
| `target2` | 456 | PPO | duckie: 9 | ball: 1 | **10.5114** | - | - | - | 475,975 | 12.09s | 2026-10-12 07:35:48 |
| `target2` | 789 | PPO | duckie: 9 | ball: 0 | **11.9428** | - | - | - | 409,374 | 12.06s | 2026-10-12 08:46:21 |
| `target2` | 1024 | PPO | duckie: 10 | ball: 0 | **12.3826** | - | - | - | 338,043 | 11.94s | 2026-10-12 09:41:56 |
| `target3` | 42 | PPO | ball: 9 | box: 1 | **10.1973** | - | - | - | 369,813 | 12.34s | 2026-10-12 10:45:38 |
| `target3` | 123 | PPO | ball: 8 | box: 0 | **10.4477** | - | - | - | 340,151 | 12.31s | 2026-10-12 11:46:34 |
| `target3` | 456 | PPO | ball: 4 | box: 0 | **5.3077** | - | - | - | 418,817 | 12.62s | 2026-10-12 13:10:57 |
| `target3` | 789 | PPO | ball: 0 | box: 1 | **1.1915** | - | - | - | 419,446 | 12.67s | 2026-10-12 14:44:06 |
| `target3` | 1024 | PPO | ball: 6 | box: 0 | **8.4360** | - | - | - | 370,443 | 13.47s | 2026-10-12 16:01:11 |

