# Agent Performance Evaluation Summary

**Last Updated:** 2026-09-21 19:52:49
- **Reasoning Engine:** `google/gemma-4-12B-it`
- **Episodes per Scenario:** `10`
- **Total Completed Runs:** `5`

---

## Overall Performance Summary (Aggregated Across Seeds)

| Environment | Target | Seeds Completed | Mean Score ± Std | Primary Metric (Mean) |
|-------------|--------|-----------------|-------------------|------------------------|
| SimplePickup | `target1` | 42, 123, 456, 789 | **0.9500 ± 0.0866** | 9.50 (Reward Objects Picked) |

---

## 1. MiniGrid (SimplePickup)

| Target | Seed | Agent | Rewarding Objects Picked | Non-Rewarding Objects Picked | Running Avg Score | Completed At |
|--------|------|-------|--------------------------|------------------------------|-------------------|--------------|
| `target1` | 42 | DQN | purple box: 10 | green ball: 0 | **1.0000** | 2026-09-21 16:58:23 |
| `target1` | 123 | DQN | purple box: 10 | green ball: 0 | **1.0000** | 2026-09-21 17:40:40 |
| `target1` | 456 | DQN | purple box: 10 | green ball: 0 | **1.0000** | 2026-09-21 18:25:05 |
| `target1` | 789 | DQN | purple box: 8 | green ball: 0 | **0.8000** | 2026-09-21 19:52:49 |

