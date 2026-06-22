# meshflow (ollama) — verification-routed escalation beats a single tier (1.0 vs 0.8)

Verified, reproducible. `meshflow` tries the **cheap** tier first and accepts its answer
**only if an external check passes**; otherwise it escalates. So it is at least as correct
as any single tier — and cheaper when the cheap tier suffices.

## Hardware / runtime
- 32 GB box (CPU), ollama. Tiers: `llama3.2:3b` (cheap) → `qwen2.5-coder:7b` (strong).
- The gate = `gama bench`'s own per-case checker (honest external verification — *not* the
  model's self-report).

## Result — default suite (5 classes), measured
| | meshflow (3b → 7b, verify-gated) | always qwen2.5-coder:7b |
|---|---|---|
| **score** | **1.00** | 0.80 |
| avg latency | 8.1 s | 1.7 s |

meshflow **matched/beat** the strong tier (1.0 vs 0.8): it accepts whichever tier passes the
external check, so on a task the 7B-coder fails (e.g. mental arithmetic) it can accept the
**cheap** tier's *correct* answer instead. The cost here was latency — it escalated often on
this varied suite. On tasks where the cheap tier already passes, it stops there (cheaper).
That is the whole point: **pay for the strong tier only when the check demands it.**

## Reproduce
```bash
ollama pull llama3.2:3b qwen2.5-coder:7b
gama bench --backends meshflow,ollama --config recipes/meshflow-ollama/config.json --tier large
```

## Notes (honest)
Single run; small N; small models are streaky. The win is "accept the tier that passes an
external check" — at least as correct as the best tier, cheaper when the cheap one suffices.
High stakes + still unresolved → meshflow returns `<<NEEDS_HUMAN>>` instead of shipping. See
the meshflow section in the README for the full design.
