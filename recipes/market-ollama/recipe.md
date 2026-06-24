# market-ollama — is combining cheaper than scaling, on YOUR ollama models?

Measure the **verification-routing market** (`gama market`) and the discriminating **hard**
suite on local ollama models — the `p > w/s` verdict from
[`soshiki-genron`](https://github.com/akihidem/soshiki-genron) §5, on your own hardware.

## What it does
`run.py` builds a cheap→expensive ladder of local ollama models, runs gama's `hard` suite
(deterministic checkers, **no LLM judge**) over each, then asks whether verification-routed
escalation Pareto-dominates the single strongest model — i.e. does the cheap tier's
solve-rate `p` clear the cost ratio `w/s`?

## Run
```bash
# ollama must be serving locally (http://localhost:11434). Pull the models first:
ollama pull gemma4:e2b && ollama pull qwen2.5:7b
python3 recipes/market-ollama/run.py              # 2-tier ladder, hard suite (all cases)
python3 recipes/market-ollama/run.py --limit 1    # 1 case/class (quick smoke)
```
Edit `LADDER` in `run.py` for your own models / costs. The cost weights are a price/compute
proxy (here ≈ param-count); set them to your real $/token or latency. On a CPU-only box the
first call to each model is a slow cold-load — `be(timeout=…)` bounds a stalled call.

## Measured (this box: WSL2 CPU, ollama, 2026-06-24)
ladder: `gemma4:e2b` (cost 1) → `qwen2.5:7b` (cost 3), `hard` suite (10 cases):

| class | gemma-e2b | qwen-7b |
|---|---|---|
| code_implementation | 1.00 | 1.00 |
| content | 1.00 | 0.75 |
| integration | 1.00 | 1.00 |
| qa | 1.00 | 0.50 |
| research | 1.00 | 0.50 |
| **OVERALL** | **1.00** | **0.75** |

```
verification-routing market vs flat-strong:
  market      : cost=10.0  pass_rate=1.0
  flat-strong : qwen-7b  cost=30.0  pass_rate=0.7
  Pareto-dominates flat-strong : True   (analytic p_weak=1.0 > p*=0.333)
```

**The twist — and the point.** Here the *cheap* tier (`gemma4:e2b`, ~2B) actually scored
**higher** than the nominally-"strong" 7B: capability is **not a monotonic ladder** (the
non-monotonic *profile* that [`soshiki-genron`](https://github.com/akihidem/soshiki-genron)
measured shows up on real local models too — the 7B lost points on strict `qa`/`research`
formatting). The market never assumes the ladder is right: it routes by *external
verification*, so it stops at `gemma` on every case, pays 1/3 the cost, and Pareto-dominates
flat-`qwen`. The verdict is honest about *your* models — not a story you told it. (Raw
per-case outputs in [`results.json`](results.json).)

## Reading it
- **market dominates** = the escalation market reached the strong model's pass-rate at
  *strictly lower cost* (you paid the cheap tier most of the time, the strong one only where
  the external check demanded it).
- **doesn't dominate is also a real result**: if the models are too close (small `w/s` gap)
  or the cheap one solves too little (`p < w/s`), a single model wins. That's the threshold,
  not a failure — see the 3 regimes in the main [README](../../README.md).

> SECURITY: the code/integration cases **execute model-generated Python** (opt-in, like a
> sandbox). Only run on models you trust.
