# ollama (official models) — structured routing wins (0.83 vs 0.5), verified

Fully reproducible with official, pullable models. Shows **why you route**: the tool helps
math but *breaks* code, so a single tool-everywhere — or a naive ensemble — doesn't win.
Structured routing does.

## Hardware / runtime
- 32 GB box (CPU), ollama. Models (all `ollama pull`-able):
  `qwen2.5-coder:7b`, `qwen2.5:7b`, `llama3.2:3b`.

## Result — hard suite (6 tasks), measured
| | qwen2.5:7b (single) | coder7b + tool | ensemble (3 small) | **gama-combined** |
|---|---|---|---|---|
| **AVG** | 0.50 | 0.50 | 0.50 | **0.83** |

Per task type:
- **math** (m1, m2): single & ensemble FAIL — small models can't do the arithmetic. **The
  tool fixes it** (the model writes Python, we run it). 0 → 1.
- **code** (c1, c2): single & ensemble pass. **Tool-everywhere BREAKS code** — PAL turns
  "write a function" into "print something", so the function check fails. 1 → 0.
- **reasoning**: r2 (look-and-say) all pass; r1 (a pigeonhole puzzle) all FAIL — a gap these
  7Bs share.

**The lesson:** the tool must be **routed** (it helps math, hurts code). `gama-combined`
routes `qa → tool`, `code → coder`, `research → ensemble` and wins (**0.83**) over every
single approach (0.50). The remaining miss (r1) is a reasoning gap these small models share
— that's where a bigger model (or a better reasoner) is still needed.

## Reproduce
```bash
ollama pull qwen2.5-coder:7b qwen2.5:7b llama3.2:3b
python3 experiments/moa_vs_strong.py recipes/ollama-official/config.json
```

## Notes (honest)
Single run; small models are streaky. No big baseline on this box (CPU / 32 GB) — this
shows the **routing** win on accessible 7Bs, not a 122B tie (see
[`mac-studio-mlx`](../mac-studio-mlx/recipe.md) for that).
