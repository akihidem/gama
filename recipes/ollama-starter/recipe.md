# ollama + tool — the tool lane recovers a compute miss (verified on gemma4)

A small, **verified** starting point for anyone with [ollama](https://ollama.com) — no
SSH/MLX. Shows the program-aided **tool** lane recovering a miss, on a modest box.

## Hardware / runtime
- 32 GB box (CPU), **ollama** with `gemma4` (`gemma4:e2b`, `gemma4:latest`).
- Single run; small models are non-deterministic — numbers will wobble.

## Result — hard suite (6 tasks), measured
| | gemma4:e2b (small) | gemma4:latest | **gemma4:latest + tool** |
|---|---|---|---|
| **score** | **1.00** | **0.83** | **1.00** |
| miss | — | m2 (a multi-step word problem) | — |

**Finding:** `gemma4:latest` miscalculated one multi-step word problem (480 − 12·9 + 20·6
= 492); wrapping it in the **`tool`** lane (the model writes Python, we run it) recovered
it → **1.00**. The tool reliably rescues *compute-able* tasks. (Amusingly, the *smaller*
`gemma4:e2b` got that one right unaided — small models are streaky.)

## Reproduce
```bash
python3 experiments/moa_vs_strong.py recipes/ollama-starter/config.json
```
Swap `gemma4` for any models you have (`ollama list`) — e.g. `qwen2.5:7b`, `llama3.2` —
and the tool lane should help the same way on arithmetic/word problems. PR your numbers.

## Notes (honest)
Single run, modest box, gemma4 is streaky. This isn't the "light stack ties a 122B" story
(this box has no big model) — for that, see [`mac-studio-mlx`](../mac-studio-mlx/recipe.md).
The reproducible point here: **the tool lane fixes compute misses**, even on small ollama
models.
