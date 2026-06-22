# ollama starter — light stack vs a bigger model (bring your own numbers)

A ready-to-run **starting point** for anyone with [ollama](https://ollama.com) — no
SSH/MLX. **Not yet verified**: swap in models you actually have and PR your bench table.

## How
1. `ollama pull` the models you want, and edit `config.json` to match `ollama list`.
2. Run: `python3 -m experiments.moa_vs_strong recipes/ollama-starter/config.json`
3. Paste your table below and open a PR.

## The combination
| task class | lane |
|---|---|
| `qa` (math) | **tool** — the model writes Python, we run it |
| `code_implementation` | a **coder** model |
| `research` | a small **heterogeneous ensemble** |

vs baseline: a single bigger ollama model.

## Result — fill me in! 🌱
| | gama-light (ollama) | baseline (ollama-big) |
|---|---|---|
| score | ? | ? |
| misses | ? | ? |

## Notes (be honest)
Report blind spots, single-run variance, and any measurement caveats. A tie is a tie —
don't claim a win. See [CONTRIBUTING](../../CONTRIBUTING.md).
