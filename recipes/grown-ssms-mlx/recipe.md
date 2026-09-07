# recipe-ss (grown by `gama grow`)

Hardware: Mac Studio M4 Max 128GB (ssms), MLX: Mistral-Small-24B-Instruct-2501-4bit + Qwen2.5-7B-Instruct-4bit, 158 cases incl. crux+edge, ratio 1:2:2

The ssh host is written as `user@mac-studio` here and in `config.json`; the run used a local
alias. What the lines below record is the part that carries the evidence: the model each port
said it was serving, as reported by the server on every call. Point the host at your own box.

Measured against (as reported by the server on every call):
- `user@mac-studio:8080/mlx-community/Mistral-Small-24B-Instruct-2501-4bit` → `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`
- `user@mac-studio:8082/mlx-community/Qwen2.5-7B-Instruct-4bit` → `mlx-community/Qwen2.5-7B-Instruct-4bit`

**Held-out verdict: IMPROVED** (+5.25 cases in content, qa on the sealed split, which resolves 1)

the held-out split agrees the champion is better than the seed.

| | seed (no structure) | grown champion |
|---|---|---|
| sealed score (n=62 cases, never used for a decision) | **0.6288** | **0.7134** |
| confirm score (the split that decided promotions) | — | 0.8081 |
| confirm score the sealed claim was made from (means over the run: 3 seed / 1 champion measurements) | 0.6915 | 0.8081 |
| search score (selection — biased upward, do not quote) | — | 0.8452 |

- promotions: 2 over 5 generations (20 designs measured)
- the champion's diagnosis saw replies cut at the token limit (in 5 of 5 generation(s); most in one generation: integration: 2, qa: 2, research: 2); what became of the prescriptions:
  - `tokens:integration(m24)x3072`: listed in 4 generation(s), confirm-measured 2 time(s), promoted 0
  - `tokens:qa(m24)x3072`: listed in 2 generation(s), confirm-measured 0 time(s), promoted 0
  - `tokens:research(m24)x3072`: listed in 5 generation(s), confirm-measured 0 time(s), promoted 0
- per-call trace: 2980 calls; cut at the token limit (finish=length): 82 (integration: 42, qa: 18, research: 22); each reply's length, tail and stop reason are in `grow-ss.trace.jsonl`
- the bar was set by RESOLUTION in 5 of 5 judged generation(s): a change has to be worth one whole confirm case, so the lever is more cases IN THE CLASS being changed (not a bigger pool, and not more `--repeats`)
- most room left: `research` 7 confirm cases (cut 2) — above the one-case floor, so a mutation there can be promoted at all
- every promotion required a held-out `confirm` win larger than the champion's own re-measurement drift; no LLM judged anything.
- grown with: {"suites": ["wide", "graded", "steep", "qadeep", "researchdeep", "crux", "edge"], "ratio": [1, 2, 2], "tier": "large", "repeats": 2, "width": 4, "generations": 5, "patience": 5, "min_margin": 0.0154, "min_margin_source": "auto(one confirm case)", "ensemble_strategy": "synthesize", "max_paired_p": null, "code": {"version": "0.1.0", "commit": "9baf076", "dirty": false, "source": "4cdd0fce8c8ed3f3"}}
- spot-check the champion (this is NOT a reproduction of the numbers above, which come from the splits recorded in config.json): `gama bench --backends system --config config.json --suite hard`

## What grew

- · gen0 `tokens:integration(m24)x3072` — search 0.6516→0.6516, confirm 0.6915→0.6915 (δ=0.0154, noise 0 cases) → confirm-not-better
- ✅ gen1 `tool:qa(m24)` — search 0.6516→0.8129, confirm 0.6915→0.7876 (δ=0.0154, noise 0 cases) → promote
  - gated on +6.25 cases; re-measured next generation +6.25, mean while it stayed champion +6.25 (cases)
- · gen2 `ensemble:code_implementation(m24+q7)` — search 0.8129→0.8387, confirm 0.7876→0.7876 (δ=0.0154, noise 0 cases) → confirm-not-better
- ✅ gen3 `ensemble:content(m24+q7)` — search 0.8129→0.8452, confirm 0.7876→0.8081 (δ=0.0154, noise 0 cases) → promote
  - gated on +1.33 cases; re-measured next generation +1.33, mean while it stayed champion +1.33 (cases)
- · gen4 `tokens:integration(m24)x3072` — search 0.8452→0.8452, confirm 0.8081→0.8081 (δ=0.0154, noise 0 cases) → confirm-not-better
