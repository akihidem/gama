# WSL2 CPU-only — `gama grow` found exactly one structural win (and threw the rest out)

The first recipe in this library that was **not written by a person**. `gama grow` searched
the config space on a CPU-only box, twice, and both times promoted the same single change:
send the `qa` class through the **`tool`** (program-aided) lane on `llama3.2:3b`, leave every
other class on the bare model. Everything else it tried — ensembling, verification-gated
escalation, swapping in the 7B coder — was measured and rejected.

## Hardware / runtime
- WSL2 on Windows, 24 GB RAM, **CPU only** (no GPU offload), ollama over `localhost:11434`.
- Lanes offered to the loop: `llama3.2:3b`, `qwen2.5:7b`, `qwen2.5-coder:7b`.
- Cases: `wide,hard,brutal` (56), split **28 search / 15 confirm / 13 sealed** — the exact
  case ids are recorded in `config.json`, so the claim "the sealed split decided nothing" is
  checkable rather than promised.

## The combination
| task class | lane |
|---|---|
| `qa` | **`tool(llama3.2:3b)`** — the model writes Python, we execute it |
| everything else | `llama3.2:3b` |

vs baseline: the same `llama3.2:3b` with **no structure at all** — that is the seed the loop
started from, so the comparison is "did structure pay", not "is this model good".

## Result — two independent runs, sealed split (never used for any decision)

| | seed (no structure) | grown champion | gain |
|---|---|---|---|
| run A (`--repeats 1`) | 0.577 | **0.846** | +0.269 |
| run B (`--repeats 2`) | 0.603 | **0.789** | +0.186 |

13 sealed cases, so one case is 0.077: the two runs agree on direction and on order of
magnitude (**2–3 sealed cases**), and differ by about one case. Read it as "2–3 cases", not
as three decimal places. 3 of the 13 sealed cases are `qa`, which is the class the promotion
changed — the size of the gain is accounted for, not just observed.

## Why the tool lane, and why nothing else

`qa` here is exact arithmetic (2^20, gcd, modular powers). A 3B model does that badly in its
head and well in a `print(...)`. That is gama's own thesis; what is new is that **nobody told
the loop** — it proposed the mutation, measured it, and had to clear a held-out split to keep it.

The rejections are the more interesting half:

| candidate | run A | run B |
|---|---|---|
| `meshflow:research(3b→coder7b)` | lost on `search` (though it scored *higher* on confirm) | won `search`, then **flat** on confirm |
| `ensemble:qa(3b+coder7b)` | rejected | rejected |
| `tool:integration(3b)` | rejected | rejected |
| `route:content→coder7b` | rejected | rejected |

`meshflow:research` failing for **opposite reasons on the two runs** is the useful signal: a
candidate that loses a different gate each time is noise, not an effect. Either run on its own
reads as "so close" in a different direction.

## Reproduce
```bash
ollama pull llama3.2:3b
# champion vs the exact model it grew from, on a suite (a spot-check, NOT the split numbers):
gama bench --backends system,ollama --config recipes/grown-wsl-ollama/config.json --suite wide

# or grow it again from scratch (~2h on CPU with repeats=2):
gama grow --models llama3.2:3b,qwen2.5:7b,qwen2.5-coder:7b --repeats 2 --generations 3 --width 4
```

## Notes (honest)
- **CPU only.** Latency in any ledger from this box is dominated by model load/swap between
  lanes; do not read it as throughput.
- Both promotions were decided when the margin floor was a hand-set constant (0.05). The floor
  is now derived as **one confirm case** (1/15 = 0.067). The winning margin was 0.133 = two
  cases, so it clears either floor — re-checked against the ledger, not assumed.
- One structural win out of six designs, on a 3B model, is the honest yield. This recipe is
  evidence that the loop **refuses** far more than it accepts; if you want more than a tool
  lane out of it, the lever is a bigger case pool, not more generations.
- The `qa` win comes from cases with exact numeric answers. A `qa` workload that is not
  computational will not see this gain — the lane is only as good as the class it was measured on.
