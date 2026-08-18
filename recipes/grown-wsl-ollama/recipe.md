# WSL2 CPU-only — `gama grow` found exactly one structural win (and threw the rest out)

The first recipe in this library that was **not written by a person**. `gama grow` searched the
config space on a CPU-only box **three times** and promoted the same single change every time:
send the `qa` class through the **`tool`** (program-aided) lane on `llama3.2:3b`, leave every
other class on the bare model. Everything else it tried — ensembling, verification-gated
escalation, swapping in the 7B coder — was measured and rejected.

## Hardware / runtime
- WSL2 on Windows, 24 GB RAM, **CPU only** (no GPU offload), ollama over `localhost:11434`.
- Lanes offered to the loop: `llama3.2:3b`, `qwen2.5:7b`, `qwen2.5-coder:7b`.
- Cases: `wide,hard,brutal` (56), split **28 search / 15 confirm / 13 sealed** — the exact case
  ids are in `config.json`, so "the sealed split decided nothing" is checkable, not promised.

## The combination
| task class | lane |
|---|---|
| `qa` | **`tool(llama3.2:3b)`** — the model writes Python, we execute it |
| everything else | `llama3.2:3b` |

vs baseline: the same `llama3.2:3b` with **no structure at all** — the seed the loop started
from, so the comparison is "did structure pay", not "is this model good".

## Result — measure the class that changed

The champion reroutes **only `qa`**. Every other class runs the identical backend, so any
difference there is measurement noise by construction. Sealed split, per case, 3 reps each:

| sealed case | class | seed | champion |
|---|---|---|---|
| `hard-math-modexp` | qa | 0 / 0 / 0 | **1 / 1 / 1** |
| `wide-qa-seconds` | qa | 0 / 0 / 0 | **1 / 1 / 1** |
| `wide-qa-gcd` | qa | 1 / 0 / 1 | **1 / 1 / 1** |
| the other 10 cases | code / content / integration / research | *unchanged lane* | *unchanged lane* (±1 case of noise, in both directions) |

**`qa` on the sealed split: 0.222 → 1.000.** The tool lane fixes exactly the class it was
promoted for, and fixes it completely — including `wide-qa-seconds`, which **no model in the
pool solves unaided** (they answer 15300 for 3h45m; asked for a program, the same 3B writes
`3*3600+45*60` and gets 13500).

## Result — and why the whole-split totals scatter

The same three runs, read as a whole-split total, look much less stable:

| run | | sealed seed | sealed champion | total gain |
|---|---|---|---|---|
| A | `--repeats 1` | 0.577 | 0.846 | +0.269 |
| B | `--repeats 2` | 0.603 | 0.789 | +0.186 |
| C | `--repeats 2`, derived margin floor | 0.654 | 0.673 | **+0.019** |

Run C looks like the effect vanished. It did not: **3 of the 13 sealed cases are `qa`, so 10
cases of unrelated noise are averaged into every one of those totals.** The per-case table
above puts the real effect at 2.33 of 3 qa cases = **+0.179 on the 13-case total**, which is
what the three noisy totals scatter around.

That is the honest lesson to take from this recipe, more than the number: **a whole-split score
dilutes a class-restricted change with the noise of every class it did not touch.** If you
change one lane, measure that lane's cases.

What *did* reproduce tightly is the decision itself — 3 promotions out of 3 runs, with confirm
margins of +0.133, +0.133, +0.100 (the gate needs one confirm case = 0.067).

## Why the tool lane, and why nothing else

`qa` here is exact arithmetic (2^20, gcd, modular powers, unit conversion). A 3B model does that
badly in its head and well in a `print(...)`. That is gama's own thesis; what is new is that
**nobody told the loop** — it proposed the mutation, measured it, and had to clear a held-out
split to keep it.

The rejections are the more interesting half:

| candidate | run A | run B | run C |
|---|---|---|---|
| `meshflow:research(3b→coder7b)` | lost on `search` (but scored *higher* on confirm) | won `search`, then flat on confirm | won `search`, confirm +0.033 = **half a case**, below the margin |
| `ensemble:qa(3b+coder7b)` | rejected | rejected | rejected |
| `tool:integration(3b)` | rejected | rejected | rejected |
| `route:content→coder7b` | rejected | rejected | rejected |

`meshflow:research` failing for a **different reason on each run** is the useful signal: a
candidate that loses a different gate every time is noise, not an effect. Any single run of it
reads as "so close", in a different direction each time.

## Reproduce
```bash
ollama pull llama3.2:3b
# champion vs the exact model it grew from (a spot-check, NOT the split numbers above):
gama bench --backends system,ollama --config recipes/grown-wsl-ollama/config.json --suite wide

# or grow it again from scratch (~2h on CPU with repeats=2):
gama grow --models llama3.2:3b,qwen2.5:7b,qwen2.5-coder:7b --repeats 2 --generations 3 --width 4
```

## Notes (honest)
- **CPU only.** Latency in any ledger from this box is dominated by model load/swap between
  lanes; do not read it as throughput.
- Runs A and B were decided when the margin floor was a hand-set 0.05; run C used the derived
  floor (one confirm case = 0.067). All three promotions cleared both, so the change did not
  manufacture this result — checked against the ledgers, not assumed.
- The `qa` win comes from cases with exact numeric answers. A `qa` workload that is not
  computational will not see it — the lane is only as good as the class it was measured on.
- One structural win out of six designs, on a 3B model, is the honest yield. This recipe is
  evidence that the loop **refuses** far more than it accepts. If you want more out of it, the
  lever is a bigger case pool, not more generations.
