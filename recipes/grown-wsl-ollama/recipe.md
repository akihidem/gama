# WSL2 CPU-only — two structural wins a loop found, and four it refused

The first recipe in this library that was **not written by a person**. `gama grow` searched
the config space on a CPU-only box across six runs and ended up with two changes to a bare
`llama3.2:3b`, each of which had to clear a held-out split to get in:

| task class | lane | what it does |
|---|---|---|
| `qa` | **`tool(llama3.2:3b)`** | the model writes Python, we execute it (exact arithmetic) |
| `research` | **`mesh(llama3.2:3b → qwen2.5-coder:7b)`** | try the 3B, escalate to the 7B coder only when the external check fails |
| everything else | `llama3.2:3b` | untouched |

vs baseline: the same `llama3.2:3b` with **no structure at all** — the seed every run started
from, so the question being answered is "did structure pay", not "is this model good".

## Hardware / runtime
- WSL2 on Windows, 24 GB RAM, **CPU only** (no GPU offload), ollama over `localhost:11434`.
- Lanes offered: `llama3.2:3b`, `qwen2.5:7b`, `qwen2.5-coder:7b`.
- Run F: all five suites (86 cases) split **43 search / 23 confirm / 20 sealed**, `--repeats 2`.
  The case ids of each split are in `config.json`, so "the sealed split decided nothing" is
  checkable rather than promised.

## Result — run F, both wins in one run

| gen | challenger | search | confirm | verdict |
|---|---|---|---|---|
| 0 | `meshflow:research(3b→coder7b)` | 0.651 → 0.709 | 0.635 → **0.832** (+0.197) | **promoted** |
| 1 | `tool:qa(3b)` | 0.709 → 0.828 | 0.761 → **0.929** (+0.168) | **promoted** |
| 2 | `meshflow:content(3b→coder7b)` | 0.828 → 0.896 | 0.859 → 0.913 (+0.054) | refused: below the bar (0.071) |
| 3 | `meshflow:integration(3b→coder7b)` | 0.828 → 0.849 | 0.908 → 0.951 (+0.044) | refused: below the bar (0.049) |

**Sealed split (20 cases, never used for any decision): 0.638 → 0.854.** That is +4.3 cases,
and 8 of the 20 sealed cases belong to the two rerouted classes — so the size is accounted
for rather than merely observed.

Generations 2 and 3 are the interesting refusals. Both *improved* confirm, by +0.054 and
+0.044, and both were refused because the champion's own re-measurement drift that generation
was larger (0.071, 0.049). The bar is `max(one confirm case, that drift)`: in gen 0–1 the case
size bound it, in gen 2–3 the noise did. Structure kept looking slightly better and the loop
kept saying "not by more than you wobble".

## Six runs, and what changed between them

| run | pool | champion | sealed |
|---|---|---|---|
| A | 26 cases, repeats 1 | `qa→tool` | 0.577 → 0.846 |
| B | 56 cases, repeats 2 | `qa→tool` | 0.603 → 0.789 |
| C | 56 cases, derived floor | `qa→tool` | 0.654 → 0.673 |
| D | `graded` only (confirm = 5) | **none** — one case = 0.2, too coarse to certify anything | 0.85 → 0.85 |
| E | 86 cases, starved frontier | `research→meshflow` | 0.669 → 0.731 |
| F | 86 cases, rotating frontier | **both** | 0.638 → **0.854** |

Three things in that table are worth more than the numbers:

- **Run C looked like the effect vanished** (+0.019). It had not: the champion rerouted only
  `qa`, 3 of 13 sealed cases, so ten cases of unrelated noise were averaged into the total.
  Measured per case, `qa` went 0.222 → 1.000 and nothing else moved. **A whole-split score
  dilutes a class-restricted change with the noise of every class it did not touch.**
- **`meshflow:research` was called noise in an earlier version of this file.** Across runs A–C
  it lost a *different* gate each time, which is exactly what noise looks like. With 23 confirm
  cases instead of 8–15 it won by +0.197. The old reading was fair for the evidence then; it
  was still wrong.
- **Run E did not reject `tool:qa` — it never proposed it.** A narrow `--width` left an
  un-challenged candidate parked at the head of its queue, and a fixed kind order dropped the
  5th kind entirely once `simplify` became reachable (which happens the moment anything is
  promoted). Absence in a ledger reads like a verdict; it wasn't one. Fixed by rotating both
  offsets per generation, which is why run F could see both wins.

## Reproduce
```bash
ollama pull llama3.2:3b qwen2.5-coder:7b
# champion vs the exact model it grew from (a spot-check, NOT the split numbers above):
gama bench --backends system,ollama --config recipes/grown-wsl-ollama/config.json --suite wide

# or grow it again from scratch (~3h on CPU):
gama grow --models llama3.2:3b,qwen2.5:7b,qwen2.5-coder:7b \
          --suites default,hard,brutal,wide,graded --repeats 2 --generations 4 --width 4
```

## Notes (honest)
- **CPU only.** Latency in these ledgers is dominated by model load/swap between lanes; the
  champion is ~3x slower per case than the bare 3B (1.73s vs 0.61s on the sealed split) because
  two of its five classes now cost extra calls. Read that as the price of the two wins.
- `qa→tool` was promoted in every run that proposed it (5 for 5 attempts). `research→meshflow`
  is 2 for 2 on splits fine enough to resolve it, and lost three times on splits that were not.
- The `qa` win comes from cases with exact numeric answers; a non-computational `qa` workload
  will not see it. The `research` win comes from the 3B failing an external check that the 7B
  coder passes — with a different pair, measure before assuming.
- Six runs produced two accepted changes out of ~20 designs measured. The loop refuses far more
  than it accepts, and three of the four refusals in run F were improvements that were simply
  smaller than the measurement noise of the day.
