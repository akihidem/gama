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

## What this recipe ships, and why only half of what the loop found

Nine runs proposed two structural changes. Their records are not comparable:

| candidate | promoted | refused | shipped here? |
|---|---|---|---|
| `tool:qa(llama3.2:3b)` | **7** | 1 (a 5-case confirm split where one case = 0.2) + 1 run that never proposed it, due to a search bug | **yes** |
| `meshflow:research(3b→coder7b)` | 3 | 6 | no — documented below |

`qa → tool` also has evidence no aggregate can give: measured per case with 3 repeats, the
three sealed `qa` cases go **0.222 → 1.000**, including `wide-qa-seconds`, which no model in
the pool solves unaided (they answer 15300 for 3h45m; asked for a program, the same 3B writes
`3*3600+45*60`). That is a mechanism, not a score.

`research → meshflow` is marginal, and this file has been wrong about it **twice, in opposite
directions**. First it was called noise, because across three runs it lost a *different* gate
each time. Then a run on the 86-case pool promoted it at +0.197 and this file called the noise
reading a mistake. Then another run on **the same 86-case pool** refused it (confirm went
down, −0.026). Final record: 3 wins, 6 losses, winning once on the smallest pool and losing
once on the largest — no clean relationship to split size at all. Each retraction came from
reading a pattern into one or two runs. The honest statement needs nine:

> On this box, escalating `research` to the 7B coder under external verification helps
> sometimes. If your box reproduces it, add the lane; the loop will tell you.

## The refusals are most of what a loop does

Across those runs the gate refused far more than it accepted, and the refusals came from every
part of it:

| reason | example |
|---|---|
| never earned the challenge | `meshflow:research` scored below the champion on `search` twice |
| no confirm gain | `ensemble:*` in six consecutive runs |
| below the margin | `meshflow:content` +0.054 against a bar of 0.071 — a real improvement, smaller than the champion's own drift that generation |
| the split could not certify it | `tool:qa` +0.075 on a 5-case confirm split where one case = 0.2 |

Two of those refusals were later shown to be right for the wrong reason, and one (`tool:qa` on
the coarse split) refused something independently known to be real. That is the cost of
refusing to certify below the resolution you have.

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
