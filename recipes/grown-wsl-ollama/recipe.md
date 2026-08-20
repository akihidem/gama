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

## What this recipe ships

Both lanes the loop found, each measured on the class it serves:

| lane | its class, sealed, 3 reps | removal test | promotion record |
|---|---|---|---|
| `qa → tool(llama3.2:3b)` | **0.222 → 1.000** | refused (−0.18 to −0.21 confirm) | 7 promoted / 1 refused / 1 never proposed |
| `research → mesh(3b→coder7b)` | **0.313 → 0.583** | refused (−0.142 confirm vs a 0.0217 band) | 3 promoted / 6 refused |

An earlier version of this file shipped only the `qa` lane, on the strength of that last
column. That was the wrong column to decide on, and the reason is worth carrying away:

> **A promotion record measures whether a change could beat whatever champion existed that
> day. It does not measure what the change contributes once it is in.**

The two questions come apart badly here. `research → mesh` won 3 of 9 attempts as an addition
— genuinely marginal. But seed a run from the champion that already contains it, and taking it
out costs 0.142 confirm (3.3 cases) against a noise band of 0.0217; measured on the four sealed
`research` cases directly, it goes 0.313 → 0.583, improving three of the four. It is easier to
see the contribution of a lane by removing it than by proposing it, because the removal is
measured against a stable high-scoring champion while the proposals were measured against
whatever noisy thing the loop happened to be holding.

`qa → tool` earns its place the same way and more strongly, including a case no model in the
pool solves unaided (asked for 3h45m in seconds they answer 15300; asked for a program, the
same 3B writes `3*3600+45*60`).

**What it costs.** The champion is roughly 3.7x slower per case than the bare model on the
sealed split (2.02s vs 0.55s), and on the classes it touches the multiplier is about 10x
(research: 6.09s vs 0.60s). Two of five classes now spend extra model calls. That is the
price, and the loop has no opinion about it: it optimises score, and only the shrink gate ever
pushes back on structure.

## The loop was asked to defend this champion, and did

Two runs seeded from it (`--start-from`) produced **zero promotions in eight generations**, and
both removals were refused. Every addition it tried was either not better on confirm or better
by less than the champion's own drift that generation. On this box, with these five suites and
this mutation set, the configuration below is a local optimum rather than a way-station.

## Does this generalise? Measured, and no

The obvious question about a grown recipe is whether the shapes it found are properties of the
task suite or of the one model it grew from. Twelve runs all seeded `llama3.2:3b`, so run M
swapped the pool for a different family entirely — `gemma4:e2b` + `qwen2.5:7b`, no llama, no
coder — and grew from bare.

Nothing was promoted in five generations, and the reason is not that the mutations failed:

| | sealed (same 20 cases) | latency / case |
|---|---|---|
| structured `llama3.2:3b` (this recipe) | 0.865 | 2.02s |
| **bare `gemma4:e2b`, no structure at all** | **0.950** | **10.85s** |

The seed already scored 0.959 on search and 0.935 on confirm. With one case of headroom left on
the sealed split, no mutation could clear a promotion bar of one confirm case — **the suite is
saturated for that model**, so this run cannot say whether the structures transfer.

It does say something else, and it cuts against the grain: on this box and this pool, **paying
for the bigger model beat structuring the smaller one** — 1.7 sealed cases better for 5.4x the
latency. "Structure, not scale" buys you a *cheap* model that works, not a better ceiling. If
you can afford 10.85s per case, the simplest thing on this hardware is to stop reading and run
gemma4:e2b.

So this recipe is scoped, deliberately: **it is a prescription for a weak seed model on this
case pool.** Do not port the two lanes to a stronger model on the strength of these numbers;
run `gama grow` there and let it tell you, which takes an afternoon and is the entire point of
the tool.

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
