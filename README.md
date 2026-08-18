```text
                                      ___
              .-"""-.   .-"""-.      (   )  ~
             /   o   \ /   o   \      )_(    puff
            |     >   V   <     |    /|\     (kiseru)
             \     '-...-'     /    / |
          _.'-------------------'-._/
         /         G A M A          \
        |          '--www--'         |
         \     croak ... croak      /
          '._                    _.'
             '-..____________..-'
```

> **Summon a toad.** Combine small local models — route, ensemble, tool — to match a
> big one. (*gama* = 蝦蟇, the toad you summon, à la Gamabunta in NARUTO.)

**English** | [日本語](README.ja.md)

# gama 🐸 — combine local LLMs

[![CI](https://github.com/akihidem/gama/actions/workflows/ci.yml/badge.svg)](https://github.com/akihidem/gama/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![deps: stdlib only](https://img.shields.io/badge/deps-stdlib%20only-brightgreen.svg)](pyproject.toml)

**Route each task to the right small local model, combine them into a mixture of agents,
give them tools — and benchmark which combination matches a big model. Stdlib-only.
Fully local.**

> **The finding that started this:** on a hard task suite, a *structured combination* of
> small local models (a 7B + 24B + 32B + a calculator tool, routed by task type) **tied a
> single 122B model (0.92 vs 0.92)** — entirely on one Mac, no cloud. Not by stacking
> copies (useless), not by naive ensembling (0.83), but by **routing each task class to
> the right light mechanism.** Structure, not scale.

`gama` is the toolkit for building and measuring those combinations on *your* hardware —
and a home to **grow a community library of recipes** (which small models + tools +
routing match a big model, on what hardware).

## Why
A small model that can't do arithmetic in its head can **write a `print(...)` and run
it**; a model weak at one kind of reasoning can be **outvoted by an ensemble**; a coder
model **beats a generalist on code**. Combine the right small specialists per task and you
can match a big model — locally, privately, cheaply.

## Install
```bash
pip install git+https://github.com/akihidem/gama        # or: pip install gama-llm
# to hack on it:
git clone https://github.com/akihidem/gama && cd gama && pip install -e .
```
No dependencies — pure Python ≥ 3.10.

## 30-second quickstart
gama talks to any OpenAI-compatible local server (**ollama**, **MLX `mlx_lm.server`**,
**LM Studio**, **vLLM**) and to subprocess CLIs.
```bash
# Free, deterministic smoke (no models needed):
gama bench --backends echo

# Benchmark your local models per task class, propose a routing table:
gama bench --backends ollama --tier large --propose routing.json
```

## The pieces
| Backend | What it does |
|---|---|
| `ollama`, `ssh-openai` | call a local model (HTTP, or an OpenAI server over SSH — no open port) |
| **`GamaBackend`** | **route** 1 task → 1 model by task class (a measured `routing_table`) |
| **`EnsembleBackend`** | **combine** N models on the same task (`synthesize` / `majority` / `first`) |
| **`ToolBackend`** | **program-aided**: the model writes Python, we run it (exact math, etc.) |
| **`MeshflowBackend`** | **escalate** cheap→strong gated by an *external check*, mesh at the edge, human-gate high stakes (the AI-native *organizational form*) |
| **`ABMCTSBackend`** | **search** — adaptively *go wider* (new candidate) or *go deeper* (refine one) per node via Thompson sampling, reward = the external check; the model per widen step is a bandit (Multi-LLM AB-MCTS) |

Compose them freely as JSON (`build_backend`): a `gama` router over `tool` / `ensemble` /
coder lanes is a *sovereign stack* you can benchmark against a single big model.
```bash
gama bench --backends gama,ssh-openai --config recipes/mac-studio-mlx/config.json --tier large
```

### Meshflow — structure *as an organization*
Routing and ensembling combine models *statically*. `MeshflowBackend` adds the missing
shape: **verification-routed escalation**. Try the cheapest tier; accept its answer **only
if an external `verify(artifact)→score` passes** (not the model's self-report); otherwise
climb to a stronger tier. When no single tier passes, **mesh** the drafts (their errors are
complementary); when it's *still* unresolved and the stakes are high, return `<<NEEDS_HUMAN>>`
rather than silently shipping — a thin human governance membrane. So you can pay the cheap
tier most of the time and reach the strong tier only when the check demands it:
```bash
gama run "<task>" --config examples/meshflow.example.json --task-type code_implementation
gama bench --backends meshflow,ssh-openai --config examples/meshflow.example.json --tier large
```
This is "structure, not scale" as an *organizational* runtime — ported from the
[`soshiki-genron`](https://github.com/akihidem/soshiki-genron) research repo
(`experiments/meshflow.py`, PAPER §6.5 "the org chart to adopt"), where the same form is
argued from first principles and shown to match a frontier model at lower cost.

### AB-MCTS — structure *as an adaptive search*
Routing, ensembling and meshflow all decide the *shape* of the combination up front. `ABMCTSBackend`
makes the shape adaptive at inference time: it grows a search tree where, at **every node**, it
Thompson-samples whether to spend the next model call **going wider** (generate a brand-new
candidate) or **going deeper** (refine an existing one), driven by the external `verify`→score as
the reward. Because each worker is one bandit *action*, a "go wider" step also learns online *which*
model to call — so a set of small models becomes **Multi-LLM AB-MCTS** for free. This is a
stdlib-only port of Sakana AI's *"Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive
Branching Tree Search"* ([arXiv:2503.04412](https://arxiv.org/abs/2503.04412), NeurIPS 2025; blog
[sakana.ai/ab-mcts](https://sakana.ai/ab-mcts/); OSS [SakanaAI/treequest](https://github.com/SakanaAI/treequest)) —
the closed-form **AB-MCTS-A** variant, whose Beta-conjugate Thompson sampling (`verify` scores are
already in `[0,1]`) needs only `random.betavariate`, no numpy/scipy/PyMC:
```bash
gama run "<task>" --config examples/abmcts.example.json --task-type code_implementation
gama bench --backends abmcts,ensemble,ssh-openai --config recipes/mac-studio-abmcts/config.json --tier large
```
"Go deeper" feeds the parent answer **and its score** back into the refine prompt — the one detail
that keeps depth from collapsing into width (i.e. into plain best-of-N). The honest test is not
"does search beat one model" but "does the *adaptive* wider/deeper search beat a *width-only*
best-of-N of the same models?" — so pair it with an `ensemble` control (see
`recipes/mac-studio-abmcts`) and read `last_trace` to confirm the winning nodes are actually
`refine`d at depth > 1. Same porting honesty as `trinity.py`: we implement AB-MCTS-A, not the
MCMC-based AB-MCTS-M (which would need PyMC and break stdlib-only).

### Measure *whether the structure pays* — `bench --suite hard`, `market`, `mesh`
Combining only helps under specific conditions; gama lets you **measure them on your own
models** instead of guessing. The default bench suite hits a ceiling (good models all score
1.0 — it can't tell them apart), so first switch to a discriminating one:
```bash
gama bench --backends ollama,ssh-openai --suite hard        # or: --suite brutal, --suite wide
```
`hard` and `brutal` are *harder*; **`wide`** is *broader* — 40 cases, 8 in each of the 5
classes, in the same difficulty band as `hard`. Depth ranks two backends; breadth is what
lets you **split** a suite into search / confirm / sealed without each case being worth a
fifth of the score (see `gama grow` below). Every expected answer in every suite is
re-derived from a reference solution in `tests/test_suite_integrity.py`, because a case with
a wrong answer quietly penalises the models that got it right.

Measured once on `wide` (2026-08-18, WSL2, CPU-only ollama — your box will differ, and
nothing in CI guards these numbers): `llama3.2:3b` **0.550**, `qwen2.5:7b` **0.725**,
`qwen2.5-coder:7b` **0.817** — monotone with capability and nowhere near the ceiling, which is
what a suite has to do before you split it. 19 of the 40 cases separate those three; 17 are
solved by all of them (they still earn their place — they are what catches a mutation that
*breaks* an easy class) and 4 by none of them.

**`--suite graded` is the one that scores in fractions.** Every other suite is effectively
pass/fail, so a score can only move in whole cases. Each of its 20 cases carries several
independently checkable requirements (three sub-answers, four format constraints, six CSV
cells) and returns the fraction satisfied — still deterministic, still no judge model, just a
finer grain. Two things need that grain: `gama grow`'s "one confirm case" floor can only
differ from a fixed constant when a real gain lands *between* half a case and a whole one, and
an inference-time search like `abmcts` steers on the verify score as a reward — with a binary
reward, "go deeper" has nothing to refine toward. Measured on the same box, **35% / 22% / 15%**
of measurements land strictly between 0 and 1 for the 3B / 7B / 7B-coder (observed values
0.33, 0.43, 0.5, 0.6, 0.67, 0.75, 0.8 — sevenths and fifths, not just halves). Honest caveat:
`graded` is *easier* than `wide` (means 0.73 / 0.90 / 0.94), so use it for gradient, not for
separating strong backends.
**`gama market` — when is escalation cheaper than scaling?** Verification-routed escalation
(meshflow) Pareto-dominates the single strong model **iff the cheap tier's solve-rate `p`
exceeds the cost ratio `w/s`** (`p > w/s`). `gama market` runs the bench over your tiers
(cheap→expensive) and prints the verdict — cost, pass-rate, and whether the market dominates:
```bash
gama market --backends gemma,haiku --suite hard --costs 1,10
```
**`gama mesh` — does ensembling actually help?** An ensemble beats its best single member
**only when members are decorrelated (`rho < 1`) and mutually complementary (not nested)** —
`gain = (1−rho)·(1−p)·(1−(1−p)^(n−1))`. `gama mesh` measures the failure correlation `rho`
and the union-vs-best gain from a bench, so you know *before* deploying whether combining
ignites or just burns tokens:
```bash
gama mesh --backends gemma,qwen,llama --suite hard
```
These are the *economic / statistical verdict layer* for the composites above, ported from
the [`soshiki-genron`](https://github.com/akihidem/soshiki-genron) research repo
(`model/market.py`, p>w/s; `model/mesh.py`, decorrelation). They turn "structure, not scale"
from a slogan into something you can **falsify on your own hardware**.

## The result
Hard 12-task suite, fully local on a Mac Studio (MLX). Measurement made fair (code
extraction + token budget) — read this as *competitive/tied*, not a clean win:

| | sovereign light stack (7B+24B+32B+tool, routed) | single 122B |
|---|---|---|
| score | **0.92** | **0.92** |
| | misses 1 (a day-of-week puzzle) | misses 1 (a roman-numeral coder task) |

Complementary blind spots, same score — all local. Reproduce:
`python3 -m experiments.moa_vs_strong <config.json>`.

## A canvas for these combinations — [yoriai 🪢](https://github.com/akihidem/yoriai)

gama's config is already a tree: `build_backend` composes `meshflow` / `ensemble` / `tool` /
`gama` recursively. **[yoriai](https://github.com/akihidem/yoriai)** is the GUI for that tree —
place small local models on a canvas as an *organisation*, and it **machine-checks whether the
organisation is sound before you run it**:

- will the weights actually fit in your VRAM? (a node with no `model_by_tier` still pulls
  `OllamaBackend.DEFAULT_MODEL` — 9.6GB — so a budgeter that counts only explicit params lies)
- is what ran the thing you placed? (trace ⇄ structure, so a silent tier fallback is caught)
- **is the consensus grounded?** an `EnsembleBackend` with no external `verify` anywhere has
  grounding *g* = 0 — and below the critical *g\* = 0.225* a unanimous vote can lock onto a
  wrong answer (souteni H2). The frontier models herd *hardest* here.
- does the mesh actually ignite? `ignites()` is `mesh_gain > 1e-9`, i.e. **True for any ρ < 1** —
  so yoriai reports the *size* of the gain, and calls anything at or below the once-retracted
  +0.042 `marginal` rather than "it fired".

Same round-trip both ways: every config in `examples/` and `recipes/` loads into the canvas and
writes back byte-identical (that's yoriai's L0-1).

### Let it grow itself — `gama grow`
`bench` measures a combination *you* wrote. **`grow` writes the combinations.** It mutates the
config one move at a time (route a class to another model, wrap a lane in `tool`, ensemble it,
escalate it under verification — or **strip structure back off**), measures every candidate with
the same deterministic checkers, and installs a new champion **only when a held-out split
confirms the win**. No model judges anything, anywhere in the loop.

```bash
gama grow --models llama3.2:3b,qwen2.5:7b,qwen2.5-coder:7b --generations 4 --width 5 \
          --out grow.jsonl --write-recipe recipes/my-box
```
It pools `wide,hard,brutal` (56 cases) by default and splits them three ways:

| split | what it may decide |
|---|---|
| `search` | measure candidates, pick **one** challenger per generation (the max of K noisy scores is biased upward, so this earns a challenge, not a promotion) |
| `confirm` | the only thing that can promote: the challenger must beat the champion here by at least **max(one confirm case, the champion's own re-measurement drift)** — a win smaller than one whole case, or smaller than your setup's noise, is not a win |
| `sealed` | nothing. Never touched until the run ends, then opened once — so the headline number is the one no decision was fitted to |

Run on a WSL2 box with CPU-only ollama, over the 56-case `wide,hard,brutal` pool
(split 28 / 15 / 13):

| generation | challenger | search | confirm | verdict |
|---|---|---|---|---|
| 0 | `ensemble:qa(llama3.2:3b+qwen2.5-coder:7b)` | 0.607 → 0.571 | 0.733 → 0.733 | rejected (didn't earn the challenge) |
| 1 | `meshflow:research(llama3.2:3b→qwen2.5-coder:7b)` | 0.607 → 0.548 | 0.667 → 0.800 | rejected (didn't earn the challenge) |
| 2 | `tool:qa(llama3.2:3b)` | 0.607 → 0.714 | 0.733 → **0.867** | **promoted** |

**Sealed split (13 cases, never used for any decision): seed 0.577 → champion 0.846.** The one
confirmed win transfers to cases no decision was fitted to, and the size of it is accounted for:
3 of the 13 sealed cases are `qa`, which is exactly the class the promotion changed.

Two things in that table are the loop working, not the loop being lucky:

- **Generation 1 lost on `search` while scoring higher on `confirm` (0.800 vs 0.667).** A loop
  that promoted on confirm alone would have taken it. It has to win the split it was selected
  on *first*, or "it won" just means "we looked at confirm enough times".
- **The champion's own confirm score moved 0.733 → 0.667 → 0.733 between generations** without
  the champion changing. That is the measurement noise, and it is why the bar δ is read off that
  drift instead of being a constant someone picked. Its floor is not a constant either: it is
  **one confirm case** (1/n), so "smaller than one case" never promotes at any suite size — a
  fixed 0.05 would be half a case on an 8-case confirm split and *stricter than a whole case*
  on a 30-case one, i.e. the gate would silently change meaning when you changed the suite.

Run it three times and it promotes that same single change every time. The whole-split sealed
totals, though, come out **+0.269 / +0.186 / +0.019** — the third one looks like the effect
vanished. It didn't: the champion reroutes only `qa`, which is 3 of the 13 sealed cases, so ten
cases of unrelated noise are averaged into every one of those totals. Measured on the cases that
actually changed (3 reps each), `qa` goes **0.222 → 1.000** and every other class sits still.

That is worth more than the number: **a whole-split score dilutes a class-restricted change with
the noise of every class it did not touch.** If you change one lane, measure that lane's cases.
All three runs, the per-case table and every rejected candidate are checked in as
[`recipes/grown-wsl-ollama`](recipes/grown-wsl-ollama), the first recipe here that no human wrote.

An earlier run of the same loop, on the 26-case pool before `wide` existed, promoted twice and
then showed **0.6 → 0.6 on a 5-case sealed split** — no transfer detectable at 0.2 per case. Same
loop, same box, same discipline: what changed is that there are now enough cases to see the
answer. Grow the suite, then grow the config.

## Recipes — grow it together 🌱
`recipes/` is a community library: each recipe is a `config.json` (a combination) +
`recipe.md` (the models, the hardware, the `gama bench` numbers). Found a small-model combo
that matches a big one on your box? **Add a recipe** — see [CONTRIBUTING](CONTRIBUTING.md).
```bash
gama recipes                       # list
gama recipes mac-studio-mlx        # print a recipe's config
gama run "compute 47*53+89*17" --config recipes/mac-studio-mlx/config.json --task-type qa
```

## Honest notes
- Combining identical copies of one model does **nothing** — diversity (different blind
  spots) is what helps. `gama mesh` quantifies this as the failure correlation `rho`:
  identical/redundant members → `rho ≈ 1` → no ignition.
- A small ensemble can't fix a gap **all members share** — a common hard core is high `rho`,
  so `gama mesh` reports `gain 0`. There you need a tool, or the big model.
- Cross-architecture benchmarking needs fair answer-extraction + enough tokens, or you
  measure the harness, not the model.
- The `tool` and code benchmark cases **execute model-generated Python** — only run on
  trusted backends (opt-in, like a sandbox).

## License
MIT. Built out of the [`tehai`](https://github.com/akihidem/tehai-core) delegation layer,
extracted into a focused, standalone tool.
