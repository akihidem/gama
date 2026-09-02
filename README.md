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

**`--suite steep` is for models that have already saturated the others.** `gemma4:e2b` scores
0.95 on the other five suites pooled, which leaves a 20-case sealed split one case of headroom
— at that point `gama grow` cannot promote anything and the ceiling, not the loop, is what
stopped it. `steep` is 20 cases of exact modular arithmetic, factorial tails, CSV escaping and
LRU eviction order: hard for a 7B *in its head*, and mostly trivial for a program, so the
structure still has somewhere to help. Measured: `llama3.2:3b` **0.468**, `qwen2.5:7b`
**0.683**, `gemma4:e2b` **0.894**. Honest limit: gemma4 still fully solves 16 of the 20, so
this raised the ceiling rather than removing it.

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
**`gama mesh` — does ensembling actually help?** Any policy that returns one member's answer
(a vote, a router, a cascade, a verified union) scores at most **`1 − β`, where β is the
co-failure rate: the share of cases *every* member gets wrong**. `gama mesh` counts β from a
bench with an exact (Clopper–Pearson) 95% interval and reads the union-vs-best gain through it,
so you know *before* deploying whether combining is **certified** to ignite (the union's 95%
lower bound beats the best member's 95% upper bound), **undetermined** (a gain was observed but
the sample cannot separate it from 0 — a one-case fluke is the typical shape; say so, don't ship
it as emergence), or **dead** (nested members). The pairwise failure correlation `rho` of the analytic law
`gain = (1−rho)·(1−p)·(1−(1−p)^(n−1))` is still printed, but last: with 3+ members, identical
marginals and pairwise correlations can hide different β (Chen 2026, arXiv:2606.27288), so `rho`
explains a verdict and never certifies one:
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
- does the mesh actually ignite? yoriai reads the co-failure counts a `gama mesh` run produced
  and applies the same `verdict_from_counts` rule: `certified` only when the union's lower bound
  beats the best member's Bonferroni upper bound, `undetermined` when a gain exists but one fluke
  could explain it (the once-retracted +0.042 lands here mechanically), `dead` for nested members.
  Given only (p, ρ) it says `analytic` — a model prediction, never "it fired": the analytic
  `ignites()` is `mesh_gain > 1e-9`, i.e. **True for any ρ < 1**, and pairwise ρ cannot see β.

Same round-trip both ways: every config in `examples/` and `recipes/` loads into the canvas and
writes back byte-identical (that's yoriai's L0-1).

### Let it grow itself — `gama grow`
`bench` measures a combination *you* wrote. **`grow` writes the combinations.** It mutates the
config one move at a time (route a class to another model, wrap a lane in `tool`, ensemble it
behind an aggregator, escalate it under verification, swap the default lane every unrouted class
sits on, nest one composite inside another — or **strip structure back off**), measures
every candidate with the same deterministic checkers, and installs a new champion **only when a
held-out split confirms the win**. No model judges anything, anywhere in the loop.

Measuring the loop's own mutations is worth doing, and it found one of them broken: proposed
ensembles used `majority`, which compares replies verbatim, so on free-text answers all counts
tie and the aggregate silently becomes the *first member*. On the graded suite that scored
**0.705 — below the bare 3B's 0.830 — while costing 14x the latency**: the mutation was paying
for two models to keep the cheaper one's answer. Proposals now synthesize through the other
member (0.975 on the same cases, better on 8 of 20 and worse on none).

```bash
gama grow --models llama3.2:3b,qwen2.5:7b,qwen2.5-coder:7b --generations 4 --width 5 \
          --out grow.jsonl --write-recipe recipes/my-box
```
It pools `wide,hard,brutal` (56 cases) by default and splits them three ways:

| split | what it may decide |
|---|---|
| `search` | measure candidates; the top one goes to `confirm` as the generation's **single** challenger (the max of K noisy scores is biased upward, so this earns a challenge, not a promotion). It is a filter with **one-case resolution**, not a race: a candidate that trails the champion by more than one search case is settled here for as long as that champion stands, and never costs a confirm measurement; one that ties or trails by less keeps its challenge, because the champion's own search score is a stale one-time maximum |
| `confirm` | the only thing that can promote: the challenger must beat the champion here by at least **max(one confirm case, the champion's own re-measurement drift)** — a win smaller than one whole case, or smaller than your setup's noise, is not a win |
| `sealed` | nothing. Never touched until the run ends, then opened once — so the headline number is the one no decision was fitted to |

Twelve runs of it on a WSL2 box with CPU-only ollama (up to 86 cases, split 43 / 23 / 20)
found two structural changes: send `qa` through the tool lane, and escalate `research` to the
7B coder under external verification. Measured on the classes they serve, `qa` goes **0.222 →
1.000** (including a case no model in the pool solves unaided — asked for 3h45m in seconds they
answer 15300, asked for a program the same 3B writes `3*3600+45*60`) and `research` goes
**0.313 → 0.583**.

Getting there took retracting two published conclusions, and the second retraction is the
useful one. `research → meshflow` won only 3 of 9 attempts as an *addition*, so this project
shipped a recipe without it. Then a run seeded from the champion that contains it tried to
remove it, and removal cost 0.142 confirm against a 0.0217 noise band. **A promotion record
measures whether a change could beat whatever champion existed that day; it does not measure
what the change contributes once it is in.** Those two questions disagree, and the second one
is the one a recipe is answering. Both lanes ship now, with all twelve runs recorded:
[`recipes/grown-wsl-ollama`](recipes/grown-wsl-ollama).

Pointed at a 48B model on another machine (`Kimi-Linear-48B-A3B` served by llama.cpp on an AWS
L4, reached over SSH), the loop first refused all five `tool:<class>` mutations — each *lowered*
the confirm score — and this README said the tool lane was a tax on a model that does not need
one. Re-splitting the same 80 cases as 20 / **40** / 20, which halves the promotion floor to
0.025, promoted `tool:qa` (+0.031) and `tool:research` (+0.031) and moved the sealed score
0.833 → 0.850.

Both measurements are exact — drift was 0.0000 throughout — so what disagrees is the aggregate,
not the instrument. **The effect is heterogeneous across cases and worth about one case in
forty; a split coarser than that reports it as zero or as harm depending on the draw.** That
is the third conclusion this project has had to retract, and all three retractions came from
the same place: a confident reading of one split.

Given enough cases in the class it acts on, the loop settles — on a *different* champion than
the 3B got. With 32 `qa` cases (half of them deliberately non-computational) and 48 confirm
cases, it promoted `qa → ensemble(two temperatures of the same weights)` at **+3.00 cases** and
`research → tool` at **+1.37**, sealed 0.861 → 0.875. In the same run `tool:qa` — the 3B's
signature win — was worth **exactly zero** and refused.

That comparison is the whole argument for running the loop rather than copying a recipe:

| box | champion |
|---|---|
| WSL2 CPU, `llama3.2:3b` | `qa → tool`, `research → mesh(3b→coder7b)` |
| AWS L4, `Kimi-48B` | `qa → ensemble(temperatures)`, `research → tool` (run R; the recipe now ships run W's `qa → tool`, `research → ensemble`) |

Both classes changed hands. See [`recipes/grown-aws-kimi48b`](recipes/grown-aws-kimi48b), which
also records why `tool:qa` fell from 1.25 cases to zero: the `qa` class stopped being purely
arithmetic. A tool lane helps arithmetic; a routing table routes classes; **a benchmark whose
class does not contain what the class is will bless the wrong granularity of decision.**

One correction to the obvious lesson, because the arithmetic bites: **"use more cases" is not a
general lever.** The floor is `1/n` and a gain is `S/n`, where `S` is the case-equivalents the
change actually improves, so the two scale together and the ratio is just `S`. Adding cases in
classes a mutation never touches moves nothing. What has to be true is `S >= 1` — the change is
worth *one whole case* — so the lever is more cases **in the class being changed**, and the
loop now reports every gain in cases rather than in fractions.

Swapping the model pool entirely (`gemma4:e2b` + `qwen2.5:7b`, growing from bare) promoted
nothing in five generations, because that seed already scored 0.959 on search — **the suite is
saturated for it**. That run cannot say whether the structures transfer, and it says something
sharper instead: on the same 20 sealed cases, the bare 7.2GB model scored **0.950 at 10.85s per
case** against the structured 3B's **0.865 at 2.02s**. Structure bought a *cheap* model that
works, not a better ceiling. If you can afford the latency, scale is simpler than structure —
which is exactly the trade `gama market` exists to put a number on.

The refusals are the bulk of what the loop does, and they come from every part of the gate: a
candidate that never earned its challenge, one whose confirm gain was real but smaller than the
champion's own re-measurement drift that generation (+0.054 against a bar of 0.071), one the
split was too coarse to certify at all. The ledger keeps all of them, including the run where
the best-known change was never proposed because a narrow search width parked it behind another
candidate — absence in a ledger reads exactly like a verdict, which is the argument for keeping
one.

#### What the sealed split says about the run itself

Every gate above decides on `confirm`. `confirm` is also selected against in *every* generation,
so it reads high for the same reason `search` does — one level further in. The `sealed` split is
opened once, after the last generation, and feeds no decision, which makes it the only thing
entitled to judge the run as a whole. Across the runs where the loop promoted something:

| run | confirm says | sealed says |
|---|---|---|
| Q | +6.25% | +1.65% |
| R | +8.85% | +1.39% |
| T | +0.45% | **−1.79%** |

Four to six times overstated, and at small gains the sign flips. Run T shipped a champion,
reported `promotions: 1`, and its held-out cases said the champion was *worse* than the seed —
the number was in the ledger and nothing looked at it. A run now ends with a three-valued
verdict banded by sealed's own resolution of one case: **improved**, **regressed**, or **not
separable**. Refusing to round "cannot tell" up to "improved" is the whole point, and it changes
what this repo can claim: of the nine completed runs, **seven come back NOT SEPARABLE**. The
WSL/`llama3.2:3b` run clears the band (0.6375 → 0.8542 on 20 sealed cases, +4.33), and so does
run W on the 48B (0.7484 → 0.8214 on 32 sealed cases, +2.34), with a qualification that matters:
W was seeded from the previous champion, so the separation is from that champion, not from the
bare model, and the first link of that chain was never separable.

So the honest headline is narrower than "the loop grows better combinations". On a small model
with a saturated-nothing suite it demonstrably did. On a 48B, seven runs of promotions (every
ledger on that box that reached a final row, the discarded one included) did not produce a
champion its held-out cases could tell apart from where it started; the eighth did, after harder
cases (`crux`) were added and the seed was already grown.

#### Where the headroom actually is

That result has a measurable cause, and it is not the gates. Counting the score still unearned
in each class on one run's 56 confirm cases:

| class | cases | score | headroom (cases left to win) |
|---|---|---|---|
| integration | 8 | 1.000 | **0.00** |
| code_implementation | 8 | 0.938 | 0.50 |
| qa | 16 | 0.922 | 1.25 |
| content | 8 | 0.658 | 2.73 |
| research | 16 | 0.812 | 3.00 |
| **total** | **56** | **0.866** | **7.48** |

Forty-six of the 56 cases are already perfect. A promotion has to be worth one whole case, so
there are at most seven promotions' worth of room in the entire suite, and any single lane
change moves one to three cases — inside the sealed split's resolution. That is the mechanism
behind six NOT SEPARABLE verdicts, and it means **the lever is harder cases, not more of them**.

"Not separable" turned out to be three different results wearing one word, and the loop now
says which. The hypothesis sealed tests is *the final champion against the seed*, so the claim
to scale is how far the champion stands above the seed on `confirm` at the end (means of the
measurements that were never used for selection: every seed measurement, and the champion's
re-measurements after its promotion, not the promotion score itself), scaled to the sealed
split's size. Run T's champion ended +0.84 of 56 confirm cases above its seed, which is 0.42 of
28 sealed cases, inside the one case sealed resolves: its verdict was **fixed before the split
was opened**. Run R's ended +4.0 of 48, which should have shown as 2.0 of 24 sealed cases, and
sealed showed +0.33: there the confirm gains **did not transfer**, and the promotions were
selection noise. Run V is the third kind: its gate certified +1.1 cases at promotion time, and
the champion, re-measured every generation after (0.784 at promotion, then 0.779, 0.764, 0.761,
0.759 against a seed at 0.767), ended **0.08 cases below the seed** on confirm. The claim had
**evaporated** before sealed was opened; the promotion score was the selection, and the
re-measurements were the correction. The first kind asks for a bigger sealed split (or larger
gains); the second asks you to distrust the promotions; the third says the gate certified
noise. A run prints the sealed split's resolution in confirm cases at the seed, so an
underpowered run announces itself three hours before it ends.

One caveat on that headroom, found by auditing the checkers rather than the models: of the ten
cases the champion loses, **four are scored on output format as well as content** (their checker
rejects a correct answer wrapped in prose), and they carry 3.25 of the 7.48 cases of headroom.
For those the suite cannot separate "the model cannot do this" from "the model did not answer in
the demanded shape". Some of that is legitimate — writing a lipogram *is* a format task — but it
means a share of the remaining room may not be reasoning room at all. Confirming which needs the
replies themselves, and the ledger stores scores rather than outputs, so it is an open question
rather than a result. `crux` was written with the looser convention throughout: numeric answers
are read as "the last integer in the reply", so a model that solves the problem and then adds a
sentence still scores it.

It also gives the loop a proof rather than a heuristic. An additive lane mutation only touches
cases of its own class, so a class with less headroom than the gate cannot produce a promotable
gain whatever you try there — and `integration` at 8/8 was being handed real GPU time anyway.
Those are skipped now. Shrink mutations are *not* skipped there: the gates are asymmetric
(additions must be better, removals need only be not worse), so a saturated class is the best
place to show a structure is buying nothing, not a place to stop looking.

#### A tool lane has a precondition, and it can fail exactly where it would help

`crux` was built to un-saturate this box, and it did: the bare model scores 0.306 on it against
0.866 on the old pool, and 11.80 of 17 cases are still winnable against 7.48 of 56 — five times
the headroom per case. Then the shipped champion, which routes `research` through a `tool` lane,
fixed **zero of the four** crux research cases. Amicable numbers, the longest Collatz chain,
derangements, Josephus: four problems a five-line program solves.

The lane was never running a program. On those prompts the model emits no ```python block at
all, so `ToolBackend` falls back to returning the raw reply — and the raw reply is unfinished
reasoning, which scores *worse* than the bare model because the bare model at least answers.
None of this raised an exception, so it arrived as an ordinary low score.

What it is not, measured rather than assumed:

| suspicion | test | result |
|---|---|---|
| the generated code times out | time the naive solutions | 2.9s and 0.7s against a 15s limit |
| the prompts conflict (case says "reply with only the integer", the PAL wrapper says "only code") | resolve the conflict explicitly | still no code block |
| the token budget is too small | 1536 / 4096 / 8192 | 1536, 4096 and **8097** completion tokens, no code block in any; at 8192 it stopped on its own |
| greedy decoding is stuck in a loop | temperature 0.0 vs 0.8, 3 cases each | 0 of 3 against 1 of 3 — inside noise, not a fix |
| the model will not *open* the fence | send ` ```python ` as the start of its reply (a trailing assistant turn) | **2–3 of 3 replies contain code**; 1 of 3 correct |

So the honest reading is about the model: `Kimi-Linear-48B-A3B-Instruct-IQ2_M` is a roughly
two-bit quantization, and reliably emitting a fenced code block on a multi-sentence prompt is
exactly the kind of instruction-following that degrades. **A PAL lane is not a free structural
win; it is a bet that the model will actually write the program.** That bet gets *worse* as the
problem gets harder, which is the opposite of where you want it.

The loop cannot fix this, but it should never have hidden it. Tool calls and successful runs are
counted per measurement now and reported, so "the tool lane scored badly" and "the tool lane was
never used" stop looking identical.

The obvious next step was to let the loop remove the lanes, and the prediction here was that it
would. It did not, and on its own numbers it was right not to: seeded from the shipped champion
with `crux` in the pool, `simplify:qa → bare` measured 0.729 against the champion's 0.761 on
confirm and `simplify:research → bare` 0.743 against 0.759 — each tool lane still earns one to
two confirm cases net, while failing every crux research case. A lane can be broken on the cases
you built to expose it and still be paying for itself on the rest. The recipe keeps both, and the
[`recipes/grown-aws-kimi48b`](recipes/grown-aws-kimi48b) notes record the run.

The last row of the table is the one thing that moved the model, and it is a change to the
*lane*, not the prompt: `ToolBackend(prefill="```python\n")` hands the model the opening of its
own reply, so the only thing left to decide is what goes inside the fence. It is opt-in per
lane, because on the suites where this model already writes code it is an unmeasured change —
and the loop is the thing that measures. So it is a mutation: once a class sits on a `tool`
lane, the next one-step refinement the loop proposes is `tool:<class>(model)+prefill`, gated
like everything else. Two of three probes reached code and one of three was right; whether that
buys a confirm case is for the ledger to say. (The reply comes back in one of three shapes
depending on how the server treats a trailing assistant turn — continued, re-opened, or opened
and never closed — and the extractor reads all three, since the difference is the server's, not
the model's.)

The queue reads that diagnosis. Every measurement records, per class, how many tool calls came
back without a code block (`tool_no_code_by_class`), and the `+prefill` for the most symptomatic
class goes to the front of the tool slot. Without it the order is the rotation's: run W's seed
said "no code, 8 of 72 confirm calls", all of it research, and gen1's tool slot still went to
`tool:qa(kimi-cold)+prefill`, a class whose tool lane was already returning code. Replaying
`propose()` from that checkpoint (no model involved) puts the research prefill in gen2, one
rotation step later. This paragraph first said "not in five generations", worked out from the
rotation rules alone; the replay corrected it, and one generation is what the diagnosis bought
in run W. What it changes is where the order comes from: the measurement, rather than the
position the rotation happened to start at.

Being proposed is not being measured on `confirm`, though. Among candidates tied on `search`
the challenger is chosen by structural size and then by label; a prefill keeps the champion's
size, and `route:` and `ensemble:` sort before `tool:`. Replaying from run W's seed with its
diagnosis (all research), the research prefill is proposed in gen0 and listed in every generation
after, and the challenger goes to `route:content`, `ensemble:research`, `default`, `route:research`,
`route:qa`: five generations, never the prescription. So the diagnosis names a *prescription*, and
the prescription takes the first seat of the generation, ahead of the kind rotation, and wins ties
for the confirm measurement. It costs one measurement, once: measured, it stays in the pool at zero
calls; rejected, it is excluded. Whether it buys a confirm case is for the ledger to say.

#### Things the loop was doing wrong and could not see

**It was not deterministic.** Among candidates tied on `search`, the challenger was picked by
measured latency — wall clock, which moves with someone else's load on a shared box. Two runs
over identical inputs could take different paths. The determinism test had existed all along and
passed every time anyone ran it; it fails 4 times in 80. Ties break on structural size now, which
is a pure function of the spec and is the thing actually generating the extra calls.

**It assumed the backend stayed the same.** One run was decided against a server that its owner
restarted onto a different model mid-run; it was caught only because the swap returned 503s and
tripped the error-rate guard. A swap that keeps returning 200s would have compared generation 0
and generation 5 across two different models and reported one smooth improvement. Responses
already name the model they came from, so the loop now checks that each destination keeps
resolving to the same weights, at zero extra calls, and records what it measured into the recipe
— "Kimi-48B" was never enough to reproduce a number when a different quantisation is a different
model.

**It raced `search` against a stale maximum.** Gate ① demanded that a challenger strictly beat
the champion on `search`. But the champion's search score is measured once, at promotion, as the
maximum over that generation's width — and never again, while its confirm score *is* re-measured
every generation (one run: 0.784 → 0.779 → 0.764 → 0.761 → 0.759). Measuring the same spec twice
on a 32-case search split moved it by exactly one case (0.896 and 0.865). At that resolution the
loop was reading a half-case deficit as a loss, and it discarded — without ever measuring them
on confirm, since confirm is only measured for the one challenger — candidates that had already
been measured at **4 wins, 1 loss** and **4 wins, 0 losses** (p = 0.0625) on confirm. `search` now
settles only what trails by more than one search case; ties and sub-case deficits keep the
challenge, and the confirm gate decides. The side effect is a saving: a candidate settled on
search costs nothing further, where before every generation's challenger paid a full confirm
measurement whatever the verdict.

**A deepened lane was a dead end.** The loop can nest a `tool` stage inside a `mesh` or
`ensemble` lane (`mesh(a->b)+tool`). Two copies of the function that finds a composite lane's
base model sat in the same file — one reading the spec, one parsing the name — and the
name-parsing one won by definition order. It recognised `tool(a)` and `mesh(a->b)` but not
`mesh(a->b)+tool`, so a class that had been deepened could never be simplified back, wrapped
differently, or recombined: only routed away. Every direction a mutation can take must stay
reachable from every shape the loop can produce, or the loop drifts one way and calls it a
result. The spec is read first now; names are parsed only inside the namespace the loop itself
mints.

**Measured designs were taking the seats.** `width` is the number of designs the loop measures
per generation. A design already in the archive costs nothing to propose again, and it has to be
proposed again: a candidate inside the band that was not the maximum is the stepping stone that
can challenge from the archive once the maximum has been rejected on `confirm`. But each one took
a width slot. In the all-tie regime (run W's search split has four unsolved cases, so most
mutations cannot move it) the replay from W's gen0 checkpoint measures 4, 3, 2, 3, 1, 0 new
designs over six generations, and the research prefill, proposed in gen2, never appears again: a
stone came back only when the rotation happened to land on it. The archive is the challenger pool
now. Every measured, unsettled design is listed every generation outside the width (the same
replay: 4, 4, 4, 3, 0, 0, the zeros because the fifteen one-step designs are all measured by then),
and the ledger separates `candidates` (listed) from `new_candidates` (measured) and records whether
the challenger was new or came from the archive.

**The recipe named the wrong code.** The ledger stamps the git commit so that two runs under the
same conditions can be told apart when the judging changed between them. It read `HEAD` at write
time. Run W's seed row says `e7e5e63`; its recipe says `ffdb5bf`, a commit that landed while the
run was still measuring and contributed nothing to its numbers. Run X was worse: it imported
`HEAD` plus an uncommitted edit, and the stamp said `HEAD`. The stamp is taken once per run now,
and it carries `dirty`, whether the package directory differed from the commit at that moment.

**The gate borrowed its noise from the wrong side.** A promotion must clear
`max(one confirm case, the champion's own re-measurement drift)`. That drift is the champion's,
and on the 48B pool the champion runs at temperature 0: drift 0.0 every generation, so the bar
was always the one-case floor. The challenger's noise entered nowhere. Run V promoted
`route:content->kimi-hot` at +1.1 cases (2 wins, 0 losses); run W measured the same design on
the same 65 confirm cases at −0.9 (1 win, 3 losses). Nothing in the ledger could have predicted
the 2.0-case swing, although the repeats that every measurement already pays for carry the
estimate: each measurement now records the standard error of its mean from its repeats, and
every comparison (the challenger's gain, the simplifier's drop, the sealed verdict) writes the
re-measurement noise of the pair beside the gain, in cases. It is a record, not a gate: no
earlier ledger holds the number, so how many past promotions it would have stopped is not known,
and a gate whose blast radius is unmeasured goes in after the number has been carried, not before.

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
  spots) is what helps. `gama mesh` counts this as the co-failure rate `β`: identical/redundant
  members fail on the same cases → the union is the best member → verdict `dead`.
- A small ensemble can't fix a gap **all members share** — a common hard core *is* β, and no
  vote/router/cascade over those members can score above `1 − β`. There you need a tool, or the
  big model. (Pairwise `rho` cannot see this tail once there are 3+ members; that is why β is
  the number `gama mesh` prints first.)
- Cross-architecture benchmarking needs fair answer-extraction + enough tokens, or you
  measure the harness, not the model.
- The `tool` and code benchmark cases **execute model-generated Python** — only run on
  trusted backends (opt-in, like a sandbox).
- A gain measured on the split that selected it is not an estimate of the gain. If you quote a
  `grow` number, quote the **sealed** one, and quote the verdict with it.
- `grow` records per-case wins and losses for every challenge and prints the sign-test p-value,
  but does **not** gate on it by default: at these case counts significance demands a near-sweep
  (5–0, or 8–1), so switching it on would freeze the loop. `--max-paired-p` is there if you want
  it. The number is worth reading either way — it is usually 1w–2l out of 56, which is the whole
  story about why these suites cannot separate structures.

## License
MIT. Built out of the [`tehai`](https://github.com/akihidem/tehai-core) delegation layer,
extracted into a focused, standalone tool.
