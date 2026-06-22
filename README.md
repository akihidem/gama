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

## The result
Hard 12-task suite, fully local on a Mac Studio (MLX). Measurement made fair (code
extraction + token budget) — read this as *competitive/tied*, not a clean win:

| | sovereign light stack (7B+24B+32B+tool, routed) | single 122B |
|---|---|---|
| score | **0.92** | **0.92** |
| | misses 1 (a day-of-week puzzle) | misses 1 (a roman-numeral coder task) |

Complementary blind spots, same score — all local. Reproduce:
`python3 -m experiments.moa_vs_strong <config.json>`.

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
  spots) is what helps.
- A small ensemble can't fix a gap **all members share** — there you need a tool, or the
  big model.
- Cross-architecture benchmarking needs fair answer-extraction + enough tokens, or you
  measure the harness, not the model.
- The `tool` and code benchmark cases **execute model-generated Python** — only run on
  trusted backends (opt-in, like a sandbox).

## License
MIT. Built out of the [`tehai`](https://github.com/akihidem/tehai-core) delegation layer,
extracted into a focused, standalone tool.
