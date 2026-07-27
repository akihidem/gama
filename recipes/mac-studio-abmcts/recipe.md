# Mac Studio — AB-MCTS-A adaptive branching search vs width-only best-of-N

Sakana AI's **AB-MCTS** ("Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive
Branching Tree Search", [arXiv:2503.04412](https://arxiv.org/abs/2503.04412), NeurIPS 2025;
blog [sakana.ai/ab-mcts](https://sakana.ai/ab-mcts/); OSS [SakanaAI/treequest](https://github.com/SakanaAI/treequest))
generalizes repeated sampling (all width) and sequential refinement (all depth) into one
search: at every node it decides — via Thompson sampling over a Bayesian reward posterior —
whether the next model call should **go wider** (a brand-new candidate) or **go deeper**
(refine an existing one). `ABMCTSBackend` (`gama/abmcts.py`) ports the closed-form
**AB-MCTS-A** variant to gama, using gama's external `verify(artifact)->score in [0,1]` as the
reward and each worker backend as one bandit "action" (so it also learns *which* model to call
= Multi-LLM AB-MCTS).

## Scope (read this before the numbers)

- **-A, not -M.** The paper's AB-MCTS-**M** (a hierarchical mixed model) needs MCMC via
  PyMC/numpyro; gama is **stdlib-only**, so this ports AB-MCTS-**A** (per-node Beta/Gaussian
  conjugate posteriors), whose Thompson sampling is just `random.betavariate`. The paper's own
  results show -A captures nearly all the benefit. This is the same honesty stance as
  `gama/trinity.py` porting openfugu's *structural* idea without its CMA-ES hidden-state head.
- **Beta reward model.** gama's verify score is already in `[0,1]`, the exact conjugate match
  for `Beta(0.5, 0.5)` (Jeffreys). Reward `r` updates `a += r; b += 1-r`.
- **The load-bearing detail:** "go deeper" feeds the parent answer **and its score** back into
  the refine prompt (`_refine_prompt`). Without that, depth degenerates into width and the whole
  thing is just best-of-N (prior art: Reflexion / LATS). That is exactly why the control below
  matters.

## Hardware / runtime
- Mac Studio, **MLX** (`mlx_lm.server`, OpenAI-compatible), reached over **SSH** (no open port;
  prompt on stdin) — `SshOpenAIBackend`.
- weak = Qwen2.5-Coder-7B-Instruct-4bit (port 8082) / strong = Devstral-Small-2-24B-Instruct-2512-4bit
  (port 8080). Set `ssh_host` to your Mac's SSH host (`user@host`).
- The Mac Studio's standing config keeps **122B solo** resident as the shared Claude-Code
  fallback (Devstral+122B+7B together OOM at 128GB), so a pass using 7B+24B only runs them for
  the duration of the bench and restores 122B-solo afterward (same make-room discipline as the
  other Mac Studio recipes).

## The combination
| | abmcts | ensemble (control) | flat-strong (24B alone) |
|---|---|---|---|
| shape | adaptive tree: wider vs deeper per node, Thompson-sampled | width-only best-of-N (strategy=first) of the same 2 workers | none (always 24B) |
| model choice | per-widen Multi-LLM bandit over {weak, strong} | fixed member list | fixed |
| reward use | posteriors steer width/depth + model, gated by the case checker | picks first non-empty (no reward steering) | n/a |
| cost model | summed cost of every generation made (`budget` calls) | one call per member | flat 24B cost |

## What to measure (not yet run on real models)

This recipe ships the config and the protocol; the numbers below are intentionally left as a
template rather than fabricated. Run it on your box and fill them in (then send a PR — that is
what `recipes/` is for). The **honest question** is not "does search beat one model" but:

> Does the *adaptive* wider/deeper search beat a *width-only* best-of-N of the same models at a
> comparable call budget?

```bash
gama bench --suite hard --backends abmcts,ensemble,ssh-openai \
  --config recipes/mac-studio-abmcts/config.json --tier large --repeats 3
```

| suite=hard, n=10/class | abmcts (budget 12) | ensemble best-of-N | flat-strong (24B) |
|---|---|---|---|
| pass_rate | _TBD_ | _TBD_ | _TBD_ |
| avg cost (price proxy) | _TBD_ | _TBD_ | _TBD_ |

Read `last_trace` (per-iteration `{model, depth, mode, score}`) to see whether the search is
actually **going deeper** on hard cases (mode=`refine` at depth>1) or collapsing to all-width —
if every winning node is depth 1, AB-MCTS bought you nothing over the `ensemble` control and you
should say so. Consistent with gama's thesis: **structure, not scale — falsify it on your own
hardware.**

## Notes
- `budget` is the number of model calls = the compute budget. Bump it for harder suites; the
  search stops early as soon as a candidate hits `pass_score` (default 1.0).
- `seed: null` gives a fresh nondeterministic search per call (so `--repeats` measures real
  variance). Set an integer `seed` for a reproducible single run (used by the tests).
- SECURITY: with `"verify": "code_runs"` (or the code bench cases) the reward **executes
  model-generated Python** — opt-in, like `ToolBackend` / `--sandbox`. Only run on trusted
  backends.
