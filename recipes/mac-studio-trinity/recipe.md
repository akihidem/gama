# Mac Studio — one-shot predictive routing (trinity) vs verified escalation (meshflow)

Does openfugu's TRINITY *structural* idea -- a cheap classifier picks ONE worker up front,
dispatch once, never retry -- beat gama's own `MeshflowBackend` (sequential cheap->expensive
escalation, gated by the case's own checker) on the exact same two local models?

## Scope (read this before the numbers)

[openfugu](https://github.com/trotsky1997/openfugu) is Sakana AI's "Fugu" OSS reimplementation.
Its TRINITY component scores workers from a small model's **hidden states**, via a head trained
with **CMA-ES** (gradient-free). This recipe does **not** reproduce that: the Mac Studio serving
stack (`mlx_lm.server`, OpenAI-compatible) does not expose hidden states, so `TrinityBackend`
(`gama/trinity.py`) picks a worker via a single cheap **prompted classification call** instead of
a trained router. What's measured here is the narrower, honest question: *is one-shot predictive
routing (the structural shape) competitive with gama's verified escalation on gama's own hard
suite?* It is **not** a reproduction of openfugu's own claimed "+107% over best single model"
number (different tasks, different models, a trained router).

The Mac Studio's standing config keeps **122B solo** resident as the shared Claude-Code
fallback (Devstral+122B+7B together were proven to OOM at 128GB), so any pass using 24B
and/or 7B here only runs them for the duration of the bench and restores 122B-solo
afterward -- see Notes. Two passes are recorded below: 24B as the strong tier first, then
(see "Second pass") 122B as the strong tier, to check whether a bigger capability gap
changes the verdict.

## Hardware / runtime
- Mac Studio, running **MLX** (`mlx_lm.server`, OpenAI-compatible), reached over **SSH**
  (no open port; prompt on stdin) -- `SshOpenAIBackend`.
- weak = Qwen2.5-Coder-7B-Instruct-4bit (port 8082) / strong = Devstral-Small-2-24B-Instruct-2512-4bit
  (port 8080, `config.json`) or Qwen3.5-122B-A10B-4bit (port 8081, thinking model,
  `config-122b.json` -- needs `max_tokens: 8192`+ or the truncation trap eats the answer
  before it's reached). Set `ssh_host` in whichever config to your Mac's SSH host (`user@host`).

## The combination
| | trinity | meshflow | flat-strong (24B alone) |
|---|---|---|---|
| decision | 1 classifier call -> 1 worker, no retry | cheap->expensive, gated by the case checker | none (always 24B) |
| cost model | scorer_cost + chosen tier cost (fixed) | cumulative cost up to the tier that passes | flat 24B cost every case |

## Result (real run, 2026-07-07, `gama bench --suite hard --backends trinity,meshflow,ssh-openai`)

| | trinity | meshflow | ssh-openai (flat-strong, 24B alone) |
|---|---|---|---|
| pass_rate (n=10) | **0.7** | **0.8** | **0.8** |
| avg latency_s | 2.61 | 0.87 | 2.63 |
| avg cost (tokens) | n/a (`unit_cost` not set) | n/a | n/a |

Per task-class (`hard` suite, 2 cases each):

| class | trinity | meshflow | ssh-openai |
|---|---|---|---|
| code_implementation | 1.0 | 1.0 | 1.0 |
| qa | 0.5 | 0.5 | 0.5 |
| **research** | **0.5** | **1.0** | **1.0** |
| content | 0.5 | 0.5 | 0.5 |
| integration | 1.0 | 1.0 | 1.0 |

**trinity loses, cleanly, on one class**: `hard-reason-weekday`. Its one-shot classifier bet
on the *weak* (7B) tier; 7B gets that riddle wrong. meshflow, given the same weak-tier miss,
escalates to strong (24B) and gets it right; ssh-openai always uses 24B and gets it right too.
Every other class ties exactly across all three backends. `trinity`'s classifier picked
correctly (matched a real label, not a parser artifact -- see Notes) on all 4 qa/research
cases; this is a genuine "no retry" loss, not a broken proxy.

**Verdict**: on this suite/these two models, one-shot predictive routing does **not** beat
gama's verified escalation, and does not beat simply always using the strong model either
-- it ties everywhere escalation isn't needed and loses exactly where escalation earns its
keep. `reject-by-measure` for a fuller trinity build; the code stays in gama as a real,
working measurement point (`ledger` id `openfugu-trinity-vs-gama-20260704`).

## Second pass (real run, 2026-07-07, weak=7B / strong=122B, `config-122b.json`)

Same question, a much bigger capability gap: does a stronger fallback change the verdict?

| | trinity | meshflow | ssh-openai (flat-122B alone) |
|---|---|---|---|
| pass_rate (n=10) | **0.9** | **1.0** | **1.0** |
| avg latency_s | 14.73 | 6.17 | 34.49 |

Per class: everything ties at 1.0 **except `research`, where trinity is still 0.5** -- the
exact same case (`hard-reason-weekday`) fails for the exact same reason as the 24B pass:
the classifier correctly picks `weak` (7B) both times (`last_fallback=False`, verified via
`last_trace`), and 7B gets that specific riddle wrong regardless of which model backs it up.
meshflow, given the identical weak-tier miss, escalates to 122B and gets it right; flat-122B
solves everything on its own.

**This reproduces, not just resembles, the first pass's structural conclusion**: making the
fallback option far more capable (122B vs 24B) does not rescue a one-shot router's wrong bet
-- it only matters if that bet gets escalated past, which trinity's design never does. Trinity's
*overall* score went up (0.9 vs 0.7) purely because 7B's other misses happened to not recur
here, not because the routing decision improved. Latency also flips the earlier picture:
meshflow (6.17s) beats flat-122B (34.49s) by staying on the fast 7B tier whenever it already
passes, while trinity (14.73s) sits in between (it always pays for whichever single tier the
classifier picked, cheap or not).

## Notes (be honest)

- **Single run, small suite** (`hard`, n=10, 2 reps/class) per pass. Numbers will wobble --
  re-run and PR your own. The 122B-tier second pass reproduced the exact same single
  failing case as the 24B pass, which is reassuring (not noise) but still n=1 per config.
- **Two real bugs found and fixed mid-measurement** (both in `TrinityBackend._pick`,
  `gama/trinity.py`), worth knowing if you build on this:
  1. The scorer's classification prompt originally asked for a label *before* the query
     text. The real 7B worker just answered the query instead of naming a label (**100%
     fallback rate** on first run). Moving the "pick a worker, don't answer it" instruction
     to *after* the query (recency-anchored) fixed this on the same model/suite.
  2. Even once the model correctly replied `"strong"`, the MLX server left the chat
     template's end-of-turn marker glued on with no separating whitespace
     (`"strong<|im_end|>\n"`), which broke naive exact-word matching and silently forced
     every call to the cheapest-worker fallback. Fixed by stripping `<\|.*?\|>`-style
     tokens before matching (regression test:
     `test_special_end_of_turn_token_glued_to_reply_still_parses`).
  Before both fixes, `TrinityBackend.last_fallback` was `True` on every single case --
  exactly the diagnostic this recipe's own design calls for: a 100% fallback rate means the
  *prompted proxy*, not the structural idea, is broken. After fixing both, fallback rate on
  the checked cases dropped to 0% and the classifier's picks were verified against
  `last_trace` to be genuine (not accidental matches).
- **Operational note**: the Mac Studio's standing config (`~/bin/llm-stack-up.sh`, dated
  2026-06-24) keeps 122B solo resident as the shared Claude-Code-fallback default --
  Devstral+122B+7B together were proven to OOM at 128GB. 24B+7B were started only for this
  measurement window and 122B was restored immediately after.
