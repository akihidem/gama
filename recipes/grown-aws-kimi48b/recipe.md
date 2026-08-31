# AWS L4 + Kimi-48B — the same loop, a different box, a different champion

The second recipe in this library grown by `gama grow`, and the reason it is worth reading next
to [`grown-wsl-ollama`](../grown-wsl-ollama): **same suites, same mutation set, same gates —
and a different answer.**

| box | champion the loop settled on |
|---|---|
| WSL2, CPU, `llama3.2:3b` | `qa → tool`, `research → mesh(3b → qwen2.5-coder:7b)` |
| **AWS L4, `Kimi-Linear-48B-A3B`** | **`qa → tool`, `research → tool`** (the `qa` lane is a coin-flip between `tool` and an ensemble — see below) |

Both classes changed hands. Structure is not portable; the loop is how you find out what your
box wants.

## Hardware / runtime
- AWS L4 24GB (a shared box), one `llama.cpp` server already running `Kimi-Linear-48B-A3B-IQ2_M`
  on localhost:8000, reached with gama's `ssh-openai` backend over an SSH control-master.
- **No second model was loaded.** The two lanes are the same weights at `temperature` 0.0 and
  0.8, so the pool costs nothing extra on a GPU somebody else is using. Sub-second per call.
- Cases: `wide,graded,steep,qadeep` (96) at `--ratio 1:2:1` → **24 search / 48 confirm / 24
  sealed**, `--repeats 2`.

## Result

| gen | challenger | worth | verdict |
|---|---|---|---|
| 0 | `ensemble:qa(cold+hot)` | **+3.00 cases** | **promoted** |
| 1 | `tool:qa(cold)` | **±0.00 cases** | refused |
| 2 | `tool:research(cold)` | **+1.37 cases** | **promoted** |
| 3 | `ensemble:content(cold+hot)` | −0.12 cases | refused |
| 4 | `deepen:qa(tool inside ens)` | ±0.00 cases | refused |
| 4 | `simplify:qa → cold` (removal) | — | refused: measurably worse |

Sealed 0.861 → 0.875. Cost: 2.49s → 4.39s per case, because the winning `qa` lane spends three
calls where the bare model spent one.

## The finding worth carrying away

`tool:qa` was promoted on this same model a run earlier, worth 1.25 cases. Here it is worth
**exactly zero**. The difference is not the model or the split discipline — it is what the `qa`
class contains.

| run | what `qa` held | `tool:qa` |
|---|---|---|
| Q | 16 cases, **all exact computation** | +1.25 cases → promoted |
| R | 32 cases, **half computation, half short-answer recall** | ±0.00 cases → refused |

The tool lane helps *arithmetic*. It does not help *the class*, and a routing table routes
classes. Adding non-computational `qa` cases (a chemical symbol, a continent, an antonym —
things no program can supply) was a deliberate choice when building `qadeep`: filling the class
with calculator problems would have produced a benchmark that proves the tool lane right by
construction. **A benchmark whose class does not contain what the class is will bless the wrong
granularity of decision.**

What took the class instead was self-consistency: two samples of the same weights at different
temperatures, aggregated. On the 20-case confirm split of an earlier run that lane measured
+0.25 cases and was refused as unprovable; with 16 `qa` cases in confirm it measures +3.00 and
clears easily. Same effect, enough cases to see it. The shrink gate then tried to take it back
out in generation 4 and refused — it is load-bearing, not decoration.

## The audit, applied to both classes

`qadeep` had shown that `qa` was 16 for 16 exact computation, so `researchdeep` did the same
thing to the other suspicious class: 16 more `research` cases, 8 computational and 8 that no
program can supply (temporal order, referent resolution, odd-one-out, analogy, contradiction,
category deduction, negation, causal chain). `tool:research` had been the one mutation
reproducing across splits, and the question was whether it survives a class that contains what
the class is supposed to be.

**It does.** With the deepened class in place (112 cases, 56 confirm), removing the lane was
put to the shrink gate and refused:

| audit | result |
|---|---|
| `simplify:qa` → bare model (run R, run T) | refused, measurably worse — `qa` needs *a* lane |
| `tool:qa` vs `ensemble:qa` (run R, after qadeep) | ±0.00 cases — tool adds nothing over the ensemble |
| `simplify:research` → bare model (run T, after researchdeep) | **refused, measurably worse — the lane earns its place** |

So of the two classes that looked like they might be named for something they did not contain,
only `qa` was. `research` holds up.

## What is still undecided, and what ships because of it

The `qa` lane is not settled, and the two measurements disagree by less than they can resolve:

| split | winner | margin |
|---|---|---|
| confirm, 56 cases | `tool` | +1.25 cases |
| sealed, 28 cases | `ensemble` | 0.5 cases |

Both inside one case, with the sign flipped. What *is* settled is that removing the lane
altogether is worse, in both runs. So `qa` needs a lane and this box cannot yet say which.

This recipe ships **`tool`**: at a statistical tie, take the cheaper and simpler option — one
call instead of three, deterministic instead of stochastic. That is the same Occam the shrink
gate encodes, applied by hand where the loop has no rule for it. If your own run separates them,
prefer what it measures over what this file shipped.

## A run that had to be thrown away

Run S ran this exact configuration first and produced a champion out of nothing. Partway
through, the owner of the shared box restarted the llama.cpp server onto a different model;
every call after that returned a 503 body, `run_bench` caught the exceptions and scored the
cases 0.0, and the loop saw a perfectly coherent measurement — champion 0.0, challenger 0.0,
drift 0.0 — and concluded that dropping the research lane cost exactly nothing. It deleted it
and reported a sealed score of 0.0 -> 0.0.

Every gate worked correctly on numbers that meant nothing. `grow` now separates "the model was
wrong" from "the call raised", and stops the run above a 20% failure rate rather than deciding
on it. Its ledger is kept as the record of what a broken measurement looks like from inside.

## Reproduce
```bash
# the pool is two temperature settings of ONE served model; adjust ssh_host/port for your box
gama grow --pool pool.json --suites wide,graded,steep,qadeep --ratio 1:2:1 \
          --generations 5 --width 4 --repeats 2 --out ~/gama-runs/run.jsonl
```
`pool.json` holds two `ssh-openai` lanes differing only in `temperature` (0.0 and 0.8).

## Notes (honest)
- **The bar was resolution, never noise**: drift measured exactly 0.0000 in all five
  generations (temperature-0 lane, deterministic checkers), so every decision was against the
  one-case floor. On a served model the loop has no measurement noise to fight, which is why
  the case count is the only thing that matters here.
- Two of five classes now cost extra calls, and `qa` costs three. Read the 1.76x latency as the
  price of +1 sealed case; the loop optimises score and has no opinion about cost.
- One box, one model, one quantisation (IQ2_M). The point of this file is not its champion but
  that its champion **differs** from the one the same loop found elsewhere.
