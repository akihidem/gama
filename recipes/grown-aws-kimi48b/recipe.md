# AWS L4 + Kimi-48B — the same loop, a different box, a different champion

The second recipe in this library grown by `gama grow`, and the reason it is worth reading next
to [`grown-wsl-ollama`](../grown-wsl-ollama): **same suites, same mutation set, same gates —
and a different answer.**

**Held-out verdict: IMPROVED over the previous recipe, by +2.34 of 32 sealed cases (run W,
band 1.0).** The champion shipped here is `qa → tool`, `research → ensemble(cold + hot)`. It was
grown from the previous champion of this file (`qa → tool`, `research → tool`), and that is the
seed the sealed split compares it against. Seven runs before it on this box ended NOT SEPARABLE
(O to V, every ledger that reached a final row; S is the discarded one, U stopped early and is
not counted), including every run that started from the bare model, so the chain reads: bare
model → first
champion (never separable), first champion → this one (separable by 2.34 sealed cases). The gap to
the bare model was not measured. The earlier reason still stands for the first link: across 56
confirm cases of the original suite there were 7.48 cases of score left to win, 46 already
perfect, `integration` at 8 of 8. **That suite is saturated for a 48B**; the `crux` cases added
for runs V and W are what gave the loop something to move. Read the sections below in order; the
numbers of the early runs are on the 96-case suite, run W's on 129.

| box | champion the loop settled on |
|---|---|
| WSL2, CPU, `llama3.2:3b` | `qa → tool`, `research → mesh(3b → qwen2.5-coder:7b)` |
| **AWS L4, `Kimi-Linear-48B-A3B`** | **`qa → tool`, `research → ensemble(cold + hot)`** (run W; the `qa` lane is a coin-flip between `tool` and an ensemble — see below. Runs R–T settled on `research → tool`) |

Both classes changed hands. Structure is not portable; the loop is how you find out what your
box wants.

## Hardware / runtime
- AWS L4 24GB (a shared box), one `llama.cpp` server already running `Kimi-Linear-48B-A3B-IQ2_M`
  on localhost:8000, reached with gama's `ssh-openai` backend over an SSH control-master.
- **No second model was loaded.** The two lanes are the same weights at `temperature` 0.0 and
  0.8, so the pool costs nothing extra on a GPU somebody else is using. Sub-second per call.
- Cases: `wide,graded,steep,qadeep` (96) at `--ratio 1:2:1` → **24 search / 48 confirm / 24
  sealed**, `--repeats 2`. Runs V and W add `researchdeep,crux` (129) → **32 / 65 / 32**.

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

## Run V: the shipped champion against `crux`, and what the loop could not see

Seeded from this recipe, with the `crux` suite added to the pool (129 cases, split 32 / 65 / 32,
`--generations 5 --width 4 --repeats 2`, code `a23eb22`). Ledger: `grow-v-crux.jsonl`.

| gen | challenger | confirm champion → challenger | paired | verdict |
|---|---|---|---|---|
| 0 | `route:content → kimi-hot` | 0.767 → 0.784 (+1.1 cases) | 2 w / 0 l | promoted |
| 1 | `meshflow:content` | 0.779 → 0.767 | 1 / 2 | search-not-better |
| 2 | `meshflow:integration` | 0.764 → 0.788 (**+1.57**) | **4 / 1** | search-not-better |
| 3 | `ensemble:content` | 0.761 → 0.787 (**+1.7**) | **4 / 0**, p = 0.0625 | search-not-better |
| 4 | `route:content → kimi-cold` | 0.759 → 0.767 | 2 / 1 | search-not-better |

Sealed: seed 0.748, champion 0.747 — **not separable** (the seventh completed run in a row on this box).
Re-read with the run's own confirm numbers in hand: the gate certified +1.1 of 65 confirm cases
at gen 0, and the champion, re-measured every generation after (0.784 at promotion, then 0.779,
0.764, 0.761, 0.759 against a seed measured twice at 0.767), ended **0.08 cases below the seed**
on confirm. The claim had **evaporated** before sealed was opened, so sealed had nothing to
test. (Run T on this box was underpowered instead: +0.84 of 56 at the end, 0.42 of 28 sealed
cases. Run R was the third kind: +4.0 of 48 at the end, 2.0 sealed cases expected, +0.33
seen — the gains did not transfer.)

Three things this run showed, none of them about the model:

- **Gate ① was a race against a stale maximum.** The champion's search score (0.901) was the
  maximum of one generation's width, measured once at promotion; every later challenger had to
  beat it strictly while its confirm score drifted 0.784 → 0.759. Rows 2 and 3 were discarded
  *after* being measured at 4/1 and 4/0 on confirm. The gate is now a band of one search case
  (`grow` ≥ `87c63d9`); on this ledger both rows would have reached the confirm gate.
- **The `research → tool` lane fixed none of the four crux research cases.** Not because the
  programs were wrong: the model never opened a ```python block on those prompts, at any
  temperature or token budget up to 8192, so the lane fell back to unfinished reasoning. Yet
  `simplify:research → bare` scored lower on search (0.852 vs 0.901) — the lane still pays for
  itself on the rest. Sending the opening fence as the start of the reply
  (`ToolBackend(prefill="```python\n")`, `ead35cd`) makes the same lane solve **3 of 4** crux
  research cases in about 2 s each; `grow` now proposes it as `tool:<class>(model)+prefill`,
  the one-step refinement of a tool lane. Run W measures whether that buys a confirm case.
- **Search-side saturation was removed** and a rounding defect fixed: scores reach the gates
  rounded to four digits while the widths are exact 1/n, so a one-case margin could pass or
  fail on the rounding direction alone (11/15 − 10/15 arrives as 0.0666 against 0.066667).

The champion of this run is not shipped: one promotion inside the band, no sealed separation.

## Run W: the first sealed separation on this box, and what it is worth

Same seed, same 129 cases and the same split as run V (the case ids are identical), the search
gate now a band, the tool lane able to propose `+prefill`. Ledger: `grow-w-prefill.jsonl`.

| gen | challenger | confirm champion → challenger | paired | verdict |
|---|---|---|---|---|
| 0 | `route:content → kimi-hot` | 0.767 → 0.753 (−0.9 cases) | 1 w / 3 l | rejected |
| 1 | `ensemble:research(cold + hot)` | 0.767 → 0.782 (**+1.0**) | 3 / 2, p = 0.5 | **promoted** |
| 2 | `tool:research(cold)` (= the seed again) | 0.790 → 0.767 (−1.5) | 1 / 2 | rejected |
| 3 | `ensemble:integration(cold + hot)` | 0.790 → 0.802 (+0.77) | 2 / 2 | below the 1-case floor |
| 4 | `deepen:research(tool inside ens)` | 0.780 → 0.775 (−0.38) | 1 / 1 | rejected |

Sealed: seed 0.7484, champion 0.8214, **+2.34 cases of 32, band 1.0: improved.** The confirm
side of the same claim, taken the way the loop now takes it (the seed's three measurements, all
0.7669, against the champion's three re-measurements after promotion, 0.790 / 0.790 / 0.780):
**+1.29 of 65 cases.** Both splits agree on the sign; the promotion itself was carried by one
case at 3 wins to 2, p = 0.5, the weakest evidence the gate accepts. This champion ships because
the split that was never used for a decision separates it from the seed, and it ships with
that p-value next to it.

What the run showed about the loop, not the model:

- **One confirm measurement of a stochastic lane cannot hold a one-case floor.** Row 0 is the
  mutation run V promoted at generation 0: `route:content → kimi-hot`, the same 65 confirm cases,
  the same seed. Run V measured it at 0.7838 (+1.1 cases, 2 wins / 0 losses); run W measured it
  at 0.7531 (−0.9 cases, 1 / 3). `kimi-hot` samples at temperature 0.8, and two measurements of
  the same design on the same cases landed two cases apart, one on each side of the champion.
  The seed, whose lanes are all at temperature 0, measured 0.7669 three times in a row. A
  floor of one case is below the spread of a single measurement of a lane that samples; the
  gate needs the lane's own spread before it can read a one-case gain from it.
- **The prescription was never measured.** The seed's tool lanes returned no code on 10 of
  their 108 search + confirm calls; run X, which measures the same seed on the same split with
  the per-class count (`b78fb81`), puts 8 of the 10 on `research` and 2 on `qa`. The `+prefill`
  refinement that treats exactly that was listed once, in generation 1, for `qa`, by the
  rotation's order, and lost the tie to the ensemble on label order. After
  generation 1 the `research` class was no longer a tool lane, so its `+prefill` was no longer
  a one-step mutation and could not be proposed at all. `grow` now reads the diagnosis
  (`cc985ab`), keeps every measured design in the running at zero cost (`07e9543`) and gives
  the prescribed design the first seat and the tie; run X measures whether that buys a case.
  This paragraph was written by hand after replaying the ledger; the ledger now counts what
  became of each prescription itself (`prescriptions` in the result, one line per prescription
  in the recipe), so the next run's recipe carries it without a person reading the rows.
- **The recipe named the wrong code.** The process started at `907a976`. Its seed row says
  `e7e5e63` and the `recipe.md` it wrote says `ffdb5bf`: both were read from `HEAD` at write
  time, while commits landed during the run. The first is harmless (test-only commits, the
  package byte-identical), the second is not (388 lines of `gama/` between them). `config.json`
  here carries `907a976` by hand, with the note. The stamp is taken once per run now (`7d995fa`),
  and it says when the tree differed from the commit.

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
