# Membrane probe — m=0 vs m>0 on gama's meshflow ship-gate

> The gama-native counterpart to tehai's upper-loop probe. In tehai the
> governance membrane sits on the *self-improvement loop*; in gama it sits on the
> **task ship-gate**: `MeshflowBackend` returns `<<NEEDS_HUMAN>>` on a high-stakes
> unresolved artifact instead of silently shipping a failing draft (soshiki-genron
> PAPER §6.5, organizational form ③ — "薄い人間統治膜").

## m=0 without touching the package

The membrane fires only when `stakes >= stakes_threshold`. So:

| gate | config | behaviour |
|---|---|---|
| **m>0** | `stakes_threshold = 0.7` | membrane on — hard high-stakes cases held for a human |
| **m=0** | `stakes_threshold = inf` | membrane off — always ships best-effort |

Same `MeshflowBackend`, same cases, **gama package unchanged** (public API only).

## Result

```bash
cd ~/Projects/gama
python3 -m experiments.membrane_probe.run
```

Over a case mix (solvable low-stakes + hard high-stakes):

- **m=0**: `ship_rate=1.0` but `bad_ship_rate>0` — it ships verified-wrong answers
  on the hard cases. `shipped_precision < 1`.
- **m>0**: `bad_ship_rate=0`, `shipped_precision=1.0` — wrong answers are held
  (`escalation_rate>0`) instead of shipped.
- **verdict**: `"membrane eliminates bad ships at the cost of escalation"`.

Solvable cases behave identically under both gates — the membrane only changes the
hard, high-stakes corner.

## The data point

Third governance-membrane measurement, all pointing the same way:

- **tehai** `experiments/auto_upper_loop` + `safe_upper_loop` — membrane on the
  *upper loop*; m=0 degrades true quality (Goodhart).
- **soshiki-genron** — *theory*: interior optimum `m* > 0`.
- **gama** (here) — membrane on the *ship-gate*; m=0 ships wrong answers.

Removing the human membrane is cheap until the corner where it isn't — which is
exactly where it earns its keep.

```bash
python3 -m unittest experiments.membrane_probe.test_membrane_probe
```
