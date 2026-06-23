"""m=0 vs m>0 on gama's meshflow membrane — head-to-head.

Runs the SAME MeshflowBackend over the SAME cases under two settings:

  m>0  stakes_threshold = 0.7   (membrane on — hard high-stakes cases held)
  m=0  stakes_threshold = inf   (membrane off — always ship best-effort)

The gama package is untouched; m=0 is achieved purely by configuration.

Usage:
    python3 -m experiments.membrane_probe.run --out runs/membrane_probe
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from gama.meshflow import NEEDS_HUMAN, MeshflowBackend
from gama.models import ModelTier

from .cases import build_tiers, checker, default_cases


def _run_one(label: str, stakes_threshold: float) -> dict:
    cases = default_cases()
    tiers = build_tiers(cases)
    backend = MeshflowBackend(tiers, verify=checker, mesh="union",
                              stakes_threshold=stakes_threshold, pass_score=1.0)

    n = len(cases)
    shipped = correct_ship = bad_ship = escalated = 0
    rows = []
    for c in cases:
        art = backend.complete(c.cid, ModelTier.SMALL, verify=checker, stakes=c.stakes)
        is_human = art == NEEDS_HUMAN
        is_correct = checker(art) >= 1.0
        if is_human:
            escalated += 1
        else:
            shipped += 1
            if is_correct:
                correct_ship += 1
            else:
                bad_ship += 1
        rows.append({"case": c.cid, "stakes": c.stakes, "gate": label,
                     "resolved_by": backend.last_resolved_by,
                     "human_gate": backend.last_human_gate,
                     "shipped": not is_human, "correct": is_correct})

    return {
        "gate": label,
        "stakes_threshold": stakes_threshold if math.isfinite(stakes_threshold) else "inf",
        "n_cases": n,
        "ship_rate": round(shipped / n, 4),
        "bad_ship_rate": round(bad_ship / n, 4),       # wrong answers shipped
        "shipped_precision": round(correct_ship / shipped, 4) if shipped else None,
        "escalation_rate": round(escalated / n, 4),
        "rows": rows,
    }


def run_compare(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    m1 = _run_one("m>0", 0.7)
    m0 = _run_one("m=0", math.inf)
    verdict = ("membrane eliminates bad ships at the cost of escalation"
               if (m0["bad_ship_rate"] > 0 and m1["bad_ship_rate"] == 0
                   and m1["escalation_rate"] > 0)
               else "inconclusive")
    comparison = {
        "membrane_axis": "task ship-gate (NEEDS_HUMAN on high-stakes unresolved)",
        "m=0": {k: m0[k] for k in ("stakes_threshold", "ship_rate", "bad_ship_rate",
                                   "shipped_precision", "escalation_rate")},
        "m>0": {k: m1[k] for k in ("stakes_threshold", "ship_rate", "bad_ship_rate",
                                   "shipped_precision", "escalation_rate")},
        "verdict": verdict,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in m1["rows"] + m0["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return comparison


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="m=0 vs m>0 on gama's meshflow membrane")
    ap.add_argument("--out", default="runs/membrane_probe")
    args = ap.parse_args(argv)
    comparison = run_compare(Path(args.out))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
