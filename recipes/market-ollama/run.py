#!/usr/bin/env python3
"""market-ollama — measure the verification-routing market on your OWN local ollama models.

Builds a cheap->expensive ladder of local ollama models, runs gama's discriminating `hard`
suite over each (deterministic checkers, no LLM judge), then asks `gama market` the question
that matters: does verification-routed escalation Pareto-dominate the strongest single model?
(the p > w/s verdict — soshiki-genron §5, productized as `gama.market.analyze`).

Honest by construction: the numbers are whatever YOUR models score. A small capability gap
may not dominate — that's a real result, not a failure (see the 3 regimes in the README).

run:  python3 recipes/market-ollama/run.py            # default 3-model ladder, hard suite
      python3 recipes/market-ollama/run.py --limit 1  # 1 case/class (fast smoke)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gama.backends import OllamaBackend
from gama.benchmark import HARD_SUITE, run_bench, summarize
from gama.market import analyze
from gama.models import ModelTier

# cheap -> expensive. (label, ollama model tag, cost weight = price/compute proxy).
# Default = a 2-tier weak->strong ladder (fast to run); add more tiers freely.
LADDER = [
    ("gemma-e2b",  "gemma4:e2b",   1.0),   # ~2B  — the cheap tier
    ("qwen-7b",    "qwen2.5:7b",   3.0),   # ~7B  — the flat-strong baseline
]
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def be(model, timeout=90):
    # one model for every tier slot, so bench at any --tier hits the same model.
    # timeout bounds a stalled ollama call (a hung request scores 0, the sweep goes on).
    return OllamaBackend(host=HOST, model_by_tier={t: model for t in ModelTier}, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cases per class (default: all)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()

    backends = {label: be(model) for label, model, _ in LADDER}
    names = [label for label, _, _ in LADDER]
    costs = [c for _, _, c in LADDER]

    print(f"# market-ollama  host={HOST}  ladder={list(zip(names, costs))}")
    records = run_bench(backends, suite=HARD_SUITE, tier=ModelTier.LARGE,
                        limit_per_class=args.limit, run_id="market-ollama")

    # --- discrimination table: does the hard suite separate the models? ---
    summ = summarize(records)
    classes = sorted(summ["by_class"])
    print("\n## hard-suite scores (deterministic checkers)")
    print(f"{'class':<20}" + "".join(f"{n:>12}" for n in names))
    for c in classes:
        row = f"{c:<20}"
        for n in names:
            v = summ["by_class"].get(c, {}).get(n)
            row += (f"{v['score']:>12.2f}" if v else f"{'-':>12}")
        print(row)
    print(f"{'OVERALL':<20}" + "".join(f"{summ['overall'][n]['score']:>12.2f}" for n in names))

    # --- the market verdict: combine cheaper than scaling? ---
    result = analyze(records, names, costs=costs, pass_score=1.0)
    m, strong, a = result["market"], result["flat_strong"], result["analytic"]
    print("\n## verification-routing market vs flat-strong")
    print(f"market        : cost={m['market_cost']:<7} pass_rate={m['pass_rate']}")
    print(f"flat-strong   : {strong['backend']} cost={strong['cost']:<7} pass_rate={strong['pass_rate']}")
    print(f"Pareto-dominates flat-strong : {result['market_dominates_flat_strong']}")
    print(f"analytic (2-tier weak->strong): p_weak={a['p_weak']} {'>' if a['dominates_2tier'] else '<='} "
          f"p*={a['p_star']}  -> dominates={a['dominates_2tier']}")
    print(f"\n  {result['thesis']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": records, "analysis": result}, f, ensure_ascii=False, indent=2)
    print(f"\nraw -> {args.out}")


if __name__ == "__main__":
    main()
