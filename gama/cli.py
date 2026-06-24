"""gama CLI — bench / run / recipes.

    gama bench --backends ollama,ssh-openai,gama,ensemble --propose table.json
    gama run "compute 47*53+89*17" --config recipe/config.json --task-type qa
    gama recipes [name]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends import get_backend
from .benchmark import SUITES, propose_routing_table, run_bench
from .config import (
    build_backend,
    ensemble_from_config,
    gama_from_config,
    load_config,
    meshflow_from_config,
)
from .decorrelation import analyze as mesh_analyze
from .logger import ExecutionLogger
from .market import analyze as market_analyze
from .models import ModelTier

BACKEND_CHOICES = ["null", "echo", "claude-cli", "claude-tui", "codex", "gemini",
                   "ollama", "ssh-openai"]


def _build_backend_map(names: list, config) -> tuple:
    """Build a ``{name: backend}`` map from a comma list, resolving the composite names
    (ensemble/gama/meshflow) from ``config``. Unknown/bad backends are skipped (the sweep
    goes on). Returns ``(backends, unavailable_names)``."""
    cfg = load_config(config)
    backends: dict = {}
    unavailable: list = []
    for n in names:
        try:
            if n == "ensemble":
                be = ensemble_from_config(config)
            elif n == "gama":
                be = gama_from_config(config)
            elif n == "meshflow":
                be = meshflow_from_config(config)
            else:
                be = get_backend(n, **cfg["backends"].get(n, {}))
        except Exception as e:  # unknown name / bad kwargs — skip, don't abort the sweep
            sys.stderr.write(f"[gama] skip backend {n!r}: {e}\n")
            continue
        backends[n] = be
        if not getattr(be, "available", False):
            unavailable.append(n)
    return backends, unavailable


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    backends, unavailable = _build_backend_map(names, args.config)
    if not backends:
        sys.stderr.write("[gama] no usable backends to benchmark\n")
        return 2
    if unavailable:
        sys.stderr.write(f"[gama] WARNING: unavailable backends will score 0: {unavailable}\n")
    sys.stderr.write("[gama] NOTE: code cases EXECUTE model-generated Python (opt-in, "
                     "like a sandbox). Only run on trusted backends.\n")
    logger = ExecutionLogger(args.out) if args.out else None
    records = run_bench(backends, suite=SUITES[args.suite], tier=ModelTier(args.tier),
                        repeats=args.repeats, limit_per_class=args.limit_per_class,
                        unit_cost=cfg.get("unit_cost") or None,
                        logger=logger, run_id=args.run_id or "bench")
    proposal = propose_routing_table(records)
    print(json.dumps(proposal, ensure_ascii=False, indent=2))
    if args.out:
        sys.stderr.write(f"[gama] bench ledger -> {args.out}\n")
    if args.propose:
        Path(args.propose).write_text(
            json.dumps({"routing_table": proposal["routing_table"]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        sys.stderr.write(f"[gama] routing_table proposal -> {args.propose}\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if "system" in raw:
        be = build_backend(raw["system"])      # {"system": <backend spec>}
    elif raw.get("ensemble"):
        be = ensemble_from_config(args.config)  # an ensemble config
    elif raw.get("meshflow"):
        be = meshflow_from_config(args.config)  # a meshflow (段階委譲) config
    else:
        be = gama_from_config(args.config)      # a gama routing config
    out = be.complete(args.prompt, ModelTier(args.tier), task_type=args.task_type)
    print(out)
    return 0


def cmd_recipes(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    if not root.exists():
        sys.stderr.write(f"[gama] no recipes directory at {root}\n")
        return 1
    if args.name:
        cfg = root / args.name / "config.json"
        if cfg.exists():
            print(cfg.read_text(encoding="utf-8"))
            return 0
        sys.stderr.write(f"[gama] recipe {args.name!r} not found\n")
        return 1
    for p in sorted(d for d in root.iterdir() if d.is_dir() and (d / "config.json").exists()):
        rm = p / "recipe.md"
        desc = rm.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip() if rm.exists() else ""
        print(f"{p.name:<26} {desc}")
    return 0


def cmd_market(args: argparse.Namespace) -> int:
    """Run a bench over the given tiers (cheap->expensive) and print the market verdict:
    does verification-routed escalation Pareto-dominate the flat-strong model? (p > w/s)."""
    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    if len(names) < 2:
        sys.stderr.write("[gama] market needs >= 2 tiers cheap->expensive, "
                         "e.g. --backends weak,strong\n")
        return 2
    backends, unavailable = _build_backend_map(names, args.config)
    tier_order = [n for n in names if n in backends]    # keep cheap->expensive order
    if len(tier_order) < 2:
        sys.stderr.write("[gama] need >= 2 usable tiers for a market\n")
        return 2
    if unavailable:
        sys.stderr.write(f"[gama] WARNING: unavailable backends score 0: {unavailable}\n")
    sys.stderr.write("[gama] NOTE: code cases EXECUTE model-generated Python (opt-in, "
                     "like a sandbox). Only run on trusted backends.\n")
    costs = [c.strip() for c in args.costs.split(",") if c.strip()] if args.costs else None
    records = run_bench(backends, suite=SUITES[args.suite], tier=ModelTier(args.tier),
                        repeats=args.repeats, run_id="market")
    try:
        result = market_analyze(records, tier_order, costs=costs, pass_score=args.pass_score)
    except ValueError as e:
        sys.stderr.write(f"[gama] {e}\n")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    m, strong, a = result["market"], result["flat_strong"], result["analytic"]
    sys.stderr.write(
        f"[gama] market cost={m['market_cost']} pass_rate={m['pass_rate']}  vs  "
        f"flat-strong({strong['backend']}) cost={strong['cost']} pass_rate={strong['pass_rate']}  "
        f"-> Pareto-dominates={result['market_dominates_flat_strong']} "
        f"(analytic p_weak={a['p_weak']} {'>' if a['dominates_2tier'] else '<='} p*={a['p_star']})\n")
    return 0


def cmd_mesh(args: argparse.Namespace) -> int:
    """Run a bench over the given ensemble members and print the decorrelation verdict: does
    combining them (union under external verification) beat the best single member? (rho < 1)."""
    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    if len(names) < 2:
        sys.stderr.write("[gama] mesh needs >= 2 members, e.g. --backends a,b,c\n")
        return 2
    backends, unavailable = _build_backend_map(names, args.config)
    members = [n for n in names if n in backends]
    if len(members) < 2:
        sys.stderr.write("[gama] need >= 2 usable members for a mesh\n")
        return 2
    if unavailable:
        sys.stderr.write(f"[gama] WARNING: unavailable backends score 0: {unavailable}\n")
    sys.stderr.write("[gama] NOTE: code cases EXECUTE model-generated Python (opt-in, "
                     "like a sandbox). Only run on trusted backends.\n")
    records = run_bench(backends, suite=SUITES[args.suite], tier=ModelTier(args.tier),
                        repeats=args.repeats, run_id="mesh")
    try:
        result = mesh_analyze(records, members, pass_score=args.pass_score)
    except ValueError as e:
        sys.stderr.write(f"[gama] {e}\n")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.stderr.write(
        f"[gama] union={result['union']} vs best-single({result['best_member']})="
        f"{result['best_single']}  gain={result['mesh_gain']}  failure_rho={result['failure_rho']}  "
        f"-> ensembling ignites={result['ignites']}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gama", description="combine local LLMs: route, ensemble, tool, benchmark")
    p.add_argument("--version", action="version", version="gama 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("bench", help="benchmark backends per task-class; propose a routing_table")
    pb.add_argument("--backends", default="echo",
                    help="comma list, e.g. ollama,ssh-openai,gama,ensemble,meshflow. "
                         "'echo' = free smoke")
    pb.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pb.add_argument("--suite", default="default", choices=["default", "hard", "brutal"],
                    help="case suite: default (5 classes, may hit a ceiling) | hard | "
                         "brutal (discriminating suites that break the ceiling effect)")
    pb.add_argument("--repeats", type=int, default=1)
    pb.add_argument("--limit-per-class", type=int, default=None)
    pb.add_argument("--out", default=None, help="write a JSONL bench ledger")
    pb.add_argument("--propose", default=None, help="write the proposed routing_table JSON")
    pb.add_argument("--run-id", default=None)
    pb.add_argument("--config", default=None,
                    help="config providing per-backend kwargs + ensemble + unit_cost")
    pb.set_defaults(func=cmd_bench)

    pr = sub.add_parser("run", help="run a combined system on a prompt")
    pr.add_argument("prompt")
    pr.add_argument("--config", required=True,
                    help="a gama routing config, an 'ensemble' config, or {'system': <spec>}")
    pr.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pr.add_argument("--task-type", default="generic",
                    help="task_type for routing (e.g. qa, code_implementation, research)")
    pr.set_defaults(func=cmd_run)

    prc = sub.add_parser("recipes", help="list community recipes, or show one's config")
    prc.add_argument("name", nargs="?", help="recipe name to show")
    prc.add_argument("--dir", default="recipes", help="recipes directory (default: ./recipes)")
    prc.set_defaults(func=cmd_recipes)

    pm = sub.add_parser(
        "market", help="is combining cheaper than scaling? the p>w/s verdict from your bench")
    pm.add_argument("--backends", default="echo,echo",
                    help="comma list, CHEAP->EXPENSIVE tiers (e.g. ollama,ssh-openai); the "
                         "last is the flat-strong baseline. 'echo,echo' = free smoke")
    pm.add_argument("--suite", default="hard", choices=["default", "hard", "brutal"],
                    help="case suite (default: hard — discriminating, so the market has gaps "
                         "to exploit)")
    pm.add_argument("--costs", default=None,
                    help="comma per-tier cost weights cheap->expensive (e.g. 1,3,10); "
                         "default 1,2,3,...")
    pm.add_argument("--pass-score", type=float, default=1.0,
                    help="a case score >= this counts as solved (its external verifier passed)")
    pm.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pm.add_argument("--repeats", type=int, default=1)
    pm.add_argument("--config", default=None,
                    help="per-backend kwargs + composites (ensemble/gama/meshflow)")
    pm.set_defaults(func=cmd_market)

    pmesh = sub.add_parser(
        "mesh", help="does ensembling help? the decorrelation (rho<1) verdict from your bench")
    pmesh.add_argument("--backends", default="echo,null",
                       help="comma list of ensemble members (>=2), e.g. gemma,qwen,llama. "
                            "'echo,null' = free smoke")
    pmesh.add_argument("--suite", default="hard", choices=["default", "hard", "brutal"],
                       help="case suite (default: hard — discriminating, so members can differ)")
    pmesh.add_argument("--pass-score", type=float, default=1.0,
                       help="a case score >= this counts as solved (its external verifier passed)")
    pmesh.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pmesh.add_argument("--repeats", type=int, default=1)
    pmesh.add_argument("--config", default=None,
                       help="per-backend kwargs + composites (ensemble/gama/meshflow)")
    pmesh.set_defaults(func=cmd_mesh)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
