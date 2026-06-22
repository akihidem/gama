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
from .benchmark import propose_routing_table, run_bench
from .config import (
    build_backend,
    ensemble_from_config,
    gama_from_config,
    load_config,
    meshflow_from_config,
)
from .logger import ExecutionLogger
from .models import ModelTier

BACKEND_CHOICES = ["null", "echo", "claude-cli", "claude-tui", "codex", "gemini",
                   "ollama", "ssh-openai"]


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    backends: dict = {}
    unavailable: list = []
    for n in names:
        try:
            if n == "ensemble":
                be = ensemble_from_config(args.config)
            elif n == "gama":
                be = gama_from_config(args.config)
            elif n == "meshflow":
                be = meshflow_from_config(args.config)
            else:
                be = get_backend(n, **cfg["backends"].get(n, {}))
        except Exception as e:  # unknown name / bad kwargs — skip, don't abort the sweep
            sys.stderr.write(f"[gama] skip backend {n!r}: {e}\n")
            continue
        backends[n] = be
        if not getattr(be, "available", False):
            unavailable.append(n)
    if not backends:
        sys.stderr.write("[gama] no usable backends to benchmark\n")
        return 2
    if unavailable:
        sys.stderr.write(f"[gama] WARNING: unavailable backends will score 0: {unavailable}\n")
    sys.stderr.write("[gama] NOTE: code cases EXECUTE model-generated Python (opt-in, "
                     "like a sandbox). Only run on trusted backends.\n")
    logger = ExecutionLogger(args.out) if args.out else None
    records = run_bench(backends, tier=ModelTier(args.tier), repeats=args.repeats,
                        limit_per_class=args.limit_per_class,
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
