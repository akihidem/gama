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
    abmcts_from_config,
    build_backend,
    ensemble_from_config,
    gama_from_config,
    load_config,
    meshflow_from_config,
    system_from_config,
    trinity_from_config,
)
from .decorrelation import analyze as mesh_analyze
from .grow import grow, ollama_pool, write_recipe
from .logger import ExecutionLogger
from .market import analyze as market_analyze
from .models import ModelTier

BACKEND_CHOICES = ["null", "echo", "claude-cli", "claude-tui", "codex", "gemini",
                   "ollama", "ssh-openai"]


def _build_backend_map(names: list, config) -> tuple:
    """Build a ``{name: backend}`` map from a comma list, resolving the composite names
    (ensemble/gama/meshflow/trinity/abmcts) from ``config``. Unknown/bad backends are skipped
    (the sweep goes on). Returns ``(backends, unavailable_names)``."""
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
            elif n == "trinity":
                be = trinity_from_config(config)
            elif n == "abmcts":
                be = abmcts_from_config(config)
            elif n == "system":
                be = system_from_config(config)   # a whole stack declared as one nested spec
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
    elif raw.get("trinity"):
        be = trinity_from_config(args.config)   # a trinity (一撃予測ルーティング) config
    elif raw.get("abmcts"):
        be = abmcts_from_config(args.config)    # an abmcts (AB-MCTS 適応分岐探索) config
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


def _positive_int(value: str) -> int:
    """argparse type: reject 0/negative up front rather than degrading into a silent no-op
    (``--width 0`` would 'run' a growth loop that proposes nothing and reports no promotion,
    which reads like a measured result)."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {n})")
    return n


def _nonneg_float(value: str) -> float:
    """argparse type: a negative promotion floor would not loosen the gate, it would remove
    it (max(negative, drift) collapses to drift, and drift can be 0)."""
    f = float(value)
    if f < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0 (got {f})")
    return f


def cmd_grow(args: argparse.Namespace) -> int:
    """Run the self-improvement loop over gama's own config space and print the champion.

    Progress goes to stderr as it happens: a real run spends minutes per candidate on local
    models, and a loop you cannot watch is indistinguishable from a hung one.
    """
    pool: dict = {}
    if args.pool:
        try:
            pool.update(json.loads(Path(args.pool).read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            sys.stderr.write(f"[gama] --pool {args.pool!r} is not readable JSON: {e}\n")
            return 2
    if args.models:
        pool.update(ollama_pool([m.strip() for m in args.models.split(",") if m.strip()],
                                host=args.host))
    if args.smoke and not pool:
        # Free deterministic smoke: echo/null solve nothing, so the honest outcome is
        # "nothing was promoted" — which is exactly the behaviour worth smoke-testing.
        pool = {"echo": {"backend": "echo"}, "null": {"backend": "null"}}
    if not pool:
        sys.stderr.write("[gama] grow needs lanes: --models m1,m2 (ollama) | --pool lanes.json "
                         "| --smoke\n")
        return 2
    for lane, spec in sorted(pool.items()):
        try:                      # 入力 JSON の壊れた spec は、走らせる前にレーン名付きで断る
            build_backend(spec)
        except Exception as e:
            sys.stderr.write(f"[gama] lane {lane!r} is not a usable backend spec: "
                             f"{type(e).__name__}: {e}\n")
            return 2
    try:
        ratio = tuple(int(x) for x in args.ratio.split(":"))
        if len(ratio) != 3 or any(r < 0 for r in ratio) or sum(ratio) <= 0:
            raise ValueError
    except ValueError:
        sys.stderr.write(f"[gama] --ratio must be three non-negative ints summing > 0, like "
                         f"2:1:1 (got {args.ratio!r})\n")
        return 2
    sys.stderr.write("[gama] NOTE: code/tool cases EXECUTE model-generated Python (opt-in, "
                     "like a sandbox). Only grow on trusted backends.\n")

    def on_event(row: dict) -> None:
        ev = row.get("event")
        if ev == "seed":
            sizes = {k: len(v) for k, v in row["splits"].items()}
            sys.stderr.write(f"[gama] splits {sizes} | seed search={row['search']['score']} "
                             f"confirm={row['confirm']['score']}\n")
            if row.get("margin_floor_coarse"):
                sys.stderr.write(
                    f"[gama] WARNING: only {len(row['splits']['confirm'])} confirm cases, so "
                    f"one case = {row['margin_floor']} — a real improvement worth less than "
                    "that cannot be certified here and will be refused. Widen --suites if you "
                    "want this run to be able to promote small gains.\n")
            if row.get("classes_unconfirmable"):
                sys.stderr.write(
                    f"[gama] NOTE: no confirm cases for {row['classes_unconfirmable']} — those "
                    "classes are left alone (a win there could never be confirmed). Widen with "
                    "--suites default,hard,brutal.\n")
        elif ev == "candidate":
            sys.stderr.write(f"[gama]  gen{row['gen']} {row['label']:<34} "
                             f"search={row['search']['score']}\n")
        elif ev == "generation":
            sys.stderr.write(
                f"[gama] gen{row['gen']} challenger={row['challenger']} "
                f"search {row['champion_search']}->{row['challenger_search']} "
                f"confirm {row['champion_confirm']}->{row['challenger_confirm']} "
                f"(delta={row['delta']}) -> {row['reason']}\n")
        elif ev == "stop":
            sys.stderr.write(f"[gama] stop: {row['reason']}\n")

    try:
        result = grow(pool, suites=[s.strip() for s in args.suites.split(",") if s.strip()],
                      ratio=ratio, generations=args.generations, width=args.width,
                      repeats=args.repeats, tier=ModelTier(args.tier),
                      min_margin=args.min_margin, patience=args.patience, ledger_path=args.out,
                      ensemble_strategy=args.ensemble_strategy, on_event=on_event)
    except ValueError as e:      # too few cases to split honestly / bad lane names / bad suite
        sys.stderr.write(f"[gama] cannot grow: {e}\n")
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "history"},
                     ensure_ascii=False, indent=2))
    sealed = result["sealed"]
    if sealed:
        sys.stderr.write(
            f"[gama] sealed (never used for a decision): seed={sealed['seed']['score']} "
            f"-> champion={sealed['champion']['score']} ({result['promotions']} promotions)\n")
    else:
        sys.stderr.write("[gama] WARNING: case pool too small for a sealed split — the reported "
                         "scores all fed decisions (optimistic). Widen --suites.\n")
    if args.write_recipe:
        d = write_recipe(result, args.write_recipe, hardware=args.hardware)
        sys.stderr.write(f"[gama] recipe -> {d}/config.json + {d}/recipe.md\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gama", description="combine local LLMs: route, ensemble, tool, benchmark")
    p.add_argument("--version", action="version", version="gama 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("bench", help="benchmark backends per task-class; propose a routing_table")
    pb.add_argument("--backends", default="echo",
                    help="comma list, e.g. ollama,ssh-openai,gama,ensemble,meshflow,abmcts,"
                         "system ('system' = the nested spec under a config's \"system\" key, "
                         "e.g. a grown recipe). 'echo' = free smoke")
    pb.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pb.add_argument("--suite", default="default", choices=["default", "hard", "brutal", "wide", "graded"],
                    help="case suite: default (5 classes, may hit a ceiling) | hard | "
                         "brutal (discriminating, break the ceiling effect) | wide (40 cases, "
                         "8 per class — breadth for splitting, same band as hard) | graded "
                         "(20 cases scored as a FRACTION of independently checked "
                         "requirements, so a score can move by less than a whole case)")
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
                    help="a gama routing config; an 'ensemble' / 'meshflow' / 'trinity' / "
                         "'abmcts' config; or {'system': <spec>}")
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
    pm.add_argument("--suite", default="hard", choices=["default", "hard", "brutal", "wide", "graded"],
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
    pmesh.add_argument("--suite", default="hard", choices=["default", "hard", "brutal", "wide", "graded"],
                       help="case suite (default: hard — discriminating, so members can differ)")
    pmesh.add_argument("--pass-score", type=float, default=1.0,
                       help="a case score >= this counts as solved (its external verifier passed)")
    pmesh.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pmesh.add_argument("--repeats", type=int, default=1)
    pmesh.add_argument("--config", default=None,
                       help="per-backend kwargs + composites (ensemble/gama/meshflow)")
    pmesh.set_defaults(func=cmd_mesh)

    pg = sub.add_parser(
        "grow", help="self-improvement loop: mutate the config, measure, keep only "
                     "held-out-confirmed wins")
    pg.add_argument("--models", default=None,
                    help="comma list of ollama models = the lane pool (e.g. "
                         "llama3.2:3b,qwen2.5:7b,gemma4:latest)")
    pg.add_argument("--host", default="http://localhost:11434", help="ollama host for --models")
    pg.add_argument("--pool", default=None,
                    help="JSON file mapping lane name -> backend spec (any backend, not just "
                         "ollama)")
    pg.add_argument("--smoke", action="store_true",
                    help="free deterministic smoke with echo/null lanes (promotes nothing)")
    pg.add_argument("--suites", default="wide,hard,brutal",
                    help="comma list of case suites to pool and split. The default pools the "
                         "discriminating ones (56 cases); add 'default' for 10 easier cases, "
                         "which widens coverage but compresses the differences you are "
                         "searching for")
    pg.add_argument("--ratio", default="2:1:1",
                    help="search:confirm:sealed case ratio (sealed is never used for a decision)")
    pg.add_argument("--generations", type=_positive_int, default=3)
    pg.add_argument("--width", type=_positive_int, default=6,
                    help="candidates proposed per generation")
    pg.add_argument("--repeats", type=_positive_int, default=1,
                    help="bench repeats per case (raise to average out sampling noise)")
    pg.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pg.add_argument("--min-margin", type=_nonneg_float, default=None,
                    help="floor on the confirm-split margin required to promote. Default: one "
                         "confirm case (1/n), so 'smaller than one case' never promotes at any "
                         "suite size; the champion's own measured drift raises it further when "
                         "the models are noisier than that")
    pg.add_argument("--patience", type=_positive_int, default=2,
                    help="stop after this many generations with no promotion")
    pg.add_argument("--ensemble-strategy", default="synthesize",
                    choices=["majority", "first", "synthesize"],
                    help="aggregation for proposed ensemble lanes. Default synthesize: the "
                         "other member writes the final answer from both drafts (one extra "
                         "call per case). majority compares replies verbatim, so on free-text "
                         "answers it ties and falls back to the first member")
    pg.add_argument("--out", default=None, help="write the JSONL grow ledger")
    pg.add_argument("--write-recipe", default=None,
                    help="write the champion to this recipe directory (config.json + recipe.md)")
    pg.add_argument("--hardware", default="(fill in: box, RAM, GPU)",
                    help="hardware line for the emitted recipe.md")
    pg.set_defaults(func=cmd_grow)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
