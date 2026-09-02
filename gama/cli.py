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
from .benchmark import SUITE_DOCS, SUITES, propose_routing_table, run_bench
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
from .grow import (MeasurementFailure, grow, ollama_pool, trace_diff, trace_lines,
                   trace_rows, trace_summary, write_recipe)
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


def cmd_calib(args: argparse.Namespace) -> int:
    """Calibrate one verifier against one backend on a suite (v3a/T8): confusion counts,
    I(verdict; correct) in bits, and the achievable selection ceiling. With --observed K,
    run the budget check: a pipeline reporting more V-driven correct picks than the ceiling
    is a measurement bug (DPI), not a discovery."""
    from .calib import budget_check, calibrate_verifier
    from .meshflow import resolve_verifier

    backends, unavailable = _build_backend_map([args.backend], args.config)
    be = backends.get(args.backend)
    if be is None:
        sys.stderr.write(f"[gama] backend {args.backend!r} not usable\n")
        return 2
    if unavailable:
        sys.stderr.write(f"[gama] WARNING: {args.backend} reports unavailable; scores will be 0\n")
    try:
        verify = resolve_verifier(args.verify)
    except (TypeError, ValueError) as e:
        sys.stderr.write(f"[gama] {e}\n")
        return 2
    if verify is None:
        sys.stderr.write("[gama] calib needs a verifier (name or callable); got none\n")
        return 2
    if not (0.0 < args.pass_score <= 1.0):
        # verify のスコアは [0,1] に clamp される（meshflow._normalize_score）。範囲外の
        # 閾値は「常に pass / 決して pass しない」の退化で、較正としては入力ミス。
        sys.stderr.write(f"[gama] --pass-score must be in (0, 1]; got {args.pass_score}\n")
        return 2
    sys.stderr.write("[gama] NOTE: code cases EXECUTE model-generated Python "
                     "(sandboxed subprocess with timeout). Only run on trusted backends.\n")
    res = calibrate_verifier(be, verify, SUITES[args.suite], ModelTier(args.tier),
                             pass_score=args.pass_score)
    if args.observed is not None:
        # --observed-n が無い呼び出しでは same-suite 照合が自己充足になる（calib 自身の
        # n を渡すので必ず一致する）。その場合は「同一 suite は呼び手の申告」だと結果に
        # 焼き込む。申告でなく検算にしたければ --observed-n で観測側の件数を渡す。
        obs_n = args.observed_n if args.observed_n is not None else res["confusion"]["n"]
        try:
            res["budget"] = budget_check(args.observed, obs_n, res["confusion"])
        except ValueError as e:
            sys.stderr.write(f"[gama] {e}\n")
            return 2
        if args.observed_n is None:
            res["budget"]["same_suite"] = "claimed (pass --observed-n to verify the count)"
    print(json.dumps(res, ensure_ascii=False, indent=2))
    cm, sel = res["confusion"], res["selection"]
    line = (f"[gama] verify calib({args.verify}@{args.backend}): I={res['i_bits']} bits  "
            f"confusion tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}  "
            f"selection ceiling {sel['ceiling_k']}/{sel['n']} "
            f"(no-verdict base {sel['base_k']}, headroom {sel['headroom_k']})")
    if "budget" in res:
        line += f"  -> {res['budget']['verdict']}"
    sys.stderr.write(line + "\n")
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
    """Run a bench over the given ensemble members and print the ignition verdict: does combining
    them (union under external verification) beat the best single member? Certified from the
    co-failure rate β (every member wrong on the same case) and its exact interval — the pairwise
    rho is printed last because it cannot see β once there are 3+ members."""
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
        result = mesh_analyze(records, members, pass_score=args.pass_score, by_class=True)
    except ValueError as e:
        sys.stderr.write(f"[gama] {e}\n")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    lo, hi = result["beta_interval"]
    glo, ghi = result["gain_bounds"]
    # β first: it is the quantity that caps every answer-selecting policy. The point gain alone
    # once reported a one-case fluke (+0.042) as emergence; the interval is what stops that.
    sys.stderr.write(
        f"[gama] co-failure beta={result['cofailure_k']}/{result['n_cases']}={result['cofailure_beta']} "
        f"[{lo}, {hi}] -> ceiling 1-beta={result['ceiling']} vs best-single({result['best_member']})="
        f"{result['best_single']}  gain={result['mesh_gain']} ({result['gain_cases']} cases; bounds [{glo}, {ghi}])  "
        f"-> verdict={result['verdict']} (pairwise rho={result['failure_rho']}, secondary)\n")
    blind = {c: v for c, v in result.get("classes", {}).items() if v["blind_spot"]}
    if blind:
        # existence is certified (k >= 1); MAGNITUDE is the interval — read it before acting.
        # Those k cases defeat every member, so where the mass is large, more members won't
        # help: add a different KIND of capability (tool/lane) or accept the ceiling.
        sys.stderr.write("[gama] certified co-failure by class (k>=1): "
                         + "  ".join(f"{c}={v['cofailure_k']}/{v['n_cases']} "
                                     f"[{v['beta_interval'][0]}, {v['beta_interval'][1]}] "
                                     f"ceiling<={v['ceiling_certified']}"
                                     for c, v in blind.items()) + "\n")
    neff = result.get("effective_votes")
    if neff is not None:
        # advisory only: Kish n_eff はペア φ 由来で β を識別しない(2605.29800 の診断)。
        # 「m 人置いて実効何票か」の直感を運ぶだけで、verdict はこれを読まない。
        worst = max(result.get("pairwise_cofailure", []),
                    key=lambda d: d["cofailure_k"], default=None)
        line = (f"[gama] advisory: effective votes n_eff={neff}/{len(result.get('members', []))} "
                f"(Kish, pairwise phi; cannot certify beta)")
        if worst and worst["cofailure_k"] > 0:
            line += (f"  worst pair={worst['pair'][0]}+{worst['pair'][1]} "
                     f"co-fail {worst['cofailure_k']}/{worst['n_cases']}")
        sys.stderr.write(line + "\n")
    if result.get("unclassified_cases"):
        sys.stderr.write(f"[gama] note: {result['unclassified_cases']} case(s) carry no "
                         "task_type and are outside the class breakdown\n")
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
        # Free deterministic smoke: echo solves nothing, so the honest outcome is "nothing was
        # promoted" — exactly the behaviour worth smoke-testing. Two echo lanes rather than
        # echo+null, because NullBackend raises by design: that pool was a measurement where
        # half the calls failed, and it only ever "passed" because the loop ignored errors.
        pool = {"echo-a": {"backend": "echo"}, "echo-b": {"backend": "echo"}}
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
    seed_spec = None
    if args.start_from:
        try:                       # a grown recipe, or any {"system": <spec>} config
            raw = json.loads(Path(args.start_from).read_text(encoding="utf-8"))
            seed_spec = raw["system"] if isinstance(raw.get("system"), dict) else raw
        except (OSError, ValueError, KeyError, TypeError) as e:
            sys.stderr.write(f"[gama] --start-from {args.start_from!r} is not a usable config: "
                             f"{e}\n")
            return 2
    sys.stderr.write("[gama] NOTE: code/tool cases EXECUTE model-generated Python (opt-in, "
                     "like a sandbox). Only grow on trusted backends.\n")
    # 台帳はこの走行の唯一の証拠で、--resume の入口でもある。それを再起動で消える場所に
    # 置くのは、数時間の測定を「reboot 一回で全損」にする賭け(実際に /tmp のスクラッチに
    # 置いた 14 走ぶんの台帳が、WSL の再起動で丸ごと消えた)。走る前に言う。
    if not args.out:
        sys.stderr.write("[gama] WARNING: no --out, so this run leaves no ledger: nothing to "
                         "resume from and no record behind its numbers.\n")
    elif str(Path(args.out).resolve()).startswith(("/tmp/", "/var/tmp/")):
        sys.stderr.write(f"[gama] WARNING: ledger at {args.out} is under /tmp and will not "
                         "survive a reboot — the evidence for this run dies with it.\n")

    seen: dict = {}

    def on_event(row: dict) -> None:
        ev = row.get("event")
        if ev in ("seed", "resumed"):
            # confirm の問数は seed/resumed 行にしか無い(同じ emit で出るので両方持っている)。
            # 世代行は持たないので、ここで控えておかないと「+1.25 問 / 全何問中」が出せない
            # (出せないまま書式だけ残ると "cases of " という尻切れになる)。
            # 台帳は古い版のものも読まされうるので、無ければ黙って諦める(表示のために落ちない)。
            conf = ((row.get("splits") or {}).get("confirm")) or []
            if conf:
                seen["n_confirm"] = len(conf)
            seen["classes"] = row.get("classes") or []
        if ev == "seed":
            sizes = {k: len(v) for k, v in row["splits"].items()}
            sys.stderr.write(f"[gama] splits {sizes} | seed search={row['search']['score']} "
                             f"confirm={row['confirm']['score']}\n")
            # 種の診断: tool レーンがコードを出さなかったクラス。処方(+prefill)はこの順で
            # tool 枠の先頭に出る。走行後に台帳を掘らないと「なぜその順で並んだか」が
            # 見えないので、並ぶ前に言う。
            symptoms: dict = {}
            for _k in ("search", "confirm"):
                for _c, _n in ((row.get(_k) or {}).get("tool_no_code_by_class") or {}).items():
                    symptoms[_c] = symptoms.get(_c, 0) + _n
            if symptoms:
                order = sorted(symptoms, key=lambda c: (-symptoms[c], c))
                sys.stderr.write("[gama] seed tool lanes returned no code on: "
                                 + ", ".join(f"{c} {symptoms[c]}" for c in order)
                                 + " (calls, search+confirm); their +prefill is proposed "
                                   "before the other tool mutations\n")
            # sealed が何を言えるかは封を開ける前に決まっている。1 問の帯を confirm の問数に
            # 直した量より小さくしか種の上に立てなかった最終形は、転移していても NOT SEPARABLE
            # で終わる。run T/V がそれで、走行後に「分からない」と読むのは 3 時間遅い。
            # 量るのは昇格時の認定の合計ではなく最終形の confirm 点(sealed が比べる相手と同じ)。
            if sizes.get("sealed") and sizes.get("confirm"):
                need = sizes["confirm"] / sizes["sealed"]
                sys.stderr.write(
                    f"[gama] the sealed split resolves one case, which is {need:.2f} confirm "
                    f"cases: a champion that ends no more than that above the seed on confirm "
                    "will be NOT SEPARABLE by construction, whatever it is worth.\n")
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
        # tool レーンが「道具として働いていない」ことは、例外にも低得点にも見えない形で
        # 混ざる(コードが取り出せないと素の返答をそのまま返すため)。一度だけ大きく言う。
        for _k in ("search", "confirm"):
            _m = row.get(_k)
            if isinstance(_m, dict) and _m.get("tool_calls") and not _m.get("tool_ran"):
                if not seen.get("tool_warned"):
                    seen["tool_warned"] = True
                    # 原因で言い分ける: コードが出ないのは prompt/モデル側、出力が空なのは
                    # 生成コード側。直し方が正反対なので「動かなかった」で束ねない。
                    nc, eo = _m.get("tool_no_code", 0), _m.get("tool_empty_out", 0)
                    why = (f"{nc} produced no code block at all"
                           if nc >= eo else
                           f"{eo} ran but printed nothing")
                    sys.stderr.write(
                        f"[gama] WARNING: a tool lane was called {_m['tool_calls']} times and "
                        f"NEVER returned a program's output ({why}). Every one of those calls "
                        "fell back to the model's raw reply, so you are not measuring a tool "
                        "lane — you are measuring the bare model with a code-writing prompt in "
                        "front of it, which scores WORSE than the bare model. No code block "
                        "means the model would not follow the format (try another model, or a "
                        "less quantized one); code that prints nothing means the generated "
                        "program is wrong.\n")
        if ev == "candidate":
            sys.stderr.write(f"[gama]  gen{row['gen']} {row['label']:<34} "
                             f"search={row['search']['score']}\n")
        elif ev == "generation":
            gain = row.get("gain_cases")
            # 「並んだ数」と「測った数」は違う(測定済みは width の外で並ぶ)。挑戦者が archive の
            # 踏み石なら、その search の点はこの世代に測ったものではないと読めるようにする。
            # キーの有無で分岐するのは、古い台帳(この 2 つを書く前の走行)を再生しても落ちない
            # ため。
            listed = (f" ({row['new_candidates']} new of {row['candidates']} listed)"
                      if "new_candidates" in row else "")
            origin = (f" [{row['challenger_from']}]" if row.get("challenger_from") else "")
            sys.stderr.write(
                f"[gama] gen{row['gen']} challenger={row.get('challenger', '(none)')}{origin}"
                f"{listed} search {row['champion_search']}->{row.get('challenger_search', '-')} "
                # search で負けた候補の confirm は測らない(測っても門 ① で落ちる)ので、
                # 無い値を "-" で見せるより「測らなかった」と分かる形にする。キーの有無で判定
                # するのは、grow が challenger_confirm を**測った直後にだけ**書く(理由に関わらず
                # 「無い = 測っていない」が成り立つ)から。
                + (f"confirm {row['champion_confirm']}->{row['challenger_confirm']} "
                   if "challenger_confirm" in row
                   else f"confirm {row['champion_confirm']} (challenger not measured) ")
                + (f"({gain:+} of {seen['n_confirm']} cases) " if gain is not None
                                                                    and seen.get("n_confirm")
                   else (f"({gain:+} cases) " if gain is not None else ""))
                + (f"[{row['paired_wins']}w-{row.get('paired_losses', '?')}l "
                   f"p={row.get('paired_p', '?')}] " if row.get("paired_wins") is not None
                   else "")
                + f"(delta={row['delta']}) -> {row['reason']}"
                # 例外で 0 点になった call がある測定で決めた世代は、その場で言う。走行を止める
                # のは 20% からで、その下の失敗は低い点として混ざり、drift 経由で門まで動かす。
                + _raised_note(row) + "\n")
        elif ev == "saturated":
            # gen0 のこの行は**候補を1つも測る前**に出る。走行の最後に「区別できなかった」と
            # 知るのは高い(実走で数時間)。打つ手が無いことが分かった時点で言う。
            room = ", ".join(f"{c} {v:g}" for c, v in sorted(row["classes"].items()))
            sys.stderr.write(
                f"[gama] gen{row['gen']} saturated: {room} (cases of headroom, gate is "
                f"{row['gate_cases']:g}) — no additive mutation there can be promoted, so those "
                "classes are skipped this generation.\n")
            if row["gen"] == 0 and not seen.get("live_classes_warned"):
                seen["live_classes_warned"] = True
                known = set(seen.get("classes") or ())
                # 空集合と比較すると `>=` は常に真になり、全クラス飽和でなくても警告が出る。
                # 「全部」と言えるのは、比べる相手を実際に知っているときだけ。
                if known and set(row["classes"]) >= known:
                    sys.stderr.write(
                        "[gama] ...that is EVERY class this run can mutate. Nothing additive can "
                        "be promoted no matter how many generations you give it: this champion "
                        "already scores too well on these cases for a one-case gain to exist. "
                        "The run will still try to REMOVE structure (that only has to be not "
                        "worse). If you wanted growth, stop now and use harder cases — "
                        "`--suites crux` exists for this.\n")
        elif ev == "stop":
            sys.stderr.write(f"[gama] stop: {row['reason']}\n")

    try:
        result = grow(pool, suites=[s.strip() for s in args.suites.split(",") if s.strip()],
                      ratio=ratio, generations=args.generations, width=args.width,
                      repeats=args.repeats, tier=ModelTier(args.tier),
                      min_margin=args.min_margin, patience=args.patience, ledger_path=args.out,
                      ensemble_strategy=args.ensemble_strategy, seed_spec=seed_spec,
                      max_paired_p=args.max_paired_p,
                      resume_from=args.resume, on_event=on_event, trace_path=args.trace)
    except MeasurementFailure as e:
        sys.stderr.write(f"[gama] STOPPED: {e}\n")
        return 3                 # 分けて返す: 設定ミスでなく実行環境の事故なので、再開が正しい対応
    except ValueError as e:      # too few cases to split honestly / bad lane names / bad suite
        sys.stderr.write(f"[gama] cannot grow: {e}\n")
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "history"},
                     ensure_ascii=False, indent=2))
    sealed = result["sealed"]
    if sealed:
        net = "" if result.get("net_change", True) else " — NET ZERO: the champion ended up " \
                                                        "identical to the seed"
        sys.stderr.write(
            f"[gama] sealed (never used for a decision): seed={sealed['seed']['score']} "
            f"-> champion={sealed['champion']['score']} ({result['promotions']} promotions"
            f"{net})\n")
    else:
        sys.stderr.write("[gama] WARNING: case pool too small for a sealed split — the reported "
                         "scores all fed decisions (optimistic). Widen --suites.\n")
    bound = result.get("bound_by") or {}
    if bound.get("floor", 0) and bound["floor"] >= bound.get("drift", 0):
        n_confirm = len(result["splits"]["confirm"])
        sys.stderr.write(
            f"[gama] the bar was set by RESOLUTION, not noise, in {bound['floor']} of "
            f"{sum(bound.values())} generations: on {n_confirm} confirm cases a change has to "
            "be worth one whole case. Note that adding cases in classes a mutation does not "
            "touch changes nothing — the floor 1/n and a gain S/n scale together — so the "
            "lever is more cases IN THE CLASS being changed, not a bigger pool. More "
            "--repeats does nothing here either.\n")
        # 一般論(「変えているクラスに case を足せ」)は測った数字で言える。どのクラスに
        # どれだけ余地が残っているかは走行が知っているので、推測させない。
        room = result.get("headroom") or {}
        if room:
            dry = sorted(c for c, v in room.items() if v < 1.0)
            listing = ", ".join(f"{c} {v:g}" for c, v in sorted(room.items(), key=lambda kv: kv[1]))
            sys.stderr.write(f"[gama] headroom left, in cases: {listing}.\n")
            if dry:
                sys.stderr.write(
                    f"[gama] {', '.join(dry)} hold less than one case of headroom, so no "
                    "additive mutation there can clear the floor whatever you try. Adding cases "
                    "to those classes only helps if the new cases are ones this champion gets "
                    "WRONG — harder, not more.\n")
    elif bound.get("drift", 0):
        sys.stderr.write(
            f"[gama] the bar was set by NOISE in {bound['drift']} of {sum(bound.values())} "
            "generations: the champion's own re-measurement moved more than one case. Raising "
            "--repeats is the lever here, not more cases.\n")

    # 昇格の**証拠の強さ**を最後に必ず言う。平均差の床だけを通った手は、held-out で
    # しぼむ/反転することが実測で出ている(confirm 比 4〜6 倍、小さい伸びでは符号反転)。
    # 台帳を読まない人にも、どの手が弱い証拠で通ったかがその場で見えるようにする。
    # 閾値は運用者が指定したものを使う。--max-paired-p を渡しているのに警告だけ 0.05 だと、
    # 「弱い」の定義が門と表示で食い違う。指定が無いときだけ 0.05 を目安に使う。
    weak_at = args.max_paired_p if args.max_paired_p is not None else 0.05
    weak = [e for e in (result.get("promotion_evidence") or [])
            if e.get("p") is not None and e["p"] > weak_at]
    if weak:
        detail = ", ".join(f"{e['challenger']} ({e['wins']}w-{e['losses']}l, p={e['p']})"
                           for e in weak)
        sys.stderr.write(
            f"[gama] {len(weak)} of {len(result['promotion_evidence'])} promotions cleared the "
            f"mean floor but NOT a per-case sign test (p>{weak_at}): {detail}. At this case "
            f"count that test "
            "needs a near-sweep, so this is a limit of the evidence, not proof the change is "
            "bad — read the sealed line as the check, and treat these lanes as provisional.\n")

    # 走行の合否は昇格数ではなく、封をした split が言う。run T は sealed が下がっているのに
    # 「昇格 1・成功」として champion を出していた。数字は出ていたが、誰も判定していなかった。
    sv = result.get("sealed_verdict") or {}
    if sv.get("verdict") == "regressed":
        sys.stderr.write(
            f"[gama] HELD-OUT VERDICT: REGRESSED ({sv['delta_cases']:+} cases). {sv['note']}. "
            "The promotions were measured on `confirm`, which is selected against every "
            "generation and therefore reads high; `sealed` is the only split that was not.\n")
    elif sv.get("verdict") == "not-separable":
        if not result.get("net_change"):
            # 何も通っていない走行に「通した手が区別できない」と書くと、直前の NET ZERO 行と
            # 矛盾する。同じ判定でも意味が違うので、言い分けないと読み手が混乱する。
            # 判定に `net_change` を使うのは、これが `spec_hash(champion) != spec_hash(seed)`
            # そのものだから(昇格の**回数**ではなく、最終形が種と同一かを直接見ている)。
            # 足して戻した走行も net_change=False で、その場合も champion は種と同一。
            sys.stderr.write(
                "[gama] HELD-OUT VERDICT: NOT SEPARABLE — trivially, because nothing was "
                "promoted: the champion IS the seed, so there was nothing for the sealed split "
                "to tell apart. The refusals are the result here.\n")
        else:
            # note は「検定力が無かった(封を開ける前から決まっていた)」と「あったが転移しな
            # かった(confirm の伸びは選択の上振れ)」を言い分けている。ここで一般論に丸めない。
            sys.stderr.write(
                f"[gama] HELD-OUT VERDICT: NOT SEPARABLE ({sv['delta_cases']:+} cases, and the "
                f"sealed split resolves {sv.get('band_cases', 1):g}). {sv['note']}. "
                "That is not a failure, but it is not an improvement either — say so when "
                "quoting these numbers.\n")
    elif sv.get("verdict") == "improved":
        sys.stderr.write(f"[gama] HELD-OUT VERDICT: IMPROVED ({sv['delta_cases']:+} cases on "
                         "cases that never fed a decision).\n")

    if result.get("identity_verified") is False:
        sys.stderr.write(
            "[gama] this run resumed from a ledger written before served-model identity was "
            "recorded, so the boundary between the old run and this one is UNVERIFIED: if the "
            "server was restarted onto different weights in between, the generations either "
            "side are not comparable. Later generations are still checked against each other.\n")

    # 切れた返答の数は走行の終わりに 1 行で言う(recipe と同じ文言・同じ関数。二か所で別の
    # 数え方をすると、読み手には別の事実に見える)。台帳の無い走行には trace も無く、何も出ない。
    for line in trace_lines(result):
        sys.stderr.write(f"[gama] {line[2:] if line.startswith('- ') else line}\n")

    if args.write_recipe:
        d = write_recipe(result, args.write_recipe, hardware=args.hardware)
        sys.stderr.write(f"[gama] recipe -> {d}/config.json + {d}/recipe.md\n")
    return 0


def _trace_labels(trace_path: Path) -> dict:
    """spec hash → label。trace は hash しか持たない(label は同じ設計に何度も付く)ので、隣の
    台帳(run.trace.jsonl → run.jsonl)の candidate 行から引く。台帳が無ければ空: hash のまま出す。
    seed と最終 champion は seed/final 行の hash から名付ける。"""
    name = trace_path.name
    if not name.endswith(".trace.jsonl"):
        return {}
    ledger = trace_path.with_name(name[:-len(".trace.jsonl")] + ".jsonl")
    if not ledger.exists():
        return {}
    names: dict = {}

    def add(h, label):
        # 同じ設計に複数の名が付く(種に戻す 1 手変異は種と同じ hash)。全部を " = " で
        # つないで出す: 「この候補は種そのもの」が表で読める
        if h and label and label not in names.setdefault(h, []):
            names[h].append(label)

    with ledger.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            if r.get("event") == "candidate":
                add(r.get("hash"), r.get("label"))
            elif r.get("event") in ("seed", "resumed"):
                add(r.get("hash"), "seed")
            elif r.get("event") == "final":
                add(r.get("champion_hash"), "champion")     # 種のままなら "seed = champion"
    return {h: " = ".join(ls) for h, ls in names.items()}


def _resolve_spec(prefix: str, rows: list) -> str:
    """hash は前方一致で受ける(12 桁を打たせない)。曖昧なら候補を並べて断る。"""
    # 空の前置きは全部に当たって「曖昧」と出るが、原因は打ち忘れなので先にそう言う
    if not prefix:
        raise ValueError("empty design prefix; pass the first characters of a hash")
    found = sorted({r.get("spec") for r in rows if r.get("spec") and r["spec"].startswith(prefix)})
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(f"no design in the trace starts with {prefix!r}")
    raise ValueError(f"{prefix!r} matches {len(found)} designs: {', '.join(found)}")


_RAISED_FIELDS = (("champion", "champion_error_rate"), ("champion search", "champion_search_error_rate"),
                  ("challenger", "challenger_error_rate"),
                  ("challenger search", "challenger_search_error_rate"),
                  # 表示名は台帳の鍵と同じ綴りにする(片方で grep した人がもう片方に届く)
                  ("simplify", "simplify_error_rate"),
                  ("simplify search", "simplify_search_error_rate"))


def _raised_note(row: dict) -> str:
    """判定に使った測定の失敗率(全部 0 なら何も言わない)。

    小さい率を `0%` と書かない: 130 コールの 1 本が落ちた 0.77% を「0%」と出すと、
    非ゼロだと言うために付けた行が「ゼロだ」と読める(codex 指摘)。
    """
    def pct(rate: float) -> str:
        if rate >= 0.01:
            return f"{rate:.0%}"
        # 0.1% 未満も「0.0%」にはしない: この行は「ゼロではない」と言うために出ている。
        return f"{rate:.1%}" if rate >= 0.001 else "<0.1%"

    parts = []
    for who, key in _RAISED_FIELDS:
        try:
            # 古い台帳・手で書いた行も通る入口なので、読めない値は黙って飛ばす(表示専用の
            # 行が例外で走行の報告を止める方が悪い)。
            rate = float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if rate:
            parts.append(f"{who} {pct(rate)}")
    return f" [calls that raised: {', '.join(parts)}]" if parts else ""


def cmd_trace(args: argparse.Namespace) -> int:
    """`gama trace run.trace.jsonl`: 台帳が言えない「なぜ」を読む。既定は要約(走行の終わりの
    行と同じ)と、設計 × 役 × split ごとの call 数・平均点・切断数。`--rows` で条件に合う call を
    1 行ずつ(末尾つき)、`--diff A B` で 2 設計の case ごとの差。"""
    path = Path(args.file)
    if not path.exists():
        sys.stderr.write(f"[gama] no trace at {path}\n")
        return 1
    rows = trace_rows(path)
    labels = _trace_labels(path)
    summary = trace_summary(path, rows)
    for line in trace_lines({"trace": summary}):
        sys.stderr.write(f"[gama] {line[2:] if line.startswith('- ') else line}\n")
    try:
        if args.diff:
            a, b = (_resolve_spec(h, rows) for h in args.diff)
            d = trace_diff(rows, a, b)
            if args.json:
                print(json.dumps(d, ensure_ascii=False, indent=2))
                return 0
            print(f"a = {a} {labels.get(a, '')}".rstrip())
            print(f"b = {b} {labels.get(b, '')}".rstrip())
            # 「cases」は台帳と同じ単位(点差の合計・1 case 満点=1.0)で、case の数の増減ではない
            print(f"shared cases: {d['shared']}  b-a: {d['delta_cases']:+} cases "
                  f"(sum of per-case score differences; {d['wins']}w-{d['losses']}l-{d['ties']}t)"
                  + (f"  by class: " + ", ".join(f"{c} {v:+}" for c, v in d["by_class"].items())
                     if d["by_class"] else ""))
            for m in d["moved"]:
                print(f"  {m['split']:<8} {m['case']:<28} {m['class'] or '-':<20} "
                      f"{m['a']:.2f} -> {m['b']:.2f} ({m['delta']:+.2f}; n={m['n_a']}/{m['n_b']})")
            if d["only_a"] or d["only_b"]:
                print(f"  not compared: only a {d['only_a']}, only b {d['only_b']}")
            return 0
        want = dict(spec=_resolve_spec(args.spec, rows) if args.spec else None,
                    finish=args.finish, **{"class": args.cls}, split=args.split, role=args.role,
                    gen=args.gen)
    except ValueError as e:
        sys.stderr.write(f"[gama] {e}\n")
        return 2
    picked = [r for r in rows
              if all(v is None or r.get(k) == v for k, v in want.items())]
    if args.rows:
        for r in picked:
            if args.json:
                print(json.dumps(r, ensure_ascii=False))
                continue
            tail = (r.get("tail") or "").replace("\n", "\\n")
            print(f"gen={r.get('gen')} {r.get('role')} {r.get('spec')} {r.get('split')} "
                  f"{r.get('case')} rep={r.get('rep')} score={r.get('score')} "
                  f"finish={r.get('finish')} chars={r.get('chars')} tokens={r.get('tokens')}"
                  + (f" error={r['error']!r}" if r.get("error") else "")
                  + f" | ...{tail[-80:]}")
        sys.stderr.write(f"[gama] {len(picked)} of {len(rows)} calls match\n")
        return 0
    # 設計 × 役 × split の表: 「この設計は何回測られ、平均は、何 call 切れたか」が一目で
    groups: dict = {}
    for r in picked:
        g = groups.setdefault((r.get("spec"), r.get("role"), r.get("split")),
                              {"calls": 0, "score": 0.0, "scored": 0, "length": 0, "gens": set()})
        g["calls"] += 1
        if r.get("score") is not None:
            g["score"] += float(r["score"])
            g["scored"] += 1
        g["length"] += r.get("finish") == "length"
        g["gens"].add(r.get("gen"))
    if args.json:
        print(json.dumps([{"spec": k[0], "label": labels.get(k[0]), "role": k[1], "split": k[2],
                           "calls": g["calls"],
                           "mean": round(g["score"] / g["scored"], 4) if g["scored"] else None,
                           "cut": g["length"],
                           "gens": sorted((x for x in g["gens"] if x is not None))}
                          for k, g in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2])))],
                         ensure_ascii=False, indent=2))
        return 0
    print(f"{'spec':<13} {'role':<11} {'split':<8} {'calls':>5} {'mean':>7} {'cut':>4}  gens  label")
    for k, g in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))):
        mean = f"{g['score'] / g['scored']:.4f}" if g["scored"] else "-"
        gens = ",".join(str(x) for x in sorted(x for x in g["gens"] if x is not None)) or "-"
        print(f"{str(k[0]):<13} {str(k[1]):<11} {str(k[2]):<8} {g['calls']:>5} {mean:>7} "
              f"{g['length']:>4}  {gens:<5} {labels.get(k[0], '')}")
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
    pb.add_argument("--suite", default="default", choices=sorted(SUITES),
                    help="case suite. " + " | ".join(f"{k} ({v})"
                                                       for k, v in sorted(SUITE_DOCS.items())))
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
    pm.add_argument("--suite", default="hard", choices=sorted(SUITES),
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
        "mesh", help="does ensembling help? the co-failure (beta) ignition verdict from your bench")
    pmesh.add_argument("--backends", default="echo,null",
                       help="comma list of ensemble members (>=2), e.g. gemma,qwen,llama. "
                            "'echo,null' = free smoke")
    pmesh.add_argument("--suite", default="hard", choices=sorted(SUITES),
                       help="case suite (default: hard — discriminating, so members can differ)")
    pmesh.add_argument("--pass-score", type=float, default=1.0,
                       help="a case score >= this counts as solved (its external verifier passed)")
    pmesh.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pmesh.add_argument("--repeats", type=int, default=1)
    pmesh.add_argument("--config", default=None,
                       help="per-backend kwargs + composites (ensemble/gama/meshflow)")
    pmesh.set_defaults(func=cmd_mesh)

    pc = sub.add_parser(
        "calib", help="how much does a verifier KNOW? confusion counts + I(verdict;correct) "
                      "bits + the achievable selection ceiling (v3a/T8)")
    pc.add_argument("--backend", default="echo",
                    help="single backend whose outputs the verifier judges ('echo' = free smoke)")
    pc.add_argument("--verify", required=True,
                    help="verifier name (built-in, e.g. code_runs / nonempty)")
    pc.add_argument("--suite", default="hard",
                    choices=sorted(SUITES))
    pc.add_argument("--pass-score", type=float, default=1.0,
                    help="verify score >= this counts as pass (same rule as the meshflow gate)")
    pc.add_argument("--tier", default="large", choices=["small", "medium", "large"])
    pc.add_argument("--observed", type=int, default=None,
                    help="observed correct picks by any V-based accept/reject policy "
                         "(count, SAME suite+tier) for the DPI budget check: observed > "
                         "ceiling means the MEASUREMENT is wrong")
    pc.add_argument("--observed-n", type=int, default=None,
                    help="case count of the observed run; lets the check verify the "
                         "same-suite premise instead of taking it on faith")
    pc.add_argument("--config", default=None,
                    help="per-backend kwargs + composites (ensemble/gama/meshflow)")
    pc.set_defaults(func=cmd_calib)

    pt = sub.add_parser("trace", help="read a grow run's per-call trace: what each reply "
                                       "looked like, which cases moved between two designs")
    pt.add_argument("file", help="the trace (run.trace.jsonl; its ledger run.jsonl beside it "
                                 "supplies the labels when present)")
    pt.add_argument("--spec", default=None, metavar="HASH",
                    help="one design (hash prefix from the ledger / the table)")
    pt.add_argument("--finish", default=None, help="only calls that stopped for this reason "
                                                   "(stop, length)")
    pt.add_argument("--class", dest="cls", default=None, help="only this task class")
    pt.add_argument("--split", default=None, help="search, confirm or sealed")
    pt.add_argument("--role", default=None,
                    help="champion, candidate, challenger, simplifier or seed")
    pt.add_argument("--gen", type=int, default=None, help="only this generation")
    pt.add_argument("--rows", action="store_true",
                    help="print the matching calls one per line (with the reply's tail) "
                         "instead of the per-design table")
    pt.add_argument("--diff", nargs=2, default=None, metavar=("A", "B"),
                    help="per-case difference between two designs (hash prefixes), mean over "
                         "every measurement of each in the file")
    pt.add_argument("--json", action="store_true", help="machine-readable output")
    pt.set_defaults(func=cmd_trace)

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
                    help="free deterministic smoke with two echo lanes (promotes nothing)")
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
    pg.add_argument("--max-paired-p", type=_nonneg_float, default=None, metavar="P",
                    help="additionally require the challenger's per-case wins to be unlikely "
                         "by chance (one-sided sign test p). Default off: at typical case "
                         "counts this demands a near-sweep and blocks almost every promotion. "
                         "Read `paired_p` in the ledger first, then decide with your numbers")
    pg.add_argument("--ensemble-strategy", default="synthesize",
                    choices=["majority", "first", "synthesize"],
                    help="aggregation for proposed ensemble lanes. Default synthesize: the "
                         "other member writes the final answer from both drafts (one extra "
                         "call per case). majority compares replies verbatim, so on free-text "
                         "answers it ties and falls back to the first member")
    pg.add_argument("--start-from", default=None, metavar="CONFIG",
                    help="grow onward from an existing champion (a grown recipe's config.json, "
                         "or any config with a 'system' spec) instead of from a bare model. "
                         "The pool from --models should contain the same models the config "
                         "references, or its lanes cannot be simplified back")
    pg.add_argument("--resume", default=None, metavar="LEDGER",
                    help="continue an interrupted run from its JSONL ledger (a long run that "
                         "dies has already paid for its measurements). Give the same path as "
                         "--out to continue in place: the ledger is appended to, not rewritten. "
                         "Refused if the ledger used a different split, since its sealed cases "
                         "would not be sealed under this one")
    pg.add_argument("--out", default=None, help="write the JSONL grow ledger")
    pg.add_argument("--trace", default=None, metavar="PATH",
                    help="per-call records (score, tokens, finish reason, reply head/tail) "
                         "beside the ledger; default is --out with its extension replaced by "
                         ".trace.jsonl (run.jsonl -> run.trace.jsonl); appended to, like the "
                         "ledger, when --resume is the --out file, started over otherwise")
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
