r"""market — the *economics* of combining vs scaling, ported from soshiki-genron.

soshiki-genron の §5「検証ルーティング市場」を gama の **経済 verdict 層** にしたもの.
出典: ``~/Projects/soshiki-genron`` ``model/market.py`` (解析) +
``experiments/config_mesh_market.py`` (実測ブリッジ).

gama の :class:`~gama.meshflow.MeshflowBackend` は段階委譲(cheap->expensive を外部検証で
gate)を *運用* する. 本モジュールはその裏返し —— 「**その段階委譲は、単一モデルを使うより
本当に安く同じ正答へ届くのか**」を判定する. それは構造でなく *分布* の問題で、解析的には
ひとつの閾値に落ちる:

    2ティアのエスカレーション市場(安い w・完全解率 p / 落ちた時だけ高い s):
        期待コスト  C = w + (1 - p)*s,   正しさ = 1 (高ティアが必ず救う)
    **支配定理**: 市場が flat-strong(コスト s・正しさ 1) を Pareto 支配 ⟺ C < s ⟺ **p > w/s**.

安いほど(w/s 小)バーは低い. 均質(w = s)なら p > 1 が必要 ＝ 不可能 ＝ 市場は勝てない.
これは gama のテーゼ "Structure, not scale" の経済的な言い換え: 構造が効くのは agent 間に
*能力差* があるとき(p と w/s)だけ. ``gama market`` は ``gama bench`` の実測からこの verdict を出す.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# Analytic model (soshiki-genron model/market.py). Deterministic, stdlib-only.
# --------------------------------------------------------------------------- #
def escalation_cost(w: float, s: float, p: float) -> float:
    """Expected cost of the 2-tier escalation market: pay ``w`` always, pay ``s`` when the
    weak tier fails (prob ``1 - p``). Correctness is 1 (the strong tier always rescues)."""
    return round(w + (1 - p) * s, 4)


def p_star(w: float, s: float) -> float:
    """Threshold weak-solve-rate above which the market Pareto-dominates flat-strong
    (= the cost ratio ``w/s``). A cheaper weak tier (smaller ``w/s``) -> a lower bar."""
    return round(w / s, 4) if s else float("inf")


def dominates(w: float, s: float, p: float) -> bool:
    """Market gives flat-strong's correctness (1) strictly cheaper  <=>  ``p > w/s``."""
    return escalation_cost(w, s, p) < s - 1e-9


def ladder(tiers: list) -> dict:
    """n-tier escalation ladder. ``tiers = [(cost, p_solve)]`` cheap->expensive. Returns the
    expected ``{"cost", "correctness"}`` (correctness = ``1 - prod(1 - p_i)``)."""
    cost, reach, fail = 0.0, 1.0, 1.0
    for c, p in tiers:
        cost += c * reach            # reach (and pay) this tier with prob `reach`
        reach *= (1 - p)             # still unsolved -> escalate further
        fail *= (1 - p)
    return {"cost": round(cost, 4), "correctness": round(1 - fail, 4)}


# --------------------------------------------------------------------------- #
# Empirical bridge — turn `gama bench` records into the market verdict.
# (soshiki-genron experiments/config_mesh_market.py, driven by gama's own bench.)
# --------------------------------------------------------------------------- #
def _scores_by_case(records: list) -> tuple:
    """``-> ({(case_id, backend): best_score}, [case_id, ...] in first-seen order)``."""
    score: dict = {}
    cases: list = []
    for r in records:
        key = (r["case_id"], r["backend"])
        score[key] = max(score.get(key, 0.0), float(r.get("score", 0.0)))
        if r["case_id"] not in cases:
            cases.append(r["case_id"])
    return score, cases


def _tier_costs(tier_order: list, costs=None) -> list:
    """Per-tier cost weights (price/compute proxy), cheap->expensive. Default ``1, 2, 3, ...``
    (the same proxy ``MeshflowBackend`` uses). Length must match ``tier_order``."""
    if costs:
        cs = [float(c) for c in costs]
        if len(cs) != len(tier_order):
            raise ValueError(
                f"costs has {len(cs)} entries but tier_order has {len(tier_order)}")
        return cs
    return [float(i + 1) for i in range(len(tier_order))]


def market_over_records(records: list, tier_order: list, costs=None,
                        pass_score: float = 1.0) -> dict:
    """Verification-routing market over a bench's ``records``.

    For each case, walk ``tier_order`` cheap->expensive, paying each tier's cost, and stop
    at the first tier whose score for that case is ``>= pass_score`` (its external verifier
    passed). The market's cost is the summed tier cost; it 'solves' a case iff some tier
    passed. Mirrors soshiki-genron's ``config_mesh_market``, driven by gama's measurements.
    """
    score, cases = _scores_by_case(records)
    cs = _tier_costs(tier_order, costs)
    total_cost, solved, ladders = 0.0, 0, []
    for cid in cases:
        c, routed = 0.0, None
        for i, b in enumerate(tier_order):
            c += cs[i]
            if score.get((cid, b), 0.0) >= pass_score:
                routed = b
                break
        total_cost += c
        solved += 1 if routed else 0
        ladders.append({"case": cid, "routed_to": routed, "cost": round(c, 4)})
    n = len(cases)
    return {
        "n_cases": n,
        "tier_order": list(tier_order),
        "costs": cs,
        "pass_score": pass_score,
        "market_cost": round(total_cost, 4),
        "cost_per_case": round(total_cost / n, 4) if n else 0.0,
        "pass_rate": round(solved / n, 4) if n else 0.0,
        "ladders": ladders,
    }


def analyze(records: list, tier_order: list, costs=None, pass_score: float = 1.0) -> dict:
    """Compare the verification-routing market against each flat (single-model) backend and
    return the Pareto-dominance verdict vs **flat-strong** (the most expensive tier).

    A flat backend's cost over N cases = its tier cost x N; its ``pass_rate`` = fraction of
    cases it solves alone (``score >= pass_score``). The market Pareto-dominates flat-strong
    iff it matches/exceeds that pass_rate at strictly lower cost. ``analytic`` adds the
    closed-form 2-tier lens (weak vs strong): ``p > w/s``.
    """
    if len(tier_order) < 2:
        raise ValueError("market needs >= 2 tiers (cheap->expensive)")
    score, cases = _scores_by_case(records)
    cs = _tier_costs(tier_order, costs)
    n = len(cases)
    market = market_over_records(records, tier_order, cs, pass_score)
    flats = []
    for i, b in enumerate(tier_order):
        solved = sum(1 for cid in cases if score.get((cid, b), 0.0) >= pass_score)
        flats.append({
            "backend": b,
            "cost": round(cs[i] * n, 4),
            "pass_rate": round(solved / n, 4) if n else 0.0,   # empirical solve-rate p_i
        })
    strong = flats[-1]
    weak = flats[0]
    verdict = (market["pass_rate"] >= strong["pass_rate"] - 1e-9
               and market["market_cost"] < strong["cost"] - 1e-9)
    w, s, p = cs[0], cs[-1], weak["pass_rate"]
    analytic = {
        "w_weak": w, "s_strong": s, "p_weak": round(p, 4),
        "p_star": p_star(w, s),
        "escalation_cost_2tier": escalation_cost(w, s, p),
        "dominates_2tier": dominates(w, s, p),
        "threshold": "market Pareto-dominates flat-strong  <=>  p_weak > w/s",
    }
    return {
        "market": market,
        "flats": flats,
        "flat_strong": strong,
        "market_dominates_flat_strong": verdict,
        "analytic": analytic,
        "thesis": ("Structure beats scale only when agents differ in capability: the "
                   "escalation market wins iff the cheap tier's solve-rate p exceeds the "
                   "cost ratio w/s (soshiki-genron §5, p > w/s)."),
    }
