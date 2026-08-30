r"""decorrelation — when does *ensembling* actually help? the ignition law for ``EnsembleBackend``.

soshiki-genron ``model/mesh.py`` を gama に反映。:mod:`gama.market`（escalation の **コスト** 閾値
``p > w/s``）の対をなす: mesh（n エージェントが独立に解き、外部検証で通った解の union を取る）が
**best-single を超えるかは *脱相関* の問題**である。

n エージェント・各完全解率 ``p``・失敗相関 ``rho``（exchangeable 一因子モデル: 確率 ``rho`` で全員が
同結果＝comonotone、確率 ``1-rho`` で独立）:

    P(all fail) = rho*(1-p) + (1-rho)*(1-p)^n
    **mesh 利得 = union − best_single = (1-rho)*(1-p)*(1 - (1-p)^(n-1))**

**点火 ⟺ ``rho < 1``（脱相関がある）かつ ``0<p<1`` かつ ``n>=2``**。``rho=1``（共通 hard core /
入れ子）→ 利得 0 ＝ ``EnsembleBackend`` を足しても best-single 止まり（単一最強で足りる）。

``EnsembleBackend`` はメンバを *盲目的に* 束ねる。本モジュールはその裏返しで「いつ束ねが効くか」を
``gama bench`` の実測から測る —— 脱相関(``rho<1``)だけでなく *相互* 相補（非入れ子）が要る点も、
``union − best`` が自然に出す（非対称＝入れ子なら gain 0）。これは gama のテーゼ "Structure, not scale"
の合議版: 多様性が効くのは agent が *違う誤り* をするとき(``rho<1``)だけ。

**主量は共倒れ率 β であって、ペア相関 rho ではない（2026-08-31 改訂）。**
上の解析モデルの ``rho`` は「全員が同時に落ちる混合の重み」（共通ショック）であって、実測で数える
ペアの φ 係数とは別物である。ペア相関は 3 体以上では「全員が同じ問いで誤る率」
β = P(all fail) を同定できない（同じ周辺分布・同じペア相関で β が異なる分布が存在する。
Chen 2026, arXiv:2606.27288 Prop. 3。67 フロンティアモデルの実測で、ペア相関から較正した単因子模型は
β を約 2.5 倍過小評価し、その誤差はプール規模で拡大した）。一方 union の正しさは恒等的に ``1 − β``
なので、答えを 1 つ選ぶどんな方策（多数決・ルータ・cascade・検証つき union）も ``1 − β`` を超えられない。
だから実測側は **β を直接数え、稀事象なので正確な二項区間（Clopper–Pearson）を添える**。ペア φ は
「なぜ β がその値か」の補助読みとしてだけ残す。soshiki-genron で一度「創発」として報告され
再測定で撤回された +0.042（N=24 の 1 ケース分）は、この区間が 0 を跨いでいたことの非公式な発見だった。
"""
from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# Analytic model (soshiki-genron model/mesh.py). Deterministic, stdlib-only.
# ``rho`` here is the common-shock mixing weight, NOT the pairwise phi below.
# --------------------------------------------------------------------------- #
def union_fail_prob(p: float, rho: float, n: int) -> float:
    """P(every one of n agents fails), one-factor exchangeable model. rho=1 comonotone, rho=0 independent."""
    q = 1.0 - p
    return rho * q + (1.0 - rho) * (q ** n)


def mesh_correctness(p: float, rho: float, n: int) -> float:
    """Union correctness: external verify keeps any passing answer -> solved iff ANY agent solves."""
    return round(1.0 - union_fail_prob(p, rho, n), 6)


def mesh_gain(p: float, rho: float, n: int) -> float:
    """``union − best_single = (1-rho)*(1-p)*(1 - (1-p)^(n-1))``. > 0 iff rho<1, 0<p<1, n>=2."""
    if n < 1:
        return 0.0
    return round((1.0 - rho) * (1.0 - p) * (1.0 - (1.0 - p) ** (n - 1)), 6)


def ignites(p: float, rho: float, n: int) -> bool:
    """Does the mesh union *strictly* beat best-single? (the decorrelation ignition condition).

    Analytic and binary: True for any ``rho < 1``. That is a statement about the model, not a
    verdict about a measured mesh — for measured members use :func:`analyze`, whose ``verdict``
    is certified from the co-failure rate and its interval."""
    return mesh_gain(p, rho, n) > 1e-9


# --------------------------------------------------------------------------- #
# Empirical bridge — measure beta / union / gain from `gama bench` records.
# Each member's per-case solve vector (1=solved) gives the co-failure count
# directly; the pairwise phi is kept only as a secondary diagnostic.
# --------------------------------------------------------------------------- #
def _phi(fa: list, fb: list):
    """phi coefficient (= Pearson) of two failure indicators. 1=comonotone, 0=independent, <0=anti.

    ``None`` when undefined (a constant vector: a member that solved everything or nothing has
    no failure variance to correlate). Reporting a number there would read as a diagnosis."""
    n = len(fa)
    if n == 0:
        return None
    ma, mb = sum(fa) / n, sum(fb) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(fa, fb)) / n
    va, vb = ma * (1 - ma), mb * (1 - mb)
    return None if va <= 0 or vb <= 0 else cov / ((va * vb) ** 0.5)


def failure_correlation(solve_vectors: list):
    """Mean pairwise failure correlation across members (secondary diagnostic), or ``None``.

    solve_vectors=[[0/1,...],...] (1=solved). Pairs whose phi is undefined (a constant member)
    are left out of the mean; if no pair is defined the result is ``None`` rather than a number.
    This is a *pairwise* statistic: with 3+ members it cannot identify the co-failure rate (see
    :func:`cofailure`), so it explains but never certifies a verdict."""
    fails = [[1 - s for s in v] for v in solve_vectors]
    phis = [_phi(fails[i], fails[j])
            for i in range(len(fails)) for j in range(i + 1, len(fails))]
    defined = [x for x in phis if x is not None]
    return round(sum(defined) / len(defined), 4) if defined else None


def _n_cases(solve_vectors: list) -> int:
    """Case count shared by every member's solve vector; refuse empty or ragged input.

    Both counters below index case ``c`` across members, so a ragged list would raise an
    IndexError deep inside a comprehension (or silently drop cases if it happened to be shorter
    for the union). These are exported functions, so say what was wrong at the door."""
    if not solve_vectors or not solve_vectors[0]:
        raise ValueError("solve_vectors must hold >= 1 member with >= 1 case")
    n = len(solve_vectors[0])
    if any(len(v) != n for v in solve_vectors):
        raise ValueError(f"solve_vectors are ragged: lengths {[len(v) for v in solve_vectors]}")
    return n


def union_solve(solve_vectors: list) -> float:
    """Correctness of the externally-verified union = fraction of cases ANY member solved."""
    n = _n_cases(solve_vectors)
    return round(sum(1 for k in range(n) if any(v[k] for v in solve_vectors)) / n, 4)


def cofailure(solve_vectors: list) -> tuple:
    """``(k, n)``: k = cases where EVERY member failed, n = cases. ``k/n`` is the co-failure rate β.

    β is the primary quantity because it is exactly what caps any answer-selecting policy:
    on a co-failure case no member is right, so no vote/router/cascade/union can be — accuracy
    ≤ 1 − β (Kuncheva's oracle bound; Chen 2026 arXiv:2606.27288). It is counted directly from
    the solve vectors instead of inferred from pairwise correlation, which cannot see it."""
    n = _n_cases(solve_vectors)
    k = sum(1 for c in range(n) if not any(v[c] for v in solve_vectors))
    return k, n


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p).

    Each term is built in log space (lgamma) and exponentiated, so no term can overflow and a
    term too small to represent underflows to 0 — harmless, it was negligible against the sum.
    The remaining error is float accumulation over <= n+1 terms (~n·1e-16), far below the
    1e-2-scale tail probabilities the bisection targets, so monotonicity in ``p`` survives at
    bench sizes. Validated range: the tests pin the bounds to published exact-interval values at
    n = 10..480 and to the closed form at n = 10,000. Beyond ~10^6 cases the accumulated error
    approaches the 1e-9-scale tails that matter there; use a beta-quantile routine instead."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma(n + 1)

    def pmf(i: int) -> float:
        return math.exp(lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)

    # Sum whichever tail is shorter: the bisection calls this ~60 times per bound, and k near
    # n/2 would otherwise cost n/2 terms each call.
    if k < n - k:
        return sum(pmf(i) for i in range(0, k + 1))
    return 1.0 - sum(pmf(i) for i in range(k + 1, n + 1))


def _solve_monotone(f, target: float, increasing: bool) -> float:
    """Root of f(p) = target on (0, 1) by bisection; f must be monotone. 60 halvings ≈ 1e-18."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple:
    """Exact (Clopper–Pearson) two-sided interval for a binomial rate ``k/n``.

    Chosen over a normal approximation because β is a rare-event rate: with k = 0..20 all-fail
    cases the Wald interval is useless (it can be empty or negative), and the whole point of the
    interval is to stop a one-case fluke (1/24 ≈ 0.042) from being reported as ignition. Solved by
    bisection on the binomial tails so the package stays stdlib-only. ``k=0`` → lower 0,
    ``k=n`` → upper 1 (the closed forms ``1 − (α/2)^(1/n)`` and ``(α/2)^(1/n)`` fall out)."""
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"clopper_pearson needs 0 <= k <= n, n > 0 (got k={k}, n={n})")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1) (got {confidence})")
    half_alpha = (1.0 - confidence) / 2.0
    # lower: largest p with P(X >= k | p) = α/2 ; P(X >= k) is increasing in p.
    lower = 0.0 if k == 0 else _solve_monotone(
        lambda p: 1.0 - _binom_cdf(k - 1, n, p), half_alpha, increasing=True)
    # upper: smallest p with P(X <= k | p) = α/2 ; P(X <= k) is decreasing in p.
    upper = 1.0 if k == n else _solve_monotone(
        lambda p: _binom_cdf(k, n, p), half_alpha, increasing=False)
    # Returned unrounded: a verdict compares these against 0 and a rounded bound can cross it.
    return lower, upper


def solve_vectors_from_records(records: list, members: list, pass_score: float = 1.0) -> list:
    """Build each member's 0/1 per-case solve vector (1 iff its score >= pass_score), aligned by case."""
    score: dict = {}
    cases: list = []
    seen: set = set()          # membership via set: the list scan was O(n²) on large benches
    for r in records:
        key = (r["case_id"], r["backend"])
        score[key] = max(score.get(key, 0.0), float(r.get("score", 0.0)))
        if r["case_id"] not in seen:
            seen.add(r["case_id"])
            cases.append(r["case_id"])   # list keeps first-seen order (aligned vectors, stable)
    return [[1 if score.get((c, m), 0.0) >= pass_score else 0 for c in cases] for m in members]


def analyze(records: list, members: list, pass_score: float = 1.0, confidence: float = 0.95) -> dict:
    """Measure whether ensembling ``members`` ignites, certified from the co-failure rate β.

    ``union = 1 − β`` exactly, so ``mesh_gain = union − best_single`` is the oracle gain of the
    externally-verified union over the best single member. The verdict does not stop at the
    point estimate: β gets an exact ``confidence`` interval and the gain is re-read through it.

    * ``dead`` — no case where the union beats the best member (nested / identical members).
    * ``certified`` — the union's lower bound beats the best member's upper bound:
      ``(1 − β_hi) − best_hi > 0``, both ends from exact intervals; the best member's interval is
      taken at ``1 − (1 − confidence)/m`` because it was selected as the max of ``m`` members on
      this sample (Bonferroni). Using the two one-sided ends together is conservative and ignores
      that union and best are paired on the same cases — so certification is harder to earn than
      a paired test would make it, never easier. More cases than one fluke can explain.
    * ``undetermined`` — a gain was observed but the sample cannot separate it from 0; report
      the size and the interval, do not say "it fired" (the soshiki-genron +0.042 retraction).

    ``gain_bounds = [(1 − β_hi) − best_hi, (1 − β_lo) − best_lo]`` are those conservative
    bounds — not a confidence interval of the paired difference — and ``gain_upper_bound`` is the
    certificate the other way: no policy that returns one member's answer can gain more than this
    over best-single at this confidence. The Bonferroni step is a bound, not a model of the
    selection: with many members and few cases, certify on a held-out split (as ``gama grow``
    does) rather than trusting the in-sample maximum."""
    if len(members) < 2:
        raise ValueError("mesh needs >= 2 members")
    vecs = solve_vectors_from_records(records, members, pass_score)
    k, n = cofailure(vecs)
    solved = [sum(v) for v in vecs]
    # Everything the verdict depends on is a case COUNT, never a rounded rate: on a 30,000-case
    # bench a one-case gain is 0.00003, which a 4-place rate rounds to 0.0 and misreads as "dead"
    # (nested). Likewise the best member is the one with the most solved cases, ties -> the
    # first listed (stable and documented, not whichever rounding happened to favour).
    best_i = max(range(len(solved)), key=lambda i: (solved[i], -i))
    union_cases = n - k                        # union ⊇ best, so this never undercounts the best
    gain_cases = union_cases - solved[best_i]  # cases the best member failed but some member solved
    beta = k / n
    beta_lo, beta_hi = clopper_pearson(k, n, confidence)
    # The best member was picked as the max of m on this same sample, so its rate is biased up
    # (winner's curse). Bonferroni over the m candidates (confidence 1 − α/m for that interval)
    # is the cheap, always-conservative correction: the more members you fan out, the harder
    # it gets to certify — the direction a pre-run gate must fail in.
    m = len(vecs)
    best_conf = 1.0 - (1.0 - confidence) / m
    best_lo, best_hi = clopper_pearson(solved[best_i], n, best_conf)
    gain_lo = (1.0 - beta_hi) - best_hi
    gain_hi = (1.0 - beta_lo) - best_lo
    if gain_cases == 0:
        verdict = "dead"
    elif gain_lo > 0.0:
        verdict = "certified"
    else:
        verdict = "undetermined"
    r6 = lambda x: round(x, 6)  # noqa: E731 — display rounding, applied after the verdict
    per = [round(c / n, 4) for c in solved]
    return {
        "members": list(members),
        "per_member_solve_rate": per,          # empirical p_i (1=solved per case)
        "best_single": per[best_i],
        "best_single_interval": [r6(best_lo), r6(best_hi)],   # at 1-(1-confidence)/members
        "best_interval_confidence": round(best_conf, 6),
        "best_member": members[best_i],
        "union": round(union_cases / n, 4),
        "cofailure_beta": round(beta, 4),      # PRIMARY: share of cases every member failed
        "cofailure_k": k,
        "n_cases": n,
        "confidence": confidence,
        "beta_interval": [r6(beta_lo), r6(beta_hi)],   # exact Clopper–Pearson
        "ceiling": round(1.0 - beta, 4),       # no answer-selecting policy scores above this
        "mesh_gain": round(gain_cases / n, 4), # union − best_single (the oracle gain, point)
        "gain_cases": gain_cases,              # the count behind it; 0 is what "dead" means
        "gain_bounds": [r6(gain_lo), r6(gain_hi)],   # conservative, from the two marginal intervals
        "gain_upper_bound": r6(gain_hi),       # certificate: max gain any such policy can deliver
        "verdict": verdict,                    # dead | undetermined | certified
        "ignites": verdict == "certified",     # binary view of the verdict; False while undetermined
        "failure_rho": failure_correlation(vecs),   # SECONDARY: pairwise phi, cannot identify beta
        "thesis": ("Any policy that returns one member's answer scores at most 1 − β, β = the "
                   "co-failure rate (every member wrong on the same case). Ensembling ignites only "
                   "when the union's gain over best-single clears β's exact interval; pairwise "
                   "failure correlation rho explains but cannot certify it (m>=3 non-identification, "
                   "Chen 2026 arXiv:2606.27288; soshiki-genron mesh, analytic gain="
                   "(1-rho)(1-p)(1-(1-p)^(n-1)))."),
    }
