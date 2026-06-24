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
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# Analytic model (soshiki-genron model/mesh.py). Deterministic, stdlib-only.
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
    """Does the mesh union *strictly* beat best-single? (the decorrelation ignition condition)."""
    return mesh_gain(p, rho, n) > 1e-9


# --------------------------------------------------------------------------- #
# Empirical bridge — measure rho / union / gain from `gama bench` records.
# Each member's per-case solve vector (1=solved) gives the *measured* failure
# correlation; ignition needs rho<1 AND mutual (non-nested) complementarity,
# which union−best reflects directly (a nested member set scores gain 0).
# --------------------------------------------------------------------------- #
def _phi(fa: list, fb: list) -> float:
    """phi coefficient (= Pearson) of two failure indicators. 1=comonotone, 0=independent, <0=anti."""
    n = len(fa)
    if n == 0:
        return 0.0
    ma, mb = sum(fa) / n, sum(fb) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(fa, fb)) / n
    va, vb = ma * (1 - ma), mb * (1 - mb)
    return 0.0 if va <= 0 or vb <= 0 else cov / ((va * vb) ** 0.5)


def failure_correlation(solve_vectors: list) -> float:
    """Mean pairwise failure correlation rho across members. solve_vectors=[[0/1,...],...] (1=solved)."""
    fails = [[1 - s for s in v] for v in solve_vectors]
    pairs = [(i, j) for i in range(len(fails)) for j in range(i + 1, len(fails))]
    return (round(sum(_phi(fails[i], fails[j]) for i, j in pairs) / len(pairs), 4)
            if pairs else 0.0)


def union_solve(solve_vectors: list) -> float:
    """Correctness of the externally-verified union = fraction of cases ANY member solved."""
    n = len(solve_vectors[0])
    return round(sum(1 for k in range(n) if any(v[k] for v in solve_vectors)) / n, 4)


def solve_vectors_from_records(records: list, members: list, pass_score: float = 1.0) -> list:
    """Build each member's 0/1 per-case solve vector (1 iff its score >= pass_score), aligned by case."""
    score: dict = {}
    cases: list = []
    for r in records:
        key = (r["case_id"], r["backend"])
        score[key] = max(score.get(key, 0.0), float(r.get("score", 0.0)))
        if r["case_id"] not in cases:
            cases.append(r["case_id"])
    return [[1 if score.get((c, m), 0.0) >= pass_score else 0 for c in cases] for m in members]


def analyze(records: list, members: list, pass_score: float = 1.0) -> dict:
    """Measure whether ensembling ``members`` ignites: does the externally-verified union beat the
    best single member, and is the measured failure correlation ``rho < 1``? ``gain > 0`` means an
    ``EnsembleBackend`` over these members can do what none does alone (decorrelated, non-nested)."""
    if len(members) < 2:
        raise ValueError("mesh needs >= 2 members")
    vecs = solve_vectors_from_records(records, members, pass_score)
    per = [round(sum(v) / len(v), 4) for v in vecs]
    union = union_solve(vecs)
    best_i = max(range(len(per)), key=lambda i: per[i])
    gain = round(union - per[best_i], 4)
    return {
        "members": list(members),
        "per_member_solve_rate": per,          # empirical p_i (1=solved per case)
        "best_single": per[best_i],
        "best_member": members[best_i],
        "union": union,
        "mesh_gain": gain,                     # union − best_single
        "failure_rho": failure_correlation(vecs),
        "ignites": gain > 1e-9,
        "thesis": ("Ensembling beats the best single model only when members are decorrelated "
                   "AND mutually complementary (not nested): union > best_single iff rho < 1 with "
                   "non-nested errors (soshiki-genron mesh, gain=(1-rho)(1-p)(1-(1-p)^(n-1)))."),
    }
