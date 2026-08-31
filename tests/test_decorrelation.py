import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.benchmark import BenchCase, run_bench
from gama.cli import build_parser
from gama.decorrelation import (
    analyze, clopper_pearson, cofailure, cofailure_by_class, effective_votes,
    failure_correlation, ignites, mesh_correctness, mesh_gain, pairwise_cofailure,
    solve_vectors_from_records, union_solve, verdict_from_counts,
)
from gama.models import ModelTier


class Canned(ModelBackend):
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        return self.reply


# Complementary: A solves c1,c2 ; B solves c3,c4 (disjoint -> union covers all).
COMPLEMENT_SUITE = [
    BenchCase("c1", "qa", "p", lambda o: 1.0 if "A" in o else 0.0),
    BenchCase("c2", "qa", "p", lambda o: 1.0 if "A" in o else 0.0),
    BenchCase("c3", "qa", "p", lambda o: 1.0 if "B" in o else 0.0),
    BenchCase("c4", "qa", "p", lambda o: 1.0 if "B" in o else 0.0),
]
# Same complementary split, but with enough cases that the union's gain clears beta's
# interval (4 cases can never certify anything: with k=0 the 95% upper bound is 1-0.025^(1/4)=0.60).
WIDE_COMPLEMENT_SUITE = (
    [BenchCase(f"a{i}", "qa", "p", lambda o: 1.0 if "A" in o else 0.0) for i in range(20)]
    + [BenchCase(f"b{i}", "qa", "p", lambda o: 1.0 if "B" in o else 0.0) for i in range(20)]
)
# Nesting probe: each case needs a distinct marker.
NESTED_SUITE = [
    BenchCase("n1", "qa", "p", lambda o: 1.0 if "X" in o else 0.0),
    BenchCase("n2", "qa", "p", lambda o: 1.0 if "Y" in o else 0.0),
    BenchCase("n3", "qa", "p", lambda o: 1.0 if "Z" in o else 0.0),
    BenchCase("n4", "qa", "p", lambda o: 1.0 if "W" in o else 0.0),
]


class TestAnalytic(unittest.TestCase):
    """The mesh ignition law (soshiki-genron model/mesh.py)."""

    def test_gain_independent_two(self):
        self.assertAlmostEqual(mesh_gain(0.5, 0.0, 2), 0.25, places=6)

    def test_gain_zero_when_comonotone(self):
        self.assertEqual(mesh_gain(0.5, 1.0, 2), 0.0)        # rho=1 -> nested/common hard core

    def test_gain_rises_with_n(self):
        self.assertAlmostEqual(mesh_gain(0.5, 0.0, 3), 0.375, places=6)

    def test_correctness_union(self):
        self.assertAlmostEqual(mesh_correctness(0.5, 0.0, 2), 0.75, places=6)
        self.assertAlmostEqual(mesh_correctness(0.5, 1.0, 2), 0.5, places=6)

    def test_ignites_iff_decorrelated(self):
        self.assertTrue(ignites(0.5, 0.0, 2))
        self.assertFalse(ignites(0.5, 1.0, 2))

    def test_no_ignition_at_p_extremes(self):
        self.assertFalse(ignites(0.0, 0.0, 2))               # nobody solves
        self.assertFalse(ignites(1.0, 0.0, 2))               # everybody solves -> no gain

    def test_phi_self_is_one(self):
        self.assertAlmostEqual(failure_correlation([[1, 0, 0], [1, 0, 0]]), 1.0, places=4)

    def test_union_solve(self):
        self.assertAlmostEqual(union_solve([[1, 0, 0, 0], [0, 1, 0, 0]]), 0.5, places=4)


class TestEmpirical(unittest.TestCase):
    def _recs(self, backends, suite):
        return run_bench(backends, suite=suite, tier=ModelTier.SMALL)

    def test_complementary_members_gain_but_four_cases_cannot_certify(self):
        r = analyze(self._recs({"A": Canned("A"), "B": Canned("B")}, COMPLEMENT_SUITE), ["A", "B"])
        self.assertEqual(r["union"], 1.0)
        self.assertEqual(r["best_single"], 0.5)
        self.assertEqual(r["mesh_gain"], 0.5)
        self.assertEqual((r["cofailure_k"], r["n_cases"], r["cofailure_beta"]), (0, 4, 0.0))
        self.assertEqual(r["ceiling"], 1.0)
        # k=0 of 4: beta's 95% upper bound is 0.60, so (1-0.60)-0.5 < 0 -> the gain is real in
        # this sample but not separable from a fluke. Say so instead of "it fired".
        self.assertEqual(r["verdict"], "undetermined")
        self.assertFalse(r["ignites"])
        self.assertLess(r["gain_bounds"][0], 0.0)
        # k=0 -> beta_lo=0, and best 2/4 has lower bound 0.0676 -> 1 - 0 - 0.0676
        self.assertAlmostEqual(r["gain_upper_bound"], 1.0 - r["best_single_interval"][0], places=6)
        self.assertLess(r["failure_rho"], 0.0)               # anti-correlated failures (secondary)

    def test_complementary_members_certify_with_enough_cases(self):
        r = analyze(self._recs({"A": Canned("A"), "B": Canned("B")}, WIDE_COMPLEMENT_SUITE),
                    ["A", "B"])
        self.assertEqual((r["cofailure_k"], r["n_cases"]), (0, 40))
        self.assertEqual(r["mesh_gain"], 0.5)
        # union's lower bound (1-0.088) beats the best member's upper bound (20/40 -> 0.656)
        self.assertGreater(r["gain_bounds"][0], 0.0)
        self.assertEqual(r["verdict"], "certified")
        self.assertTrue(r["ignites"])

    def test_nested_members_do_not_ignite(self):
        r = analyze(self._recs({"A": Canned("X"), "B": Canned("XYZ")}, NESTED_SUITE), ["A", "B"])
        self.assertEqual(r["best_member"], "B")
        self.assertEqual(r["union"], r["best_single"])       # B nests A
        self.assertEqual(r["mesh_gain"], 0.0)
        self.assertEqual(r["verdict"], "dead")               # rho<1 yet nested -> no gain
        self.assertFalse(r["ignites"])
        self.assertEqual(r["cofailure_beta"], round(1.0 - r["union"], 4))   # union == 1 - beta
        self.assertLess(r["failure_rho"], 1.0)

    def test_identical_members_rho_one(self):
        r = analyze(self._recs({"A": Canned("X"), "B": Canned("X")}, NESTED_SUITE), ["A", "B"])
        self.assertEqual(r["mesh_gain"], 0.0)
        self.assertEqual(r["verdict"], "dead")
        self.assertFalse(r["ignites"])
        self.assertAlmostEqual(r["failure_rho"], 1.0, places=4)   # comonotone failures

    def test_needs_two_members(self):
        with self.assertRaises(ValueError):
            analyze(self._recs({"A": Canned("X")}, NESTED_SUITE), ["A"])

    def test_solve_vectors_alignment(self):
        recs = self._recs({"A": Canned("A"), "B": Canned("B")}, COMPLEMENT_SUITE)
        self.assertEqual(solve_vectors_from_records(recs, ["A", "B"]), [[1, 1, 0, 0], [0, 0, 1, 1]])


class TestCofailure(unittest.TestCase):
    """β is the primary quantity: counted directly, with an exact interval, never inferred from rho."""

    def test_cofailure_counts_cases_every_member_failed(self):
        self.assertEqual(cofailure([[1, 0, 0, 0], [0, 1, 0, 0]]), (2, 4))
        self.assertEqual(cofailure([[1, 1], [0, 1]]), (0, 2))

    def test_clopper_pearson_closed_forms(self):
        # k=0 and k=n have closed forms; the bisection must land on them.
        lo, hi = clopper_pearson(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** (1 / 20), places=5)
        lo, hi = clopper_pearson(20, 20)
        self.assertAlmostEqual(lo, 0.025 ** (1 / 20), places=5)
        self.assertEqual(hi, 1.0)

    def test_clopper_pearson_matches_published_intervals(self):
        # Chen 2026 (arXiv:2606.27288) Table 2 reports exact CP 95% intervals for its all-wrong
        # counts: 16/480 -> [0.019, 0.054], 6/200 -> [0.011, 0.064]. Same numbers, stdlib-only.
        lo, hi = clopper_pearson(16, 480)
        self.assertAlmostEqual(lo, 0.019, delta=0.0006)
        self.assertAlmostEqual(hi, 0.054, delta=0.0006)
        lo, hi = clopper_pearson(6, 200)
        self.assertAlmostEqual(lo, 0.011, delta=0.0006)
        self.assertAlmostEqual(hi, 0.064, delta=0.0006)

    def test_clopper_pearson_contains_point_and_narrows_with_n(self):
        for k, n in [(1, 10), (5, 50), (17, 330)]:
            lo, hi = clopper_pearson(k, n)
            self.assertLessEqual(lo, k / n)
            self.assertGreaterEqual(hi, k / n)
        self.assertLess(clopper_pearson(10, 1000)[1] - clopper_pearson(10, 1000)[0],
                        clopper_pearson(1, 100)[1] - clopper_pearson(1, 100)[0])

    def test_clopper_pearson_matches_reference_table(self):
        # Standard exact-interval values (95%): 1/10, 5/50, 50/100.
        for k, n, lo_ref, hi_ref in [(1, 10, 0.0025, 0.4450), (5, 50, 0.0333, 0.2181),
                                     (50, 100, 0.3983, 0.6017)]:
            lo, hi = clopper_pearson(k, n)
            self.assertAlmostEqual(lo, lo_ref, delta=6e-4)
            self.assertAlmostEqual(hi, hi_ref, delta=6e-4)

    def test_more_members_make_certification_harder(self):
        # Same best member and same union; adding members that solve nothing new widens the
        # best member's Bonferroni interval, so the gain lower bound can only go down.
        n = 40
        a = [1] * 20 + [0] * 20
        b = [0] * 20 + [1] * 20
        dup = [0] * 40
        recs = []
        for name, vec in (("A", a), ("B", b), ("D1", dup), ("D2", dup)):
            recs += [{"case_id": f"c{i}", "backend": name, "score": float(vec[i])} for i in range(n)]
        two = analyze(recs, ["A", "B"])
        four = analyze(recs, ["A", "B", "D1", "D2"])
        self.assertLess(four["best_single_interval"][0], two["best_single_interval"][0])
        self.assertGreater(four["best_single_interval"][1], two["best_single_interval"][1])
        self.assertLess(four["gain_bounds"][0], two["gain_bounds"][0])
        self.assertAlmostEqual(two["best_interval_confidence"], 0.975, places=6)
        self.assertAlmostEqual(four["best_interval_confidence"], 0.9875, places=6)

    def test_failure_correlation_undefined_pairs_are_excluded(self):
        # A member that fails nothing has no failure variance: its pairs are undefined, not 1.0.
        self.assertIsNone(failure_correlation([[1, 1, 1], [1, 1, 1]]))
        self.assertIsNone(failure_correlation([[1, 1, 1], [1, 0, 0]]))
        self.assertAlmostEqual(failure_correlation([[1, 1, 1], [1, 0, 0], [1, 0, 0]]), 1.0, places=4)

    def test_clopper_pearson_stable_at_large_n(self):
        # The plain log-space sum must stay monotone and tight at bench-sized n.
        lo, hi = clopper_pearson(100, 10_000)
        self.assertLess(lo, 0.01)
        self.assertGreater(hi, 0.01)
        self.assertLess(hi - lo, 0.005)
        lo0, hi0 = clopper_pearson(0, 10_000)
        self.assertEqual(lo0, 0.0)
        self.assertAlmostEqual(hi0, 1 - 0.025 ** (1 / 10_000), places=7)

    def test_cofailure_rejects_empty_or_ragged(self):
        with self.assertRaises(ValueError):
            cofailure([])
        with self.assertRaises(ValueError):
            cofailure([[1, 0], [1]])
        with self.assertRaises(ValueError):
            union_solve([[1, 0], [1, 0, 1]])

    def test_clopper_pearson_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            clopper_pearson(5, 4)
        with self.assertRaises(ValueError):
            clopper_pearson(0, 0)
        with self.assertRaises(ValueError):
            clopper_pearson(1, 4, confidence=1.0)

    def test_pairwise_rho_cannot_identify_beta(self):
        # Three members, identical marginals (each fails half the cases) and identical pairwise
        # failure correlation (0: pairwise independent), yet beta differs: even-parity failures
        # never all fail together, independent coins do 1/8 of the time. This is why rho was
        # demoted — it is exact for pairs and blind to the joint tail from m=3 on.
        even_parity = [[1, 1, 0, 0, 1, 1, 0, 0],    # failure patterns 000,011,101,110 twice each
                       [1, 0, 1, 0, 1, 0, 1, 0],
                       [1, 0, 0, 1, 1, 0, 0, 1]]
        independent = [[1, 1, 1, 1, 0, 0, 0, 0],    # all 8 failure patterns once
                       [1, 1, 0, 0, 1, 1, 0, 0],
                       [1, 0, 1, 0, 1, 0, 1, 0]]
        self.assertAlmostEqual(failure_correlation(even_parity), 0.0, places=4)
        self.assertAlmostEqual(failure_correlation(independent), 0.0, places=4)
        self.assertEqual(cofailure(even_parity), (0, 8))
        self.assertEqual(cofailure(independent), (1, 8))

    def test_verdict_uses_case_counts_not_rounded_rates(self):
        # 30,000 cases: A solves 15,000, B solves the same 15,000 plus one more. The one-case gain
        # (0.00003) rounds to a 0.0 rate and A and B round to the same 0.5 — but B is the best
        # member by count and the mesh is not "dead" (a case exists where the union beats it).
        n = 30_000
        a = [1] * 15_000 + [0] * 15_000
        b = [1] * 15_000 + [1] + [0] * (n - 15_001)
        c = [0] * 15_001 + [1] + [0] * (n - 15_002)   # rescues one case B fails
        recs = []
        for name, vec in (("A", a), ("B", b), ("C", c)):
            recs += [{"case_id": f"c{i}", "backend": name, "score": float(vec[i])} for i in range(n)]
        r = analyze(recs, ["A", "B", "C"])
        self.assertEqual(r["best_member"], "B")
        self.assertEqual(r["gain_cases"], 1)
        self.assertEqual(r["mesh_gain"], 0.0)                 # the rounded rate alone would mislead
        self.assertNotEqual(r["verdict"], "dead")

    def test_best_member_tie_goes_to_first_listed(self):
        recs = [{"case_id": f"c{i}", "backend": m, "score": 1.0 if i < 2 else 0.0}
                for m in ("A", "B") for i in range(4)]
        self.assertEqual(analyze(recs, ["A", "B"])["best_member"], "A")
        self.assertEqual(analyze(recs, ["B", "A"])["best_member"], "B")

    def test_verdict_from_counts_is_the_rule_analyze_uses(self):
        # Same counts -> same verdict dict as analyze() (minus the member-specific keys).
        r = analyze(self._recs_wide(), ["A", "B"])
        v = verdict_from_counts(r["cofailure_k"], r["n_cases"], r["best_solved"], members=2)
        for key in ("verdict", "gain_bounds", "beta_interval", "best_single_interval",
                    "ceiling", "mesh_gain", "gain_cases", "ignites"):
            self.assertEqual(v[key], r[key], key)
        self.assertEqual(v["verdict"], "certified")

    def test_verdict_from_counts_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            verdict_from_counts(5, 4, 0, members=2)   # k > n
        with self.assertRaises(ValueError):
            verdict_from_counts(2, 10, 9, members=2)  # best solved a co-failure case
        with self.assertRaises(ValueError):
            verdict_from_counts(0, 10, 5, members=0)
        with self.assertRaises(ValueError):
            verdict_from_counts(1.0, 10, 5, members=2)      # counts are ints, not rates
        with self.assertRaises(ValueError):
            verdict_from_counts(True, 10, 5, members=2)
        with self.assertRaises(ValueError):
            verdict_from_counts(0, 10, 5, members=2, confidence=1.0)

    def _recs_wide(self):
        recs = []
        for name, on in (("A", "A"), ("B", "B")):
            recs += [{"case_id": c.case_id, "backend": name,
                      "score": c.checker(on)} for c in WIDE_COMPLEMENT_SUITE]
        return recs

    def test_one_case_fluke_is_undetermined_not_ignition(self):
        # The soshiki-genron shape: N=24, best member 12/24, union 13/24 (+0.042 = one case).
        # k=11 all-fail of 24 -> beta's 95% upper bound ~0.66 -> (1-0.66)-0.5 < 0.
        best = [1] * 12 + [0] * 12
        other = [0] * 12 + [1] + [0] * 11
        recs = [{"case_id": f"c{i}", "backend": "best", "score": float(best[i])} for i in range(24)]
        recs += [{"case_id": f"c{i}", "backend": "other", "score": float(other[i])} for i in range(24)]
        r = analyze(recs, ["best", "other"])
        self.assertEqual(r["mesh_gain"], round(1 / 24, 4))
        self.assertEqual((r["cofailure_k"], r["n_cases"]), (11, 24))
        self.assertEqual(r["verdict"], "undetermined")
        self.assertFalse(r["ignites"])


class TestPairwiseAdvisory(unittest.TestCase):
    """SECONDARY 表示(pairwise 件数と Kish n_eff)の性質。verdict はこれらを読まない。"""

    def test_pairwise_counts_are_exact(self):
        # A は case0,1 で失敗 / B は case1,2 で失敗 / C は全勝
        vecs = [[0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
        pw = pairwise_cofailure(vecs, ["A", "B", "C"])
        by_pair = {tuple(d["pair"]): d["cofailure_k"] for d in pw}
        self.assertEqual(by_pair[("A", "B")], 1)   # case1 のみ両落ち
        self.assertEqual(by_pair[("A", "C")], 0)   # C は落ちない
        self.assertEqual(by_pair[("B", "C")], 0)
        self.assertTrue(all(d["n_cases"] == 4 for d in pw))

    def test_pairwise_refuses_mismatched_members(self):
        with self.assertRaises(ValueError):
            pairwise_cofailure([[0, 1], [1, 0]], ["only-one"])

    def test_n_eff_closed_forms(self):
        # φ=1(同一ベクトル・非定数) → n_eff = m/m = 1.0
        v = [0, 1, 0, 1]
        self.assertEqual(effective_votes([v, list(v)]), 1.0)
        # φ=0 は「互いに素な失敗」ではない(それは反相関で n_eff>m になる)。
        # cov=0 の構成: A の失敗 {c0,c1} / B の失敗 {c0,c2} → φ=0 → n_eff=2.0
        self.assertEqual(effective_votes([[0, 0, 1, 1], [0, 1, 0, 1]]), 2.0)
        # 互いに素な失敗は反相関(φ=-1/3)なので独立 2 票より価値が高い: 3.0
        self.assertEqual(effective_votes([[0, 1, 1, 1], [1, 0, 1, 1]]), 3.0)

    def test_n_eff_matches_published_nine_judges_value(self):
        # 外部アンカー: 2605.29800 Table 2 は m=9, φ̄=0.391 で n_eff=2.18。
        # failure_correlation を経由できない(実ベクトル非公開)ので、式の段だけを
        # 公表値と突き合わせる: m/(1+(m-1)*0.391) = 2.18(2 桁丸め)。
        m, phi_bar = 9, 0.391
        self.assertAlmostEqual(round(m / (1 + (m - 1) * phi_bar), 2), 2.18)

    def test_n_eff_none_when_undefined(self):
        # 定数メンバーだけ → φ 未定義 → None(数を出すと診断に読まれる)
        self.assertIsNone(effective_votes([[1, 1, 1], [1, 1, 1]]))
        # 1 人では独立票の概念が立たない
        self.assertIsNone(effective_votes([[0, 1, 0]]))

    def test_n_eff_none_on_negative_denominator(self):
        # 完全反相関(φ=-1, m=2)は分母 0 → None
        self.assertIsNone(effective_votes([[0, 1], [1, 0]]))

    def test_analyze_carries_advisory_fields(self):
        recs = ([{"backend": "a", "case_id": f"c{i}", "score": 1.0 if i < 3 else 0.0,
                  "task_type": "qa"} for i in range(4)]
                + [{"backend": "b", "case_id": f"c{i}", "score": 1.0 if i in (0, 3) else 0.0,
                    "task_type": "qa"} for i in range(4)])
        out = analyze(recs, ["a", "b"])
        self.assertIn("pairwise_cofailure", out)
        self.assertIn("effective_votes", out)
        self.assertEqual(out["pairwise_cofailure"][0]["pair"], ["a", "b"])


class TestByClass(unittest.TestCase):
    """L0-5b: per-class beta. The blind-spot flag is certified (interval), never a point guess."""

    def _recs(self):
        # class "qa": co-failure 0 of 12 -> the only way the flag stays off (lower bound 0).
        # class "code": EVERY member fails all 12 -> beta_lo = 0.025^(1/12) ~ 0.735 -> blind spot.
        # class "math": 1 co-failure of 12 -> CP lower bound is small but POSITIVE -> certified
        #   (a case that defeats every member was observed); magnitude is read from the interval.
        recs = []
        for i in range(12):
            recs += [{"case_id": f"q{i}", "backend": "A", "score": 1.0, "task_type": "qa"},
                     {"case_id": f"q{i}", "backend": "B", "score": 0.0, "task_type": "qa"},
                     {"case_id": f"c{i}", "backend": "A", "score": 0.0, "task_type": "code"},
                     {"case_id": f"c{i}", "backend": "B", "score": 0.0, "task_type": "code"},
                     {"case_id": f"m{i}", "backend": "A", "score": 0.0 if i == 0 else 1.0,
                      "task_type": "math"},
                     {"case_id": f"m{i}", "backend": "B", "score": 0.0, "task_type": "math"}]
        return recs

    def test_blind_spot_is_certified_not_pointwise(self):
        by = cofailure_by_class(self._recs(), ["A", "B"])
        self.assertEqual(set(by), {"qa", "code", "math"})
        self.assertEqual((by["qa"]["cofailure_k"], by["qa"]["blind_spot"]), (0, False))
        code = by["code"]
        self.assertEqual((code["cofailure_k"], code["n_cases"]), (12, 12))
        self.assertTrue(code["blind_spot"] and code["beta_interval"][0] > 0.7)
        self.assertEqual(code["ceiling"], 0.0)
        # 1 co-failure: the flag is on (a co-failure was observed, so mass 0 is excluded),
        # but the interval shows it is small — the reader judges magnitude, the flag only
        # certifies existence. Only k=0 leaves the class unflagged.
        math_c = by["math"]
        self.assertEqual(math_c["cofailure_k"], 1)
        self.assertTrue(math_c["blind_spot"])
        self.assertLess(math_c["beta_interval"][0], 0.05)
        self.assertGreater(math_c["beta_interval"][0], 0.0)
        self.assertEqual(by["qa"]["per_member_solved"], {"A": 12, "B": 0})

    def test_analyze_carries_classes_only_when_asked(self):
        r = analyze(self._recs(), ["A", "B"])
        self.assertNotIn("classes", r)                      # API compat: opt-in
        r2 = analyze(self._recs(), ["A", "B"], by_class=True)
        self.assertEqual(r2["classes"]["code"]["blind_spot"], True)
        # the per-class counts must reconcile with the global ones
        self.assertEqual(sum(v["cofailure_k"] for v in r2["classes"].values()), r2["cofailure_k"])
        self.assertEqual(sum(v["n_cases"] for v in r2["classes"].values()), r2["n_cases"])
        self.assertEqual(r2["unclassified_cases"], 0)

    def test_cases_without_task_type_are_counted_as_unclassified(self):
        recs = [{"case_id": "x", "backend": "A", "score": 0.0},
                {"case_id": "x", "backend": "B", "score": 0.0}]
        self.assertEqual(cofailure_by_class(recs, ["A", "B"]), {})
        r = analyze(recs + [{"case_id": "y", "backend": "A", "score": 1.0, "task_type": "qa"},
                            {"case_id": "y", "backend": "B", "score": 0.0, "task_type": "qa"}],
                    ["A", "B"], by_class=True)
        self.assertEqual(r["unclassified_cases"], 1)
        self.assertEqual(sum(v["n_cases"] for v in r["classes"].values()) + 1, r["n_cases"])

    def test_conflicting_task_types_for_one_case_are_refused_only_in_by_class(self):
        recs = [{"case_id": "x", "backend": "A", "score": 0.0, "task_type": "qa"},
                {"case_id": "x", "backend": "B", "score": 0.0, "task_type": "code"}]
        with self.assertRaises(ValueError):
            cofailure_by_class(recs, ["A", "B"])
        with self.assertRaises(ValueError):
            cofailure_by_class([{"case_id": "x", "backend": "A", "score": 0.0, "task_type": 7}],
                               ["A"])
        # スコアだけ見る経路（by_class 無し）では task_type は補助情報: 落とさない
        self.assertEqual(solve_vectors_from_records(recs, ["A", "B"]), [[0], [0]])
        r = analyze(recs + [{"case_id": "y", "backend": "A", "score": 1.0},
                            {"case_id": "y", "backend": "B", "score": 0.0}], ["A", "B"])
        self.assertEqual(r["verdict"], "dead")              # 例外にならず通常判定（union==best）

    def test_ceiling_certified_bounds_the_point_estimate(self):
        by = cofailure_by_class(self._recs(), ["A", "B"])
        for v in by.values():
            self.assertGreaterEqual(v["ceiling_certified"], v["ceiling"])


class TestMeshCli(unittest.TestCase):
    def test_subcommand_parses(self):
        a = build_parser().parse_args(["mesh", "--backends", "a,b,c"])
        self.assertEqual(a.backends, "a,b,c")
        self.assertEqual(a.func.__name__, "cmd_mesh")

    def test_default_suite_is_hard(self):
        self.assertEqual(build_parser().parse_args(["mesh"]).suite, "hard")


if __name__ == "__main__":
    unittest.main()
