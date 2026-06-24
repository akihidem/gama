import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.benchmark import BenchCase, run_bench
from gama.cli import build_parser
from gama.market import (
    analyze, dominates, escalation_cost, ladder, market_over_records, p_star,
)
from gama.models import ModelTier


class Canned(ModelBackend):
    """Fake backend returning a fixed reply (so bench scores are deterministic)."""

    available = True

    def __init__(self, reply):
        self.reply = reply
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        return self.reply


# A suite with a capability gap: 'easy' cases need an "E", 'hard' cases need an "H".
GAP_SUITE = [
    BenchCase("easy1", "qa", "p", lambda o: 1.0 if "E" in o else 0.0),
    BenchCase("easy2", "qa", "p", lambda o: 1.0 if "E" in o else 0.0),
    BenchCase("hard1", "qa", "p", lambda o: 1.0 if "H" in o else 0.0),
    BenchCase("hard2", "qa", "p", lambda o: 1.0 if "H" in o else 0.0),
]


class TestAnalytic(unittest.TestCase):
    """The closed-form market model (soshiki-genron model/market.py)."""

    def test_escalation_cost(self):
        # gemma x haiku calibration (§5 regime ③): w=0.2, s=1, p=0.8888
        self.assertAlmostEqual(escalation_cost(0.2, 1.0, 0.8888), 0.3112, places=4)

    def test_p_star_is_cost_ratio(self):
        self.assertAlmostEqual(p_star(0.2, 1.0), 0.2, places=4)
        self.assertAlmostEqual(p_star(1.0, 10.0), 0.1, places=4)

    def test_dominates_on_large_gap(self):
        self.assertTrue(dominates(0.2, 1.0, 0.8888))     # p=0.889 > w/s=0.2

    def test_homogeneous_cannot_dominate(self):
        self.assertFalse(dominates(1.0, 1.0, 1.0))       # p* = 1 -> need p > 1 (impossible)

    def test_dominance_is_strict_at_threshold(self):
        self.assertFalse(dominates(0.5, 1.0, 0.5))       # p == w/s : not strictly cheaper
        self.assertTrue(dominates(0.5, 1.0, 0.51))       # p just over the bar

    def test_ladder_three_tier(self):
        r = ladder([(0.2, 0.8888), (1.0, 1.0), (15.0, 1.0)])
        self.assertAlmostEqual(r["cost"], 0.3112, places=4)
        self.assertAlmostEqual(r["correctness"], 1.0, places=4)

    def test_ladder_partial_correctness(self):
        r = ladder([(1.0, 0.5), (2.0, 0.5)])             # cost 1 + 2*0.5=2 ; corr 1-0.25=0.75
        self.assertAlmostEqual(r["cost"], 2.0, places=4)
        self.assertAlmostEqual(r["correctness"], 0.75, places=4)


class TestEmpiricalMarket(unittest.TestCase):
    def _gap_records(self):
        return run_bench({"weak": Canned("E"), "strong": Canned("EH")},
                         suite=GAP_SUITE, tier=ModelTier.SMALL)

    def test_market_routes_cheap_then_escalates(self):
        m = market_over_records(self._gap_records(), ["weak", "strong"], costs=[1, 3])
        routed = {d["case"]: d["routed_to"] for d in m["ladders"]}
        self.assertEqual(routed["easy1"], "weak")        # cheap tier clears the bar
        self.assertEqual(routed["hard1"], "strong")      # escalate only where needed
        self.assertEqual(m["market_cost"], 10.0)         # 1 + 1 + 4 + 4
        self.assertEqual(m["pass_rate"], 1.0)

    def test_market_dominates_flat_strong_on_a_gap(self):
        r = analyze(self._gap_records(), ["weak", "strong"], costs=[1, 3])
        self.assertTrue(r["market_dominates_flat_strong"])
        self.assertEqual(r["market"]["market_cost"], 10.0)
        self.assertEqual(r["flat_strong"]["backend"], "strong")
        self.assertEqual(r["flat_strong"]["cost"], 12.0)        # 3 x 4 cases
        self.assertEqual(r["flats"][0]["pass_rate"], 0.5)       # weak solves half (p_weak)
        self.assertTrue(r["analytic"]["dominates_2tier"])
        self.assertAlmostEqual(r["analytic"]["p_star"], 1 / 3, places=4)

    def test_homogeneous_equal_cost_does_not_dominate(self):
        recs = run_bench({"a": Canned("EH"), "b": Canned("EH")},
                         suite=GAP_SUITE, tier=ModelTier.SMALL)
        r = analyze(recs, ["a", "b"], costs=[1, 1])
        self.assertFalse(r["market_dominates_flat_strong"])     # same cost, same capability
        self.assertFalse(r["analytic"]["dominates_2tier"])
        self.assertAlmostEqual(r["analytic"]["p_star"], 1.0, places=4)

    def test_costs_length_must_match_tiers(self):
        with self.assertRaises(ValueError):
            analyze(self._gap_records(), ["weak", "strong"], costs=[1, 2, 3])

    def test_needs_at_least_two_tiers(self):
        with self.assertRaises(ValueError):
            analyze(self._gap_records(), ["weak"])

    def test_default_costs_are_one_two_three(self):
        m = market_over_records(self._gap_records(), ["weak", "strong"])  # default 1,2
        self.assertEqual(m["costs"], [1.0, 2.0])


class TestMarketCli(unittest.TestCase):
    def test_subcommand_parses(self):
        a = build_parser().parse_args(["market", "--backends", "weak,strong", "--costs", "1,3"])
        self.assertEqual(a.backends, "weak,strong")
        self.assertEqual(a.costs, "1,3")
        self.assertEqual(a.func.__name__, "cmd_market")

    def test_default_suite_is_hard(self):
        self.assertEqual(build_parser().parse_args(["market"]).suite, "hard")

    def test_pass_score_is_float(self):
        self.assertEqual(build_parser().parse_args(["market", "--pass-score", "0.5"]).pass_score, 0.5)


if __name__ == "__main__":
    unittest.main()
