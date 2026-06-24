import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.benchmark import BenchCase, run_bench
from gama.cli import build_parser
from gama.decorrelation import (
    analyze, failure_correlation, ignites, mesh_correctness, mesh_gain,
    solve_vectors_from_records, union_solve,
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

    def test_complementary_members_ignite(self):
        r = analyze(self._recs({"A": Canned("A"), "B": Canned("B")}, COMPLEMENT_SUITE), ["A", "B"])
        self.assertEqual(r["union"], 1.0)
        self.assertEqual(r["best_single"], 0.5)
        self.assertEqual(r["mesh_gain"], 0.5)
        self.assertTrue(r["ignites"])
        self.assertLess(r["failure_rho"], 0.0)               # anti-correlated failures

    def test_nested_members_do_not_ignite(self):
        r = analyze(self._recs({"A": Canned("X"), "B": Canned("XYZ")}, NESTED_SUITE), ["A", "B"])
        self.assertEqual(r["best_member"], "B")
        self.assertEqual(r["union"], r["best_single"])       # B nests A
        self.assertEqual(r["mesh_gain"], 0.0)
        self.assertFalse(r["ignites"])                       # rho<1 yet nested -> no gain
        self.assertLess(r["failure_rho"], 1.0)

    def test_identical_members_rho_one(self):
        r = analyze(self._recs({"A": Canned("X"), "B": Canned("X")}, NESTED_SUITE), ["A", "B"])
        self.assertEqual(r["mesh_gain"], 0.0)
        self.assertFalse(r["ignites"])
        self.assertAlmostEqual(r["failure_rho"], 1.0, places=4)   # comonotone failures

    def test_needs_two_members(self):
        with self.assertRaises(ValueError):
            analyze(self._recs({"A": Canned("X")}, NESTED_SUITE), ["A"])

    def test_solve_vectors_alignment(self):
        recs = self._recs({"A": Canned("A"), "B": Canned("B")}, COMPLEMENT_SUITE)
        self.assertEqual(solve_vectors_from_records(recs, ["A", "B"]), [[1, 1, 0, 0], [0, 0, 1, 1]])


class TestMeshCli(unittest.TestCase):
    def test_subcommand_parses(self):
        a = build_parser().parse_args(["mesh", "--backends", "a,b,c"])
        self.assertEqual(a.backends, "a,b,c")
        self.assertEqual(a.func.__name__, "cmd_mesh")

    def test_default_suite_is_hard(self):
        self.assertEqual(build_parser().parse_args(["mesh"]).suite, "hard")


if __name__ == "__main__":
    unittest.main()
