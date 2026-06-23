"""Tests for the gama meshflow membrane probe (m=0 vs m>0)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from gama.meshflow import NEEDS_HUMAN, MeshflowBackend
from gama.models import ModelTier

from .cases import CORRECT, WRONG, build_tiers, checker, default_cases
from .run import run_compare


class TestCases(unittest.TestCase):
    def test_tiers_solve_by_rank(self):
        cases = default_cases()
        tiers = build_tiers(cases)
        # easy-0 solved by cheapest tier; hard-0 by none.
        self.assertEqual(tiers[0].complete("easy-0", ModelTier.SMALL), CORRECT)
        self.assertEqual(tiers[0].complete("mid-0", ModelTier.SMALL), WRONG)
        self.assertEqual(tiers[2].complete("mid-0", ModelTier.SMALL), CORRECT)
        self.assertEqual(tiers[2].complete("hard-0", ModelTier.SMALL), WRONG)


class TestMembraneToggle(unittest.TestCase):
    def setUp(self):
        self.cases = default_cases()
        self.tiers = build_tiers(self.cases)

    def test_m_pos_holds_hard_highstakes(self):
        be = MeshflowBackend(self.tiers, verify=checker, stakes_threshold=0.7)
        art = be.complete("hard-0", ModelTier.SMALL, verify=checker, stakes=0.9)
        self.assertEqual(art, NEEDS_HUMAN)
        self.assertTrue(be.last_human_gate)

    def test_m_zero_ships_wrong_answer(self):
        # stakes_threshold = inf -> membrane never fires -> ships best-effort (WRONG).
        be = MeshflowBackend(self.tiers, verify=checker, stakes_threshold=math.inf)
        art = be.complete("hard-0", ModelTier.SMALL, verify=checker, stakes=0.9)
        self.assertNotEqual(art, NEEDS_HUMAN)
        self.assertEqual(checker(art), 0.0)         # a verified-wrong artifact shipped
        self.assertFalse(be.last_human_gate)

    def test_easy_case_identical_under_both_gates(self):
        for thr in (0.7, math.inf):
            be = MeshflowBackend(self.tiers, verify=checker, stakes_threshold=thr)
            art = be.complete("easy-0", ModelTier.SMALL, verify=checker, stakes=0.3)
            self.assertEqual(art, CORRECT)


class TestComparison(unittest.TestCase):
    def test_membrane_eliminates_bad_ships(self):
        with tempfile.TemporaryDirectory() as d:
            comp = run_compare(Path(d))
            self.assertGreater(comp["m=0"]["bad_ship_rate"], 0)     # m=0 ships wrong answers
            self.assertEqual(comp["m>0"]["bad_ship_rate"], 0)       # m>0 holds them
            self.assertEqual(comp["m=0"]["ship_rate"], 1.0)         # m=0 ships everything
            self.assertGreater(comp["m>0"]["escalation_rate"], 0)   # at the cost of escalation
            self.assertEqual(comp["m>0"]["shipped_precision"], 1.0)
            self.assertEqual(comp["verdict"],
                             "membrane eliminates bad ships at the cost of escalation")

    def test_isolation_no_writes_into_gama(self):
        import gama
        pkg = Path(gama.__file__).resolve().parent
        before = {p: p.stat().st_mtime_ns for p in pkg.rglob("*.py")}
        with tempfile.TemporaryDirectory() as d:
            run_compare(Path(d))
        after = {p: p.stat().st_mtime_ns for p in pkg.rglob("*.py")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
