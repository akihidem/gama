"""Sealed fixtures: synthesis, conversion overflow, and a shared consumer."""
from decimal import Decimal
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_mesh_sealed"
ROOT = Path.cwd().resolve() / "gama"
SPEC = importlib.util.spec_from_file_location(
    NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PKG = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = PKG
SPEC.loader.exec_module(PKG)
mesh = sys.modules[NAME + ".meshflow"]
TIER = PKG.ModelTier.LARGE


class Reply(PKG.ModelBackend):
    available = True

    def __init__(self, *texts):
        self.texts = iter(texts)

    def complete(self, prompt, tier, **kwargs):
        return next(self.texts)


class FiniteScoresSealed(unittest.TestCase):
    def test_unrated_synthesis_cannot_resolve_high_stakes(self):
        backend = mesh.MeshflowBackend(
            [("a", Reply("a")), ("b", Reply("b"))],
            aggregator=Reply("merged"), mesh="synthesize", stakes=0.95,
            verify=lambda text: float("nan") if text == "merged" else 0.0)
        self.assertEqual(backend.complete("decide", TIER), mesh.NEEDS_HUMAN)
        self.assertEqual(backend.last_trace[-1]["score"], 0.0)

    def test_negative_infinity_stays_invalid(self):
        self.assertEqual(mesh._normalize_score("-Infinity"), 0.0)

    def test_conversion_overflow_is_an_invalid_rating(self):
        self.assertEqual(mesh._normalize_score(10 ** 5000), 0.0)

    def test_finite_decimal_can_meet_a_fractional_threshold(self):
        backend = mesh.MeshflowBackend(
            [Reply("rated")], verify=lambda _: Decimal("0.625"),
            pass_score=0.625, mesh=False, stakes=0.99)
        self.assertEqual(backend.complete("decide", TIER), "rated")
        self.assertFalse(backend.last_human_gate)

    def test_no_verifier_preserves_best_effort(self):
        backend = mesh.MeshflowBackend(
            [("cheap", Reply("cheap")), ("strong", Reply("strong"))],
            verify=None, mesh=False)
        self.assertEqual(backend.complete("question", TIER), "strong")

    def test_abmcts_reuses_safe_normalization(self):
        backend = PKG.ABMCTSBackend(
            [Reply("unrated", "verified")], budget=2, seed=11,
            verify=lambda text: float("nan") if text == "unrated" else 1.0)
        self.assertEqual(backend.complete("question", TIER), "verified")
