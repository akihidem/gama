"""Confirmation fixtures: infinity, serialized NaN, and unresolved stakes."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_mesh_confirm"
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

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, tier, **kwargs):
        return self.text


class FiniteScoresConfirm(unittest.TestCase):
    def test_positive_infinity_is_not_full_credit(self):
        self.assertEqual(mesh._normalize_score(float("inf")), 0.0)

    def test_string_nan_is_invalid(self):
        self.assertEqual(mesh._normalize_score("NaN"), 0.0)

    def test_nan_cannot_bypass_high_stakes_gate(self):
        backend = mesh.MeshflowBackend(
            [Reply("unverified recommendation")], verify=lambda _: float("nan"),
            mesh=False, stakes=0.9)
        self.assertEqual(backend.complete("decide", TIER), mesh.NEEDS_HUMAN)
        self.assertTrue(backend.last_human_gate)

    def test_finite_mesh_can_still_resolve_complementary_drafts(self):
        backend = mesh.MeshflowBackend(
            [("left", Reply("left")), ("right", Reply("right"))],
            verify=lambda text: 1.0 if text == "left\nright" else 0.2,
            mesh="union", stakes=0.9)
        self.assertEqual(backend.complete("combine", TIER), "left\nright")
        self.assertEqual(backend.last_resolved_by, "mesh")

    def test_float_compatible_finite_rating_is_preserved(self):
        class Rating:
            def __float__(self):
                return 0.8125
        self.assertEqual(mesh._normalize_score(Rating()), 0.8125)

    def test_rejected_conversion_remains_zero(self):
        class Unrated:
            def __float__(self):
                raise ValueError("no rating")
        self.assertEqual(mesh._normalize_score(Unrated()), 0.0)
