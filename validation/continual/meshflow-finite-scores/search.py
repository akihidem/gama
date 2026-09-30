"""Search fixtures: invalid ratings cannot terminate verified escalation."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_mesh_search"
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
        self.text, self.calls = text, 0

    def complete(self, prompt, tier, **kwargs):
        self.calls += 1
        return self.text


class FiniteScoresSearch(unittest.TestCase):
    def test_nan_is_invalid(self):
        self.assertEqual(mesh._normalize_score(float("nan")), 0.0)

    def test_nan_draft_cannot_prevent_a_verified_answer(self):
        draft, answer = Reply("unverified draft"), Reply("checked answer")
        backend = mesh.MeshflowBackend(
            [("draft", draft), ("answer", answer)], mesh=False,
            verify=lambda text: float("nan") if text == draft.text else 1.0)
        self.assertEqual(backend.complete("question", TIER), answer.text)
        self.assertEqual(answer.calls, 1)
        self.assertEqual(backend.last_trace[0]["score"], 0.0)

    def test_booleans_finite_values_and_clamping_are_preserved(self):
        inputs = [True, False, 0.375, -2.0, 3.0]
        self.assertEqual([mesh._normalize_score(x) for x in inputs],
                         [1.0, 0.0, 0.375, 0.0, 1.0])

    def test_finite_numeric_strings_remain_supported(self):
        self.assertEqual(mesh._normalize_score(" 0.125 "), 0.125)

    def test_unconvertible_results_remain_invalid(self):
        self.assertEqual([mesh._normalize_score(x) for x in (None, [], "unknown")],
                         [0.0, 0.0, 0.0])

    def test_verifier_exception_still_allows_escalation(self):
        def verify(text):
            if text == "first":
                raise ValueError("checker could not score this draft")
            return True
        backend = mesh.MeshflowBackend(
            [("first", Reply("first")), ("second", Reply("second"))],
            verify=verify, mesh=False)
        self.assertEqual(backend.complete("question", TIER), "second")
