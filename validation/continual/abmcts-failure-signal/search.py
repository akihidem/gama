"""Search fixtures: infrastructure failure versus genuine model answers."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_abmcts_search"
ROOT = Path.cwd().resolve() / "gama"
SPEC = importlib.util.spec_from_file_location(
    NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PKG = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = PKG
SPEC.loader.exec_module(PKG)
Unavailable = sys.modules[NAME + ".backends"].MeasurementUnavailable
TIER = PKG.ModelTier.LARGE


class Replies(PKG.ModelBackend):
    available = True

    def __init__(self, *items):
        self.items, self.calls = items, 0

    def complete(self, prompt, tier, **kwargs):
        item = self.items[min(self.calls, len(self.items) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class FailureSignalSearch(unittest.TestCase):
    def test_all_generation_errors_signal_unavailable(self):
        worker = Replies(RuntimeError("offline generator"))
        backend = PKG.ABMCTSBackend([worker], budget=3, seed=2, verify=lambda _: 0)
        with self.assertRaises(Unavailable):
            backend.complete("answer", TIER)
        self.assertLessEqual(worker.calls, 3)

    def test_missing_verifier_does_not_hide_backend_failure(self):
        backend = PKG.ABMCTSBackend(
            [Replies(ConnectionError("not connected"))], budget=2, seed=4)
        with self.assertRaises(Unavailable):
            backend.complete("answer", TIER)

    def test_genuine_empty_answers_are_still_answers(self):
        backend = PKG.ABMCTSBackend([Replies("")], budget=2, verify=lambda _: 0)
        self.assertEqual(backend.complete("answer", TIER), "")

    def test_successful_answer_stops_at_verified_success(self):
        worker = Replies("valid")
        backend = PKG.ABMCTSBackend([worker], budget=5, verify=lambda _: 1)
        self.assertEqual(backend.complete("answer", TIER), "valid")
        self.assertEqual(worker.calls, 1)

    def test_later_success_survives_an_earlier_failure(self):
        backend = PKG.ABMCTSBackend(
            [Replies(RuntimeError("temporary outage"), "recovered")],
            budget=2, seed=6, verify=lambda text: float(text == "recovered"))
        self.assertEqual(backend.complete("answer", TIER), "recovered")

    def test_failed_generation_cannot_pass_a_permissive_verifier(self):
        backend = PKG.ABMCTSBackend(
            [Replies(RuntimeError("no response"), "actual answer")],
            budget=2, seed=8, verify=lambda _: 1)
        self.assertEqual(backend.complete("answer", TIER), "actual answer")
