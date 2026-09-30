"""Sealed fixtures: stale outcomes, equal scores, and non-error emptiness."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_abmcts_sealed"
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


class FailureSignalSealed(unittest.TestCase):
    def test_provider_os_errors_are_visible_to_the_caller(self):
        backend = PKG.ABMCTSBackend(
            [Replies(BrokenPipeError("provider disconnected"))],
            budget=3, seed=21, verify=lambda _: 0)
        with self.assertRaises(Unavailable):
            backend.complete("respond", TIER)

    def test_failed_call_does_not_keep_an_earlier_winner(self):
        backend = PKG.ABMCTSBackend(
            [Replies("previous success", RuntimeError("now unavailable"))],
            budget=2, seed=23, verify=lambda text: float(bool(text)))
        self.assertEqual(backend.complete("first", TIER), "previous success")
        with self.assertRaises(Unavailable):
            backend.complete("second", TIER)
        self.assertIsNone(backend.last_resolved_by)

    def test_failed_placeholder_cannot_win_a_zero_score_tie(self):
        backend = PKG.ABMCTSBackend(
            [Replies(RuntimeError("no candidate"), "real draft")],
            budget=2, seed=25, verify=lambda _: 0)
        self.assertEqual(backend.complete("respond", TIER), "real draft")

    def test_successful_whitespace_without_errors_remains_valid_output(self):
        backend = PKG.ABMCTSBackend(
            [Replies(" \t")], budget=2, seed=27, verify=lambda _: 0)
        self.assertEqual(backend.complete("respond", TIER), " \t")

    def test_complete_verifier_override_preserves_success(self):
        backend = PKG.ABMCTSBackend(
            [Replies("override answer")], budget=2, seed=29, verify=lambda _: 0)
        self.assertEqual(backend.complete("respond", TIER, verify=lambda _: 1),
                         "override answer")

    def test_nonpassing_answer_remains_available_without_errors(self):
        backend = PKG.ABMCTSBackend(
            [Replies("best effort")], budget=3, seed=31, verify=lambda _: 0.125)
        self.assertEqual(backend.complete("respond", TIER), "best effort")
