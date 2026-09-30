"""Confirmation fixtures: multiple failing arms and partial valid results."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_abmcts_confirm"
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


class FailureSignalConfirm(unittest.TestCase):
    def test_all_selected_arms_failing_is_unavailable(self):
        backend = PKG.ABMCTSBackend(
            [("a", Replies(OSError("arm a down"))),
             ("b", Replies(RuntimeError("arm b down")))],
            budget=4, seed=7, verify=lambda _: 0)
        with self.assertRaises(Unavailable):
            backend.complete("solve", TIER)

    def test_timeout_is_not_an_empty_answer(self):
        backend = PKG.ABMCTSBackend(
            [Replies(TimeoutError("generation deadline"))],
            budget=1, seed=9, verify=lambda _: 0)
        with self.assertRaises(Unavailable):
            backend.complete("solve", TIER)

    def test_failure_followed_only_by_empty_answers_is_unavailable(self):
        backend = PKG.ABMCTSBackend(
            [Replies(RuntimeError("outage"), "")],
            budget=3, seed=12, verify=lambda _: 0)
        with self.assertRaises(Unavailable):
            backend.complete("solve", TIER)

    def test_partial_answer_survives_later_failures(self):
        backend = PKG.ABMCTSBackend(
            [Replies("partial solution", RuntimeError("later outage"))],
            budget=3, seed=14, verify=lambda text: 0.4 if text else 0.0)
        self.assertEqual(backend.complete("solve", TIER), "partial solution")

    def test_zero_text_is_a_nonempty_success(self):
        backend = PKG.ABMCTSBackend(
            [Replies("0")], budget=2, seed=15, verify=lambda text: text == "0")
        self.assertEqual(backend.complete("solve", TIER), "0")

    def test_unverified_but_successfully_generated_answer_is_preserved(self):
        backend = PKG.ABMCTSBackend(
            [Replies("draft")], budget=2, seed=16, verify=lambda _: 0)
        self.assertEqual(backend.complete("solve", TIER), "draft")
