import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import EnsembleBackend, ModelBackend
from gama.config import build_backend, ensemble_from_config
from gama.models import ModelTier


class Fixed(ModelBackend):
    """Returns a fixed string regardless of prompt."""
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        return self.reply


class CapturingAgg(ModelBackend):
    available = True

    def __init__(self):
        self.seen = None
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        self.seen = prompt
        return "FINAL"


class Boom(ModelBackend):
    available = True

    def complete(self, prompt, tier, **kw):
        raise RuntimeError("boom")


class TestEnsemble(unittest.TestCase):
    def test_majority(self):
        e = EnsembleBackend([Fixed("yes"), Fixed("yes"), Fixed("no")], strategy="majority")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "yes")

    def test_majority_normalizes_whitespace(self):
        e = EnsembleBackend([Fixed("396"), Fixed(" 396 "), Fixed("394")], strategy="majority")
        self.assertEqual(e.complete("q", ModelTier.LARGE).strip(), "396")

    def test_first_skips_empty(self):
        e = EnsembleBackend([Fixed(""), Fixed("A"), Fixed("B")], strategy="first")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "A")

    def test_synthesize_feeds_candidates_to_aggregator(self):
        agg = CapturingAgg()
        e = EnsembleBackend([Fixed("c1"), Fixed("c2")], strategy="synthesize", aggregator=agg)
        out = e.complete("the task", ModelTier.LARGE)
        self.assertEqual(out, "FINAL")
        self.assertIn("c1", agg.seen)
        self.assertIn("c2", agg.seen)
        self.assertIn("the task", agg.seen)

    def test_records_candidates(self):
        e = EnsembleBackend([Fixed("x"), Fixed("y")], strategy="first")
        e.complete("q", ModelTier.LARGE)
        self.assertEqual(e.last_candidates, ["x", "y"])

    def test_member_failure_tolerated(self):
        e = EnsembleBackend([Boom(), Fixed("ok")], strategy="first")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "ok")
        self.assertEqual(e.last_candidates, ["", "ok"])

    def test_all_empty_returns_empty(self):
        e = EnsembleBackend([Fixed(""), Fixed("")], strategy="majority")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "")

    def test_empty_members_rejected(self):
        with self.assertRaises(ValueError):
            EnsembleBackend([])

    def test_available_reflects_members(self):
        self.assertTrue(EnsembleBackend([Fixed("x")]).available)


class TestEnsembleFromConfig(unittest.TestCase):
    def test_member_times_n(self):
        e = ensemble_from_config({"ensemble": {"member": {"backend": "echo"}, "n": 4,
                                               "strategy": "first"}})
        self.assertEqual(len(e.members), 4)
        self.assertEqual(e.strategy, "first")

    def test_explicit_members(self):
        e = ensemble_from_config({"ensemble": {"members": [{"backend": "echo"},
                                                           {"backend": "null"}]}})
        self.assertEqual(len(e.members), 2)

    def test_aggregator_built(self):
        e = ensemble_from_config({"ensemble": {"member": {"backend": "echo"}, "n": 2,
                                               "aggregator": {"backend": "echo"}}})
        self.assertIsNotNone(e.aggregator)

    def test_missing_spec_raises(self):
        with self.assertRaises(ValueError):
            ensemble_from_config({"ensemble": {}})


class TestBuildBackend(unittest.TestCase):
    def test_plain(self):
        from gama.backends import EchoBackend
        self.assertIsInstance(build_backend({"backend": "echo"}), EchoBackend)

    def test_tool_wraps_inner(self):
        from gama.backends import EchoBackend, ToolBackend
        b = build_backend({"backend": "tool", "kwargs": {"inner": {"backend": "echo"}}})
        self.assertIsInstance(b, ToolBackend)
        self.assertIsInstance(b.backend, EchoBackend)

    def test_ensemble_members(self):
        from gama.backends import EnsembleBackend
        b = build_backend({"backend": "ensemble",
                           "kwargs": {"members": [{"backend": "echo"}, {"backend": "null"}]}})
        self.assertIsInstance(b, EnsembleBackend)
        self.assertEqual(len(b.members), 2)

    def test_gama_nested_composites(self):
        from gama.backends import GamaBackend, ToolBackend
        b = build_backend({"backend": "gama", "kwargs": {
            "backends": {"t": {"backend": "tool", "kwargs": {"inner": {"backend": "echo"}}},
                         "e": {"backend": "echo"}},
            "routing_table": {"qa": "t"}, "default": "e"}})
        self.assertIsInstance(b, GamaBackend)
        self.assertIsInstance(b.backends["t"], ToolBackend)
        self.assertEqual(b.pick("qa"), "t")
        self.assertEqual(b.pick("other"), "e")


if __name__ == "__main__":
    unittest.main()
