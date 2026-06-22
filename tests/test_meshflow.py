import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.config import build_backend, meshflow_from_config
from gama.meshflow import (
    NEEDS_HUMAN,
    MeshflowBackend,
    resolve_verifier,
    verify_code_runs,
    verify_nonempty,
)
from gama.models import ModelTier


class Fixed(ModelBackend):
    """Returns a fixed string and counts how many times it was called."""
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        self.calls += 1
        return self.reply


class CapturingAgg(ModelBackend):
    available = True

    def __init__(self, reply="MERGED"):
        self.reply = reply
        self.seen = None
        self.last_usage = {"total_tokens": 7}

    def complete(self, prompt, tier, **kw):
        self.seen = prompt
        return self.reply


def good(art):
    return 1.0 if art == "GOOD" else 0.0


class TestMeshflowEscalation(unittest.TestCase):
    def test_cheap_solves_stops_early(self):
        cheap, strong = Fixed("GOOD"), Fixed("X")
        m = MeshflowBackend([("cheap", cheap), ("strong", strong)], verify=good)
        out = m.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "GOOD")
        self.assertEqual(m.last_resolved_by, "cheap")
        self.assertEqual(cheap.calls, 1)
        self.assertEqual(strong.calls, 0)            # never escalated
        self.assertEqual(m.last_trace, [{"tier": "cheap", "score": 1.0}])
        self.assertEqual(m.last_cost, 1.0)

    def test_escalates_when_verify_fails(self):
        cheap, mid, strong = Fixed("BAD"), Fixed("GOOD"), Fixed("X")
        m = MeshflowBackend([("cheap", cheap), ("mid", mid), ("strong", strong)], verify=good)
        out = m.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "GOOD")
        self.assertEqual(m.last_resolved_by, "mid")
        self.assertEqual((cheap.calls, mid.calls, strong.calls), (1, 1, 0))
        self.assertEqual(m.last_cost, 3.0)           # cost 1 + 2
        self.assertEqual(m.last_human_gate, False)

    def test_default_label_is_backend_name(self):
        m = MeshflowBackend([Fixed("GOOD")], verify=good)
        m.complete("q", ModelTier.LARGE)
        self.assertEqual(m.last_resolved_by, "abstract")  # ModelBackend.name default

    def test_failing_tier_is_escalated_past(self):
        # A flaky tier (network/SSH/subprocess error) must not crash the run — escalate.
        class Boom(ModelBackend):
            available = True
            def complete(self, prompt, tier, **kw):
                raise RuntimeError("boom")
        m = MeshflowBackend([("flaky", Boom()), ("strong", Fixed("GOOD"))], verify=good)
        out = m.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "GOOD")
        self.assertEqual(m.last_resolved_by, "strong")
        self.assertEqual(m.last_trace[0], {"tier": "flaky", "score": 0.0})


class TestMeshflowEdgeMesh(unittest.TestCase):
    def test_union_mesh_at_edge(self):
        # No single tier passes; the union of their distinct lines does.
        tiers = [Fixed("A"), Fixed("B"), Fixed("C")]
        m = MeshflowBackend(tiers, verify=lambda a: 1.0 if a == "A\nB\nC" else 0.0, mesh="union")
        out = m.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "A\nB\nC")
        self.assertEqual(m.last_resolved_by, "mesh")
        self.assertEqual(m.last_trace[-1], {"tier": "mesh", "score": 1.0})
        self.assertEqual(m.last_cost, 1 + 2 + 3 + 0.5)

    def test_synthesize_mesh_uses_aggregator(self):
        agg = CapturingAgg("MERGED")
        m = MeshflowBackend([Fixed("c1"), Fixed("c2")],
                            verify=lambda a: 1.0 if a == "MERGED" else 0.0,
                            mesh="synthesize", aggregator=agg)
        out = m.complete("the task", ModelTier.LARGE)
        self.assertEqual(out, "MERGED")
        self.assertEqual(m.last_resolved_by, "mesh")
        self.assertIn("c1", agg.seen)               # drafts were handed to the aggregator
        self.assertIn("c2", agg.seen)
        self.assertIn("the task", agg.seen)
        self.assertEqual(m.last_usage, {"total_tokens": 7})  # usage from the aggregator

    def test_mesh_disabled_skips_to_membrane(self):
        m = MeshflowBackend([Fixed("A"), Fixed("B")], verify=good, mesh=False)
        out = m.complete("q", ModelTier.LARGE)       # nothing passes, low stakes
        self.assertEqual(out, "B")                   # best-effort = strongest draft
        self.assertEqual(m.last_resolved_by, "best-effort")


class TestMeshflowMembrane(unittest.TestCase):
    def test_human_gate_on_high_stakes(self):
        m = MeshflowBackend([Fixed("A"), Fixed("B")], verify=good, mesh=False,
                            stakes=0.9, stakes_threshold=0.7)
        out = m.complete("q", ModelTier.LARGE)
        self.assertEqual(out, NEEDS_HUMAN)
        self.assertTrue(m.last_human_gate)
        self.assertIsNone(m.last_resolved_by)

    def test_best_effort_on_low_stakes(self):
        m = MeshflowBackend([Fixed("A"), Fixed("B")], verify=good, mesh=False, stakes=0.0)
        out = m.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "B")                   # strongest tier's draft, flagged
        self.assertEqual(m.last_resolved_by, "best-effort")
        self.assertFalse(m.last_human_gate)

    def test_stakes_via_kwargs(self):
        m = MeshflowBackend([Fixed("A")], verify=good, mesh=False, stakes_threshold=0.5)
        self.assertEqual(m.complete("q", ModelTier.LARGE, stakes=0.9), NEEDS_HUMAN)
        self.assertEqual(m.complete("q", ModelTier.LARGE, stakes=0.1), "A")


class TestMeshflowVerifyKwarg(unittest.TestCase):
    def test_kwargs_verify_overrides_and_gates(self):
        # Constructed with no verifier; the bench-style `verify` kwarg drives escalation.
        cheap, strong = Fixed("GOOD"), Fixed("X")
        m = MeshflowBackend([("cheap", cheap), ("strong", strong)], verify=None)
        out = m.complete("q", ModelTier.LARGE, verify=good)
        self.assertEqual(out, "GOOD")
        self.assertEqual(m.last_resolved_by, "cheap")
        self.assertEqual(strong.calls, 0)
        # control kwargs are NOT forwarded to sub-backends (they take **kwargs anyway)

    def test_no_verifier_runs_all_then_best_effort(self):
        cheap, strong = Fixed("a"), Fixed("b")
        m = MeshflowBackend([cheap, strong], verify=None, mesh=False)
        out = m.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "b")
        self.assertEqual((cheap.calls, strong.calls), (1, 1))
        self.assertEqual(m.last_resolved_by, "best-effort")


class TestVerifiers(unittest.TestCase):
    def test_resolve_builtin_and_callable_and_none(self):
        self.assertIs(resolve_verifier(None), None)
        self.assertIs(resolve_verifier(good), good)
        self.assertIs(resolve_verifier("code_runs"), verify_code_runs)
        self.assertIs(resolve_verifier("nonempty"), verify_nonempty)

    def test_resolve_unknown_raises(self):
        with self.assertRaises(ValueError):
            resolve_verifier("does-not-exist")

    def test_nonempty(self):
        self.assertEqual(verify_nonempty("x"), 1.0)
        self.assertEqual(verify_nonempty("  "), 0.0)
        self.assertEqual(verify_nonempty(NEEDS_HUMAN), 0.0)

    def test_code_runs_executes(self):
        self.assertEqual(verify_code_runs("```python\nprint(2 + 2)\n```"), 1.0)
        self.assertEqual(verify_code_runs("```python\nimport sys; sys.exit(1)\n```"), 0.0)
        self.assertEqual(verify_code_runs("not code at all"), 0.0)
        self.assertEqual(verify_code_runs(""), 0.0)


class TestMeshflowConstruction(unittest.TestCase):
    def test_empty_tiers_rejected(self):
        with self.assertRaises(ValueError):
            MeshflowBackend([])

    def test_available_reflects_tiers(self):
        self.assertTrue(MeshflowBackend([Fixed("x")]).available)

    def test_costs_override(self):
        m = MeshflowBackend([Fixed("BAD"), Fixed("GOOD")], verify=good, costs=[0.2, 5.0])
        m.complete("q", ModelTier.LARGE)
        self.assertEqual(m.last_cost, round(0.2 + 5.0, 3))


class TestMeshflowFromConfig(unittest.TestCase):
    def test_build_backend_constructs_meshflow(self):
        b = build_backend({"backend": "meshflow", "kwargs": {
            "tiers": [{"backend": "echo"}, {"backend": "echo"}],
            "verify": "nonempty", "mesh": "union", "stakes_threshold": 0.5}})
        self.assertIsInstance(b, MeshflowBackend)
        self.assertEqual(len(b.tiers), 2)
        self.assertIs(b.verify, verify_nonempty)
        self.assertEqual(b.stakes_threshold, 0.5)

    def test_build_backend_labels_and_nested_composite(self):
        b = build_backend({"backend": "meshflow", "kwargs": {"tiers": [
            {"label": "small", "backend": "echo"},
            {"label": "tooled", "backend": "tool", "kwargs": {"inner": {"backend": "echo"}}},
        ]}})
        from gama.backends import ToolBackend
        self.assertEqual([lbl for lbl, _ in b.tiers], ["small", "tooled"])
        self.assertIsInstance(b.tiers[1][1], ToolBackend)

    def test_meshflow_from_config(self):
        b = meshflow_from_config({"meshflow": {"tiers": [{"backend": "echo"}],
                                               "verify": "nonempty"}})
        self.assertIsInstance(b, MeshflowBackend)

    def test_meshflow_from_config_needs_tiers(self):
        with self.assertRaises(ValueError):
            meshflow_from_config({"meshflow": {}})


if __name__ == "__main__":
    unittest.main()
