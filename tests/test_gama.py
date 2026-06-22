import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import GamaBackend, ModelBackend
from gama.config import gama_from_config
from gama.models import ModelTier


class Tagged(ModelBackend):
    """Fake sub-backend: returns its own name and records the task_type it saw."""

    available = True

    def __init__(self, name):
        self.name = name
        self.seen = []
        self.last_usage = {"total_tokens": 7}

    def complete(self, prompt, tier, **kw):
        self.seen.append(kw.get("task_type"))
        return f"[{self.name}]{prompt}"


class TestGamaBackend(unittest.TestCase):
    def setUp(self):
        self.a = Tagged("alpha")
        self.b = Tagged("beta")
        self.gama = GamaBackend(
            {"alpha": self.a, "beta": self.b},
            routing_table={"code_implementation": "beta", "qa": "alpha"},
            default="alpha",
        )

    def test_routes_by_task_type(self):
        out = self.gama.complete("x", ModelTier.LARGE, task_type="code_implementation")
        self.assertEqual(out, "[beta]x")
        self.assertEqual(self.gama.last_route, ("code_implementation", "beta"))

    def test_mapped_class_routes_to_alpha(self):
        self.assertEqual(self.gama.complete("y", ModelTier.SMALL, task_type="qa"), "[alpha]y")

    def test_unmapped_falls_back_to_default(self):
        out = self.gama.complete("z", ModelTier.MEDIUM, task_type="security_review")
        self.assertEqual(out, "[alpha]z")
        self.assertEqual(self.gama.last_route, ("security_review", "alpha"))

    def test_missing_task_type_uses_default(self):
        self.assertEqual(self.gama.complete("w", ModelTier.LARGE), "[alpha]w")

    def test_task_type_kwarg_propagates_to_subbackend(self):
        self.gama.complete("x", ModelTier.LARGE, task_type="code_implementation")
        self.assertIn("code_implementation", self.b.seen)

    def test_last_usage_propagated(self):
        self.gama.complete("x", ModelTier.LARGE, task_type="qa")
        self.assertEqual(self.gama.last_usage, {"total_tokens": 7})

    def test_pick_unknown_backend_in_table_falls_back(self):
        m = GamaBackend({"alpha": self.a}, routing_table={"qa": "ghost"}, default="alpha")
        self.assertEqual(m.pick("qa"), "alpha")

    def test_default_must_exist(self):
        with self.assertRaises(ValueError):
            GamaBackend({"alpha": self.a}, default="nope")

    def test_empty_backends_rejected(self):
        with self.assertRaises(ValueError):
            GamaBackend({})

    def test_available_reflects_subbackends(self):
        self.assertTrue(self.gama.available)


class TestGamaFromConfig(unittest.TestCase):
    def test_builds_named_subbackends(self):
        cfg = {
            "default_backend": "echo",
            "routing_table": {"code_implementation": "echo", "qa": "null"},
            "backends": {},
        }
        gama = gama_from_config(cfg)
        self.assertIn("echo", gama.backends)
        self.assertIn("null", gama.backends)        # named in the table -> instantiated
        self.assertEqual(gama.default, "echo")
        self.assertEqual(gama.pick("code_implementation"), "echo")
        self.assertEqual(gama.pick("qa"), "null")
        self.assertEqual(gama.pick("unmapped_type"), "echo")

    def test_forwards_backend_kwargs(self):
        cfg = {
            "default_backend": "ollama",
            "routing_table": {},
            "backends": {"ollama": {"host": "http://example:9999"}},
        }
        gama = gama_from_config(cfg)
        self.assertEqual(gama.backends["ollama"].host, "http://example:9999")


if __name__ == "__main__":
    unittest.main()
