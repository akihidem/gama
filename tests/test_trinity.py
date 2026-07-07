import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.config import build_backend
from gama.models import ModelTier
from gama.trinity import TrinityBackend


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


class TestTrinityOneShot(unittest.TestCase):
    def test_scorer_picks_a_worker_and_only_that_worker_runs(self):
        scorer = Fixed("strong")
        weak, strong = Fixed("weak-answer"), Fixed("strong-answer")
        t = TrinityBackend([("weak", weak), ("strong", strong)], scorer=scorer)
        out = t.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "strong-answer")
        self.assertEqual(t.last_resolved_by, "strong")
        self.assertEqual(scorer.calls, 1)
        self.assertEqual((weak.calls, strong.calls), (0, 1))   # only the picked worker runs
        self.assertFalse(t.last_fallback)

    def test_special_end_of_turn_token_glued_to_reply_still_parses(self):
        # Measured on a real MLX server: some chat templates leave `<|im_end|>` glued
        # onto content with no separating whitespace ("strong<|im_end|>\n"), which broke
        # naive whitespace-only stripping and always fell back to the cheapest worker.
        scorer = Fixed("strong<|im_end|>\n")
        weak, strong = Fixed("weak-answer"), Fixed("strong-answer")
        t = TrinityBackend([("weak", weak), ("strong", strong)], scorer=scorer)
        out = t.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "strong-answer")
        self.assertEqual(t.last_resolved_by, "strong")
        self.assertFalse(t.last_fallback)

    def test_cost_is_scorer_plus_chosen_worker_fixed_no_retry(self):
        scorer = Fixed("weak")
        weak, strong = Fixed("A"), Fixed("B")
        t = TrinityBackend([("weak", weak), ("strong", strong)], scorer=scorer)
        t.complete("q", ModelTier.LARGE)
        # default costs [1.0, 2.0]; scorer_cost defaults to costs[0]=1.0; picked "weak" (cost 1.0)
        self.assertEqual(t.last_cost, 1.0 + 1.0)

    def test_unparseable_scorer_reply_falls_back_to_cheapest(self):
        scorer = Fixed("I have no idea, sorry!")
        weak, strong = Fixed("weak-answer"), Fixed("strong-answer")
        t = TrinityBackend([("weak", weak), ("strong", strong)], scorer=scorer)
        out = t.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "weak-answer")
        self.assertEqual(t.last_resolved_by, "weak")
        self.assertTrue(t.last_fallback)
        self.assertEqual((weak.calls, strong.calls), (1, 0))    # never fell to the EXPENSIVE tier

    def test_scorer_exception_fails_closed_not_crash(self):
        class Boom(ModelBackend):
            available = True
            def complete(self, prompt, tier, **kw):
                raise RuntimeError("boom")

        weak, strong = Fixed("weak-answer"), Fixed("strong-answer")
        t = TrinityBackend([("weak", weak), ("strong", strong)], scorer=Boom())
        out = t.complete("q", ModelTier.LARGE)   # must not raise
        self.assertEqual(out, "weak-answer")
        self.assertTrue(t.last_fallback)

    def test_single_worker_skips_the_classification_call(self):
        scorer = Fixed("irrelevant")
        only = Fixed("the-answer")
        t = TrinityBackend([("only", only)], scorer=scorer)
        out = t.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "the-answer")
        self.assertEqual(scorer.calls, 0)      # no ambiguity to resolve -> no scorer call
        self.assertEqual(only.calls, 1)

    def test_failing_worker_does_not_crash_the_run(self):
        class Boom(ModelBackend):
            available = True
            def complete(self, prompt, tier, **kw):
                raise RuntimeError("boom")

        scorer = Fixed("flaky")
        t = TrinityBackend([("flaky", Boom()), ("strong", Fixed("GOOD"))], scorer=scorer)
        out = t.complete("q", ModelTier.LARGE)
        self.assertEqual(out, "")               # a failing dispatch -> empty artifact, not a retry
        self.assertEqual(t.last_resolved_by, "flaky")   # trinity does not escalate past it


class TestTrinityFromConfig(unittest.TestCase):
    def test_build_backend_wires_workers_recursively(self):
        spec = {
            "backend": "trinity",
            "kwargs": {
                "workers": [
                    {"label": "weak", "backend": "echo"},
                    {"label": "strong", "backend": "echo"},
                ],
            },
        }
        be = build_backend(spec)
        self.assertIsInstance(be, TrinityBackend)
        self.assertEqual([label for label, _ in be.workers], ["weak", "strong"])
        out = be.complete("hello", ModelTier.LARGE)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
