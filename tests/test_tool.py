import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend, ToolBackend
from gama.models import ModelTier


class FakeCode(ModelBackend):
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        return self.reply


class TestToolBackend(unittest.TestCase):
    def test_runs_code_returns_stdout(self):
        be = ToolBackend(FakeCode("```python\nprint(47 * 53 + 89 * 17)\n```"))
        self.assertEqual(be.complete("compute 47*53+89*17", ModelTier.LARGE), "4004")

    def test_picks_longest_block(self):
        be = ToolBackend(FakeCode("draft:\n```python\nprint(1)\n```\nfinal:\n"
                                  "```python\nprint(2 + 3)\n```"))
        self.assertEqual(be.complete("q", ModelTier.LARGE), "5")

    def test_fallback_when_no_code(self):
        be = ToolBackend(FakeCode("the answer is 5"))
        self.assertEqual(be.complete("q", ModelTier.LARGE), "the answer is 5")

    def test_fallback_on_empty_stdout(self):
        be = ToolBackend(FakeCode("```python\nx = 1\n```"))  # no print -> fall back to raw
        self.assertIn("x = 1", be.complete("q", ModelTier.LARGE))

    def test_available_reflects_inner(self):
        self.assertTrue(ToolBackend(FakeCode("x")).available)


if __name__ == "__main__":
    unittest.main()
