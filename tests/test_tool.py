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
        self.seen_kwargs = None

    def complete(self, prompt, tier, **kw):
        self.seen_kwargs = kw
        return self.reply


class FakeChat(FakeCode):
    """A backend that knows what to do with a prefill (it declares so)."""
    supports_prefill = True


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

    def test_an_unclosed_fence_is_still_the_program(self):
        # A length stop eats the closing fence. Running "```python\n..." as Python is a
        # SyntaxError on line 1, so the lane used to fall back to prose for a reply that
        # contained a complete program.
        be = ToolBackend(FakeCode("thinking...\n```python\nprint(6 * 7)\n"))
        self.assertEqual(be.complete("q", ModelTier.LARGE), "42")


class TestPrefill(unittest.TestCase):
    def test_prefill_reaches_the_inner_backend_only_when_configured(self):
        inner = FakeChat("```python\nprint(1)\n```")
        ToolBackend(inner).complete("q", ModelTier.LARGE)
        self.assertNotIn("prefill", inner.seen_kwargs)
        ToolBackend(inner, prefill=ToolBackend.PREFILL).complete("q", ModelTier.LARGE)
        self.assertEqual(inner.seen_kwargs["prefill"], "```python\n")

    def test_refuses_a_backend_that_would_drop_the_prefill(self):
        # every adapter takes **kwargs, so without the positive flag the prefill would vanish
        # and the lane be scored as if it had one
        with self.assertRaises(ValueError):
            ToolBackend(FakeCode("x"), prefill="```python\n")

    def test_reads_back_every_shape_a_server_returns_for_a_prefill(self):
        pf = ToolBackend.PREFILL
        # 1. the server continued our fence: body, closing fence, then chatter
        be = ToolBackend(FakeChat("print(2 + 2)\n```\nThat prints 4."), prefill=pf)
        self.assertEqual(be.complete("q", ModelTier.LARGE), "4")
        # 2. the server started a new turn and the model re-opened the fence itself —
        #    re-attaching ours here would match the empty block between the two fences
        be = ToolBackend(FakeChat("```python\nprint(3 + 3)\n```"), prefill=pf)
        self.assertEqual(be.complete("q", ModelTier.LARGE), "6")
        # 3. continued, but never closed (length stop)
        be = ToolBackend(FakeChat("print(4 + 4)\n"), prefill=pf)
        self.assertEqual(be.complete("q", ModelTier.LARGE), "8")
        # 3'. the model opened its own fence mid-reply (tagged and closed, or untagged and
        #     not): re-attaching ours would pair our opener with its opener and hand the prose
        #     between them to the interpreter — the codex reviewer's case, reproduced
        be = ToolBackend(FakeChat("thinking...\n```python\nprint(5 * 5)\n```"), prefill=pf)
        self.assertEqual(be.complete("q", ModelTier.LARGE), "25")
        be = ToolBackend(FakeChat("Here:\n```\nprint(7 * 7)"), prefill=pf)
        self.assertEqual(be.complete("q", ModelTier.LARGE), "49")
        # 4. prose only: falls back to the reply. The fence around it was OURS, so this is
        #    the model declining to write code (no_code), not code that printed nothing.
        from gama.backends import reset_tool_stats, tool_stats
        reset_tool_stats()
        be = ToolBackend(FakeChat("I would rather explain."), prefill=pf)
        self.assertEqual(be.complete("q", ModelTier.LARGE), "I would rather explain.")
        self.assertEqual(tool_stats()["no_code"], 1)
        self.assertEqual(tool_stats()["empty_out"], 0)
        # ...whereas a fence the model opened itself, with nothing runnable inside, is code
        # that failed (the run-V josephus reply: 1536 tokens of prose inside an unclosed fence)
        be = ToolBackend(FakeChat("```python\nLet me think about this differently."), prefill=pf)
        be.complete("q", ModelTier.LARGE)
        self.assertEqual(tool_stats()["empty_out"], 1)


if __name__ == "__main__":
    unittest.main()
