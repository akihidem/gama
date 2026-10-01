"""Search fixtures: stdout is an answer only after successful execution."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_tool_search"
ROOT = Path.cwd().resolve() / "gama"
SPEC = importlib.util.spec_from_file_location(
    NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PKG = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = PKG
SPEC.loader.exec_module(PKG)
backends = sys.modules[NAME + ".backends"]
TIER = PKG.ModelTier.LARGE


class Reply(PKG.ModelBackend):
    available = True

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, tier, **kwargs):
        return self.text


def execute(raw):
    return PKG.ToolBackend(Reply(raw), timeout=3).complete("calculate", TIER)


class ExitStatusSearch(unittest.TestCase):
    def setUp(self):
        backends.reset_tool_stats()

    def test_nonzero_exit_discards_partial_stdout(self):
        raw = "```python\nprint('partial')\nraise SystemExit(7)\n```"
        self.assertEqual(execute(raw), raw)

    def test_exception_after_printing_uses_raw_fallback(self):
        raw = "```python\nprint('unfinished')\nraise RuntimeError('failed')\n```"
        self.assertEqual(execute(raw), raw)

    def test_failure_is_not_counted_as_ran(self):
        execute("```python\nprint('not an answer')\nraise SystemExit(3)\n```")
        self.assertEqual(backends.tool_stats()["ran"], 0)
        self.assertEqual(backends.tool_stats()["calls"], 1)

    def test_successful_stdout_is_stripped(self):
        self.assertEqual(execute("```python\nprint(' 42 ')\n```"), "42")
        self.assertEqual(backends.tool_stats()["ran"], 1)

    def test_no_code_preserves_original_reply(self):
        self.assertEqual(execute("I cannot express this as a program."),
                         "I cannot express this as a program.")

    def test_success_without_stdout_preserves_raw_reply(self):
        raw = "```python\nvalue = 9\n```"
        self.assertEqual(execute(raw), raw)
