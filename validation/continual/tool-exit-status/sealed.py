"""Sealed fixtures: termination, multiple blocks, and preserved success."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_tool_sealed"
ROOT = Path.cwd().resolve() / "gama"
SPEC = importlib.util.spec_from_file_location(
    NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PKG = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = PKG
SPEC.loader.exec_module(PKG)
TIER = PKG.ModelTier.LARGE


class Reply(PKG.ModelBackend):
    available = True

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, tier, **kwargs):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def execute(raw):
    return PKG.ToolBackend(Reply(raw), timeout=3).complete("calculate", TIER)


class ExitStatusSealed(unittest.TestCase):
    def test_bare_program_failure_keeps_raw_fallback(self):
        raw = "print('untrusted partial')\nraise SystemExit(2)"
        self.assertEqual(execute(raw), raw)

    def test_selected_longest_program_failure_keeps_entire_reply(self):
        raw = ("```python\nprint(1)\n```\n"
               "```python\nprint('unfinished longer result')\nraise SystemExit(4)\n```")
        self.assertEqual(execute(raw), raw)

    def test_signal_termination_discards_flushed_stdout(self):
        raw = ("```python\nimport os, signal\nprint('partial', flush=True)\n"
               "os.kill(os.getpid(), signal.SIGTERM)\n```")
        self.assertEqual(execute(raw), raw)

    def test_zero_stdout_is_a_successful_answer(self):
        self.assertEqual(execute("```python\nprint(0)\n```"), "0")

    def test_success_preserves_unicode_and_internal_newlines(self):
        raw = "```python\nprint('  café')\nprint('二  ')\n```"
        self.assertEqual(execute(raw), "café\n二")

    def test_inner_backend_exception_is_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError, "generator offline"):
            execute(RuntimeError("generator offline"))
