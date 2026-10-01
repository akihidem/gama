"""Confirmation fixtures: prefill, stderr, exit messages, and timeouts."""
import importlib.util
from pathlib import Path
import sys
import unittest

NAME = "_starter_tool_confirm"
ROOT = Path.cwd().resolve() / "gama"
SPEC = importlib.util.spec_from_file_location(
    NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PKG = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = PKG
SPEC.loader.exec_module(PKG)
TIER = PKG.ModelTier.LARGE


class Reply(PKG.ModelBackend):
    available = True
    supports_prefill = True

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, tier, **kwargs):
        return self.text


class ExitStatusConfirm(unittest.TestCase):
    def run_reply(self, raw, **kwargs):
        return PKG.ToolBackend(Reply(raw), **kwargs).complete("calculate", TIER)

    def test_exit_message_does_not_validate_prior_stdout(self):
        raw = "```python\nprint('candidate')\nraise SystemExit('incomplete')\n```"
        self.assertEqual(self.run_reply(raw), raw)

    def test_unclosed_fence_keeps_failure_fallback(self):
        raw = "Working:\n```python\nprint('draft')\nraise SystemExit(9)\n"
        self.assertEqual(self.run_reply(raw), raw)

    def test_prefilled_failure_returns_original_continuation(self):
        raw = "print('partial continuation')\nraise SystemExit(5)\n```"
        self.assertEqual(self.run_reply(raw, prefill=PKG.ToolBackend.PREFILL), raw)

    def test_stderr_diagnostic_does_not_invalidate_zero_exit(self):
        raw = "```python\nimport sys\nprint('diagnostic', file=sys.stderr)\nprint(27)\n```"
        self.assertEqual(self.run_reply(raw), "27")

    def test_prefilled_success_remains_executable(self):
        raw = "print(6 * 8)\n```"
        self.assertEqual(self.run_reply(raw, prefill=PKG.ToolBackend.PREFILL), "48")

    def test_timeout_does_not_publish_partial_stdout(self):
        raw = "```python\nimport time\nprint('partial', flush=True)\ntime.sleep(10)\n```"
        self.assertEqual(self.run_reply(raw, timeout=0.2), raw)
