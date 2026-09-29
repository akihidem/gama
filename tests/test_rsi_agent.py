import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from gama.rsi_agent import propose_patch
from gama.rsi_process import ProcessError


PATCH = (
    "diff --git a/gama/example.py b/gama/example.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/gama/example.py\n"
    "+++ b/gama/example.py\n"
    "@@ -1 +1 @@\n"
    "-value = 1\n"
    "+value = 2\n"
)


class TestRSIAgent(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gama-rsi-agent-test-")
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        self.request = {
            "goal": "Improve measured performance while preserving correctness.",
            "parent": "0123456789abcdef",
            "files": {"gama/example.py": "value = 1\n# 日本語\n"},
            "evaluation": {"score": 0.5, "failures": ["independent-case-2"]},
            "papers": [{"title": "Automated Design of Agentic Systems",
                        "url": "https://www.alphaxiv.org/abs/2408.08435"}],
        }

    def command_agent(self, output=PATCH, stderr="", returncode=0):
        return {
            "name": "external-fake",
            "command": [
                sys.executable, "-I", "-c",
                f"import sys; sys.stdout.write({output!r}); "
                f"sys.stderr.write({stderr!r}); sys.exit({returncode})",
            ],
        }

    def propose(self, agent, **kwargs):
        return propose_patch(
            agent, request=self.request, cwd=self.cwd, timeout=kwargs.pop("timeout", 5),
            **kwargs,
        )

    def test_command_receives_exact_json_request_and_returns_raw_patch(self):
        script = self.cwd / "emitter.py"
        script.write_text(
            "import json, pathlib, sys\n"
            "request = json.load(sys.stdin)\n"
            "pathlib.Path('observed-request.json').write_text(json.dumps(request))\n"
            f"sys.stdout.write({PATCH!r})\n",
            encoding="utf-8",
        )
        proposal = self.propose({
            "name": "request-reader", "command": [sys.executable, "-I", str(script)]})
        self.assertEqual(proposal.patch, PATCH)
        self.assertEqual(proposal.output, PATCH)
        self.assertIsNone(proposal.usage)
        self.assertEqual(
            json.loads((self.cwd / "observed-request.json").read_text()), self.request)

    def test_single_diff_fence_preserves_raw_output_and_extracts_patch(self):
        output = f"\n```diff\n{PATCH}```\n"
        proposal = self.propose(self.command_agent(output))
        self.assertEqual(proposal.patch, PATCH)
        self.assertEqual(proposal.output, output)

    def test_command_disables_bytecode_while_preserving_inherited_environment(self):
        (self.cwd / "local_helper.py").write_text(f"PATCH = {PATCH!r}\n", encoding="utf-8")
        script = self.cwd / "importing_emitter.py"
        script.write_text(
            "import os, sys\n"
            "from local_helper import PATCH\n"
            "assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n"
            "assert os.environ['GAMA_RSI_AGENT_TEST'] == 'keep-value'\n"
            f"assert os.environ.get('HOME') == {os.environ.get('HOME')!r}\n"
            "sys.stdout.write(PATCH)\n",
            encoding="utf-8",
        )
        before = sorted(self.cwd.rglob("*"))
        with mock.patch.dict(os.environ, {
                "GAMA_RSI_AGENT_TEST": "keep-value", "PYTHONDONTWRITEBYTECODE": "0"}):
            proposal = self.propose({
                "name": "importing-command", "command": [sys.executable, str(script)]})
            self.assertEqual(os.environ["PYTHONDONTWRITEBYTECODE"], "0")
        self.assertEqual(proposal.patch, PATCH)
        self.assertEqual(sorted(self.cwd.rglob("*")), before)
        self.assertFalse(list(self.cwd.rglob("__pycache__")))

    def test_multiple_files_hunks_new_files_and_no_newline_markers(self):
        patch = (
            "--- a/gama/example.py\n+++ b/gama/example.py\n"
            "@@ -1,3 +1,3 @@ optional heading\n"
            " context\n-old\n+new\n \n"
            "@@ -10 +10 @@\n-last\n\\ No newline at end of file\n"
            "+updated\n\\ No newline at end of file\n"
            "diff --git a/gama/new.py b/gama/new.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/gama/new.py\n"
            "@@ -0,0 +1,2 @@\n+first = 1\n+second = 2\n"
        )
        self.assertEqual(self.propose(self.command_agent(patch)).patch, patch)

    def test_patch_can_change_fences_and_preserves_trailing_blank_context(self):
        patch = (
            "--- a/gama/example.py\n+++ b/gama/example.py\n"
            "@@ -1,2 +1,2 @@\n-```python\n+```diff\n \n"
        )
        output = f"```diff\n{patch}```"
        self.assertEqual(self.propose(self.command_agent(output)).patch, patch)

    def test_crlf_source_data_is_preserved_in_the_final_changed_line(self):
        patch = (
            "--- a/gama/example.py\n+++ b/gama/example.py\n"
            "@@ -1 +1 @@\n-value = 1\r\n+value = 2\r\n"
        )
        for output in [patch, f"```diff\n{patch}```\n"]:
            with self.subTest(fenced=output.startswith("```")):
                self.assertEqual(self.propose(self.command_agent(output)).patch, patch)

    def test_path_policy_is_left_to_the_workspace(self):
        patch = PATCH.replace("gama/example.py", "outside-the-allowlist.py")
        self.assertEqual(self.propose(self.command_agent(patch)).patch, patch)

    def test_empty_ambiguous_and_malformed_outputs_are_rejected(self):
        bad_outputs = [
            "",
            "Here is an improvement.",
            f"Explanation\n{PATCH}",
            f"```diff\n{PATCH}```\n```diff\n{PATCH}```",
            f"```python\n{PATCH}```",
            f"```diff\n{PATCH}",
            f"{PATCH}Postscript\n",
            PATCH.replace("@@ -1 +1 @@", "@@ broken @@"),
            PATCH.replace("@@ -1 +1 @@", "@@ -1,2 +1,2 @@"),
            PATCH + "+unaccounted line\n",
            "--- a/file\n+++ b/file\n",
            "--- a/file\n+++ b/file\n@@ -1 +1 @@\n unchanged\n",
            "--- \n+++ b/file\n@@ -1 +1 @@\n-a\n+b\n",
            "--- a/file\n+++ b/file\n@@ -0 +1 @@\n-a\n+b\n",
            "--- a/file\n+++ b/file\n@@ -0,0 +0,0 @@\n",
            PATCH + "\0",
        ]
        for output in bad_outputs:
            with self.subTest(output=output), self.assertRaisesRegex(
                    ProcessError, "invalid patch"):
                self.propose(self.command_agent(output))

    def test_nonzero_exit_rejects_even_a_valid_patch_and_bounds_stderr_tail(self):
        with self.assertRaises(ProcessError) as caught:
            self.propose(self.command_agent(
                stderr="discarded-prefix\n" + "x" * 10000 + "\nfailed-tail",
                returncode=17,
            ))
        message = str(caught.exception)
        self.assertIn("external-fake", message)
        self.assertIn("status 17", message)
        self.assertIn("failed-tail", message)
        self.assertNotIn("discarded-prefix", message)
        self.assertLess(len(message), 4300)

    def test_command_timeout_and_cancellation(self):
        agent = {"name": "sleeping", "command": [
            sys.executable, "-I", "-c", "import time; time.sleep(30)"]}
        with self.assertRaisesRegex(ProcessError, "timed out"):
            self.propose(agent, timeout=0.15)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(ProcessError, "cancelled"):
            self.propose(agent, cancel=cancel)

    def test_exactly_one_agent_mode_is_required(self):
        backend = {"backend": "unused-fake"}
        for agent in [
            {"name": "both", "command": ["unused"], "backend": backend},
            {"name": "neither"},
            {"name": "", "command": ["unused"]},
            {"name": "bad-spec", "backend": "unused-fake"},
        ]:
            with self.subTest(agent=agent), self.assertRaises(ValueError):
                self.propose(agent)

    def make_parent_backend(self, *, output=PATCH, extra_code=""):
        package = self.cwd / "gama"
        package.mkdir()
        (package / "__init__.py").write_text(
            "print('parent package import chatter')\n", encoding="utf-8")
        (package / "models.py").write_text(
            "from enum import Enum\n"
            "class ModelTier(str, Enum):\n"
            "    LARGE = 'large'\n",
            encoding="utf-8",
        )
        (package / "config.py").write_text(
            textwrap.dedent(f"""\
                import json
                import os
                from pathlib import Path
                import subprocess
                import sys
                import time

                CALLS = 0
                class ParentBackend:
                    last_usage = None

                    def complete(self, prompt, tier, **kwargs):
                        global CALLS
                        CALLS += 1
                        print('provider print chatter')
                        os.write(1, b'provider fd chatter\\n')
                        subprocess.run([sys.executable, '-I', '-c', "print('child chatter')"],
                                       check=True)
                        Path('observed-call.json').write_text(json.dumps({{
                            'prompt': prompt, 'tier': tier.value, 'kwargs': kwargs,
                            'calls': CALLS, 'config_path': __file__,
                            'isolated': sys.flags.isolated,
                            'bytecode_disabled': sys.dont_write_bytecode,
                        }}))
                        self.last_usage = {{'prompt_tokens': 7, 'completion_tokens': 5,
                                           'total_tokens': 12}}
                        {extra_code or 'pass'}
                        return {output!r}

                def build_backend(spec):
                    print('parent build_backend chatter')
                    Path('observed-spec.json').write_text(json.dumps(spec))
                    return ParentBackend()
                """),
            encoding="utf-8",
        )
        return {
            "name": "selected-parent",
            "backend": {"backend": "parent-only-fake",
                        "kwargs": {"temperature": 0.25, "max_tokens": 100}},
        }

    def test_backend_worker_imports_selected_parent_and_isolates_all_stdout(self):
        output = f"```diff\n{PATCH}```\n"
        agent = self.make_parent_backend(output=output)
        self.assertFalse((self.cwd / "gama" / "rsi_agent.py").exists())
        self.assertFalse((self.cwd / "gama" / "rsi_process.py").exists())
        proposal = self.propose(agent)
        self.assertEqual(proposal.patch, PATCH)
        self.assertEqual(proposal.output, output)
        self.assertEqual(proposal.usage, {
            "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12})
        self.assertEqual(
            json.loads((self.cwd / "observed-spec.json").read_text()), agent["backend"])
        observed = json.loads((self.cwd / "observed-call.json").read_text())
        self.assertEqual(Path(observed["config_path"]), self.cwd / "gama" / "config.py")
        self.assertEqual(observed["isolated"], 1)
        self.assertTrue(observed["bytecode_disabled"])
        self.assertFalse(list(self.cwd.rglob("__pycache__")))
        self.assertEqual(observed["tier"], "large")
        self.assertEqual(observed["kwargs"], {"task_type": "code_implementation"})
        prompt = observed["prompt"]
        self.assertIn(self.request["goal"], prompt)
        self.assertIn(self.request["parent"], prompt)
        self.assertIn(json.dumps(self.request["files"], ensure_ascii=False, indent=2), prompt)
        self.assertIn("independent-case-2", prompt)
        self.assertIn("https://www.alphaxiv.org/abs/2408.08435", prompt)
        self.assertIn("ONLY a unified diff", prompt)

    def test_each_backend_proposal_has_fresh_parent_globals(self):
        agent = self.make_parent_backend()
        for _ in range(2):
            self.assertEqual(self.propose(agent).patch, PATCH)
            observed = json.loads((self.cwd / "observed-call.json").read_text())
            self.assertEqual(observed["calls"], 1)

    def test_missing_parent_backend_does_not_fall_back_to_controller_package(self):
        with self.assertRaisesRegex(ProcessError, "selected parent has no gama/config.py"):
            self.propose({"name": "missing-parent", "backend": {"backend": "echo"}})

    def test_backend_failure_keeps_provider_stderr_context(self):
        agent = self.make_parent_backend(extra_code="raise RuntimeError('parent-failure')")
        with self.assertRaises(ProcessError) as caught:
            self.propose(agent)
        self.assertIn("parent-failure", str(caught.exception))
        self.assertIn("provider fd chatter", str(caught.exception))

    def test_backend_worker_has_a_real_subprocess_timeout(self):
        agent = self.make_parent_backend(extra_code="time.sleep(30)")
        started = time.monotonic()
        with self.assertRaisesRegex(ProcessError, "timed out"):
            self.propose(agent, timeout=0.3)
        self.assertLess(time.monotonic() - started, 3)

    def test_malformed_backend_output_is_a_failure(self):
        agent = self.make_parent_backend(output="A suggestion without a patch.")
        with self.assertRaisesRegex(ProcessError, "malformed unified diff"):
            self.propose(agent)


if __name__ == "__main__":
    unittest.main()
