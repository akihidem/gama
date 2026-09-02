import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import SshOpenAIBackend


class TestSshOpenAIBackendRemoteCmd(unittest.TestCase):
    def test_default_path_produces_expected_url(self):
        be = SshOpenAIBackend(ssh_host="user@host", port=8080)
        self.assertIn("http://localhost:8080/v1/chat/completions", be._remote_cmd())

    def test_malicious_path_cannot_break_out_of_quoting(self):
        # `path` is config-controlled; the remote command is executed through a shell on
        # the remote host, so a naive `'{url}'` interpolation lets a path containing a
        # single quote inject arbitrary remote shell syntax.
        be = SshOpenAIBackend(ssh_host="user@host", port=8080,
                              path="/x'; touch /tmp/pwned; echo '")
        cmd = be._remote_cmd()
        # shlex.split must reproduce exactly one curl invocation with one URL argument --
        # if the injection worked, splitting would instead surface extra shell tokens
        # (a bare `touch` command, a stray `echo`, etc.) after unquoting.
        tokens = __import__("shlex").split(cmd)
        self.assertEqual(tokens[0:2], ["curl", "-sS"])
        self.assertNotIn("touch", tokens)
        self.assertNotIn("/tmp/pwned", tokens)

    def test_ssh_cmd_keeps_remote_command_as_a_single_argv_element(self):
        be = SshOpenAIBackend(ssh_host="user@host", port=8080)
        argv = be._ssh_cmd()
        self.assertEqual(argv[0], "ssh")
        self.assertEqual(argv[-2], "user@host")
        self.assertEqual(argv[-1], be._remote_cmd())


class TestSshOpenAIBackendSaysWhyItFailed(unittest.TestCase):
    """走行を止めた出来事の記録はこの 1 行だけ(台帳は点しか持たない・trace は例外の文字列を
    そのまま持つ)。理由の落ちた行は「止まった」以上のことを言えない。"""

    def _run(self, rc=0, out="", err=""):
        import subprocess
        import types
        be = SshOpenAIBackend(ssh_host="h", port=8000)
        calls = []

        def fake_run(argv, **kw):
            calls.append(argv)
            return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)

        real, subprocess.run = subprocess.run, fake_run
        try:
            from gama.models import ModelTier
            return be.complete("hi", ModelTier.LARGE)
        finally:
            subprocess.run = real

    def test_a_dead_server_is_named_with_curls_own_reason(self):
        with self.assertRaises(RuntimeError) as cm:
            self._run(rc=7, err="curl: (7) Failed to connect to localhost port 8000")
        self.assertIn("Failed to connect", str(cm.exception))
        self.assertIn("h:8000/v1/chat/completions", str(cm.exception))

    def test_a_failure_with_no_stderr_still_says_so_rather_than_ending_in_a_colon(self):
        with self.assertRaises(RuntimeError) as cm:
            self._run(rc=255, err="")
        self.assertIn("no stderr (exit 255)", str(cm.exception))
        self.assertFalse(str(cm.exception).rstrip().endswith(":"))

    def test_a_non_json_reply_carries_the_first_bytes(self):
        with self.assertRaises(RuntimeError) as cm:
            self._run(rc=0, out="<html>502 Bad Gateway</html>")
        self.assertIn("no JSON", str(cm.exception))
        self.assertIn("502 Bad Gateway", str(cm.exception))

    def test_a_json_error_reply_is_not_read_as_an_empty_answer(self):
        with self.assertRaises(RuntimeError) as cm:
            self._run(rc=0, out='{"error": {"message": "model not found"}}')
        self.assertIn("no choices", str(cm.exception))
        self.assertIn("model not found", str(cm.exception))

    def test_a_json_array_reply_is_refused_with_the_same_diagnostic(self):
        with self.assertRaises(RuntimeError) as cm:
            self._run(rc=0, out="[]")
        self.assertIn("no choices", str(cm.exception))

    def test_a_good_reply_still_returns_its_content(self):
        out = ('{"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}], '
               '"usage": {"prompt_tokens": 3, "completion_tokens": 1}, "model": "kimi"}')
        self.assertEqual(self._run(rc=0, out=out), "hello")


if __name__ == "__main__":
    unittest.main()
