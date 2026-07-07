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
        self.assertEqual(tokens[0:2], ["curl", "-s"])
        self.assertNotIn("touch", tokens)
        self.assertNotIn("/tmp/pwned", tokens)

    def test_ssh_cmd_keeps_remote_command_as_a_single_argv_element(self):
        be = SshOpenAIBackend(ssh_host="user@host", port=8080)
        argv = be._ssh_cmd()
        self.assertEqual(argv[0], "ssh")
        self.assertEqual(argv[-2], "user@host")
        self.assertEqual(argv[-1], be._remote_cmd())


if __name__ == "__main__":
    unittest.main()
