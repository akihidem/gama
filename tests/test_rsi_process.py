import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest

from gama.rsi_process import ProcessError, run_process


class TestRSIProcess(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gama-rsi-process-test-")
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)

    def run_code(self, code, **kwargs):
        return run_process(
            [sys.executable, "-I", "-c", code], cwd=self.cwd, **kwargs)

    def test_complete_output_stdin_cwd_and_inherited_environment(self):
        text = "source 日本語\n" * 10000
        result = self.run_code(
            "import json, os, sys\n"
            "sys.stdout.write(sys.stdin.read())\n"
            "sys.stderr.write(json.dumps({'cwd': os.getcwd(), "
            "'home_value': os.environ.get('HOME')}))\n",
            input_text=text,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, text)
        self.assertEqual(json.loads(result.stderr), {
            "cwd": str(self.cwd), "home_value": os.environ.get("HOME")})
        self.assertGreater(result.elapsed_s, 0)

    def test_arguments_are_not_interpreted_by_a_shell(self):
        literal = "$(touch should-not-exist); `echo surprise`"
        result = run_process(
            [sys.executable, "-I", "-c", "import sys; print(sys.argv[1])", literal],
            cwd=self.cwd,
        )
        self.assertEqual(result.stdout, literal + "\n")
        self.assertFalse((self.cwd / "should-not-exist").exists())

    def test_explicit_environment_is_passed_without_modification(self):
        process_env = dict(os.environ, GAMA_RSI_PROCESS_TEST="specific-value")
        original = process_env.copy()
        result = self.run_code(
            "import json, os; print(json.dumps(["
            "os.environ.get('GAMA_RSI_PROCESS_TEST'), os.environ.get('HOME')]))",
            env=process_env,
        )
        self.assertEqual(
            json.loads(result.stdout), ["specific-value", os.environ.get("HOME")])
        self.assertEqual(process_env, original)

    def test_nonzero_exit_is_available_to_evaluators(self):
        result = self.run_code(
            "import sys; print('partial result'); "
            "sys.stderr.write('check failed'); sys.exit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "partial result\n")
        self.assertEqual(result.stderr, "check failed")

    def test_timeout_with_unread_large_stdin_and_stderr_diagnostic(self):
        started = time.monotonic()
        with self.assertRaisesRegex(ProcessError, "timed out.*waiting for input"):
            self.run_code(
                "import sys, time; print('waiting for input', file=sys.stderr, flush=True); "
                "time.sleep(30)",
                input_text="x" * 1048576, timeout=0.3,
            )
        self.assertLess(time.monotonic() - started, 3)

    def test_cancellation_during_execution(self):
        cancel = threading.Event()
        timer = threading.Timer(0.3, cancel.set)
        self.addCleanup(timer.cancel)
        timer.start()
        started = time.monotonic()
        with self.assertRaisesRegex(ProcessError, "cancelled.*working"):
            self.run_code(
                "import sys, time; print('working', file=sys.stderr, flush=True); "
                "time.sleep(30)",
                timeout=10, cancel=cancel,
            )
        self.assertLess(time.monotonic() - started, 3)

    def test_preexisting_cancellation_does_not_launch_a_process(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(ProcessError, "cancelled before launch"):
            self.run_code(
                "from pathlib import Path; Path('launched').touch()", cancel=cancel)
        self.assertFalse((self.cwd / "launched").exists())

    def test_fast_output_overflow_is_a_failure_not_truncated_success(self):
        with self.assertRaisesRegex(ProcessError, "output limit.*1000 bytes"):
            self.run_code("import sys; sys.stdout.write('x' * 1001)", max_output_bytes=1000)

    def test_output_limit_applies_to_both_streams_combined(self):
        with self.assertRaisesRegex(ProcessError, "combined output limit"):
            self.run_code(
                "import sys; sys.stdout.write('x' * 600); sys.stderr.write('y' * 600)",
                max_output_bytes=1000,
            )

    def test_live_output_overflow_stops_process_and_bounds_diagnostic(self):
        started = time.monotonic()
        with self.assertRaises(ProcessError) as caught:
            self.run_code(
                "import sys, time\n"
                "sys.stderr.write('irrelevant-prefix\\n' + 'x' * 20000 + '\\ntail-marker')\n"
                "sys.stderr.flush()\n"
                "time.sleep(30)\n",
                max_output_bytes=1000, timeout=10,
            )
        message = str(caught.exception)
        self.assertIn("output limit", message)
        self.assertIn("tail-marker", message)
        self.assertNotIn("irrelevant-prefix", message)
        self.assertLess(len(message), 4500)
        self.assertLess(time.monotonic() - started, 3)

    def test_invalid_utf8_is_not_silently_replaced_in_a_patch_or_score(self):
        with self.assertRaisesRegex(ProcessError, "invalid UTF-8.*diagnostic"):
            self.run_code(
                "import os; os.write(1, b'\\xff'); os.write(2, b'diagnostic')")

    def test_launch_error_names_the_missing_executable(self):
        with self.assertRaisesRegex(ProcessError, "could not start.*missing-executable"):
            run_process([str(self.cwd / "missing-executable")], cwd=self.cwd)

    def test_invalid_limits_and_commands_are_rejected_before_launch(self):
        for timeout in [0, -1, float("inf"), float("nan"), True]:
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.run_code("raise AssertionError('launched')", timeout=timeout)
        for limit in [0, -1, 1.5, True]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.run_code("raise AssertionError('launched')", max_output_bytes=limit)
        for command in [[], "echo unsafe", [""], ["echo", "\0"], ["echo", 1]]:
            with self.subTest(command=command), self.assertRaises(ValueError):
                run_process(command, cwd=self.cwd)

    def _child_is_running(self, pid):
        try:
            status = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return False
        return status.rsplit(")", 1)[1].strip().split()[0] not in {"Z", "X"}

    @unittest.skipUnless(sys.platform.startswith("linux"), "inspects Linux process states")
    def test_timeout_and_cancel_kill_descendants_that_ignore_sigterm(self):
        for cancellation in (False, True):
            with self.subTest(cancellation=cancellation):
                pid_file = self.cwd / f"child-{cancellation}.json"
                cancel = threading.Event() if cancellation else None
                child_code = (
                    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "time.sleep(30)")
                code = (
                    "import json, os, pathlib, subprocess, sys, time\n"
                    f"child = subprocess.Popen([sys.executable, '-I', '-c', {child_code!r}])\n"
                    f"pathlib.Path({str(pid_file)!r}).write_text("
                    "json.dumps({'child': child.pid, 'group': os.getpid()}))\n"
                    "time.sleep(30)\n"
                )
                timer = threading.Timer(0.4, cancel.set) if cancel else None
                if timer:
                    timer.start()
                try:
                    with self.assertRaisesRegex(
                            ProcessError, "cancelled" if cancellation else "timed out"):
                        self.run_code(code, timeout=10 if cancel else 0.4, cancel=cancel)
                    ids = json.loads(pid_file.read_text())
                    deadline = time.monotonic() + 1
                    while self._child_is_running(ids["child"]) and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertFalse(self._child_is_running(ids["child"]))
                finally:
                    if timer:
                        timer.cancel()
                    if pid_file.exists():
                        group = json.loads(pid_file.read_text())["group"]
                        try:
                            os.killpg(group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
