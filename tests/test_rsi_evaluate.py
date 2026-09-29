"""Exercise RSI scoring with real, small subprocesses and temporary files."""

import json
import math
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from gama.rsi_evaluate import Evaluation, EvaluationError, evaluate, promotion, run_checks
from gama.rsi_process import ProcessError


class CommandTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)

    def python(self, source, *args):
        return [sys.executable, "-c", source, *args]

    def output(self, stdout, stderr="", returncode=0):
        (self.cwd / "stdout.txt").write_text(stdout, encoding="utf-8")
        (self.cwd / "stderr.txt").write_text(stderr, encoding="utf-8")
        return self.python(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.stdout.write(Path('stdout.txt').read_text(encoding='utf-8'))\n"
            "sys.stderr.write(Path('stderr.txt').read_text(encoding='utf-8'))\n"
            f"sys.exit({returncode})\n"
        )

    def invoke(self, operation, command, *, timeout=5, cancel=None):
        if operation == "evaluate":
            return evaluate(command, cwd=self.cwd, timeout=timeout, cancel=cancel)
        return run_checks([command], cwd=self.cwd, timeout=timeout, cancel=cancel)


class TestEvaluate(CommandTestCase):
    def test_default_repeat_preserves_json_details(self):
        details = {"score": 0.75, "cases": [1, 0, 1], "metadata": {"ok": True, "name": "評価"}}
        result = evaluate(
            self.output(" \n" + json.dumps(details) + "\t\n", "diagnostic\n"),
            cwd=self.cwd, timeout=5,
        )
        self.assertIsInstance(result, Evaluation)
        self.assertIsInstance(result.score, float)
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.samples, (0.75,))
        self.assertEqual(result.results, (details,))

    def test_accepts_integer_endpoints_and_numeric_scores(self):
        for score in (0, 1, 0.0, 1.0, 0.125, 1e-10):
            with self.subTest(score=score):
                result = evaluate(
                    self.output(json.dumps({"score": score})), cwd=self.cwd, timeout=5,
                )
                self.assertEqual(result.score, score)
                self.assertIsInstance(result.samples[0], float)

    def test_repeats_are_fresh_processes_and_use_arithmetic_mean(self):
        command = self.python(
            "import json, os\n"
            "from pathlib import Path\n"
            "counter = Path('counter.txt')\n"
            "index = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(index + 1))\n"
            "print(json.dumps({'score': [0.2, 0.5, 0.9][index], "
            "'repeat': index, 'pid': os.getpid(), 'details': {'count': index + 1}}))\n"
        )
        result = evaluate(command, cwd=self.cwd, timeout=5, repeats=3)
        self.assertAlmostEqual(result.score, 1.6 / 3)
        self.assertEqual(result.samples, (0.2, 0.5, 0.9))
        self.assertEqual([row["repeat"] for row in result.results], [0, 1, 2])
        self.assertEqual([row["details"] for row in result.results],
                         [{"count": 1}, {"count": 2}, {"count": 3}])
        pids = {row["pid"] for row in result.results}
        self.assertEqual(len(pids), 3)
        self.assertNotIn(os.getpid(), pids)
        self.assertEqual((self.cwd / "counter.txt").read_text(), "3")

    def test_rejects_output_that_is_not_exactly_one_json_object(self):
        outputs = [
            "", " \n\t", "not JSON", "[]", "null", "0.5", '"score"',
            '{"score":', '{"score": 0.5,}', '{"score": 0.5}\n{"score": 1}',
            'log line\n{"score": 0.5}', '{"score": 0.5}\nlog line',
            '```json\n{"score": 0.5}\n```', '\ufeff{"score": 0.5}',
            '{"score": 0, "score": 1}',
            '{"score": 0.5, "details": {"value": 0, "value": 1}}',
            '{"score": 0.5, "details": NaN}',
            '{"score": 0.5, "details": Infinity}',
        ]
        for output in outputs:
            with self.subTest(output=output):
                with self.assertRaises(EvaluationError):
                    evaluate(self.output(output), cwd=self.cwd, timeout=5)

    def test_rejects_missing_or_invalid_scores(self):
        outputs = ["{}"] + [
            '{"score": ' + value + "}"
            for value in (
                "true", "false", "null", '"0.5"', "[]", "{}",
                "NaN", "Infinity", "-Infinity", "1e309", "-1e309",
                "-0.001", "1.001", "9" * 500,
            )
        ]
        for output in outputs:
            with self.subTest(output=output):
                with self.assertRaises(EvaluationError):
                    evaluate(self.output(output), cwd=self.cwd, timeout=5)

    def test_deeply_nested_invalid_json_is_an_evaluation_error(self):
        output = '{"score": 0.5, "details": ' + "[" * 2000 + "0" + "]" * 1999 + "}"
        with self.assertRaises(EvaluationError):
            evaluate(self.output(output), cwd=self.cwd, timeout=5)

    def test_valid_json_with_nonzero_exit_is_rejected(self):
        command = self.output('{"score": 1}', "checker broke", returncode=7)
        with self.assertRaises(EvaluationError) as caught:
            evaluate(command, cwd=self.cwd, timeout=5)
        self.assertIn("code 7", str(caught.exception))
        self.assertIn("checker broke", str(caught.exception))
        self.assertIn('{"score": 1}', str(caught.exception))

    def test_later_failed_repeat_stops_without_partial_evaluation(self):
        for failure in ("print('invalid JSON')", "print('{\"score\": 1}'); sys.exit(3)"):
            with self.subTest(failure=failure):
                counter = self.cwd / "counter.txt"
                counter.write_text("0")
                command = self.python(
                    "import sys\n"
                    "from pathlib import Path\n"
                    "counter = Path('counter.txt')\n"
                    "index = int(counter.read_text())\n"
                    "counter.write_text(str(index + 1))\n"
                    "if index == 1:\n"
                    f"    {failure}\n"
                    "else:\n"
                    "    print('{\"score\": 0.5}')\n"
                )
                with self.assertRaisesRegex(EvaluationError, "repeat 2/3"):
                    evaluate(command, cwd=self.cwd, timeout=5, repeats=3)
                self.assertEqual(counter.read_text(), "2")

    def test_rejects_invalid_repeats_before_running(self):
        command = self.python("from pathlib import Path; Path('started').touch()")
        for repeats in (0, -1, True, False, 1.5, "2", None, math.inf, math.nan):
            with self.subTest(repeats=repeats):
                with self.assertRaisesRegex(EvaluationError, "repeats"):
                    evaluate(command, cwd=self.cwd, timeout=5, repeats=repeats)
        self.assertFalse((self.cwd / "started").exists())

    def test_command_arguments_are_literal(self):
        argument = "a value; $(touch injected) * ' \""
        command = self.python(
            "import json, sys\n"
            "print(json.dumps({'score': 0.5, 'argument': sys.argv[1]}))\n",
            argument,
        )
        result = evaluate(command, cwd=self.cwd, timeout=5)
        self.assertEqual(result.results[0]["argument"], argument)
        self.assertFalse((self.cwd / "injected").exists())


class TestRunChecks(CommandTestCase):
    def test_no_checks_returns_empty_receipts(self):
        self.assertEqual(run_checks([], cwd=self.cwd, timeout=5), [])

    def test_checks_run_in_order_in_cwd_and_retain_receipts(self):
        commands = [
            self.python(
                "import sys\n"
                "from pathlib import Path\n"
                "Path('state.txt').write_text('first')\n"
                "print('first passed')\n"
                "print('first diagnostic', file=sys.stderr)\n"
            ),
            self.python(
                "from pathlib import Path\n"
                "state = Path('state.txt')\n"
                "assert state.read_text() == 'first'\n"
                "state.write_text('second')\n"
                "print('second passed')\n"
            ),
        ]
        results = run_checks(commands, cwd=self.cwd, timeout=5)
        self.assertEqual((self.cwd / "state.txt").read_text(), "second")
        self.assertEqual([row["command"] for row in results], commands)
        self.assertEqual([row["returncode"] for row in results], [0, 0])
        self.assertEqual([row["stdout"] for row in results],
                         ["first passed\n", "second passed\n"])
        self.assertEqual([row["stderr"] for row in results], ["first diagnostic\n", ""])
        self.assertTrue(all(math.isfinite(row["elapsed_s"]) and row["elapsed_s"] >= 0
                            for row in results))

    def test_failed_check_prevents_later_checks(self):
        commands = [
            self.python("from pathlib import Path; Path('first').touch()"),
            self.output("check stdout", "check stderr", returncode=4),
            self.python("from pathlib import Path; Path('last').touch()"),
        ]
        with self.assertRaises(EvaluationError) as caught:
            run_checks(commands, cwd=self.cwd, timeout=5)
        message = str(caught.exception)
        self.assertIn("check 2", message)
        self.assertIn("code 4", message)
        self.assertIn("check stdout", message)
        self.assertIn("check stderr", message)
        self.assertTrue((self.cwd / "first").exists())
        self.assertFalse((self.cwd / "last").exists())

    def test_commands_are_validated_before_any_check_runs(self):
        command = self.python("from pathlib import Path; Path('started').touch()")
        with self.assertRaises(EvaluationError):
            run_checks([command, []], cwd=self.cwd, timeout=5)
        self.assertFalse((self.cwd / "started").exists())
        for commands in (None, "echo pass"):
            with self.subTest(commands=commands):
                with self.assertRaises(EvaluationError):
                    run_checks(commands, cwd=self.cwd, timeout=5)


class TestProcessFailures(CommandTestCase):
    def test_failure_feedback_is_bounded_and_includes_both_streams(self):
        stdout = "stdout start\n" + "x" * 20000 + "\nstdout end"
        stderr = "stderr start\n" + "y" * 20000 + "\nstderr end"
        command = self.output(stdout, stderr, returncode=2)
        for operation in ("evaluate", "checks"):
            with self.subTest(operation=operation):
                with self.assertRaises(EvaluationError) as caught:
                    self.invoke(operation, command)
                message = str(caught.exception)
                self.assertLess(len(message), 10000)
                for marker in ("stdout start", "stdout end", "stderr start", "stderr end"):
                    self.assertIn(marker, message)
                self.assertIn("truncated", message)

    def test_timeout_is_wrapped_as_evaluation_error(self):
        command = self.python("import time; time.sleep(10)")
        for operation in ("evaluate", "checks"):
            with self.subTest(operation=operation):
                with self.assertRaises(EvaluationError) as caught:
                    self.invoke(operation, command, timeout=0.1)
                self.assertIsInstance(caught.exception.__cause__, ProcessError)

    def test_cancellation_is_forwarded_and_prevents_execution(self):
        cancel = threading.Event()
        cancel.set()
        command = self.python("from pathlib import Path; Path('started').touch()")
        for operation in ("evaluate", "checks"):
            with self.subTest(operation=operation):
                with self.assertRaises(EvaluationError) as caught:
                    self.invoke(operation, command, cancel=cancel)
                self.assertIsInstance(caught.exception.__cause__, ProcessError)
                self.assertFalse((self.cwd / "started").exists())

    def test_output_overflow_is_rejected_by_process_helper(self):
        for stream in ("stdout", "stderr"):
            command = self.python(
                "import sys\n"
                "print('{\"score\": 0.5}', flush=True)\n"
                f"sys.{stream}.write('x' * (1048576 + 1))\n"
            )
            for operation in ("evaluate", "checks"):
                with self.subTest(stream=stream, operation=operation):
                    with self.assertRaises(EvaluationError) as caught:
                        self.invoke(operation, command)
                    self.assertIsInstance(caught.exception.__cause__, ProcessError)

    def test_missing_executable_is_an_evaluation_error(self):
        command = [str(self.cwd / "missing-executable")]
        for operation in ("evaluate", "checks"):
            with self.subTest(operation=operation):
                with self.assertRaises(EvaluationError):
                    self.invoke(operation, command)

    def test_invalid_timeout_prevents_execution(self):
        command = self.python("from pathlib import Path; Path('started').touch()")
        for timeout in (0, -1, math.nan, math.inf, -math.inf, True, "1", None):
            for operation in ("evaluate", "checks"):
                with self.subTest(timeout=timeout, operation=operation):
                    with self.assertRaisesRegex(EvaluationError, "timeout"):
                        self.invoke(operation, command, timeout=timeout)
        self.assertFalse((self.cwd / "started").exists())

    def test_rejects_invalid_argument_lists(self):
        for command in ([], "", "echo pass", [""], [sys.executable, None], ["bad\0name"]):
            for operation in ("evaluate", "checks"):
                with self.subTest(command=command, operation=operation):
                    with self.assertRaises(EvaluationError):
                        self.invoke(operation, command)


class TestPromotion(unittest.TestCase):
    def evaluation(self, samples, score=0.5):
        return Evaluation(score=score, samples=samples, results=())

    def test_nonoverlapping_samples_pass(self):
        ok, reason = promotion(
            self.evaluation((0.8, 0.9)), self.evaluation((0.4, 0.5)), min_gain=0.1,
        )
        self.assertTrue(ok)
        self.assertIn("conservative non-overlap", reason)

    def test_tie_touching_overlap_and_worse_samples_fail(self):
        for candidate, incumbent in (
            ((0.5,), (0.5,)),
            ((0.5, 0.8), (0.2, 0.5)),
            ((0.5, 0.9, 0.9), (0.2, 0.6)),
            ((0.1, 0.2), (0.3, 0.4)),
        ):
            with self.subTest(candidate=candidate, incumbent=incumbent):
                ok, reason = promotion(self.evaluation(candidate), self.evaluation(incumbent))
                self.assertFalse(ok)
                self.assertTrue(reason)

    def test_minimum_gain_is_strict_without_tolerance(self):
        incumbent = self.evaluation((0.5,))
        boundary = self.evaluation((0.75,))
        self.assertFalse(promotion(boundary, incumbent, min_gain=0.25)[0])
        self.assertTrue(promotion(boundary, incumbent, min_gain=0.125)[0])
        just_above = self.evaluation((math.nextafter(0.75, 1.0),))
        self.assertTrue(promotion(just_above, incumbent, min_gain=0.25)[0])

    def test_gate_uses_samples_instead_of_aggregate_score(self):
        self.assertFalse(promotion(
            self.evaluation((0.4, 0.9), score=1.0),
            self.evaluation((0.5, 0.6), score=0.0),
        )[0])
        self.assertTrue(promotion(
            self.evaluation((0.8,), score=0.0),
            self.evaluation((0.5, 0.6), score=1.0),
        )[0])

    def test_rejects_invalid_minimum_gain(self):
        for gain in (-0.01, math.nan, math.inf, -math.inf, True, "0", None, 10 ** 400):
            with self.subTest(gain=gain):
                with self.assertRaisesRegex(EvaluationError, "min_gain"):
                    promotion(self.evaluation((0.9,)), self.evaluation((0.5,)), gain)

    def test_empty_or_bad_sample_sets_fail_for_either_side(self):
        invalid_sets = [
            (), [], None, "0.8", {"score": 0.8}, {0.8},
            (math.nan, 0.9), (0.9, math.nan), (math.inf,), (-math.inf,),
            (True,), (False,), ("0.8",), (None,), ([],), ({},),
            (-0.01,), (1.01,), (10 ** 400,),
        ]
        for samples in invalid_sets:
            bad = self.evaluation(samples)
            for label, candidate, incumbent in (
                ("candidate", bad, self.evaluation((0.0,))),
                ("incumbent", self.evaluation((1.0,)), bad),
            ):
                with self.subTest(samples=samples, side=label):
                    ok, reason = promotion(candidate, incumbent)
                    self.assertFalse(ok)
                    self.assertIn(label, reason)
                    self.assertIn("samples", reason)


if __name__ == "__main__":
    unittest.main()
