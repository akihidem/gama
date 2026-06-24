import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.benchmark import (
    BRUTAL_SUITE, DEFAULT_SUITE, HARD_SUITE, SUITES, BenchCase, _check_func,
    _extract_code, propose_routing_table, run_bench, score_output, summarize,
)
from gama.cli import build_parser
from gama.logger import ExecutionLogger
from gama.models import ModelTier

VALID_CLASSES = {"code_implementation", "qa", "research", "content", "integration"}


class Canned(ModelBackend):
    """Fake backend returning a fixed string regardless of prompt."""

    available = True

    def __init__(self, reply):
        self.reply = reply
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        return self.reply


# Tiny suite: two classes, marker-string checkers -> deterministic winners.
SUITE = [
    BenchCase("c1", "code_implementation", "p", lambda o: "ALPHA" in o),
    BenchCase("c2", "qa", "p", lambda o: 1.0 if "BETA" in o else 0.0),
]


class TestScoreOutput(unittest.TestCase):
    def test_bool_true(self):
        self.assertEqual(score_output(BenchCase("x", "qa", "p", lambda o: True), "z"), 1.0)

    def test_float_clamped(self):
        self.assertEqual(score_output(BenchCase("x", "qa", "p", lambda o: 2.5), "z"), 1.0)
        self.assertEqual(score_output(BenchCase("x", "qa", "p", lambda o: -1), "z"), 0.0)

    def test_checker_exception_is_zero(self):
        def boom(o):
            raise ValueError("x")
        self.assertEqual(score_output(BenchCase("x", "qa", "p", boom), "z"), 0.0)


class TestRunBench(unittest.TestCase):
    def setUp(self):
        self.backends = {"alpha": Canned("ALPHA wins"), "beta": Canned("BETA wins")}

    def test_records_shape(self):
        recs = run_bench(self.backends, suite=SUITE, tier=ModelTier.SMALL)
        self.assertEqual(len(recs), 4)  # 2 backends x 2 cases x 1 repeat
        self.assertTrue(all(r["backend"] in ("alpha", "beta") for r in recs))

    def test_proposal_picks_per_class_winner(self):
        prop = propose_routing_table(run_bench(self.backends, suite=SUITE, tier=ModelTier.SMALL))
        self.assertEqual(prop["routing_table"]["code_implementation"], "alpha")
        self.assertEqual(prop["routing_table"]["qa"], "beta")

    def test_summarize_scores(self):
        summ = summarize(run_bench(self.backends, suite=SUITE, tier=ModelTier.SMALL))
        self.assertEqual(summ["by_class"]["code_implementation"]["alpha"]["score"], 1.0)
        self.assertEqual(summ["by_class"]["code_implementation"]["beta"]["score"], 0.0)

    def test_deterministic(self):
        a = propose_routing_table(run_bench(self.backends, suite=SUITE, tier=ModelTier.SMALL))
        b = propose_routing_table(run_bench(self.backends, suite=SUITE, tier=ModelTier.SMALL))
        self.assertEqual(a["routing_table"], b["routing_table"])

    def test_limit_per_class_on_default_suite(self):
        recs = run_bench(self.backends, tier=ModelTier.SMALL, limit_per_class=1)
        self.assertEqual(len({r["task_type"] for r in recs}), 5)  # 5 classes
        per = {}
        for r in recs:
            per[(r["backend"], r["task_type"])] = per.get((r["backend"], r["task_type"]), 0) + 1
        self.assertTrue(all(v == 1 for v in per.values()))

    def test_ledger_is_logrecord_compatible(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bench.jsonl"
            run_bench(self.backends, suite=SUITE, tier=ModelTier.SMALL,
                      logger=ExecutionLogger(p), run_id="t")
            rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["run_id"], "t")
            self.assertIn(rows[0]["selected_model"], ("alpha", "beta"))
            self.assertIn("review_score", rows[0])


class TestCodeExtraction(unittest.TestCase):
    """A model that wraps code in prose + fences must still be scored on correctness."""

    def test_fenced_code_in_prose(self):
        reply = "Sure! Here is the function:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
        self.assertEqual(_extract_code(reply).strip(), "def f(x):\n    return x + 1")

    def test_raw_code_passthrough(self):
        self.assertIn("def f", _extract_code("def f(x):\n    return x + 1\n"))

    def test_check_func_with_prose_wrapped_code(self):
        reply = "Here you go:\n```python\ndef nth(n):\n    return n * n\n```\nThat squares it."
        self.assertEqual(_check_func(reply, "nth", [((3,), 9), ((4,), 16)]), 1.0)

    def test_check_func_picks_longest_block(self):
        reply = "```\nx = 1\n```\nthen the real one:\n```python\ndef g(a):\n    return a * 2\n```"
        self.assertEqual(_check_func(reply, "g", [((5,), 10)]), 1.0)


class TestNamedSuites(unittest.TestCase):
    def test_registry_keys(self):
        self.assertEqual(set(SUITES), {"default", "hard", "brutal"})

    def test_default_is_unchanged(self):
        self.assertIs(SUITES["default"], DEFAULT_SUITE)

    def test_task_types_are_real_classes(self):
        for name, suite in SUITES.items():
            for c in suite:
                self.assertIn(c.task_type, VALID_CLASSES, f"{name}:{c.case_id}")

    def test_case_ids_unique_within_suite(self):
        for name, suite in SUITES.items():
            ids = [c.case_id for c in suite]
            self.assertEqual(len(ids), len(set(ids)), name)

    def test_hard_and_brutal_have_cases(self):
        self.assertGreaterEqual(len(HARD_SUITE), 8)
        self.assertGreaterEqual(len(BRUTAL_SUITE), 4)

    def test_run_bench_accepts_named_suite(self):
        recs = run_bench({"x": Canned("9")}, suite=SUITES["hard"],
                         tier=ModelTier.SMALL, limit_per_class=1)
        self.assertTrue(recs)
        self.assertTrue(all(r["task_type"] in VALID_CLASSES for r in recs))


class TestHardBrutalCheckers(unittest.TestCase):
    """Every discriminating case must score a correct answer 1.0 and a wrong answer
    < 1.0. That property is exactly what lets the suite *separate* backends."""

    _LONGPAL_OK = (
        "def longest_palindrome(s):\n"
        "    best = ''\n"
        "    for i in range(len(s)):\n"
        "        for j in range(i, len(s)):\n"
        "            sub = s[i:j + 1]\n"
        "            if sub == sub[::-1] and len(sub) > len(best):\n"
        "                best = sub\n"
        "    return best\n"
    )
    _MERGE_OK = (
        "def merge_intervals(intervals):\n"
        "    intervals = sorted(intervals)\n"
        "    out = []\n"
        "    for x in intervals:\n"
        "        if out and x[0] <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], x[1])\n"
        "        else:\n"
        "            out.append(list(x))\n"
        "    return out\n"
    )

    GOOD = {
        "hard-code-longpal": _LONGPAL_OK,
        "hard-code-merge": _MERGE_OK,
        "hard-math-mult": "1059",
        "hard-math-modexp": "9",
        "hard-reason-weekday": "Friday",
        "hard-reason-lookandsay": "312211",
        "hard-write-acrostic": ("Curious machine hums\nObserve a machine learn\n"
                                "Deep machine dreams\nEvery machine wakes"),
        "hard-write-primelist": "53,59,61,67",
        "hard-tool-json-nested": ('{"tool": "search", "args": {"query": "gama", '
                                  '"limit": 5}, "tags": ["a", "b", "c"]}'),
        "hard-tool-json-squares": "[1, 4, 9, 16, 25]",
        "brutal-qa-trailzeros": "24",
        "brutal-qa-powmod": "624",
        "brutal-research-knights": "knight",
        "brutal-research-distinct": "648",
        "brutal-content-palindrome": "A man, a plan, a canal, Panama",
        "brutal-content-alliteration": "Peter picked plump purple plums past peculiar ponds",
    }
    BAD = {
        "hard-code-longpal": "def longest_palindrome(s):\n    return ''\n",
        "hard-code-merge": "def merge_intervals(intervals):\n    return intervals\n",
        "hard-math-mult": "1000",
        "hard-math-modexp": "1",
        "hard-reason-weekday": "Monday",
        "hard-reason-lookandsay": "111221",
        "hard-write-acrostic": "Apple pie\nBanana bread\nCherry cake\nDate loaf",
        "hard-write-primelist": "53,59,61,67,71",
        "hard-tool-json-nested": "{}",
        "hard-tool-json-squares": "[1, 2, 3, 4, 5]",
        "brutal-qa-trailzeros": "20",
        "brutal-qa-powmod": "1000",
        "brutal-research-knights": "knave",
        "brutal-research-distinct": "1000",
        "brutal-content-palindrome": "hello world this is not",
        "brutal-content-alliteration": "Peter picked plums",
    }

    def _cases(self):
        return HARD_SUITE + BRUTAL_SUITE

    def test_fixtures_cover_every_case(self):
        ids = {c.case_id for c in self._cases()}
        self.assertEqual(ids, set(self.GOOD))
        self.assertEqual(ids, set(self.BAD))

    def test_correct_answers_score_one(self):
        for c in self._cases():
            self.assertEqual(score_output(c, self.GOOD[c.case_id]), 1.0, c.case_id)

    def test_wrong_answers_score_below_one(self):
        for c in self._cases():
            self.assertLess(score_output(c, self.BAD[c.case_id]), 1.0, c.case_id)


class TestBenchCli(unittest.TestCase):
    def test_suite_flag_parses(self):
        self.assertEqual(build_parser().parse_args(["bench", "--suite", "hard"]).suite, "hard")

    def test_suite_defaults_to_default(self):
        self.assertEqual(build_parser().parse_args(["bench"]).suite, "default")

    def test_bad_suite_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["bench", "--suite", "nope"])


if __name__ == "__main__":
    unittest.main()
