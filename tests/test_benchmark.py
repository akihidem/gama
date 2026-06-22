import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.benchmark import (
    BenchCase, _check_func, _extract_code, propose_routing_table, run_bench,
    score_output, summarize,
)
from gama.logger import ExecutionLogger
from gama.models import ModelTier


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


if __name__ == "__main__":
    unittest.main()
