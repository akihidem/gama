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


class TestSandboxedCheck(unittest.TestCase):
    """Cancellation floor: 生成コード 1 件が採点側を止められないこと。

    2026-08-31 の実験1で TinyLlama の非停止コードが in-process exec の採点を
    実運用時間内に返さなくした(gama-runs/kl-vs-beta 逸脱1)。この床は
    「タイムアウト⇒不正解(0.0)・壁時計が上限で切れる」を機械判定する。
    """

    def _short_timeout(self, seconds):
        import gama.benchmark as bm
        orig = bm._CHECK_TIMEOUT_S
        bm._CHECK_TIMEOUT_S = seconds
        self.addCleanup(setattr, bm, "_CHECK_TIMEOUT_S", orig)

    def test_nonterminating_function_scores_zero_within_budget(self):
        self._short_timeout(1.0)
        import time as _t
        t0 = _t.monotonic()
        score = _check_func("def f(x):\n    while True:\n        pass\n",
                            "f", [((1,), 1)])
        self.assertEqual(score, 0.0)
        # 上限 1.0s + 子の起動/後始末。5s を超えたら打ち切りが機能していない。
        self.assertLess(_t.monotonic() - t0, 5.0)

    def test_nonterminating_toplevel_scores_zero(self):
        self._short_timeout(1.0)
        score = _check_func("while True:\n    pass\n", "f", [((1,), 1)])
        self.assertEqual(score, 0.0)

    def test_hang_mid_suite_keeps_earlier_partial_credit(self):
        # summarize/routing は平均 score(部分点)で勝者を決めるので、hang した
        # 1 件が「そこまでに完走した正解」まで消してはいけない。子は cell を
        # 逐次送る設計なので、hang の前は生き、hang 以降だけ不合格になる。
        self._short_timeout(1.0)
        code = ("def f(x):\n"
                "    if x == 0:\n"
                "        while True:\n"
                "            pass\n"
                "    return x\n")
        # hang が先頭: 何も完走していないので 0.0
        self.assertEqual(_check_func(code, "f", [((0,), 0), ((2,), 2)]), 0.0)
        # hang が 2 件目: 1 件目の正解は保持され 0.5
        self.assertEqual(_check_func(code, "f", [((2,), 2), ((0,), 0)]), 0.5)

    def test_works_from_non_main_thread(self):
        # yoriai server の measure はワーカスレッドから採点を呼ぶ。SIGALRM 案が
        # 使えない理由がこれで、subprocess 案が守るべき性質。
        import threading
        out = {}

        def run():
            out["score"] = _check_func(
                "def h(a):\n    return a + 1\n", "h", [((1,), 2)])

        t = threading.Thread(target=run)
        t.start()
        t.join(30)
        self.assertEqual(out.get("score"), 1.0)

    def test_unpicklable_return_is_wrong_not_hang(self):
        self._short_timeout(2.0)
        code = ("def f(x):\n"
                "    return lambda: x\n")
        score = _check_func(code, "f", [((1,), 1)])
        self.assertEqual(score, 0.0)

    def test_reduce_payload_is_rejected_not_executed(self):
        # __reduce__ を持つ戻り値は素の unpickle だと親プロセスで任意コードを
        # 実行できる(隔離の自己矛盾)。親は find_class 全拒否の unpickler で
        # 復元するので、reduce が無害な (list, ((1,2),)) であっても GLOBAL 参照の
        # 時点で拒否され、その 1 件は err(=不正解)になる。実行されていれば
        # [1,2] == [1,2] で 1.0 になるはずなので、0.0 がそのまま非実行の証拠。
        self._short_timeout(5.0)
        code = ("class E:\n"
                "    def __reduce__(self):\n"
                "        return (list, ((1, 2),))\n"
                "def f(x):\n"
                "    return E()\n")
        self.assertEqual(_check_func(code, "f", [((0,), [1, 2])]), 0.0)

    def test_scores_correctly_from_c_entrypoint(self):
        # 回帰: forkserver は子で親の __main__ を path 再 import するため、main が実ファイル
        # でない起動(python -c / heredoc)だと bootstrap FileNotFoundError で正解コードが
        # 0.0 に化けていた(2026-08-31 実測)。単スレッド親は fork を使うことで、あらゆる
        # 起動経路で一様に動くべき。実際に別プロセスの `python -c` から確かめる。
        import subprocess
        code = (
            "from gama.benchmark import _check_func\n"
            "print(_check_func('def f(x):\\n    return x*x\\n', 'f', "
            "[((3,), 9), ((4,), 16)]))\n"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(__file__)), timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "1.0", f"stdout={r.stdout!r} stderr={r.stderr!r}")

    def test_multithreaded_nonfile_main_fails_loud_not_silent(self):
        # コンボの穴(多スレッド親 + 非ファイル __main__): fork へ逃がせず forkserver/
        # spawn の bootstrap が非ファイル main の再 import で失敗し得る。ここは正解コードを
        # 正しく採れないが、**沈黙で 0 点化してはいけない**(measurement-goes-silent)。
        # 別スレッドを立てて active_count>1 を作り、python -c から _check_func を呼ぶ。
        # 期待: 沈黙でなく stderr に 1 度きりの警告が出る(スコアが 0.0 でも可視)。
        # `python -` (stdin main) は __main__.__file__='<stdin>' になり、forkserver の
        # 子が main を path 再 import して FileNotFoundError で死ぬ。スレッドを立てて
        # active_count>1 にすると fork へ逃がせず、まさに doom コンボに落ちる。これを
        # subprocess の stdin で決定的に踏む。
        import subprocess
        script = (
            "import threading\n"
            "stop = threading.Event()\n"
            "t = threading.Thread(target=lambda: stop.wait()); t.start()\n"
            "from gama.benchmark import _check_func\n"
            "s = _check_func('def f(x):\\n    return x*x\\n', 'f', [((3,), 9)])\n"
            "print('score', s)\n"
            "stop.set(); t.join()\n"
        )
        r = subprocess.run([sys.executable, "-"], input=script, capture_output=True,
                           text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
                           timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        # 沈黙の 0.0 は禁止(measurement-goes-silent): score が 1.0 でないなら、必ず
        # 「なぜ 0 なのか」の警告が stderr に出ていること。この環境では bootstrap が
        # 失敗して loud 0.0 になる想定だが、成功して 1.0 でも契約は満たす。
        if "score 1.0" not in r.stdout:
            self.assertIn("sandbox child died during bootstrap", r.stderr,
                          f"silent 0.0 forbidden. stdout={r.stdout!r} stderr={r.stderr!r}")

    def test_child_hard_exit_returns_fast(self):
        # 生成コードが q.put 無しで即死する経路(os._exit / ハードクラッシュ)。
        # queue 待ち一本だと満額 timeout を待つ回帰があったので、死の検知で
        # 早期に 0.0 へ折れることを壁時計で判定する。
        self._short_timeout(10.0)
        import time as _t
        t0 = _t.monotonic()
        score = _check_func("import os\nos._exit(0)\n", "f", [((1,), 1)])
        self.assertEqual(score, 0.0)
        self.assertLess(_t.monotonic() - t0, 5.0)

    def test_user_hard_exit_stays_silent(self):
        # started 印の後に死ぬのは生成コード自身の選択(os._exit)なので、bootstrap 失敗の
        # 警告を出してはいけない(codex 指摘: 環境失敗と user コードの hard-exit を区別)。
        # 別プロセスで stderr を捕まえ、警告が漏れないことを確認する。
        import subprocess
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from gama.benchmark import _check_func\n"
            "print('score', _check_func('import os\\nos._exit(0)\\n', 'f', [((1,), 1)]))\n"
            % os.path.dirname(os.path.dirname(__file__))
        )
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(r.stdout.strip(), "score 0.0", r.stderr)
        self.assertNotIn("sandbox child died during bootstrap", r.stderr)


class TestNamedSuites(unittest.TestCase):
    def test_registry_keys(self):
        self.assertEqual(set(SUITES), {"default", "hard", "brutal", "wide", "graded", "steep",
                                      "qadeep", "researchdeep", "crux"})

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


class TestSuiteChoicesTrackTheRegistry(unittest.TestCase):
    """suite を足したのに CLI から選べない、が起きないようにする。

    実害: `crux` を SUITES に足したが、`--suite` の choices が 4 箇所ベタ書きだったので
    `gama bench --suite crux` が「invalid choice」で弾かれた。足す側が 4 箇所を思い出す設計に
    なっていたのが原因なので、**選択肢は registry から引く**ことを試験で固定する。
    """

    def test_every_registered_suite_is_selectable_from_the_cli(self):
        from gama.cli import build_parser
        parser = build_parser()
        found = 0
        for action in _walk_actions(parser):
            if action.dest == "suite" and action.choices is not None:
                found += 1
                self.assertEqual(set(action.choices), set(SUITES),
                                 f"--suite choices drifted from the registry: "
                                 f"{sorted(set(SUITES) ^ set(action.choices))}")
        self.assertGreater(found, 0, "no --suite argument found to check")

    def test_every_registered_suite_has_a_one_line_description(self):
        # help を手書きの写しにしていたので、qadeep / researchdeep / crux の 3 本が登録済みなのに
        # 説明に出てこない状態が続いていた。出所を 1 つにしたことを試験で固定する。
        from gama.benchmark import SUITE_DOCS
        self.assertEqual(set(SUITE_DOCS), set(SUITES),
                         f"SUITE_DOCS drifted from the registry: "
                         f"{sorted(set(SUITE_DOCS) ^ set(SUITES))}")
        for name, doc in SUITE_DOCS.items():
            self.assertTrue(doc.strip(), f"{name} has an empty description")


def _walk_actions(parser):
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            for sub in action.choices.values():          # subparsers
                yield from _walk_actions(sub)
        else:
            yield action
