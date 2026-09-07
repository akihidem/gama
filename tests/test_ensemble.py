import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import (EnsembleBackend, MeasurementUnavailable, ModelBackend,
                           clear_finish_reason, member_failures_of)
from gama.config import build_backend, ensemble_from_config
from gama.models import ModelTier


class Fixed(ModelBackend):
    """Returns a fixed string regardless of prompt."""
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        return self.reply


class CapturingAgg(ModelBackend):
    available = True

    def __init__(self):
        self.seen = None
        self.last_usage = None

    def complete(self, prompt, tier, **kw):
        self.seen = prompt
        return "FINAL"


class Boom(ModelBackend):
    available = True

    def complete(self, prompt, tier, **kw):
        raise RuntimeError("boom")


class TestEnsemble(unittest.TestCase):
    def test_majority(self):
        e = EnsembleBackend([Fixed("yes"), Fixed("yes"), Fixed("no")], strategy="majority")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "yes")

    def test_majority_normalizes_whitespace(self):
        e = EnsembleBackend([Fixed("396"), Fixed(" 396 "), Fixed("394")], strategy="majority")
        self.assertEqual(e.complete("q", ModelTier.LARGE).strip(), "396")

    def test_first_skips_empty(self):
        e = EnsembleBackend([Fixed(""), Fixed("A"), Fixed("B")], strategy="first")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "A")

    def test_synthesize_feeds_candidates_to_aggregator(self):
        agg = CapturingAgg()
        e = EnsembleBackend([Fixed("c1"), Fixed("c2")], strategy="synthesize", aggregator=agg)
        out = e.complete("the task", ModelTier.LARGE)
        self.assertEqual(out, "FINAL")
        self.assertIn("c1", agg.seen)
        self.assertIn("c2", agg.seen)
        self.assertIn("the task", agg.seen)

    def test_records_candidates(self):
        e = EnsembleBackend([Fixed("x"), Fixed("y")], strategy="first")
        e.complete("q", ModelTier.LARGE)
        self.assertEqual(e.last_candidates, ["x", "y"])

    def test_member_failure_tolerated(self):
        e = EnsembleBackend([Boom(), Fixed("ok")], strategy="first")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "ok")
        self.assertEqual(e.last_candidates, ["", "ok"])

    def test_all_empty_returns_empty(self):
        # 誰も落ちずに全員が空を返したなら、それは本当に「空の答え」。0 点でよい。
        e = EnsembleBackend([Fixed(""), Fixed("")], strategy="majority")
        self.assertEqual(e.complete("q", ModelTier.LARGE), "")

    def test_all_members_failing_raises_instead_of_answering_empty(self):
        # 全員落ちた合議が空文字を返すと、台帳には「不正解」として入り、error は 1 件も
        # 増えないので走行の門(error_rate)が気づけない。実測(run AA): サーバが消えていた窓で
        # research クラスの 114 call が chars=0 の非エラーとして記録され、伸びしろが一番
        # 大きいクラスの点を黙って押し下げていた。合成物は外から見ると答えたように見えるので、
        # 答えを持って帰れない時は名乗って落ちる。
        e = EnsembleBackend([Boom(), Boom()], strategy="first")
        with self.assertRaises(RuntimeError) as cm:
            e.complete("q", ModelTier.LARGE)
        msg = str(cm.exception)
        self.assertIn("ensemble", msg)          # どの合成物が答えられなかったか
        self.assertIn("boom", msg)              # 最初の原因も台帳の 1 行に載る
        self.assertIn("2 of 2", msg)            # 何人中何人が落ちたか(「全員」と決め打たない)
        self.assertIsNotNone(cm.exception.__cause__)
        # 型で区別できる: 後から「基盤の失敗」を error_rate から分けたくなった時に、
        # 台帳の文字列を解析しなくて済む
        self.assertIsInstance(cm.exception, MeasurementUnavailable)
        self.assertIsInstance(cm.exception, RuntimeError)   # 掃引の except Exception が拾う

    def test_one_member_failing_and_one_answering_empty_is_still_unavailable(self):
        # 実装の条件は「1 人以上落ちた かつ 非空が 0」で、「全員落ちた」ではない。
        # 落ちた人が居て答えも無いなら、答えを持って帰れていない。
        e = EnsembleBackend([Boom(), Fixed("")], strategy="first")
        with self.assertRaises(MeasurementUnavailable) as cm:
            e.complete("q", ModelTier.LARGE)
        self.assertIn("1 of 2", str(cm.exception))          # 事実と違うことを言わない

    def test_a_partly_degraded_ensemble_is_visible_even_though_it_answered(self):
        # 3 人中 2 人落ちて 1 人が答えた合議は普通に答えを返すので、点だけ見ても
        # 「実際に走ったのは単体モデル 1 本」だと分からない。走らなかった構成の数字が
        # 設計の証拠として台帳に残るのは、全滅を空文字で返していたのと同じ型の欠陥。
        from gama.benchmark import BenchCase, _run_one

        e = EnsembleBackend([Boom(), Boom(), Fixed("ok")], strategy="first")
        case = BenchCase("x1", "research", "q", lambda o: 1.0 if o == "ok" else 0.0)
        rec = _run_one("ens", e, case, ModelTier.LARGE, 0, {})
        self.assertIsNone(rec["error"])          # 答えは返っている(error ではない)
        self.assertEqual(rec["score"], 1.0)
        self.assertEqual(rec["member_failures"], 2)
        self.assertEqual(member_failures_of(e), 2)

    def test_the_same_composite_called_twice_keeps_both_calls_failures(self):
        # 外側の合議が同じ内側を 1 コールの中で 2 回呼ぶ形。置き換えだと 2 回目の成功が
        # 1 回目の部分障害を消し、台帳には member_failures=0 が残る。
        class BoomOnce(ModelBackend):
            available = True

            def __init__(self):
                self.calls = 0
                self.last_usage = None

            def complete(self, prompt, tier, **kw):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("boom")
                return "ok"

        inner = EnsembleBackend([BoomOnce(), Fixed("ok")], strategy="first")
        outer = EnsembleBackend([inner, inner], strategy="first")
        clear_finish_reason(outer)
        self.assertEqual(outer.complete("q", ModelTier.LARGE), "ok")
        self.assertEqual(member_failures_of(outer), 1)

    def test_failures_are_cleared_by_the_same_walker_that_clears_finish_reason(self):
        # 読む側は木を全部見るので、今回呼ばれなかった枝に前回の失敗が残っていると
        # それを今回のものとして数える。消す側と読む側で walker を 1 つにする。
        e = EnsembleBackend([Boom(), Fixed("ok")], strategy="first")
        e.complete("q", ModelTier.LARGE)
        self.assertEqual(member_failures_of(e), 1)
        clear_finish_reason(e)
        self.assertEqual(member_failures_of(e), 0)

    def test_a_failing_ensemble_is_recorded_as_an_error_not_a_wrong_answer(self):
        # 上の約束が効くのは、走行が読む 1 行にそれが出た時だけ。sweep の記録側で押さえる:
        # error が入り、score は 0 でも「間違えた」ではないと区別できること。
        from gama.benchmark import BenchCase, _run_one

        case = BenchCase("x1", "research", "q", lambda o: 1.0 if o else 0.0)
        rec = _run_one("ens", EnsembleBackend([Boom(), Boom()], strategy="first"),
                       case, ModelTier.LARGE, 0, {})
        self.assertIsNotNone(rec["error"])
        self.assertIn("ensemble", rec["error"])
        self.assertEqual(rec["score"], 0.0)
        self.assertEqual(rec["output_chars"], 0)
        # 比較: 全員が空を返しただけなら error は付かない(本当に不正解)
        ok = _run_one("ens", EnsembleBackend([Fixed(""), Fixed("")], strategy="first"),
                      case, ModelTier.LARGE, 0, {})
        self.assertIsNone(ok["error"])
        self.assertEqual(ok["score"], 0.0)

    def test_empty_members_rejected(self):
        with self.assertRaises(ValueError):
            EnsembleBackend([])

    def test_available_reflects_members(self):
        self.assertTrue(EnsembleBackend([Fixed("x")]).available)


class TestEnsembleFromConfig(unittest.TestCase):
    def test_member_times_n(self):
        e = ensemble_from_config({"ensemble": {"member": {"backend": "echo"}, "n": 4,
                                               "strategy": "first"}})
        self.assertEqual(len(e.members), 4)
        self.assertEqual(e.strategy, "first")

    def test_explicit_members(self):
        e = ensemble_from_config({"ensemble": {"members": [{"backend": "echo"},
                                                           {"backend": "null"}]}})
        self.assertEqual(len(e.members), 2)

    def test_aggregator_built(self):
        e = ensemble_from_config({"ensemble": {"member": {"backend": "echo"}, "n": 2,
                                               "aggregator": {"backend": "echo"}}})
        self.assertIsNotNone(e.aggregator)

    def test_missing_spec_raises(self):
        with self.assertRaises(ValueError):
            ensemble_from_config({"ensemble": {}})


class TestBuildBackend(unittest.TestCase):
    def test_plain(self):
        from gama.backends import EchoBackend
        self.assertIsInstance(build_backend({"backend": "echo"}), EchoBackend)

    def test_tool_wraps_inner(self):
        from gama.backends import EchoBackend, ToolBackend
        b = build_backend({"backend": "tool", "kwargs": {"inner": {"backend": "echo"}}})
        self.assertIsInstance(b, ToolBackend)
        self.assertIsInstance(b.backend, EchoBackend)

    def test_ensemble_members(self):
        from gama.backends import EnsembleBackend
        b = build_backend({"backend": "ensemble",
                           "kwargs": {"members": [{"backend": "echo"}, {"backend": "null"}]}})
        self.assertIsInstance(b, EnsembleBackend)
        self.assertEqual(len(b.members), 2)

    def test_gama_nested_composites(self):
        from gama.backends import GamaBackend, ToolBackend
        b = build_backend({"backend": "gama", "kwargs": {
            "backends": {"t": {"backend": "tool", "kwargs": {"inner": {"backend": "echo"}}},
                         "e": {"backend": "echo"}},
            "routing_table": {"qa": "t"}, "default": "e"}})
        self.assertIsInstance(b, GamaBackend)
        self.assertIsInstance(b.backends["t"], ToolBackend)
        self.assertEqual(b.pick("qa"), "t")
        self.assertEqual(b.pick("other"), "e")


if __name__ == "__main__":
    unittest.main()
