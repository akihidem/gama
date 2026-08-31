"""v3a/T8: verifier calibration の性質テスト。

固定するのは (1) I(V;C) の閉形式、(2) selection ceiling が件数の MAP であること、
(3) budget_check の DPI 検出（超過=測定バグ）と same-suite 拘束、(4) 実走較正が
meshflow の門と同じ二値化（_normalize_score >= pass_score）を使うこと。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gama.backends import ModelBackend
from gama.benchmark import BenchCase
from gama.calib import (
    budget_check, calibrate_verifier, confusion_counts, mutual_information_bits,
    selection_ceiling_counts,
)
from gama.cli import build_parser
from gama.models import ModelTier


def _cm(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}


class Canned(ModelBackend):
    available = True

    def __init__(self, replies):
        self.replies = dict(replies)   # case_id -> output

    def complete(self, prompt, tier, **kw):
        return self.replies[kw.get("_case_id")] if "_case_id" in kw else self._by_prompt(prompt)

    def _by_prompt(self, prompt):
        for k, v in self.replies.items():
            if k in prompt:
                return v
        return ""


class TestInformation(unittest.TestCase):
    def test_perfect_verifier_balanced_is_one_bit(self):
        self.assertEqual(mutual_information_bits(_cm(5, 0, 0, 5)), 1.0)

    def test_independent_verifier_is_zero_bits(self):
        self.assertEqual(mutual_information_bits(_cm(25, 25, 25, 25)), 0.0)

    def test_inverted_perfect_verifier_still_one_bit_but_ceiling_saves_it(self):
        # pass/fail を逆に読む verifier: I は同じ 1 bit（情報はラベルの向きに依らない）。
        # 運用上限を I でなく MAP ceiling で読む理由がこれ: MAP は向きを直して満点に立つ。
        cm = _cm(0, 5, 5, 0)
        self.assertEqual(mutual_information_bits(cm), 1.0)
        self.assertEqual(selection_ceiling_counts(cm)["ceiling_k"], 10)

    def test_empty_is_none_not_zero(self):
        self.assertIsNone(mutual_information_bits(_cm(0, 0, 0, 0)))


class TestCeiling(unittest.TestCase):
    def test_map_counts_by_hand(self):
        # pass 側 {tp=7, fp=3} は正解に賭けて 7、fail 側 {fn=2, tn=8} は不正解側に賭けて 8
        sel = selection_ceiling_counts(_cm(7, 3, 2, 8))
        self.assertEqual(sel["ceiling_k"], 15)
        self.assertEqual(sel["base_k"], 11)     # V 無視の最善 = max(9, 11)
        self.assertEqual(sel["headroom_k"], 4)

    def test_useless_verifier_has_zero_headroom(self):
        sel = selection_ceiling_counts(_cm(25, 25, 25, 25))
        self.assertEqual(sel["headroom_k"], 0)


class TestBudget(unittest.TestCase):
    def test_within_and_exceeds(self):
        cm = _cm(7, 3, 2, 8)
        ok = budget_check(15, 20, cm)
        self.assertTrue(ok["within_budget"])
        bug = budget_check(16, 20, cm)
        self.assertFalse(bug["within_budget"])
        self.assertIn("MEASUREMENT BUG", bug["verdict"])

    def test_refuses_cross_suite_comparison(self):
        with self.assertRaises(ValueError):
            budget_check(3, 10, _cm(7, 3, 2, 8))   # n=10 vs calib n=20


class TestLiveCalibration(unittest.TestCase):
    def _suite(self):
        # checker は「OK を含む」。出力は 4 通り: 正解+verify通過 / 正解+不通過 /
        # 不正解+通過 / 不正解+不通過 を 1 件ずつ作る
        return [
            BenchCase("c1", "qa", "say c1", lambda o: "OK" in o),
            BenchCase("c2", "qa", "say c2", lambda o: "OK" in o),
            BenchCase("c3", "qa", "say c3", lambda o: "OK" in o),
            BenchCase("c4", "qa", "say c4", lambda o: "OK" in o),
        ]

    def test_confusion_from_live_run(self):
        be = Canned({"c1": "OK pass", "c2": "OK", "c3": "pass", "c4": ""})
        verify = lambda o: "pass" in o          # noqa: E731
        res = calibrate_verifier(be, verify, self._suite(), ModelTier.LARGE)
        self.assertEqual(res["confusion"], _cm(1, 1, 1, 1))
        self.assertEqual(res["i_bits"], 0.0)    # 独立: verify は正しさを何も知らない
        self.assertEqual(len(res["per_case"]), 4)

    def test_pass_score_threshold_matches_gate_rule(self):
        # float を返す verifier: 門と同じ「_normalize_score >= pass_score」で二値化する。
        be = Canned({"c1": "OK", "c2": "OK", "c3": "x", "c4": "y"})
        verify = lambda o: 0.6                  # noqa: E731
        strict = calibrate_verifier(be, verify, self._suite(), ModelTier.LARGE,
                                    pass_score=1.0)
        loose = calibrate_verifier(be, verify, self._suite(), ModelTier.LARGE,
                                   pass_score=0.5)
        self.assertEqual(strict["confusion"]["tp"] + strict["confusion"]["fp"], 0)
        self.assertEqual(loose["confusion"]["tp"] + loose["confusion"]["fp"], 4)

    def test_verifier_exception_counts_as_fail(self):
        be = Canned({"c1": "OK", "c2": "OK", "c3": "x", "c4": "y"})

        def verify(o):
            raise RuntimeError("gate crashed")

        res = calibrate_verifier(be, verify, self._suite(), ModelTier.LARGE)
        self.assertEqual(res["confusion"]["tp"] + res["confusion"]["fp"], 0)


class TestBackendFailureIsolation(unittest.TestCase):
    def test_backend_exception_never_reaches_verifier(self):
        # 出力が存在しないケースに定数 True の verifier が fp を作ると較正が汚染される。
        # docstring の約束(fail 固定)を機構で守っていることの確認(doc-promise-needs-mechanism)。
        class Boom(ModelBackend):
            available = True

            def complete(self, prompt, tier, **kw):
                raise RuntimeError("backend down")

        suite = [BenchCase("c1", "qa", "say", lambda o: True)]
        res = calibrate_verifier(Boom(), lambda o: True, suite, ModelTier.LARGE)
        self.assertEqual(res["confusion"], {"tp": 0, "fp": 0, "fn": 0, "tn": 1, "n": 1})
        self.assertTrue(res["per_case"][0]["backend_error"])


class TestCliWiring(unittest.TestCase):
    def test_parser_accepts_calib(self):
        args = build_parser().parse_args(
            ["calib", "--backend", "echo", "--verify", "nonempty", "--suite", "hard",
             "--observed", "3", "--observed-n", "24"])
        self.assertEqual(args.command, "calib")
        self.assertEqual(args.observed, 3)
        self.assertEqual(args.observed_n, 24)


if __name__ == "__main__":
    unittest.main()
