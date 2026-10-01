import unittest

from gama.backends import EnsembleBackend, MeasurementUnavailable, ModelBackend
from gama.models import ModelTier


class ScriptedBackend(ModelBackend):
    available = True

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, prompt, tier, **kwargs):
        self.calls.append((prompt, tier, kwargs))
        if not self.outcomes:
            raise AssertionError('unexpected backend call')
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SealedTests(unittest.TestCase):
    def test_default_aggregator_failure_after_first_member_failure(self):
        first = ScriptedBackend(RuntimeError('member down'), RuntimeError('aggregate down'))
        answer = '  saved\n'
        survivor = ScriptedBackend(answer)
        later = ScriptedBackend('a longer alternative that must not win')
        ensemble = EnsembleBackend([first, survivor, later], strategy='synthesize')
        self.assertEqual(ensemble.complete('question', ModelTier.SMALL), answer)
        self.assertEqual(ensemble.last_candidates, [
            '', answer, 'a longer alternative that must not win'])
        self.assertEqual(len(ensemble.last_failures), 1)
        self.assertEqual([len(b.calls) for b in (first, survivor, later)], [2, 1, 1])

    def test_default_aggregator_failure_after_whitespace_member_reply(self):
        first = ScriptedBackend(' \t\r\n', TimeoutError('aggregation timeout'))
        answer = ' \n0\t'
        survivor = ScriptedBackend(answer)
        ensemble = EnsembleBackend([first, survivor])
        self.assertEqual(ensemble.complete('question', ModelTier.MEDIUM), answer)
        self.assertEqual(ensemble.last_candidates, [' \t\r\n', answer])
        self.assertEqual(ensemble.last_failures, [])
        self.assertEqual([len(first.calls), len(survivor.calls)], [2, 1])

    def test_repeated_calls_select_current_candidates_in_member_order(self):
        first = ScriptedBackend('', 'new first answer')
        second = ScriptedBackend('old survivor', 'new second answer')
        aggregator = ScriptedBackend(RuntimeError('first failure'), RuntimeError('second failure'))
        ensemble = EnsembleBackend([first, second], aggregator=aggregator)
        self.assertEqual(ensemble.complete('first question', ModelTier.SMALL), 'old survivor')
        self.assertEqual(ensemble.complete('second question', ModelTier.SMALL), 'new first answer')
        self.assertEqual(ensemble.last_candidates, ['new first answer', 'new second answer'])
        self.assertEqual([len(b.calls) for b in (first, second, aggregator)], [2, 2, 2])

    def test_no_usable_member_and_a_failure_still_raise(self):
        for second_outcome in (' \t', ValueError('second member down')):
            with self.subTest(second_outcome=repr(second_outcome)):
                aggregator = ScriptedBackend('must remain unused')
                ensemble = EnsembleBackend([
                    ScriptedBackend(RuntimeError('first member down')),
                    ScriptedBackend(second_outcome)], aggregator=aggregator)
                with self.assertRaises(MeasurementUnavailable):
                    ensemble.complete('question', ModelTier.SMALL)
                self.assertEqual(aggregator.calls, [])

    def test_all_blank_members_remain_a_real_empty_answer(self):
        aggregator = ScriptedBackend('must remain unused')
        ensemble = EnsembleBackend([
            ScriptedBackend(''), ScriptedBackend(' \t\n')], aggregator=aggregator)
        self.assertEqual(ensemble.complete('question', ModelTier.SMALL), '')
        self.assertEqual(ensemble.last_failures, [])
        self.assertEqual(aggregator.calls, [])

    def test_successful_aggregator_reply_is_not_replaced_by_a_member(self):
        for answer in ('', ' \t', 'synthesized answer'):
            with self.subTest(answer=answer):
                aggregator = ScriptedBackend(answer)
                ensemble = EnsembleBackend([
                    ScriptedBackend(''), ScriptedBackend('usable member')],
                    aggregator=aggregator)
                self.assertEqual(ensemble.complete('question', ModelTier.LARGE), answer)
                self.assertEqual(len(aggregator.calls), 1)


if __name__ == '__main__':
    unittest.main()
