import unittest

from gama.backends import ModelBackend, synthesize
from gama.models import ModelTier


class RecordingBackend(ModelBackend):
    available = True

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def complete(self, prompt, tier, **kwargs):
        self.calls.append((prompt, tier, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class ConfirmTests(unittest.TestCase):
    def test_first_usable_candidate_is_verbatim_and_untruncated(self):
        answer = ' \t' + ('answer line\n' * 180) + '\r\n'
        for prefix in ([], [''], [' \t\r\n', '', '\u2003']):
            with self.subTest(prefix=prefix):
                candidates = prefix + [answer, 'longer later alternative ' * 200]
                before = list(candidates)
                aggregator = RecordingBackend(TimeoutError('aggregation timeout'))
                result = synthesize(aggregator, 'question', ModelTier.SMALL, candidates)
                self.assertEqual(result, answer)
                self.assertEqual(candidates, before)
                self.assertEqual(len(aggregator.calls), 1)

    def test_no_usable_candidate_yields_empty_string(self):
        for candidates in ([], [''], [' \t\r\n'], ['', '\u2003', '  ']):
            with self.subTest(candidates=candidates):
                before = list(candidates)
                aggregator = RecordingBackend(RuntimeError('offline'))
                self.assertEqual(
                    synthesize(aggregator, 'question', ModelTier.SMALL, candidates), '')
                self.assertEqual(candidates, before)
                self.assertEqual(len(aggregator.calls), 1)

    def test_successful_aggregation_is_authoritative_even_when_blank(self):
        for output in ('composed answer', '', ' \t\n'):
            with self.subTest(output=output):
                aggregator = RecordingBackend(output)
                candidates = ['', 'available member answer']
                result = synthesize(aggregator, 'question', ModelTier.MEDIUM, candidates)
                self.assertEqual(result, output)
                self.assertEqual(len(aggregator.calls), 1)

    def test_request_forwarding_is_preserved_during_fallback(self):
        aggregator = RecordingBackend(ValueError('aggregation failed'))
        instruction = 'Keep the requested answer format.'
        result = synthesize(
            aggregator, 'original question', ModelTier.LARGE,
            ['', '  recovered answer  ', 'later answer'],
            instruction=instruction, task_type='math', effort='low', prefill='prefix')
        self.assertEqual(result, '  recovered answer  ')
        self.assertEqual(len(aggregator.calls), 1)
        prompt, tier, kwargs = aggregator.calls[0]
        self.assertIn('original question', prompt)
        self.assertIn('  recovered answer  ', prompt)
        self.assertIn('later answer', prompt)
        self.assertIn(instruction, prompt)
        self.assertIs(tier, ModelTier.LARGE)
        self.assertEqual(kwargs, {
            'task_type': 'math', 'effort': 'low', 'prefill': 'prefix'})


if __name__ == '__main__':
    unittest.main()
