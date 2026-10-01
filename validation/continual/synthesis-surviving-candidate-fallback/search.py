import unittest

from gama.backends import EnsembleBackend, ModelBackend, synthesize
from gama.models import ModelTier


class Stub(ModelBackend):
    available = True

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def complete(self, prompt, tier, **kwargs):
        self.calls.append((prompt, tier, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class SearchTests(unittest.TestCase):
    def test_helper_skips_blank_prefix_after_aggregation_error(self):
        aggregator = Stub(RuntimeError('aggregation unavailable'))
        candidates = ['', ' \t\n', '  usable answer\n', 'later alternative']
        result = synthesize(aggregator, 'question', ModelTier.SMALL, candidates)
        self.assertEqual(result, '  usable answer\n')
        self.assertEqual(len(aggregator.calls), 1)

    def test_failed_first_member_does_not_hide_survivor(self):
        answer = 'surviving member'
        aggregator = Stub(ValueError('bad aggregation'))
        ensemble = EnsembleBackend(
            [Stub(RuntimeError('member unavailable')), Stub(answer)],
            strategy='synthesize', aggregator=aggregator)
        self.assertEqual(ensemble.complete('question', ModelTier.MEDIUM), answer)
        self.assertEqual(ensemble.last_candidates, ['', answer])
        self.assertEqual(len(ensemble.last_failures), 1)
        self.assertEqual(len(aggregator.calls), 1)


if __name__ == '__main__':
    unittest.main()
