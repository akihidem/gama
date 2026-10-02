import copy
import unittest

from gama.config import load_config


class LoadConfigSnapshotSearch(unittest.TestCase):
    def test_loaded_backend_mutations_do_not_change_source(self):
        raw = {
            'backends': {
                'local': {
                    'model': 'original',
                    'options': {'stop': ['END'], 'limits': {'tokens': 64}},
                },
            },
        }
        before = copy.deepcopy(raw)
        loaded = load_config(raw)
        self.assertEqual(raw, before)
        loaded['backends']['local']['model'] = 'changed'
        loaded['backends']['local']['options']['stop'].append('STOP')
        loaded['backends']['local']['options']['limits']['tokens'] = 1
        self.assertEqual(raw, before, 'Loaded backend kwargs must not alias the source')

    def test_source_mutations_do_not_rewrite_loaded_ensemble(self):
        raw = {'ensemble': {
            'members': [{'backend': 'local', 'kwargs': {
                'options': {'tags': ['original']},
            }}],
            'strategy': 'first',
        }}
        loaded = load_config(raw)
        before = copy.deepcopy(loaded)
        raw['ensemble']['members'][0]['kwargs']['options']['tags'].append('changed')
        raw['ensemble']['strategy'] = 'majority'
        self.assertEqual(loaded, before, 'Loaded composite specs must be snapshots')


if __name__ == '__main__':
    unittest.main()
