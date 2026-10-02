import copy
import unittest

from gama.config import load_config


class LoadConfigSnapshotSealed(unittest.TestCase):
    def test_shared_nested_data_and_later_loads(self):
        shared = {'layers': [{'items': ['seed', {'value': 'before'}]}]}
        leaf = {'backend': 'leaf', 'kwargs': {'payload': shared}}
        raw = {
            'backends': {'left': {'payload': shared}, 'right': {'payload': shared}},
            'ensemble': {'member': leaf, 'n': 2, 'aggregator': leaf},
            'meshflow': {'tiers': [
                {'backend': 'tool', 'kwargs': {'inner': leaf}}
            ]},
            'trinity': {'workers': [leaf], 'scorer': leaf},
            'abmcts': {'workers': [leaf]},
        }
        original = copy.deepcopy(raw)
        first = load_config(raw)
        expected_first = copy.deepcopy(first)
        self.assertEqual(raw, original)

        shared['layers'][0]['items'][1]['value'] = 'caller'
        self.assertEqual(first, expected_first)

        second = load_config(raw)
        self.assertEqual(second['backends']['left']['payload'], shared)
        self.assertEqual(second['ensemble']['member']['kwargs']['payload'], shared)
        third = load_config(raw)
        expected_third = copy.deepcopy(third)
        expected_source = copy.deepcopy(raw)

        payload = second['trinity']['scorer']['kwargs']['payload']
        payload['layers'][0]['items'].append('result')
        self.assertEqual(raw, expected_source)
        self.assertEqual(first, expected_first)
        self.assertEqual(third, expected_third)

    def test_isolation_preserves_filtering_and_normalization(self):
        raw = {
            'default_backend': 17,
            'routing_table': {3: 'local', 'discard': None},
            'backends': {
                3: {'options': {'labels': ['seed']}},
                'discard': [],
            },
            'unit_cost': {3: 2, 'fraction': 0.125, 'flag': True, 'discard': '3.0'},
            'ensemble': {
                'member': {'backend': 'local', 'kwargs': {'labels': ['x']}},
                'n': 2,
            },
            'meshflow': [],
            'trinity': 'invalid',
            'abmcts': {
                'workers': [{'backend': 'local', 'kwargs': {}}],
                'prior': [0.5, 1.0],
            },
            'unrecognized': {'payload': [9]},
        }
        expected = {
            'default_backend': 'ollama',
            'routing_table': {'3': 'local'},
            'backends': {'3': {'options': {'labels': ['seed']}}},
            'unit_cost': {'3': 2.0, 'fraction': 0.125, 'flag': 1.0},
            'ensemble': {
                'member': {'backend': 'local', 'kwargs': {'labels': ['x']}},
                'n': 2,
            },
            'meshflow': {},
            'trinity': {},
            'abmcts': {
                'workers': [{'backend': 'local', 'kwargs': {}}],
                'prior': [0.5, 1.0],
            },
        }
        original = copy.deepcopy(raw)
        loaded = load_config(raw)
        self.assertEqual(raw, original)
        self.assertEqual(loaded, expected)
        for value in loaded['unit_cost'].values():
            self.assertIsInstance(value, float)

        raw['backends'][3]['options']['labels'].append('changed')
        raw['ensemble']['member']['kwargs']['labels'][0] = 'changed'
        raw['abmcts']['prior'][0] = 9.0
        self.assertEqual(loaded, expected)


if __name__ == '__main__':
    unittest.main()
