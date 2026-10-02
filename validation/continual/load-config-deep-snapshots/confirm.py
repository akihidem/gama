import copy
import unittest

from gama.config import load_config


COMPOSITES = (
    ('ensemble', 'members'),
    ('meshflow', 'tiers'),
    ('trinity', 'workers'),
    ('abmcts', 'workers'),
)


def source_for(section, collection):
    return {section: {collection: [
        {'backend': 'leaf', 'kwargs': {
            'options': {'labels': ['keep'], 'bounds': {'limit': 8}}
        }}
    ]}}


class LoadConfigSnapshotConfirmation(unittest.TestCase):
    def test_every_composite_is_insulated_from_source_edits(self):
        for section, collection in COMPOSITES:
            with self.subTest(section=section):
                raw = source_for(section, collection)
                loaded = load_config(raw)
                expected = copy.deepcopy(loaded)
                options = raw[section][collection][0]['kwargs']['options']
                options['bounds']['limit'] = 99
                options['labels'].clear()
                raw[section][collection].append({'backend': 'added'})
                self.assertEqual(loaded, expected)

    def test_every_composite_leaves_source_unchanged_when_edited(self):
        for section, collection in COMPOSITES:
            with self.subTest(section=section):
                raw = source_for(section, collection)
                expected = copy.deepcopy(raw)
                loaded = load_config(raw)
                loaded[section][collection][0]['kwargs']['options']['labels'].append('new')
                loaded[section][collection].append({'backend': 'added'})
                self.assertEqual(raw, expected)

    def test_separate_calls_do_not_share_composite_data(self):
        for section, collection in COMPOSITES:
            with self.subTest(section=section):
                raw = source_for(section, collection)
                first = load_config(raw)
                second = load_config(raw)
                expected_second = copy.deepcopy(second)
                first[section][collection][0]['kwargs']['options']['bounds']['limit'] = 0
                first[section][collection].clear()
                self.assertEqual(second, expected_second)


if __name__ == '__main__':
    unittest.main()
