import json
import tempfile
import unittest
from pathlib import Path
from gama.logger import ExecutionLogger


class UnicodeFramingConfirm(unittest.TestCase):
    def read_valid(self, logger):
        try:
            return logger.read()
        except json.JSONDecodeError as exc:
            self.fail('Valid JSON strings containing Unicode separators must remain intact: ' + str(exc))

    def test_external_nested_records_and_physical_delimiters(self):
        separators = ''.join(chr(n) for n in (0x85, 0x2028, 0x2029))
        expected = [
            {'key' + separators: {'items': [separators, 'x' + separators + 'y', {'leaf': separators}]}},
            {'text': '日本語' + separators + 'done', 'literal': chr(92) + 'u2028'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'external.jsonl'
            for ending in ('\n', '\r\n'):
                for terminated in (False, True):
                    with self.subTest(ending=repr(ending), terminated=terminated):
                        lines = [
                            '', ' \t ', json.dumps(expected[0], ensure_ascii=False),
                            '', json.dumps(expected[1], ensure_ascii=False),
                        ]
                        text = ending.join(lines) + (ending if terminated else '')
                        before = text.encode('utf-8')
                        path.write_bytes(before)
                        logger = ExecutionLogger(path)
                        self.assertEqual(self.read_valid(logger), expected)
                        self.assertEqual(self.read_valid(logger), expected)
                        self.assertEqual(path.read_bytes(), before)

    def test_malformed_records_still_raise_without_rewriting(self):
        valid = json.dumps({'text': 'a' + chr(0x2028) + 'b'}, ensure_ascii=False)
        bodies = ('{\n' + valid + '\n', valid + '\n{\n{}\n', valid + '\n{')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.jsonl'
            for index, text in enumerate(bodies):
                with self.subTest(case=index):
                    before = text.encode('utf-8')
                    path.write_bytes(before)
                    with self.assertRaises(json.JSONDecodeError):
                        ExecutionLogger(path).read()
                    self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
