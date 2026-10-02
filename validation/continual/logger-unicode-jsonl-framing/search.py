import json
import tempfile
import unittest
from pathlib import Path
from gama.logger import ExecutionLogger, LogRecord


class UnicodeFramingSearch(unittest.TestCase):
    def test_logged_unicode_separators_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = ExecutionLogger(Path(directory) / 'ledger.jsonl')
            expected = []
            for index, codepoint in enumerate((0x85, 0x2028, 0x2029)):
                text = 'left' + chr(codepoint) + 'right'
                record = LogRecord(
                    run_id='run-' + str(index), task_id=text,
                    task_type='benchmark', selected_model='model',
                    failure_reason=text, ts='2026-10-02T00:00:00+00:00',
                )
                expected.append(record.to_dict())
                self.assertIs(logger.log(record), record)
            before = logger.path.read_bytes()
            for codepoint in (0x85, 0x2028, 0x2029):
                self.assertIn(chr(codepoint).encode('utf-8'), before)
            try:
                actual = logger.read()
            except json.JSONDecodeError as exc:
                self.fail('Valid Unicode text must not split a JSONL record: ' + str(exc))
            self.assertEqual(actual, expected)
            self.assertEqual(logger.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
