import json
import tempfile
import unittest
from pathlib import Path
from gama.logger import ExecutionLogger, LogRecord


class UnicodeFramingSealed(unittest.TestCase):
    def read_valid(self, logger):
        try:
            return logger.read()
        except json.JSONDecodeError as exc:
            self.fail('Unicode separators inside valid JSON are data: ' + str(exc))

    def test_append_to_external_records_and_read_between_appends(self):
        separators = ''.join(chr(n) for n in (0x85, 0x2028, 0x2029))
        seed = {'seed': {separators: ['x' + separators + 'y']}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'append.jsonl'
            path.write_bytes((json.dumps(seed, ensure_ascii=False) + '\n').encode('utf-8'))
            logger = ExecutionLogger(path)
            expected = [seed]
            for index, codepoint in enumerate((0x85, 0x2028, 0x2029)):
                separator = chr(codepoint)
                text = separator + 'value' + separator + chr(34) + chr(92) + '\n\r\t'
                record = LogRecord(
                    run_id='run' + separator + str(index), task_id=text,
                    task_type='type' + separator, selected_model='model' + separator,
                    review_score=0.75, actual_cost=0.1, elapsed_seconds=2.0,
                    judge_decision='pass' + separator, failure_reason=text,
                    ts='2026-10-02T00:00:00+00:00',
                )
                expected.append(record.to_dict())
                self.assertIs(logger.log(record), record)
                self.assertEqual(record.to_dict(), expected[-1])
                before = path.read_bytes()
                self.assertEqual(self.read_valid(logger), expected)
                self.assertEqual(path.read_bytes(), before)
            self.assertEqual(self.read_valid(ExecutionLogger(path)), expected)
            self.assertEqual(path.read_bytes(), before)

    def test_literal_and_escaped_unicode_encodings_are_equivalent(self):
        text = ''.join(chr(n) for n in (0x2029, 0x85, 0x2028))
        record = {text: [text, {'literal': chr(92) + 'u2028'}, text[::-1]]}
        payload = '\n'.join([
            json.dumps(record, ensure_ascii=True),
            json.dumps(record, ensure_ascii=False),
        ]).encode('utf-8')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encodings.jsonl'
            path.write_bytes(payload)
            self.assertEqual(self.read_valid(ExecutionLogger(path)), [record, record])
            self.assertEqual(path.read_bytes(), payload)

    def test_missing_blank_and_unterminated_valid_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'ledger.jsonl'
            logger = ExecutionLogger(path)
            self.assertEqual(logger.read(), [])
            self.assertFalse(path.exists())
            blank = b'\n \t\r\n\r\n'
            path.write_bytes(blank)
            self.assertEqual(logger.read(), [])
            self.assertEqual(path.read_bytes(), blank)
            record = {'message': ''.join(chr(n) for n in (0x85, 0x2028, 0x2029))}
            payload = json.dumps(record, ensure_ascii=False).encode('utf-8')
            path.write_bytes(payload)
            self.assertEqual(self.read_valid(logger), [record])
            self.assertEqual(path.read_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
