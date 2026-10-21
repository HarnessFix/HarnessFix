import json
import tempfile
import unittest
from pathlib import Path
from eval.eval_appworld import evaluate


class SuccessTypeTests(unittest.TestCase):
    def test_non_boolean_success_is_not_a_pass(self):
        for value in ['false', 'true', 1, 0, [], {}, ['false']]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for task, success in [('invalid', value), ('valid', True)]:
                    path = root / task
                    path.mkdir()
                    (path / 'result.json').write_text(json.dumps({'success': success}))
                result = evaluate([{'task_id': 'invalid'}, {'task_id': 'valid'}], root)
                self.assertEqual(result['resolved_ids'], ['valid'])
                self.assertEqual(result['error_ids'], ['invalid'])
                self.assertEqual(result['accuracy'], .5)

    def test_false_and_missing_success_remain_unsolved_or_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task, result in [('failed', {'success': False}), ('missing', {})]:
                path = root / task
                path.mkdir()
                (path / 'result.json').write_text(json.dumps(result))
                if task == 'failed':
                    (path / 'failed.traj.json').write_text('{"info":{"exit_status":"unsolved"}}')
            result = evaluate([{'task_id': 'failed'}, {'task_id': 'missing'}], root)
            self.assertEqual(result['unresolved_ids'], ['failed'])
            self.assertEqual(result['empty_patch_ids'], ['missing'])
