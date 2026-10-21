import json
import tempfile
import unittest
from pathlib import Path
from eval.eval_terminal_bench import evaluate


class TerminalRewardTests(unittest.TestCase):
    def test_invalid_rewards_do_not_resolve_or_abort_other_tasks(self):
        for reward in [float('inf'), float('-inf'), float('nan'), 'invalid', {}, []]:
            for source in ['trajectory', 'result']:
                with self.subTest(reward=reward, source=source), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    for task, value in [('bad', reward), ('good', 1.0)]:
                        path = root / task
                        path.mkdir()
                        data = {'info': {'reward': value}} if source == 'trajectory' else {'verifier_result': {'rewards': {'reward': value}}}
                        filename = f'{task}.traj.json' if source == 'trajectory' else 'result.json'
                        (path / filename).write_text(json.dumps(data))
                    report = evaluate(['bad', 'good'], root)
                    self.assertEqual(report['resolved_ids'], ['good'])
                    self.assertEqual(report['error_ids'], ['bad'])
                    self.assertIsNone(report['rewards']['bad'])
                    json.dumps(report, allow_nan=False)

    def test_finite_missing_and_exception_rewards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task, info in [('zero', {'reward': 0}), ('fraction', {'reward': 0.5}), ('one', {'reward': '1.0'}), ('missing', {}), ('error', {'reward': 1, 'exit_status': 'error'})]:
                path = root / task
                path.mkdir()
                (path / f'{task}.traj.json').write_text(json.dumps({'info': info}))
            report = evaluate(['zero', 'fraction', 'one', 'missing', 'error'], root)
            self.assertEqual(report['resolved_ids'], ['one'])
            self.assertEqual(report['unresolved_ids'], ['fraction', 'zero'])
            self.assertEqual(report['empty_patch_ids'], ['missing'])
            self.assertEqual(report['error_ids'], ['error'])
