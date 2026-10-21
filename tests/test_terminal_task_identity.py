import json
import tempfile
import unittest
from pathlib import Path
from task_agent.terminal_bench_agent.run_terminal_bench_entry import _convert_outputs, _normalize_task_id


class TaskIdentityTests(unittest.TestCase):
    def test_explicit_task_names_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / 'job'
            job.mkdir()
            (job / 'result.json').write_text('{}')
            for name, reward in [('group__first', 1), ('group__second', 0)]:
                trial = job / (name + '__random')
                trial.mkdir()
                (trial / 'result.json').write_text(json.dumps({'task_name': name, 'verifier_result': {'rewards': {'reward': reward}}}))
            output = root / 'out'
            _convert_outputs(job, output, 'fixture', 'terminus-2')
            self.assertEqual(json.loads((output / 'preds.json').read_text()), {'group__first': '1', 'group__second': '0'})
            for name in ['group__first', 'group__second']:
                self.assertTrue((output / name / (name + '.traj.json')).is_file())

    def test_explicit_paths_and_trial_fallback(self):
        self.assertEqual(_normalize_task_id({'path': '/tasks/group__first'}, 'unused'), 'group__first')
        self.assertEqual(_normalize_task_id(None, 'simple__random'), 'simple')
