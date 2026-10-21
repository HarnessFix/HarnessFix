import json
import tempfile
import unittest
from pathlib import Path

from task_agent.terminal_bench_agent.run_terminal_bench_entry import _convert_outputs


class ConfigOnlyTrialTests(unittest.TestCase):
    def test_incomplete_trial_does_not_abort_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / 'job'
            job.mkdir()
            (job / 'result.json').write_text('{}')
            incomplete = job / 'incomplete__abc'
            incomplete.mkdir()
            (incomplete / 'config.json').write_text('{"task": {"path": "incomplete"}}')
            complete = job / 'complete__def'
            complete.mkdir()
            (complete / 'result.json').write_text(json.dumps({
                'task_name': 'complete', 'verifier_result': {'rewards': {'reward': 1}}
            }))
            output = root / 'out'
            _convert_outputs(job, output, 'fixture', 'terminus-2')
            preds = json.loads((output / 'preds.json').read_text())
            self.assertEqual(preds, {'complete': '1', 'incomplete': ''})
            info = json.loads((output / 'run_info.json').read_text())
            self.assertEqual(info['n_predictions'], 2)
            self.assertTrue((output / 'incomplete' / 'incomplete.traj.json').exists())
