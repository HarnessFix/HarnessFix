import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


GATE = Path(__file__).resolve().parents[1] / 'failure_analysis' / 'check_val_gate.py'


class GateInputTests(unittest.TestCase):
    def run_gate(self, root, extra):
        (root / 'ids.txt').write_text('task\n')
        (root / 'baseline.json').write_text('{"resolved_ids": []}')
        (root / 'current.json').write_text('{"resolved_ids": ["task"]}')
        return subprocess.run([sys.executable, str(GATE), '--baseline-eval', str(root / 'baseline.json'), '--current-eval', str(root / 'current.json'), '--val-ids', str(root / 'ids.txt'), '--output', str(root / 'gate.json'), *extra], capture_output=True, text=True)

    def test_explicit_target_plan_requires_trace_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'fixes': [{'target_metrics': ['error_rate']}]}))
            process = self.run_gate(root, ['--plan-spec', str(plan)])
            self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
            self.assertFalse((root / 'gate.json').exists())

    def test_partial_trace_pair_is_not_silently_ignored(self):
        for option in ['--baseline-traces', '--current-traces']:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                process = self.run_gate(root, [option, str(root)])
                self.assertEqual(process.returncode, 2, process.stdout + process.stderr)

    def test_missing_explicit_plan_is_not_defaulted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = self.run_gate(root, ['--plan-spec', str(root / 'absent.json')])
            self.assertEqual(process.returncode, 2, process.stdout + process.stderr)

    def test_trace_pair_requires_real_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = self.run_gate(root, ['--baseline-traces', str(root / 'absent'), '--current-traces', str(root / 'absent')])
            self.assertEqual(process.returncode, 2, process.stdout + process.stderr)

    def test_resolved_only_mode_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = self.run_gate(Path(tmp), [])
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_paired_traces_still_evaluate_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ['baseline', 'current']:
                task = root / arm / 'task'
                task.mkdir(parents=True)
                (task / 'task.traj.json').write_text('{}')
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'fixes': [{'target_metrics': ['resolved_rate']}]}))
            process = self.run_gate(root, ['--plan-spec', str(plan), '--baseline-traces', str(root / 'baseline'), '--current-traces', str(root / 'current')])
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads((root / 'gate.json').read_text())
            self.assertEqual(report['target_metric_results']['improved_metric_count'], 1)
