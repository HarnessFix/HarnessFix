import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).resolve().parents[1] / 'failure_analysis' / 'check_val_gate.py'


class CohortTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'ids').write_text('selected\n')
        (root / 'plan').write_text(json.dumps({'fixes': [{'target_metrics': ['avg_api_calls']}]}))
        for arm, calls, resolved in [('baseline', 10, []), ('current', 5, ['selected'])]:
            (root / (arm + '.json')).write_text(json.dumps({'resolved_ids': resolved, 'total': 1, 'accuracy': len(resolved)}))
            task = root / arm / 'selected'
            task.mkdir(parents=True)
            (task / 'selected.traj.json').write_text(json.dumps({'info': {'model_stats': {'api_calls': calls}}, 'messages': [], 'trajectory': []}))

    def run_gate(self, root, mode='swe', traces=True, plan=True):
        command = [sys.executable, str(GATE), '--mode', mode, '--val-ids', str(root / 'ids'), '--baseline-eval', str(root / 'baseline.json'), '--current-eval', str(root / 'current.json')]
        if traces:
            command += ['--baseline-traces', str(root / 'baseline'), '--current-traces', str(root / 'current')]
        if plan:
            command += ['--plan-spec', str(root / 'plan')]
        return subprocess.run(command, capture_output=True, text=True)

    def test_outside_trace_cannot_mask_selected_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'current' / 'selected' / 'selected.traj.json').write_text('{"info":{"model_stats":{"api_calls":20}}}')
            for arm, calls in [('baseline', 1000), ('current', 0)]:
                task = root / arm / 'outside'
                task.mkdir()
                (task / 'outside.traj.json').write_text(json.dumps({'info': {'model_stats': {'api_calls': calls}}}))
            result = self.run_gate(root)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_missing_duplicate_and_noncanonical_traces_are_rejected(self):
        for kind in ['missing', 'duplicate', 'noncanonical']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                task = root / 'current' / 'selected'
                if kind == 'missing':
                    (task / 'selected.traj.json').unlink()
                elif kind == 'duplicate':
                    (task / 'extra.traj.json').write_text('{}')
                else:
                    (task / 'selected.traj.json').rename(task / 'other.traj.json')
                result = self.run_gate(root)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_report_cohort_must_match(self):
        for change in [{'error_ids': ['outside']}, {'resolved_ids': ['selected', 'outside']}, {'total': 2}, {'total_instances': 2}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                path = root / 'current.json'
                report = json.loads(path.read_text())
                report.update(change)
                path.write_text(json.dumps(report))
                result = self.run_gate(root)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_empty_cohort_rejected_in_resolved_only_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'ids').write_text('')
            result = self.run_gate(root, traces=False, plan=False)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_matching_cohort_passes_all_modes(self):
        for mode in ['swe', 'gaia', 'appworld', 'terminal_bench']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                result = self.run_gate(root, mode=mode, plan=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resolved_only_mode_can_filter_broader_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'current.json').write_text('{"resolved_ids":["selected","outside"],"total":2}')
            result = self.run_gate(root, traces=False, plan=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
