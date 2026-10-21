import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'failure_analysis' / 'print_htir_compact.py'


class CompactLimitTests(unittest.TestCase):
    def run_cli(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bundle.json'
            path.write_text(json.dumps({'agent_trace_steps': [
                {'step_id': f'step_{i:03d}', 'summary': 'diagnostic ' * 100}
                for i in range(10)]}))
            return subprocess.run([sys.executable, str(SCRIPT), str(path), *args],
                                  text=True, capture_output=True)

    def test_positive_character_limits_bound_entire_stdout(self):
        for limit in (1, 20, 79, 80, 120):
            with self.subTest(limit=limit):
                result = self.run_cli('--char-limit', str(limit))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertLessEqual(len(result.stdout), limit)

    def test_zero_character_limit_remains_unlimited(self):
        result = self.run_cli('--char-limit', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(len(result.stdout), 1000)
        self.assertIsInstance(json.loads(result.stdout), dict)

    def test_small_step_limits_bound_actual_steps(self):
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                result = self.run_cli('--step-limit', str(limit), '--char-limit', '0')
                self.assertEqual(result.returncode, 0, result.stderr)
                steps = json.loads(result.stdout)['agent_trace_steps']
                actual = [step for step in steps if step.get('role') != 'omitted']
                self.assertEqual(len(actual), limit)
                self.assertEqual(actual[-1]['step_id'], 'step_009')

    def test_nonpositive_step_limits_are_rejected(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                result = self.run_cli('--step-limit', str(limit))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('--step-limit must be positive', result.stderr)
