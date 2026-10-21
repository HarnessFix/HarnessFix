import json
import tempfile
import unittest
from pathlib import Path
from failure_analysis.validation_metrics import compute_terminal_bench_metrics


class CommandIdentityTests(unittest.TestCase):
    def rate(self, calls):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / 'traces' / 'task'
            task.mkdir(parents=True)
            (task / 'task.traj.json').write_text('{}')
            (task / 'task.atif.json').write_text(json.dumps({'steps': [{'tool_calls': [call]} for call in calls]}))
            report = root / 'report.json'
            report.write_text('{}')
            return compute_terminal_bench_metrics(root / 'traces', report)['metrics']['repeated_command_rate']

    def test_unique_invocation_ids_do_not_hide_repeated_command(self):
        calls = [{'tool_call_id': str(i), 'function_name': 'shell', 'arguments': {'command': 'pwd'}, 'extra': {'retry': i}} for i in range(4)]
        self.assertEqual(self.rate(calls), 1)

    def test_argument_key_order_is_not_command_identity(self):
        calls = [{'tool_call_id': str(i), 'function_name': 'shell', 'arguments': dict([('command', 'pwd'), ('timeout', 10)][::1 if i % 2 else -1])} for i in range(4)]
        self.assertEqual(self.rate(calls), 1)

    def test_distinct_arguments_or_tools_do_not_form_a_loop(self):
        for field in ['arguments', 'function_name']:
            calls = [{'tool_call_id': 'same', 'function_name': 'shell', 'arguments': {'command': "printf 'a b'"}} for _ in range(4)]
            calls[2][field] = {'command': "printf 'a  b'"} if field == 'arguments' else 'other_tool'
            with self.subTest(field=field):
                self.assertEqual(self.rate(calls), 0)
