import json
import tempfile
import unittest
from pathlib import Path
from failure_analysis.htir import compile_appworld_htir, _node, _step_external_effect


class NoEffectTests(unittest.TestCase):
    def test_read_only_api_log_does_not_claim_external_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / 'trace.json'
            trace.write_text('{"messages": []}')
            logs = root / 'appworld_experiments' / 'outputs' / 'run' / 'tasks' / 'task' / 'logs'
            logs.mkdir(parents=True)
            (logs / 'api_calls.jsonl').write_text(json.dumps({'method': 'GET', 'path': '/items'}) + '\n')
            bundle = compile_appworld_htir('task', 'wrong', {'traj_path': str(trace)}, 'inspect items')
            effect = bundle['agent_trace_steps'][-1]['external_effect']
            self.assertFalse(effect['has_external_effect'])
            self.assertEqual(effect['api_call_count'], 1)
            self.assertTrue(effect['state_nodes'])

    def test_explicit_positive_effect_is_preserved(self):
        node = _node('e', 'StateDeltaRecord', 1, 'tool', 'changed', 'source',
                     {'effect_delta': {'has_task_state_change': True}})
        self.assertTrue(_step_external_effect([node])['has_external_effect'])

    def test_legacy_state_node_remains_positive(self):
        node = _node('e', 'StateDeltaRecord', 1, 'tool', 'changed', 'source')
        self.assertTrue(_step_external_effect([node])['has_external_effect'])

    def test_observed_database_change_remains_positive(self):
        node = _node('e', 'StateDeltaRecord', 1, 'tool', 'changed', 'source',
                     {'effect_delta': {'db_changes': [{'app': 'notes'}]}})
        self.assertTrue(_step_external_effect([node])['has_external_effect'])
