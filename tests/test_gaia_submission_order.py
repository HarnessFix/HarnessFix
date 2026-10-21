import json
import tempfile
import unittest
from pathlib import Path
from failure_analysis.htir import compile_gaia_htir


class GaiaSubmissionOrderTests(unittest.TestCase):
    def compile(self, calls, limit=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps({'trajectory': [{'_type': 'ActionStep'}],
                                        'model_calls': calls,
                                        'info': {'iteration_limit_exceeded': limit}}))
            return compile_gaia_htir('task', 'wrong', str(path), 'question', '42', '41')

    def test_submission_follows_auxiliary_reformulation(self):
        bundle = self.compile([{'call_index': 1, 'component': 'manager'},
                               {'call_index': 2, 'component': 'reformulator',
                                'response_message': {'content': '41'}}], limit=True)
        nodes = bundle['graph']['nodes']
        submission = next(node for node in nodes if node['type'] == 'SubmissionEvent')
        calls = [node for node in nodes if node['type'] == 'ModelInvocationEvent']
        self.assertGreater(submission['step_id'], max(node['step_id'] for node in calls))
        exception = next(node for node in nodes if node['source_ref'] == 'traj.info.iteration_limit_exceeded')
        self.assertEqual(exception['step_id'], submission['step_id'])
        self.assertEqual(bundle['agent_trace_steps'][-1]['numeric_step_id'], submission['step_id'])

    def test_without_auxiliary_calls_submission_keeps_next_step(self):
        bundle = self.compile([])
        submission = next(node for node in bundle['graph']['nodes'] if node['type'] == 'SubmissionEvent')
        self.assertEqual(submission['step_id'], 2)
