import json
import tempfile
import unittest
from pathlib import Path
from failure_analysis.htir import compile_gaia_htir


class InlineEvidenceTests(unittest.TestCase):
    def compile(self, step, calls=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps({'trajectory': [step], 'model_calls': calls or []}))
            bundle = compile_gaia_htir('task', 'wrong', str(path), 'question', '42', '41')
            return next(node for node in bundle['graph']['nodes'] if node['type'] == 'ModelInvocationEvent')

    def test_inline_message_reference_points_to_inline_evidence(self):
        step = {'model_input_messages': [{'content': 'question'}],
                'model_output_message': {'content': '41'}}
        node = self.compile(step)
        self.assertEqual(node['source_ref'], 'trajectory[0].model_output_message')

    def test_inline_text_reference_points_to_inline_evidence(self):
        node = self.compile({'model_input_messages': [{'content': 'question'}], 'model_output': '41'})
        self.assertEqual(node['source_ref'], 'trajectory[0].model_output')

    def test_recorded_call_reference_is_preserved(self):
        node = self.compile({}, [{'call_index': 1, 'component': 'manager',
                                'response_message': {'content': '41'}}])
        self.assertEqual(node['source_ref'], 'traj.model_calls[0].response_message')
