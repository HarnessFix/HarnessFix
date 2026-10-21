import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Run the real CLI parser/main, replacing only optional import-time dependencies.
BOOTSTRAP = '''import runpy,sys,types
sys.modules['dotenv'] = types.SimpleNamespace(load_dotenv=lambda *a, **kw: None)
sys.modules['open_deep_research'] = types.ModuleType('open_deep_research')
sys.modules['open_deep_research.scoring'] = types.SimpleNamespace(question_scorer=lambda a,b: a == b)
script = sys.argv.pop(1)
sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
runpy.run_path(script, run_name='__main__')
'''


class EvaluatorTaskIdTests(unittest.TestCase):
    def run_cli(self, mode, ids, source='jsonl'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / 'subset'
            traces = root / 'traces'
            subset.mkdir()
            traces.mkdir()
            output = root / 'output.json'
            if source == 'jsonl':
                (subset / 'data.jsonl').write_text(''.join(json.dumps({'task_id': task_id, 'true_answer': 'yes'}) + '\n' for task_id in ids))
            elif source == 'ids':
                (subset / 'instance_ids.txt').write_text('\n'.join(ids) + '\n')
            else:
                (traces / 'run_info.json').write_text(json.dumps({'tasks': [{'task_id': task_id} for task_id in ids]}))
            for task_id in ids:
                if isinstance(task_id, str) and task_id.strip():
                    task_dir = traces / task_id
                    task_dir.mkdir(exist_ok=True)
                    (task_dir / 'result.json').write_text('{"success": true, "verifier_result": {"rewards": {"reward": 1}}}')
            preds = traces / 'preds.json'
            preds.write_text(json.dumps({str(task_id): 'yes' for task_id in ids}))
            command = [sys.executable, '-c', BOOTSTRAP, str(ROOT / 'eval' / f'eval_{mode}.py'), '--subset', str(subset), '--traces-dir', str(traces), '--output', str(output)]
            if mode == 'gaia':
                command += ['--preds', str(preds)]
            result = subprocess.run(command, text=True, capture_output=True)
            return result, json.loads(output.read_text()) if output.exists() else None

    def test_duplicate_tasks_rejected_without_report(self):
        for mode in ('gaia', 'appworld', 'terminal_bench'):
            with self.subTest(mode=mode):
                result, report = self.run_cli(mode, ['same', 'same'])
                self.assertNotEqual(result.returncode, 0, report)
                self.assertIn('Duplicate task ID', result.stderr)
                self.assertIsNone(report)

    def test_nonempty_string_ids_required(self):
        for mode in ('gaia', 'appworld', 'terminal_bench'):
            for task_id in ('', '   ', None, 7):
                with self.subTest(mode=mode, task_id=task_id):
                    result, report = self.run_cli(mode, [task_id])
                    self.assertNotEqual(result.returncode, 0, report)
                    self.assertIn('nonempty string', result.stderr)
                    self.assertIsNone(report)

    def test_terminal_text_and_run_info_duplicates_rejected(self):
        for source in ('ids', 'run_info'):
            with self.subTest(source=source):
                result, report = self.run_cli('terminal_bench', ['same', 'same'], source)
                self.assertNotEqual(result.returncode, 0, report)
                self.assertIn('Duplicate task ID', result.stderr)
                self.assertIsNone(report)

    def test_unique_ids_keep_counts_consistent(self):
        for mode in ('gaia', 'appworld', 'terminal_bench'):
            with self.subTest(mode=mode):
                result, report = self.run_cli(mode, ['first', 'second'])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(report['total'], 2)
                self.assertEqual(report['resolved'], 2)
                self.assertEqual(len(set(report['resolved_ids'])), 2)
