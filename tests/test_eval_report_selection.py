import json
import tempfile
import unittest
from pathlib import Path
from failure_analysis.validation_metrics import load_eval_json


class ReportSelectionTests(unittest.TestCase):
    def test_multiple_reports_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['older.json', 'newer.json']:
                (root / name).write_text(json.dumps({'resolved_ids': [name]}))
            with self.assertRaisesRegex(ValueError, 'Multiple JSON reports'):
                load_eval_json(root)
            path, data = load_eval_json(root / 'newer.json')
            self.assertEqual(path, root / 'newer.json')
            self.assertEqual(data['resolved_ids'], ['newer.json'])

    def test_single_report_ignores_run_metadata_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'run_info.json').write_text('{}')
            (root / 'directory.json').mkdir()
            (root / 'report.json').write_text('{"resolved_ids": []}')
            self.assertEqual(load_eval_json(root), (root / 'report.json', {'resolved_ids': []}))

    def test_no_report_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'run_info.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):
                load_eval_json(root)
