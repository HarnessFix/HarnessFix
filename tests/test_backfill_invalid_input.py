import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from task_agent.appworld_agent import backfill_official_traces as backfill


class BackfillInputTests(unittest.TestCase):
    def test_invalid_original_is_never_overwritten(self):
        for original in ['{incomplete', 'null', '[]', '"text"']:
            with self.subTest(original=original), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / 'task.traj.json'
                path.write_text(original)
                with patch.object(backfill, 'read_official_model_calls', return_value=[]), patch.object(backfill, 'read_official_messages', return_value=[{'role': 'user', 'content': 'log'}]):
                    result = backfill.backfill_trace(root, path, dry_run=False, force=True)
                self.assertEqual(result['status'], 'skipped')
                self.assertEqual(path.read_text(), original)

    def test_valid_original_keeps_unrelated_evidence(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'task.traj.json'
            path.write_text('{"info":{"prediction":"42"},"custom":"keep"}')
            with patch.object(backfill, 'read_official_model_calls', return_value=[]), patch.object(backfill, 'read_official_messages', return_value=[{'role': 'user', 'content': 'log'}]):
                result = backfill.backfill_trace(root, path, dry_run=False, force=False)
            self.assertEqual(result['status'], 'updated')
            data = json.loads(path.read_text())
            self.assertEqual(data['custom'], 'keep')
            self.assertEqual(data['info']['prediction'], '42')
