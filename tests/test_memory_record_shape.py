import json
import tempfile
import unittest
from pathlib import Path
from failure_analysis.harness_memory import load_memory_entries, retrieve_relevant_memories


class MemoryRecordShapeTests(unittest.TestCase):
    def test_nonobject_json_rows_do_not_break_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = {"mode": "swe", "summary": "keep this repair"}
            rows = [None, [], "text", 1, False, valid]
            (root / "accepted_repairs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            self.assertEqual(retrieve_relevant_memories(root, "swe", []), [valid])

    def test_malformed_lines_and_empty_lines_preserve_valid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rejected_repairs.jsonl").write_text('\n{invalid\n{"mode":"gaia"}\n')
            self.assertEqual(load_memory_entries(root), [{"mode": "gaia"}])

    def test_missing_memory_files_are_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_memory_entries(Path(directory)), [])
