import tempfile
import unittest
from pathlib import Path

from failure_analysis.plan_diff_audit import audit_candidate


class PythonDeletionAuditTests(unittest.TestCase):
    def audit(self, original_files, candidate_files, budget=None):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            candidate = Path(directory) / "candidate"
            for root, files in ((original, original_files), (candidate, candidate_files)):
                root.mkdir()
                for name, content in files.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
            return audit_candidate(original, candidate, {
                "edit_budget": budget or {},
                "fixes": [{"id": "remove-obsolete", "target_files": ["obsolete.py"]}],
            })

    def test_planned_python_deletion_passes_and_covers_target(self):
        result = self.audit({"obsolete.py": "value = 1\n"}, {})
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["changed_files"], ["obsolete.py"])
        self.assertEqual(result["changed_file_count"], 1)
        self.assertEqual(result["syntax_failures"], [])
        self.assertTrue(result["fix_coverage"]["remove-obsolete"]["covered"])

    def test_deletion_still_counts_against_budget(self):
        result = self.audit({"obsolete.py": "pass\n"}, {}, {"max_files_to_modify": 0})
        self.assertEqual(result["violations"], ["changed_files_exceed_budget:1>0"])

    def test_forbidden_deletion_is_rejected(self):
        result = self.audit({"obsolete.py": "pass\n"}, {}, {"forbidden_paths": ["obsolete.py"]})
        self.assertEqual(result["violations"], ["forbidden_paths_touched"])

    def test_out_of_scope_deletion_is_rejected(self):
        result = self.audit({"obsolete.py": "pass\n"}, {}, {"allowed_paths": ["src/**"]})
        self.assertEqual(result["violations"], ["changed_files_outside_allowed_paths"])

    def test_invalid_added_python_is_rejected_alongside_deletion(self):
        result = self.audit({"obsolete.py": "pass\n"}, {"added.py": "def broken(\n"})
        self.assertEqual(result["violations"], ["syntax_failures"])
        self.assertEqual(len(result["syntax_failures"]), 1)
        self.assertTrue(result["syntax_failures"][0].startswith("added.py: SyntaxError:"))

    def test_invalid_modified_python_is_rejected_alongside_deletion(self):
        result = self.audit({"obsolete.py": "pass\n", "modified.py": "pass\n"},
                            {"modified.py": "def broken(\n"})
        self.assertEqual(result["violations"], ["syntax_failures"])
        self.assertEqual(len(result["syntax_failures"]), 1)
        self.assertTrue(result["syntax_failures"][0].startswith("modified.py: SyntaxError:"))

    def test_directory_replacing_python_file_is_not_silently_skipped(self):
        result = self.audit({"obsolete.py": "pass\n"}, {"obsolete.py/data.txt": "data"})
        self.assertEqual(result["violations"], ["syntax_failures"])
        self.assertTrue(result["syntax_failures"][0].startswith("obsolete.py: IsADirectoryError:"))


if __name__ == "__main__":
    unittest.main()
