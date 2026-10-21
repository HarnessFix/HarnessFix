import unittest
from failure_analysis.artifact_sanitizer import sanitize_for_prompt


class EmptyOutputTests(unittest.TestCase):
    def test_unavailable_raw_output_keeps_content(self):
        for raw in (None, "", {}, []):
            with self.subTest(raw=raw):
                message = {"role": "user", "content": "Command timed out", "extra": {"raw_output": raw}}
                self.assertEqual(sanitize_for_prompt(message).get("content"), "Command timed out")

    def test_available_raw_output_still_removes_duplicate(self):
        message = {"content": "wrapped output", "extra": {"raw_output": "output"}}
        self.assertNotIn("content", sanitize_for_prompt(message))
        self.assertEqual(sanitize_for_prompt(message)["extra"]["raw_output"], "output")

    def test_message_without_raw_output_keeps_content(self):
        self.assertEqual(sanitize_for_prompt({"content": "error", "extra": {"cost": 1}}), {"content": "error"})
