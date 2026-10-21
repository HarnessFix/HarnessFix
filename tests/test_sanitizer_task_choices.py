import unittest
from failure_analysis.artifact_sanitizer import sanitize_for_prompt


class TaskChoiceTests(unittest.TestCase):
    def test_task_choices_and_identifier_survive(self):
        task = {'id': 'quiz-1', 'question': 'Which option?', 'choices': ['A', 'B']}
        self.assertEqual(sanitize_for_prompt(task), task)

    def test_structured_task_choices_survive(self):
        task = {'id': 'quiz-2', 'choices': [{'label': 'A', 'value': 1}]}
        self.assertEqual(sanitize_for_prompt(task), task)

    def test_typed_provider_response_is_still_removed(self):
        response = {'object': 'chat.completion', 'id': 'completion-id', 'choices': [{'message': {'content': 'answer'}}]}
        self.assertEqual(sanitize_for_prompt(response), {})

    def test_untyped_completion_choices_are_still_removed(self):
        for choice in [{'message': {'content': 'answer'}}, {'delta': {'content': 'part'}}, {'text': 'completion'}]:
            with self.subTest(choice=choice):
                response = {'id': 'completion-id', 'choices': [dict(index=0, **choice)]}
                self.assertEqual(sanitize_for_prompt(response), {})
