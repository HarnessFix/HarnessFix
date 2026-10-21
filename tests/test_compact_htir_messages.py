import unittest
from failure_analysis.htir import _finalize_paper_htir, _node
from failure_analysis.print_htir_compact import build_compact_bundle


class CompactMessageTests(unittest.TestCase):
    def test_generated_step_messages_survive_projection(self):
        node = _node("evt_001", "ModelInvocationEvent", 1, "controller", "answer",
                     "traj.messages[2]", {"request_summary": "inspect failing test",
                                          "response_summary": "run pytest"})
        bundle = _finalize_paper_htir({"graph": {"nodes": [node], "edges": []}})
        step = bundle["agent_trace_steps"][0]
        compact = build_compact_bundle(bundle, 16)["agent_trace_steps"][0]
        self.assertEqual(compact.get("request_message"), step["request_message"])
        self.assertEqual(compact.get("response_message"), step["response_message"])

    def test_legacy_summaries_remain_available(self):
        step = {"step_id": "step_001", "request_summary": "question", "response_summary": "answer"}
        compact = build_compact_bundle({"agent_trace_steps": [step]}, 16)["agent_trace_steps"][0]
        self.assertEqual(compact["request_summary"], "question")
        self.assertEqual(compact["response_summary"], "answer")

    def test_empty_messages_are_omitted(self):
        step = {"step_id": "step_001", "request_message": "", "response_message": ""}
        compact = build_compact_bundle({"agent_trace_steps": [step]}, 16)["agent_trace_steps"][0]
        self.assertNotIn("request_message", compact)
        self.assertNotIn("response_message", compact)
