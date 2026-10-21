import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from failure_analysis.validation_metrics import evaluate_target_metrics


class ElapsedTimeMetricTests(unittest.TestCase):
    def test_direction_and_ties(self):
        plan = {"fixes": [{"target_metrics": ["avg_elapsed_seconds"]}]}
        for current, expected_delta, improved in [(5.0, 5.0, True), (15.0, -5.0, False), (10.0, 0.0, False)]:
            with self.subTest(current=current):
                result = evaluate_target_metrics(
                    plan,
                    {"mode": "gaia", "metrics": {"avg_elapsed_seconds": 10.0}},
                    {"mode": "gaia", "metrics": {"avg_elapsed_seconds": current}},
                )
                self.assertEqual(result["metrics"][0]["direction"], "lower")
                self.assertEqual(result["aggregate_delta"], expected_delta)
                self.assertEqual(result["improved_metric_count"], int(improved))

    def test_gate_rejects_slowdown_and_accepts_speedup(self):
        gate = Path(__file__).resolve().parents[1] / "failure_analysis" / "check_val_gate.py"
        for elapsed, expected_pass in [(20.0, False), (5.0, True)]:
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "ids.txt").write_text("task-a\n")
                (root / "plan.json").write_text(json.dumps({"fixes": [{"target_metrics": ["avg_elapsed_seconds"]}]}))
                for label, duration, resolved in [("baseline", 10.0, []), ("current", elapsed, ["task-a"])]:
                    (root / f"{label}.json").write_text(json.dumps({"resolved_ids": resolved, "accuracy": len(resolved)}))
                    task = root / label / "task-a"
                    task.mkdir(parents=True)
                    (task / "task-a.traj.json").write_text(json.dumps({"info": {"elapsed_seconds": duration, "prediction": "answer"}, "trajectory": []}))
                process = subprocess.run([
                    sys.executable, str(gate), "--mode", "gaia",
                    "--baseline-eval", str(root / "baseline.json"),
                    "--current-eval", str(root / "current.json"),
                    "--baseline-traces", str(root / "baseline"),
                    "--current-traces", str(root / "current"),
                    "--plan-spec", str(root / "plan.json"),
                    "--val-ids", str(root / "ids.txt"),
                    "--output", str(root / "result.json"),
                ], capture_output=True, text=True)
                self.assertEqual(process.returncode, 0 if expected_pass else 1, process.stdout + process.stderr)
                result = json.loads((root / "result.json").read_text())
                self.assertTrue(result["resolved_pass"])
                self.assertEqual(result["target_pass"], expected_pass)
                self.assertEqual(result["passed"], expected_pass)


if __name__ == "__main__":
    unittest.main()
