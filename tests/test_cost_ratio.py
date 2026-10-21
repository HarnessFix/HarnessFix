"""Cost comparisons must preserve zero measurements and matching units."""
import unittest

from failure_analysis.validation_metrics import compute_cost_ratio


class CostRatioTests(unittest.TestCase):
    def ratio(self, baseline, current):
        return compute_cost_ratio({"metrics": baseline}, {"metrics": current})

    def test_zero_current_cost_is_not_replaced_with_step_ratio(self):
        self.assertEqual(self.ratio(
            {"avg_instance_cost": 2, "avg_steps": 10},
            {"avg_instance_cost": 0, "avg_steps": 20}), 0)

    def test_zero_current_steps_are_not_replaced_with_api_calls(self):
        self.assertEqual(self.ratio(
            {"avg_steps": 10, "avg_api_calls": 5},
            {"avg_steps": 0, "avg_api_calls": 20}), 0)

    def test_zero_current_api_calls_are_not_replaced_with_baseline(self):
        self.assertEqual(self.ratio({"avg_api_calls": 5}, {"avg_api_calls": 0}), 0)

    def test_missing_current_steps_uses_shared_api_call_metric(self):
        self.assertEqual(self.ratio(
            {"avg_steps": 10, "avg_api_calls": 2}, {"avg_api_calls": 4}), 2)

    def test_missing_baseline_steps_uses_shared_api_call_metric(self):
        self.assertEqual(self.ratio(
            {"avg_api_calls": 2}, {"avg_steps": 10, "avg_api_calls": 4}), 2)

    def test_cost_takes_precedence_over_steps(self):
        self.assertEqual(self.ratio(
            {"avg_instance_cost": 2, "avg_steps": 10},
            {"avg_instance_cost": 3, "avg_steps": 20}), 1.5)

    def test_zero_baseline_cost_falls_back_to_shared_steps(self):
        self.assertEqual(self.ratio(
            {"avg_instance_cost": 0, "avg_steps": 10},
            {"avg_instance_cost": 0, "avg_steps": 20}), 2)

    def test_no_comparable_denominator_returns_neutral_ratio(self):
        for baseline, current in [({}, {}), ({"avg_steps": 0}, {"avg_steps": 2}),
                                  ({"avg_steps": 2}, {"avg_api_calls": 4})]:
            with self.subTest(baseline=baseline, current=current):
                self.assertEqual(self.ratio(baseline, current), 1)

    def test_none_values_fall_back_to_shared_steps(self):
        self.assertEqual(self.ratio(
            {"avg_instance_cost": None, "avg_steps": 2},
            {"avg_instance_cost": None, "avg_steps": 4}), 2)


if __name__ == "__main__":
    unittest.main()
