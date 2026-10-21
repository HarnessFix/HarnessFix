import json
import tempfile
import unittest
from pathlib import Path
from failure_analysis.validation_metrics import compute_run_metrics


class RatePopulationTests(unittest.TestCase):
    def test_missing_traces_do_not_change_evaluation_rates(self):
        for mode in ['swe', 'gaia', 'appworld', 'terminal_bench']:
            for traces in [0, 1, 3]:
                with self.subTest(mode=mode, traces=traces), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    report = root / 'report.json'
                    report.write_text(json.dumps({'total': 2, 'total_instances': 2, 'resolved_ids': ['a'], 'error_ids': ['b']}))
                    for i in range(traces):
                        task = root / 'traces' / str(i)
                        task.mkdir(parents=True)
                        (task / (str(i) + '.traj.json')).write_text('{}')
                    metrics = compute_run_metrics(mode, root / 'traces', report)['metrics']
                    key = 'accuracy' if mode == 'gaia' else 'resolved_rate'
                    self.assertEqual(metrics[key], .5)
                    self.assertEqual(metrics['error_rate'], .5)

    def test_report_id_union_supports_legacy_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / 'report.json'
            report.write_text('{"resolved_ids": ["a"], "unresolved_ids": ["b"]}')
            self.assertEqual(compute_run_metrics('swe', root / 'traces', report)['metrics']['resolved_rate'], .5)
