import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FreshAuditTests(unittest.TestCase):
    def invoke(self, mode, fresh_result=None, dry_run=False):
        with patch.dict(sys.modules, {"dotenv": types.SimpleNamespace(load_dotenv=lambda *a, **kw: None)}):
            module = importlib.import_module("run_pipeline_" + mode)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            # All modes use a different report name; seed their known output paths.
            names = ["audit_swe_v1.json", "gaia_audit_v1.json", "audit_appworld_v1.json", "audit_terminal_bench_v1.json"]
            for name in names:
                (root / "results" / name).write_text('{"passed": true, "old": true}')
            def run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                if fresh_result is not None:
                    output.write_text(json.dumps(fresh_result))
                return subprocess.CompletedProcess(command, 0 if fresh_result and fresh_result["passed"] else 1)
            args = [1, "model", 0] if mode == "gaia" else [1, 0]
            with patch.object(module, "FAILURE_ANALYSIS_DIR", root), patch.object(module.subprocess, "run", side_effect=run) as mock_run:
                result = module.step_plan_diff_audit(*args, root / "plan.md", root / "candidate", dry_run)
                if dry_run:
                    mock_run.assert_not_called()
                    self.assertTrue(all((root / "results" / name).exists() for name in names))
                return result

    def test_failed_audit_does_not_reuse_old_approval(self):
        for mode in ("swe", "gaia", "appworld", "terminal_bench"):
            with self.subTest(mode=mode):
                result = self.invoke(mode)
                self.assertFalse(result["passed"])
                self.assertIn("audit_output_missing", result["violations"])

    def test_fresh_rejection_details_are_retained(self):
        report = {"passed": False, "violations": ["forbidden_paths_touched"]}
        for mode in ("swe", "gaia", "appworld", "terminal_bench"):
            with self.subTest(mode=mode):
                self.assertEqual(self.invoke(mode, report), report)

    def test_fresh_approval_is_accepted(self):
        for mode in ("swe", "gaia", "appworld", "terminal_bench"):
            with self.subTest(mode=mode):
                self.assertEqual(self.invoke(mode, {"passed": True}), {"passed": True})

    def test_dry_run_preserves_existing_evidence(self):
        for mode in ("swe", "gaia", "appworld", "terminal_bench"):
            with self.subTest(mode=mode):
                self.assertTrue(self.invoke(mode, dry_run=True)["dry_run"])


class FreshGateAndComparisonTests(unittest.TestCase):
    def invoke(self, mode, kind, fresh=None, returncode=1, dry_run=False):
        with patch.dict(sys.modules, {"dotenv": types.SimpleNamespace(load_dotenv=lambda *a, **kw: None)}):
            module = importlib.import_module("run_pipeline_" + mode)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            if kind == 'gate':
                name = 'gaia_gate_v1.json' if mode == 'gaia' else f'gate_{mode}_v1.json'
            else:
                name = f'{mode}_train_compare_v1.json'
            output = root / 'results' / name
            output.write_text('{"passed": true, "net_change": 7, "old": true}')
            def run(command, **kwargs):
                self.assertEqual(Path(command[command.index('--output') + 1]), output)
                if fresh is not None:
                    output.write_text(json.dumps(fresh))
                return subprocess.CompletedProcess(command, returncode)
            with patch.object(module, 'FAILURE_ANALYSIS_DIR', root), patch.object(module.subprocess, 'run', side_effect=run) as mock_run:
                if kind == 'gate':
                    result = module.step_gate_check(1, root, root, root, root, root / 'plan', 0, 1, 1.5, dry_run)
                elif mode == 'gaia':
                    result = module.step_compare_train(1, 'model', root, root, dry_run)
                else:
                    result = module.step_compare_train(1, 'model', root, root, root, root, root / 'plan', dry_run)
                if dry_run:
                    mock_run.assert_not_called()
                    self.assertTrue(json.loads(output.read_text())['old'])
                return result

    def test_missing_gate_output_never_approves(self):
        for mode in ('gaia', 'appworld', 'terminal_bench'):
            for returncode in (0, 1):
                with self.subTest(mode=mode, returncode=returncode):
                    self.assertFalse(self.invoke(mode, 'gate', returncode=returncode)['passed'])

    def test_missing_comparison_output_never_reuses_old_improvement(self):
        for mode in ('gaia', 'appworld', 'terminal_bench'):
            with self.subTest(mode=mode):
                self.assertEqual(self.invoke(mode, 'compare')['net_change'], 0)

    def test_fresh_reports_are_preserved(self):
        for mode in ('gaia', 'appworld', 'terminal_bench'):
            for kind in ('gate', 'compare'):
                for report in ({'passed': True, 'net_change': 2}, {'passed': False, 'net_change': -1}):
                    with self.subTest(mode=mode, kind=kind, report=report):
                        self.assertEqual(self.invoke(mode, kind, fresh=report, returncode=0 if report['passed'] else 1), report)

    def test_dry_runs_leave_existing_reports_intact(self):
        for mode in ('gaia', 'appworld', 'terminal_bench'):
            for kind in ('gate', 'compare'):
                with self.subTest(mode=mode, kind=kind):
                    self.assertTrue(self.invoke(mode, kind, dry_run=True)['dry_run'])
