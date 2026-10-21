import contextlib
import importlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class StopBeforeInference(Exception):
    pass


class ForceCandidateCacheTests(unittest.TestCase):
    def exercise(self, mode, flags, should_clear):
        with patch.dict(sys.modules, {'dotenv': types.SimpleNamespace(load_dotenv=lambda *a, **kw: None)}):
            module = importlib.import_module('run_pipeline_' + mode)
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            constants = ('TASK_AGENT_DIR', 'TRACES_DIR', 'EVAL_DIR', 'LOGS_DIR', 'TRAIN_DIR', 'VAL_DIR', 'TEST_DIR', 'FULL_DATASET_DIR', 'TRAIN_DATASET_DIR', 'VAL_DATASET_DIR')
            for name in constants:
                if hasattr(module, name):
                    path = root / name
                    path.mkdir()
                    (path / 'instance_ids.txt').write_text('one\n')
                    stack.enter_context(patch.object(module, name, path))
            if mode == 'terminal_bench':
                (root / 'agent' / 'harbor' / 'src').mkdir(parents=True)
                stack.enter_context(patch.object(module, 'TERMINAL_AGENT_DIR', root / 'agent'))
            model = 'test/model'
            version = 2
            if mode == 'gaia':
                agent = module.base_agent_dir(model, version)
                caches = [module.enhanced_train_traces_dir(model, version), module.enhanced_train_eval_dir(model, version)]
            else:
                agent = module.base_agent_dir(version)
                caches = [module.train_traces_dir(model, version), module.train_eval_dir(model, version)]
            caches += [module.val_enhanced_traces_dir(model, version), module.val_enhanced_eval_dir(model, version)]
            if mode == 'swe':
                caches += [module.train_logs_dir(model, version), module.LOGS_DIR / module.val_enhanced_run_id(model, version) / module.eval_model_slug(model)]
            for path in [agent, *caches]:
                path.mkdir(parents=True)
                (path / 'old-result').write_text('old')
            baseline = root / 'TRACES_DIR' / 'baseline-to-preserve'
            baseline.mkdir()
            stack.enter_context(patch.object(module, 'step_val_baseline_inference', return_value=baseline))
            stack.enter_context(patch.object(module, 'step_val_baseline_evaluate', return_value=root))
            stack.enter_context(patch.object(module, 'step_train_inference', side_effect=StopBeforeInference))
            argv = ['pipeline', '--model', model, '--start-version', '2', '--max-iterations', '1', *flags]
            stack.enter_context(patch('sys.argv', argv))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(StopBeforeInference):
                module.main()
            for path in [agent, *caches]:
                self.assertEqual(path.exists(), not should_clear, str(path))
            self.assertTrue(baseline.exists())

    def test_force_clears_candidate_code_traces_reports_and_harness_logs(self):
        for mode in ('swe', 'gaia', 'appworld', 'terminal_bench'):
            with self.subTest(mode=mode):
                self.exercise(mode, ['--force'], True)

    def test_normal_resume_preserves_candidate_results(self):
        for mode in ('swe', 'gaia', 'appworld', 'terminal_bench'):
            with self.subTest(mode=mode):
                self.exercise(mode, [], False)

    def test_forced_dry_run_preserves_candidate_results(self):
        for mode in ('swe', 'gaia', 'appworld', 'terminal_bench'):
            with self.subTest(mode=mode):
                self.exercise(mode, ['--force', '--dry-run'], False)
