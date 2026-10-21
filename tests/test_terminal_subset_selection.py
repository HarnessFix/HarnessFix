import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from task_agent.terminal_bench_agent import run_terminal_bench_entry as entry


class SubsetSelectionTests(unittest.TestCase):
    def test_explicit_missing_subset_does_not_launch_full_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sys.argv', ['runner', '--subset', str(root / 'typo'), '--output', str(root / 'output')]), patch.object(entry, '_run_harbor') as run, patch.object(entry, '_convert_outputs'):
                with self.assertRaises(SystemExit) as error:
                    entry.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_default_still_supports_registered_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = root / 'not-downloaded'
            with patch.object(entry, 'DEFAULT_DATASET', default), patch('sys.argv', ['runner', '--output', str(root / 'output')]), patch.object(entry, '_run_harbor') as run, patch.object(entry, '_convert_outputs'):
                entry.main()
                self.assertEqual(entry._dataset_args(run.call_args.args[0].subset, None), ['--dataset', 'terminal-bench@2.0'])
