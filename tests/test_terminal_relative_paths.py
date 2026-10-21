import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task_agent.terminal_bench_agent import run_terminal_bench_entry as entry


class RelativePathTests(unittest.TestCase):
    def test_cli_resolves_paths_before_harbor_changes_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            caller = Path(tmp).resolve()
            (caller / 'subset').mkdir()
            (caller / 'agent').mkdir()
            previous = Path.cwd()
            try:
                os.chdir(caller)
                def run(args):
                    self.assertEqual(args.subset, caller / 'subset')
                    self.assertEqual(args.output, caller / 'output')
                    self.assertEqual(args.mini_swe_agent_src, caller / 'agent')
                    return args.output / 'harbor_jobs' / 'output'
                with patch('sys.argv', ['runner', '--subset', 'subset', '--output', 'output', '--mini-swe-agent-src', 'agent']), patch.object(entry, '_run_harbor', side_effect=run), patch.object(entry, '_convert_outputs') as convert:
                    entry.main()
                self.assertEqual(convert.call_args.args[1], caller / 'output')
            finally:
                os.chdir(previous)
