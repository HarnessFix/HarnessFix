import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import download_terminal_bench as download


class DownloadManifestTests(unittest.TestCase):
    def test_custom_manifest_used_for_revision_and_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'custom.json'
            manifest.write_text('{"source_commit": "pinned-revision"}')
            argv = ['download', '--dest', str(root / 'checkout'), '--manifest', str(manifest)]
            with patch('sys.argv', argv), patch.object(download, '_run') as run:
                download.main()
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn('pinned-revision', commands[1])
            sample = commands[-1]
            self.assertIn('--manifest', sample)
            self.assertEqual(Path(sample[sample.index('--manifest') + 1]), manifest.resolve())

    def test_relative_manifest_resolved_before_child_changes_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['download', '--source', 'github', '--dest', str(Path(directory) / 'checkout'), '--manifest', 'custom.json']
            with patch('sys.argv', argv), patch.object(download, '_run') as run:
                download.main()
            sample = run.call_args.args[0]
            self.assertIn('--manifest', sample)
            self.assertEqual(Path(sample[sample.index('--manifest') + 1]), Path('custom.json').resolve())

    def test_skip_sampling_does_not_invoke_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['download', '--source', 'github', '--dest', str(Path(directory) / 'checkout'), '--skip-sampling']
            with patch('sys.argv', argv), patch.object(download, '_run') as run:
                download.main()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][:2], ['git', 'clone'])
