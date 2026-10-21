import json
import tempfile
import unittest
from pathlib import Path

from data.sample_terminal_bench import _find_task_dirs, _load_manifest_splits, _sample_random_splits


class SplitIntegrityTests(unittest.TestCase):
    def manifest(self, train, val=(), test=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'splits': {
                'terminal_bench_train': list(train),
                'terminal_bench_val': list(val),
                'terminal_bench_test': list(test),
            }}))
            return _load_manifest_splits(path, {name: Path(name) for name in ('a', 'b', 'c')})

    def test_duplicate_within_each_split_rejected(self):
        for index in range(3):
            splits = [[], [], []]
            splits[index] = ['a', 'a']
            with self.subTest(index=index), self.assertRaisesRegex(SystemExit, 'Duplicate'):
                self.manifest(*splits)

    def test_duplicate_between_splits_still_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'Duplicate'):
            self.manifest(['a'], ['a'])

    def test_valid_manifest_preserves_order(self):
        result = self.manifest(['b', 'a'], [], ['c'])
        self.assertEqual(result['terminal_bench_train'], [Path('b'), Path('a')])
        self.assertEqual(result['terminal_bench_test'], [Path('c')])

    def test_negative_counts_rejected(self):
        tasks = [Path(str(index)) for index in range(10)]
        for counts in ((-1, 2, 2), (2, -1, 2), (2, 2, -1)):
            with self.subTest(counts=counts), self.assertRaisesRegex(SystemExit, 'non-negative'):
                _sample_random_splits(tasks, *counts, seed=42)

    def test_zero_and_positive_counts_are_disjoint_and_reproducible(self):
        tasks = [Path(str(index)) for index in range(10)]
        result = _sample_random_splits(tasks, 0, 3, 5, 42)
        self.assertEqual(result, _sample_random_splits(tasks, 0, 3, 5, 42))
        self.assertEqual([len(values) for values in result.values()], [0, 3, 5])
        flat = [task for split in result.values() for task in split]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(tasks, [Path(str(index)) for index in range(10)])

    def test_oversized_request_still_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'only found'):
            _sample_random_splits([Path('a')], 1, 1, 1, 42)


class SourceTaskIdentityTests(unittest.TestCase):
    def test_duplicate_directory_names_are_rejected_before_splitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group in ('first', 'second'):
                task = root / group / 'same-task'
                task.mkdir(parents=True)
                (task / 'task.toml').write_text('')
            with self.assertRaisesRegex(SystemExit, 'Duplicate.*same-task'):
                _find_task_dirs(root)

    def test_distinct_names_remain_sorted_and_hidden_tasks_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('group/b', 'a', '.hidden/a'):
                task = root / name
                task.mkdir(parents=True)
                (task / 'task.toml').write_text('')
            self.assertEqual([task.name for task in _find_task_dirs(root)], ['a', 'b'])
