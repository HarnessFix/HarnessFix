import importlib
import sys
import types
import unittest
from unittest.mock import patch

with patch.dict(sys.modules, {'dotenv': types.SimpleNamespace(load_dotenv=lambda *a, **kw: None)}):
    sampler = importlib.import_module('data.sample_gaia')


class GaiaSplitRequestsTests(unittest.TestCase):
    def test_levels_requested_only_by_validation_or_test_are_sampled(self):
        pool = {'1': [{'task_id': 'train'}], '2': [{'task_id': 'val'}], '3': [{'task_id': 'test'}]}
        self.assertEqual(sampler.stratified_sample_no_overlap(pool, {'1': 1}, {'2': 1}, {'3': 1}, 42),
                         ([{'task_id': 'train'}], [{'task_id': 'val'}], [{'task_id': 'test'}]))

    def test_negative_counts_rejected_for_every_split(self):
        pool = {'1': [{'task_id': str(index)} for index in range(10)]}
        for counts in ((-1, 2, 2), (2, -1, 2), (2, 2, -1)):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'non-negative'):
                sampler.stratified_sample_no_overlap(pool, *({'1': count} for count in counts), 42)

    def test_missing_pool_for_validation_only_level_raises(self):
        with self.assertRaisesRegex(ValueError, 'Not enough level-2'):
            sampler.stratified_sample_no_overlap({}, {}, {'2': 1}, {}, 42)

    def test_negative_count_on_test_only_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            sampler.stratified_sample_no_overlap({}, {}, {}, {'3': -1}, 42)

    def test_seeded_sampling_remains_disjoint_and_does_not_mutate_pool(self):
        pool = {'1': [{'task_id': str(index)} for index in range(10)]}
        original = list(pool['1'])
        counts = ({'1': 2}, {'1': 3}, {'1': 4})
        result = sampler.stratified_sample_no_overlap(pool, *counts, 42)
        self.assertEqual(result, sampler.stratified_sample_no_overlap(pool, *counts, 42))
        self.assertEqual(tuple(map(len, result)), (2, 3, 4))
        self.assertEqual(len({task['task_id'] for split in result for task in split}), 9)
        self.assertEqual(pool['1'], original)

    def test_empty_and_zero_requests_are_valid(self):
        self.assertEqual(sampler.stratified_sample_no_overlap({}, {}, {}, {}, 42), ([], [], []))
        self.assertEqual(sampler.stratified_sample_no_overlap({}, {}, {'2': 0}, {'3': 0}, 42), ([], [], []))
