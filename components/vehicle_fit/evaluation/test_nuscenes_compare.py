import json
from pathlib import Path
import tempfile
import unittest

from .nuscenes_compare import compare


class RepeatComparisonTests(unittest.TestCase):
    def test_mask_and_outcome_drift_are_not_hidden_by_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'a', Path(temp) / 'b'
            identity = {'manifest_sha256': 'same', 'source_hashes': {'geometry.py': 'same'},
                        'config_sha256': 'same', 'timeout_ms': 1700, 'seed_tolerance_m': .05}
            row = {'case_id': 'car', 'sample_token': 'frame', 'route': 'vehicles/car',
                   'request': {'seed': {'x': 1}}, 'response': {'status': 'no-fit', 'reason': 'ambiguous',
                                                             'diagnostics': {'maskIndices': [1, 2]}}}
            for path in (left, right):
                path.mkdir()
                (path / 'identity.json').write_text(json.dumps(identity))
                (path / 'cases.jsonl').write_text(json.dumps(row) + '\n')
            self.assertTrue(compare(left, right)['exact_match'])
            row['response']['diagnostics']['maskIndices'] = [1, 3]
            (right / 'cases.jsonl').write_text(json.dumps(row) + '\n')
            result = compare(left, right)
            self.assertFalse(result['exact_match'])
            self.assertEqual(result['changed_masks'][0]['symmetric_difference'], 2)
            self.assertEqual(result['changed_outcomes'], [])
            row['response']['reason'] = 'insufficient-evidence'
            (right / 'cases.jsonl').write_text(json.dumps(row) + '\n')
            self.assertEqual(compare(left, right)['changed_outcomes'], ['car'])
            identity['config_sha256'] = 'different'
            (right / 'identity.json').write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError, 'identical config'):
                compare(left, right)


if __name__ == '__main__':
    unittest.main()
