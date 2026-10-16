"""Independent metric oracles and deterministic selection guards for PR0."""
import unittest

import numpy as np

from .metrics import Cuboid, euler_xyz_matrix
from .nuscenes_baseline import bev_iou, rate, select_cases, summarize, valid_success


class BaselineTests(unittest.TestCase):
    def box(self, center=(0, 0, 0), rotation=(0, 0, 0)):
        return Cuboid(np.array(center), np.array([2., 2., 2.]), euler_xyz_matrix(rotation))

    def test_bev_exact_overlap_and_partial_translation(self):
        self.assertAlmostEqual(bev_iou(self.box(), self.box()), 1.)
        self.assertAlmostEqual(bev_iou(self.box(), self.box((1, 0, 100))), 1 / 3)
        self.assertEqual(bev_iou(self.box(), self.box((5, 0, 0))), 0.)

    def test_bev_rotated_square_matches_analytic_area(self):
        self.assertAlmostEqual(bev_iou(self.box(), self.box(rotation=(0, 0, np.pi/4))), 1 / np.sqrt(2))

    def test_selector_permutation_independent_no_unseedable_or_dev_leak(self):
        def case(token, split='held_out', supported=True, points='1-4'):
            return {'case_id': token, 'split': split, 'target_label': 'vehicles', 'vehicles_type': 'car',
                    'seeds': [{'point': [0, 0, 0]}] if supported else [], 'bins': {'points': points}}
        cases = [case('a'), case('b'), case('c', points='50+'), case('d', split='dev'), case('e', supported=False)]
        manifest = {'random_seed': 20261016, 'cases': cases}
        first = select_cases(manifest, 'held_out', 2)
        second = select_cases({**manifest, 'cases': list(reversed(cases))}, 'held_out', 2)
        self.assertEqual(first, second)
        self.assertIn('c', first['case_ids'])
        self.assertEqual(first['eligible_by_route'], {'vehicles/car': 3})
        self.assertEqual(first['unseedable_by_route'], {'vehicles/car': 1})

    def test_no_output_is_not_accuracy_or_containment_success(self):
        summary = summarize([{'route': 'vehicles/car', 'response': {'status': 'no-fit', 'reason': 'timeout'},
                              'outcome': 'timeout', 'http_seconds': 1.7, 'bins': {}}])['all']
        self.assertEqual(summary['coverage']['rate'], 0.)
        self.assertIsNone(summary['metrics']['iou_3d']['p50'])
        self.assertIsNone(summary['seed_containment']['rate'])

    def test_wilson_does_not_claim_certainty_from_one_sample(self):
        self.assertIsNone(rate(0, 0)['rate'])
        self.assertLess(rate(1, 1)['wilson_95'][0], .21)
        self.assertGreater(rate(0, 1)['wilson_95'][1], .79)

    def test_stale_identity_wrong_provider_and_invalid_geometry_never_count(self):
        meta = {'requestId': 'run-id', 'frame': 2, 'allowEstimated': True}
        response = {**meta, 'status': 'ok', 'provider': 'snap', 'quality': 'confident',
                    'center': [0, 0, 0], 'size': [4, 2, 1.5], 'rotation': [0, 0, 0]}
        self.assertTrue(valid_success(response, meta))
        for wrong in ({'frame': 3}, {'requestId': 'stale'}, {'provider': 'mock'},
                      {'size': [0, 2, 1]}, {'center': [float('nan'), 0, 0]},
                      {'quality': 'estimated-ish'}, {'provider': 'geometry-only'}):
            self.assertFalse(valid_success({**response, **wrong}, meta))
        self.assertTrue(valid_success({**response, 'provider': 'geometry-only', 'quality': 'estimated'}, meta))


if __name__ == '__main__':
    unittest.main()
