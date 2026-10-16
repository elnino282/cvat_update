import unittest
from unittest.mock import patch

import numpy as np

from vehicle_fit.geometry import NoFit
from vehicle_fit.large_vehicle import fit_truck, orientation_candidates
from vehicle_fit.service import FitRuntime
from vehicle_fit.tests.test_geometry import road_scene
from vehicle_fit.tests.test_service import metadata


def box_scene(length=7., width=2.5, height=3., *, y=4., side_only=False):
    base = road_scene()
    ground = base[base[:, 2] == 0]
    x = np.linspace(10 - length / 2, 10 + length / 2, int(np.ceil(length / .22)) + 1)
    ys = np.linspace(y - width / 2, y + width / 2, int(np.ceil(width / .22)) + 1)
    z = np.linspace(.2, height, int(np.ceil(height / .20)) + 1)
    xx, zz = np.meshgrid(x, z)
    surfaces = [np.column_stack((xx.ravel(), np.full(xx.size, y - width / 2), zz.ravel(), np.zeros(xx.size)))]
    if not side_only:
        surfaces.append(np.column_stack((xx.ravel(), np.full(xx.size, y + width / 2), zz.ravel(), np.zeros(xx.size))))
        yy, zz = np.meshgrid(ys, z)
        for edge in (10 - length / 2, 10 + length / 2):
            surfaces.append(np.column_stack((np.full(yy.size, edge), yy.ravel(), zz.ravel(), np.zeros(yy.size))))
        xx, yy = np.meshgrid(x, ys)
        surfaces.append(np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, height), np.zeros(xx.size))))
    return np.vstack((ground, *surfaces)), len(ground)


class TruckTests(unittest.TestCase):
    def test_geometry_candidates_are_bounded_deterministic_and_cover_methods(self):
        cloud, index = box_scene()
        proposals = orientation_candidates(cloud[index:], lambda: False)
        self.assertEqual(proposals, orientation_candidates(cloud[index:], lambda: False))
        self.assertLessEqual(len(proposals), 10)
        self.assertIn('robust-pca', {p[1] for p in proposals})

    def test_full_truck_uses_own_prior_and_ground(self):
        cloud, index = box_scene()
        result = fit_truck(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(result['quality'], 'estimated')
        self.assertTrue(result['diagnostics']['prototypeKey'].startswith('vehicle.truck:'))
        np.testing.assert_allclose(result['size'], [7.02, 2.52, 3.02], atol=.04)
        self.assertAlmostEqual(result['center'][2] - result['size'][2] / 2, 0.)

    def test_single_long_side_is_preserved_during_hidden_width_completion(self):
        cloud, index = box_scene(side_only=True)
        result = fit_truck(cloud, cloud[index, :3], seed_index=index)
        self.assertAlmostEqual(result['rotation'][2], 0., places=4)
        self.assertAlmostEqual(result['center'][1] - result['size'][1] / 2, 2.75, places=4)
        self.assertGreater(result['size'][1], 2.29)
        self.assertEqual(result['diagnostics']['dimensionWeights'][1], 0.)

    def test_detached_neighbor_is_not_selected_even_if_larger_mask(self):
        cloud, index = box_scene(length=5.3, width=2., height=1.9, y=2.)
        other, start = box_scene(y=6.)
        cloud = np.vstack((cloud, other[start:]))
        result = fit_truck(cloud, cloud[index, :3], seed_index=index, mask=cloud[:, 2] > .1)
        self.assertLess(result['center'][1], 3.)
        self.assertGreater(result['diagnostics']['detachedMaskPoints'], result['diagnostics']['selectedPoints'])

    def test_connected_long_trailer_cannot_be_clamped_into_truck_prior(self):
        cloud, index = box_scene(length=15., side_only=True)
        with self.assertRaises(NoFit) as caught:
            fit_truck(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(caught.exception.reason, 'unsafe-fit')
        self.assertEqual(caught.exception.diagnostics['safetyGate'], 'class-extent')

    def test_sparse_truck_estimate_contains_seed_and_uses_truck_dimensions(self):
        cloud, index = box_scene()
        cloud = np.vstack((cloud[:index], [[10, 4, 1, 0]]))
        result = fit_truck(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(result['quality'], 'estimated')
        self.assertGreater(result['size'][0], 4.7)
        self.assertTrue(result['diagnostics']['seedContained'])
        self.assertTrue(result['diagnostics']['yawUncertain'])

    def test_cancelled_candidate_search_ends(self):
        cloud, index = box_scene()
        with self.assertRaises(NoFit) as caught:
            fit_truck(cloud, cloud[index, :3], seed_index=index, cancelled=lambda: True)
        self.assertEqual(caught.exception.reason, 'cancelled')

    def test_route_only_enabled_explicitly(self):
        with patch.dict('os.environ', VEHICLE_FIT_BACKEND='geometry-only', VEHICLE_FIT_EXPERIMENTAL_CLASSES='vehicles/truck'):
            runtime = FitRuntime()
        cloud, index = box_scene()
        cloud = cloud.astype('<f4')
        meta = dict(metadata(), labelName='vehicles', vehicleType='truck', allowEstimated=True)
        meta['seed'] = dict(zip(('x', 'y', 'z'), cloud[index, :3].tolist()), pointIndex=index)
        result = runtime.fit_once(meta, cloud.tobytes())
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['autoFit']['class_key'], 'vehicles/truck')
        self.assertEqual(runtime.health()['enabledExperimentalClasses'], ['vehicles/truck'])


if __name__ == '__main__':
    unittest.main()
