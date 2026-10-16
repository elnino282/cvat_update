import unittest
from unittest.mock import patch

import numpy as np

from vehicle_fit.car_van import fit_car_van
from vehicle_fit.geometry import NoFit
from vehicle_fit.service import FitRuntime
from vehicle_fit.tests.test_large_vehicle import box_scene
from vehicle_fit.tests.test_service import metadata, fake_fit


class CarVanTests(unittest.TestCase):
    def test_van_uses_reliable_car_van_pool_without_claiming_subtype_truth(self):
        cloud, index = box_scene(length=4.7, width=2., height=1.8)
        result = fit_car_van(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(result['quality'], 'estimated')
        self.assertTrue(result['diagnostics']['prototypeKey'].startswith('car_van:'))
        self.assertEqual(result['diagnostics']['acceptanceGroundTruth'], 'insufficient-ground-truth')
        np.testing.assert_allclose(result['size'], [4.72, 2.02, 1.82], atol=.03)

    def test_isolated_sparse_return_estimates_with_ground_and_eligible_prior(self):
        cloud, index = box_scene()
        cloud = np.vstack((cloud[:index], [[10, 4, 1, 0]]))
        result = fit_car_van(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(result['quality'], 'estimated')
        self.assertTrue(result['diagnostics']['seedContained'])
        self.assertTrue(result['diagnostics']['yawUncertain'])
        self.assertGreater(result['size'][0], 4.)
        self.assertLess(result['size'][0], 5.2)

    def test_visible_side_remains_fixed_when_width_is_completed(self):
        cloud, index = box_scene(length=4.7, width=2., height=1.8, side_only=True)
        result = fit_car_van(cloud, cloud[index, :3], seed_index=index)
        self.assertAlmostEqual(result['center'][1] - result['size'][1] / 2, 3., places=4)
        self.assertEqual(result['diagnostics']['dimensionWeights'][1], 0.)

    def test_larger_detached_component_never_replaces_clicked_van(self):
        cloud, index = box_scene(length=4.7, width=2., height=1.8, y=2.)
        other, start = box_scene(length=8., width=2.7, height=3., y=7.)
        cloud = np.vstack((cloud, other[start:]))
        result = fit_car_van(cloud, cloud[index, :3], seed_index=index, mask=cloud[:, 2] > .1)
        self.assertLess(result['center'][1], 3.)
        self.assertLess(result['size'][0], 5.2)
        self.assertGreater(result['diagnostics']['detachedMaskPoints'], result['diagnostics']['selectedPoints'])

    def test_sparse_estimate_cannot_swallow_detached_neighbor(self):
        cloud, index = box_scene()
        cloud = np.vstack((cloud[:index], [[10, 4, 1, 0]],
                           [[10.6, 4, 1 + .1*i, 0] for i in range(4)]))
        with self.assertRaises(NoFit) as caught:
            fit_car_van(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(caught.exception.diagnostics['safetyGate'], 'competing-instance')

    def test_van_feature_flag_does_not_override_car_fitter(self):
        with patch.dict('os.environ', VEHICLE_FIT_BACKEND='geometry-only', VEHICLE_FIT_EXPERIMENTAL_CLASSES='vehicles/van'):
            runtime = FitRuntime()
        cloud = np.array([[1., 2., 3., .5]], dtype='<f4')
        with patch('vehicle_fit.service.fit_vehicle', side_effect=fake_fit) as car:
            result = runtime.fit_once(metadata(), cloud.tobytes())
        car.assert_called_once()
        self.assertEqual(result['autoFit']['class_key'], 'vehicles/car')
        self.assertFalse(result['autoFit']['experimental'])

    def test_van_route_records_van_semantics(self):
        with patch.dict('os.environ', VEHICLE_FIT_BACKEND='geometry-only', VEHICLE_FIT_EXPERIMENTAL_CLASSES='vehicles/van'):
            runtime = FitRuntime()
        cloud, index = box_scene(length=4.7, width=2., height=1.8)
        cloud = cloud.astype('<f4')
        meta = dict(metadata(), labelName='vehicles', vehicleType='van', allowEstimated=True)
        meta['seed'] = dict(zip(('x', 'y', 'z'), cloud[index, :3].tolist()), pointIndex=index)
        result = runtime.fit_once(meta, cloud.tobytes())
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['autoFit']['class_key'], 'vehicles/van')
        self.assertTrue(result['autoFit']['prototype_key'].startswith('car_van:'))


if __name__ == '__main__':
    unittest.main()
