import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from vehicle_fit.geometry import NoFit
from vehicle_fit.service import FitRuntime
from vehicle_fit.two_wheels import fit_two_wheels
from vehicle_fit.tests.test_large_vehicle import box_scene
from vehicle_fit.tests.test_service import metadata


def composite_scene():
    cloud, index = box_scene(length=2., width=.6, height=.9)
    xx, yy, zz = np.meshgrid(np.linspace(9.7, 10.3, 5), np.linspace(3.77, 4.23, 5),
                             [1.3, 1.48, 1.66, 1.84])
    rider = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel(), np.zeros(xx.size)))
    return np.vstack((cloud, rider)), index, len(cloud)


class TwoWheelsTests(unittest.TestCase):
    def test_sparse_no_rider_uses_eligible_pool_and_retains_one_public_class(self):
        cloud, index = box_scene()
        cloud = np.vstack((cloud[:index], [[10., 4., .7, 0]]))
        result = fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=False)
        self.assertEqual(result['quality'], 'estimated')
        self.assertIsNotNone(result['diagnostics']['prototypeKey'])
        self.assertFalse(result['diagnostics']['rideState'])
        self.assertNotIn('subtype', result)
        self.assertTrue(result['diagnostics']['yawUncertain'])

    def test_sparse_rider_pool_cannot_supply_missing_dimensions(self):
        cloud, index = box_scene()
        cloud = np.vstack((cloud[:index], [[10., 4., 1.5, 0]]))
        with self.assertRaises(NoFit) as caught:
            fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=True)
        self.assertEqual(caught.exception.reason, 'unsafe-fit')
        self.assertIn('prior-unavailable', caught.exception.diagnostics['diagnosticFlags'])

    def test_separated_vehicle_and_rider_are_merged_only_with_spatial_support(self):
        cloud, index, _ = composite_scene()
        result = fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=True,
                                mask=cloud[:, 2] > .1)
        self.assertEqual(result['quality'], 'estimated')
        self.assertEqual(result['diagnostics']['fitMode'], 'geometry')
        self.assertIsNone(result['diagnostics']['prototypeKey'])
        self.assertIsNone(result['diagnostics']['priorVersion'])
        self.assertEqual(result['diagnostics']['riderComponentsMerged'], 1)
        np.testing.assert_allclose(result['size'], [2.02, .62, 1.86], atol=.03)

    def test_click_on_rider_finds_unique_vehicle_below_without_changing_seed(self):
        cloud, _, index = composite_scene()
        result = fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=True,
                                mask=cloud[:, 2] > .1)
        self.assertTrue(result['diagnostics']['seedContained'])
        np.testing.assert_allclose(result['size'], [2.02, .62, 1.86], atol=.03)

    def test_adjacent_pedestrian_is_not_merged_into_rider_box(self):
        cloud, index, _ = composite_scene()
        xx, yy, zz = np.meshgrid([9.9, 10., 10.1], [4.9, 5., 5.1], np.arange(.2, 1.9, .15))
        pedestrian = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel(), np.zeros(xx.size)))
        cloud = np.vstack((cloud, pedestrian))
        result = fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=True,
                                mask=cloud[:, 2] > .1)
        self.assertLess(result['size'][1], .7)
        self.assertGreaterEqual(result['diagnostics']['detachedMaskPoints'], len(pedestrian))

    def test_touching_pedestrian_cannot_enlarge_lower_vehicle_to_join_rider(self):
        cloud, index, _ = composite_scene()
        xx, yy, zz = np.meshgrid([9.9, 10., 10.1], [4.5, 4.6, 4.7], np.arange(.2, 1.9, .15))
        pedestrian = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel(), np.zeros(xx.size)))
        cloud = np.vstack((cloud, pedestrian))
        with self.assertRaises(NoFit) as caught:
            fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=True,
                           mask=cloud[:, 2] > .1)
        self.assertEqual(caught.exception.reason, 'unsafe-fit')

    def test_only_vehicle_with_ride_state_true_has_no_fabricated_rider(self):
        cloud, index = box_scene(length=2., width=.6, height=.9)
        with self.assertRaises(NoFit) as caught:
            fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=True)
        self.assertIn('prior-unavailable', caught.exception.diagnostics['diagnosticFlags'])

    def test_cancellation_prevents_composite_completion(self):
        cloud, index, _ = composite_scene()
        with self.assertRaises(NoFit) as caught:
            fit_two_wheels(cloud, cloud[index, :3], seed_index=index, ride_state=True, cancelled=lambda: True)
        self.assertEqual(caught.exception.reason, 'cancelled')

    def test_service_requires_boolean_ride_state_and_persists_public_class_only(self):
        with patch.dict('os.environ', VEHICLE_FIT_BACKEND='geometry-only', VEHICLE_FIT_EXPERIMENTAL_CLASSES='two-wheels'):
            runtime = FitRuntime()
        meta = dict(metadata(), labelName='two-wheels', allowEstimated=True)
        self.assertEqual(runtime.fit_once(meta, b'')['reason'], 'invalid-coordinates')
        cloud, index = box_scene()
        cloud = np.vstack((cloud[:index], [[10., 4., .7, 0]])).astype('<f4')
        meta['rideState'] = False
        meta['seed'] = dict(zip(('x', 'y', 'z'), cloud[index, :3].tolist()), pointIndex=index)
        result = runtime.fit_once(meta, cloud.tobytes())
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['autoFit']['class_key'], 'two-wheels')
        self.assertTrue(result['autoFit']['experimental'])
        self.assertNotIn('subtype', result['autoFit'])
        persisted = json.dumps(result['autoFit'])
        self.assertNotIn('bicycle', persisted)
        self.assertNotIn('motorcycle', persisted)
        latent_key = result['diagnostics']['prototypeKey']
        expected_key = 'two-wheels:' + hashlib.sha256(latent_key.encode('utf-8')).hexdigest()[:12]
        self.assertEqual(result['autoFit']['prototype_key'], expected_key)
        self.assertIs(type(result['diagnostics']['rideState']), bool)
        self.assertFalse(result['diagnostics']['rideState'])


if __name__ == '__main__':
    unittest.main()
