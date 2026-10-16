"""Focused safety/lifecycle regression for the experimental class routes."""
import os
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np

from vehicle_fit.geometry import NoFit
from vehicle_fit.pedestrian import fit_pedestrian
from vehicle_fit.priors import REGISTRY
from vehicle_fit.service import FitRuntime
from vehicle_fit.shared import validate_seed
from vehicle_fit.tests.test_geometry import road_scene
from vehicle_fit.tests.test_service import metadata


def pedestrian_scene(count=10, offset=0.):
    base = road_scene()
    ground = base[base[:, 2] == 0]
    support = np.array([[10., offset + .05 * (i % 2), .3 + .1 * i, 0.] for i in range(count)])
    return np.vstack((ground, support)), len(ground)


class SharedEvidenceTests(unittest.TestCase):
    def test_prior_filters_independent_count_and_preserves_public_taxonomy(self):
        self.assertEqual(len(REGISTRY.pool('pedestrian')), 3)
        self.assertEqual(len(REGISTRY.pool('vehicles/truck')), 2)
        self.assertEqual(len(REGISTRY.pool('vehicles/van')), 2)
        self.assertFalse(REGISTRY.pool('vehicles/bus'))
        self.assertFalse(REGISTRY.pool('two-wheels', True))
        self.assertEqual(len(REGISTRY.pool('two-wheels', False)), 2)
        self.assertTrue(all(r['split'] == 'dev' for r in REGISTRY.records))

    def test_seed_index_must_match_measured_coordinate(self):
        cloud = np.array([[1., 2., 3.], [4., 5., 6.]])
        for index in (-1, 2, True, None, 1):
            with self.subTest(index=index), self.assertRaises(NoFit) as caught:
                validate_seed(cloud, dict(x=1, y=2, z=3, pointIndex=index))
            self.assertEqual(caught.exception.reason, 'invalid-coordinates')
        self.assertEqual(validate_seed(cloud, dict(x=1, y=2, z=3, pointIndex=0))[1], 0)

    def test_nonfinite_seed_is_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(NoFit):
                validate_seed(np.array([[1., 2., 3.]]), dict(x=value, y=2, z=3, pointIndex=0))


class PedestrianTests(unittest.TestCase):
    def test_sparse_5_10_15_returns_get_ground_anchored_estimate(self):
        for count in (5, 10, 15):
            with self.subTest(count=count):
                cloud, index = pedestrian_scene(count)
                result = fit_pedestrian(cloud, cloud[index, :3], seed_index=index)
                self.assertEqual(result['quality'], 'estimated')
                self.assertAlmostEqual(result['center'][2] - result['size'][2] / 2, 0)
                self.assertTrue(result['diagnostics']['yawUncertain'])
                self.assertTrue(result['diagnostics']['seedContained'])
                self.assertEqual(result['diagnostics']['selectedPoints'], count)

    def test_single_supported_return_can_estimate_with_prior_and_ground(self):
        cloud, index = pedestrian_scene(1)
        result = fit_pedestrian(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(result['quality'], 'estimated')
        self.assertEqual(result['diagnostics']['point_count'], 1)

    def test_larger_detached_neighbor_cannot_steal_seed_component(self):
        cloud, index = pedestrian_scene(5)
        neighbor = np.array([[10., 1.2 + .02*(i % 8), .3 + .1*(i % 15), 0.] for i in range(100)])
        cloud = np.vstack((cloud, neighbor))
        mask = cloud[:, 2] > .1
        result = fit_pedestrian(cloud, cloud[index, :3], seed_index=index, mask=mask)
        self.assertLess(abs(result['center'][1]), .3)
        self.assertEqual(result['diagnostics']['selectedPoints'], 5)
        self.assertEqual(result['diagnostics']['detachedMaskPoints'], 100)

    def test_estimate_cannot_swallow_competing_vertical_component(self):
        cloud, index = pedestrian_scene(1)
        neighbor = np.array([[10.05, .05, 1.2 + .1*i, 0.] for i in range(5)])
        cloud = np.vstack((cloud, neighbor))
        with self.assertRaises(NoFit) as caught:
            fit_pedestrian(cloud, cloud[index, :3], seed_index=index, mask=cloud[:, 2] > .1)
        self.assertEqual(caught.exception.reason, 'unsafe-fit')
        self.assertEqual(caught.exception.diagnostics['safetyGate'], 'competing-instance')

    def test_empty_mask_falls_back_to_local_seed_support(self):
        cloud, index = pedestrian_scene()
        result = fit_pedestrian(cloud, cloud[index, :3], seed_index=index, mask=np.zeros(len(cloud), bool))
        self.assertEqual(result['diagnostics']['maskSource'], 'seed-connected')
        self.assertIn('empty-mask', result['diagnostics']['diagnosticFlags'])

    def test_missing_ground_cannot_invent_anchor(self):
        cloud, _ = pedestrian_scene()
        cloud = cloud[cloud[:, 2] > .1]
        with self.assertRaises(NoFit) as caught:
            fit_pedestrian(cloud, cloud[0, :3], seed_index=0)
        self.assertIn('ground-unavailable', caught.exception.diagnostics['diagnosticFlags'])

    def test_ground_return_is_not_object_support(self):
        cloud, _ = pedestrian_scene()
        with self.assertRaises(NoFit) as caught:
            fit_pedestrian(cloud, cloud[0, :3], seed_index=0)
        self.assertEqual(caught.exception.reason, 'no-seed-support')

    def test_cancelled_fit_stops(self):
        cloud, index = pedestrian_scene()
        with self.assertRaises(NoFit) as caught:
            fit_pedestrian(cloud, cloud[index, :3], seed_index=index, cancelled=lambda: True)
        self.assertEqual(caught.exception.reason, 'cancelled')


class MulticlassRuntimeTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, VEHICLE_FIT_BACKEND='geometry-only', VEHICLE_FIT_EXPERIMENTAL_CLASSES='pedestrian'):
            self.runtime = FitRuntime()
        self.cloud, index = pedestrian_scene()
        self.cloud = self.cloud.astype('<f4')
        self.meta = dict(metadata(), labelName='pedestrian', allowEstimated=True)
        self.meta['seed'] = dict(zip(('x', 'y', 'z'), self.cloud[index, :3].tolist()), pointIndex=index)

    def test_disabled_routes_refuse_before_decoding(self):
        self.runtime.experimental_classes.clear()
        with patch('vehicle_fit.service.decode_cloud', side_effect=AssertionError('must not decode')):
            result = self.runtime.fit_once(self.meta, b'')
        self.assertEqual(result['reason'], 'unsupported-type')

    def test_runtime_preserves_estimated_metadata_and_idempotency(self):
        first = self.runtime.fit_once(self.meta, self.cloud.tobytes())
        second = self.runtime.fit_once(self.meta, self.cloud.tobytes())
        self.assertEqual(first, second)
        self.assertEqual(first['status'], 'ok')
        self.assertEqual(first['autoFit']['class_key'], 'pedestrian')
        self.assertTrue(first['autoFit']['experimental'])
        self.assertEqual(first['autoFit']['prior_version'], REGISTRY.version)
        self.assertEqual(first['autoFit']['request_id'], self.meta['requestId'])

    def test_wrong_coordinate_rejected_before_snap(self):
        self.runtime.backend_name = 'snap'
        self.runtime.backend = Mock()
        self.meta['seed']['x'] = 1000
        result = self.runtime.fit_once(dict(self.meta, timeout_ms=8000), self.cloud.tobytes())
        self.assertEqual(result['reason'], 'invalid-coordinates')
        self.runtime.backend.segment.assert_not_called()

    def test_finite_cloud_index_not_original_file_index(self):
        dirty = np.vstack(([float('nan'), 0, 0, 0], self.cloud)).astype('<f4')
        result = self.runtime.fit_once(self.meta, dirty.tobytes())
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['diagnostics']['maskIndices'][0], self.meta['seed']['pointIndex'] + 1)

    def test_car_geometry_call_is_unchanged_and_receives_no_new_prior(self):
        from vehicle_fit.tests.test_service import fake_fit
        meta = metadata(str(uuid.uuid4()))
        cloud = np.array([[1, 2, 3, .5]], dtype='<f4')
        with patch('vehicle_fit.service.fit_vehicle', side_effect=fake_fit) as fitter:
            result = self.runtime.fit_once(meta, cloud.tobytes())
        self.assertEqual(fitter.call_count, 1)
        self.assertEqual(fitter.call_args.kwargs['priors'].length, 3.70)
        self.assertFalse(result['autoFit']['experimental'])
        self.assertIsNone(result['autoFit']['prior_version'])
