"""PRD 1.1 PR0: freeze real behavior without changing the production fitter.

Expected failures are executable PR1/PR2 requirements, not accepted behavior.
unittest fails on unexpected success, so remove a marker when the gap is fixed.
SNAP is mocked only in service ordering tests; geometry uses real code.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from vehicle_fit.evaluation.metrics import Cuboid, calibration, euler_xyz_matrix
from vehicle_fit.geometry import (
    NoFit, SEED_TOLERANCE, _prepare_evidence, _validate_geometry, fit_vehicle,
)
from vehicle_fit.service import FitRuntime
from vehicle_fit.tests.test_geometry import road_scene
from vehicle_fit.tests.test_service import metadata


def competing_scene():
    """Neighbor has twice the returns and 20% more length than the target."""
    scene = road_scene()
    car = scene[scene[:, 2] > .1]
    target = car[::2].copy()
    target[:, 1] -= 1.5
    neighbor = car.copy()
    neighbor[:, 1] += 1.5
    neighbor[:, 0] = 10 + (neighbor[:, 0] - 10) * 1.2
    cloud = np.vstack((scene[scene[:, 2] == 0], target, neighbor))
    seed = target[np.linalg.norm(target[:, :3] - [10.05, -.6, .75], axis=1).argmin(), :3]
    return cloud, seed


class PR0GeometryCharacterization(unittest.TestCase):
    def assert_contains(self, result, seed):
        box = Cuboid(np.asarray(result['center']), np.asarray(result['size']),
                     euler_xyz_matrix(result['rotation']))
        self.assertTrue(box.contains(np.asarray([seed]), tolerance=SEED_TOLERANCE)[0])

    def test_car_geometry_snapshot_and_containment(self):
        seed = np.array([10.05, .9, .75])
        result = fit_vehicle(road_scene(), seed)
        # Existing algorithm/prior snapshot; this is not a nuScenes accuracy claim.
        np.testing.assert_allclose(result['center'], [10, 0, .8], atol=1e-10)
        np.testing.assert_allclose(result['size'], [4.3, 1.73, 1.6], atol=1e-10)
        np.testing.assert_allclose(result['rotation'], [0, 0, 0], atol=1e-10)
        self.assertEqual(result['quality'], 'confident')
        self.assert_contains(result, seed)

    def test_larger_denser_neighbor_does_not_win_seed_connected_fit(self):
        cloud, seed = competing_scene()
        result = fit_vehicle(cloud, seed, allow_estimated=True)
        self.assertLess(result['center'][1], 0)
        self.assertLess(abs(result['center'][1] + 1.5), .1)
        retained = cloud[result['diagnostics']['mask_indices']]
        self.assertTrue(np.all(retained[:, 1] < 0))
        self.assert_contains(result, seed)

    def test_larger_neighbor_in_snap_mask_is_currently_refused_safely(self):
        cloud, seed = competing_scene()
        with self.assertRaises(NoFit) as caught:
            fit_vehicle(cloud, seed, mask=cloud[:, 2] > .1, allow_estimated=True)
        self.assertEqual(caught.exception.reason, 'insufficient-evidence')

    @unittest.expectedFailure
    def test_requirement_competing_mask_attempts_seed_component_estimate(self):
        """PR1 gap: full SNAP mask retained; valid target is not selected for fallback."""
        cloud, seed = competing_scene()
        result = fit_vehicle(cloud, seed, mask=cloud[:, 2] > .1, allow_estimated=True)
        self.assertLess(abs(result['center'][1] + 1.5), .5)
        self.assert_contains(result, seed)

    def test_seed_beyond_99th_quantile_is_in_final_box(self):
        # Nine connected bumper returns are below the 1% quantile trim budget.
        tail = np.array([[12.27, y, z, 0] for y in (-.06, 0, .06) for z in (.69, .75, .81)])
        cloud = np.vstack((road_scene(), tail))
        seed = tail[4, :3]
        mask = cloud[:, 2] > .1
        self.assertGreater(seed[0], np.quantile(cloud[mask, 0], .99) + SEED_TOLERANCE)
        evidence = _prepare_evidence(cloud, seed, mask, lambda: False)
        trimmed = {'center': [10, 0, .8], 'size': [4.3, 1.8, 1.6], 'rotation': [0, 0, 0]}
        with self.assertRaises(NoFit) as caught:
            _validate_geometry(trimmed, evidence, lambda: False)
        self.assertEqual(caught.exception.diagnostics['safetyGate'], 'seed-outside')
        result = fit_vehicle(cloud, seed, mask=mask, allow_estimated=True)
        self.assert_contains(result, seed)

    def test_rigid_xy_transform_preserves_car_fit_and_seed(self):
        cloud = road_scene()
        seed = np.array([10.05, .9, .75])
        original = fit_vehicle(cloud, seed)
        # Quarter turn preserves the current discrete yaw proposal grid exactly.
        rotation = euler_xyz_matrix([0, 0, np.pi / 2])
        translation = np.array([-12, 7, 0])
        cloud[:, :3] = cloud[:, :3] @ rotation.T + translation
        transformed_seed = seed @ rotation.T + translation
        transformed = fit_vehicle(cloud, transformed_seed, allow_estimated=True)
        np.testing.assert_allclose(transformed['center'],
                                   np.asarray(original['center']) @ rotation.T + translation, atol=1e-8)
        np.testing.assert_allclose(transformed['size'], original['size'], atol=1e-8)
        self.assert_contains(transformed, transformed_seed)


class PR0CoordinateAndRoutingCharacterization(unittest.TestCase):
    def setUp(self):
        with patch.dict('os.environ', {'VEHICLE_FIT_BACKEND': 'geometry-only'}):
            self.runtime = FitRuntime()
        self.cloud = road_scene().astype('<f4')
        self.raw = self.cloud.tobytes()
        self.meta = metadata()
        self.meta.update(timeout_ms=8000, allowEstimated=True)
        self.meta['seed'] = dict(x=10.05, y=.9, z=.75,
                                pointIndex=int(np.argmin(np.linalg.norm(self.cloud[:, :3] - [10.05, .9, .75], axis=1))))
        # No real model/thread initialization: deterministic SNAP contract fixture.
        self.runtime.backend_name = 'snap'
        self.runtime.backend = Mock()
        self.runtime.backend.segment.return_value = (self.cloud[:, 2] > .1, {'backend': 'mocked-snap'})

    def test_nonfinite_seed_never_calls_snap_or_fitter(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                self.meta['requestId'] = str(value)
                self.meta['seed']['x'] = value
                with patch('vehicle_fit.service.fit_vehicle') as fitter:
                    result = self.runtime.fit_once(self.meta, self.raw)
                self.assertEqual(result['reason'], 'invalid-coordinates')
                self.runtime.backend.segment.assert_not_called()
                fitter.assert_not_called()

    def test_foreign_prepared_frame_never_calls_snap(self):
        prepared = self.runtime.prepare(self.meta, self.raw)
        self.meta.update(frameToken=prepared['frameToken'], frame=self.meta['frame'] + 1)
        result = self.runtime.fit_once(self.meta)
        self.assertEqual(result['reason'], 'invalid-coordinates')
        self.assertIn('scope mismatch', result['diagnostics']['detail'])
        self.runtime.backend.segment.assert_not_called()

    def test_wrong_coordinate_space_is_rejected_before_segmentation(self):
        self.meta['seed'].update(x=1000, y=1000, z=1000)
        result = self.runtime.fit_once(self.meta, self.raw)
        self.assertEqual(result['reason'], 'invalid-coordinates')
        self.runtime.backend.segment.assert_not_called()

    def test_requirement_wrong_coordinate_space_rejected_before_snap(self):
        """PR1 closes the PR0 pre-segmentation coordinate validation gap."""
        self.meta['seed'].update(x=1000, y=1000, z=1000)
        self.runtime.fit_once(self.meta, self.raw)
        self.runtime.backend.segment.assert_not_called()

    def test_noncar_vehicles_routes_refuse_before_snap(self):
        for kind in ('truck', 'bus', 'van', 'other'):
            with self.subTest(kind=kind):
                result = self.runtime.fit_once(dict(self.meta, labelName='vehicles', vehicleType=kind), b'')
                self.assertEqual(result['reason'], 'unsupported-type')
                self.runtime.backend.segment.assert_not_called()

    def test_sparse_pedestrian_route_is_gated_until_explicitly_enabled(self):
        for count in (5, 10, 15):
            with self.subTest(count=count):
                support = np.array([[10, .05 * (i % 2), .3 + .1 * i, 0] for i in range(count)], dtype='<f4')
                raw = np.vstack((self.cloud[self.cloud[:, 2] == 0], support)).tobytes()
                result = self.runtime.fit_once(dict(self.meta, labelName='pedestrian'), raw)
                self.assertEqual(result['reason'], 'unsupported-type')
                self.runtime.backend.segment.assert_not_called()

    def test_requirement_sparse_pedestrian_has_class_fit_attempt(self):
        """PR2 enables a separate fitter with a supported dev prototype."""
        self.runtime.experimental_classes.add('pedestrian')
        support = np.array([[10, .05 * (i % 2), .3 + .1 * i, 0] for i in range(10)], dtype='<f4')
        cloud = np.vstack((self.cloud[self.cloud[:, 2] == 0], support))
        self.meta.update(labelName='pedestrian', seed=dict(x=10, y=0, z=.3, pointIndex=len(cloud) - 10))
        self.runtime.backend.segment.return_value = (cloud[:, 2] > .1, {})
        result = self.runtime.fit_once(self.meta, cloud.tobytes())
        self.runtime.backend.segment.assert_called_once()
        self.assertEqual(result['status'], 'ok')
        self.assertIn(result['quality'], ('estimated', 'confident'))


class PR0ExistingCalibrationParserCharacterization(unittest.TestCase):
    """Legacy evaluator parser only. Production has no calibration schema yet."""
    def read_calibration(self, rotation):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'calibration.txt'
            path.write_text('R0_rect: ' + ' '.join(map(str, rotation.flat)) + '\n'
                            'Tr_velo_to_cam: 1 0 0 1 0 1 0 2 0 0 1 3\n'
                            'P2: 1 0 0 0 0 1 0 0 0 0 1 0\n')
            return calibration(path)

    def test_valid_transform_round_trip(self):
        transform, _ = self.read_calibration(euler_xyz_matrix([.1, -.2, .3]))
        point = np.array([4., 5., 6., 1.])
        np.testing.assert_allclose(np.linalg.inv(transform) @ (transform @ point), point, atol=1e-12)

    def test_current_parser_accepts_nonfinite_and_singular_calibration(self):
        nonfinite = np.eye(3)
        nonfinite[0, 0] = np.nan
        transform, _ = self.read_calibration(nonfinite)
        self.assertFalse(np.isfinite(transform).all())
        transform, _ = self.read_calibration(np.zeros((3, 3)))
        self.assertEqual(np.linalg.det(transform), 0)

    @unittest.expectedFailure
    def test_requirement_nan_calibration_rejected_at_read_boundary(self):
        """PR1 gap: legacy KITTI parser returns NaN matrix without validation."""
        rotation = np.eye(3)
        rotation[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.read_calibration(rotation)

    @unittest.expectedFailure
    def test_requirement_singular_calibration_rejected_at_read_boundary(self):
        """PR1 gap: legacy KITTI parser returns singular matrix without validation."""
        with self.assertRaises(ValueError):
            self.read_calibration(np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()
