import unittest
from unittest.mock import patch

import numpy as np

from vehicle_fit.bus import fit_bus
from vehicle_fit.geometry import NoFit
from vehicle_fit.service import FitRuntime
from vehicle_fit.shared import rotation
from vehicle_fit.tests.test_large_vehicle import box_scene
from vehicle_fit.tests.test_service import metadata


class BusTests(unittest.TestCase):
    def test_measured_rigid_bus_does_not_borrow_truck_or_sparse_bus_prior(self):
        cloud, index = box_scene(length=12., width=3., height=3.5)
        result = fit_bus(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(result['quality'], 'estimated')
        self.assertEqual(result['diagnostics']['fitMode'], 'geometry')
        self.assertIsNone(result['diagnostics']['prototypeKey'])
        self.assertIsNone(result['diagnostics']['priorVersion'])
        np.testing.assert_allclose(result['size'], [12.02, 3.02, 3.52], atol=.01)
        self.assertEqual(result['diagnostics']['dimensionWeights'], [1., 1., 1.])

    def test_single_observed_side_cannot_invent_hidden_bus_width(self):
        cloud, index = box_scene(length=12., width=3., height=3.5, side_only=True)
        with self.assertRaises(NoFit) as caught:
            fit_bus(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(caught.exception.reason, 'unsafe-fit')
        self.assertIn('prior-unavailable', caught.exception.diagnostics['diagnosticFlags'])

    def test_sparse_bus_has_no_made_up_dimensions(self):
        cloud, index = box_scene()
        cloud = np.vstack((cloud[:index], [[10, 4, 1, 0], [10.1, 4, 1.1, 0]]))
        with self.assertRaises(NoFit) as caught:
            fit_bus(cloud, cloud[index, :3], seed_index=index)
        self.assertEqual(caught.exception.diagnostics['safetyGate'], 'bus-observability')

    def test_missing_top_prevents_height_completion_from_insufficient_prior(self):
        cloud, index = box_scene(length=12., width=3., height=3.5)
        # One vertical pole is a max-Z return, not supported roof/height extent.
        cloud = cloud[cloud[:, 2] < 2.]
        cloud = np.vstack((cloud, [[10., 2.5, z, 0] for z in np.arange(2., 3.6, .2)]))
        with self.assertRaises(NoFit) as caught:
            fit_bus(cloud, cloud[index, :3], seed_index=index)
        self.assertIn('prior-unavailable', caught.exception.diagnostics['diagnosticFlags'])

    def test_joined_bendy_body_cannot_be_one_extreme_box(self):
        cloud, index = box_scene(length=12., width=3., height=3.5)
        right = (cloud[:, 0] > 10) & (cloud[:, 2] > .1)
        cloud[right, :2] = (cloud[right, :2] - [10., 4.]) @ rotation(np.pi / 4).T + [10., 4.]
        with self.assertRaises(NoFit) as caught:
            fit_bus(cloud, cloud[index, :3], seed_index=index, mask=cloud[:, 2] > .1)
        self.assertEqual(caught.exception.reason, 'unsafe-fit')

    def test_service_reports_experimental_measured_geometry_metadata(self):
        with patch.dict('os.environ', VEHICLE_FIT_BACKEND='geometry-only', VEHICLE_FIT_EXPERIMENTAL_CLASSES='vehicles/bus'):
            runtime = FitRuntime()
        cloud, index = box_scene(length=12., width=3., height=3.5)
        cloud = cloud.astype('<f4')
        meta = dict(metadata(), labelName='vehicles', vehicleType='bus', allowEstimated=True)
        meta['seed'] = dict(zip(('x', 'y', 'z'), cloud[index, :3].tolist()), pointIndex=index)
        result = runtime.fit_once(meta, cloud.tobytes())
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['autoFit']['experimental'])
        self.assertEqual(result['autoFit']['fit_mode'], 'geometry')
        self.assertIsNone(result['autoFit']['prior_version'])


if __name__ == '__main__':
    unittest.main()
