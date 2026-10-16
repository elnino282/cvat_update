import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vehicle_fit.evaluation.nuscenes_replay import (
    compare, observed_mask, replay_case, semantic_rows, verified_config,
    verified_frame, verified_seed, verify_coordinates,
)


class NuScenesReplayTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'coordinate_frame': 'LIDAR_TOP', 'dimension_order': 'length,width,height',
                         'rotation_convention': 'Three.js Euler XYZ (Rx @ Ry @ Rz)'}
        self.frame = {'lidar_to_global': np.eye(4).tolist()}
        self.case = {'case_id': 'target', 'sample_token': 'frame', 'target_label': 'vehicles',
                     'vehicles_type': 'car', 'bins': {}, 'seeds': [{'pointIndex': 2, 'point': [100., 3., 4.]}]}
        self.item = {'case_id': 'target', 'sample_token': 'frame', 'route': 'vehicles/car', 'outcome': 'no-fit',
                     'request': {'requestId': 'request', 'frame': 7, 'dataVersion': 'manifest', 'cloud_format': 'bin',
                                 'labelName': 'vehicles', 'vehicleType': 'car', 'allowEstimated': True,
                                 'seed': {'pointIndex': 2, 'x': 100., 'y': 3., 'z': 4.}},
                     'response': {'requestId': 'request', 'frame': 7, 'status': 'no-fit',
                                  'diagnostics': {'cloudSHA256': 'cloud', 'maskIndices': [2, 0]}}}
        self.cloud = np.array([[0., 0., 0., 1.], [100., 3., 4., 1.]], dtype='<f4')
        self.indices = np.array([0, 2])

    def test_rejects_wrong_frame_nan_singular_and_reflected_transform(self):
        verify_coordinates(self.manifest, self.frame)
        for matrix in (np.full((4, 4), np.nan), np.zeros((4, 4)), np.diag([-1., 1., 1., 1.])):
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, 'coordinate transform'):
                    verify_coordinates(self.manifest, {'lidar_to_global': matrix.tolist()})
        with self.assertRaisesRegex(ValueError, 'coordinate frame'):
            verify_coordinates({**self.manifest, 'coordinate_frame': 'ego'}, self.frame)

    def test_cloud_hash_verification_preserves_raw_index_after_nonfinite_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cloud.bin'
            raw = np.array([[0., 0., 0., 1., 0.], [np.nan, 1., 2., 1., 0.], [100., 3., 4., 1., 0.]], dtype='<f4')
            path.write_bytes(raw.tobytes())
            frame = {**self.frame, 'point_count': 3, 'point_dimensions': 5, 'lidar_path': path.name,
                     'sha256': hashlib.sha256(raw.tobytes()).hexdigest()}
            manifest = {**self.manifest, 'frames': {'frame': frame}}
            cloud, indices, service_hash = verified_frame(manifest, 'frame', directory)
            np.testing.assert_array_equal(cloud, self.cloud)
            np.testing.assert_array_equal(indices, self.indices)
            self.assertEqual(service_hash, hashlib.sha256(raw[:, :4].copy().tobytes()).hexdigest())
            path.write_bytes(np.zeros_like(raw).tobytes())
            with self.assertRaisesRegex(ValueError, 'Frozen cloud changed'):
                verified_frame(manifest, 'frame', directory)

    def test_primary_seed_and_request_response_identity_are_checked(self):
        verified_seed(self.case, self.item, self.cloud, self.indices, 'manifest')
        for key, value in [('x', 100.0001), ('pointIndex', 1), ('z', float('nan'))]:
            item = copy.deepcopy(self.item)
            item['request']['seed'][key] = value
            with self.assertRaisesRegex(ValueError, 'seed mismatch'):
                verified_seed(self.case, item, self.cloud, self.indices, 'manifest')
        item = copy.deepcopy(self.item)
        item['response']['frame'] = 8
        with self.assertRaisesRegex(ValueError, 'response request/frame identity'):
            verified_seed(self.case, item, self.cloud, self.indices, 'manifest')

    def test_observed_mask_is_validated_against_cloud_not_ground_truth(self):
        mask, digest = observed_mask(self.item, self.indices, 'cloud')
        np.testing.assert_array_equal(mask, [True, True])
        reverse = copy.deepcopy(self.item)
        reverse['response']['diagnostics']['maskIndices'].reverse()
        self.assertEqual(observed_mask(reverse, self.indices, 'cloud')[1], digest)
        with self.assertRaisesRegex(ValueError, 'cloud hash mismatch'):
            observed_mask(self.item, self.indices, 'other cloud')
        for bad_indices in ([1], [-1], [2.0], [True], [2, 2]):
            item = copy.deepcopy(self.item)
            item['response']['diagnostics']['maskIndices'] = bad_indices
            with self.assertRaisesRegex(ValueError, 'Invalid observed mask'):
                observed_mask(item, self.indices, 'cloud')

    def test_missing_mask_stays_explicit_and_never_calls_fitter(self):
        self.item['response']['diagnostics'].pop('maskIndices')
        self.assertEqual(observed_mask(self.item, self.indices, 'cloud'), (None, None))
        with patch('vehicle_fit.evaluation.nuscenes_replay.geometry.fit_vehicle') as fitter:
            result = replay_case(self.case, self.item, self.frame, self.cloud, np.zeros(3), None, None, {}, .05)
        fitter.assert_not_called()
        self.assertEqual(result['outcome'], 'missing-mask')
        self.assertNotIn('metrics', result)

    def test_production_source_and_configuration_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.py'
            path.write_text('frozen source')
            cloud = Path(directory) / 'cloud.py'
            cloud.write_text('frozen cloud decoder')
            identity = {'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, cloud)},
                        'health': {'vehiclePriorsMetres': {'length': 3.7, 'width': 1.61, 'height': 1.49},
                                   'maxSurfaceScore': .1, 'adaptiveSizeRefinement': False},
                        'timeout_ms': 1700, 'seed_tolerance_m': .05}
            config = {'health': identity['health'], 'timeout_ms': 1700, 'allowEstimated': True}
            identity['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            self.assertFalse(verified_config(identity, directory)['adaptive_refinement'])
            identity['health']['maxSurfaceScore'] = .2
            with self.assertRaisesRegex(ValueError, 'config hash mismatch'):
                verified_config(identity, directory)
            path.write_text('changed source')
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                verified_config(identity, directory)

    def test_repeat_digest_excludes_only_runtime_and_requires_same_cohort(self):
        row = {'case_id': 'one', 'outcome': 'estimated', 'response': {'size': [1, 2, 3]}, 'geometry_seconds': 1.}
        later = {**row, 'geometry_seconds': 2.}
        self.assertEqual(semantic_rows([row]), semantic_rows([later]))
        later['response'] = {'size': [1, 2, 4]}
        self.assertEqual(compare([later], [row])['changed_geometry'], ['one'])
        self.assertNotEqual(semantic_rows([row]), semantic_rows([later]))
        with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
            compare([row], [])


if __name__ == '__main__':
    unittest.main()
