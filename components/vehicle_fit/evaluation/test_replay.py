import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np

from vehicle_fit.evaluation.interactive import run
from vehicle_fit.evaluation.metrics import Cuboid
from vehicle_fit.evaluation.replay import summarize, verified_frame


class ReplayTests(unittest.TestCase):
    def test_calibration_hash_is_checked_before_geometry_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cloud = root/'cloud.bin'
            cloud.write_bytes(np.ones((1, 4), dtype='<f4').tobytes())
            calib = root/'calib.txt'
            calib.write_text('frozen calibration')
            manifest = {'frames': {'1': {'assets': {'velodyne': str(cloud), 'calib': str(calib)},
                                        'sha256': {key: hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for key, p in [('velodyne', cloud), ('calib', calib)]}}}}
            verified_frame(manifest, '1')
            calib.write_text('changed calibration')
            with self.assertRaisesRegex(ValueError, 'calib'):
                verified_frame(manifest, '1')

    def test_transport_failure_is_flushed_and_distinct_from_algorithm_no_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'frames': {'1': {'sha256': {'velodyne': 'frozen'}}},
                        'cases': [{'frame': '1', 'case_id': 'one', 'split': 'tune', 'adequate': True,
                                   'gt_point_count': 10, 'seeds': [{'point': [0, 0, 0], 'pointIndex': 0}]}]}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            args = SimpleNamespace(manifest=str(path), split='tune', endpoint='http://unused',
                                   output=str(root/'out'), tracklet_zip=None)
            with patch('vehicle_fit.evaluation.interactive.verified_frame', return_value=np.ones((1, 4))), \
                 patch('vehicle_fit.evaluation.interactive.truth_boxes', return_value=([Cuboid(np.zeros(3), np.ones(3), np.eye(3))], [0])), \
                 patch('vehicle_fit.evaluation.interactive.http', side_effect=urllib.error.URLError('offline')):
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
            rows = [json.loads(line) for line in (root/'out/cases.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['status'], 'transport-error')
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertTrue(summary['completed'])
            self.assertEqual(summary['all']['transport_error'], 1)
            self.assertEqual(summary['all']['no_fit'], 0)

    def test_missing_masks_and_unseedable_cases_are_explicit(self):
        summary = summarize([{'status': 'missing-mask'}, {'status': 'no-fit'},
                             {'status': 'unseedable'}, {'status': 'transport-error'}])
        self.assertEqual(summary['cases'], 4)
        for key in ('missing_mask', 'no_fit', 'unseedable', 'transport_error'):
            self.assertEqual(summary[key], 1)
