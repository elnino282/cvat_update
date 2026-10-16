import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vehicle_fit.evaluation.drive_0005 import compare_routes, correct_instance, dataset_manifest, load_cloud, run


def pcd(points):
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {}\nHEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA binary\n"
    ).format(len(points), len(points)).encode()
    return header + np.asarray(points, dtype="<f4").tobytes()


class Drive0005ManifestTests(unittest.TestCase):
    def test_empty_corpus_cannot_pass_execution_gate(self):
        manifest = {'corpus': 'empty', 'annotation_sha256': 'a', 'cloud_archive_sha256': 'b',
                    'tracklet_sha256': 'c', 'frames': {}, 'cases': []}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(dataset='unused', output=str(Path(directory) / 'report'),
                                   endpoint='http://unused', limit=0, paired_legacy=9)
            with patch('vehicle_fit.evaluation.drive_0005.dataset_manifest', return_value=manifest), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result['gates']['execution'], 'BLOCKED')

    def test_route_parity_does_not_pass_matching_outages_or_different_provider(self):
        for reason in ('timeout', 'unavailable', 'cancelled', 'frame-expired'):
            response = {'status': 'no-fit', 'reason': reason, 'provider': 'snap'}
            self.assertIsNone(compare_routes(response, response))
        response = {'status': 'ok', 'provider': 'snap', 'quality': 'estimated',
                    'center': [0, 0, 0], 'size': [4, 2, 1.5], 'rotation': [0, 0, 0]}
        self.assertTrue(compare_routes(response, response))
        self.assertFalse(compare_routes(response, dict(response, provider='geometry-only')))
        self.assertFalse(compare_routes(response, dict(response, quality='confident')))

    def test_zero_overlap_and_ties_are_not_correct_instances(self):
        self.assertFalse(correct_instance([0, 0], 0))
        self.assertFalse(correct_instance([.8, .8], 0))
        self.assertFalse(correct_instance([.2, .8], 0))
        self.assertTrue(correct_instance([.8, .2], 0))

    def test_manifest_freezes_pcd_and_visible_vehicle_gt_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations/annotations").mkdir(parents=True)
            (root / "metadata").mkdir()
            cloud_path = root / "cvat_data_3d.zip"
            points = np.asarray([
                [-0.5, 0, 0, 0.1], [0.5, 0, 0, 0.2],
                [4.5, 0, 0, 0.3], [5.5, 0, 0, 0.4],
            ], dtype="<f4")
            with zipfile.ZipFile(cloud_path, "w") as archive:
                archive.writestr("pointcloud/0000000000.pcd", pcd(points))
            source_annotations = {
                "categories": {"label": {"labels": [{"name": "Car"}, {"name": "Van"}]}},
                "items": [{
                    "id": "pointcloud/0000000000",
                    "annotations": [
                        {"id": 1, "type": "cuboid_3d", "label_id": 0,
                         "position": [0, 0, 0], "scale": [2, 2, 2], "rotation": [0, 0, 0],
                         "attributes": {"track_id": 7, "kitti_track_id": "4", "outside": False}},
                        {"id": 2, "type": "cuboid_3d", "label_id": 1,
                         "position": [5, 0, 0], "scale": [2, 2, 2], "rotation": [0, 0, 0],
                         "attributes": {"track_id": 8, "kitti_track_id": "5", "outside": False}},
                        {"id": 3, "type": "cuboid_3d", "label_id": 0,
                         "position": [10, 0, 0], "scale": [2, 2, 2], "rotation": [0, 0, 0],
                         "attributes": {"track_id": 9, "kitti_track_id": "6", "outside": True}},
                    ],
                }],
            }
            annotation_path = root / "annotations/annotations/default.json"
            annotation_path.write_text(json.dumps(source_annotations))
            tracklet_path = root / "metadata/tracklet_labels.xml"
            tracklet_path.write_text("<tracklets />")
            before = {path: path.read_bytes() for path in (cloud_path, annotation_path, tracklet_path)}

            manifest = dataset_manifest(root)

            self.assertEqual(manifest["visible_cases_by_class"], {"Car": 1, "Van": 1})
            self.assertEqual(len(manifest["cases"]), 2)
            car = next(case for case in manifest["cases"] if case["class"] == "Car")
            self.assertEqual(car["track_id"], 7)
            self.assertEqual(car["gt_point_count"], 2)
            self.assertEqual([seed["pointIndex"] for seed in car["seeds"]], [0, 1])
            payload, cloud = load_cloud(manifest, "0000000000")
            self.assertEqual(cloud.shape, (4, 4))
            self.assertEqual(len(payload), len(pcd(points)))
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_duplicate_annotation_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations/annotations").mkdir(parents=True)
            (root / "metadata").mkdir()
            with zipfile.ZipFile(root / "cvat_data_3d.zip", "w") as archive:
                archive.writestr("pointcloud/0000000000.pcd", pcd([[0, 0, 0, 0]]))
            annotation = {
                "categories": {"label": {"labels": [{"name": "Car"}]}},
                "items": [{"id": "pointcloud/0000000000", "annotations": [
                    {"id": 1, "type": "cuboid_3d", "label_id": 0, "position": [0, 0, 0],
                     "scale": [2, 2, 2], "rotation": [0, 0, 0], "attributes": {"outside": False}},
                    {"id": 1, "type": "cuboid_3d", "label_id": 0, "position": [0, 0, 0],
                     "scale": [2, 2, 2], "rotation": [0, 0, 0], "attributes": {"outside": False}},
                ]}],
            }
            (root / "annotations/annotations/default.json").write_text(json.dumps(annotation))
            (root / "metadata/tracklet_labels.xml").write_text("<tracklets />")
            with self.assertRaisesRegex(ValueError, "Duplicate annotation ID"):
                dataset_manifest(root)

    def test_unavailable_snap_preparation_is_blocked_and_never_a_quality_pass(self):
        case = {"case_id": "case-1", "frame": "0000000000", "class": "Car",
                "track_id": 1, "kitti_track_id": "1", "gt_point_count": 4,
                "seeds": [{"point": [0, 0, 0], "pointIndex": 0}],
                "gt_center": [0, 0, 0], "gt_size": [2, 2, 2], "gt_rotation": [0, 0, 0]}
        manifest = {"corpus": "fixture", "annotation_sha256": "a", "cloud_archive_sha256": "b",
                    "tracklet_sha256": "c", "frames": {case["frame"]: {"sha256": "d"}},
                    "cases": [case]}
        args = SimpleNamespace(dataset="unused", output="unused", endpoint="http://unused",
                               limit=0, paired_legacy=1)
        with tempfile.TemporaryDirectory() as directory:
            args.output = str(Path(directory) / "report")
            with patch("vehicle_fit.evaluation.drive_0005.dataset_manifest", return_value=manifest), \
                 patch("vehicle_fit.evaluation.drive_0005.load_cloud", return_value=(b"pcd", np.zeros((1, 4)))), \
                 patch("vehicle_fit.evaluation.drive_0005.prepare",
                       return_value={"state": "unavailable", "frameToken": "token"}), \
                 patch("vehicle_fit.evaluation.drive_0005.request", return_value={"backend": "snap"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = run(args)
        self.assertEqual(summary["gates"]["execution"], "BLOCKED")
        self.assertEqual(summary["gates"]["legacy_route_parity"], "BLOCKED")
        self.assertEqual(summary["gates"]["snap_quality"], "NOT_ASSESSED")


if __name__ == "__main__":
    unittest.main()
