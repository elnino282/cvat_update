"""Coordinate, split leakage, provenance and reproducibility checks without raw data."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..cloud import decode_cloud
from .metrics import euler_xyz_matrix
from .nuscenes_dataset import (TABLES, build_priors, cloud_to_pcd, create_manifest, footprint_distance,
                               inventory, lidar_cuboid, load_cloud, load_manifest, matrix_to_euler_xyz,
                               pose_matrix, quaternion_matrix, scene_splits, select_cases, source_path,
                               taxonomy, write_artifact)


def miniature_dataset(root):
    tables = {name: [] for name in TABLES}
    tables["category"] = [{"token": "car-category", "name": "vehicle.car"},
                          {"token": "bicycle-category", "name": "vehicle.bicycle"},
                          {"token": "barrier-category", "name": "movable_object.barrier"}]
    tables["attribute"] = [{"token": "rider", "name": "cycle.with_rider"}]
    tables["visibility"] = [{"token": "4", "level": "v80-100"}]
    tables["sensor"] = [{"token": "lidar", "channel": "LIDAR_TOP", "modality": "lidar"}]
    tables["log"] = [{"token": "log"}]
    tables["calibrated_sensor"] = [{"token": "calib", "sensor_token": "lidar", "translation": [0, 0, 0],
                                    "rotation": [1, 0, 0, 0]}]
    for i in range(3):
        scene, sample, data, ego = f"scene{i}", f"sample{i}", f"data{i}", f"ego{i}"
        tables["scene"].append({"token": scene, "name": scene, "log_token": "log", "nbr_samples": 1,
                                "first_sample_token": sample, "last_sample_token": sample})
        tables["sample"].append({"token": sample, "scene_token": scene, "prev": "", "next": ""})
        tables["ego_pose"].append({"token": ego, "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]})
        filename = f"samples/LIDAR_TOP/{data}.pcd.bin"
        (root / filename).parent.mkdir(parents=True, exist_ok=True)
        np.array([[0, 0, 0, 12, 0], [1, 0, 0, 30, 1], [np.nan, 0, 0, 1, 1]], dtype="<f4").tofile(root / filename)
        tables["sample_data"].append({"token": data, "sample_token": sample, "ego_pose_token": ego,
                                      "calibrated_sensor_token": "calib", "is_key_frame": True,
                                      "filename": filename, "timestamp": i, "prev": "", "next": ""})
        for kind, center in [("car", [0, 0, 0]), ("bicycle", [10, 0, 0]), ("barrier", [0, 2, 0])]:
            ann, instance = f"{kind}-ann{i}", f"{kind}-instance{i}"
            tables["instance"].append({"token": instance, "category_token": f"{kind}-category",
                                       "first_annotation_token": ann, "last_annotation_token": ann})
            tables["sample_annotation"].append({"token": ann, "sample_token": sample, "instance_token": instance,
                                                 "attribute_tokens": ["rider"] if kind == "bicycle" else [],
                                                 "visibility_token": "4", "translation": center,
                                                 "size": [2, 4, 2], "rotation": [1, 0, 0, 0],
                                                 "num_lidar_pts": 2 if kind == "car" else 0, "prev": "", "next": ""})
    (root / "v1.0-mini").mkdir()
    for name, rows in tables.items():
        (root / "v1.0-mini" / f"{name}.json").write_text(json.dumps(rows))


class NuScenesDatasetTests(unittest.TestCase):
    def test_taxonomy_preserves_locked_contract_and_unknown_rider(self):
        self.assertEqual(taxonomy("vehicle.car"), ("vehicles", "car", None))
        self.assertEqual(taxonomy("vehicle.bus.bendy"), ("vehicles", "bus", None))
        self.assertEqual(taxonomy("vehicle.trailer"), (None, None, None))
        self.assertEqual(taxonomy("human.pedestrian.child"), ("pedestrian", None, None))
        for category in ("vehicle.bicycle", "vehicle.motorcycle"):
            self.assertEqual(taxonomy(category, ["cycle.with_rider"]), ("two-wheels", None, True))
            self.assertEqual(taxonomy(category, ["cycle.without_rider"]), ("two-wheels", None, False))
            self.assertEqual(taxonomy(category), ("two-wheels", None, None))
            self.assertEqual(taxonomy(category, ["cycle.with_rider", "cycle.without_rider"])[2], None)

    def test_global_to_sensor_composes_both_poses_and_swaps_width_length(self):
        q = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        calibration = {"rotation": q, "translation": [10, 0, 0]}
        ego = {"rotation": q, "translation": [20, 0, 0]}
        ann = {"translation": [19, 8, 3], "size": [2, 4, 2], "rotation": [0, 0, 0, 1]}
        box = lidar_cuboid(ann, calibration, ego)
        np.testing.assert_allclose(box.center, [1, 2, 3], atol=1e-12)
        np.testing.assert_allclose(box.size, [4, 2, 2])
        np.testing.assert_allclose(box.axes, np.eye(3), atol=1e-12)
        global_point = (pose_matrix(ego) @ pose_matrix(calibration) @ np.r_[box.center, 1])[:3]
        np.testing.assert_allclose(global_point, ann["translation"])

    def test_rotation_roundtrip_retains_pitch_roll_and_rejects_bad_quaternions(self):
        for angles in ([.1, -.2, .8], [-.4, .3, -2.8], [.3, math.pi / 2, 0]):
            axes = euler_xyz_matrix(angles)
            np.testing.assert_allclose(euler_xyz_matrix(matrix_to_euler_xyz(axes)), axes, atol=1e-7)
        for quaternion in ([0, 0, 0, 0], [math.nan, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0]):
            with self.assertRaises(ValueError):
                quaternion_matrix(quaternion)

    def test_adjacent_obb_gap_not_only_center_distance(self):
        a = {"translation": [0, 0, 0], "size": [2, 10, 2], "rotation": [1, 0, 0, 0]}
        b = {**a, "translation": [10.5, 0, 0]}
        self.assertAlmostEqual(footprint_distance(a, b), .5)
        self.assertEqual(footprint_distance(a, {**b, "translation": [0, 1, 0]}), 0)
        self.assertAlmostEqual(footprint_distance(a, {**b, "translation": [0, 5, 0]}), 3)

    def test_scene_split_order_independent_no_overlap(self):
        scenes = [{"token": str(i)} for i in range(10)]
        split = scene_splits(scenes)
        self.assertEqual(split, scene_splits(list(reversed(scenes))))
        self.assertEqual([list(split.values()).count(key) for key in ("dev", "validation", "held_out")], [6, 2, 2])
        with self.assertRaises(ValueError):
            scene_splits(scenes[:2])

    def test_full_generator_keeps_unseedable_and_unmapped_neighbors_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            miniature_dataset(root)
            before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            manifest, audit = create_manifest(root, "v1.0-mini")
            self.assertEqual(audit["missing_files"], [])
            self.assertEqual(audit["broken_references"], [])
            self.assertEqual(len(manifest["cases"]), 6)
            self.assertEqual(len(manifest["frames"]), 3)
            self.assertEqual(sum(bool(case["seeds"]) for case in manifest["cases"]), 3)
            for case in manifest["cases"]:
                frame = manifest["frames"][case["sample_token"]]
                self.assertEqual(len(frame["annotations"]), 3)
                self.assertEqual(frame["split"], case["split"])
                if case["vehicles_type"] == "car":
                    self.assertEqual(case["bins"]["adjacency"], "adjacent")
                    self.assertEqual(case["gt_point_count"], 2)
                    cloud = load_cloud(manifest, case["sample_token"])
                    decoded, indices = decode_cloud(cloud_to_pcd(cloud), "pcd")
                    np.testing.assert_array_equal(indices, [0, 1])
                    for seed in case["seeds"]:
                        np.testing.assert_array_equal(decoded[seed["pointIndex"], :3], seed["point"])
                else:
                    self.assertEqual(case["seeds"], [])
                    self.assertIs(case["ride_state"], True)
            self.assertEqual(before, {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()})
            self.assertEqual(manifest, create_manifest(root, "v1.0-mini")[0])
            selected = select_cases(manifest, "held_out", 2)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected, select_cases(manifest, "held_out", 2))
            self.assertEqual(len(select_cases(manifest, "held_out", 2, seedable_only=False)), 2)
            lidar_path = root / next(iter(manifest["frames"].values()))["lidar_path"]
            lidar_path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Frozen cloud changed"):
                load_cloud(manifest, next(iter(manifest["frames"])))

    def test_inventory_detects_broken_reference_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            miniature_dataset(root)
            path = root / "v1.0-mini/sample_data.json"
            rows = json.loads(path.read_text())
            rows[0]["ego_pose_token"] = "missing"
            rows[0]["filename"] = "samples/LIDAR_TOP/missing.bin"
            path.write_text(json.dumps(rows))
            audit, _ = inventory(root, "v1.0-mini")
            self.assertEqual(len(audit["broken_references"]), 1)
            self.assertEqual(audit["missing_files"], ["samples/LIDAR_TOP/missing.bin"])
            with self.assertRaisesRegex(ValueError, "Broken token"):
                create_manifest(root, "v1.0-mini")

    def test_dev_only_instance_weighted_prototypes_and_van_limitation(self):
        cases = []
        for i in range(24):
            for repeat in range(1 if i % 2 else 4):
                cases.append({"case_id": f"{i}-{repeat}", "annotation_token": f"{i}-{repeat}",
                              "instance_token": str(i), "split": "dev", "target_label": "vehicles",
                              "vehicles_type": "car", "source_category": "vehicle.car", "ride_state": None,
                              "gt_size": [4 + i * .1, 2 + i * .01, 1.5 + i * .01]})
        manifest = {"cases": cases, "dataset_version": "test", "random_seed": 1, "source_hashes": {},
                    "script_sha256": "test", "scene_splits": {"train": "dev", "test": "held_out"}}
        original = build_priors(manifest, "2026-10-16")
        changed = copy.deepcopy(manifest)
        changed["cases"].append({**cases[0], "split": "held_out", "gt_size": [100, 100, 100]})
        self.assertEqual(original, build_priors(changed, "2026-10-16"))
        car = [record for record in original["records"] if record["vehicles_type"] == "car"]
        van = [record for record in original["records"] if record["vehicles_type"] == "van"]
        self.assertEqual(len(car), 3)
        self.assertEqual(sum(record["instance_count"] for record in car), 24)
        self.assertEqual(sum(record["sample_count"] for record in car), len(cases))
        for c, v in zip(car, van):
            self.assertEqual(c["median_lwh"], v["median_lwh"])
            self.assertEqual(v["acceptance_gt"], "insufficient-ground-truth")
            self.assertTrue(np.all(np.array(c["p10_lwh"]) <= c["median_lwh"]))
            self.assertTrue(np.all(np.array(c["median_lwh"]) <= c["p90_lwh"]))
            self.assertEqual(c["split"], "dev")

    def test_artifacts_byte_reproducible_and_frozen_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json.gz"
            digest = write_artifact(path, {"value": [1, 2, 3]}, compressed=True)
            self.assertEqual(write_artifact(path, {"value": [1, 2, 3]}, compressed=True), digest)
            self.assertEqual(load_manifest(path), {"value": [1, 2, 3]})
            with self.assertRaisesRegex(ValueError, "Refusing"):
                write_artifact(path, {"value": [1]}, compressed=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                source_path(directory, "../outside")


if __name__ == "__main__":
    unittest.main()
