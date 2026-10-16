"""Focused contracts for the retained fixture helper; no CVAT writes."""

import importlib.util
import copy
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("handoff", Path(__file__).with_name("multiclass_handoff.py"))
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.manifest_hash, index = handoff.read_manifest(
            handoff.EVIDENCE / "nuscenes_manifest.json.gz"
        )
        cls.ledger, _ = handoff.read_fixed_ledger(index)

    def test_fixed_car_selection_retains_two_accepted_and_one_no_fit(self):
        cases = handoff.select_cases(self.manifest, "vehicles/car", 3, self.ledger)
        self.assertEqual([item[1] for item in cases], ["pr0-accepted", "pr0-accepted", "pr0-no-fit"])
        self.assertEqual(cases, handoff.select_cases(self.manifest, "vehicles/car", 3, self.ledger))
        self.assertEqual(len({case[0]["case_id"] for case in cases}), 3)

    def test_new_routes_use_dev_and_support_strata_without_fit_results(self):
        for route in handoff.ROUTES[1:]:
            with self.subTest(route=route):
                cases = handoff.select_cases(self.manifest, route, 4)
                self.assertEqual(len(cases), 4)
                self.assertTrue(all(case["split"] == "dev" and ref is None for case, _, ref in cases))
                self.assertEqual([name for _, name, _ in cases], ["dense", "5-15-points", "sparse-1-4", "adjacent"])
                if route == "two-wheels":
                    self.assertEqual({case["ride_state"] for case, _, _ in cases}, {True, False})
                if route == "vehicles/van":
                    self.assertTrue(all(case["source_category"] == "vehicle.car" for case, _, _ in cases))

    def test_pcd_preserves_xyzi_bytes_and_original_point_order(self):
        raw = struct.pack("<10f", 1, 2, 3, 42, 7, -1, -2, -3, 99, 8)
        pcd = handoff.cloud_to_pcd(raw)
        header, payload = pcd.split(b"DATA binary\n", 1)
        self.assertIn(b"FIELDS x y z intensity\n", header)
        self.assertEqual(payload, raw[:16] + raw[20:36])
        for bad in (b"", b"x", struct.pack("<5f", 1, float("nan"), 3, 4, 5)):
            with self.assertRaises(ValueError):
                handoff.cloud_to_pcd(bad)

    def test_source_hash_and_seed_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = struct.pack("<5f", 1, 2, 3, 4, 5)
            Path(directory, "cloud.bin").write_bytes(raw)
            frame = {"lidar_path": "cloud.bin", "sha256": handoff.digest(raw), "point_dimensions": 5}
            seed = {"x": 1, "y": 2, "z": 3, "pointIndex": 0}
            self.assertTrue(handoff.load_cloud(directory, frame, seed))
            with self.assertRaises(ValueError):
                handoff.load_cloud(directory, {**frame, "sha256": "0" * 64}, seed)
            with self.assertRaises(ValueError):
                handoff.load_cloud(directory, frame, {**seed, "pointIndex": 1})
            with self.assertRaises(ValueError):
                handoff.load_cloud(directory, frame, {**seed, "x": 9})

    def test_fixture_count_is_bounded(self):
        for count in (0, -1, 6):
            with self.assertRaises(ValueError):
                handoff.select_cases(self.manifest, "pedestrian", count)

    def test_runtime_schema_accepts_cvat_text_normalization_and_rejects_extra_subtype(self):
        labels = copy.deepcopy(handoff.LABELS)
        for index, label in enumerate(labels):
            label["id"] = index + 11
            for attr_index, attr in enumerate(label["attributes"]):
                attr["id"] = index * 3 + attr_index + 21
                if attr["input_type"] == "text":
                    attr["values"] = [""]
        result = handoff.resolve_schema(labels)
        self.assertEqual(result["vehicleLabelId"], 11)
        self.assertEqual(result["rideStateAttributeId"], 27)
        labels[2]["attributes"].append({"name": "subtype"})
        with self.assertRaises(ValueError):
            handoff.resolve_schema(labels)


if __name__ == "__main__":
    unittest.main()
