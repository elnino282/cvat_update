# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

import unittest
import importlib.util
from pathlib import Path

# This validator has no Django dependency; keep it runnable on the host, too.
_spec = importlib.util.spec_from_file_location(
    "auto_fit_metadata", Path(__file__).parents[1] / "auto_fit_metadata.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_auto_fit_metadata = _module.validate_auto_fit_metadata


METADATA = {
    "version": 1, "request_id": "fit-test", "frame": 0, "class_key": "pedestrian",
    "quality": "estimated", "fit_mode": "hybrid_prior", "yaw_uncertain": True,
    "prototype_key": "pedestrian-adult", "prior_version": "pr0-dev", "experimental": True,
}


class AutoFitMetadataTests(unittest.TestCase):
    def test_legacy_empty_and_complete_provenance(self):
        self.assertEqual(validate_auto_fit_metadata({}), {})
        result = validate_auto_fit_metadata(METADATA)
        self.assertEqual(result, METADATA)
        self.assertIsNot(result, METADATA)
        self.assertEqual(validate_auto_fit_metadata({**METADATA, "prototype_key": None,
            "prior_version": None})["prototype_key"], None)

    def test_rejects_unbounded_unknown_and_non_json_values(self):
        for value in (None, [], "text", {"version": 1}, {**METADATA, "points": []},
            {**METADATA, "frame": float("nan")}, {**METADATA, "frame": -1},
            {**METADATA, "frame": True}, {**METADATA, "version": True},
            {**METADATA, "quality": "ok"}, {**METADATA, "fit_mode": "unknown"},
            {**METADATA, "experimental": 1}, {**METADATA, "yaw_uncertain": "true"},
            {**METADATA, "prototype_key": {}}, {**METADATA, "prior_version": object()},
            {**METADATA, "request_id": "x" * 129}, {**METADATA, "class_key": "a\nb"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_auto_fit_metadata(value)


if __name__ == "__main__":
    unittest.main()
