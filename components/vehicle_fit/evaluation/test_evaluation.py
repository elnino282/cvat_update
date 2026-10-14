import unittest

import numpy as np

from .cli import summarize
from .manifest import point_seeds
from .metrics import Cuboid


class EvaluationTests(unittest.TestCase):
    def test_seeds_are_measured_finite_and_frozen(self):
        box = Cuboid(np.zeros(3), np.ones(3) * 4, np.eye(3))
        cloud = np.array([[0, 0, 0, 1], [1, 1, 1, 1], [-1, -1, 1, 1],
                          [20, 0, 0, 1], [float("nan"), 0, 0, 1]], dtype=np.float32)
        seeds, count = point_seeds(cloud, box)
        self.assertEqual(count, 3)
        self.assertEqual(len(seeds), 3)
        self.assertEqual((seeds, count), point_seeds(cloud, box))
        self.assertEqual(len({s["pointIndex"] for s in seeds}), 3)
        for seed in seeds:
            np.testing.assert_array_equal(seed["point"], cloud[seed["pointIndex"], :3])
        self.assertEqual(point_seeds(cloud[3:], box), ([], 0))

    def test_refusals_unseedable_and_retry_remain_in_denominator(self):
        metrics = {"iou_3d": .8, "center_error_m": .1, "yaw_error_rad_mod_pi": .1,
                   "dimension_error_m_lwh": [.1, .1, .1], "dimension_relative_error_lwh": [.01, .02, .03]}
        attempt = {"http_latency_s": 1, "cache_group": "warm_frame", "response": None}
        records = [{"adequate": True, "accepted": True, "correct_vehicle": True,
                    "metrics": metrics, "attempts": [attempt, attempt], "outcome": "PASS"},
                   {"adequate": True, "accepted": False, "correct_vehicle": False,
                    "attempts": [attempt], "outcome": "NO_FIT"},
                   {"adequate": False, "accepted": False, "correct_vehicle": False,
                    "attempts": [], "outcome": "UNSEEDABLE"}]
        summary = summarize(records)
        self.assertEqual(summary["all"]["cases"], 3)
        self.assertAlmostEqual(summary["all"]["success_rate_all_cases"], 1 / 3)
        self.assertAlmostEqual(summary["all"]["retry_rate"], .5)
        self.assertAlmostEqual(summary["all"]["iou_all_cases_zero_for_no_fit"]["mean"], .8 / 3)
        self.assertAlmostEqual(summary["adequate"]["coverage"], .5)
        self.assertEqual(summary["http_latency_seconds"]["all"]["count"], 3)


if __name__ == "__main__":
    unittest.main()
