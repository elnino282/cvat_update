import math
import unittest

import numpy as np

from .metrics import Cuboid, euler_xyz_matrix, errors, label_cuboid, oriented_iou, project


class MetricsTests(unittest.TestCase):
    def box(self, center=(0, 0, 0), size=(4, 2, 2), rotation=(0, 0, 0)):
        return Cuboid(np.array(center, dtype=float), np.array(size, dtype=float), euler_xyz_matrix(rotation))

    def test_identical_disjoint_touching_and_translation(self):
        box = self.box()
        self.assertAlmostEqual(oriented_iou(box, box), 1)
        self.assertEqual(oriented_iou(box, self.box((5, 0, 0))), 0)
        self.assertEqual(oriented_iou(box, self.box((4, 0, 0))), 0)
        self.assertAlmostEqual(oriented_iou(box, self.box((2, 0, 0))), 1 / 3)
        self.assertAlmostEqual(oriented_iou(box, self.box((0, 0, 1))), 1 / 3)

    def test_true_3d_rotation_and_symmetry(self):
        left, right = self.box(), self.box(rotation=(0.2, 0.3, math.pi / 2))
        self.assertAlmostEqual(oriented_iou(left, right), oriented_iou(right, left))
        self.assertGreater(oriented_iou(left, right), 0)
        self.assertLess(oriented_iou(left, right), 1)
        self.assertAlmostEqual(oriented_iou(left, self.box(rotation=(0, 0, math.pi / 2))), 1 / 3)

    def test_containment(self):
        self.assertAlmostEqual(oriented_iou(self.box(), self.box(size=(2, 1, 1))), 1 / 8)

    def test_yaw_mod_pi(self):
        self.assertAlmostEqual(errors(self.box(rotation=(0, 0, math.pi)), self.box())["yaw_error_rad_mod_pi"], 0)

    def test_calibration_and_bottom_center(self):
        # velo X→camera Z, velo Y→camera -X, velo Z→camera -Y.
        transform = np.array([[0, -1, 0, .1], [0, 0, -1, .2], [1, 0, 0, .3], [0, 0, 0, 1]])
        label = {"hwl": [2, 2, 4], "location": [3, 2, 10], "ry": .2}
        box = label_cuboid(label, transform)
        center = (transform @ np.r_[box.center, 1])[:3]
        np.testing.assert_allclose(center, [3, 1, 10])
        camera_corners = (transform @ np.column_stack((box.corners, np.ones(8))).T).T[:, :3]
        self.assertAlmostEqual(camera_corners[:, 1].max(), 2)
        self.assertAlmostEqual(camera_corners[:, 1].min(), 0)
        self.assertEqual(box.contains(box.corners).sum(), 8)
        camera = np.array([[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]])
        pixels, valid = project(np.array([box.center]), transform, camera)
        self.assertTrue(valid[0])
        np.testing.assert_allclose(pixels[0], [80, 60])

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            self.box(size=(0, 2, 2))
        with self.assertRaises(ValueError):
            self.box(center=(float("nan"), 0, 0))


if __name__ == "__main__":
    unittest.main()
