import unittest

import numpy as np

from vehicle_fit.cloud import decode_cloud
from vehicle_fit.snap_backend import deterministic_voxels


class CloudTests(unittest.TestCase):
    def test_kitti_bin_finite_point_and_intensity_mapping(self):
        raw = np.array([[1, 2, 3, .5], [np.nan, 0, 0, .1], [4, 5, 6, .9]], dtype="<f4")
        cloud, original = decode_cloud(raw.tobytes(), "bin")
        np.testing.assert_array_equal(original, [0, 2])
        np.testing.assert_array_equal(cloud[:, :3], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(cloud[:, 3], [.5, .9])

    def test_ascii_pcd_field_order_and_compressed_voxel_inverse(self):
        pcd = (b"VERSION .7\nFIELDS intensity z x y\nSIZE 4 4 4 4\nTYPE F F F F\n"
               b"COUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n"
               b".1 0.01 0.01 0.01\n.9 0.02 0.02 0.02\n.5 0.30 0.30 0.30\n")
        cloud, original = decode_cloud(pcd, "pcd")
        np.testing.assert_array_equal(original, [0, 1, 2])
        np.testing.assert_allclose(cloud[:, 0], [.01, .02, .30])
        np.testing.assert_allclose(cloud[:, 3], [.1, .9, .5])
        representatives, inverse, grid = deterministic_voxels(cloud, .05)
        np.testing.assert_array_equal(representatives, [0, 2])
        np.testing.assert_array_equal(inverse, [0, 0, 1])
        np.testing.assert_array_equal(grid, [[0, 0, 0], [6, 6, 6]])

    def test_rejects_truncated_cloud(self):
        with self.assertRaises(ValueError):
            decode_cloud(b"bad", "bin")


if __name__ == "__main__":
    unittest.main()
