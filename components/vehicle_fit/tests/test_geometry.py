import unittest

import numpy as np

from vehicle_fit.geometry import NoFit, fit_vehicle
from vehicle_fit.evaluation.metrics import Cuboid, euler_xyz_matrix, oriented_iou


def road_scene(vehicle_centers=((10.0, 0.0),)):
    ground_x, ground_y = np.meshgrid(np.arange(-5, 36, 1.0), np.arange(-15, 16, 1.0))
    points = [np.column_stack((ground_x.ravel(), ground_y.ravel(), np.zeros(ground_x.size)))]
    for x, y in vehicle_centers:
        longitudinal = np.arange(-2.05, 2.06, .15)
        lateral = np.arange(-.90, .91, .15)
        vertical = np.arange(.15, 1.51, .15)
        surfaces = []
        for sign in (-1, 1):
            lx, lz = np.meshgrid(longitudinal, vertical)
            surfaces.extend([np.column_stack((x + lx.ravel(), np.full(lx.size, y + sign * .9), lz.ravel()))])
            ly, lz = np.meshgrid(lateral, vertical)
            surfaces.extend([np.column_stack((x + sign * 2.1 + np.zeros(ly.size),
                                               y + ly.ravel(), lz.ravel()))])
        for z in (1.35, 1.5):
            lx, ly = np.meshgrid(longitudinal, lateral)
            surfaces.append(np.column_stack((x + lx.ravel(), y + ly.ravel(), np.full(lx.size, z))))
        points.extend(surfaces)
    return np.column_stack((np.concatenate(points), np.zeros(sum(len(p) for p in points))))


class GeometryTests(unittest.TestCase):
    def test_full_observed_car_returns_whole_finite_cuboid(self):
        cloud = road_scene()
        seed = np.array([10.05, .9, .75])
        result = fit_vehicle(cloud, seed)
        box = Cuboid(np.asarray(result["center"]), np.asarray(result["size"]),
                     euler_xyz_matrix(result["rotation"]))
        expected = Cuboid(np.array([10.0, 0.0, .775]), np.array([4.2, 1.8, 1.55]), np.eye(3))
        self.assertGreaterEqual(oriented_iou(box, expected), .70)
        self.assertTrue(np.isfinite(box.corners).all())
        self.assertGreater(result["diagnostics"]["point_count"], 100)

    def test_neighboring_cars_do_not_join_into_an_oversized_fit(self):
        cloud = road_scene(((10.0, -1.45), (10.0, 1.45)))
        result = fit_vehicle(cloud, np.array([10.05, -.55, .75]))
        component = cloud[np.asarray(result["diagnostics"]["mask_indices"], dtype=int)]
        self.assertLess(np.median(component[:, 1]), 0)
        self.assertLess(result["center"][1], 0)
        self.assertLess(result["size"][1], 2.0)

    def test_ground_seed_and_insufficient_surface_are_refused(self):
        cloud = road_scene()
        with self.assertRaises(NoFit):
            fit_vehicle(cloud, np.array([10.0, 0.0, 0.0]))
        sparse = np.array([[10.0, .9, .75, 0.0], [10.1, .9, .75, 0.0],
                           [10.2, .9, .75, 0.0]], dtype=float)
        with self.assertRaises(NoFit):
            fit_vehicle(sparse, sparse[0, :3])


if __name__ == "__main__":
    unittest.main()

class EstimateTests(unittest.TestCase):
    def test_estimate_is_opt_in_and_ground_never_creates_box(self):
        cloud = road_scene()
        seed = np.array([10.05, .9, .75])
        mask = np.zeros(len(cloud), dtype=bool)
        points = cloud[:, :3]
        # A fragment with measured breadth and height but insufficient full-car surfaces.
        mask = ((points[:, 0] >= 9.9) & (points[:, 0] <= 10.55) &
                (points[:, 1] >= .60) & (points[:, 1] <= .91) & (points[:, 2] > .1))
        with self.assertRaises(NoFit):
            fit_vehicle(cloud, seed, mask=mask)
        result = fit_vehicle(cloud, seed, mask=mask, allow_estimated=True)
        self.assertEqual(result['quality'], 'estimated')
        self.assertTrue(np.isfinite(result['center']).all())
        self.assertGreaterEqual(result['size'][0], 2.6)
        with self.assertRaises(NoFit):
            fit_vehicle(cloud, np.array([10, 0, 0]), allow_estimated=True)

    def test_merged_cars_are_not_enclosed_in_one_estimate(self):
        cloud = road_scene(((10, -3), (10, 3)))
        xyz = cloud[:, :3]
        result = fit_vehicle(cloud, np.array([10.05, -2.1, .75]), mask=xyz[:, 2] > .1,
                             allow_estimated=True)
        self.assertLess(result['center'][1], 0)
        self.assertLessEqual(result['size'][1], 2.3)

    def test_prior_completed_footprint_is_marked_for_review_even_if_surface_gate_passes(self):
        from unittest.mock import patch
        candidate = {'center': [10, 0, 1], 'size': [4, 1.8, 1.5], 'rotation': [0, 0, 0],
                     'diagnostics': {'observed_footprint': [2.4, 1.6]}}
        with patch('vehicle_fit.geometry._fit_vehicle', return_value=candidate):
            result = fit_vehicle(road_scene(), [10, .9, .75], allow_estimated=True)
        self.assertEqual(result['quality'], 'estimated')
        self.assertEqual(result['diagnostics']['reviewReason'], 'partially-observed-footprint')

    def test_sparse_mask_with_detached_neighbor_returns_keeps_only_seed_vehicle(self):
        cloud = road_scene(((10, -3), (10, 3)))
        xyz = cloud[:, :3]
        near = np.flatnonzero((abs(xyz[:, 1] + 2.1) < .01) & (xyz[:, 0] >= 9.9) &
                              (xyz[:, 0] <= 10.5) & (xyz[:, 2] >= .6) & (xyz[:, 2] <= .9))[:12]
        far = np.flatnonzero((abs(xyz[:, 1] - 2.1) < .01) & (xyz[:, 0] >= 9.9) &
                             (xyz[:, 0] <= 10.5) & (xyz[:, 2] >= .6) & (xyz[:, 2] <= .9))[:12]
        mask = np.zeros(len(cloud), dtype=bool)
        mask[np.r_[near, far]] = True
        from vehicle_fit.geometry import estimate_vehicle
        try:
            result = estimate_vehicle(cloud, xyz[near[0]], mask=mask)
        except NoFit:
            return  # Refusal is also valid when the isolated fragment has no width evidence.
        self.assertLess(result['center'][1], 0)
        self.assertTrue(np.all(xyz[result['diagnostics']['mask_indices'], 1] < 0))

    def test_sparse_isotropic_patch_is_refused_instead_of_guessing_vehicle_orientation(self):
        cloud = road_scene()
        xyz = cloud[:, :3]
        selected = np.flatnonzero((xyz[:, 0] >= 9.9) & (xyz[:, 0] <= 10.5) &
                                 (xyz[:, 1] >= .6) & (xyz[:, 1] <= .9) &
                                 (xyz[:, 2] >= .6) & (xyz[:, 2] <= 1.1))[:20]
        mask = np.zeros(len(cloud), dtype=bool)
        mask[selected] = True
        # Keep only the observed fragment plus the road: no surrounding vehicle returns.
        subset = np.flatnonzero((xyz[:, 2] == 0) | mask)
        fragment = cloud[subset]
        from vehicle_fit.geometry import estimate_vehicle
        with self.assertRaises(NoFit):
            estimate_vehicle(fragment, xyz[selected[0]], mask=mask[subset])
