"""Adversarial instance and observed-end regression, with refinement enabled."""
import unittest

import numpy as np

from vehicle_fit.geometry import (NoFit, VehiclePriors, _finish, _faces, _prepare_evidence, _validate_geometry,
                                  estimate_vehicle, fit_vehicle)
from vehicle_fit.tests.test_geometry import road_scene


class InstanceSafetyTests(unittest.TestCase):
    def test_two_compact_cars_with_touching_bumpers_are_not_one_long_car(self):
        scene = road_scene()
        car = scene[scene[:, 2] > .1].copy()
        car[:, 0] = 10+(car[:, 0]-10)*2.65/4.2
        second = car.copy()
        second[:, 0] += 2.70
        cloud = np.vstack((scene[scene[:, 2] == 0], car, second))
        seed = car[np.linalg.norm(car[:, :3]-[10, .9, .75], axis=1).argmin(), :3]
        for mask in (None, cloud[:, 2] > .1):
            with self.subTest(path='fit', mask=mask is not None), self.assertRaises(NoFit):
                fit_vehicle(cloud, seed, mask, allow_estimated=True, adaptive_refinement=True)
            with self.subTest(path='estimate', mask=mask is not None), self.assertRaises(NoFit):
                estimate_vehicle(cloud, seed, mask, adaptive_refinement=True)

    def test_touching_bumpers_do_not_make_one_seed_a_neighbor_instance(self):
        cloud = road_scene(((10, 0), (14.25, 0)))
        xyz = cloud[:, :3]
        seed = np.array([12.1, 0, .75])
        mask = (xyz[:, 0] > 12.12) & (xyz[:, 2] > .1)
        mask[np.linalg.norm(xyz-seed, axis=1).argmin()] = True
        for fitter in (estimate_vehicle, fit_vehicle):
            kwargs = {'allow_estimated': True} if fitter is fit_vehicle else {}
            with self.subTest(fitter=fitter.__name__), self.assertRaises(NoFit) as caught:
                fitter(cloud, seed, mask, adaptive_refinement=True, **kwargs)
            self.assertEqual(caught.exception.diagnostics['safetyGate'], 'seed-local-mask')

    def test_seed_outside_winning_candidate_is_terminal_even_when_components_touch(self):
        cloud = road_scene(((10, 0), (10, 2.1)))
        xyz = cloud[:, :3]
        seed = np.array([10.05, 1.2, .75])
        mask = (xyz[:, 2] > .1) & (xyz[:, 1] <= .91)
        patch = np.flatnonzero((abs(xyz[:, 1]-1.2) < .01) & (xyz[:, 0] >= 9.9) &
                              (xyz[:, 0] <= 10.5) & (xyz[:, 2] >= .6) & (xyz[:, 2] <= .9))
        # Include the selected return and a few points, below quantile trimming.
        mask[patch[:7]] = True
        mask[np.linalg.norm(xyz-seed, axis=1).argmin()] = True
        for allow in (False, True):
            with self.subTest(allow=allow), self.assertRaises(NoFit) as caught:
                fit_vehicle(cloud, seed, mask, allow_estimated=allow)
            self.assertIn(caught.exception.diagnostics['safetyGate'], ('seed-outside', 'seed-cluster', 'seed-local-mask'))

    def test_seed_containment_tolerance_and_vertical_axis(self):
        cloud = road_scene()
        seed = np.array([10.05, .9, .75])
        evidence = _prepare_evidence(cloud, seed, None, lambda: False)
        # Keep normal supporting surfaces within contamination tolerance while
        # making the selected end exceed the much smaller seed tolerance.
        for displacement, axis in ((.15, 1), (.95, 2)):
            box = {'center': [10, -displacement if axis == 1 else 0, .8+displacement if axis == 2 else .8],
                   'size': [4.3, 1.8, 1.6], 'rotation': [0, 0, 0]}
            with self.subTest(axis=axis), self.assertRaises(NoFit) as caught:
                _validate_geometry(box, evidence, lambda: False)
            self.assertEqual(caught.exception.diagnostics['safetyGate'], 'seed-outside')

    def test_geometry_only_and_cancelled_refinement_cannot_bypass_safety(self):
        cloud = road_scene()
        for fitter in (fit_vehicle, estimate_vehicle):
            with self.subTest(fitter=fitter.__name__), self.assertRaises(NoFit) as caught:
                fitter(cloud, [10.05, .9, .75], cancelled=lambda: True, adaptive_refinement=True)
            self.assertEqual(caught.exception.reason, 'cancelled')

    def test_neighbor_mask_with_one_selected_seed_is_terminal_in_every_path(self):
        cloud = road_scene(((10, -1.45), (10, 1.45)))
        seed = np.array([10.05, -.55, .75])
        mask = cloud[:, 1] > .5
        mask[np.linalg.norm(cloud[:, :3] - seed, axis=1).argmin()] = True
        for allow in (False, True):
            for gate in (.10, .05):
                with self.subTest(allow=allow, gate=gate), self.assertRaises(NoFit):
                    fit_vehicle(cloud, seed, mask, max_surface_score=gate, allow_estimated=allow)
        with self.assertRaises(NoFit):
            estimate_vehicle(cloud, seed, mask)

    def test_isolated_measured_seed_cannot_borrow_a_neighbor_cluster(self):
        cloud = road_scene(((10, 3),))
        cloud = np.vstack((cloud, [10, 0, .75, 0]))
        for fitter in (estimate_vehicle, fit_vehicle):
            with self.subTest(fitter=fitter.__name__), self.assertRaises(NoFit):
                fitter(cloud, cloud[-1, :3])


class RefinementTests(unittest.TestCase):
    def crop(self, sign):
        cloud = road_scene()
        xyz = cloud[:, :3]
        mask = (xyz[:, 2] > .1) & (xyz[:, 0] <= 11.3 if sign == 1 else xyz[:, 0] >= 8.7)
        return cloud, np.array([10.05, .9, .75]), mask

    def test_one_end_extension_preserves_anchor_and_improves_length(self):
        for sign in (-1, 1):
            cloud, seed, mask = self.crop(sign)
            old = fit_vehicle(cloud, seed, mask, allow_estimated=True)
            new = fit_vehicle(cloud, seed, mask, allow_estimated=True, adaptive_refinement=True)
            with self.subTest(sign=sign):
                self.assertGreater(new['size'][0], old['size'][0])
                self.assertLess(abs(4.2-new['size'][0]), abs(4.2-old['size'][0]))
                self.assertEqual(new['size'][1:], old['size'][1:])
                self.assertEqual(new['rotation'], old['rotation'])
                a = -sign
                u = np.array([np.cos(old['rotation'][2]), np.sin(old['rotation'][2]), 0])
                np.testing.assert_allclose(np.array(new['center'])+a*new['size'][0]/2*u,
                                           np.array(old['center'])+a*old['size'][0]/2*u)
                self.assertEqual(new['quality'], 'estimated')

    def test_full_observation_and_unknown_ends_do_not_extend(self):
        cloud = road_scene()
        xyz = cloud[:, :3]
        for mask in (xyz[:, 2] > .1, (xyz[:, 2] > .1) & (xyz[:, 0] > 8.5) & (xyz[:, 0] < 11.5)):
            old = fit_vehicle(cloud, [10.05, .9, .75], mask, allow_estimated=True)
            new = fit_vehicle(cloud, [10.05, .9, .75], mask, allow_estimated=True, adaptive_refinement=True)
            self.assertEqual(old['size'], new['size'])
            self.assertFalse(new['diagnostics']['sizeRefinement']['applied'])

    def test_quality_is_assessed_before_strict_policy(self):
        cloud, seed, mask = self.crop(1)
        relaxed = fit_vehicle(cloud, seed, mask, allow_estimated=True, adaptive_refinement=True)
        with self.assertRaises(NoFit) as caught:
            fit_vehicle(cloud, seed, mask, allow_estimated=False, adaptive_refinement=True)
        self.assertEqual(caught.exception.diagnostics['assessedQuality'], relaxed['quality'])

    def test_refinement_default_prior_matches_runtime(self):
        self.assertEqual(VehiclePriors(), VehiclePriors(3.7, 1.61, 1.49))

    def test_extension_into_detached_support_is_blocked(self):
        cloud = road_scene()
        mask = (cloud[:, 2] > .1) & (cloud[:, 0] <= 10.3)
        subset = (cloud[:, 2] == 0) | mask
        cloud, mask = cloud[subset], mask[subset]
        neighbor = np.array([[10.95, y, z, 0] for y in (-.1, .1) for z in (.4, .7, 1., 1.3)])
        cloud = np.vstack((cloud, neighbor))
        mask = np.r_[mask, np.zeros(len(neighbor), dtype=bool)]
        evidence = _prepare_evidence(cloud, [10.05, .9, .75], mask, lambda: False)
        base = {'center': [9.2, 0, .8], 'size': [2.6, 1.8, 1.6], 'rotation': [0, 0, 0],
                'diagnostics': {'plausible_yaws': [0]}}
        result = _finish(base, evidence, VehiclePriors(), True, lambda: False)
        self.assertEqual(result['size'][0], 2.6)
        self.assertEqual(result['diagnostics']['sizeRefinement']['blockedReason'], 'neighbor-overlap')

    def test_no_extension_with_unstable_yaw_or_horizontal_only_cap_patch(self):
        cloud, seed, mask = self.crop(1)
        evidence = _prepare_evidence(cloud, seed, mask, lambda: False)
        base = {'center': [9.575, 0, .8], 'size': [3.4, 1.8, 1.6], 'rotation': [0, 0, 0],
                'diagnostics': {'plausible_yaws': [0, .3]}}
        result = _finish(base, evidence, VehiclePriors(), True, lambda: False)
        self.assertFalse(result['diagnostics']['sizeRefinement']['applied'])
        self.assertFalse(result['diagnostics']['sizeRefinement']['yawStable'])
        mask &= (cloud[:, 2] >= .6) & (cloud[:, 2] <= .9)
        evidence = _prepare_evidence(cloud, seed, mask, lambda: False)
        self.assertEqual(_faces(evidence, 0)[2][0], [False, False])

    def test_supported_small_car_is_preserved_below_prior(self):
        cloud = road_scene()
        vehicle = cloud[:, 2] > .1
        cloud[vehicle, 0] = 10+(cloud[vehicle, 0]-10)*3/4.2
        seed = np.array([10+.05*3/4.2, .9, .75])
        result = fit_vehicle(cloud, seed, adaptive_refinement=True, allow_estimated=True)
        self.assertLess(result['size'][0], 3.3)
        self.assertFalse(result['diagnostics']['sizeRefinement']['applied'])

    def test_supported_long_car_does_not_shrink_or_trigger_joined_surface_gate(self):
        cloud = road_scene()
        vehicle = cloud[:, 2] > .1
        cloud[vehicle, 0] = 10+(cloud[vehicle, 0]-10)*5.4/4.2
        result = fit_vehicle(cloud, [10+.05*5.4/4.2, .9, .75], adaptive_refinement=True, allow_estimated=True)
        self.assertGreaterEqual(result['size'][0], 5.4)
        self.assertFalse(result['diagnostics']['sizeRefinement']['applied'])

    def test_rotated_translated_end_anchoring(self):
        cloud, seed, mask = self.crop(1)
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        cloud[:, :2] = cloud[:, :2] @ rotation.T + [-20, 5]
        seed[:2] = seed[:2] @ rotation.T + [-20, 5]
        old = fit_vehicle(cloud, seed, mask, allow_estimated=True)
        new = fit_vehicle(cloud, seed, mask, allow_estimated=True, adaptive_refinement=True)
        self.assertGreater(new['size'][0], old['size'][0])
        yaw = old['rotation'][2]
        u = np.array([np.cos(yaw), np.sin(yaw), 0])
        a = -1 if new['diagnostics']['sizeRefinement']['extensionDirection'] == 'positive-local-x' else 1
        np.testing.assert_allclose(np.array(old['center'])+a*old['size'][0]/2*u,
                                   np.array(new['center'])+a*new['size'][0]/2*u, atol=1e-7)

    def test_fragment_with_no_supported_ends_is_refused_in_both_policies(self):
        cloud = road_scene()
        xyz = cloud[:, :3]
        mask = ((xyz[:, 0] >= 9.9) & (xyz[:, 0] <= 10.55) &
                (xyz[:, 1] >= .60) & (xyz[:, 1] <= .91) & (xyz[:, 2] > .1))
        for allow in (False, True):
            with self.subTest(allow=allow), self.assertRaises(NoFit):
                fit_vehicle(cloud, [10.05, .9, .75], mask, allow_estimated=allow, adaptive_refinement=True)


if __name__ == '__main__':
    unittest.main()
