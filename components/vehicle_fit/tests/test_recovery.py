"""Frozen tune regressions. Dataset absence is a skip, never a quality PASS."""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from vehicle_fit import geometry
from vehicle_fit.evaluation.replay import verified_frame
from vehicle_fit.tests.test_geometry import road_scene


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT/'components/vehicle_fit/evidence/object-manifest.json'
INPUTS = ROOT/'components/vehicle_fit/evidence/fast-final-tune/cases.jsonl'
DATA_AVAILABLE = MANIFEST.exists() and INPUTS.exists() and Path('/home/quynhhn/Downloads/100_training/training/velodyne').exists()


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cloud = road_scene()
        self.seed = np.array([10.05,.9,.75])
        self.evidence = geometry._prepare_evidence(self.cloud,self.seed,None,lambda:False)
        self.priors = geometry.VehiclePriors()

    def hypothesis(self, score=.01, center=(10,0), length=4.3, width=1.88):
        return (score,np.array(center,dtype=float),length,width,1.6,0.,0.,np.array([4.2,1.8]))

    def test_same_instance_geometry_ambiguity_is_estimated_not_instance_conflict(self):
        hypotheses = [self.hypothesis(),self.hypothesis(.011,(10.6,0),5.3)]
        result = geometry._select_hypotheses(hypotheses,self.evidence,self.priors,lambda:False,.10)
        self.assertEqual(result['quality'],'estimated')
        self.assertEqual(result['diagnostics']['reviewReason'],'geometry-ambiguity-same-instance')
        self.assertTrue(result['diagnostics']['candidateCounts']['completed'])
        geometry._validate_geometry(result,self.evidence,lambda:False)

    def test_p0_neighbor_check_is_independent_of_geometry_score_band(self):
        neighbor = np.array([[10,y,z,0] for y in (1.4,1.45) for z in (.4,.7,1.,1.3)])
        cloud = np.vstack((self.cloud,neighbor))
        mask = np.r_[self.cloud[:,2]>.1,np.zeros(len(neighbor),bool)]
        evidence = geometry._prepare_evidence(cloud,self.seed,mask,lambda:False)
        # The preferred geometry encloses detached context. The fallback is
        # far outside the geometry ambiguity score band but remains safe.
        hypotheses = [self.hypothesis(.001,(10,.25),width=2.3),self.hypothesis(1.)]
        trace = []
        result = geometry._select_hypotheses(hypotheses,evidence,self.priors,lambda:False,.10,trace)
        self.assertEqual(trace[0]['reason'],'neighbor-overlap')
        self.assertEqual(result['diagnostics']['candidateCounts']['firstSafeRank'],2)
        geometry._validate_geometry(result,evidence,lambda:False)

    def test_no_small_top_k_limit_can_hide_the_first_safe_candidate(self):
        hypotheses = [self.hypothesis(i*.0001,(13+i*.01,0)) for i in range(60)]
        hypotheses.append(self.hypothesis(.008))
        result = geometry._select_hypotheses(hypotheses,self.evidence,self.priors,lambda:False,.10)
        self.assertEqual(result['diagnostics']['candidateCounts']['firstSafeRank'],61)
        self.assertEqual(result['diagnostics']['candidateCounts']['evaluated'],61)

    def test_cancel_during_ambiguity_cannot_publish_the_already_safe_candidate(self):
        calls = 0
        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(geometry.NoFit) as caught:
            geometry._select_hypotheses([self.hypothesis(),self.hypothesis(.011)],self.evidence,self.priors,cancelled,.10)
        self.assertEqual(caught.exception.reason,'cancelled')
        self.assertFalse(caught.exception.diagnostics['candidateCounts']['completed'])
        self.assertEqual(caught.exception.diagnostics['candidateCounts']['safe'],1)

    def test_global_evidence_failure_cannot_search_another_instance(self):
        with patch('vehicle_fit.geometry._select_hypotheses') as select:
            with self.assertRaises(geometry.NoFit):
                geometry.fit_vehicle(self.cloud,[10,0,0],allow_estimated=True)
        select.assert_not_called()

    def test_measurement_padding_is_symmetric_and_missing_faces_do_not_gate_yaw(self):
        lower,upper = np.array([0.,0.]),np.array([3.,1.5])
        placements = geometry._candidate_placements(lower,upper,np.array([3.7,1.58]),[[False,False],[False,False]])
        self.assertTrue(placements)
        for centre,basis in placements:
            self.assertAlmostEqual(centre[1],.75)
            self.assertEqual(basis[0],'estimated')
        unsupported = [[False, False], [False, False]]
        provenance = geometry._placement_provenance(['observed-bounds', 'observed-bounds'], unsupported)
        self.assertEqual(provenance['placementSource'], 'observed-bounds')
        self.assertEqual(provenance['supportedFacesLW'], unsupported)
        self.assertEqual(geometry._placement_provenance(['estimated', 'observed-bounds'], unsupported)['placementSource'], 'estimated')

    def test_axis_specific_observation_recovers_the_baseline_estimator_placement(self):
        placements=geometry._candidate_placements(np.zeros(2),np.array([.7,1.4]),np.array([3.7,1.61]),[[False,False],[False,False]])
        expected=np.array([1.85,.7])
        self.assertTrue(any(np.allclose(centre,expected) for centre,_ in placements))

    def test_refinement_provenance_records_prior_blend(self):
        cloud=road_scene()
        mask=(cloud[:,0] <= 11.3) & (cloud[:,2] > .1)
        evidence=geometry._prepare_evidence(cloud,self.seed,mask,lambda:False)
        base={'center':[9.575,0,.8],'size':[3.4,1.8,1.6],'rotation':[0,0,0],
              'diagnostics':{'plausible_yaws':[0],'dimensionSourceLWH':[['prior'],['measured-margin'],['measured-margin']]}}
        result=geometry._finish(base,evidence,self.priors,True,lambda:False)
        self.assertTrue(result['diagnostics']['sizeRefinement']['applied'])
        self.assertEqual(result['diagnostics']['dimensionSourceLWH'][0],['prior-blend'])
        self.assertEqual(result['diagnostics']['dimensionSourceBeforeRefinementLWH'][0],['prior'])
        self.assertEqual(result['diagnostics']['placementBasisLW'][0], 'observed-face')


@unittest.skipUnless(DATA_AVAILABLE, 'Frozen KITTI tune assets required; no quality claim without them')
class FrozenRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.inputs = {r['case_id']: r for r in map(json.loads, INPUTS.read_text().splitlines())}

    def fit(self, case_id):
        case = next(c for c in self.manifest['cases'] if c['case_id'] == case_id)
        self.assertEqual(case['split'], 'tune')
        cloud = verified_frame(self.manifest, case['frame'])
        mask = np.zeros(len(cloud), bool)
        mask[self.inputs[case_id]['response']['diagnostics']['maskIndices']] = True
        seed = np.array(case['seeds'][0]['point'])
        result = geometry.fit_vehicle(cloud, seed, mask=mask, priors=geometry.VehiclePriors(3.7,1.61,1.49),
                                      max_surface_score=.10, allow_estimated=True)
        evidence = geometry._prepare_evidence(cloud, seed, mask, lambda: False)
        geometry._validate_geometry(result, evidence, lambda: False)
        return result

    def test_shortening_does_not_replace_a_safe_prior_hypothesis(self):
        result = self.fit('object-000010-car-4')
        self.assertGreaterEqual(result['size'][0], 3.4)

    def test_missing_cap_does_not_rotate_yaw_to_pass_placement(self):
        result = self.fit('object-000002-car-1')
        baseline_yaw = math.radians(8)
        difference = abs((result['rotation'][2]-baseline_yaw+math.pi/2)%math.pi-math.pi/2)
        self.assertLessEqual(difference, math.radians(4))
        self.assertGreaterEqual(result['size'][0], 3.4)
        self.assertEqual(result['quality'], 'estimated')

    def test_rejected_top_candidate_does_not_hide_safe_candidates_after_rank_47(self):
        result = self.fit('object-000008-car-0')
        self.assertTrue(np.isfinite(result['center']).all())
        self.assertGreaterEqual(result['size'][0], 2.6)

    def test_dense_end_cap_does_not_let_prior_choose_yaw_and_hidden_length(self):
        result = self.fit('object-000008-car-5')
        self.assertEqual(result['diagnostics']['estimateEvidence'],'dense-end-cap-hidden-body')
        self.assertLess(result['size'][0],3.)
        self.assertEqual(result['quality'],'estimated')
        self.assertIn('bound-limited',result['diagnostics']['dimensionSourceLWH'][0])

    def test_real_geometry_and_provenance_are_json_serializable(self):
        result = self.fit('object-000002-car-1')
        decoded = json.loads(json.dumps(result,allow_nan=False))
        self.assertEqual(decoded['size'],result['size'])


if __name__ == '__main__':
    unittest.main()
