import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vehicle_fit.evaluation.recovery import assess_stage, associate, verify_policy
from vehicle_fit.evaluation.metrics import Cuboid


class RecoveryProtocolTests(unittest.TestCase):
    def row(self,case='a',iou=.9,error=.1):
        return {'case_id':case,'status':'ok','correct_vehicle':True,'seed_inside':True,
                'metrics':{'iou_3d':iou,'dimension_error_m_lwh':[error,.1,.1]},
                'response':{'size':[3.7,1.61,1.49]},'counts':{'completed':True}}

    def policy(self):
        return {'numeric_epsilon':1e-6,'minimum_iou':.7,'maximum_protected_iou_drop':.02,
                'minimum_correct_boxes':1,'minimum_good_boxes':1}

    def test_iou_90_to_71_fails_even_when_a_new_good_case_replaces_it(self):
        baseline=[self.row()]
        protected={'cases':{'a':{'reference_iou':.9}}}
        assessed=assess_stage([self.row(iou=.71),self.row('new')],baseline,protected,self.policy())
        self.assertFalse(assessed['pass'])
        self.assertFalse(assessed['gates']['protected_cases'])
        self.assertAlmostEqual(assessed['protected'][0]['minimum_allowed_iou'],.88)

    def test_exact_frozen_drop_limit_is_allowed_and_smaller_drop_is_reported(self):
        baseline=[self.row()]
        protected={'cases':{'a':{'reference_iou':.9}}}
        assessed=assess_stage([self.row(iou=.88)],baseline,protected,self.policy())
        self.assertTrue(assessed['pass'])
        self.assertEqual(len(assessed['changes_vs_baseline_p0']),1)
        self.assertFalse(assess_stage([self.row(iou=.879)],baseline,protected,self.policy())['pass'])

    def test_missing_reference_case_cannot_improve_paired_error_by_removal(self):
        baseline=[self.row(),self.row('hard',error=1.)]
        assessed=assess_stage([self.row(error=.01)],baseline,{'cases':{'a':{'reference_iou':.9}}},self.policy())
        self.assertFalse(assessed['gates']['paired_length'])
        self.assertEqual(assessed['lost_reference_case_ids'],['hard'])

    def test_policy_checksum_prevents_threshold_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.json'
            path.write_text(json.dumps(self.policy()))
            path.with_suffix('.sha256').write_text('incorrect')
            with self.assertRaisesRegex(ValueError,'policy changed'):
                verify_policy(path)

    def test_target_neighbor_tie_is_not_correct_instance_and_does_not_modify_geometry(self):
        result={'center':[0,0,0],'size':[4,2,2],'rotation':[0,0,0]}
        before=json.dumps(result)
        box=Cuboid(np.zeros(3),np.array([4.,2.,2.]),np.eye(3))
        neighbors=[{'id':0,'class':'Car','box':box},{'id':1,'class':'Van','box':box}]
        row=associate(result,np.zeros(3),{'label':{'line_index':0}},neighbors,1e-6)
        self.assertFalse(row['correct_vehicle'])
        self.assertEqual(row['association']['classification'],'tie')
        self.assertEqual(json.dumps(result),before)
