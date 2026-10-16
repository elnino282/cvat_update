"""Focused semantic routing contracts; no database or real inference writes."""
import json
from types import SimpleNamespace

from django.test import SimpleTestCase, override_settings
from rest_framework.serializers import ValidationError

from cvat.apps.engine.auto_fit import fit_job_frame
from cvat.apps.engine.tests import test_auto_fit as base_tests


@override_settings(
    CVAT_AUTOFIT_ENABLED=True, CVAT_AUTOFIT_BACKEND='geometry-only',
    CVAT_AUTOFIT_EXPERIMENTAL_CLASSES='pedestrian,vehicles/truck,vehicles/bus,vehicles/van,two-wheels',
)
class MultiClassAdapterTests(SimpleTestCase):
    setUp = base_tests.AutoFitAdapterTests.setUp
    def label(self, name, attributes=()):
        self.job.get_labels = lambda: [SimpleNamespace(id=4, name=name, attributes=list(attributes))]

    def success(self, key):
        self.data['allowEstimated'] = True
        self.result.update(quality='estimated', autoFit={
            'version': 1, 'request_id': self.data['requestId'], 'frame': 3,
            'class_key': key, 'quality': 'estimated', 'fit_mode': 'hybrid_prior',
            'yaw_uncertain': True, 'prototype_key': 'dev', 'prior_version': 'pr0', 'experimental': True,
        })

    def test_pedestrian_semantic_route_preserves_empty_attributes(self):
        self.label('pedestrian')
        self.success('pedestrian')
        response = fit_job_frame(self.request, self.job)
        self.assertEqual(response.data['status'], 'ok')
        sent = json.loads(self.post.call_args.kwargs['data']['metadata'])
        self.assertEqual(sent['classKey'], 'pedestrian')
        self.assertNotIn('vehicleType', sent)
        self.assertNotIn('age', sent)
        self.assertNotIn('pose', sent)

    def test_two_wheels_runtime_checkbox_identity_and_boolean(self):
        self.label('two-wheels', [SimpleNamespace(id=819, name='ride_state', input_type='checkbox', mutable=False)])
        self.data.update(rideState=True, rideStateAttributeId=819)
        self.success('two-wheels')
        self.assertEqual(fit_job_frame(self.request, self.job).data['status'], 'ok')
        sent = json.loads(self.post.call_args.kwargs['data']['metadata'])
        self.assertIs(sent['rideState'], True)
        self.assertEqual(sent['classKey'], 'two-wheels')
        self.post.reset_mock()
        self.data['rideStateAttributeId'] = 820
        with self.assertRaises(ValidationError):
            fit_job_frame(self.request, self.job)
        self.post.assert_not_called()

    def test_vehicle_routes_are_distinct_and_other_never_reads_frame(self):
        self.label('vehicles', [SimpleNamespace(id=763, name='type', input_type='select', mutable=False,
                                              values=['car', 'truck', 'bus', 'van', 'other'])])
        for kind in ('truck', 'bus', 'van'):
            self.data.update(vehicleType=kind, typeAttributeId=763)
            self.success('vehicles/' + kind)
            self.assertEqual(fit_job_frame(self.request, self.job).data['status'], 'ok')
            sent = json.loads(self.post.call_args.kwargs['data']['metadata'])
            self.assertEqual(sent['vehicleType'], kind)
        self.post.reset_mock()
        self.provider.get_frame.reset_mock()
        self.data['vehicleType'] = 'other'
        self.assertEqual(fit_job_frame(self.request, self.job).data['reason'], 'unsupported-type')
        self.post.assert_not_called()
        self.provider.get_frame.assert_not_called()

    @override_settings(CVAT_AUTOFIT_EXPERIMENTAL_CLASSES='')
    def test_disabled_experiment_stops_before_cloud_and_inference(self):
        self.label('pedestrian')
        self.assertEqual(fit_job_frame(self.request, self.job).data['reason'], 'unsupported-type')
        self.post.assert_not_called()
        self.provider.get_frame.assert_not_called()

    def test_wrong_metadata_identity_and_unsupported_coordinates_are_rejected(self):
        self.label('pedestrian')
        self.success('pedestrian')
        self.result['autoFit']['frame'] = 7
        self.assertEqual(fit_job_frame(self.request, self.job).data['reason'], 'invalid-geometry')
        self.post.reset_mock()
        self.data['units'] = 'millimetres'
        with self.assertRaises(ValidationError):
            fit_job_frame(self.request, self.job)
        self.post.assert_not_called()

