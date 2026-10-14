# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

"""Focused adapter tests; no test database or annotation writes required."""

import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
from django.test import SimpleTestCase, override_settings
from rest_framework.serializers import ValidationError

from cvat.apps.engine.auto_fit import AutoFitRequestSerializer, cancel_job_fit, fit_job_frame, prepare_job_frame
from cvat.apps.engine.models import FrameQuality
from cvat.apps.engine.permissions import JobPermission


@override_settings(
    CVAT_AUTOFIT_ENABLED=True,
    CVAT_AUTOFIT_BACKEND="geometry-only",
    CVAT_AUTOFIT_SERVICE_URL="http://vehicle_fit:8093",
)
class AutoFitAdapterTests(SimpleTestCase):
    def setUp(self):
        self.data = {
            "requestId": "74f95c66-04e5-4fea-ac39-38d80d0d49ba",
            "requestedAt": 1000,
            "taskId": 1, "jobId": 2, "frame": 3, "labelId": 4, "objectType": "shape",
            "seed": {"x": 5.0, "y": 6.0, "z": 7.0, "pointIndex": 0},
        }
        self.request = SimpleNamespace(data=self.data, user=SimpleNamespace(id=7))
        self.task = SimpleNamespace(id=1, dimension="3d", data=SimpleNamespace(deleted_frames=[]))
        self.job = SimpleNamespace(
            id=2, segment=SimpleNamespace(task=self.task, task_id=1),
            get_labels=lambda: [SimpleNamespace(id=4, name="Car")],
        )
        self.result = {
            "status": "ok", "requestId": self.data["requestId"], "frame": 3,
            "provider": "geometry-only", "center": [5.0, 6.0, 7.0],
            "size": [4.2, 1.8, 1.6], "rotation": [0, 0, 1.2], "confidence": 0.8,
        }
        self.provider_patch = patch("cvat.apps.engine.auto_fit.JobFrameProvider")
        self.provider = self.provider_patch.start().return_value
        self.addCleanup(self.provider_patch.stop)
        self.provider.get_frame.return_value = SimpleNamespace(data=BytesIO(b"actual-pcd"))
        self.post_patch = patch("cvat.apps.engine.auto_fit.requests.post")
        self.post = self.post_patch.start()
        self.addCleanup(self.post_patch.stop)
        self.post.return_value = MagicMock(status_code=200)
        self.post.return_value.json.return_value = self.result

    def test_permission_requires_frame_read_and_annotation_write(self):
        for action in ["auto_fit", "cancel_auto_fit", "prepare_auto_fit"]:
            self.assertEqual(
                JobPermission._get_scopes(self.request, SimpleNamespace(action=action), self.job),
                [JobPermission.Scopes.VIEW_DATA, JobPermission.Scopes.UPDATE_ANNOTATIONS],
            )

    def test_reads_matching_job_pcd_and_only_returns_service_geometry(self):
        response = fit_job_frame(self.request, self.job)
        self.assertEqual(response.data, self.result)
        self.provider.validate_frame_number.assert_called_once_with(3)
        self.assertEqual(self.provider.get_frame.call_args.kwargs["quality"], FrameQuality.COMPRESSED)
        self.provider.unload.assert_called_once()
        upload = self.post.call_args.kwargs
        self.assertEqual(upload["files"]["cloud"][1], b"actual-pcd")
        metadata = json.loads(upload["data"]["metadata"])
        self.assertEqual(metadata["seed"], self.data["seed"])
        self.assertEqual(metadata["userId"], 7)
        self.assertEqual(metadata["cloud_format"], "pcd")
        self.assertEqual(metadata["labelName"], "Car")
        self.assertLessEqual(metadata["timeout_ms"], 16000)

    def test_feature_off_never_reads_cloud_or_calls_service(self):
        with override_settings(CVAT_AUTOFIT_ENABLED=False):
            self.assertEqual(fit_job_frame(self.request, self.job).status_code, 503)
        self.provider.get_frame.assert_not_called()
        self.post.assert_not_called()

    def test_rejects_wrong_identity_label_deleted_and_out_of_job_frames(self):
        for field in ["taskId", "jobId", "labelId"]:
            original = self.data[field]
            self.data[field] = 99
            with self.assertRaises(ValidationError):
                fit_job_frame(self.request, self.job)
            self.data[field] = original
        self.job.get_labels = lambda: [SimpleNamespace(id=4, name="Pedestrian")]
        with self.assertRaises(ValidationError):
            fit_job_frame(self.request, self.job)
        self.job.get_labels = lambda: [SimpleNamespace(id=4, name="Car")]
        self.task.data.deleted_frames = [3]
        with self.assertRaises(ValidationError):
            fit_job_frame(self.request, self.job)
        self.task.data.deleted_frames = []
        self.provider.validate_frame_number.side_effect = ValidationError("out of job")
        with self.assertRaises(ValidationError):
            fit_job_frame(self.request, self.job)
        self.provider.get_frame.assert_not_called()
        self.post.assert_not_called()

    def test_serializer_rejects_nonfinite_seed_invalid_uuid_and_type(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            data = {**self.data, "seed": {**self.data["seed"], "x": value}}
            self.assertFalse(AutoFitRequestSerializer(data=data).is_valid())
        for override in [{"requestId": "old-id"}, {"objectType": "tag"}, {"frame": -1}]:
            self.assertFalse(AutoFitRequestSerializer(data={**self.data, **override}).is_valid())

    def test_explicit_timeout_is_bounded_and_independent_of_policy(self):
        for value in (0, -1, 16001):
            self.assertFalse(AutoFitRequestSerializer(data={**self.data, 'timeout_ms': value}).is_valid())
        for allow in (False, True):
            self.data.update(allowEstimated=allow, timeout_ms=8000)
            fit_job_frame(self.request, self.job)
            forwarded = json.loads(self.post.call_args.kwargs['data']['metadata'])
            self.assertGreater(forwarded['timeout_ms'], 7000)
            self.assertLessEqual(forwarded['timeout_ms'], 8000)
        self.data.pop('timeout_ms')
        fit_job_frame(self.request, self.job)
        forwarded = json.loads(self.post.call_args.kwargs['data']['metadata'])
        self.assertLessEqual(forwarded['timeout_ms'], 1700)

    @override_settings(CVAT_AUTOFIT_BACKEND='snap')
    def test_strict_fallback_rejection_is_valid_no_fit(self):
        rejection = {'status': 'no-fit', 'requestId': self.data['requestId'], 'frame': 3,
                     'provider': 'snap', 'reason': 'insufficient-evidence',
                     'diagnostics': {'attemptedProvider': 'geometry-only', 'assessedQuality': 'estimated'}}
        self.post.return_value.json.return_value = rejection
        self.assertEqual(fit_job_frame(self.request, self.job).data, rejection)

    def test_invalid_geometry_and_wrong_provenance_create_no_fit(self):
        for override in [
            {"center": [float("nan"), 0, 0]}, {"size": [-1, 1, 1]},
            {"rotation": [0, 0]}, {"provider": "snap"}, {"frame": 99},
            {"requestId": "wrong"}, {"confidence": 2},
        ]:
            self.post.return_value.json.return_value = {**self.result, **override}
            response = fit_job_frame(self.request, self.job)
            self.assertEqual(response.data["status"], "no-fit")
            self.assertEqual(response.data["reason"], "invalid-geometry")

    def test_timeout_unavailable_and_idempotency_conflict(self):
        for error, code, reason in [
            (requests.Timeout(), 504, "timeout"), (requests.ConnectionError(), 503, "unavailable"),
        ]:
            self.post.side_effect = error
            response = fit_job_frame(self.request, self.job)
            self.assertEqual(response.status_code, code)
            self.assertEqual(response.data["reason"], reason)
        self.post.side_effect = None
        self.post.return_value.status_code = 409
        self.assertEqual(fit_job_frame(self.request, self.job).status_code, 409)

    def test_cancellation_forwards_only_authenticated_scope(self):
        with patch("cvat.apps.engine.auto_fit.requests.delete") as delete:
            delete.return_value.status_code = 204
            response = cancel_job_fit(self.request, self.job, self.data["requestId"])
            self.assertEqual(response.status_code, 204)
            self.assertEqual(delete.call_args.kwargs["params"], {"job_id": 2, "task_id": 1, "user_id": 7})

    def test_prepared_fit_validates_frame_but_does_not_read_or_upload_pcd(self):
        self.data.update(frameToken="a" * 32, allowEstimated=True)
        response = fit_job_frame(self.request, self.job)
        self.assertEqual(response.data, self.result)
        self.provider.validate_frame_number.assert_called_once_with(3)
        self.provider.get_frame.assert_not_called()
        payload = self.post.call_args.kwargs
        self.assertNotIn("files", payload)
        self.assertEqual(payload["json"]["frameToken"], "a" * 32)
        self.assertLessEqual(payload["json"]["timeout_ms"], 1700)

    def test_estimate_requires_opt_in_and_explicit_fallback_provider(self):
        estimate = {**self.result, "quality": "estimated"}
        self.post.return_value.json.return_value = estimate
        self.assertEqual(fit_job_frame(self.request, self.job).data["reason"], "invalid-geometry")
        self.data["allowEstimated"] = True
        with override_settings(CVAT_AUTOFIT_BACKEND="snap"):
            self.assertEqual(fit_job_frame(self.request, self.job).data, estimate)
            self.post.return_value.json.return_value = {**estimate, "quality": "confident"}
            self.assertEqual(fit_job_frame(self.request, self.job).data["reason"], "invalid-geometry")

    def test_prepare_and_status_forward_authenticated_scope_and_data_version(self):
        self.job.segment.chunks_updated_date = "v1"
        ready = {"state": "ready", "frame": 3, "frameToken": "a" * 32, "backend": "geometry-only"}
        self.post.return_value.json.return_value = ready
        request = SimpleNamespace(method="POST", data={"taskId": 1, "jobId": 2, "frame": 3},
                                  user=SimpleNamespace(id=7))
        self.assertEqual(prepare_job_frame(request, self.job).data, ready)
        metadata = json.loads(self.post.call_args.kwargs["data"]["metadata"])
        self.assertEqual(metadata["userId"], 7)
        self.assertEqual(metadata["dataVersion"], "v1")
        self.provider.get_frame.reset_mock()
        request.method = "GET"
        request.query_params = {**request.data, "frameToken": "a" * 32}
        with patch("cvat.apps.engine.auto_fit.requests.get") as get:
            get.return_value = MagicMock(status_code=200)
            get.return_value.json.return_value = ready
            self.assertEqual(prepare_job_frame(request, self.job).data, ready)
            self.assertEqual(get.call_args.kwargs["params"]["userId"], 7)
        self.provider.get_frame.assert_not_called()
