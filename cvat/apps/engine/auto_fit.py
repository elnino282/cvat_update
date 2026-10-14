# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

"""Authenticated CVAT adapter for the isolated point-cloud fitting service.

The adapter only reads a job frame and returns geometry. Annotation creation and
history remain in the existing canvas pipeline.
"""

import json
import logging
import math
import time

import requests
from django.conf import settings
from rest_framework import serializers, status
from rest_framework.response import Response

from cvat.apps.engine.media_io.frame_provider import FrameOutputType, JobFrameProvider
from cvat.apps.engine.models import DimensionType, FrameQuality

logger = logging.getLogger(__name__)


class FiniteFloatField(serializers.FloatField):
    def to_internal_value(self, data):
        value = super().to_internal_value(data)
        if not math.isfinite(value):
            self.fail("invalid")
        return value


class AutoFitSeedSerializer(serializers.Serializer):
    x = FiniteFloatField()
    y = FiniteFloatField()
    z = FiniteFloatField()
    pointIndex = serializers.IntegerField(min_value=0)


class AutoFitRequestSerializer(serializers.Serializer):
    requestId = serializers.UUIDField()
    requestedAt = FiniteFloatField(min_value=0)
    taskId = serializers.IntegerField(min_value=1)
    jobId = serializers.IntegerField(min_value=1)
    frame = serializers.IntegerField(min_value=0)
    labelId = serializers.IntegerField(min_value=1)
    objectType = serializers.ChoiceField(choices=["shape", "track"])
    seed = AutoFitSeedSerializer()
    frameToken = serializers.RegexField(r"^[0-9a-f]{32}$", required=False)
    allowEstimated = serializers.BooleanField(default=False)
    timeout_ms = serializers.IntegerField(min_value=1, max_value=16000, default=1700)


class AutoFitPrepareSerializer(serializers.Serializer):
    taskId = serializers.IntegerField(min_value=1)
    jobId = serializers.IntegerField(min_value=1)
    frame = serializers.IntegerField(min_value=0)
    frameToken = serializers.RegexField(r"^[0-9a-f]{32}$", required=False)


def _no_fit(data, reason, *, response_status=status.HTTP_200_OK):
    return Response(
        {
            "status": "no-fit",
            "requestId": str(data["requestId"]),
            "frame": data["frame"],
            "reason": reason,
            "provider": settings.CVAT_AUTOFIT_BACKEND,
        },
        status=response_status,
    )


def _valid_result(result, metadata):
    if not isinstance(result, dict) or (
        result.get("requestId") != metadata["requestId"]
        or result.get("frame") != metadata["frame"]
    ):
        return False
    fallback = (metadata.get("allowEstimated") is True and settings.CVAT_AUTOFIT_BACKEND == "snap"
                and result.get("provider") == "geometry-only" and result.get("quality") == "estimated")
    if result.get("provider") != settings.CVAT_AUTOFIT_BACKEND and not fallback:
        return False
    if result.get("status") == "no-fit":
        return result.get("reason") in {
            "no-points", "ambiguous", "insufficient-evidence", "unavailable", "timeout",
            "cancelled", "invalid-geometry", "invalid-seed",
            "frame-expired",
        }
    if result.get("status") != "ok":
        return False
    if result.get("quality", "confident") not in {"confident", "estimated"}:
        return False
    if result.get("quality") == "estimated" and not metadata.get("allowEstimated"):
        return False
    for name in ["center", "rotation", "size"]:
        value = result.get(name)
        if not isinstance(value, list) or len(value) != 3 or not all(
            isinstance(n, (float, int)) and not isinstance(n, bool) and math.isfinite(n)
            for n in value
        ):
            return False
    if any(n <= 0 for n in result["size"]):
        return False
    confidence = result.get("confidence")
    return confidence is None or (
        isinstance(confidence, (float, int))
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
        and 0 <= confidence <= 1
    )


def fit_job_frame(request, job):
    if not settings.CVAT_AUTOFIT_ENABLED:
        return Response({"detail": "Auto Fit is disabled"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    started = time.monotonic()
    serializer = AutoFitRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    metadata = dict(serializer.validated_data)
    metadata["requestId"] = str(metadata["requestId"])
    task = job.segment.task
    if metadata["taskId"] != task.id or metadata["jobId"] != job.id:
        raise serializers.ValidationError("Auto Fit identity does not match this job")
    if task.dimension != DimensionType.DIM_3D:
        raise serializers.ValidationError("Auto Fit requires a 3D job")
    if metadata["frame"] in task.data.deleted_frames:
        raise serializers.ValidationError("Auto Fit frame has been deleted")
    label = next((label for label in job.get_labels() if label.id == metadata["labelId"]), None)
    if label is None or label.name.strip().lower() != "car":
        raise serializers.ValidationError("Auto Fit requires the Car label from this job")
    provider = JobFrameProvider(job)
    try:
        provider.validate_frame_number(metadata["frame"])
        # Match the compressed PCD rendered by the browser, preserving seed coordinates.
        cloud = None
        if not metadata.get("frameToken"):
            frame = provider.get_frame(
                metadata["frame"], quality=FrameQuality.COMPRESSED, out_type=FrameOutputType.BUFFER
            )
            cloud = frame.data.getvalue()
    finally:
        provider.unload()
    remaining = metadata["timeout_ms"] / 1000 - (time.monotonic() - started)
    if remaining <= 0:
        return _no_fit(metadata, "timeout", response_status=status.HTTP_504_GATEWAY_TIMEOUT)
    metadata.update(
        cloud_format="pcd",
        labelName=label.name,
        userId=request.user.id,
        timeout_ms=int(remaining * 1000),
        dataVersion=str(getattr(job.segment, "chunks_updated_date", "")),
    )
    try:
        payload = {"json": metadata} if cloud is None else {
            "data": {"metadata": json.dumps(metadata, allow_nan=False)},
            "files": {"cloud": ("frame.pcd", cloud, "application/octet-stream")},
        }
        response = requests.post(
            settings.CVAT_AUTOFIT_SERVICE_URL + "/v1/fit",
            **payload,
            timeout=(min(.2, remaining), remaining),
        )
        if response.status_code == status.HTTP_409_CONFLICT:
            return Response({"detail": "Auto Fit request ID conflicts with another payload"}, status=409)
        response.raise_for_status()
        result = response.json()
        if not _valid_result(result, metadata):
            logger.warning("Auto Fit invalid response request=%s job=%s", metadata["requestId"], job.id)
            return _no_fit(metadata, "invalid-geometry")
        logger.info(
            "Auto Fit request=%s job=%s frame=%s status=%s elapsed_ms=%.1f",
            metadata["requestId"], job.id, metadata["frame"], result["status"],
            (time.monotonic() - started) * 1000,
        )
        return Response(result)
    except requests.Timeout:
        # Runtime uses a monotonic deadline, including its GPU queue.
        return _no_fit(metadata, "timeout", response_status=status.HTTP_504_GATEWAY_TIMEOUT)
    except (requests.RequestException, ValueError):
        logger.warning("Auto Fit service unavailable request=%s job=%s", metadata["requestId"], job.id)
        return _no_fit(metadata, "unavailable", response_status=status.HTTP_503_SERVICE_UNAVAILABLE)


def prepare_job_frame(request, job):
    if not settings.CVAT_AUTOFIT_ENABLED:
        return Response({"detail": "Auto Fit is disabled"}, status=503)
    serializer = AutoFitPrepareSerializer(data=request.query_params if request.method == "GET" else request.data)
    serializer.is_valid(raise_exception=True)
    metadata = dict(serializer.validated_data)
    task = job.segment.task
    if metadata["taskId"] != task.id or metadata["jobId"] != job.id or task.dimension != DimensionType.DIM_3D:
        raise serializers.ValidationError("Auto Fit requires the matching 3D job")
    if metadata["frame"] in task.data.deleted_frames:
        raise serializers.ValidationError("Auto Fit frame has been deleted")
    if not any(label.name.strip().lower() == "car" for label in job.get_labels()):
        raise serializers.ValidationError("Auto Fit requires a Car label")
    metadata.update(userId=request.user.id, cloud_format="pcd",
                    dataVersion=str(getattr(job.segment, "chunks_updated_date", "")))
    provider = JobFrameProvider(job)
    try:
        provider.validate_frame_number(metadata["frame"])
        if request.method == "GET":
            if not metadata.get("frameToken"):
                raise serializers.ValidationError("Prepared frame token is required")
            response = requests.get(settings.CVAT_AUTOFIT_SERVICE_URL + "/v1/prepare",
                                    params=metadata, timeout=(.2, 2))
        else:
            frame = provider.get_frame(metadata["frame"], quality=FrameQuality.COMPRESSED,
                                       out_type=FrameOutputType.BUFFER)
            response = requests.post(
                settings.CVAT_AUTOFIT_SERVICE_URL + "/v1/prepare",
                data={"metadata": json.dumps(metadata, allow_nan=False)},
                files={"cloud": ("frame.pcd", frame.data.getvalue(), "application/octet-stream")},
                timeout=(.2, 5),
            )
        if response.status_code == 410:
            return Response({"detail": "Prepared frame expired"}, status=410)
        response.raise_for_status()
        result = response.json()
        if (not isinstance(result, dict) or result.get("frame") != metadata["frame"]
                or result.get("state") not in {"preparing", "ready", "unavailable"}
                or not isinstance(result.get("frameToken"), str)
                or len(result["frameToken"]) != 32 or result.get("backend") != settings.CVAT_AUTOFIT_BACKEND):
            return Response({"detail": "Invalid preparation response"}, status=502)
        return Response(result, status=response.status_code)
    except (requests.RequestException, ValueError):
        return Response({"detail": "Auto Fit preparation unavailable"}, status=503)
    finally:
        provider.unload()


def cancel_job_fit(request, job, request_id):
    if not settings.CVAT_AUTOFIT_ENABLED:
        return Response(status=status.HTTP_204_NO_CONTENT)
    try:
        response = requests.delete(
            settings.CVAT_AUTOFIT_SERVICE_URL + f"/v1/requests/{request_id}",
            params={"job_id": job.id, "task_id": job.segment.task_id, "user_id": request.user.id},
            timeout=(1, 1),
        )
        if response.status_code == status.HTTP_409_CONFLICT:
            return Response({"detail": "Auto Fit cancellation scope mismatch"}, status=409)
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("Auto Fit cancellation unavailable request=%s job=%s", request_id, job.id)
    return Response(status=status.HTTP_204_NO_CONTENT)
