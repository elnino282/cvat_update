# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

"""Compact creation provenance for Auto Fit annotations (never fitting diagnostics)."""

AUTO_FIT_METADATA_KEYS = (
    "version", "request_id", "frame", "class_key", "quality", "fit_mode",
    "yaw_uncertain", "prototype_key", "prior_version", "experimental",
)


def validate_auto_fit_metadata(value):
    """Accept legacy/manual {}, or the complete, JSON-safe v1 creation record."""
    if not isinstance(value, dict):
        raise ValueError("auto_fit must be an object")
    if not value:
        return {}
    if set(value) != set(AUTO_FIT_METADATA_KEYS):
        raise ValueError("auto_fit must contain exactly the version 1 provenance fields")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("auto_fit.version must be 1")
    if type(value["frame"]) is not int or not 0 <= value["frame"] <= 2147483647:
        raise ValueError("auto_fit.frame must be a non-negative integer")
    for key in ("request_id", "class_key", "prototype_key", "prior_version"):
        item = value[key]
        if item is None and key in ("prototype_key", "prior_version"):
            continue
        if not isinstance(item, str) or not 0 < len(item) <= 128:
            raise ValueError(f"auto_fit.{key} must be a non-empty string of at most 128 characters")
        if any(ord(char) < 32 for char in item):
            raise ValueError(f"auto_fit.{key} must not contain control characters")
    if value["quality"] not in ("confident", "estimated"):
        raise ValueError("auto_fit.quality must be confident or estimated")
    if value["fit_mode"] not in ("geometry", "hybrid_prior"):
        raise ValueError("auto_fit.fit_mode must be geometry or hybrid_prior")
    for key in ("yaw_uncertain", "experimental"):
        if type(value[key]) is not bool:
            raise ValueError(f"auto_fit.{key} must be boolean")
    return {key: value[key] for key in AUTO_FIT_METADATA_KEYS}
