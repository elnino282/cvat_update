"""Read-only Auto Fit regression against the supplied CVAT KITTI drive_0005 GT.

The source ZIP and Datumaro annotation are never modified. Seeds contain only
observed points; GT geometry is used by this evaluator only for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

import numpy as np

from ..cloud import decode_cloud
from .manifest import point_seeds, sha256 as file_sha256
from .metrics import Cuboid, errors, euler_xyz_matrix, oriented_iou


INFRASTRUCTURE_REASONS = {'timeout', 'unavailable', 'cancelled', 'frame-expired'}


def compare_routes(current, legacy):
    """None means no comparable algorithm result, not successful parity."""
    for response in (current, legacy):
        if response.get('status') not in {'ok', 'no-fit'} or (
            response.get('status') == 'no-fit' and response.get('reason') in INFRASTRUCTURE_REASONS
        ):
            return None
    for field in ('status', 'provider', 'quality'):
        if current.get(field) != legacy.get(field):
            return False
    if current['status'] == 'no-fit':
        return bool(current.get('reason')) and current['reason'] == legacy.get('reason')
    new = current.get('cuboid') or current.get('geometry') or current
    old = legacy.get('cuboid') or legacy.get('geometry') or legacy
    return all(np.allclose(new[key], old[key], rtol=0, atol=1e-6)
               for key in ('center', 'size', 'rotation'))


def correct_instance(scores, target_index):
    target = scores[target_index]
    return bool(target > 0 and all(target > score + 1e-9
                                  for index, score in enumerate(scores) if index != target_index))


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dataset_manifest(dataset: Path) -> dict:
    """Freeze source hashes, GT boxes, and observed-only seeds in memory."""
    dataset = dataset.expanduser().resolve()
    cloud_archive = dataset / "cvat_data_3d.zip"
    annotation_path = dataset / "annotations/annotations/default.json"
    tracklet_path = dataset / "metadata/tracklet_labels.xml"
    annotations = json.loads(annotation_path.read_text())
    labels = annotations["categories"]["label"]["labels"]
    cases_by_frame: dict[str, list[dict]] = {}
    frames = {}
    with zipfile.ZipFile(cloud_archive) as archive:
        members = set(archive.namelist())
        for item in annotations["items"]:
            frame = Path(item["id"]).name
            frame_name = f"pointcloud/{frame}.pcd"
            if frame_name not in members:
                raise ValueError(f"Point cloud missing from source archive: {frame_name}")
            payload = archive.read(frame_name)
            digest = sha256(payload)
            frames[frame] = {
                "archive": str(cloud_archive),
                "asset": frame_name,
                "sha256": digest,
                "cloud_format": "pcd",
            }
            cloud, original_indices = decode_cloud(payload, "pcd")
            seen_annotation_ids = set()
            for shape in item["annotations"]:
                if shape["type"] != "cuboid_3d":
                    continue
                attributes = shape.get("attributes", {})
                annotation_id = shape.get("id")
                if annotation_id in seen_annotation_ids:
                    raise ValueError(f"Duplicate annotation ID in frame {frame}: {annotation_id}")
                seen_annotation_ids.add(annotation_id)
                label_id = int(shape["label_id"])
                label_name = labels[label_id]["name"]
                if attributes.get("outside"):
                    continue
                if label_name not in {"Car", "Van", "Truck", "Bus"}:
                    continue
                truth = Cuboid(
                    np.asarray(shape["position"], dtype=float),
                    np.asarray(shape["scale"], dtype=float),
                    euler_xyz_matrix(shape["rotation"]),
                )
                seeds, gt_point_count = point_seeds(cloud, truth)
                for seed in seeds:
                    seed['pointIndex'] = int(original_indices[seed['pointIndex']])
                track_id = int(attributes.get("track_id", -1))
                kitti_track_id = attributes.get("kitti_track_id")
                case = {
                    "case_id": f"drive0005-frame-{frame}-annotation-{annotation_id}",
                    "frame": frame,
                    "annotation_id": annotation_id,
                    "track_id": track_id,
                    "kitti_track_id": kitti_track_id,
                    "class": label_name,
                    "gt_center": truth.center.tolist(),
                    "gt_size": truth.size.tolist(),
                    "gt_rotation": shape["rotation"],
                    "gt_point_count": gt_point_count,
                    "seeds": seeds,
                }
                cases_by_frame.setdefault(frame, []).append(case)
    tracklet_hash = sha256(tracklet_path.read_bytes())
    cases = [case for frame in sorted(cases_by_frame) for case in cases_by_frame[frame]]
    return {
        "version": 1,
        "corpus": "CVAT KITTI drive_0005 154-frame Datumaro 3D sample",
        "dataset_path": str(dataset),
        "cloud_archive_sha256": file_sha256(cloud_archive),
        "annotation_sha256": sha256(annotation_path.read_bytes()),
        "tracklet_sha256": tracklet_hash,
        "seed_policy": "first observed point nearest GT-contained point median; GT geometry is never sent to fitter",
        "visible_cases_by_class": {
            name: sum(case["class"] == name for case in cases)
            for name in sorted({case["class"] for case in cases})
        },
        "frames": frames,
        "cases": cases,
    }


def load_cloud(manifest: dict, frame: str) -> tuple[bytes, np.ndarray]:
    asset = manifest["frames"][frame]
    with zipfile.ZipFile(asset["archive"]) as archive:
        payload = archive.read(asset["asset"])
    if sha256(payload) != asset["sha256"]:
        raise ValueError(f"Frozen cloud changed: {frame}")
    cloud, _ = decode_cloud(payload, "pcd")
    return payload, cloud


def request(endpoint: str, method: str, path: str, metadata: dict, payload: bytes | None = None):
    if method == "GET":
        path += "?" + urllib.parse.urlencode(metadata)
        body, headers = None, {}
    elif payload is None:
        body = json.dumps(metadata, allow_nan=False).encode()
        headers = {"Content-Type": "application/json"}
    else:
        boundary = "autofit-" + uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\n".encode()
            + json.dumps(metadata, allow_nan=False).encode()
            + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"cloud\"; filename=\"frame.pcd\"\r\n".encode()
            + b"Content-Type: application/octet-stream\r\n\r\n"
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(endpoint + path, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def prepare(endpoint: str, meta: dict, cloud_bytes: bytes) -> dict:
    state = request(endpoint, "POST", "/v1/prepare", meta, cloud_bytes)
    deadline = time.monotonic() + 45
    while state["state"] == "preparing" and time.monotonic() < deadline:
        time.sleep(0.2)
        status_meta = {key: meta[key] for key in ("userId", "taskId", "jobId", "frame", "dataVersion")}
        status_meta["frameToken"] = state["frameToken"]
        state = request(endpoint, "GET", "/v1/prepare", status_meta)
    return state


def fit_request(endpoint: str, meta: dict, frame_token: str, seed: dict, label_name: str, vehicle_type: str) -> dict:
    request_meta = {
        **meta,
        "frameToken": frame_token,
        "requestId": str(uuid.uuid4()),
        "labelName": label_name,
        "vehicleType": vehicle_type,
        "seed": {"x": seed["point"][0], "y": seed["point"][1], "z": seed["point"][2],
                 "pointIndex": seed["pointIndex"]},
        "timeout_ms": 1700,
        "allowEstimated": True,
    }
    return request(endpoint, "POST", "/v1/fit", request_meta)


def run(args) -> dict:
    if args.limit < 0 or args.paired_legacy < 0:
        raise ValueError('Case and pair limits must be nonnegative')
    dataset = Path(args.dataset)
    manifest = dataset_manifest(dataset)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    selected = [case for case in manifest["cases"] if case["class"] == "Car"]
    if args.limit:
        selected = selected[:args.limit]
    pairs_remaining = args.paired_legacy
    rows = []
    output_identity = {
        "dataset": manifest["corpus"],
        "annotation_sha256": manifest["annotation_sha256"],
        "cloud_archive_sha256": manifest["cloud_archive_sha256"],
        "tracklet_sha256": manifest["tracklet_sha256"],
        "endpoint": args.endpoint,
        "backend": None,
        "health": None,
        "expected_cases": len(selected),
        "completed_cases": 0,
        "complete": False,
    }
    (output / "run.json").write_text(json.dumps(output_identity, indent=2) + "\n")
    with (output / "cases.jsonl").open("w") as log:
        for frame in sorted({case["frame"] for case in selected}):
            frame_cases = [case for case in selected if case["frame"] == frame]
            cloud_bytes, _ = load_cloud(manifest, frame)
            frame_meta = {
                "userId": 700001,
                "taskId": 700001,
                "jobId": 700001,
                "frame": int(frame),
                "cloud_format": "pcd",
                "dataVersion": manifest["frames"][frame]["sha256"],
            }
            try:
                prepared = prepare(args.endpoint, frame_meta, cloud_bytes)
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                prepared = {"state": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}
            if output_identity["backend"] is None:
                try:
                    output_identity["health"] = request(args.endpoint, "GET", "/health", {})
                    output_identity["backend"] = output_identity["health"]["backend"]
                except (urllib.error.URLError, TimeoutError, ValueError):
                    pass
            for case in frame_cases:
                row = {"case_id": case["case_id"], "frame": frame, "track_id": case["track_id"],
                       "kitti_track_id": case["kitti_track_id"], "gt_point_count": case["gt_point_count"]}
                if not case["seeds"]:
                    row.update(status="unseedable", quality=None)
                elif prepared.get("state") != "ready":
                    row.update(status="prepare-unavailable", quality=None, prepare=prepared)
                else:
                    try:
                        response = fit_request(args.endpoint, frame_meta, prepared["frameToken"],
                                               case["seeds"][0], "Vehicles", "car")
                        row.update(status=response.get("status"), quality=response.get("quality"),
                                   provider=response.get("provider"), reason=response.get("reason"),
                                   response=response)
                        if response.get("status") == "ok":
                            prediction = response.get("cuboid") or response.get("geometry") or response
                            pred = Cuboid(np.asarray(prediction["center"], dtype=float),
                                          np.asarray(prediction["size"], dtype=float),
                                          euler_xyz_matrix(prediction["rotation"]))
                            truth = Cuboid(np.asarray(case["gt_center"]), np.asarray(case["gt_size"]),
                                           euler_xyz_matrix(case["gt_rotation"]))
                            row["metrics"] = errors(pred, truth)
                            scene_cases = [other for other in manifest["cases"]
                                           if other["frame"] == frame and other["class"] in {"Car", "Van", "Truck", "Bus"}]
                            scene_scores = [oriented_iou(pred, Cuboid(
                                np.asarray(other["gt_center"]), np.asarray(other["gt_size"]),
                                euler_xyz_matrix(other["gt_rotation"]))) for other in scene_cases]
                            target_index = next(index for index, other in enumerate(scene_cases)
                                                if other["case_id"] == case["case_id"])
                            row["correct_instance"] = correct_instance(scene_scores, target_index)
                        if pairs_remaining > 0 and case["seeds"]:
                            legacy = fit_request(args.endpoint, frame_meta, prepared["frameToken"],
                                                 case["seeds"][0], "Car", "car")
                            row["legacy_car_response"] = legacy
                            row["paired_legacy_geometry_equal"] = compare_routes(response, legacy)
                            pairs_remaining -= 1
                    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                        row.update(status="transport-error", quality=None,
                                   error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
            output_identity["completed_cases"] = len(rows)
            (output / "run.json").write_text(json.dumps(output_identity, indent=2) + "\n")
            print(json.dumps({'frame': frame, 'completed_cases': len(rows),
                              'expected_cases': len(selected)}), flush=True)
    accepted = [row for row in rows if row.get("status") == "ok"]
    paired = [row for row in rows if "paired_legacy_geometry_equal" in row]
    summary = {
        **output_identity,
        "complete": len(rows) == len(selected),
        "status_counts": {status: sum(row.get("status") == status for row in rows)
                          for status in ("ok", "no-fit", "unseedable", "prepare-unavailable", "transport-error")},
        "quality_counts": {quality: sum(row.get("quality") == quality for row in rows)
                           for quality in ("confident", "estimated")},
        "correct_instance_accepts": sum(row.get("correct_instance") is True for row in accepted),
        "wrong_instance_accepts": sum(row.get("correct_instance") is False for row in accepted),
        "iou_3d": {"n": len(accepted), "mean": float(np.mean([row["metrics"]["iou_3d"] for row in accepted]))
                   if accepted else None,
                   "median": float(np.median([row["metrics"]["iou_3d"] for row in accepted])) if accepted else None,
                   "ge_070": sum(row["metrics"]["iou_3d"] >= .70 for row in accepted)},
        "paired_legacy_count": len(paired),
        "paired_legacy_equal": sum(row["paired_legacy_geometry_equal"] is True for row in paired),
        "paired_legacy_mismatch": sum(row["paired_legacy_geometry_equal"] is False for row in paired),
        "paired_legacy_incomparable": sum(row["paired_legacy_geometry_equal"] not in (True, False)
                                          for row in paired),
    }
    pair_target = min(max(0, args.paired_legacy), len(selected))
    execution_ready = bool(rows) and summary["complete"] and all(
        row.get("status") in {"ok", "no-fit"} and row.get("gt_point_count", 0) > 0 and
        row.get('reason') not in INFRASTRUCTURE_REASONS for row in rows
    )
    summary["gates"] = {
        "execution": "PASS" if execution_ready else "BLOCKED",
        "legacy_route_parity": (
            "NOT_RUN" if pair_target == 0 else
            "FAIL" if summary['paired_legacy_mismatch'] else
            "PASS" if len(paired) == pair_target and summary["paired_legacy_mismatch"] == 0 and
            summary["paired_legacy_incomparable"] == 0 else "BLOCKED"
        ),
        "snap_quality": "NOT_ASSESSED",
    }
    output_identity['complete'] = summary['complete']
    (output / 'run.json').write_text(json.dumps(output_identity, indent=2) + '\n')
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Read-only cvat_drive_0005_3d directory")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8093")
    parser.add_argument("--output", required=True, help="New output directory for evidence")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N Car GT cases")
    parser.add_argument("--paired-legacy", type=int, default=9,
                        help="Compare N Vehicles/car results with legacy Car on the same seed")
    result = run(parser.parse_args())
    if result["gates"]["execution"] != "PASS" or result["gates"]["legacy_route_parity"] in {"BLOCKED", "FAIL"}:
        raise SystemExit(2)
