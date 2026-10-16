"""Read-only nuScenes inventory, frozen evaluation corpus, and dev-only priors.

No nuScenes devkit, model, camera or sweeps are needed for generation. Boxes
are transformed global -> ego -> LIDAR_TOP, preserving all three rotations.
The source dataset is never written. Source category is provenance, never a
new public subtype. In particular vehicle.car is NOT independently labelled Van.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np

from .manifest import point_seeds, sha256
from .metrics import Cuboid


SCRIPT_VERSION = "nuscenes-pr0-v1"
DEFAULT_SEED = 20261016
TABLES = ("attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log",
          "map", "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility")
SOURCES = [
    "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md",
    "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md",
]


def taxonomy(category, attributes=()):
    """Map only locked public names; absence of a rider attribute stays unknown."""
    label, vehicle_type, ride = None, None, None
    if category in {"vehicle.car", "vehicle.truck"}:
        label, vehicle_type = "vehicles", category.rsplit(".", 1)[1]
    elif category.startswith("vehicle.bus."):
        label, vehicle_type = "vehicles", "bus"
    elif category.startswith("human.pedestrian."):
        label = "pedestrian"
    elif category in {"vehicle.bicycle", "vehicle.motorcycle"}:
        label = "two-wheels"
        with_rider, without_rider = "cycle.with_rider" in attributes, "cycle.without_rider" in attributes
        if with_rider != without_rider:
            ride = with_rider
    return label, vehicle_type, ride


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid nuScenes quaternion")
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError("nuScenes quaternion must be normalized")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def pose_matrix(record):
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(record["rotation"])
    translation = np.asarray(record["translation"], dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Invalid nuScenes translation")
    result[:3, 3] = translation
    return result


def matrix_to_euler_xyz(axes):
    """Inverse of the existing evaluator / Three.js Rx @ Ry @ Rz convention."""
    axes = np.asarray(axes)
    y = math.asin(float(np.clip(axes[0, 2], -1, 1)))
    if abs(axes[0, 2]) < 0.9999999:
        x, z = math.atan2(-axes[1, 2], axes[2, 2]), math.atan2(-axes[0, 1], axes[0, 0])
    else:
        x, z = math.atan2(axes[2, 1], axes[1, 1]), 0.0
    return [x, y, z]


def lidar_cuboid(annotation, calibrated_sensor, ego_pose):
    lidar_to_global = pose_matrix(ego_pose) @ pose_matrix(calibrated_sensor)
    global_to_lidar = np.linalg.inv(lidar_to_global)
    center = (global_to_lidar @ np.r_[annotation["translation"], 1])[:3]
    axes = global_to_lidar[:3, :3] @ quaternion_matrix(annotation["rotation"])
    width, length, height = annotation["size"]
    return Cuboid(center, np.array([length, width, height]), axes)


def scene_splits(scenes, seed=DEFAULT_SEED):
    tokens = sorted(scene["token"] for scene in scenes)
    if len(tokens) < 3:
        raise ValueError("At least three scenes are required for isolated dev/validation/held_out")
    random.Random(seed).shuffle(tokens)
    hold_count = max(1, len(tokens) // 5)
    dev_count = len(tokens) - 2 * hold_count
    return {token: "dev" if i < dev_count else "validation" if i < dev_count + hold_count else "held_out"
            for i, token in enumerate(tokens)}


def source_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Dataset path escapes dataroot: {relative}")
    return path


def load_cloud(manifest, sample_token, dataroot=None):
    frame = manifest["frames"][sample_token]
    path = source_path(dataroot or manifest["dataset_root"], frame["lidar_path"])
    if sha256(path) != frame["sha256"]:
        raise ValueError(f"Frozen cloud changed: {sample_token}")
    raw = path.read_bytes()
    if not raw or len(raw) % 20:
        raise ValueError(f"nuScenes LIDAR_TOP must contain five float32 columns: {path}")
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 5)


def cloud_to_pcd(cloud):
    """Unfiltered XYZ+intensity; original point indices survive PCD conversion."""
    values = np.asarray(cloud[:, :4], dtype="<f4")
    header = ("VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
              f"COUNT 1 1 1 1\nWIDTH {len(values)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS {len(values)}\nDATA binary\n")
    return header.encode("ascii") + values.tobytes()


def load_manifest(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "status": "insufficient-data"}
    median = np.median(values, axis=0)
    return {"count": len(values), "min": values.min(axis=0).tolist(),
            "p10": np.quantile(values, .1, axis=0).tolist(), "median": median.tolist(),
            "p90": np.quantile(values, .9, axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mad": np.median(np.abs(values - median), axis=0).tolist()}


def inventory(root, version):
    root = Path(root).expanduser().resolve()
    tables = {name: json.loads((root / version / f"{name}.json").read_text()) for name in TABLES}
    indexed = {name: {row["token"]: row for row in rows} for name, rows in tables.items()}
    errors = []
    for name, rows in tables.items():
        if len(indexed[name]) != len(rows):
            errors.append({"table": name, "error": "duplicate-token"})
    references = {
        "calibrated_sensor": {"sensor_token": "sensor"},
        "instance": {"category_token": "category", "first_annotation_token": "sample_annotation",
                     "last_annotation_token": "sample_annotation"},
        "sample": {"scene_token": "scene", "prev": "sample", "next": "sample"},
        "sample_annotation": {"sample_token": "sample", "instance_token": "instance",
                              "visibility_token": "visibility", "attribute_tokens": "attribute",
                              "prev": "sample_annotation", "next": "sample_annotation"},
        "sample_data": {"sample_token": "sample", "calibrated_sensor_token": "calibrated_sensor",
                        "ego_pose_token": "ego_pose", "prev": "sample_data", "next": "sample_data"},
        "scene": {"log_token": "log", "first_sample_token": "sample", "last_sample_token": "sample"},
        "map": {"log_tokens": "log"},
    }
    for name, fields in references.items():
        for row in tables[name]:
            for field, target in fields.items():
                values = row[field] if isinstance(row[field], list) else [row[field]]
                for value in values:
                    if value and value not in indexed[target]:
                        errors.append({"table": name, "token": row["token"], "field": field, "missing": value})
    channels, missing = defaultdict(Counter), []
    for row in tables["sample_data"]:
        path = source_path(root, row["filename"])
        sensor = indexed["sensor"].get(indexed["calibrated_sensor"].get(
            row["calibrated_sensor_token"], {}).get("sensor_token"), {})
        channel = sensor.get("channel", "broken-reference")
        channels[channel]["keyframes" if row["is_key_frame"] else "sweeps"] += 1
        if path.is_file():
            channels[channel]["existing_files"] += 1
            channels[channel]["bytes"] += path.stat().st_size
        else:
            missing.append(row["filename"])
    for row in tables["map"]:
        if not source_path(root, row["filename"]).is_file():
            missing.append(row["filename"])
    by_category, by_attribute, rider_counts = Counter(), Counter(), defaultdict(Counter)
    category_annotations = defaultdict(list)
    for ann in tables["sample_annotation"]:
        instance = indexed["instance"].get(ann["instance_token"], {})
        category = indexed["category"].get(instance.get("category_token"), {}).get("name", "broken-reference")
        attrs = [indexed["attribute"].get(token, {}).get("name", "broken-reference")
                 for token in ann["attribute_tokens"]]
        by_category[category] += 1
        category_annotations[category].append(ann)
        by_attribute.update(attrs)
        if category in {"vehicle.bicycle", "vehicle.motorcycle"}:
            ride = taxonomy(category, attrs)[2]
            rider_counts[category]["unknown" if ride is None else "with_rider" if ride else "without_rider"] += 1
    result = {"dataset_root": str(root), "dataset_version": version,
              "available_versions": sorted(path.name for path in root.glob("v1.0-*") if path.is_dir()),
              "table_counts": {name: len(rows) for name, rows in tables.items()},
              "source_hashes": {f"{version}/{name}.json": sha256(root / version / f"{name}.json") for name in TABLES},
              "sensor_files": dict(channels), "missing_files": sorted(missing), "broken_references": errors,
              "annotations_by_source_category": dict(by_category), "annotations_by_attribute": dict(by_attribute),
              "source_category_distributions": {
                  category: {"num_lidar_pts": _stats([ann["num_lidar_pts"] for ann in anns]),
                             "dimensions_lwh": _stats([[ann["size"][1], ann["size"][0], ann["size"][2]] for ann in anns]),
                             "visibility": dict(Counter(ann["visibility_token"] for ann in anns))}
                  for category, anns in sorted(category_annotations.items())},
              "nonblocking_map_log_references": [error for error in errors if error["table"] == "map"],
              "two_wheels_rider_annotations": dict(rider_counts), "sources": SOURCES}
    return result, indexed


def _motion(attributes):
    if any(value in attributes for value in ("vehicle.moving", "pedestrian.moving")):
        return "moving"
    if any(value in attributes for value in ("vehicle.stopped", "vehicle.parked", "pedestrian.standing",
                                             "pedestrian.sitting_lying_down")):
        return "static"
    return "unknown"


def _bins(distance, points, visibility, motion, adjacent, ride):
    return {"distance": "0-20m" if distance <= 20 else "20-40m" if distance <= 40 else ">40m",
            "points": "0" if points == 0 else "1-4" if points <= 4 else "5-15" if points <= 15
            else "16-50" if points <= 50 else ">50", "visibility": visibility or "unknown",
            "motion": motion, "adjacency": "adjacent" if adjacent else "isolated",
            "ride_state": "unknown" if ride is None else "rider" if ride else "no-rider"}


def footprint_distance(left, right):
    """Exact XY OBB boundary distance in global frame (nuScenes boxes are upright)."""
    polygons = []
    for ann in (left, right):
        width, length, _ = ann["size"]
        axes = quaternion_matrix(ann["rotation"])[:2, :2]
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length / 2, width / 2]
        polygons.append(corners @ axes.T + np.asarray(ann["translation"][:2]))
    a, b = polygons
    separates = False
    for poly in polygons:
        edges = np.roll(poly, -1, axis=0) - poly
        for axis in np.column_stack((-edges[:, 1], edges[:, 0])):
            pa, pb = a @ axis, b @ axis
            separates |= pa.max() < pb.min() or pb.max() < pa.min()
    if not separates:
        return 0.0
    distances = []
    for points, polygon in ((a, b), (b, a)):
        starts, edges = polygon, np.roll(polygon, -1, axis=0) - polygon
        offset = points[:, None, :] - starts[None, :, :]
        weights = np.clip(np.sum(offset * edges, axis=2) / np.sum(edges * edges, axis=1), 0, 1)
        distances.append(float(np.linalg.norm(offset - weights[:, :, None] * edges, axis=2).min()))
    return min(distances)


def create_manifest(root, version, seed=DEFAULT_SEED):
    audit, tables = inventory(root, version)
    blocking_references = [error for error in audit["broken_references"] if error["table"] != "map"]
    if blocking_references:
        raise ValueError(f"Broken token references: {blocking_references[:3]}")
    root = Path(audit["dataset_root"])
    splits = scene_splits(tables["scene"].values(), seed)
    annotations = defaultdict(list)
    for ann in tables["sample_annotation"].values():
        annotations[ann["sample_token"]].append(ann)
    lidar_data = {}
    for row in tables["sample_data"].values():
        sensor = tables["sensor"][tables["calibrated_sensor"][row["calibrated_sensor_token"]]["sensor_token"]]
        if sensor["channel"] == "LIDAR_TOP" and row["is_key_frame"]:
            if row["sample_token"] in lidar_data:
                raise ValueError("Duplicate LIDAR_TOP keyframe")
            lidar_data[row["sample_token"]] = row
    frames, cases = {}, []
    for sample_token, sample in sorted(tables["sample"].items()):
        if sample_token not in lidar_data:
            raise ValueError(f"Missing LIDAR_TOP keyframe: {sample_token}")
        row = lidar_data[sample_token]
        calibration, ego = tables["calibrated_sensor"][row["calibrated_sensor_token"]], tables["ego_pose"][row["ego_pose_token"]]
        path = source_path(root, row["filename"])
        if not path.is_file():
            raise ValueError(f"Missing LiDAR file: {row['filename']}")
        raw = path.read_bytes()
        if not raw or len(raw) % 20:
            raise ValueError(f"Malformed nuScenes LiDAR: {row['filename']}")
        cloud = np.frombuffer(raw, dtype="<f4").reshape(-1, 5)
        frame = {"lidar_path": row["filename"], "sha256": hashlib.sha256(raw).hexdigest(),
                 "point_dimensions": 5, "point_count": len(cloud),
                 "finite_xyz_points": int(np.isfinite(cloud[:, :3]).all(axis=1).sum()),
                 "sample_data_token": row["token"], "timestamp": row["timestamp"],
                 "scene_token": sample["scene_token"], "split": splits[sample["scene_token"]],
                 "calibrated_sensor_token": row["calibrated_sensor_token"], "ego_pose_token": row["ego_pose_token"],
                 "lidar_to_global": (pose_matrix(ego) @ pose_matrix(calibration)).tolist(), "annotations": []}
        frames[sample_token] = frame
        frame_anns = sorted(annotations[sample_token], key=lambda ann: ann["token"])
        # Distance prefilter is a lower bound; exact OBB comparison follows nearby pairs.
        centers = np.array([ann["translation"][:2] for ann in frame_anns])
        radii = np.array([np.linalg.norm(ann["size"][:2]) / 2 for ann in frame_anns])
        for index, ann in enumerate(frame_anns):
            category = tables["category"][tables["instance"][ann["instance_token"]]["category_token"]]["name"]
            attrs = sorted(tables["attribute"][token]["name"] for token in ann["attribute_tokens"])
            label, vehicle_type, ride = taxonomy(category, attrs)
            box = lidar_cuboid(ann, calibration, ego)
            frame["annotations"].append({"annotation_token": ann["token"], "instance_token": ann["instance_token"],
                                         "source_category": category, "gt_center": box.center.tolist(),
                                         "gt_size": box.size.tolist(), "gt_rotation_matrix": box.axes.tolist()})
            if label is None:
                continue
            seeds, count = point_seeds(cloud, box)
            distance = float(np.linalg.norm(np.asarray(ann["translation"][:2]) - np.asarray(ego["translation"][:2])))
            candidates = np.flatnonzero(np.linalg.norm(centers - centers[index], axis=1) - radii - radii[index] <= 1)
            adjacent = []
            for other in candidates:
                if other == index:
                    continue
                neighbor = frame_anns[other]
                if abs(ann["translation"][2] - neighbor["translation"][2]) > (ann["size"][2] + neighbor["size"][2]) / 2:
                    continue
                if footprint_distance(ann, neighbor) <= 1:
                    adjacent.append(neighbor["token"])
            motion = _motion(attrs)
            cases.append({"case_id": ann["token"], "annotation_token": ann["token"], "sample_token": sample_token,
                          "scene_token": sample["scene_token"], "instance_token": ann["instance_token"],
                          "split": frame["split"], "source_category": category, "source_attributes": attrs,
                          "target_label": label, "vehicles_type": vehicle_type, "ride_state": ride,
                          "lidar_path": row["filename"], "gt_center": box.center.tolist(), "gt_size": box.size.tolist(),
                          "gt_rotation": matrix_to_euler_xyz(box.axes), "gt_rotation_matrix": box.axes.tolist(),
                          "num_lidar_pts": ann["num_lidar_pts"], "gt_point_count": count, "seeds": seeds,
                          "distance_m": distance, "visibility": ann["visibility_token"], "motion_state": motion,
                          "adjacent_annotation_tokens": sorted(adjacent),
                          "bins": _bins(distance, count, ann["visibility_token"], motion, bool(adjacent), ride)})
    manifest = {"schema_version": 1, "script_version": SCRIPT_VERSION, "script_sha256": sha256(__file__),
                "dataset_root": str(root), "dataset_version": version, "random_seed": seed,
                "source_hashes": audit["source_hashes"], "coordinate_frame": "LIDAR_TOP",
                "dimension_order": "length,width,height", "rotation_convention": "Three.js Euler XYZ (Rx @ Ry @ Rz)",
                "split_policy": "Sort scene tokens, Python Random(seed).shuffle, first 60% dev, next 20% validation, remainder held_out; never split a scene",
                "scene_splits": splits, "scene_names": {token: row["name"] for token, row in tables["scene"].items()},
                "seed_policy": "GT-assisted observed-point median-nearest primary and two farthest-point alternatives; no GT box in inference input",
                "adjacency_policy": "Global XY OBB boundary gap <=1m and vertical intervals overlap; all source annotation categories; GT audit stratifier only",
                "motion_policy": "Only explicit vehicle/pedestrian attributes; cycle rider is not motion evidence; no temporal inference",
                "limitations": ["vehicle.car includes Car/Van/SUV: Van insufficient-ground-truth",
                                "Mini splits are local scene holdouts, not official nuScenes test acceptance",
                                "Visibility is source camera-annotation metadata; no camera or sweep inference inputs",
                                "GT seed selection is oracle-assisted and does not measure UI picking accuracy",
                                "human.pedestrian.* maps per PRD, including personal_mobility; category counts retained"],
                "frames": frames, "cases": sorted(cases, key=lambda case: case["case_id"])}
    audit["scene_splits"] = {split: sorted(manifest["scene_names"][token] for token, value in splits.items() if value == split)
                             for split in ("dev", "validation", "held_out")}
    audit["mapped_annotations"] = len(cases)
    audit["distributions"] = summarize_cases(cases)
    audit["lidar_point_count_differences"] = {"annotations_differing_from_num_lidar_pts": sum(c["gt_point_count"] != c["num_lidar_pts"] for c in cases),
                                             "difference_observed_minus_source": _stats([c["gt_point_count"] - c["num_lidar_pts"] for c in cases])}
    audit["limitations"] = manifest["limitations"]
    return manifest, audit


def class_key(case):
    return f"vehicles/{case['vehicles_type']}" if case["target_label"] == "vehicles" else case["target_label"]


def summarize_cases(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[(case["split"], class_key(case))].append(case)
    result = {}
    for (split, label), members in sorted(groups.items()):
        bins = {name: dict(Counter(c["bins"][name] for c in members)) for name in members[0]["bins"]}
        expected = {"distance": ["0-20m", "20-40m", ">40m"], "points": ["0", "1-4", "5-15", "16-50", ">50"],
                    "visibility": ["1", "2", "3", "4"], "motion": ["moving", "static"],
                    "adjacency": ["isolated", "adjacent"]}
        if label == "two-wheels":
            expected["ride_state"] = ["rider", "no-rider"]
        result.setdefault(split, {})[label] = {
            "annotations": len(members), "instances": len({c["instance_token"] for c in members}),
            "seedable": sum(bool(c["seeds"]) for c in members),
            "dimensions_lwh": _stats([c["gt_size"] for c in members]),
            "distance_m": _stats([c["distance_m"] for c in members]),
            "num_lidar_pts": _stats([c["num_lidar_pts"] for c in members]),
            "observed_lidar_points": _stats([c["gt_point_count"] for c in members]),
            "bins": bins,
            "insufficient_data_bins": {name: [value for value in values if not bins[name].get(value)]
                                       for name, values in expected.items()},
            "held_out_100_target_status": "insufficient-data" if split == "held_out" and sum(bool(c["seeds"]) for c in members) < 100 else "not-applicable" if split != "held_out" else "available",
        }
    return result


def select_cases(manifest, split="held_out", limit_per_class=100, seedable_only=True):
    """Freeze balanced composite-bin round robin before seeing inference outputs."""
    groups = defaultdict(lambda: defaultdict(list))
    for case in manifest["cases"]:
        if case["split"] != split or (seedable_only and not case["seeds"]):
            continue
        bins = tuple(case["bins"][name] for name in sorted(case["bins"]))
        groups[class_key(case)][bins].append(case)
    chosen = []
    for label, buckets in sorted(groups.items()):
        for bucket in buckets.values():
            bucket.sort(key=lambda case: hashlib.sha256(f"{manifest['random_seed']}:{case['case_id']}".encode()).hexdigest())
        count, cursor = 0, 0
        while any(len(bucket) > cursor for bucket in buckets.values()) and (not limit_per_class or count < limit_per_class):
            for _, bucket in sorted(buckets.items()):
                if len(bucket) > cursor and (not limit_per_class or count < limit_per_class):
                    chosen.append(bucket[cursor])
                    count += 1
            cursor += 1
    return chosen


def _clusters(sizes, maximum=3, minimum_instances=8):
    """Deterministic log-size k-means; geometric keys do not assert truck subtypes."""
    sizes = np.asarray(sizes)
    values = np.log(sizes)
    k = min(maximum, max(1, len(values) // minimum_instances), len(np.unique(values, axis=0)))
    first = int(np.argmin(np.sum((values - np.median(values, axis=0)) ** 2, axis=1)))
    centers = [values[first]]
    while len(centers) < k:
        distances = np.min(np.stack([np.sum((values - center) ** 2, axis=1) for center in centers]), axis=0)
        centers.append(values[int(np.argmax(distances))])
    centers = np.asarray(centers)
    assignment = np.full(len(values), -1)
    for _ in range(100):
        updated = np.argmin(np.sum((values[:, None] - centers[None]) ** 2, axis=2), axis=1)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for index in range(k):
            if np.any(assignment == index):
                centers[index] = values[assignment == index].mean(axis=0)
    ordered = sorted(set(assignment.tolist()), key=lambda index: tuple(np.median(sizes[assignment == index], axis=0)))
    return [np.flatnonzero(assignment == index) for index in ordered]


def build_priors(manifest, created_at):
    date.fromisoformat(created_at)
    pools = defaultdict(list)
    for case in manifest["cases"]:
        if case["split"] != "dev":
            continue
        # Source categories and rider state remain private clustering provenance.
        pool = "car_van" if case["vehicles_type"] == "car" else case["source_category"]
        ride = case["ride_state"] if case["target_label"] == "two-wheels" else None
        pools[(case["target_label"], case["vehicles_type"], pool, str(ride))].append(case)
    records = []
    for (label, vehicle_type, pool, _), members in sorted(pools.items(), key=lambda pair: str(pair[0])):
        by_instance = defaultdict(list)
        for member in members:
            by_instance[member["instance_token"]].append(member)
        tokens = sorted(by_instance)
        sizes = np.array([np.median([case["gt_size"] for case in by_instance[token]], axis=0) for token in tokens])
        for index, indices in enumerate(_clusters(sizes)):
            cluster_tokens = {tokens[i] for i in indices}
            cluster_cases = [case for case in members if case["instance_token"] in cluster_tokens]
            stats = _stats(sizes[indices])
            for target_type in ("car", "van") if vehicle_type == "car" else (vehicle_type,):
                records.append({"target_label": label, "vehicles_type": target_type,
                                "latent_prototype_key": f"{pool}:{members[0]['ride_state']}:geometry-{index:02d}",
                                "ride_state": members[0]["ride_state"], "source_pool": pool,
                                "sample_count": len(cluster_cases), "instance_count": len(indices),
                                "median_lwh": stats["median"], "p10_lwh": stats["p10"], "p90_lwh": stats["p90"],
                                "mad_lwh": stats["mad"], "source_instance_tokens": sorted(cluster_tokens),
                                "dataset_version": manifest["dataset_version"], "split": "dev",
                                "created_at": created_at, "script_version": SCRIPT_VERSION,
                                "status": "insufficient-data" if len(indices) < 8 else "development-prototype",
                                "acceptance_gt": "insufficient-ground-truth" if target_type == "van" else "source-category"})
    inputs = [{key: case[key] for key in ("annotation_token", "instance_token", "gt_size", "source_category", "ride_state")}
              for case in manifest["cases"] if case["split"] == "dev"]
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "prior_version": f"{SCRIPT_VERSION}-{digest[:12]}", "created_at": created_at,
            "script_version": SCRIPT_VERSION, "script_sha256": manifest["script_sha256"],
            "dataset_version": manifest["dataset_version"], "random_seed": manifest["random_seed"],
            "source_hashes": manifest["source_hashes"], "dev_input_sha256": digest,
            "training_scene_tokens": sorted(token for token, split in manifest["scene_splits"].items() if split == "dev"),
            "method": "Dev only; per-instance median dimensions avoid temporal count bias; log-size deterministic k-means k<=3, floor(instances/8); source-category/rider pools; stats across instance medians",
            "limitations": ["Prototype clusters are geometric; do not assert pickup/rigid/tractor labels",
                            "Sparse pools retained and flagged insufficient-data; not accepted production priors",
                            "P10/P90 are descriptive dev quantiles, not validated safety ranges",
                            "Van reuses Car-Van pool; no independent Van ground truth"], "records": records}


def write_artifact(path, value, compressed=False):
    payload = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":") if compressed else None,
                          indent=None if compressed else 2) + "\n").encode()
    if compressed:
        payload = gzip.compress(payload, compresslevel=9, mtime=0)
    path = Path(path)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Refusing to overwrite frozen artifact: {path}; choose a new output directory")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default=os.environ.get("NUSCENES_DATAROOT", "~/Datasets/nuScenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--created-at", required=True, help="ISO date pinned for byte-reproducible prior output")
    args = parser.parse_args()
    root, output = Path(args.dataroot).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the read-only dataset")
    date.fromisoformat(args.created_at)
    manifest, audit = create_manifest(root, args.version, args.seed)
    priors = build_priors(manifest, args.created_at)
    output.mkdir(parents=True, exist_ok=True)
    audit["artifacts"] = {
        "nuscenes_manifest.json.gz": write_artifact(output / "nuscenes_manifest.json.gz", manifest, compressed=True),
        "nuscenes_priors.json": write_artifact(output / "nuscenes_priors.json", priors),
    }
    write_artifact(output / "nuscenes_inventory.json", audit)
    print(json.dumps({"output": str(output), "frames": len(manifest["frames"]), "cases": len(manifest["cases"]),
                      "priors": len(priors["records"]), "missing_files": len(audit["missing_files"]),
                      "broken_references": len(audit["broken_references"]), "sha256": audit["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
