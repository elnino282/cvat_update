"""Freeze corpus hashes, labels and observed-point seeds before inference."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .metrics import Cuboid, calibration, euler_xyz_matrix, kitti_labels, label_cuboid


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_seeds(cloud, box):
    indices = np.flatnonzero(box.contains(cloud))
    if not len(indices):
        return [], 0
    points = cloud[indices, :3]
    median = np.median(points, axis=0)
    primary = int(np.argmin(np.sum((points - median) ** 2, axis=1)))
    chosen = [primary]
    # Deterministic farthest-point sampling gives independent observed points.
    # Labels select supported seed candidates only; they never enter inference.
    while len(chosen) < min(3, len(points)):
        distances = np.min(np.stack([np.sum((points - points[i]) ** 2, axis=1) for i in chosen]), axis=0)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return [{"pointIndex": int(indices[i]), "point": points[i].tolist()} for i in chosen], len(indices)


def freeze_object(root):
    root = Path(root).resolve()
    frames, cases = {}, []
    for frame_number in range(100):
        frame = f"{frame_number:06d}"
        assets = {kind: str(root / kind / f"{frame}{suffix}") for kind, suffix in (
            ("velodyne", ".bin"), ("image_2", ".png"), ("calib", ".txt"), ("label_2", ".txt"))}
        frames[frame] = {"assets": assets, "sha256": {kind: sha256(path) for kind, path in assets.items()}}
        cloud = np.fromfile(assets["velodyne"], dtype="<f4").reshape(-1, 4)
        transform, _ = calibration(assets["calib"])
        for label in kitti_labels(assets["label_2"]):
            box = label_cuboid(label, transform)
            seeds, count = point_seeds(cloud, box)
            cases.append({"case_id": f"object-{frame}-car-{label['line_index']}", "frame": frame,
                          "split": "tune" if frame_number < 20 else "eval", "label": label,
                          "gt_point_count": count, "seeds": seeds,
                          "adequate": count >= 100 and label["occlusion"] in (0, 1)
                                      and label["truncation"] <= .30})
    return {"version": 1, "corpus": "KITTI 3D Object training local first 100",
            "seed_policy": "GT-assisted observed-point median-nearest primary and at most two frozen farthest-point retries; retry only rejection or transport failure; no labels in request",
            "adequate_policy": "finite points inside exact transformed GT >=100; occlusion in [0,1]; truncation<=0.30",
            "frames": frames, "cases": cases}


def freeze_raw(raw_zip, tracklet_zip):
    drive = "2011_09_26/2011_09_26_drive_0005_sync"
    frames, cases = {}, []
    with zipfile.ZipFile(tracklet_zip) as archive:
        xml = archive.read(f"{drive}/tracklet_labels.xml")
    tracklets = ET.fromstring(xml).find("tracklets").findall("item")
    with zipfile.ZipFile(raw_zip) as archive:
        for frame_number in range(104, 154):
            frame = f"{frame_number:010d}"
            assets = {kind: f"{drive}/{kind}/data/{frame}{suffix}" for kind, suffix in (
                ("velodyne_points", ".bin"), ("image_02", ".png"))}
            cloud_bytes = archive.read(assets["velodyne_points"])
            cloud = np.frombuffer(cloud_bytes, dtype="<f4").reshape(-1, 4)
            frames[frame] = {"archive": str(Path(raw_zip).resolve()), "assets": assets,
                             "sha256": {kind: hashlib.sha256(cloud_bytes if kind == "velodyne_points"
                                                        else archive.read(member)).hexdigest()
                                        for kind, member in assets.items()}}
            for track_id, tracklet in enumerate(tracklets):
                if tracklet.findtext("objectType") != "Car":
                    continue
                first_frame = int(tracklet.findtext("first_frame"))
                poses = tracklet.find("poses").findall("item")
                index = frame_number - first_frame
                if not 0 <= index < len(poses):
                    continue
                pose = poses[index]
                height, width, length = [float(tracklet.findtext(key)) for key in ("h", "w", "l")]
                center = np.array([float(pose.findtext(key)) for key in ("tx", "ty", "tz")])
                center[2] += height / 2
                rotation = [float(pose.findtext(key)) for key in ("rx", "ry", "rz")]
                box = Cuboid(center, np.array([length, width, height]), euler_xyz_matrix(rotation))
                seeds, count = point_seeds(cloud, box)
                cases.append({"case_id": f"raw-{frame}-track-{track_id}", "frame": frame,
                              "track_id": track_id, "gt_point_count": count, "seeds": seeds,
                              "range_m": float(np.linalg.norm(center[:2])), "gt_center": center.tolist(),
                              "gt_size": box.size.tolist(), "gt_rotation": rotation,
                              "occlusion": pose.findtext("occlusion"), "truncation": pose.findtext("truncation")})
    return {"version": 1, "corpus": "KITTI RAW drive 2011_09_26_drive_0005_sync frames104-153",
            "tracklet_sha256": hashlib.sha256(xml).hexdigest(),
            "seed_policy": "GT-assisted observed-point median-nearest primary and at most two frozen farthest-point retries",
            "frames": frames, "cases": cases}


def write_frozen(manifest, path):
    path = Path(path)
    contents = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != contents:
            raise ValueError(f"Refusing to overwrite frozen manifest: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    return sha256(path)
