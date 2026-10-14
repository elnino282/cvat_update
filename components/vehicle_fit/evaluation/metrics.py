"""NumPy-only calibration and true oriented 3-D cuboid overlap.

Axes are columns in the cloud coordinate system; sizes are length, width,
height. KITTI camera labels locate the bottom centre, not geometric centre.
The intersection is the convex polyhedron cut out by both sets of six planes.
This deliberately does not substitute a bird's-eye-view overlap.
"""
from dataclasses import dataclass
from itertools import combinations, product
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Cuboid:
    center: np.ndarray
    size: np.ndarray
    axes: np.ndarray

    def __post_init__(self):
        for value in (self.center, self.size, self.axes):
            if not np.isfinite(value).all():
                raise ValueError("non-finite cuboid")
        if self.center.shape != (3,) or self.size.shape != (3,) or self.axes.shape != (3, 3):
            raise ValueError("invalid cuboid shape")
        if np.any(self.size <= 0):
            raise ValueError("non-positive dimensions")
        if not np.allclose(self.axes.T @ self.axes, np.eye(3), atol=2e-6):
            raise ValueError("non-orthogonal axes")

    @property
    def corners(self):
        signs = np.asarray(list(product((-1.0, 1.0), repeat=3)))
        return self.center + (signs * self.size / 2) @ self.axes.T

    @property
    def volume(self):
        return float(np.prod(self.size) * abs(np.linalg.det(self.axes)))

    def contains(self, points, tolerance=1e-6):
        points = np.asarray(points)[:, :3]
        return np.isfinite(points).all(axis=1) & np.all(
            np.abs((points - self.center) @ np.linalg.inv(self.axes).T) <= self.size / 2 + tolerance, axis=1,
        )

    def planes(self):
        inverse = np.linalg.inv(self.axes)
        normals = np.concatenate((inverse, -inverse))
        lengths = np.linalg.norm(normals, axis=1)
        offsets = (normals @ self.center + np.tile(self.size / 2, 2)) / lengths
        normals = normals / lengths[:, None]
        return normals, offsets


def calibration(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            values[key] = np.fromstring(raw, sep=" ")
    rect, extrinsic = np.eye(4), np.eye(4)
    rect[:3, :3] = values["R0_rect"].reshape(3, 3)
    extrinsic[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
    return rect @ extrinsic, values["P2"].reshape(3, 4)


def kitti_labels(path):
    labels = []
    for line_index, line in enumerate(Path(path).read_text().splitlines()):
        parts = line.split()
        if parts[0] != "Car":
            continue
        labels.append({"line_index": line_index, "type": parts[0],
                       "truncation": float(parts[1]), "occlusion": int(parts[2]),
                       "bbox": list(map(float, parts[4:8])),
                       "hwl": list(map(float, parts[8:11])),
                       "location": list(map(float, parts[11:14])), "ry": float(parts[14])})
    return labels


def label_cuboid(label, velo_to_rect):
    height, width, length = label["hwl"]
    center = np.array(label["location"], dtype=float)
    center[1] -= height / 2
    yaw = label["ry"]
    # Camera length is local X, width local Z, height upwards (-Y).
    camera_axes = np.array([[math.cos(yaw), math.sin(yaw), 0],
                            [0, 0, -1], [-math.sin(yaw), math.cos(yaw), 0]])
    rect_to_velo = np.linalg.inv(velo_to_rect)
    axes = rect_to_velo[:3, :3] @ camera_axes
    # Retain transformed corners exactly; calibration decimal precision leaves
    # ~1e-7 non-orthogonality, accepted by the constructor tolerance.
    center = (rect_to_velo @ np.r_[center, 1])[:3]
    return Cuboid(center, np.array([length, width, height]), axes)


def euler_xyz_matrix(rotation):
    x, y, z = map(float, rotation)
    a, b, c, d, e, f = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    # Three.js Euler order XYZ is Rx @ Ry @ Rz.
    return np.array([[c * e, -c * f, d],
                     [a * f + b * d * e, a * e - b * d * f, -b * c],
                     [b * f - a * d * e, b * e + a * d * f, a * c]])


def result_cuboid(result):
    geometry = result.get("cuboid") or result.get("geometry") or result
    return Cuboid(np.asarray(geometry["center"], dtype=float),
                  np.asarray(geometry["size"], dtype=float),
                  euler_xyz_matrix(geometry["rotation"]))


def intersection_volume(left, right):
    if np.any(left.corners.max(axis=0) < right.corners.min(axis=0)) or np.any(
            right.corners.max(axis=0) < left.corners.min(axis=0)):
        return 0.0
    ln, lo = left.planes()
    rn, ro = right.planes()
    normals, offsets = np.concatenate((ln, rn)), np.r_[lo, ro]
    tolerance = 1e-7
    vertices = []
    for indices in combinations(range(12), 3):
        matrix = normals[list(indices)]
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        vertex = np.linalg.solve(matrix, offsets[list(indices)])
        if np.all(normals @ vertex <= offsets + tolerance) and not any(
                np.linalg.norm(vertex - prior) <= tolerance for prior in vertices):
            vertices.append(vertex)
    if len(vertices) < 4:
        return 0.0
    vertices = np.asarray(vertices)
    interior = vertices.mean(axis=0)
    volume = 0.0
    seen_faces = set()
    for normal, offset in zip(normals, offsets):
        indices = np.flatnonzero(np.abs(vertices @ normal - offset) <= tolerance * 4)
        key = tuple(sorted(map(int, indices)))
        if len(indices) < 3 or key in seen_faces:
            continue
        seen_faces.add(key)
        face = vertices[indices]
        center = face.mean(axis=0)
        basis = face[0] - center
        basis /= np.linalg.norm(basis)
        tangent = np.cross(normal, basis)
        order = np.argsort(np.arctan2((face - center) @ tangent, (face - center) @ basis))
        face = face[order]
        for i in range(1, len(face) - 1):
            volume += abs(float(np.linalg.det(np.stack((face[0] - interior,
                                                       face[i] - interior,
                                                       face[i + 1] - interior))))) / 6
    return min(left.volume, right.volume, max(0.0, volume))


def oriented_iou(left, right):
    overlap = intersection_volume(left, right)
    return overlap / (left.volume + right.volume - overlap)


def errors(prediction, ground_truth):
    p_heading, g_heading = prediction.axes[:, 0], ground_truth.axes[:, 0]
    yaw_difference = math.atan2(p_heading[1], p_heading[0]) - math.atan2(g_heading[1], g_heading[0])
    yaw_error = abs((yaw_difference + math.pi / 2) % math.pi - math.pi / 2)
    delta = np.abs(prediction.size - ground_truth.size)
    return {"iou_3d": oriented_iou(prediction, ground_truth),
            "center_error_m": float(np.linalg.norm(prediction.center - ground_truth.center)),
            "yaw_error_rad_mod_pi": yaw_error,
            "dimension_error_m_lwh": delta.tolist(),
            "dimension_relative_error_lwh": (delta / ground_truth.size).tolist()}


def project(points, velo_to_rect, p2):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    pixels = (p2 @ velo_to_rect @ homogeneous.T).T
    valid = pixels[:, 2] > 1e-6
    projected = np.full((len(points), 2), np.nan)
    projected[valid] = pixels[valid, :2] / pixels[valid, 2, None]
    return projected, valid
