"""Bounded, seed-anchored evidence and safety checks for experimental fitters.

Car deliberately keeps geometry.py and its existing configuration.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .geometry import NoFit, ground_plane

SEED_MATCH_TOLERANCE = .02
SEED_BOX_TOLERANCE = .05


def check_cancelled(cancelled):
    if cancelled():
        raise NoFit('cancelled', 'Operation ended')


def validate_seed(cloud, seed_payload):
    """pointIndex indexes the finite XYZ cloud used by the viewer, not raw PCD."""
    seed = np.array([seed_payload[k] for k in ('x', 'y', 'z')], dtype=float)
    if seed.shape != (3,) or not np.isfinite(seed).all():
        raise NoFit('invalid-coordinates', 'Seed coordinates must be finite')
    index = seed_payload.get('pointIndex')
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cloud):
        raise NoFit('invalid-coordinates', 'pointIndex is outside the finite cloud')
    if np.linalg.norm(cloud[index, :3] - seed) > SEED_MATCH_TOLERANCE:
        raise NoFit('invalid-coordinates', 'Seed XYZ and finite-cloud pointIndex disagree')
    return seed, index


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass
class Evidence:
    xyz: np.ndarray
    seed: np.ndarray
    index: int
    plane: np.ndarray
    above: np.ndarray
    selected: np.ndarray
    local: np.ndarray
    source: str
    radius: float
    diagnostics: dict

    @property
    def points(self):
        return self.xyz[self.selected]


def component_indices(points, start, radius, cancelled):
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    pending = [int(start)]
    visited[start] = True
    for cursor, index in enumerate(pending):
        if cursor % 32 == 0:
            check_cancelled(cancelled)
        neighbors = tree.query_ball_point(points[index], radius)
        added = [n for n in neighbors if not visited[n]]
        visited[added] = True
        pending.extend(added)
        if len(pending) > 12000:
            raise NoFit('unsafe-fit', 'Component joins excessive scene structure', {'safetyGate': 'component-size'})
    return np.flatnonzero(visited)


def evidence_for(cloud, seed, index, mask, class_key, cancelled):
    check_cancelled(cancelled)
    xyz = np.asarray(cloud[:, :3], dtype=float)
    try:
        plane = ground_plane(xyz, seed)
    except NoFit as exc:
        raise NoFit('unsafe-fit', 'No supported local ground anchor', {'diagnosticFlags': ['ground-unavailable']}) from exc
    check_cancelled(cancelled)
    above = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])
    if above[index] <= .10:
        raise NoFit('no-seed-support', 'Selected return is ground rather than object support')
    pedestrian = class_key == 'pedestrian'
    cycle = class_key == 'two-wheels'
    max_height, reach = (2.6, 2.5) if pedestrian else (2.8, 4.5) if cycle else (5.0, 18.0)
    if above[index] > max_height:
        raise NoFit('unsafe-fit', 'Seed exceeds class vertical extent', {'safetyGate': 'seed-height'})
    local = (np.linalg.norm(xyz[:, :2] - seed[:2], axis=1) <= reach) & (above > .10) & (above < max_height)
    radius = float(np.clip(.22 + np.linalg.norm(seed[:2]) * .008, .25, .42 if pedestrian else .65))
    diagnostics = {}
    source = 'seed-connected'
    eligible = local.copy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(xyz),):
            raise NoFit('unsafe-fit', 'SNAP mask does not match finite cloud', {'safetyGate': 'mask-shape'})
        if mask[index]:
            eligible &= mask
            source = 'snap'
        else:
            diagnostics['diagnosticFlags'] = ['empty-mask' if not mask.any() else 'mask-seed-mismatch']
    selected = np.flatnonzero(eligible)
    position = int(np.flatnonzero(selected == index)[0])
    connected = component_indices(xyz[selected], position, radius, cancelled)
    retained = selected[connected]
    support = np.zeros(len(xyz), dtype=bool)
    support[retained] = True
    diagnostics.update(maskPoints=int(eligible.sum()), selectedPoints=len(retained),
                       detachedMaskPoints=int(eligible.sum()) - len(retained),
                       seedMatchDistance=float(np.linalg.norm(xyz[index] - seed)),
                       componentRadius=radius, maskSource=source)
    return Evidence(xyz, seed, index, plane, above, support, local, source, radius, diagnostics)


def measured_bounds(evidence, yaw):
    local = evidence.points[:, :2] @ rotation(yaw)
    # Include extremities and the selected return; quantile clipping cannot drop seed.
    lower, upper = local.min(axis=0), local.max(axis=0)
    observed = np.r_[upper - lower, float(evidence.above[evidence.selected].max())]
    return lower, upper, observed


def face_support(evidence, yaw, lower, upper):
    local = evidence.points[:, :2] @ rotation(yaw)
    heights = evidence.above[evidence.selected]
    spans = upper - lower
    faces = []
    for axis in (0, 1):
        other = 1 - axis
        current = []
        for edge in (lower[axis], upper[axis]):
            selected = np.abs(local[:, axis] - edge) <= .12
            current.append(bool(selected.sum() >= 4 and spans[other] > .20
                                and np.ptp(local[selected, other]) >= .45 * spans[other]
                                and np.ptp(heights[selected]) >= .30))
        if spans[axis] < .18 and any(current):
            # Coincident bounds describe one visible plane, not two faces.
            # Sensor origin is (0, 0) in native LiDAR coordinates.
            current = [True, False] if (lower[axis] + upper[axis]) / 2 > 0 else [False, True]
        faces.append(current)
    return faces


def place_xy(evidence, yaw, lower, upper, dimensions, faces):
    center = (lower + upper) / 2
    local_sensor = np.zeros(2) @ rotation(yaw)
    for axis in (0, 1):
        observed_span = upper[axis] - lower[axis]
        if dimensions[axis] <= observed_span + .02:
            continue
        if faces[axis][0] != faces[axis][1]:
            center[axis] = lower[axis] + dimensions[axis] / 2 if faces[axis][0] else upper[axis] - dimensions[axis] / 2
        elif not all(faces[axis]) and observed_span < dimensions[axis] * .30:
            # A sufficiently broad observed surface constrains the visible face.
            # A single return gives only an XY support anchor, not a guessed face.
            if len(evidence.points) >= 6 and upper[1-axis] - lower[1-axis] > .4:
                center[axis] = lower[axis] + dimensions[axis] / 2 if local_sensor[axis] < center[axis] else upper[axis] - dimensions[axis] / 2
    return center @ rotation(yaw).T


def finish(evidence, center_xy, dimensions, yaw, *, prototype=None, prior_version=None,
           yaw_uncertain=True, mode='hybrid_prior', diagnostics=None, cancelled=lambda: False):
    check_cancelled(cancelled)
    dimensions = np.asarray(dimensions, dtype=float)
    ground = float(center_xy @ evidence.plane[:2] + evidence.plane[2])
    center = np.r_[center_xy, ground + dimensions[2] / 2]
    if not np.isfinite(np.r_[center, dimensions, yaw]).all() or np.any(dimensions <= 0):
        raise NoFit('unsafe-fit', 'Nonfinite or nonpositive candidate')
    local = (evidence.xyz[:, :2] - center_xy) @ rotation(yaw)
    inside_xy = np.all(np.abs(local) <= dimensions[:2] / 2 + SEED_BOX_TOLERANCE, axis=1)
    inside_z = np.abs(evidence.xyz[:, 2] - center[2]) <= dimensions[2] / 2 + SEED_BOX_TOLERANCE
    inside = inside_xy & inside_z
    if not inside[evidence.index]:
        raise NoFit('unsafe-fit', 'Candidate does not contain selected return', {'safetyGate': 'seed-containment'})
    if np.mean(inside[evidence.selected]) < .95:
        raise NoFit('unsafe-fit', 'Candidate loses measured object support', {'safetyGate': 'support-containment'})
    # Count occupied voxels, not duplicated returns. A detached neighbor may not be
    # swallowed even when SNAP included both objects in the same mask.
    outsiders = inside & evidence.local & ~evidence.selected
    outsider_voxels = np.unique(np.floor(evidence.xyz[outsiders] / .08), axis=0)
    if len(outsider_voxels) >= 3:
        raise NoFit('unsafe-fit', 'Box would include detached neighboring support',
                    {'safetyGate': 'competing-instance', 'competingVoxels': len(outsider_voxels)})
    if abs(float(evidence.seed[:2] @ evidence.plane[:2] + evidence.plane[2]) - ground) > .35:
        raise NoFit('unsafe-fit', 'Candidate ground differs from seed local ground', {'safetyGate': 'ground-anchor'})
    check_cancelled(cancelled)
    details = dict(evidence.diagnostics, **(diagnostics or {}))
    details.update(mask_indices=np.flatnonzero(evidence.selected), point_count=int(evidence.selected.sum()),
                   groundPlane=evidence.plane.tolist(), groundZ=ground, fitMode=mode,
                   yawUncertain=yaw_uncertain, priorVersion=prior_version,
                   prototypeKey=prototype['latent_prototype_key'] if prototype else None,
                   experimental=True, seedContained=True)
    return {'center': center.tolist(), 'size': dimensions.tolist(), 'rotation': [0., 0., float(yaw)],
            'confidence': 0., 'quality': 'estimated', 'diagnostics': details}
