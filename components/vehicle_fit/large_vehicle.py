"""Experimental Truck geometry candidates with class-specific PR0 priors.

Candidates are deterministic and bounded; no Car constants or fitter are changed.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .geometry import NoFit
from .priors import REGISTRY
from .shared import (check_cancelled, evidence_for, face_support, finish, measured_bounds,
                     place_xy, rotation)


def _yaw(vector):
    return float((math.atan2(vector[1], vector[0]) + math.pi / 2) % math.pi - math.pi / 2)


def orientation_candidates(points, cancelled):
    """Robust PCA, minimum-area rectangle and bounded two-edge L-shape search."""
    check_cancelled(cancelled)
    xy = np.unique(np.round(points[:, :2], 3), axis=0)
    if len(xy) < 3:
        return [(0., 'neutral', True)]
    centered = xy - np.median(xy, axis=0)
    radial = np.linalg.norm(centered, axis=1)
    core = xy[radial <= np.quantile(radial, .95)]
    if len(core) < 3:
        core = xy
    values, vectors = np.linalg.eigh(np.cov(core, rowvar=False))
    pca_yaw = _yaw(vectors[:, -1])
    elongated = values[-1] > max(values[0], .01) * 3 and np.ptp(core @ vectors[:, -1]) > 1.5
    proposals = [(pca_yaw, 'robust-pca', not elongated), (pca_yaw + math.pi / 2, 'robust-pca', not elongated)]
    try:
        hull = xy[ConvexHull(xy).vertices]
        rectangle_candidates = []
        for edge in np.roll(hull, -1, axis=0) - hull:
            check_cancelled(cancelled)
            if np.linalg.norm(edge) < .10:
                continue
            yaw = _yaw(edge)
            span = np.ptp(xy @ rotation(yaw), axis=0)
            rectangle_candidates.append((float(np.prod(span)), yaw))
        for _, yaw in sorted(rectangle_candidates)[:2]:
            proposals.extend([(yaw, 'minimum-area', not elongated),
                              (yaw + math.pi / 2, 'minimum-area', not elongated)])
    except QhullError:
        pass  # A line still has a usable longitudinal PCA cue.
    rng = np.random.default_rng(29)
    l_shapes = []
    for _ in range(32):
        check_cancelled(cancelled)
        pair = xy[rng.choice(len(xy), 2, replace=False)]
        if np.linalg.norm(pair[1] - pair[0]) < .5:
            continue
        yaw = _yaw(pair[1] - pair[0])
        local = xy @ rotation(yaw)
        lower, upper = np.quantile(local, [.01, .99], axis=0)
        # Each return supports either of the two visible orthogonal surfaces.
        residuals = []
        for x_edge in (lower[0], upper[0]):
            for y_edge in (lower[1], upper[1]):
                residuals.append(np.minimum(abs(local[:, 0] - x_edge), abs(local[:, 1] - y_edge)))
        score = min(float(np.quantile(residual, .80)) for residual in residuals)
        l_shapes.append((score, yaw))
    for _, yaw in sorted(l_shapes)[:2]:
        proposals.extend([(yaw, 'l-shape', not elongated), (yaw + math.pi / 2, 'l-shape', not elongated)])
    result, seen = [], set()
    for yaw, method, uncertain in proposals:
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        key = round(yaw, 3)
        if key not in seen:
            result.append((yaw, method, uncertain))
            seen.add(key)
    return result


def _surface_error(points, yaw):
    local = points[:, :2] @ rotation(yaw)
    lower, upper = local.min(axis=0), local.max(axis=0)
    distances = np.minimum(local - lower, upper - local)
    return float(np.quantile(distances.min(axis=1), .80))


def _reject_bent_body(evidence, yaw):
    """Reject two substantial longitudinal sections with incompatible axes."""
    local = evidence.points[:, :2] @ rotation(yaw)
    if len(local) < 30 or np.ptp(local[:, 0]) < 5:
        return
    axes = []
    for part in (local[local[:, 0] <= np.quantile(local[:, 0], .40)],
                 local[local[:, 0] >= np.quantile(local[:, 0], .60)]):
        values, vectors = np.linalg.eigh(np.cov(part, rowvar=False))
        if values[-1] <= max(values[0], .01) * 2 or np.ptp(part @ vectors[:, -1]) < 1.8:
            return
        axes.append(vectors[:, -1])
    difference = math.acos(float(np.clip(abs(np.dot(*axes)), 0, 1)))
    if difference > math.radians(20):
        raise NoFit('unsafe-fit', 'Support contains incompatible rigid section directions',
                    {'safetyGate': 'articulated-or-competing-instance', 'sectionAngleRadians': difference})


def fit_with_priors(evidence, class_key, registry, cancelled, *, ride_state=None):
    prototypes = registry.pool(class_key, ride_state)
    if not prototypes:
        raise NoFit('unsafe-fit', 'Class has no trustworthy completion prior',
                    {'diagnosticFlags': ['prior-unavailable']})
    proposals, rejected = [], []
    for yaw, method, uncertain in orientation_candidates(evidence.points, cancelled):
        check_cancelled(cancelled)
        lower, upper, observed = measured_bounds(evidence, yaw)
        faces = face_support(evidence, yaw, lower, upper)
        surface_error = _surface_error(evidence.points, yaw)
        for prototype in prototypes:
            low, high = np.array(prototype['p10_lwh']), np.array(prototype['p90_lwh'])
            median = np.array(prototype['median_lwh'])
            if np.any(observed > high + .03):
                continue
            roof_observed = (len(evidence.points) >= 25 and np.ptp(evidence.above[evidence.selected]) >= .7 * median[2]
                             and np.count_nonzero(evidence.above[evidence.selected] >= observed[2] - .15) >= 4)
            weights = np.r_[[float(all(pair)) for pair in faces], float(roof_observed)]
            dimensions = np.clip(weights * observed + (1 - weights) * median, low, high)
            dimensions = np.maximum(dimensions, np.minimum(observed + .02, high))
            center = place_xy(evidence, yaw, lower, upper, dimensions, faces)
            missing = float(np.sum((dimensions - np.minimum(observed, dimensions)) / dimensions))
            score = surface_error / .15 + .2 * missing
            proposals.append((score, yaw, method, uncertain, prototype, dimensions, center, observed, faces, weights))
    if not proposals:
        raise NoFit('unsafe-fit', 'Observed support exceeds trustworthy class prototype ranges',
                    {'diagnosticFlags': ['prior-unavailable'], 'safetyGate': 'class-extent'})
    for _, yaw, method, uncertain, prototype, dimensions, center, observed, faces, weights in sorted(proposals, key=lambda p: p[0]):
        check_cancelled(cancelled)
        try:
            _reject_bent_body(evidence, yaw)
            return finish(evidence, center, dimensions, yaw, prototype=prototype, prior_version=registry.version,
                          yaw_uncertain=uncertain,
                          diagnostics={'candidateMethod': method, 'candidateCount': len(proposals),
                                       'observedLWH': observed.tolist(), 'visibleSides': faces,
                                       'dimensionWeights': weights.tolist()}, cancelled=cancelled)
        except NoFit as exc:
            if exc.reason == 'cancelled':
                raise
            rejected.append(exc)
    raise rejected[0]


def fit_truck(cloud, seed, *, seed_index, mask=None, cancelled=lambda: False, registry=REGISTRY):
    evidence = evidence_for(cloud, seed, seed_index, mask, 'vehicles/truck', cancelled)
    return fit_with_priors(evidence, 'vehicles/truck', registry, cancelled)
