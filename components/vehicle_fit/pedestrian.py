"""Experimental ground-to-top pedestrian fitting, independent of Car geometry."""
import numpy as np

from .geometry import NoFit
from .priors import REGISTRY
from .shared import check_cancelled, evidence_for, face_support, finish, measured_bounds, place_xy


def fit_pedestrian(cloud, seed, *, seed_index, mask=None, cancelled=lambda: False, registry=REGISTRY):
    evidence = evidence_for(cloud, seed, seed_index, mask, 'pedestrian', cancelled)
    # Human footprint PCA alone does not establish body heading. Neutral yaw is
    # explicitly uncertain until a heading cue exists in the request contract.
    yaw = 0.
    lower, upper, observed = measured_bounds(evidence, yaw)
    faces = face_support(evidence, yaw, lower, upper)
    candidates = []
    for prototype in registry.pool('pedestrian'):
        check_cancelled(cancelled)
        median = np.asarray(prototype['median_lwh'])
        low, high = np.asarray(prototype['p10_lwh']), np.asarray(prototype['p90_lwh'])
        if np.any(observed > high + .03):
            continue
        # Ground-to-top height is measured independently of the XY footprint.
        weights = np.array([float(all(faces[axis])) for axis in (0, 1)] + [
            float(len(evidence.points) >= 12 and np.ptp(evidence.above[evidence.selected]) >= .7 * median[2])])
        dimensions = np.clip(weights * observed + (1 - weights) * median, low, high)
        dimensions = np.maximum(dimensions, np.minimum(observed + .02, high))
        center = place_xy(evidence, yaw, lower, upper, dimensions, faces)
        penalty = float(np.sum(np.abs(np.log(np.maximum(observed, .1) / median))))
        candidates.append((penalty, prototype, dimensions, center, weights))
    if not candidates:
        raise NoFit('unsafe-fit', 'No reliable pedestrian prototype matches observed extent',
                    {'diagnosticFlags': ['prior-unavailable'], 'observedLWH': observed.tolist()})
    errors = []
    for _, prototype, dimensions, center, weights in sorted(candidates, key=lambda item: item[0]):
        try:
            return finish(evidence, center, dimensions, yaw, prototype=prototype, prior_version=registry.version,
                          diagnostics={'observedLWH': observed.tolist(), 'dimensionWeights': weights.tolist(),
                                       'visibleSides': faces}, cancelled=cancelled)
        except NoFit as exc:
            if exc.reason == 'cancelled':
                raise
            errors.append(exc)
    raise errors[0]
