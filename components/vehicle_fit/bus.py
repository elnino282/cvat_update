"""Experimental Bus fitter: measured rigid geometry, no sparse-pool completion.

PR0 contains six rigid and two bendy instances. Neither pool is a trustworthy
completion prior. The limits below only reject implausible observed geometry;
they are never used to fill or enlarge a missing dimension.
"""
import numpy as np

from .geometry import NoFit
from .large_vehicle import _reject_bent_body, _surface_error, orientation_candidates
from .shared import check_cancelled, evidence_for, face_support, finish, measured_bounds, rotation

GEOMETRY_MINIMUM_LWH = np.array([4., 1.5, 1.8])
GEOMETRY_MAXIMUM_LWH = np.array([15., 4., 4.8])


def fit_bus(cloud, seed, *, seed_index, mask=None, cancelled=lambda: False):
    evidence = evidence_for(cloud, seed, seed_index, mask, 'vehicles/bus', cancelled)
    points = evidence.points
    unique = np.unique(np.floor(points / .08), axis=0)
    if len(unique) < 40:
        raise NoFit('unsafe-fit', 'Bus lacks independent measured support for three dimensions',
                    {'diagnosticFlags': ['prior-unavailable'], 'safetyGate': 'bus-observability'})
    candidates, rejected = [], []
    for yaw, method, uncertain in orientation_candidates(points, cancelled):
        check_cancelled(cancelled)
        lower, upper, observed = measured_bounds(evidence, yaw)
        faces = face_support(evidence, yaw, lower, upper)
        if np.any(observed < GEOMETRY_MINIMUM_LWH) or np.any(observed > GEOMETRY_MAXIMUM_LWH):
            continue
        heights = evidence.above[evidence.selected]
        roof = points[heights >= observed[2] - .12, :2] @ rotation(yaw)
        roof_observed = len(roof) >= 8 and np.all(np.ptp(roof, axis=0) >= .35 * observed[:2])
        base_observed = np.count_nonzero(heights < .5) >= 4
        if not all(all(pair) for pair in faces) or not roof_observed or not base_observed:
            continue
        # Every dimension comes from measured bounds. Two-centimetre numerical
        # padding is stated explicitly and does not substitute for hidden sides.
        dimensions = observed + .02
        center = ((lower + upper) / 2) @ rotation(yaw).T
        candidates.append((_surface_error(points, yaw), yaw, method, uncertain,
                           dimensions, center, observed, faces))
    if not candidates:
        raise NoFit('unsafe-fit', 'Bus has unobserved dimensions and no trustworthy completion prior',
                    {'diagnosticFlags': ['prior-unavailable'], 'safetyGate': 'bus-observability'})
    for _, yaw, method, uncertain, dimensions, center, observed, faces in sorted(candidates, key=lambda c: c[0]):
        check_cancelled(cancelled)
        try:
            _reject_bent_body(evidence, yaw)
            return finish(evidence, center, dimensions, yaw, mode='geometry', yaw_uncertain=uncertain,
                          diagnostics={'candidateMethod': method, 'observedLWH': observed.tolist(),
                                       'visibleSides': faces, 'dimensionWeights': [1., 1., 1.],
                                       'geometryPaddingMetres': .02, 'completionPriorAvailable': False},
                          cancelled=cancelled)
        except NoFit as exc:
            if exc.reason == 'cancelled':
                raise
            rejected.append(exc)
    raise rejected[0]
