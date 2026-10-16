"""One public Two-Wheels label, with explicit rider evidence and no subtype.

No-rider completion uses reliable PR0 pools. Rider pools have only three
independent instances each, so rider=True requires measured composite geometry.
"""
import numpy as np

from .geometry import NoFit
from .large_vehicle import fit_with_priors, orientation_candidates
from .priors import REGISTRY
from .shared import (check_cancelled, component_indices, evidence_for, finish,
                     measured_bounds, rotation)


def _unavailable(detail, gate='rider-observability'):
    raise NoFit('unsafe-fit', detail, {'diagnosticFlags': ['prior-unavailable'], 'safetyGate': gate})


def _base_description(points, heights):
    low = points[heights < 1.1]
    if len(low) < 8 or heights.min() >= .65:
        return None
    values, vectors = np.linalg.eigh(np.cov(low[:, :2], rowvar=False))
    if values[-1] < max(values[0], .01) * 2:
        return None
    axis = vectors[:, -1]
    yaw = float(np.arctan2(axis[1], axis[0]))
    local = low[:, :2] @ rotation(yaw)
    lower, upper = local.min(axis=0), local.max(axis=0)
    spans = upper - lower
    if not (.8 <= spans[0] <= 3.2 and .12 <= spans[1] <= .9):
        return None
    return yaw, lower, upper, float(heights[heights < 1.1].max()), float(np.median(local[:, 1]))


def _is_rider_component(points, heights, base):
    if len(points) < 4 or heights.min() < .85 or heights.max() > 2.6:
        return False
    yaw, lower, upper, base_top, centerline = base
    local = points[:, :2] @ rotation(yaw)
    center = np.median(local, axis=0)
    # Vertical separation may be a LiDAR sampling gap. Lateral separation is not
    # evidence of a rider and must not absorb a standing neighbor.
    inside = np.all((local >= lower - .15) & (local <= upper + .15), axis=1)
    return bool(heights.min() - base_top <= .8 and inside.mean() >= .95
                and np.all(center >= lower) and np.all(center <= upper)
                and abs(center[1] - centerline) <= .25
                and np.ptp(local[:, 1]) <= 1.1)


def _merge_rider(evidence, mask, cancelled):
    eligible = evidence.local.copy()
    if mask is not None and np.asarray(mask, bool)[evidence.index]:
        eligible &= np.asarray(mask, bool)
    remaining = eligible & ~evidence.selected
    components = []
    while remaining.any():
        check_cancelled(cancelled)
        if len(components) >= 20:
            _unavailable('Too many competing components near selected cycle', 'competing-instance')
        indices = np.flatnonzero(remaining)
        linked = component_indices(evidence.xyz[indices], 0, evidence.radius, cancelled)
        component = np.zeros(len(eligible), bool)
        component[indices[linked]] = True
        remaining &= ~component
        components.append(component)
    base = _base_description(evidence.points, evidence.above[evidence.selected])
    if base is None:
        # The annotator may click a rider. Find exactly one measured vehicle
        # immediately underneath that seed component before any union is allowed.
        bases = []
        for component in components:
            candidate = _base_description(evidence.xyz[component], evidence.above[component])
            if candidate is not None and _is_rider_component(evidence.points, evidence.above[evidence.selected], candidate):
                bases.append((component, candidate))
        if len(bases) != 1:
            _unavailable('Selected rider has no unique supported vehicle underneath', 'rider-vehicle-association')
        component, base = bases[0]
        evidence.selected |= component
    merged = 0
    for component in components:
        if np.any(component & evidence.selected):
            continue
        if _is_rider_component(evidence.xyz[component], evidence.above[component], base):
            evidence.selected |= component
            merged += 1
    if merged > 2:
        _unavailable('More than two upper components compete for the selected vehicle', 'competing-instance')
    # Reject a connected standing neighbor as well as detached neighbors. Upper
    # support must project over the lower vehicle footprint.
    upper = evidence.selected & (evidence.above >= 1.1)
    if not upper.any() or not _is_rider_component(evidence.xyz[upper], evidence.above[upper], base):
        _unavailable('Upper support is not securely associated with vehicle footprint', 'rider-vehicle-association')
    evidence.diagnostics.update(riderComponentsMerged=merged, selectedPoints=int(evidence.selected.sum()),
                                detachedMaskPoints=int(np.count_nonzero(eligible & ~evidence.selected)))
    return base


def _measured_rider_fit(evidence, mask, cancelled):
    _merge_rider(evidence, mask, cancelled)
    points, heights = evidence.points, evidence.above[evidence.selected]
    if len(np.unique(np.floor(points / .06), axis=0)) < 20:
        _unavailable('Composite lacks measured vehicle and rider extent')
    low_points = points[heights < 1.1]
    proposals = orientation_candidates(low_points, cancelled)
    rejected = []
    for yaw, method, uncertain in proposals:
        check_cancelled(cancelled)
        lower, upper, observed = measured_bounds(evidence, yaw)
        if np.any(observed < [1.2, .3, 1.3]) or np.any(observed > [3.2, 1.3, 2.6]):
            continue
        # Both longitudinal ends and both lateral boundaries of the vehicle must
        # have real support; an isolated torso or one bicycle side cannot fill it.
        local = low_points[:, :2] @ rotation(yaw)
        low_min, low_max = local.min(axis=0), local.max(axis=0)
        spans = low_max - low_min
        supported = []
        for axis in (0, 1):
            pair = []
            for edge in (low_min[axis], low_max[axis]):
                patch = abs(local[:, axis] - edge) < .12
                pair.append(bool(patch.sum() >= 3 and spans[axis] >= .25
                                 and np.ptp(local[patch, 1-axis]) >= .35 * spans[1-axis]))
            supported.append(pair)
        top = points[heights >= observed[2] - .12]
        top_supported = len(np.unique(np.floor(top / .06), axis=0)) >= 3
        body_span = float(np.ptp(heights[heights >= 1.1]))
        if not all(all(pair) for pair in supported) or not top_supported or body_span < .35 or heights.min() > .5:
            continue
        center = ((lower + upper) / 2) @ rotation(yaw).T
        try:
            return finish(evidence, center, observed + .02, yaw, mode='geometry', yaw_uncertain=uncertain,
                          diagnostics={'candidateMethod': method, 'observedLWH': observed.tolist(),
                                       'dimensionWeights': [1., 1., 1.], 'visibleSides': supported,
                                       'geometryPaddingMetres': .02, 'rideState': True,
                                       'completionPriorAvailable': False}, cancelled=cancelled)
        except NoFit as exc:
            if exc.reason == 'cancelled':
                raise
            rejected.append(exc)
    if rejected:
        raise rejected[0]
    _unavailable('Rider/vehicle dimensions are not independently observed; rider prior is insufficient')


def fit_two_wheels(cloud, seed, *, seed_index, ride_state, mask=None, cancelled=lambda: False, registry=REGISTRY):
    if type(ride_state) is not bool:
        raise NoFit('invalid-coordinates', 'rideState must be a boolean')
    evidence = evidence_for(cloud, seed, seed_index, mask, 'two-wheels', cancelled)
    if ride_state:
        return _measured_rider_fit(evidence, mask, cancelled)
    result = fit_with_priors(evidence, 'two-wheels', registry, cancelled, ride_state=False)
    result['diagnostics']['rideState'] = False
    return result
