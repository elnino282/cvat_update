"""Experimental Van route using the PR0 Car–Van prototype pool.

nuScenes vehicle.car has no independent Van subtype ground truth. This adapter
does not infer or rewrite semantic type and never calls or modifies CarFitter.
"""
from .large_vehicle import fit_with_priors
from .priors import REGISTRY
from .shared import evidence_for


def fit_car_van(cloud, seed, *, seed_index, mask=None, cancelled=lambda: False, registry=REGISTRY):
    evidence = evidence_for(cloud, seed, seed_index, mask, 'vehicles/van', cancelled)
    result = fit_with_priors(evidence, 'vehicles/van', registry, cancelled)
    result['diagnostics']['acceptanceGroundTruth'] = 'insufficient-ground-truth'
    result['diagnostics']['priorSourcePool'] = 'car_van'
    return result
