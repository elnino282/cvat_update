"""Conservative whole-car hypotheses from measured LiDAR surfaces."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class VehiclePriors:
    length: float = 4.20
    width: float = 1.80
    height: float = 1.55


class NoFit(Exception):
    def __init__(self, reason, detail, diagnostics=None):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.diagnostics = diagnostics or {}


def ground_plane(xyz, seed):
    local = xyz[np.linalg.norm(xyz[:, :2] - seed[:2], axis=1) < 15]
    if len(local) < 100:
        raise NoFit("insufficient-evidence", "No supported local ground plane")
    # Lowest measured return per spatial cell avoids fitting a plane through roofs.
    cells = np.floor(local[:, :2] / 1.5).astype(np.int64)
    order = np.lexsort((local[:, 2], cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    first = np.r_[True, (sorted_cells[1:] != sorted_cells[:-1]).any(axis=1)]
    low = local[order[first]]
    rng = np.random.default_rng(17)
    best = None
    for _ in range(100):
        if len(low) < 3:
            break
        sample = low[rng.choice(len(low), 3, replace=False)]
        matrix = np.column_stack((sample[:, :2], np.ones(3)))
        if abs(np.linalg.det(matrix)) < 0.1:
            continue
        coefficients = np.linalg.solve(matrix, sample[:, 2])
        if np.linalg.norm(coefficients[:2]) > 0.20:
            continue
        distances = np.abs(low[:, 2] - (low[:, :2] @ coefficients[:2] + coefficients[2]))
        inliers = distances < 0.12
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = score, coefficients, inliers
    if best is None or best[0] < max(15, len(low) * 0.30):
        raise NoFit("insufficient-evidence", "Local ground estimate is unstable")
    coefficients = np.linalg.lstsq(np.column_stack((low[best[2], :2], np.ones(best[0]))), low[best[2], 2], rcond=None)[0]
    return coefficients


def seed_cluster(xyz, seed, plane, min_points=30):
    above = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])
    local = (np.linalg.norm(xyz[:, :2] - seed[:2], axis=1) <= 8) & (above > 0.10) & (above < 3.0)
    indices = np.flatnonzero(local)
    if len(indices) < min_points:
        raise NoFit("insufficient-evidence", "Too few non-ground points near seed")
    points = xyz[indices]
    tree = cKDTree(points)
    distance, start = tree.query(seed)
    if distance > 0.08:
        raise NoFit("no-points", "Seed is ground or outside supported vehicle points")
    radius = float(np.clip(0.22 + np.linalg.norm(seed[:2]) * 0.008, 0.28, 0.60))
    visited = np.zeros(len(points), dtype=bool)
    pending = [int(start)]
    visited[start] = True
    cursor = 0
    while cursor < len(pending):
        neighbours = tree.query_ball_point(points[pending[cursor]], radius)
        new = [index for index in neighbours if not visited[index]]
        visited[new] = True
        pending.extend(new)
        cursor += 1
        if len(pending) > 20000:
            raise NoFit("ambiguous", "Seed component is oversized or joins scene clutter")
    mask = np.zeros(len(xyz), dtype=bool)
    mask[indices[visited]] = True
    return mask


def _rotation(yaw):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]])


def _fit_vehicle(cloud, seed, mask=None, priors=None, cancelled=lambda: False, max_surface_score=0.10):
    if not np.isfinite(max_surface_score) or max_surface_score <= 0:
        raise ValueError("max_surface_score must be finite and positive")
    priors = priors or VehiclePriors()
    xyz = np.asarray(cloud[:, :3], dtype=np.float64)
    seed = np.asarray(seed, dtype=np.float64)
    distance, seed_index = cKDTree(xyz).query(seed)
    if distance > 0.02:
        raise NoFit("no-points", "Seed does not match a measured cloud point")
    plane = ground_plane(xyz, seed)
    if seed[2] - (seed[:2] @ plane[:2] + plane[2]) <= .10:
        raise NoFit("no-points", "Ground is not a car")
    if cancelled():
        raise NoFit("cancelled", "Operation cancelled")
    if mask is None:
        mask = seed_cluster(xyz, seed, plane)
    else:
        mask = np.asarray(mask, dtype=bool).copy()
        if mask.shape != (len(xyz),) or not mask[seed_index]:
            raise NoFit("ambiguous", "Instance mask is not consistent with seed")
        height = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])
        mask &= height > 0.10
    points = xyz[mask]
    if len(points) < 30:
        raise NoFit("insufficient-evidence", "Fewer than 30 supported vehicle returns")
    span = np.ptp(points, axis=0)
    if np.linalg.norm(span[:2]) > 7.0 or span[2] > 3.2:
        raise NoFit("ambiguous", "Mask joins multiple vehicles or unsupported structure")
    heights = points[:, 2] - (points[:, :2] @ plane[:2] + plane[2])
    roof = float(np.quantile(heights, 0.98))
    if not 0.8 <= roof <= 2.4 or roof < priors.height - 0.45:
        raise NoFit("insufficient-evidence", "Observed height is not supported as Car",
                    {"observedHeight98Metres": roof, "vehiclePriorHeightMetres": priors.height})
    # L-shape face residual, rather than area alone, proposes observed-side orientations.
    angles = np.arange(0, math.pi / 2, math.pi / 90)
    proposals = []
    for angle in angles:
        local = points[:, :2] @ _rotation(angle)
        lower, upper = np.quantile(local, [0.01, 0.99], axis=0)
        residual = np.minimum(np.abs(local - lower), np.abs(local - upper)).min(axis=1)
        proposals.append((float(np.mean(np.minimum(residual, 0.35) ** 2)), angle))
    proposals.sort()
    covariance = np.cov(points[:, :2], rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    pca = math.atan2(vectors[1, -1], vectors[0, -1])
    yaws = [angle for _, angle in proposals[:4]] + [pca]
    yaws = [angle + turn for angle in yaws for turn in (0, math.pi / 2)]
    hypotheses = []
    for yaw in yaws:
        rotation = _rotation(yaw)
        local = points[:, :2] @ rotation
        lower, upper = np.quantile(local, [0.01, 0.99], axis=0)
        measured = upper - lower
        if measured[0] > 5.6 or measured[1] > 2.3 or min(measured) < 0.35:
            continue
        lengths = sorted(set([max(2.6, measured[0] + 0.10), priors.length, priors.length - 0.30, priors.length + 0.30]))
        widths = sorted(set([max(1.45, measured[1] + 0.08), priors.width, priors.width - 0.12, priors.width + 0.12]))
        for length in lengths:
            for width in widths:
                if not 2.6 <= length <= 5.6 or not 1.45 <= width <= 2.3:
                    continue
                if measured[0] > length + 0.12 or measured[1] > width + 0.12:
                    continue
                # Require observed support in both footprint axes; priors alone cannot fill a scan fragment.
                if measured[0] / length < 0.55 or measured[1] / width < 0.40:
                    continue
                dimensions = np.array([length, width])
                centres = [0.5 * (lower + upper)]
                visible = np.where(0.5 * (lower + upper) > 0, lower + dimensions / 2, upper - dimensions / 2)
                centres.append(visible)
                for centre in centres:
                    relative = local - centre
                    outside = np.maximum(np.abs(relative) - dimensions / 2, 0)
                    contamination = float((np.linalg.norm(outside, axis=1) > 0.18).mean())
                    faces = np.abs(np.abs(relative) - dimensions / 2)
                    residual = np.minimum(faces[:, 0], faces[:, 1])
                    # Height-return support is scored separately, preventing roof-only seeds.
                    supported_x = faces[:, 0] < 0.16
                    supported_y = faces[:, 1] < 0.16
                    if min(supported_x.sum(), supported_y.sum()) < max(5, len(points) * 0.04):
                        continue
                    # Development-set quartiles show tight width variation and
                    # modest length variation; keep the soft prior useful when
                    # a partial scan leaves a hidden side unconstrained.
                    prior_penalty = ((length - priors.length) / 0.70) ** 2 + ((width - priors.width) / 0.20) ** 2
                    score = float(np.mean(np.minimum(residual, 0.6) ** 2) + contamination * 0.5 + prior_penalty * 0.02)
                    world = centre @ rotation.T
                    height = float(np.clip(max(roof + 0.10, priors.height), 1.0, 2.4))
                    z_ground = float(world @ plane[:2] + plane[2])
                    hypotheses.append((score, world, length, width, height, yaw, contamination, measured))
    if cancelled():
        raise NoFit("cancelled", "Operation cancelled")
    if not hypotheses:
        raise NoFit(
            "insufficient-evidence",
            "Observed surfaces do not constrain a whole-car hypothesis",
            {"maskPointCount": int(len(points)), "observedFootprint": measured.tolist(),
             "observedHeight95Metres": roof},
        )
    hypotheses.sort(key=lambda item: item[0])
    best = hypotheses[0]
    score, centre_xy, length, width, height, yaw, contamination, measured = best
    ground = float(centre_xy @ plane[:2] + plane[2])
    proposed_geometry = {
        "center": [float(centre_xy[0]), float(centre_xy[1]), ground + height / 2],
        "size": [float(length), float(width), float(height)],
        "rotation": [0.0, 0.0, float(yaw)],
    }
    if score > max_surface_score or contamination > 0.08:
        raise NoFit(
            "ambiguous", "Surface residual or mask contamination is too high",
            {"maskPointCount": int(len(points)), "surfaceScore": score,
             "maskContamination": contamination, "observedFootprint": measured.tolist(),
             "proposedDimensionsLWH": [length, width, height],
             "proposedCuboid": proposed_geometry},
        )
    # Equally plausible, materially different placements are evidence of hidden-extent ambiguity.
    for alternative in hypotheses[1:]:
        if alternative[0] - score > 0.002:
            break
        if np.linalg.norm(alternative[1] - centre_xy) > 0.50 or abs(alternative[2] - length) > 0.70:
            raise NoFit(
                "ambiguous", "Competing whole-car hypotheses remain unresolved",
                {"bestSurfaceScore": score, "alternativeSurfaceScore": alternative[0],
                 "centerDistanceMetres": float(np.linalg.norm(alternative[1] - centre_xy)),
                 "bestLengthMetres": length, "alternativeLengthMetres": alternative[2],
                 "proposedCuboid": proposed_geometry},
            )
    yaw = (float(yaw) + math.pi / 2) % math.pi - math.pi / 2
    return {
        "center": [float(centre_xy[0]), float(centre_xy[1]), ground + height / 2],
        "size": [float(length), float(width), float(height)],
        "rotation": [0.0, 0.0, yaw],
        "confidence": float(np.clip(1 - score / max_surface_score, 0, 1)),
        "diagnostics": {
            "mask_indices": np.flatnonzero(mask).tolist(),
            "point_count": int(mask.sum()),
            "ground_plane": plane.tolist(),
            "surface_score": score,
            "surface_residual_and_contamination": score - .02 * (
                ((length - priors.length) / .70) ** 2 + ((width - priors.width) / .20) ** 2),
            "prior_uncertainty_lw": [.70, .20],
            "mask_contamination": contamination,
            "observed_footprint": measured.tolist(),
            "hypothesis_count": len(hypotheses),
            "priors": {"length": priors.length, "width": priors.width, "height": priors.height},
        },
    }


def estimate_vehicle(cloud, seed, mask=None, priors=None, cancelled=lambda: False):
    """Editable, explicitly uncertain whole-car geometry from measured support.

    Never make a prior-only box: the seed, local road and a non-ground component
    must support it. Keep disconnected SNAP returns unless the mask is oversized.
    """
    priors = priors or VehiclePriors()
    xyz = np.asarray(cloud[:, :3], dtype=np.float64)
    seed = np.asarray(seed, dtype=np.float64)
    distance, index = cKDTree(xyz).query(seed)
    if distance > .02:
        raise NoFit("invalid-seed", "Seed does not match a measured cloud point")
    plane = ground_plane(xyz, seed)
    above = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])
    if above[index] <= .10:
        raise NoFit("no-points", "Ground is not a car")
    if mask is None:
        support = seed_cluster(xyz, seed, plane, min_points=8)
        mask_source = "seed-connected"
    else:
        mask_source = "snap"
        support = np.asarray(mask, dtype=bool).copy()
        if support.shape != (len(xyz),) or not support[index]:
            raise NoFit("ambiguous", "Mask does not contain the selected point")
        support &= above > .10
        if 0 < support.sum() < 30:
            # A sparse neural mask can be an isolated part rather than a car.
            # Rebuild measured support from the seed instead of extending that
            # fragment several metres into a neighbouring vehicle.
            support = seed_cluster(xyz, seed, plane, min_points=8)
            mask_source = "seed-connected"
        elif support.sum() and (np.linalg.norm(np.ptp(xyz[support, :2], axis=0)) > 7 or
                              np.ptp(xyz[support, 2]) > 3.2):
            # Isolate a seed-supported candidate, rather than enclosing two cars.
            support &= seed_cluster(xyz, seed, plane, min_points=8)
    points = xyz[support]
    if len(points) < 8 or cancelled():
        raise NoFit("cancelled" if cancelled() else "insufficient-evidence", "Insufficient vehicle support")
    span = np.ptp(points, axis=0)
    if np.linalg.norm(span[:2]) > 7 or span[2] > 3.2:
        raise NoFit("ambiguous", "Component joins scene clutter or several vehicles")
    if len(points) < 30 and np.linalg.norm(span[:2]) < 1.0:
        raise NoFit("insufficient-evidence", "Sparse local patch does not constrain a car orientation")
    roof = float(np.quantile(above[support], .98))
    if not .45 <= roof <= 2.4:
        raise NoFit("insufficient-evidence", "Observed height does not support a car")
    _, vectors = np.linalg.eigh(np.cov(points[:, :2], rowvar=False))
    pca = math.atan2(vectors[1, -1], vectors[0, -1])
    candidates = []
    for yaw in [pca, pca + math.pi / 2, *np.arange(0, math.pi, math.pi / 45)]:
        rotation = _rotation(yaw)
        local = points[:, :2] @ rotation
        lower, upper = np.quantile(local, [.01, .99], axis=0)
        measured = upper - lower
        if measured[0] < .35 or measured[1] < .15 or measured[0] > 5.6 or measured[1] > 2.3:
            continue
        dimensions = np.maximum(measured + [.10, .08], [priors.length, priors.width])
        dimensions = np.clip(dimensions, [2.6, 1.45], [5.6, 2.3])
        midpoint = (lower + upper) / 2
        visible = np.where(midpoint > 0, lower + dimensions / 2, upper - dimensions / 2)
        centre = np.where(measured / dimensions >= .75, midpoint, visible)
        outside = np.maximum(np.abs(local - centre) - dimensions / 2, 0)
        contamination = float((np.linalg.norm(outside, axis=1) > .18).mean())
        if contamination > .08:
            continue
        residual = np.minimum(np.abs(local - lower), np.abs(local - upper)).min(axis=1)
        score = float(np.mean(np.minimum(residual, .35) ** 2) + contamination * .5)
        candidates.append((score, yaw, centre @ rotation.T, dimensions))
    if not candidates:
        raise NoFit("ambiguous", "No isolated car footprint supports an estimate")
    _, yaw, centre, dimensions = min(candidates, key=lambda item: item[0])
    height = float(np.clip(max(roof + .10, priors.height), 1.0, 2.4))
    ground = float(centre @ plane[:2] + plane[2])
    return {
        "center": [float(centre[0]), float(centre[1]), ground + height / 2],
        "size": [float(dimensions[0]), float(dimensions[1]), height],
        "rotation": [0.0, 0.0, (float(yaw) + math.pi / 2) % math.pi - math.pi / 2],
        "quality": "estimated",
        "diagnostics": {"mask_indices": np.flatnonzero(support).tolist(), "point_count": int(support.sum()),
                        "ground_plane": plane.tolist(), "maskSource": mask_source,
                        "reviewReason": "hidden-extent-or-uncertain-fit"},
    }


def fit_vehicle(cloud, seed, mask=None, priors=None, cancelled=lambda: False,
                max_surface_score=.10, allow_estimated=False):
    try:
        result = _fit_vehicle(cloud, seed, mask=mask, priors=priors, cancelled=cancelled,
                              max_surface_score=max_surface_score)
        result["quality"] = "confident"
        # Passing the surface gate is not evidence for unseen vehicle extents.
        # Keep the fitted geometry, but request human review when a substantial
        # part of either footprint dimension is supplied by the vehicle prior.
        if allow_estimated:
            observed = np.asarray(result["diagnostics"]["observed_footprint"])
            if np.any(observed / np.asarray(result["size"][:2]) < .85):
                result["quality"] = "estimated"
                result["diagnostics"]["reviewReason"] = "partially-observed-footprint"
        return result
    except NoFit as exc:
        if not allow_estimated or exc.reason not in {"ambiguous", "insufficient-evidence"}:
            raise
        proposed = exc.diagnostics.get("proposedCuboid")
        if cancelled():
            raise NoFit("cancelled", "Operation cancelled")
        if proposed and exc.diagnostics.get("maskContamination", 0) <= .08:
            # Retain the best surface-supported whole-car hypothesis when the
            # uncertainty is residual/hidden-placement, rather than recomputing
            # a poorer PCA box. Never promote it to a confident fit.
            centre, size = np.asarray(proposed["center"]), np.asarray(proposed["size"])
            yaw = proposed["rotation"][2]
            relative_xy = (np.asarray(seed)[:2] - centre[:2]) @ _rotation(yaw)
            if (np.isfinite(centre).all() and np.isfinite(size).all()
                    and np.all(size >= [2.6, 1.45, 1.0]) and np.all(size <= [5.6, 2.3, 2.4])
                    and np.all(np.abs(relative_xy) <= size[:2] / 2 + .25)
                    and abs(seed[2] - centre[2]) <= size[2] / 2 + .15):
                return {**proposed, "quality": "estimated", "diagnostics": {
                    **exc.diagnostics, "reviewReason": exc.detail,
                }}
        result = estimate_vehicle(cloud, seed, mask=mask, priors=priors, cancelled=cancelled)
        result["diagnostics"]["reviewReason"] = exc.detail
        return result
