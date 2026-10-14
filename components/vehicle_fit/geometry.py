"""Conservative whole-car hypotheses from measured LiDAR surfaces."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class VehiclePriors:
    length: float = 3.70
    width: float = 1.61
    height: float = 1.49


FACE_BAND = .16
SEED_TOLERANCE = .05  # One SNAP voxel; raw seed matching remains stricter (.02 m).


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


def seed_cluster(xyz, seed, plane, min_points=30, cancelled=lambda: False, starts=None):
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
    pending = [int(start)] if starts is None else np.flatnonzero(starts[indices]).tolist()
    visited[pending] = True
    cursor = 0
    while cursor < len(pending):
        if cursor % 64 == 0 and cancelled():
            raise NoFit("cancelled", "Operation cancelled")
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


@dataclass
class _Evidence:
    xyz: np.ndarray
    seed: np.ndarray
    plane: np.ndarray
    support: np.ndarray
    component: np.ndarray
    above: np.ndarray
    source: str
    seed_mask_coverage: float = 1.0


def _unsafe(gate, detail, **diagnostics):
    # Terminal failures cannot be reinterpreted as estimated geometry.
    raise NoFit("ambiguous", detail, {"safetyGate": gate, **diagnostics})


def _prepare_evidence(cloud, seed, mask, cancelled):
    xyz = np.asarray(cloud, dtype=np.float64)[:, :3]
    seed = np.asarray(seed, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or seed.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(seed).all():
        raise ValueError("Finite cloud and three-coordinate seed required")
    if cancelled():
        raise NoFit("cancelled", "Operation cancelled")
    distance, index = cKDTree(xyz).query(seed)
    if distance > .02:
        raise NoFit("invalid-seed", "Seed does not match a measured cloud point")
    plane = ground_plane(xyz, seed)
    above = xyz[:, 2] - (xyz[:, :2] @ plane[:2] + plane[2])
    if above[index] <= .10:
        raise NoFit("no-points", "Ground is not a car")
    component = seed_cluster(xyz, seed, plane, min_points=8, cancelled=cancelled)
    seed_mask_coverage = 1.0
    if mask is None:
        support, source = component.copy(), "seed-connected"
    else:
        support, source = np.asarray(mask, dtype=bool).copy(), "snap"
        if support.shape != (len(xyz),) or not support[index]:
            _unsafe("mask-seed-mismatch", "Instance mask does not contain the selected return")
        support &= (above > .10) & (above < 3.0)
        radius = float(np.clip(.22 + np.linalg.norm(seed[:2]) * .008, .28, .60))
        local = (np.linalg.norm(xyz-seed, axis=1) <= radius) & (above > .10) & (above < 3.0)
        # Connected bumpers can make the isolated clicked return join a neighbor
        # mask. Require a local majority with a 5% ambiguity margin, on distinct
        # measured voxels rather than density/duplicate counts. This is a CVAT
        # safety hypothesis tested against touching cars, not model confidence.
        local_count = len(np.unique(np.floor(xyz[local]/.05), axis=0))
        masked_count = len(np.unique(np.floor(xyz[local & support]/.05), axis=0))
        seed_mask_coverage = masked_count / max(1, local_count)
        if seed_mask_coverage < .55:
            _unsafe("seed-local-mask", "Mask does not support the selected local surface",
                    seedLocalMaskCoverage=seed_mask_coverage)
        # Check connectivity IN THE MASK: nearby full-scene returns cannot turn
        # one selected point in an unrelated mask into an instance association.
        selected = np.flatnonzero(support)
        try:
            linked = seed_cluster(xyz[selected], seed, plane, min_points=8, cancelled=cancelled)
        except NoFit as exc:
            if exc.reason == "cancelled":
                raise
            _unsafe("seed-cluster", "Mask lacks a real cluster around the selected point")
        retained = selected[linked]
        unique = np.unique(np.floor(xyz[retained] / .05), axis=0)
        if len(unique) < 8:
            _unsafe("seed-cluster", "Mask lacks a real cluster around the selected point")
        if len(retained) < len(selected) and len(retained) < 30 and np.linalg.norm(np.ptp(xyz[retained, :2], axis=0)) < 1.0:
            _unsafe("seed-orientation", "A local patch cannot borrow orientation from detached mask components")
        if np.linalg.norm(np.ptp(xyz[support, :2], axis=0)) > 7 or np.ptp(xyz[support, 2]) > 3.2:
            _unsafe("mask-extent", "Mask joins several vehicles or unsupported structure")
        # LiDAR scan rings can disconnect a single vehicle. A minority seed
        # component alone is not proof of another instance. Keep the neural
        # support, but require the final geometry to contain the seed and fit
        # that support. An isolated seed remains a terminal failure above.
        # Context connected to ANY real neural return belongs to the supplied
        # mask evidence; detached unmasked context is checked as a neighbor.
        component = seed_cluster(xyz, seed, plane, min_points=8,
                                 cancelled=cancelled, starts=support)
    if len(np.unique(np.floor(xyz[support] / .05), axis=0)) < 8:
        _unsafe("seed-cluster", "Selected component lacks eight distinct supported voxels")
    return _Evidence(xyz, seed, plane, support, component, above, source, seed_mask_coverage)


def _faces(evidence, yaw):
    points = evidence.xyz[evidence.support]
    local = points[:, :2] @ _rotation(yaw)
    lower, upper = np.quantile(local, [.01, .99], axis=0)
    heights = evidence.above[evidence.support]
    roof = float(np.quantile(heights, .98))
    supported, counts = [], []
    for axis in (0, 1):
        other = 1 - axis
        core = ((local[:, other] > lower[other] + FACE_BAND) &
                (local[:, other] < upper[other] - FACE_BAND) & (heights < roof - .20))
        axis_support, axis_counts = [], []
        for edge in (lower[axis], upper[axis]):
            patch = core & (np.abs(local[:, axis] - edge) < FACE_BAND)
            count = int(patch.sum())
            broad = (count >= max(5, .04 * len(points)) and
                     np.ptp(local[patch, other]) >= .40 * (upper[other] - lower[other]) and
                     np.ptp(heights[patch]) >= .45)
            axis_support.append(bool(broad))
            axis_counts.append(count)
        supported.append(axis_support)
        counts.append(axis_counts)
    return lower, upper, supported, counts


def _placement(lower, upper, dimensions, supported):
    centre = (lower + upper) / 2
    for axis in (0, 1):
        if dimensions[axis] > upper[axis] - lower[axis] + .16:
            if sum(supported[axis]) != 1:
                return None  # No origin-based or symmetric hidden-side completion.
            a = -1 if supported[axis][0] else 1
            edge = lower[axis] if a == -1 else upper[axis]
            centre[axis] = edge - a * dimensions[axis] / 2
    return centre


def _validate_geometry(result, evidence, cancelled):
    if cancelled():
        raise NoFit("cancelled", "Operation cancelled")
    centre, size, rotation = (np.asarray(result[key], dtype=float) for key in ("center", "size", "rotation"))
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (centre, size, rotation)) or not np.allclose(rotation[:2], 0):
        _unsafe("invalid-geometry", "Malformed cuboid geometry")
    if np.any(size < [2.6, 1.45, 1.0]) or np.any(size > [5.6, 2.3, 2.4]):
        _unsafe("dimension-bounds", "Cuboid exceeds supported Car dimensions")
    relative = evidence.xyz - centre
    relative[:, :2] = relative[:, :2] @ _rotation(rotation[2])
    seed_relative = evidence.seed - centre
    seed_relative[:2] = seed_relative[:2] @ _rotation(rotation[2])
    if np.any(np.abs(seed_relative) > size / 2 + SEED_TOLERANCE):
        _unsafe("seed-outside", "Selected seed is outside the final oriented cuboid")
    outside = np.maximum(np.abs(relative[evidence.support]) - size / 2, 0)
    if (np.linalg.norm(outside, axis=1) > .18).mean() > .08:
        _unsafe("support-contamination", "Final box does not contain its retained instance support")
    # Two compact cars can join at their bumpers and still fit the 5.6 m Car
    # bound. A broad interior transverse cap with at least one minimum Car
    # length (2.6 m) on BOTH sides is evidence against a single body. Reuse the
    # observed-cap support criteria; dimensions alone cannot detect this case.
    points = relative[evidence.support, :2]
    lower, upper = np.quantile(points, [.01, .99], axis=0)
    if upper[0] - lower[0] >= 2 * 2.6:
        heights = evidence.above[evidence.support]
        core = ((points[:, 1] > lower[1]+FACE_BAND) & (points[:, 1] < upper[1]-FACE_BAND) &
                (heights < np.quantile(heights, .98)-.20))
        for plane in np.arange(lower[0]+2.6, upper[0]-2.6+FACE_BAND/2, FACE_BAND/2):
            patch = core & (np.abs(points[:, 0]-plane) < FACE_BAND)
            if (patch.sum() >= max(5, .04*len(points)) and
                    np.ptp(points[patch, 1]) >= .40*(upper[1]-lower[1]) and np.ptp(heights[patch]) >= .45):
                _unsafe("joined-surfaces", "Interior transverse surface supports two joined vehicle bodies")
    inside = np.all(np.abs(relative) <= size / 2 + SEED_TOLERANCE, axis=1)
    detached = inside & ~evidence.component & (evidence.above > .10) & (evidence.above < 3.0)
    if len(np.unique(np.floor(evidence.xyz[detached] / .05), axis=0)) >= 8:
        _unsafe("neighbor-overlap", "Cuboid encloses detached non-ground support")


def _finish(result, evidence, priors, adaptive_refinement, cancelled):
    _validate_geometry(result, evidence, cancelled)
    yaw = result["rotation"][2]
    lower, upper, faces, counts = _faces(evidence, yaw)
    observed = upper - lower
    diagnostics = result.setdefault("diagnostics", {})
    plausible = diagnostics.pop("plausible_yaws", [yaw])
    stable = all(observed[0] * abs(math.sin(angle - yaw)) <= FACE_BAND for angle in plausible)
    original = list(result["size"])
    refinement = {"applied": False, "originalDimensionsLWH": original,
                  "refinedDimensionsLWH": original, "appliedPriorLWH": [priors.length, priors.width, priors.height],
                  "observedCoverageLW": (observed / np.asarray(original[:2])).tolist(),
                  "extensionDirection": "unknown", "alpha": 0.0, "endFaceSupport": faces[0],
                  "endFaceCorePoints": counts[0], "yawStable": stable}
    if adaptive_refinement and original[0] < priors.length and evidence.support.sum() >= 30 and observed[0] / priors.length >= .55 and observed[1] / original[1] >= .85 and sum(faces[0]) == 1 and stable:
        # CVAT proposal; .70 m is the existing length prior uncertainty.
        alpha = float(np.clip((priors.length - observed[0]) / .70, 0, 1))
        length = original[0] + alpha * (priors.length - original[0])
        a = -1 if faces[0][0] else 1
        # Require the base candidate face to coincide with the supported cap.
        anchor = np.asarray(result["center"][:2]) @ _rotation(yaw)
        edge = lower[0] if a == -1 else upper[0]
        if abs(anchor[0] + a * original[0]/2 - edge) <= FACE_BAND and length > original[0]:
            proposed = {**result, "center": list(result["center"]), "size": [length, *original[1:]]}
            shift = -a * (length - original[0]) / 2 * np.array([math.cos(yaw), math.sin(yaw)])
            proposed["center"][:2] = (np.asarray(result["center"][:2]) + shift).tolist()
            try:
                _validate_geometry(proposed, evidence, cancelled)
            except NoFit as exc:
                if exc.diagnostics.get("safetyGate") != "neighbor-overlap":
                    raise
                refinement["blockedReason"] = "neighbor-overlap"
            else:
                result = proposed
                refinement.update(applied=True, refinedDimensionsLWH=list(result["size"]),
                                  extensionDirection="positive-local-x" if a == -1 else "negative-local-x", alpha=alpha)
    diagnostics.update(mask_indices=np.flatnonzero(evidence.support).tolist(), point_count=int(evidence.support.sum()),
                       seedLocalMaskCoverage=evidence.seed_mask_coverage,
                       ground_plane=evidence.plane.tolist(), maskSource=evidence.source,
                       observed_footprint=observed.tolist(), priors={"length": priors.length, "width": priors.width, "height": priors.height},
                       sizeRefinement=refinement)
    coverage = observed / np.asarray(result["size"][:2])
    if result.get("quality") == "estimated" or sum(faces[0]) != 2 or np.any(coverage < .85) or not stable:
        result["quality"] = "estimated"
        diagnostics.setdefault("reviewReason", "partially-observed-footprint")
    else:
        result["quality"] = "confident"
    diagnostics["assessedQuality"] = result["quality"]
    relative = (evidence.xyz[evidence.support, :2] - result["center"][:2]) @ _rotation(yaw)
    faces_distance = np.abs(np.abs(relative) - np.asarray(result["size"][:2])/2)
    diagnostics["finalSurfaceResidualSquaredMetres"] = float(np.mean(np.minimum(faces_distance.min(axis=1), .6)**2))
    _validate_geometry(result, evidence, cancelled)
    return result


def _fit_vehicle(cloud, seed, mask=None, priors=None, cancelled=lambda: False, max_surface_score=0.10, evidence=None):
    if not np.isfinite(max_surface_score) or max_surface_score <= 0:
        raise ValueError("max_surface_score must be finite and positive")
    priors = priors or VehiclePriors()
    evidence = evidence or _prepare_evidence(cloud, seed, mask, cancelled)
    xyz, seed, plane, mask = evidence.xyz, evidence.seed, evidence.plane, evidence.support
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
        _, _, face_support, _ = _faces(evidence, yaw)
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
                placement = _placement(lower, upper, dimensions, face_support)
                centres = [] if placement is None else [placement]
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
    _validate_geometry(proposed_geometry, evidence, cancelled)
    candidate_diagnostics = {"plausible_yaws": [h[5] for h in hypotheses if h[0] <= score + .002]}
    if score > max_surface_score or contamination > 0.08:
        raise NoFit(
            "ambiguous", "Surface residual or mask contamination is too high",
            {"maskPointCount": int(len(points)), "surfaceScore": score,
             "maskContamination": contamination, "observedFootprint": measured.tolist(),
             "proposedDimensionsLWH": [length, width, height],
             "proposedCuboid": proposed_geometry, **candidate_diagnostics},
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
                 "proposedCuboid": proposed_geometry, **candidate_diagnostics},
            )
    yaw = (float(yaw) + math.pi / 2) % math.pi - math.pi / 2
    return {
        "center": [float(centre_xy[0]), float(centre_xy[1]), ground + height / 2],
        "size": [float(length), float(width), float(height)],
        "rotation": [0.0, 0.0, yaw],
        "confidence": float(np.clip(1 - score / max_surface_score, 0, 1)),
        "diagnostics": {
            **candidate_diagnostics,
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


def _estimate_candidate(evidence, priors, cancelled):
    xyz, support, plane = evidence.xyz, evidence.support, evidence.plane
    points = xyz[support]
    if cancelled():
        raise NoFit("cancelled", "Operation cancelled")
    span = np.ptp(points, axis=0)
    if np.linalg.norm(span[:2]) > 7 or span[2] > 3.2:
        _unsafe("component-extent", "Component joins scene clutter or several vehicles")
    if len(points) < 30 and np.linalg.norm(span[:2]) < 1.0:
        raise NoFit("insufficient-evidence", "Sparse patch does not constrain orientation")
    roof = float(np.quantile(evidence.above[support], .98))
    if not .45 <= roof <= 2.4:
        raise NoFit("insufficient-evidence", "Observed height does not support a car")
    _, vectors = np.linalg.eigh(np.cov(points[:, :2], rowvar=False))
    pca = math.atan2(vectors[1, -1], vectors[0, -1])
    candidates = []
    for yaw in [pca, pca + math.pi / 2, *np.arange(0, math.pi, math.pi / 45)]:
        if cancelled():
            raise NoFit("cancelled", "Operation cancelled")
        rotation = _rotation(yaw)
        local = points[:, :2] @ rotation
        lower, upper, face_support, _ = _faces(evidence, yaw)
        measured = upper - lower
        if measured[0] < .35 or measured[1] < .15 or measured[0] > 5.6 or measured[1] > 2.3:
            continue
        # A fallback does not force every fragment up to the median size.
        dimensions = np.clip(measured + [.10, .08], [2.6, 1.45], [5.6, 2.3])
        centre = _placement(lower, upper, dimensions, face_support)
        if centre is None:
            continue
        outside = np.maximum(np.abs(local - centre) - dimensions / 2, 0)
        contamination = float((np.linalg.norm(outside, axis=1) > .18).mean())
        if contamination > .08:
            continue
        residual = np.minimum(np.abs(local - lower), np.abs(local - upper)).min(axis=1)
        score = float(np.mean(np.minimum(residual, .35) ** 2) + contamination * .5)
        candidates.append((score, yaw, centre @ rotation.T, dimensions))
    if not candidates:
        raise NoFit("insufficient-evidence", "No isolated car footprint supports an estimate")
    candidates.sort(key=lambda item: item[0])
    score, yaw, centre, dimensions = candidates[0]
    height = float(np.clip(max(roof + .10, priors.height), 1.0, 2.4))
    ground = float(centre @ plane[:2] + plane[2])
    return {
        "center": [float(centre[0]), float(centre[1]), ground + height / 2],
        "size": [float(dimensions[0]), float(dimensions[1]), height],
        "rotation": [0.0, 0.0, (float(yaw) + math.pi / 2) % math.pi - math.pi / 2],
        "quality": "estimated",
        "diagnostics": {"surface_score": score,
                        "plausible_yaws": [c[1] for c in candidates if c[0] <= score + .002],
                        "reviewReason": "hidden-extent-or-uncertain-fit"},
    }


def estimate_vehicle(cloud, seed, mask=None, priors=None, cancelled=lambda: False,
                     adaptive_refinement=False):
    """Validated uncertain geometry; direct callers cannot bypass instance safety."""
    priors = priors or VehiclePriors()
    evidence = _prepare_evidence(cloud, seed, mask, cancelled)
    return _finish(_estimate_candidate(evidence, priors, cancelled), evidence, priors,
                   adaptive_refinement, cancelled)


def apply_quality_policy(result, allow_estimated):
    if result["quality"] == "estimated" and not allow_estimated:
        diagnostics = dict(result["diagnostics"])
        diagnostics.pop("mask_indices", None)
        raise NoFit("insufficient-evidence", "Estimated geometry requires allowEstimated", diagnostics)
    return result


def fit_vehicle(cloud, seed, mask=None, priors=None, cancelled=lambda: False,
                max_surface_score=.10, allow_estimated=False, adaptive_refinement=False):
    priors = priors or VehiclePriors()
    evidence = _prepare_evidence(cloud, seed, mask, cancelled)
    try:
        result = _fit_vehicle(cloud, seed, priors=priors, cancelled=cancelled,
                              max_surface_score=max_surface_score, evidence=evidence)
    except NoFit as exc:
        if exc.diagnostics.get("safetyGate") or exc.reason not in {"ambiguous", "insufficient-evidence"}:
            raise
        proposed = exc.diagnostics.get("proposedCuboid")
        if proposed and exc.diagnostics.get("maskContamination", 0) <= .08:
            result = {**proposed, "quality": "estimated", "diagnostics": {
                **exc.diagnostics, "reviewReason": exc.detail}}
            result["diagnostics"].pop("proposedCuboid", None)
        else:
            result = _estimate_candidate(evidence, priors, cancelled)
            result["diagnostics"]["reviewReason"] = exc.detail
    result = _finish(result, evidence, priors, adaptive_refinement, cancelled)
    return apply_quality_policy(result, allow_estimated)
