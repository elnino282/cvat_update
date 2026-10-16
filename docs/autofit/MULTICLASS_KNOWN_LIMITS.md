# Experimental multi-class Auto Fit: known limits

These are implementation and interaction findings, not accuracy acceptance. The
new routes return **Estimated / Experimental** boxes when their safety checks
pass. Car continues to use its PR0 fitter and configuration. No benchmark was
expanded for this investigation.

## Prior availability

Completion uses only PR0 records marked `development-prototype`, from the dev
split, with at least eight independent instances. Descriptive P10/P90 ranges
remain experimental and are not calibrated accuracy guarantees.

| Route/pool | Independent instances | Current limitation |
| --- | ---: | --- |
| Pedestrian adult prototypes | 13 / 81 / 86 | Child and other sparse source pools cannot supply completion sizes. |
| Truck prototypes | 13 / 9 | Priors can complete dimensions; they cannot establish whether a detached fragment belongs to the clicked instance. |
| Bus rigid / bendy | 6 / 2 | No reliable completion prior. A box requires independently observed dimensions; incomplete geometry returns `unsafe-fit` with `prior-unavailable`. |
| Van, shared Car–Van pool | 148 / 115 | No independent Van ground truth exists in this corpus. Diagnostic status remains `insufficient-ground-truth`. The one-instance large prototype is excluded. |
| Two-Wheels, no rider | 11 / 12 | Two eligible internal prototype pools; no public bicycle/motorcycle subtype is created. |
| Two-Wheels, rider present | 3 / 3 | No reliable completion prior. Measured vehicle/rider geometry and a safe spatial association are required. Sparse or ambiguous combinations return `unsafe-fit` with `prior-unavailable`. |

Ground support, seed containment and instance association remain mandatory even
when a size prior is available. No prior is used to invent missing ground or to
justify including a neighboring object.

Two-Wheels persists an opaque provenance identifier in `autoFit.prototype_key`:
`"two-wheels:" + sha256(latent_prototype_key.encode("utf-8")).hexdigest()[:12]`.
This mapping is stable for a registry key and can be independently recomputed
from the versioned PR0 registry. Source bicycle/motorcycle names remain internal
diagnostics and are not written as annotation subtype metadata. The selected
`ride_state` remains a boolean task attribute. Geometry-only rider results have
no prototype identifier.

## Truck: fixed local fixture, first real API probe

Evidence is recorded in
[`truck/fixture.json`](../../artifacts/autofit-eval/local-runs/multiclass/truck/fixture.json)
and
[`truck/probes-r1.json`](../../artifacts/autofit-eval/local-runs/multiclass/truck/probes-r1.json).
These local run artifacts are generated evidence, not a portable task ID contract.
The four cases were selected before inference. All four returned safe no-fit:

- Frame 0, dense: detached support could not be associated safely with the seed component.
- Frames 1 and 2, sparse/far: a supported local ground anchor was unavailable.
- Frame 3, adjacent: the candidate failed the competing-instance check. This case was not investigated further.

Only frame 0 was replayed for diagnosis, using the saved SNAP mask and the same
read-only LiDAR cloud. Case token: `013addaa841442d1b941fd11c57f8c27`;
sample token: `0373e9d8886648779c8a73304919ab5f`.

- SNAP returned 1,232 mask points and **did not include the selected seed**.
  The fitter therefore used its existing full-cloud seed-connected fallback.
- That component contained 9,910 points. The first candidate encountered 108
  additional returns, representing 61 occupied 0.08 m voxels, inside its box.
  Their nearest distances to the selected component ranged from 0.253 to 0.828 m.
- As a diagnostic only, comparison with the clicked GT cuboid using a 0.05 m
  tolerance placed all 9,910 selected points and all 108 additional returns
  inside that cuboid. Only 10 of the 1,232 SNAP mask points were inside it.
  No GT information entered fitting or candidate selection.
- The additional returns formed three detached components with 49, 14 and 45
  points. **105 of 108** lay outside the selected support's 3D convex hull by
  more than 0.05 m; the maximum signed violation was approximately 0.613 m.
- An observed XY hull with the observed Z interval also failed to establish
  enclosure: of 103 additional returns within the selected points' XYZ bounds
  plus 0.05 m, only 59 were within the XY hull tolerance.

This demonstrates a segmentation/association limitation on the selected case.
The GT diagnostic suggests fragments of the same annotated truck, but the
single-frame inference evidence does not prove that relationship. A prior-expanded
box cannot serve as instance identity evidence. Increasing the connectivity
radius or relaxing the competing-point threshold would also risk absorbing a
neighbor, so **no association algorithm, threshold, prior or production source
was changed during this investigation**. The existing no-fit result is retained.

Truck has a running class-specific fitter and focused synthetic safety tests;
these four real requests do not demonstrate successful real-data box creation.
Bus and rider-present Two-Wheels can likewise remain no-fit when geometry is
incomplete. They must not be described as accuracy-accepted or fully usable on
these incomplete observations merely because their API routes are available.
