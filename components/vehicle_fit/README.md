# Auto Fit vehicle service

This service accepts point-cloud bytes and a measured LiDAR point from the
authenticated CVAT job endpoint. It returns one cuboid or an explicit no-fit;
it does not write CVAT annotations. The browser creates a cuboid only after
the existing provider validates that result.

## Local build and service

The default geometry-only image is separate from CVAT. SNAP runs in its own
CUDA 12.8 image and reads the Outdoor checkpoint from a read-only bind mount.
The checkpoint is not baked into either application image.

```bash
CVAT_AUTOFIT_SERVICE_DOCKERFILE=components/vehicle_fit/Dockerfile.snap \
CVAT_AUTOFIT_SNAP_CHECKPOINT=/tmp/SNAP_Outdoor.pth \
docker compose -f docker-compose.yml -f docker-compose.autofit.pr3.yml build vehicle_fit

CVAT_AUTOFIT_ENABLED=true \
CVAT_AUTOFIT_SERVICE_DOCKERFILE=components/vehicle_fit/Dockerfile.snap \
CVAT_AUTOFIT_SNAP_CHECKPOINT=/tmp/SNAP_Outdoor.pth \
docker compose -f docker-compose.yml -f docker-compose.autofit.pr3.yml \
  up -d --no-deps cvat_server vehicle_fit
```

The development override exposes service port 8093 on loopback for frozen
dataset replay. Production Compose does not publish that port. Set
`CVAT_AUTOFIT_ENABLED=false` or leave it unset for the production default.
`CVAT_AUTOFIT_BACKEND=geometry-only` selects the controlled fallback; it is
recorded in each result and never reported as SNAP.
The development gate starts at `VEHICLE_FIT_MAX_SURFACE_SCORE=0.10`; this was
raised from 0.075 on the 43-Car development split after observed masks reached
97.4% target-point purity and 99.0% coverage while a supported complete-car
hypothesis at score 0.089 was rejected. The frozen 80-frame split determines
whether that prior and gate generalize. A positive mask alone never overrides
the geometry score or contamination gate.

`GET /health` reports the exact backend, checkpoint/source hashes, device and
initialization error. `POST /v1/fit` accepts multipart metadata plus a KITTI
`.bin` or CVAT `.pcd` cloud. Repeating the same request ID and payload returns
the same result; reusing its ID for a different payload returns HTTP 409.
`DELETE /v1/requests/{requestId}` invalidates pending or running results for the
authenticated job/user scope forwarded by CVAT.

## Data and fitting

SNAP's 0.05 m Outdoor voxelization uses deterministic real-point representatives
and restores its predicted mask to original cloud indices. Geometry fitting
uses that full-resolution mask. Geometry-only mode uses a seed-connected
component after local ground estimation. Both backends use the same L-Shape,
PCA/yaw-hypothesis, vehicle prior and evidence gate. Priors must come from the
frozen KITTI Object development IDs 000000–000019; do not tune them on IDs
000020–000099. Ground truth is used only by the evaluation scripts.

## Focused validation

```bash
PYTHONPATH=components python -m unittest discover \
  -s components/vehicle_fit -p 'test_*.py' -v

python -m components.vehicle_fit.evaluation.cli freeze \
  --object-root /home/quynhhn/Downloads/100_training/training \
  --output components/vehicle_fit/evidence/object-manifest.json \
  --raw-zip /home/quynhhn/Datasets/2011_09_26_drive_0005_sync.zip \
  --tracklet-zip /home/quynhhn/Datasets/2011_09_26_drive_0005_tracklets.zip \
  --raw-output components/vehicle_fit/evidence/raw50-manifest.json
```

`evaluation.cli run --split tune` operates only on the frozen first-20 Object
frames. After freezing the prior values, `--split eval` measures the remaining
80 frames. Each record includes no-fit, unseedable, incorrect-vehicle, retries,
exact oriented 3D IoU, calibration, latency and overlay evidence. The RAW50
Cypress spec separately exercises Auto Fit in the CVAT UI on frames 104–153.

SNAP's public checkpoint page does not state a separate checkpoint license.
Keep the source attribution and report that licensing limitation when redistributing
the checkpoint. Never start with random weights or hide missing active weights.

## Prepared interactive runtime

The fast runtime keeps ordinary double-click behavior unchanged. A preview is
attached only after Draw a cuboid, Auto Fit and Shape/Track are selected. On a
3D frame with Car labels, background preparation uploads its compressed PCD
once, decodes it and warms both SNAP backbone and decoder without showing a
box or entering drawing mode. Hover never sends inference requests.

Authenticated `POST /api/jobs/{id}/auto-fit/prepare` takes taskId, jobId and
frame. `GET` on the same URL adds frameToken and reports preparing/ready/
unavailable. Tokens are scoped to authenticated user, task, job, frame and
segment chunks_updated_date. The service's CPU LRU holds at most 3 frames,
128 MiB and 120 seconds; the GPU retains only one frame. Current-frame
preparation replaces older queued work. Missing tokens can be re-prepared
once within the original gesture deadline.

New fit requests opt in with allowEstimated=true and optionally frameToken.
The canvas deadline is 2 seconds, provider 1.85 seconds and adapter 1.7 seconds
including frame access. The private runtime applies its deadline to owners,
duplicates, GPU queue and result publication. CUDA work already running is
not preempted; cancelled/expired results cannot create annotations.

Results keep status=ok/no-fit and add quality=confident/estimated. CPU fallback
is explicitly provider=geometry-only, never reported as SNAP. Estimated boxes
create normal editable annotations and display a transient review hint; no
new annotation attribute is saved. Passing surface residual checks does not
prove hidden extent: a prior-completed footprint is also marked estimated.
Ground, empty space and sparse local patches without orientation evidence
remain no-fit. Geometry defaults are the frozen development priors
[3.70, 1.61, 1.49] m, with length/width prior uncertainty [.70, .20] m.

`docker-compose.autofit.fast.yml` selects the verified local images, UI bundle
and persistent read-only checkpoint in components/vehicle_fit/weights. This
folder and dist-autofit-fast are ignored by Git. Build the UI from cvat-ui;
Dockerfile.runtime reuses the already installed local CUDA/SNAP image. A clean
machine should first build Dockerfile.snap. The weights remain outside images.

Non-browser first-click evaluation:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=components /tmp/cvat-pointsam-venv/bin/python \
  -m vehicle_fit.evaluation.interactive \
  --manifest components/vehicle_fit/evidence/object-manifest.json --split eval \
  --endpoint http://127.0.0.1:8093 --output /tmp/auto-fit-eval
```

The same harness accepts raw50-manifest.json. It reports every vehicle, no-fit,
unseedable, confident/estimated quality, exact 3D IoU and wrong-vehicle results
on the first seed with no retry. Prepared private HTTP latency is distinct
from browser gesture-to-cuboid latency; the latter is not measured here.
