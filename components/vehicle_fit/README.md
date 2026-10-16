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
`allowEstimated` now controls only acceptance of an assessed estimate. The
public job API accepts optional integer `timeout_ms` in 1..16000, default 1700;
the private API retains its 16000 ms default. Readiness and budget select
inference independently of quality policy. Budgets up to 2000 ms use only a
prepared SNAP frame and wait at most 200 ms for its GPU lock; cold/busy/error
cases attempt validated geometry estimates, then apply the requested policy.
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

The same harness accepts raw50-manifest.json with `--tracklet-zip` pointing to
the matching frozen tracklet archive. It reports every vehicle, no-fit,
unseedable, confident/estimated quality, exact 3D IoU and wrong-vehicle results
on the first seed with no retry. Prepared private HTTP latency is distinct
from browser gesture-to-cuboid latency; the latter is not measured here.

## Experimental adaptive size refinement

`VEHICLE_FIT_ADAPTIVE_SIZE_REFINEMENT=false` is the default; Compose exposes it
as `CVAT_AUTOFIT_ADAPTIVE_SIZE_REFINEMENT`. Keep it disabled until geometry and
integration acceptance pass. Disabling it rolls back only the additional
length refinement. Instance validation and honest quality classification stay
active. Both geometry and runtime default to the frozen tune medians
`[3.70, 1.61, 1.49]` m; explicit environment overrides remain available.

All paths require real supporting returns around the selected seed, final
oriented containment with 0.05 m tolerance, physical dimension bounds,
support containment and detached-neighbor checks. Mask connectivity alone is
insufficient when parked cars touch: the mask must cover at least 55% of
distinct local non-ground voxels around the seed. Sparse disconnected scan
rings may still belong to the supplied SNAP instance; final geometry must
contain the seed and fit those returns. Safety failures cannot invoke an
estimated/proposed-cuboid rescue. Strict policy rejects an estimate without
changing its assessed quality.
An internal transverse cap with ≥2.6 m of observed length on either side also
rejects a candidate that plausibly joins two compact cars bumper-to-bumper.

The independently implemented observed-end detector requires a cap patch
inside a 0.16 m face band, away from side edges and the upper 0.20 m of the
roof, with sufficient support across width and height. Only one supported end
and stable yaw permit length growth toward the existing prior. Alpha is
`clip((prior_length - observed_length)/0.70, 0, 1)`. The supported face stays
fixed while the opposite end moves; width, height and yaw remain unchanged.
Unknown direction and two supported ends prevent growth. A refined box stays
estimated and passes all safety checks again. No sensor origin is assumed for
extension direction; the existing range-based connectivity radius is retained.

This alpha formula is a CVAT proposal, not an Autoware or OpenBox algorithm.
See the [audit and evaluation report](../../docs/vehicle-size-refinement/README.md)
for exact thresholds, upstream references, completed A/B results and limitations.
The feature is experimental: synthetic improvement does not establish a KITTI
or browser acceptance PASS.

Frozen-mask replay operates without GPU inference or annotation writes:

```bash
PYTHONPATH=components python3 -m vehicle_fit.evaluation.replay \
  --manifest components/vehicle_fit/evidence/object-manifest.json --split eval \
  --inputs components/vehicle_fit/evidence/fast-final-heldout/cases.jsonl \
  --baseline /tmp/vehicle-size-baseline/components/vehicle_fit/geometry.py \
  --output components/vehicle_fit/evidence/vehicle-size-refinement/final-eval
```

Extract baseline source from the reported SHA into a separate directory before
using that command. Replay verifies frozen frame assets including calibration;
RAW additionally requires `--tracklet-zip`. It uses the first frozen seed and
the same historical mask for every variant, never GT geometry as fitter input.
The evaluator flushes each case, separates transport errors and unseedable
cases, and reports coverage, seed containment and common-accepted metrics.

## Recovery after 9277fe1

The [follow-up recovery report](../../docs/vehicle-size-refinement/recovery/README.md)
compares immutable f2bdb5b, baseline+P0, 9277fe1 and cumulative fixes using the
same tune inputs. Protected case IDs and the maximum 0.02 absolute IoU drop
were frozen before tuning. The earlier report remains a historical reference.

Candidate failures no longer reject a request if another candidate passes P0;
global evidence failures remain terminal. Geometry ambiguity among placements
of the same instance stays estimated. The 0.002 score band only limits geometry
comparison, never the scene/neighbor checks. Generation and validation honor
the existing deadline and cancellation; no small top-k cutoff is used.

`diagnostics.candidateCounts` records generated/evaluated/safe candidates and
completion of geometry verification. `dimensionSourceLWH`, `placementSource`
and related fields distinguish measurements, priors and estimated placement.
Complete traces are evaluator-only. Refinement remains disabled by default;
new independent evaluation is blocked until a genuinely unused dataset exists.

## Vehicles label and migration

For new 3D tasks or projects, use [`vehicles-label.json`](vehicles-label.json):
it defines a cuboid label with immutable select attribute `type` and values
`car`, `truck`, `bus`, `van`, and `other`. `other` is the manual drawing
default. Auto Fit requires a deliberate selection in the Draw Cuboid popover.

The migration command is read-only by default. First inspect the selected scope,
then create a protected annotation/schema snapshot before applying changes:

```bash
python manage.py migrate_autofit_vehicles --task-id 123 --mode dry-run
python manage.py migrate_autofit_vehicles --task-id 123 --mode backup \
  --backup /secure/path/task-123-before.json
python manage.py migrate_autofit_vehicles --task-id 123 --mode apply \
  --backup /secure/path/task-123-before.json --confirm-scope task:123
python manage.py migrate_autofit_vehicles --task-id 123 --mode rollback \
  --backup /secure/path/task-123-before.json
```

For a project-wide scope, use `--project-id 456` and confirm `project:456`.
Store the snapshot outside the CVAT database and retain it until reload and
export checks pass. Apply requires a matching unchanged snapshot and explicit
scope confirmation. Rollback stops if the affected annotations or schema changed
after apply.
Both apply (including repeated apply) and rollback verify the stored `before`
snapshot checksum before writing backup status or database rows. A mismatched
checksum is rejected. This detects accidental snapshot edits; it is not a
signature against someone who can replace both the snapshot and its checksum.
Copy/paste preserves the source vehicle type; the drawing preference applies
only to newly drawn cuboids.

## Vehicles implementation evidence

Audit baseline was `e634cbe53` on `feat/auto-fit-pedestrian`; the existing
`.gitignore` edit and untracked `docker-compose.local.yml` were left untouched.
The [acceptance record](vehicles-acceptance.json) contains source hashes,
per-PR gates, and paths to the captured logs and GT snapshots.

**PR1 — Schema and UX:** The complete Chrome acceptance spec passes **33/33 twice**, in independent Cypress processes with **retries=0**, identical source hashes, no skipped tests, and 33 passing Allure results per run. Both Shape/Track paste regressions, failed-fit preservation, and UX16 pass. Final TypeScript, ESLint and production webpack build pass; the build reports two bundle-size warnings and was not deployed. The mock fitter validates UI lifecycle only, not SNAP quality.

Draw Cuboid now controls the popover's `open` state and closes it when drawing starts, instead of hiding an internally open popover with CSS. This prevents stale open/close state on the next draw. Cuboid Shape/Track shortcut tooltips appear below their buttons so they do not cover Auto Fit. Test interactions use one normal opening sequence and retain the original successful-fit-then-rejection scenario, checking cuboid identity, geometry and event counts.

For Auto Fit Cypress specs, the failure listener registered before Allure bounds `actual`/`expected` diagnostics (including DOM and shared object graphs), preserving the error message, stack and failure status. A separate negative probe confirmed that two deliberately failing DOM/shared-graph assertions remain FAIL in Cypress and Allure with their original messages and no `RangeError`. These expected failures are separate from the 33-case acceptance suite.

The backend, fitter, SNAP and migration results below are retained from the earlier passing runs; their source hashes still match. The browser fixes were verified separately in the two new acceptance runs.

**PR2 — Router:** The vehicle fitter passes **78 tests**, including paired
legacy-Car versus Vehicles/car routing through the same fitter and early no-fit
for truck, bus, van, and other. The Django HTTP view was also exercised against
the imported GT in QA: four unsupported types did not read a frame or call the
fitter, four invalid selections returned 400, and car forwarded the real PCD
once to a stubbed service. Stored annotations stayed identical.
The canvas Node suite, type check, lint, and webpack build pass.

After the user rebooted the host, CUDA initialization returned 0 and the real
SNAP regression **passed all 248 paired cases** across the 45 frames containing
Car GT. Every mask and fit/no-fit result matched the legacy Car route; maximum
geometry difference was exactly 0. There were 184 returned cuboids, all marked
`estimated`, and 64 no-fits (55 ambiguous, nine insufficient-evidence). No returned box matched a different GT vehicle.
180/184 returned boxes contained the seed at tolerance 1e-6; four seeds were
1.6–7.2 mm outside, identically in both legacy Car and Vehicles/car routes.
Accepted-box 3D IoU was mean 0.76456, median 0.80443; 152 reached IoU >= 0.70.
Coverage was 74.19%; mean IoU with no-fits scored as zero was 0.56726.
This establishes compatibility with the existing Car fitter. No separate
absolute production-quality threshold was specified; estimated boxes still
require review. Full provenance and metrics are in the
[SNAP evidence](vehicles-snap-evidence.json).

The earlier run was blocked by host CUDA error 999 and kernel Xid 154
(`Node Reboot Required`). After this successful run, the original fitter was restored with the same
image and verified healthy. No driver settings were changed. The runner checks
host CUDA before interrupting a service and always restores the original
container in cleanup.

**PR3 — Migration:** Eight ORM fixture tests cover missing/valid/invalid Car type,
repeat apply, tampered snapshots before any write, task and multi-task project
scopes, backup/apply/rollback, shape and track IDs, geometry, keyframes, mutable
and immutable attributes, and rollback refusal after a later edit. Full GT
import, backup/apply, Datumaro 3D export/import and rollback now **PASS** in the
dedicated `cvat-vehicles-qa` stack (154 frames, 15 tracks). Migration and rollback
preserve exact annotation values and IDs; export/import preserves semantics
with geometry tolerance 1e-6. No production task or project was migrated.
The initial CVAT import contains 722 keyframes (717 visible, five outside),
with original custom attributes and flags preserved. Geometry is rounded to
two decimals by the existing import path before migration (maximum observed
coordinate difference 0.00499933). Migration comparisons use that imported
snapshot; fitter scoring uses the original full-precision GT file.
The 25 current Django adapter and migration tests
passed in a temporary test database, which Django destroyed after the run.

The supplied GT audit found 154 frames, 248 visible Car cuboids, 293 Van
cuboids, and five outside track-termination shapes. All 717 visible cuboids
matched the source tracklet geometry; all 248 Cars have at least one observed
seed. The PCD payload hashes for frames 104–153 and the tracklet XML hash match
the existing RAW50 corpus. This is a regression sample, not an independent
held-out dataset.

The read-only evaluator freezes the input hashes and GT-derived observed seeds,
then compares Vehicles/car against the legacy Car route on identical seeds.
Use the isolated GPU runner after authorizing the temporary service interruption:

```bash
docker build -f components/vehicle_fit/Dockerfile.runtime \
  --build-arg VEHICLE_FIT_RUNTIME_BASE=cvat/vehicle-fit:autofit-fast \
  -t cvat/vehicle-fit:vehicles-qa .
python3 tests/vehicles_gpu_regression.py \
  --dataset ~/Datasets/cvat_drive_0005_3d \
  --output components/vehicle_fit/evidence/vehicles-next-run \
  --allow-service-interruption
```

The completed run is in `evidence/vehicles-post-reboot/` in this workspace.
The evaluator has tests that mark unavailable preparation as **BLOCKED** and
never as a quality pass. The fitter algorithm, model, priors, thresholds, and
fallback were not changed.

Commands validated: `node_modules/.bin/tsc -p cvat-ui/tsconfig.json --noEmit`,
ESLint on changed TypeScript/Cypress files,
`node --test cvat-canvas3d/tests/auto-fit.test.cjs`, Django tests
`cvat.apps.engine.tests.test_auto_fit` and `test_vehicle_migration`, the complete
`components/vehicle_fit` Python suite (**78 passed**), and a development
webpack build into `/tmp`.

Reproduce the isolated browser/backend checks using these helpers:

```bash
docker build -f Dockerfile.autofit-pr3 -t cvat/server:autofit-pr3 .
python3 tests/vehicles_qa_compose.py --output /tmp/cvat-vehicles-qa.yml
docker compose -f /tmp/cvat-vehicles-qa.yml up -d --pull never
docker compose -f /tmp/cvat-vehicles-qa.yml exec -T \
  -e DJANGO_SUPERUSER_PASSWORD=12qwaszx cvat_server python manage.py createsuperuser \
  --noinput --username admin --email admin@localhost.company
# Start the frontend from cvat-ui, in another terminal:
CVAT_AUTOFIT_MOCK=true CVAT_UI_HOST=localhost CVAT_UI_PORT=13000 \
  ../node_modules/.bin/webpack serve --config webpack.config.js \
  --mode development --env API_URL=http://localhost:18080
# From tests, in another terminal:
BASE_URL=http://localhost:13000 ./node_modules/.bin/cypress run --browser chrome \
  --config 'specPattern=cypress/e2e/canvas3d_functionality_2/auto_fit_box_mock.js,video=false,retries=0'
# From the repository root:
python3 tests/vehicles_gt_roundtrip.py --dataset ~/Datasets/cvat_drive_0005_3d \
  --output components/vehicle_fit/evidence/vehicles-gt-roundtrip-new
docker compose -f /tmp/cvat-vehicles-qa.yml exec -T cvat_server \
  python manage.py shell < tests/vehicles_qa_router.py
```

The [browser evidence](vehicles-browser-evidence.json) records both complete 33-case acceptance runs. Logs, source hashes and separate Allure artifacts are retained under `evidence/vehicles-final-browser`; the earlier failures and negative reporter probe remain under `evidence/vehicles-native-pointer`. Run the full browser command twice in separate Cypress processes to reproduce the gate. The migration backup was recovered
from the stopped QA container to `evidence/vehicles-post-reboot/migration-backup.json`;
its rolled-back status and both snapshot checksums were verified. Detailed new
SNAP results are stored in the workspace rather than `/tmp`.
The QA stack is stopped with its volumes retained. The GPU runner requires explicit
`--allow-service-interruption` authorization and restores the original fitter
in a `finally` block; it must never be launched silently on a shared GPU.

## Local Vehicles review deployment

The earlier Vehicles review baseline was deployed at `http://localhost:8080` using
`docker-compose.vehicles-review.yml` after the existing local override.
The frontend is a production build with real SNAP enabled and mock fitting
disabled, stored separately in `cvat-ui/dist/vehicles-review`.
The served JavaScript hash and deployed backend/fitter source hashes were
verified against the build and workspace at that time. The current popover,
tooltip and reporter fixes were built locally and tested in isolated QA only;
they have not been deployed. The startup command checks the
existing database schema without running migrations. Deployment evidence is
in `evidence/vehicles-deployment.json`.

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml \
  -f docker-compose.vehicles-review.yml up -d --no-deps --no-build --pull never \
  cvat_server cvat_ui vehicle_fit
```

For a new 3D test task, use `vehicles-label.json` to expose the vehicle type
selector. Existing Car annotations are not converted by deploying the code.
Reload the browser after deployment to load the new bundle.

To restore the previous local code and UI bundle:

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml \
  up -d --no-deps --no-build --pull never cvat_server cvat_ui vehicle_fit
```
