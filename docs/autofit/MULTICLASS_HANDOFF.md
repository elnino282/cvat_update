# Local multi-class Auto Fit — experimental handoff

Base: PR0 `7313007d7`. Car geometry/prior configuration is unchanged; the PR0 SNAP
pooling correction is now deployed. These are small interaction checks, not a new
accuracy benchmark. Newly supported routes are experimental.

## Local tasks

All tasks below were created through the API with runtime IDs and belong to the
provided account. Existing user tasks were not changed. Frame numbering is zero-based.

| Checkpoint | Task/job | Suggested frames | State |
| --- | --- | --- | --- |
| Shared / Car | [67 / 64](http://localhost:8080/tasks/67/jobs/64) | 0–1 estimated, 2 known no-fit | Deployed, real API + browser pass |
| Pedestrian | [84 / 81](http://localhost:8080/tasks/84/jobs/81) | 1–2 estimated; 0/3 rejected by safety checks | Deployed, real API + Shape/Track browser pass |
| Truck | [87 / 84](http://localhost:8080/tasks/87/jobs/84) | 0/3 unsafe component association; 1/2 unavailable ground | Enabled; four real no-fit cases, browser safety pass |
| Bus | [88 / 85](http://localhost:8080/tasks/88/jobs/85) | 3 demonstrates unavailable completion prior; 0–2 ground/support failures | Enabled; four real no-fit cases, browser safety pass |
| Van | [89 / 86](http://localhost:8080/tasks/89/jobs/86) | 1 estimated; 0/3 association failures, 2 ground unavailable | Deployed, real API + Shape browser pass; source Car pool, not independent Van GT |
| Two-Wheels | [90 / 87](http://localhost:8080/tasks/90/jobs/87) | 3 estimated with ride_state=false; 1 ride_state=true safely rejected; 0/2 no-fit | Deployed; real API + no-rider lifecycle and rider safety browser pass |

Choose the corresponding label/type, enable Auto Fit and double-click an observed
return. Estimated/Experimental badges record creation provenance, not guaranteed
accuracy after edits. No camera or extra LiDAR sweeps enter inference.

The locked API seeds below locate the intended objects in the source LiDAR frame
(XYZ in metres, rounded here). Full coordinates and finite-cloud indices are in
each retained fixture; browser clicks can pick a nearby visible return instead.

| Route / frame | Seed XYZ | Point index |
| --- | --- | ---: |
| Car / 0 | 10.469, -9.516, -1.652 | 21426 |
| Pedestrian / 1 | -6.415, 27.362, -0.030 | 7415 |
| Truck / 3 | -6.867, 22.279, 0.009 | 7063 |
| Bus / 3 | 3.412, 1.970, -0.297 | 14164 |
| Van / 1 | 11.874, 15.718, -0.952 | 12277 |
| Two-Wheels / 3 | -10.590, -0.024, -1.237 | 34674 |

## Completed checks

- Before metadata migration, PostgreSQL custom-format dump was created and verified
  with `pg_restore --list`. Original Shape/Track/keyframe counts and row hashes are
  identical before/after migration.
- The additive migration only adds `auto_fit` JSON, default `{}`, to shapes/tracks.
  No taxonomy migration was run.
- 26 focused API/metadata tests pass; core metadata lifecycle tests pass.
- 14 focused mocked browser cases pass, including duplicate/late-result guards,
  Shape/Track, Undo/Redo, Save/Reload, ride_state and copy provenance.
- All 8 final XYZI real browser smokes pass: Car Shape; Pedestrian Shape/Track;
  Van Shape; Truck/Bus safety; Two-Wheels no-rider Shape and rider safety. These
  repeat only the same chosen cases after the fixture channel correction.
- Existing Cypress cuboid copy spec only: 3/3 pass. Its temporary owned task was
  removed; no existing task was used.
- Car: three fixed real API cases match PR0 status/quality/reason/geometry exactly;
  actual browser interaction, Undo/Redo and Save/Reload pass. The same three cases
  still match exactly after the final Two-Wheels deployment.
- Pedestrian: four fixed dev cases; two estimated and two safe no-fit. Real Shape
  and Track smoke passes on the moderate-support case, including persistence.
- Truck: four fixed dev cases return unsafe-fit; no successful real Truck box is
  claimed. Browser safety smoke confirms intended-instance seed, retained seed,
  a visible no-fit reason and zero annotations. Investigation found a seed-missing
  SNAP mask plus unproven scan-fragment association; no safety threshold was lowered.
- Bus: four fixed dev cases produce no box; frame 3 explicitly refuses missing
  geometry dimensions without a trustworthy completion prior. Real browser checks
  correct-instance input, friendly reason, retained seed and no annotation creation.
- Van: frame 1 produces an estimated box, with independently checked seed containment
  and dimensions inside the selected reliable Car–Van range. The other three
  fixed cases remain no-fit. Real Shape browser smoke passes with `type=van`,
  metadata, Undo/Redo and Save/Reload. This is interaction evidence, not Van GT accuracy.
- Two-Wheels: frame 3 produces an estimated box with `ride_state=false`; independent
  seed containment, reliable prior bounds and opaque prototype mapping pass.
  SNAP missed the seed in this case; the seed-anchored fallback supplied support.
  Frame 1 refuses an unproven rider/vehicle association with no reliable completion
  prior; frame 0 rejects competing support and frame 2 lacks ground support.
  Corrected XYZI browser smoke passes both no-rider lifecycle (one box, opaque
  prototype ID, Undo/Redo, Save/Reload) and rider safety (retained seed, zero boxes).

Recorded fit POST round-trip times below exclude frame preparation/model startup.
They describe only the fixed local cases, not p95 latency or accuracy acceptance.
The client deadline remains 2 seconds; cold frame preparation can initially return
503. That failed attempt is retained before retrying the same case after readiness.

| Route | Fixed cases | Estimated / no-fit | Fit POST milliseconds, frame order |
| --- | ---: | --- | --- |
| Car, final deployment | 3 | 2 / 1 | 64.7, 158.1, 39.7 |
| Pedestrian | 4 | 2 / 2 | 41.6, 32.4, 38.0, 51.2 |
| Truck | 4 | 0 / 4 | 243.6, 36.2, 32.1, 53.3 |
| Bus | 4 | 0 / 4 | 34.2, 33.4, 31.5, 56.8 |
| Van, source Car pool | 4 | 1 / 3 | 67.8, 51.8, 28.4, 63.6 |
| Two-Wheels | 4 | 1 / 3 | 55.3, 37.2, 33.3, 42.7 |

Real test projection normalizes the camera and dispatches the double-click in one
browser turn to avoid CameraControls resetting an asynchronously changed test
camera. It still runs the production point picker, API and SNAP. The picked return
must match the rendered finite-cloud vertex and lie inside the intended GT instance.
Physical mouse gestures are additionally covered by the mocked browser cases.
The optional `AUTOFIT_SEED_VIEW=close-up` uses one fixed camera offset from the
locked seed, allowing the real picker to resolve its individual return. It neither
changes the seed/point cloud nor searches for a successful fitting outcome.

The real fixture now preserves original XYZI bytes and indices, dropping only the
nuScenes ring channel. During Two-Wheels verification, the earlier browser fixture
was found to write XYZ-only PCD, which zeroed intensity used by SNAP. On the fixed
frame, that changed 34,259 input strengths while leaving XYZ/voxel ordering intact.
That explains the non-equivalent input; earlier XYZ-only browser artifacts are
retained for diagnosis and must not be used as evidence of XYZI replay. The corrected
fixture records source and uploaded-PCD hashes and is checked against the retained
API fixture. No fitter or safety threshold was changed for this correction.

## Runtime and evidence

Compose project `cvat`: base + `docker-compose.local.yml` +
`docker-compose.vehicles-review.yml` + `docker-compose.autofit.multiclass.yml`.
Named volumes remain attached. Server/workers use `cvat/server:multiclass-shared`;
UI assets are `cvat-ui/dist/multiclass-r2`. Final fitter image is
`cvat/vehicle-fit:multiclass-two-wheels`, runtime label `multiclass-two-wheels-r1`.
All five experimental routes are enabled. The final container image IDs, mounts,
UI asset hashes and source hashes are recorded in the ignored
`artifacts/autofit-eval/local-runs/multiclass/runtime-final.json`.

| Artifact | Final SHA256 |
| --- | --- |
| Server/workers image | `ed6cae04cb1d8166df2e1dfb1da84637c7c9360405ebead55ec347634314f5e8` |
| Fitter image | `ac61f376752b0f01f686b6ebe4da6ed47bda48a56e58322ee492696f6db09656` |
| UI asset manifest digest | `2d918047b329c2933948da5eaa8de746da7bad87a1b75616192cc76fdf767f8b` |

The generated `stack.env` beside that manifest contains only nonsecret image,
asset, build and route settings. To reapply this exact local configuration:

```bash
docker compose --env-file artifacts/autofit-eval/local-runs/multiclass/stack.env \
  -p cvat -f docker-compose.yml -f docker-compose.local.yml \
  -f docker-compose.vehicles-review.yml -f docker-compose.autofit.multiclass.yml \
  up -d --no-deps cvat_server cvat_ui vehicle_fit
```

After this migration, keep metadata-aware server/workers in place. To roll back an
experimental fitter, disable its route or restore the fitter/UI image; do not switch
back to an old writer that omits the new non-null JSON fields. Restoring a database
backup would discard later annotations and is never an automatic rollback step.

Retained fixture selections, runtime IDs, source hashes and API responses are under
ignored `artifacts/autofit-eval/local-runs/multiclass/<class>/`. Private backups are
under its `backups/` directory with restricted permissions, excluded from Git and
Docker build context. Browser traces/logs are in the private runtime directory.
Credentials are never committed or copied into image build context.
Final browser payloads, responses, picked vertices and lifecycle JSON are also
retained under `local-runs/multiclass/browser-xyzi/`, with a checksummed summary.
Each response's fitted cloud hash matches its uploaded XYZI PCD hash.

Backup: `before-auto-fit-metadata-20261016T040014Z.dump`, SHA256
`1883a126638724a61d771767aaa1417b1dcfe976d382ea3382cb5ae4db12a59c`.

Recreate a separate small fixture with `tests/multiclass_handoff.py create`; probe
only its fixed cases with `probe`. Existing tasks are never searched by class name
and repurposed. Keep credentials in an external JSON file with mode 0600.

## Changed code paths

1. Shared: `cvat-ui/src/utils/auto-fit.ts` resolves semantic label/attribute IDs;
   the draw popover and `canvas-wrapper3D.tsx` pass them to
   `cvat-canvas3d/src/typescript/autoFit.ts` and `canvas3dView.ts`.
   `cvat/apps/engine/auto_fit.py` validates schema/deadline/coordinates and calls
   the existing SNAP service. `components/vehicle_fit/shared.py` and `priors.py`
   provide seed association, ground, containment and versioned reliable priors.
2. Pedestrian: `components/vehicle_fit/pedestrian.py`, with adult prototype
   completion, ground anchoring and uncertain-yaw provenance.
3. Vehicles: `large_vehicle.py` implements Truck candidates; `bus.py` requires
   observed dimensions without an eligible completion prior; `car_van.py` uses
   the reliable Car–Van pool. `geometry.py` is byte-identical to PR0 for Car.
4. Two-Wheels: `two_wheels.py` uses internal no-rider prototypes or measured
   rider geometry with association checks. The public label has only `ride_state`.
5. Persistence: `cvat-core/src/auto-fit-metadata.ts`, shape/track object state and
   saver carry creation metadata; engine models/serializers, migration `0109`
   and `dataset_manager/task.py` preserve it, including older-client updates.
   The object sidebar displays Estimated/Experimental creation provenance.

P1 camera/multi-frame work remains outside this implementation.

## Focused test commands

Run from the repository root with existing dependencies. These are targeted
checks; no full KITTI/nuScenes benchmark or full repository test suite was run.

```bash
PYTHONPATH=components python3 -m unittest \
  vehicle_fit.tests.test_multiclass \
  vehicle_fit.tests.test_large_vehicle \
  vehicle_fit.tests.test_bus \
  vehicle_fit.tests.test_car_van \
  vehicle_fit.tests.test_two_wheels \
  vehicle_fit.tests.test_service.RuntimeLifecycleTests \
  vehicle_fit.tests.test_pr0_characterization.PR0CoordinateAndRoutingCharacterization -v
node cvat-core/tests/auto-fit-metadata.test.cjs
node cvat-canvas3d/tests/auto-fit.test.cjs
node tests/playwright-autofit/support/contract.test.mjs
python3 -m unittest discover -s tests -p test_multiclass_handoff.py
python3 cvat/apps/engine/tests/test_auto_fit_metadata.py
node_modules/.bin/tsc -p cvat-ui/tsconfig.json --noEmit --incremental false
node_modules/.bin/tsc -p cvat-canvas3d/tsconfig.json --noEmit --incremental false
```

The fitter selection passed 65 tests; the final metadata-only opaque-ID change
was rechecked with its three relevant tests (included in that 65, not additional
unique tests). Core lifecycle: 7; canvas: 19; fixture contract: 5; handoff helper: 6;
pure metadata validator: 2. The 26 API/serializer checks use only `SimpleTestCase`:

```bash
docker exec cvat_server python -c '
import django, unittest
django.setup()
names = [
    "cvat.apps.engine.tests.test_auto_fit",
    "cvat.apps.engine.tests.test_auto_fit_multiclass_api",
    "cvat.apps.engine.tests.test_auto_fit_metadata_serializers",
]
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
raise SystemExit(not result.wasSuccessful())
'
```

For browser runs, provide `CVAT_USERNAME`/`CVAT_PASSWORD` through the process
environment, never command arguments or committed files. The test creates a
separate UUID-named task through the API, discovers IDs and deletes only that
owned temporary task. Retained handoff tasks above are not modified by these tests.

```bash
AUTOFIT_BROWSER_CHANNEL=chrome AUTOFIT_VIDEO=off \
node tests/playwright-autofit/node_modules/playwright/cli.js test \
  -c tests/playwright-autofit/playwright.config.mjs \
  --project=autofit-ui-mocked --reporter=list --output=/tmp/autofit-ui-smoke
```

Real GPU project: use `--project=autofit-real-nuscenes`, with
`AUTOFIT_ACCEPTANCE=1`, `AUTOFIT_MANIFEST` pointing to the existing PR0 manifest,
`AUTOFIT_MANIFEST_SHA256=f4b2fe3beac2fab88c731d08e57487863a56360a54e8021fb45792c9368f0a1c`,
`AUTOFIT_GPU_HEALTH_URL=http://127.0.0.1:8093/health` and an explicit
`AUTOFIT_CASE_ID` from the retained fixture. Set `AUTOFIT_OBJECT_MODE=Track` for
Track, `AUTOFIT_VEHICLE_TYPE=van` for the source Car pool, or
`AUTOFIT_EXPECT_STATUS=no-fit AUTOFIT_EXPECT_REASON=unsafe-fit` for the recorded
safety cases. The default expected status is `ok`; unsupported or timeout does
not satisfy a supported-class safety smoke.

## Known limits

- Bus rigid/bendy and rider=true prior pools fail the minimum independent-instance
  requirement. They cannot fill missing dimensions; insufficient evidence yields
  `unsafe-fit`/`prior-unavailable`.
- Van uses trustworthy Car–Van prototypes but remains `insufficient-ground-truth`
  for independent subtype accuracy.
- Safety rejection can remain conservative for dense mixed masks. No threshold is
  weakened merely to produce a box, and no accuracy claim follows from creation.
- Initial frame preparation can return a transient unavailable response during
  model/stack startup; interactive late results are not committed.
- See [specific data and association limitations](MULTICLASS_KNOWN_LIMITS.md).
