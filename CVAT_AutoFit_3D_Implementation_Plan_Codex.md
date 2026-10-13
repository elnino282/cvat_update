# CVAT 3D LiDAR — Auto Fit Box

**Implementation plan + execution prompt for Codex**  
Date: 2026-10-13  
Repository: https://github.com/elnino282/cvat_update  
Observed remote branch: `develop` @ `4c40eaad31fbf0b835af1082dbea3ccb01ed2af5` (check local working tree before proceeding)  
Target platform: Ubuntu 22.04, NVIDIA RTX 5060 Laptop GPU 8 GB, `localhost:8080`  
Primary object class: Car  
Initial validation: 50 consecutive KITTI RAW frames; additionally use labeled KITTI 3D Object training examples for quality measurement.

## 1. Product contract (do not reinterpret UX)

Annotator workflow:

1. Click **Draw Cuboid 3D**.
2. Choose label **Car**.
3. See an **Auto Fit Box** checkbox **immediately below Label**; default OFF.
4. Check Auto Fit Box and click existing **Shape** or **Track** button.
5. Move/drag the pointer/seed cuboid onto **actual LiDAR points** of the intended vehicle; no mandatory rectangle selection and no demand for a well-sized initial box.
6. **Double-click** to confirm the seed. This starts exactly one Auto Fit operation.
7. Once the returned cuboid is valid, insert **exactly one** CVAT cuboid annotation with the selected label/object type; do not insert a rough cuboid beforehand and do not require a separate Preview/Apply click.
8. When no seed, error, uncertain output, or cancellation: preserve existing annotations; show a concise retry hint.

When Auto Fit is OFF, preserve all original CVAT Draw Cuboid behavior including normal double-click semantics, redraw/paste, navigation, and other tools. This change must be 3D-only; never show the checkbox in 2D annotation modes. No automatic multiframe tracking, camera fusion, or World Static Freeze in PR1.

### Mock-mode UX indicator
When using the development mock provider, display a small **Mock Auto Fit** indicator in dev/test environments. The mock's dimensions are fixture values, **not evidence that a vehicle was fitted**. Mock mode must be disabled by default for production deploys. On unavailable real AI in production, report that Auto Fit is unavailable rather than silently using a mock and saving a fabricated box.

## 2. Critical source audit (verified in remote `develop`)

- `cvat-ui/src/components/annotation-page/standard-workspace/controls-side-bar/draw-shape-popover.tsx`: existing Label, then Shape/Track. The component already uses `DimensionType` to differentiate 2D and 3D.
- `cvat-ui/src/containers/annotation-page/standard-workspace/controls-side-bar/draw-shape-popover.tsx`: owns selected label/shape/track flow; calls `canvasInstance.draw(...)` and dispatches `onDrawStart(...)`.
- `cvat-ui/src/components/annotation-page/standard-workspace/controls-side-bar/draw-cuboid-control.tsx`: uses shared popover and `Canvas3d`.
- `cvat-canvas3d/src/typescript/canvas3dModel.ts`: `DrawData`, mode transitions, `draw()` and `cancel()`.
- `cvat-canvas3d/src/typescript/canvas3dView.ts`: has existing `dblclick` handler in perspective view which currently calls `canvas.drawn`/`canvas.edited` immediately in `Mode.DRAW` and then cancels draw. Do not add an independent duplicate dblclick handler.
- `cvat-canvas3d/src/typescript/canvas3dView.ts`: `renderRayCaster` currently locates point cloud intersections when drawing; inspect actual `THREE.Points` raycast and nearest point selection. Ensure double-click uses its own current pointer coordinates, not only a previous mousemove.
- `cvat-ui/src/components/annotation-page/canvas/views/canvas3d/canvas-wrapper3D.tsx`: `onCanvasShapeDrawn` constructs `ObjectState` then calls `onCreateAnnotations([objectState])`, and currently assigns the **current frame**. Any asynchronous result must check original task/job/frame before emitting a creation event.
- `tests/cypress/e2e/canvas3d_functionality_2/case_64_canvas3d_functionality_cuboid.js`: existing creation test.
- `tests/cypress/e2e/canvas3d_functionality/case_85_canvas3d_functionality_cuboid_cancel_drawing.js`: existing cancellation/repeat-draw test.
- `tests/cypress/e2e/canvas3d_functionality/case_88_canvas3d_functionality_save_job_remove_annotation.js`: Save/Reload test.
- `tests/cypress_canvas3d.config.js`: Canvas3D Cypress configuration.

**Do not assume local tree matches the GitHub branch.** First run `pwd`, `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, inspect local modifications and confirm actual files. Preserve unrelated changes. Implement on an isolated `feat/lidar-autofit-ux-mock` branch unless such a branch exists already.

## 3. Delivery gates

### PR1 — UX, point picking, mock provider, annotation lifecycle

**Scope only:** UI checkbox, state propagation, seed picking, single asynchronous request, deterministic mock response, exactly-once commit, cancellation, relevant Cypress tests. Do not install PyTorch, integrate SNAP, invent accuracy numbers or change existing real AI code.

#### UI
- `Auto Fit Box` checkbox directly below Label for 3D cuboids only; default OFF. Preserve user choice while continuing drawing in the same annotation session, but default OFF after a fresh session unless current CVAT preference design warrants otherwise.
- Shape and Track use the existing controls. Checkbox value is captured at draw start and should not change mid-flight unexpectedly.
- While processing show nonblocking progress. Avoid a mandatory Preview step. Support Escape/cancel, keyboard and error states.

#### Seed selection
- Use real LiDAR point geometry, not an unverified empty-space ray or only the draw-template cube center.
- Convert the double-click event's client coordinates to local/NDC coordinates then raycast point geometry with an explicit point-hit tolerance calibrated against point size and distance. Return the actual `pointIndex`/3D XYZ used as seed.
- Guard NaN/Infinity/out-of-cloud, click on empty space and scene/frame not ready; do not create annotations when no seed exists.
- Keep point XYZ in the exact coordinate system of the current frame. Do not silently assume it is camera/world coordinates.
- Draw-template geometry must not intercept seed picking.

#### Event lifecycle
- Existing `dblclick` path should branch only when `Mode.DRAW && autoFitEnabled && new cuboid creation`. Preserve vanilla redraw/paste/other operations.
- Under Auto Fit, **do not dispatch `canvas.drawn` until a validated response is received**. Do not set `Mode.IDLE` prematurely if it prevents retry/cancel.
- Use request token/requestId and `inFlight` guard **synchronously** at the gesture entry point (not only a React state update) to protect against duplicate events.
- Snapshot taskId, jobId, frame, labelId, objectType, seed, requestedAt before invoking provider. Discard stale results when any relevant identity or lifetime is invalid (frame change, task/job change, Escape, component unmount, new draw, explicitly changed label/mode).
- Successful commit uses the existing `canvas.drawn` -> `onCanvasShapeDrawn` -> CVAT ObjectState action **exactly once**, with known frame, label and type; save and history must use the existing CVAT mechanisms.
- Error or rejected geometry: zero annotations created; keep/reenter usable Draw mode for retry, with no lingering template or stuck loading state.
- Avoid memory leaks: detach new event listeners and dispose/abort in-flight operations on unmount.

#### Provider interface (conceptual TypeScript shape; adjust to project typing)

```typescript
type FitRequest = {
  requestId: string;
  taskId: number;
  jobId: number;
  frame: number;
  labelId: number;
  objectType: 'shape' | 'track';
  seed: { x: number; y: number; z: number; pointIndex?: number };
  initialCuboid?: { center: [number, number, number]; size: [number, number, number]; rotation: [number, number, number] };
};

type FitResult =
  | { status: 'ok'; requestId: string; frame: number; center: [number, number, number]; size: [number, number, number]; rotation: [number, number, number]; provider: 'mock' | 'snap'; confidence?: number }
  | { status: 'no-fit'; requestId: string; frame: number; reason: 'no-points' | 'ambiguous' | 'insufficient-evidence' | 'unavailable' };

interface AutoFitProvider {
  fit(request: FitRequest, signal?: AbortSignal): Promise<FitResult>;
}
```

Keep `requestId` and frame matching mandatory. Preserve a common interface for swapping mock and real backend. No need to make a new server endpoint for PR1 if a local mocked provider is simpler, provided it is testable and production-gated.

#### Mock specifics
- Deterministic fixture cuboid from a valid seed; return strictly finite numeric fields and positive lengths.
- Configurable delayed success, no-fit, rejection, timeout-like delay, invalid numeric fields (test fixtures only).
- No random response or expensive network dependency; use injectible provider so Cypress/unit tests are deterministic.
- Mock does **not** claim to segment a real vehicle and is never an emergency silent fallback in production.

#### Validation before commit
- Reject wrong requestId/frame, invalid/nonfinite position, rotation and size, any size <= 0, unexpected provider and stale requests.
- Convert to the exact CVAT cuboid point array format using a shared adapter; verify unit convention, x/y/z axis semantics and angle units. Existing canvas code uses `[x,y,z, rx,ry,rz, width,height,depth, 0,...]` and must remain aligned with `ObjectState`.
- Validation must not mutate an existing annotation.

#### Explicit state transitions

```text
IDLE -> DRAW_MANUAL (checkbox OFF) -> normal CVAT save path
IDLE -> DRAW_AUTOFIT_READY (checkbox ON)
DRAW_AUTOFIT_READY -- no hit --> DRAW_AUTOFIT_READY + hint
DRAW_AUTOFIT_READY -- valid double-click --> FITTING
FITTING -- valid matching result --> COMMIT_ONCE -> IDLE or repeat-draw per existing UX
FITTING -- no-fit/error --> DRAW_AUTOFIT_READY + retry hint
FITTING -- Esc/unmount/frame change --> CANCELLED; any late result is ignored
FITTING -- extra double-click --> no-op
```

### PR1 focused test matrix

| ID | Scenario | Expected |
|---|---|---|
| UX01 | Checkbox OFF, 3D | Manual drawing is unchanged |
| UX02 | Checkbox ON and label selected | Checkbox displayed below Label and captured at draw start |
| UX03 | 2D modes | Auto Fit checkbox absent |
| UX04 | Valid point + double-click + successful mock | One request, exactly one cuboid, correct label/type |
| UX05 | Rapid/repeated double-click | Exactly one in-flight operation and one annotation |
| UX06 | Empty-space double-click | No request, no annotation, explanatory retry hint |
| UX07 | No-fit / provider rejects | Zero annotations, UI recovers, user can retry |
| UX08 | Slow mock + Escape | Abort/ignore late response, zero annotations |
| UX09 | Slow mock + frame change | Late response discarded, zero wrong-frame annotations |
| UX10 | Slow mock + label/type change or unmount | No wrong-label/type commit |
| UX11 | Invalid geometry NaN/Infinity/negative dimensions | Rejected, zero annotations |
| UX12 | Shape mode | CVAT Shape, no phantom Track |
| UX13 | Track mode | CVAT Track created at correct frame/ID |
| UX14 | Save -> Reload | Exactly one persisted cuboid, same position/dimensions/rotation |
| UX15 | Undo -> Redo | Exactly one object removed/restored; no duplicate |
| UX16 | Repeated draw + Checkbox OFF after ON | No leaked mode or altered original draw behavior |

Test only relevant unit code + Cypress Canvas3D spec(s); use existing test fixtures and setup. Example for **focused** test after setup/prerequisites verified:

```bash
cd tests
# Confirm versioned yarn install/config and fixture initialization first.
yarn cypress run --headed --browser chrome \
  --config-file cypress_canvas3d.config.js \
  --spec 'cypress/e2e/canvas3d_functionality_2/auto_fit_box_mock.cy.js'
```

Make actual test path match the created spec (existing files are `.js`). Some versions require the Canvas3D setup task to have run; do not claim full runtime PASS from lint/build alone. Do not run the full repository test suite.

**PR1 acceptance:** builds/lints for touched modules; focused UI tests passing in an actual browser; original basic Draw test still passing; no duplicate or stale annotation; Save/Reload and Undo/Redo demonstrated. If browser prerequisites unavailable, mark runtime items NOT VERIFIED (not PASS).

### PR2 — SNAP-Outdoor + geometric cuboid fitting (separate AI service)

Only after PR1 PASS. Create a separate inference service/container, do not bake the model into CVAT UI/server image. Check checkpoint provenance/license and compatibility first.

Model: SNAP-Outdoor from https://github.com/neu-vi/SNAP. Public Python interface (`python_demo.py`) includes `PointCloudData`, `SegmentationModel`, `model.intialize_pointcloud(...)`, `model.extract_backbone_features(...)`, `model.segment(point_cloud_data, prompt_points, text_prompt=None)` returning `masks, text_labels, iou_scores`. Verify actual API/typos against pinned commit before coding. Their README documents Outdoor prompts and KITTI `.bin`, default outdoor grid_size `0.05`.

Pipeline:

```text
request seed + validated frame identity
 -> load/cache KITTI XYZI / CVAT actual point cloud (XYZ, optional intensity)
 -> validate seed coordinate and its presence in point cloud
 -> optional ground filtering (do not delete the vehicle's tires/body)
 -> SNAP interactive mask based on positive seed and optional corrections
 -> choose one instance containing/consistent with seed (do NOT require a detector-predicted box)
 -> reject merged neighboring cars / low-evidence masks
 -> robust geometric fit:
       initialization via PCA/minimal OBB
       road-plane / ground constraints
       vehicle-sized soft priors, yaw alternatives and point-surface fit
       robust outlier handling, partial occlusion-aware size estimate
 -> quality gate
 -> return center, size, rotation, model/version, mask diagnostics
```

The mask identifies observed points, **not** the true full extent of an occluded vehicle. Keep uncertainty; never claim perfect size recovery from a single partial scan.

RTX 5060 Laptop 8GB: SNAP's documented torch 2.4.1/CUDA 12.4 environment may not support Blackwell; do an isolated compatibility spike using an appropriate CUDA 12.8+ PyTorch build and verify `torch.cuda.get_device_capability()`, flash-attn, PyG extensions/spconv, inference memory, warmup, and latency. Avoid assuming checkpoint works unmodified. Run one checkpoint at a time, cache frame features and consider cropped inference without losing the whole car. If the service cannot load/run, stop with blocker evidence rather than fake PASS.

Contract: authenticated/authorized CVAT backend should resolve actual job frame; do not accept arbitrary server filesystem paths from the browser. Use internal service networking, frame/task permissions, request bounds, timeout, cancellation, and structured errors.

### PR3 — Real provider integration

Use same provider/typed contract; replace development mock behind a feature flag. Reuse existing CVAT annotation transaction/history. Ensure correct shape/track, per-frame coordinate systems, no late commits, no duplicate requests, persistence/reload and UI retry. Mock remains available only in development/test fixtures.

### PR4 — KITTI acceptance

A) KITTI RAW: **50 consecutive frames** from known synchronized KITTI RAW sequence, retaining frame alignment and metadata; functional scenarios: near/far cars, partial cars, cars close together, clutter, low points, multiple vehicles, failed seed, frame change during request. KITTI RAW can have tracklets where provided, but do not assume every sequence/frame has ground-truth cuboids.

B) KITTI 3D Object **training split**: use labeled examples + calibration to compute 3D IoU, yaw error, dimension errors, seed-selection success, failure/retry rate, p50/p95 end-to-end latency and 8 GB GPU peak memory. Treat 3D Object frame IDs as a separate evaluation dataset; do not align them with RAW 50-frame sequence by assuming matching file indexes. 3D Object labels are camera-coordinate-based and Velodyne points use another frame: validate conversion of centers, bottom-center convention, dimensions, yaw and calibration before IoU.

Target thresholds (proposed, **not measured**): 3D IoU >= 0.70 for selected, adequately observed cars where correct evaluation is possible; no unintended modification to existing annotations (100%); desired p95 <= 5 seconds after warmup, subject to measured hardware throughput. Publish numerator/denominator and failure cases, not just selected successes; benchmark-free tuning is fine, but acceptance must measure correctness.

Deliver annotated screenshots, fixture IDs, test logs, JSON/CSV for metrics and cases that fail. No broad quality claim from synthetic-only data.

### PR5 — Camera & calibration (later)

Audit that each camera image corresponds to the LiDAR frame, timestamps and intrinsic/extrinsic matrices. Start with **projection-only validation**; compare LiDAR points and reprojected cuboid corners on camera image. Only then use 2D masks to support choice of instance for close/occluded vehicles. If any transform is missing/inconsistent, bypass camera fusion and retain LiDAR-only behavior; never apply an unknown transform.

## 4. Delivery sequence & out-of-scope

- Separate PRs; do not combine UI, GPU stack, KITTI evaluation and calibration into one change.
- No VoxelNeXt refactor, no new default production model, no World Static Freeze, no multi-frame AI tracking, no redesign of 2D tools in PR1.
- Reuse CVAT's built-in Undo/Redo, creation transaction and object IDs; do not independently POST direct annotations from a 3D DOM event.
- Avoid installing SNAP, torch or heavy AI dependencies until PR2.
- Do not rebuild/replace an existing populated production-like CVAT stack blindly. Inspect `docker compose ...` overrides and use existing development frontend/hot-reload arrangement on `localhost:8080` when verified. Report exact run commands.
- Require a short before/after diff, tests actually run, evidence and known limitations for every PR.

## 5. CODEx execution prompt — IMPLEMENT PR1 NOW

> You are working locally in my fork `https://github.com/elnino282/cvat_update` on Ubuntu 22.04. Implement only **PR1: Draw Cuboid 3D Auto Fit Box checkbox + actual LiDAR seed picker + asynchronous deterministic mock provider + exactly-once cuboid creation + focused tests**. Do not implement AI inference, SNAP, fitting, KITTI quality metrics or camera fusion in this PR.
>
> 1. Audit the current local git branch, working tree and actual code before editing. Never overwrite uncommitted changes. Confirm paths, event lifecycle, shape/track and existing Canvas3D tests. Branch from the current intended base on `feat/lidar-autofit-ux-mock` if appropriate.
> 2. Preserve exact UX: Draw Cuboid 3D -> Label -> checkbox Auto Fit Box immediately below Label -> Shape/Track -> move/drag box seed onto genuine LiDAR points -> double-click -> wait for mock -> insert exactly one cuboid. No mandatory rectangular selection, no separate Apply/Preview button, no premature rough box, no change when checkbox OFF.
> 3. Introduce a typed, replaceable provider interface and deterministic dev-only mock with test failure/delay modes. Mock must not be silently enabled in production. The feature must remain easy to connect to a SNAP service in PR2/PR3.
> 4. Extend `DrawData` and use the existing `dblclick` handler in `cvat-canvas3d/src/typescript/canvas3dView.ts`; do not register an unrelated second handler that causes duplicate annotations. Obtain true point-cloud XYZ from a correctly normalized mouse ray; reject empty-space seeds. Do not trust template box center alone.
> 5. Use a synchronous in-flight/request-token guard and capture original task/job/frame/label/objectType. Handle double click races, Esc, provider rejection/timeout, invalid numeric geometry, frame change, task/job change, label/type change, unmount and stale results. Never emit `canvas.drawn` before successful validated fit. On success commit through existing CVAT action/history exactly once; on failure zero annotations and UI can retry.
> 6. Add focused Cypress tests to existing `tests/cypress/e2e/canvas3d_functionality*` conventions and minimal unit tests where helpful. Verify OFF/ON, success, error, duplicate double-click, no-hit, cancel, stale response, Shape/Track, Undo/Redo and Save/Reload. Run only impacted tests plus lint/type/build for touched packages. If the browser cannot run, report NOT VERIFIED; do not label tests PASS just because they compile.
> 7. Do not modify existing unrelated data/tasks, do not install heavy AI libraries, and do not run the entire CVAT test suite. Keep local `localhost:8080` workflow without destroying volumes. Do not push/merge without my authorization; commit locally only if appropriate to current user workflow.
> 8. Deliver: summary of audit findings, files changed, exact UX behavior, how mock mode is enabled, focused test commands/results, screenshots/evidence if available, any remaining blockers, and the next PR2 interface boundaries. The current task ends when PR1 is complete and documented—**stop rather than proceeding automatically to PR2**.
>
> Follow the requirements and test matrix in this file as the acceptance contract. If environment constraints prevent complete runtime validation, implement the safe parts and clearly state what remains unverified.

## 6. Relevant sources

- CVAT 3D annotation: https://docs.cvat.ai/docs/manual/basics/3d-object-annotation/
- CVAT tests: https://docs.cvat.ai/docs/contributing/running-tests/
- SNAP repository and Outdoor python demo: https://github.com/neu-vi/SNAP
- PyTorch 2.7 Blackwell / CUDA 12.8 support: https://pytorch.org/blog/pytorch-2-7/
- Open3D oriented bounding boxes: https://www.open3d.org/docs/latest/python_api/open3d.geometry.OrientedBoundingBox.html
- KITTI RAW: https://www.cvlibs.net/datasets/kitti/raw_data.php
- KITTI 3D Object: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- SUSTechPOINTS interactive grow-box reference: https://github.com/naurril/SUSTechPOINTS/blob/dev-auto-annotate/public/js/lidar.js

**Status at document creation:** research + remote source audit only; no local code edits, runtime build, Cypress run, model inference, or dataset evaluation performed by this assistant.
