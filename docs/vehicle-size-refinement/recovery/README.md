# Geometry recovery after 9277fe1

This report preserves the earlier 9277fe1 result as a historical reference.
Baseline source is `f2bdb5b28655a191e2764b75c74eb840158426a6`; reference is
`9277fe1e6ce380638b22dff79857278bd71e2e00`. Both remain intact.

## Pre-registered gates and controls

[policy.json](policy.json) and [policy.sha256](policy.sha256) were written before
any geometry fix. Protected-case IoU may fall by at most **0.02 absolute** from
the better baseline+P0/reference result, and must remain at least 0.70. No
threshold was relaxed after seeing results. [protected-cases.json](protected-cases.json)
contains the 16 case IDs and immutable reference metrics; its checksum is
verified before replay. New good cases cannot replace a protected case.

Fresh controls reproduce **32/16, 28/15, 22/16** (accepted/IoU≥0.70). The
baseline+P0 wrapper does not patch the frozen generator, scoring or placement.
Its ledger records the original result, input/final validator results, assessed
quality and exact equality of center/size/rotation for every accepted result.
Four baseline candidates fail P0 neighbor overlap: 000008-car-0,
000008-car-5, 000016-car-0, 000016-car-3. Other reasons remain explicit.

The protected control identity was captured immediately before writing the
control run's final completion marker; the full control ledger was already
flushed. The final `controls/run.json` records completed=true. The protected
manifest and its metrics were not rewritten during tuning.

## Runtime changes

**A — candidate selection:** reject only a candidate that fails its geometry
validation, then continue by score. Global input/instance failures remain
terminal. No small top-k limit exists. The original rank 47 failure is covered
by a frozen tune regression; a CPU fixture requires reaching rank 61.

**B — placement/yaw:** missing cap support no longer eliminates a yaw or all
prior-sized candidates. Midpoint, estimated reference and supported-face
placements are explicit hypotheses. Two supported ends still prevent forcing
a compact complete vehicle up to the prior. Reference placement is an estimate,
not a sensor-origin or visibility claim.

**C — estimator dimensions:** measured/bounded and prior hypotheses coexist.
Compact patches without orientation evidence remain no-fit. A dense broad end
cap with less than 1 m observed depth retains a bounded observed estimate with
unknown body length; a size prior must not choose a different heading. Its
30-point, 1 m, .85 coverage and cap tests reuse existing evidence thresholds.

**D/E — source-aware geometry stability:** same-instance near-ties use measured
width provenance and stable baseline-compatible placements. With incomplete
length and measured width, symmetric width padding is treated as measurement
uncertainty rather than evidence favoring an off-center fit. All choices stay
within the fixed 0.002 geometry ambiguity band and use no GT or case IDs.

**Final axis placement:** the baseline estimator centered each well-observed
axis independently (`observed/dimension≥.75`). Midpoint/reference on both axes
alone omitted this mixed-axis hypothesis and degraded sparse cases. That
hypothesis is restored explicitly. An attempted auxiliary-component variant
did not improve these cases and is not retained in the delivered algorithm.

All successful paths—including estimates and proposed geometry—pass the shared
validator again. `_prepare_evidence` and `_validate_geometry` are unchanged
from 9277fe1 (AST comparison), including their thresholds. Instance checks inspect
the complete evidence/context and do not depend on the score band. The band
only defines which *geometry* alternatives need comparison. Multiple dimensions
or placements for the same retained instance yield estimated geometry
ambiguity, not an invented competing instance.

Cancellation/deadline checks exist in generation, candidate validation,
ambiguity and before publication. Expiring during ambiguity cannot return an
already-safe but incompletely checked candidate. The service rechecks its
deadline after diagnostics/index construction. No deadline was increased.

Provenance records per-axis measured-margin/prior/bound-limited sources,
observed spans and margins, and estimated, observed-bounds or face-supported
placement with explicit face-support flags. It also records prior-blend
provenance if refinement runs. Passing P0 does not prove dimension
accuracy. Unobserved extent stays estimated; acceptance is not increased by
promoting quality to confident.

## Instance matching and dataset roles

The evaluator compares the selected target annotation with **all valid 3D
annotated neighbors** in that frame, excluding DontCare/invalid geometry.
Correct instance means target overlap is positive and exceeds every neighbor
by more than 1e-6. Ties are not correct. The ledger includes target ID/IoU,
neighbor IDs/classes/IoUs and match margin. Expanding the original Car-only
comparison does not change the verified control counts.

GT is evaluated only after the runtime function returns. It is never passed to
candidate generation, ranking, validation or tie-breaking. The frozen seeds
are GT-assisted observed-point seeds, disclosed and identical for all stages.
Historical masks may already be ground-filtered; this is a geometry comparison,
not a new SNAP inference benchmark.

Object 000000–000019 is tune. Object 000020–000099 and RAW50 drive 0005
frames 104–153 are **regression sets**, previously exposed. Other frames of the
same drive do not become an independent holdout. Fresh independent evaluation
is **BLOCKED: no new dataset has been supplied**. Do not rename old data.

## Verification and reproducibility

The three new frozen-data tests failed on 9277fe1 before fixes: shortening,
yaw rotation and rejection despite safe alternatives. CPU tests also cover
same-instance geometry ambiguity, touching neighbors, score-band-independent
neighbor checks, rank 61 search, cancellation before ambiguity completion,
global-failure rejection, JSON serialization and prior-blend provenance.
The real-data tests require the frozen assets; absence is a skip and cannot
establish a quality PASS.

Final focused validation: **70 vehicle-fit tests PASS**, **13 adapter tests
PASS**, and **canvas Node tests PASS**. The expected logged CUDA fault is an
injected recovery test, not an unhandled test failure. No dependency/model
downloads, SNAP/model replacement, deployment, push or merge were performed.

```bash
docker run --rm --network none --read-only --tmpfs /tmp:rw \
  -e VEHICLE_FIT_BACKEND=geometry-only -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/components -e OPENBLAS_NUM_THREADS=1 \
  -v /home/quynhhn/Desktop/cvat_update:/workspace:ro \
  -v /home/quynhhn/Downloads/100_training:/home/quynhhn/Downloads/100_training:ro \
  --workdir /workspace --entrypoint python3 cvat/vehicle-fit:autofit-fast \
  -m unittest discover -s components/vehicle_fit -p 'test_*.py' -v

node --test cvat-canvas3d/tests/auto-fit.test.cjs
```

Recovery replay uses the same existing image, with writable output under the
Git-ignored evidence directory and data mounted read-only. The final matrix
contains controls and every retained cumulative fix. See [replay arguments](replay-arguments.json)
and [freeze.json](freeze.json) for exact hashes/configuration. No-trace runs
measure geometry latency; traced per-stage ledgers provide candidate audit.

The partial `fix-e` prototype failed ledger serialization of a NumPy boolean;
its completion flag remains false and it is excluded from acceptance. The
corrected prototype has a regression ensuring JSON-serializable provenance.
The skin-only loss trial did not satisfy the protected-case gates and was not
selected. Trial and intermediate ledgers remain available, not overwritten.

## Final tune results (VERIFIED)

The final replay completed all 43 frozen inputs. Counts are accepted / correct-instance boxes with IoU ≥ 0.70. All accepted boxes in these runs match the target under the disclosed evaluator. CPU timing excludes cloud loading, SNAP inference, HTTP and UI; it is not production end-to-end latency.

| Stage | Accepted | IoU ≥ 0.70 | Protected failures | Paired length MAE / median (m) | CPU p50 / p95 (ms) |
|---|---:|---:|---:|---|---|
| baseline | 32 | 16 | control | see fixed-cohort ledger below | 42.52 / 52.00 |
| baseline_p0 | 28 | 15 | control | 0.3348 / 0.3000 (n=28) | 78.10 / 195.19 |
| reference9277 | 22 | 16 | control | see fixed-cohort ledger below | 50.56 / 166.24 |
| fix_a | 23 | 16 | 2 | 0.4154 / 0.3500 (n=21; lost=7) | 54.61 / 341.33 |
| fix_b | 23 | 16 | 2 | 0.3567 / 0.3500 (n=21; lost=7) | 58.92 / 398.57 |
| fix_c | 31 | 15 | 3 | 0.3348 / 0.3000 (n=28; lost=0) | 85.68 / 398.85 |
| fix_d | 31 | 16 | 3 | 0.3292 / 0.3000 (n=28; lost=0) | 86.85 / 391.00 |
| fix_e | 31 | 16 | 0 | 0.3328 / 0.3000 (n=28; lost=0) | 88.68 / 396.36 |
| final | 31 | 16 | 0 | 0.3328 / 0.3000 (n=28; lost=0) | 87.42 / 399.80 |

Intermediate A–D fail the frozen gates and are not deliverable. E passes the registered gates but loses substantial IoU on several nonprotected sparse cases; the final per-axis hypothesis removes those declines. The [full summary](final-summary.json) includes every stage, paired cohort membership, losses, rejection causes and candidate statistics. The [per-case ledger](case-ledger.jsonl) records all 387 stage/case outcomes, target/neighbor IoUs, provenance, rejection and validator results.

For baseline and reference comparisons, use the full fixed 28 baseline+P0 IDs rather than each stage’s changing accepted cohort. If a stage loses a member, paired values below describe survivors and cannot pass the paired gate.

| Stage | Retained of 28 | Paired length MAE (m) | Paired median (m) |
|---|---:|---:|---:|
| baseline | 28 | 0.334771 | 0.300000 |
| baseline_p0 | 28 | 0.334771 | 0.300000 |
| reference9277 | 21 | 0.415372 | 0.350000 |
| fix_a | 21 | 0.415372 | 0.350000 |
| fix_b | 21 | 0.356663 | 0.350000 |
| fix_c | 28 | 0.334771 | 0.300000 |
| fix_d | 28 | 0.329226 | 0.300000 |
| fix_e | 28 | 0.332756 | 0.300000 |
| final | 28 | 0.332756 | 0.300000 |

Final retains **28/28** baseline+P0 cases and **16/16** protected cases. Maximum protected IoU drop is **0.01533083**, within the pre-registered 0.02. All 31 returned boxes remain **estimated**; zero are promoted to confident. Wrong-instance accepted = 0, seed outside accepted = 0 (P0 tolerance 0.05 m), timeout = 0, incomplete candidate checks on accepted boxes = 0.

### Every protected case

| Case ID | Better control IoU | Minimum allowed | Final IoU | Drop | Gate |
|---|---:|---:|---:|---:|---|
| object-000003-car-0 | 0.744075 | 0.724075 | 0.744075 | 0.000000 | PASS |
| object-000006-car-2 | 0.744682 | 0.724682 | 0.742339 | 0.002342 | PASS |
| object-000007-car-0 | 0.774293 | 0.754293 | 0.774293 | 0.000000 | PASS |
| object-000008-car-1 | 0.827698 | 0.807698 | 0.817949 | 0.009749 | PASS |
| object-000008-car-3 | 0.869922 | 0.849922 | 0.884979 | -0.015057 | PASS |
| object-000008-car-5 | 0.881198 | 0.861198 | 0.881198 | 0.000000 | PASS |
| object-000009-car-0 | 0.807287 | 0.787287 | 0.818454 | -0.011167 | PASS |
| object-000010-car-0 | 0.714109 | 0.700000 | 0.714109 | 0.000000 | PASS |
| object-000010-car-1 | 0.827790 | 0.807790 | 0.842683 | -0.014893 | PASS |
| object-000010-car-5 | 0.781086 | 0.761086 | 0.781086 | 0.000000 | PASS |
| object-000010-car-6 | 0.735640 | 0.715640 | 0.735640 | 0.000000 | PASS |
| object-000011-car-2 | 0.771667 | 0.751667 | 0.771667 | 0.000000 | PASS |
| object-000011-car-4 | 0.923548 | 0.903548 | 0.908217 | 0.015331 | PASS |
| object-000013-car-0 | 0.771032 | 0.751032 | 0.774411 | -0.003380 | PASS |
| object-000015-car-0 | 0.893048 | 0.873048 | 0.893048 | 0.000000 | PASS |
| object-000019-car-1 | 0.836348 | 0.816348 | 0.836348 | 0.000000 | PASS |

A negative drop means improvement. [Protected results](protected-results.json) retain the results for every intermediate stage, including failures.

### Every final decline relative to baseline+P0

| Case ID | IoU delta | Length-error delta (m) | Length delta (m) |
|---|---:|---:|---:|
| object-000002-car-1 | -0.011006 | 0.000000 | 0.000000 |
| object-000008-car-1 | -0.003353 | 0.000000 | 0.000000 |
| object-000010-car-3 | -0.006403 | 0.000000 | 0.000000 |
| object-000011-car-4 | -0.010968 | -0.046560 | 0.046560 |

No baseline+P0 case is lost. No paired length worsens or becomes shorter. The small decline on 000011-car-4 accompanies a 0.04656 m length-error improvement; it is still reported and must satisfy its independent IoU gate. [changes.json](changes.json) also reports every loss, shorter box, per-axis error increase and quality downgrade against baseline and 9277fe1 for every stage. These comparisons may differ because the reference accepted cohorts differ.

### Candidate work and rejection causes

| Final candidate count per case (43 cases) | p50 | p95 |
|---|---:|---:|
| generated | 105.00 | 1811.20 |
| evaluated | 2.00 | 22.80 |
| safe | 2.00 | 15.60 |
| ambiguityCompared | 1.00 | 14.60 |

Generated counts describe the retained hypothesis pool; evaluated counts are calls to the full candidate P0 validator. [Rank47 audit](rank47-audit.json) preserves the historical first-safe-rank trace, including earlier rejects. The rank 61 fixture and interrupted-ambiguity fixture pass. No top-k or deadline increase was used.

| Final rejection cause | Cases |
|---|---:|
| insufficient-evidence | 4 |
| no-points | 4 |
| no-safe-candidate | 1 |
| seed-cluster | 3 |

The remaining 000016-car-3 case is explicitly no-fit: no candidate passes P0. It is not restored by bypassing instance validation.

## Freeze and remaining acceptance

**TUNE PASS; overall acceptance PARTIAL.** [freeze.json](freeze.json) records source, protocol, protected cases, inputs, runtime image and final ledger hashes. [Sequential patches](replay-arguments.json) preserve each separate cumulative fix. The final metadata audit replay changed no geometry, quality or metric from the preceding acceptance-final run.

[Dataset registry](datasets.json) explicitly identifies tune and previously exposed regression sets. Fresh independent held-out evaluation is **BLOCKED** until independent data exists. The older Object eval and RAW50 results are not new holdout evidence and were not rerun in this recovery. HTTP/UI p50/p95 and fresh-model inference quality remain unmeasured by this geometry replay. No production deployment was performed. When independent data is available, verify the freeze hashes, evaluate that frozen source without tuning, and report the same gates and losses separately.

## Local build update after source freeze

At the user's subsequent request, the frozen source was built into
`cvat/vehicle-fit:recovery-46ba9fa4330f` and activated on the local CVAT
`vehicle_fit` service through the existing `autofit-fast` tag. All five installed
runtime files match the frozen hashes. **70 tests pass on the built image's
installed runtime code**, with no skipped tests; see [build-tests.log](build-tests.log).
The container is healthy, SNAP is model-ready on the existing RTX 5060 GPU,
and CVAT's API is reachable. Checkpoint/source provenance matches the previous
runtime, priors remain 3.70/1.61/1.49 m and adaptive refinement remains disabled.

[build-update.json](build-update.json) records image IDs, health, activation and
rollback commands. The previous image remains available under
`cvat/vehicle-fit:before-recovery-07111e8f6837`. The original freeze record describes
the state before this requested local update and is retained unchanged.
For historical evaluation replay, replace the mutable image tag in the recorded
argv with that previous image ID or backup tag. Independent held-out acceptance
remains BLOCKED; this local build update does not establish new quality results.
