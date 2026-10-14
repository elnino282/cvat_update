# Vehicle size refinement: audit and verification

Baseline: `f2bdb5b28655a191e2764b75c74eb840158426a6`, branch
`feat/lidar-autofit-ux-mock`. Implementation and focused verification performed
2026-10-14. No model/checkpoint changes or runtime deployment were made.

## VERIFIED audit

Baseline references below use `git show <baseline>:<path>`; current links point
to the revised source. The baseline snapshot used by replay is preserved by
its source hash in each run identity.

| Finding | Baseline evidence | Reproduction |
|---|---|---|
| A mask containing the seed did not establish target instance | geometry.py:108; successful return at 219; estimator return at 313 lacked common seed containment | 1548 returns from B plus one A seed returned B confident, with seed 1.135 m outside; direct estimate was also wrong |
| Proposed containment could be bypassed | geometry.py:349–359 fell through to estimator | max surface score .05 rejected the proposed B box, then returned B estimated |
| Prior/face scoring favored a short crop | geometry.py:145,164–175; coverage compared against fitted length at 334–335 | True synthetic length 4.20 m, observed 3.35 m, fitted 3.40 m, score .07864775, quality confident |
| Quality changed with policy | geometry.py:333–337 | Fresh replay of tune object-000002-car-1: 53 saved SNAP returns; observed 1.918521 × 1.376196 m; L=3.40 m vs GT=4.36 m; score .084598436; strict confident vs relaxed estimated on identical geometry |
| Origin sign selected extension direction | geometry.py:158,299–300; no sensor-origin/pose/calibration contract in fitter | Cropping the negative end could extend beyond the supported positive cap at x=12.1 m |
| Defaults were inconsistent | geometry.py:9 and service.py:43 | Geometry 4.20/1.80/1.55; running service/health 3.70/1.61/1.49 m |

SNAP remains the pinned Outdoor adapter: revision
`59fa79b3948e2a9aec4e69efccb1f0677bdb4295`, checkpoint SHA256
`266b00df105b494e9c86d416f4715bf615baf2ba6c830c298f02e3dbebedd8bb`.
Its sigmoid >0.5 mask is mapped from deterministic 0.05 m voxels to raw
indices. Its measured-seed check and predicted IoU do not themselves verify
instance identity or hidden body length.

## Implementation and hypotheses

The shared evidence/validator in [geometry](../../components/vehicle_fit/geometry.py)
guards normal fits, proposed candidates, direct estimates and geometry-only
fallbacks. Seed matching remains 0.02 m; cuboid containment tolerance is
0.05 m, one SNAP voxel. At least eight distinct supporting voxels must belong
to the seed-linked mask cluster. A sparse local patch cannot borrow orientation
from detached mask components. Oversized masks are terminal failures.

Additional adversarial testing found that connectivity and containment together
still accepted a wrong car when two bumpers were 0.05 m apart. A local mask
coverage gate now requires at least 55% of distinct non-ground voxels in the
existing connectivity neighborhood around the seed. The threshold is a new
CVAT safety hypothesis: majority support plus a 5% ambiguity margin, validated
on touching-car regressions. It is not a learned probability or ground-truth
instance oracle.

During tune replay, using connected-component majority alone wrongly rejected
disconnected scan rings of a single car. That rule was removed. A real local
seed cluster, local coverage, final seed containment and geometric support must
jointly justify the result. Context connected to supplied mask returns is
associated with that evidence; other non-ground context is checked for overlap.
This remains conservative and cannot prove correctness for every possible
merged neural mask.

Another synthetic counterexample joined two 2.65 m cars bumper-to-bumper into
a dimensionally plausible long Car, including through geometry-only fallback.
The common validator now rejects a broad interior transverse cap with at least
the existing minimum Car length (2.6 m) on both sides. It reuses the cap support
tests; a genuine 5.4 m single-car synthetic control remains accepted. This is
an additional conservative surface hypothesis, not tracking-based association.

Candidate placement no longer selects sides from coordinate sign. Prior
completion requires an observed supported face; unconstrained symmetric
completion is omitted. The estimator starts from observed dimensions plus the
existing margins rather than forcing every fragment to the median prior.
Quality is assessed once before `allowEstimated`; incomplete/single-cap fits
remain estimated even when their residual is small. A strict rejection retains
assessed quality in diagnostics. Deadline expiry is reported as timeout, while
explicit cancellation stays cancelled.

Refinement is disabled by default. It requires ≥30 supported returns,
observed length/prior ≥.55, observed width/fitted width ≥.85, exactly one cap
and stable yaw. A cap needs a 0.16 m band, 0.16 m side-edge exclusion,
0.20 m roof exclusion, ≥max(5,.04N) points, horizontal span ≥.40 observed width
and vertical span ≥.45 m. Plausible hypotheses within best score+.002 must have
transverse yaw drift ≤.16 m. Roof exclusion and vertical-span thresholds are
geometric hypotheses supported by synthetic tests, not broad empirical proof.

`alpha=clip((L_prior-observed_L)/.70,0,1)` and
`L_new=L_fit+alpha*(L_prior-L_fit)` preserve dimensions above prior. For supported
end sign a and unit heading u, `center_new=center_old-a*(L_new-L_fit)/2*u` keeps
that end fixed. Width, height, yaw and center z remain unchanged. Extension is
blocked when detached support enters the box; an already safe original can be
returned estimated. All other safety failures reject terminally.

The API adds optional `timeout_ms` (1..16000, default 1700); the private default
remains 16000. Budgets ≤2000 ms select the prepared-frame path with ≤200 ms GPU
queue wait, independently of allowEstimated. Fallback is assessed then accepted
or rejected. A strict fallback rejection retains configured top-level provider
and records attempted provider in diagnostics, preserving adapter validation.

## Primary research and attribution

[Autoware L-shape fitting](https://github.com/autowarefoundation/autoware_universe/blob/46b1cc3d34e8ed48cf88a909976cf8a0af44d6b3/perception/autoware_shape_estimation/lib/model/bounding_box.cpp#L57)
supports searching orientation using projected surfaces. Its
[vehicle filter](https://github.com/autowarefoundation/autoware_universe/blob/46b1cc3d34e8ed48cf88a909976cf8a0af44d6b3/perception/autoware_shape_estimation/lib/filter/utils.cpp#L21)
is physical plausibility checking, not target-instance verification.
[correctWithDefaultValue and reference correction](https://github.com/autowarefoundation/autoware_universe/blob/46b1cc3d34e8ed48cf88a909976cf8a0af44d6b3/perception/autoware_shape_estimation/lib/corrector/utils.cpp#L38)
use dimensional rules and rotated center corrections; they do not implement
the proposed alpha blend. Nearest-origin/reference anchors require trustworthy
frame/reference semantics.

[OpenBox paper, §3.2](https://www.arxiv.org/pdf/2512.01352) combines multiple
frames, camera observations, ego poses and tracks. Its
[extension implementation](https://github.com/oliver0922/OpenBox/blob/987df20f135c4cab11d3437833389c711e572372/boxgen/openbox_boxgen/boxes.py#L691)
anchors visible faces relative to an assumed origin. Two adjacent faces extend
along two axes; this does not establish symmetric longitudinal growth. Its
original box-generation code license was not verified; no upstream code was
copied. The implementation here uses independent NumPy/SciPy geometry.

**BLOCKED:** sensor-ray visibility, camera-IoU placement and ego-motion heading
until their inputs and coordinate transforms are explicitly supplied and
verified. Observed-end anchoring does not prove the other end is occluded.

## Evaluation protocol

Freshly replay three variants: baseline source at the stated SHA, revised
safety/quality with refinement off, and revised safety/quality with it on.
Use the same historical mask and first frozen seed, no retry. Saved masks may
already be ground-filtered; these results measure geometry, not fresh SNAP
segmentation or a complete service/UI benchmark. Explicit geometry-only input
rows use each variant's estimator. GT enters only seed provenance and metrics,
never fitter inputs or refinement selection.

Object manifest SHA256:
`9502eef757b57cdac7054f95ae84c573ccd61659c7e4d762b0560bd6a1532c14`.
Tune is 000000–000019 (43 Cars); eval is 000020–000099 (259 Cars). The prior
remains the 43-label tune median 3.70/1.61/1.49 m. No eval labels were used to
choose priors or thresholds. The late local-coverage fix came from an independent
synthetic touching-bumper counterexample; all final replay runs were repeated.

RAW50 has 248 Car tracklet cases and is reported separately. Cloud/image hashes
and tracklet XML hash
`37b2fb315b0ed2cb8c1231fc2a685379ba3bccae25195ce22c68164bfc77072b`
are checked. Object calibration, labels, cloud and image hashes are verified
before metrics. Wrong-instance counts use maximum oriented GT IoU association;
this metric is not a universal proof of instance identity.

Each completed case is flushed to a ledger, including no-fit, unseedable,
missing-mask and transport failure. Progress and completed state are saved.
Dimension, center, yaw, exact oriented IoU and common-accepted metrics are
reported together with coverage. CPU geometry latency excludes model, HTTP and
browser time; memory and new UI deadlines are not measured.

## Verification commands

The immutable baseline snapshot was extracted before implementation:

```bash
mkdir -p /tmp/vehicle-size-baseline
git archive f2bdb5b28655a191e2764b75c74eb840158426a6 components/vehicle_fit \
  | tar -x -C /tmp/vehicle-size-baseline
```

```bash
docker run --rm --network none --read-only --tmpfs /tmp:rw \
  -e VEHICLE_FIT_BACKEND=geometry-only -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/components -e OPENBLAS_NUM_THREADS=1 \
  -v /home/quynhhn/Desktop/cvat_update:/workspace:ro --workdir /workspace \
  --entrypoint python3 cvat/vehicle-fit:autofit-fast \
  -m unittest discover -s components/vehicle_fit -p 'test_*.py' -v

docker run --rm --network none --read-only --tmpfs /tmp:rw \
  -e CVAT_BASE_DIR=/tmp -e DJANGO_SETTINGS_MODULE=cvat.settings.production \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace \
  -v /home/quynhhn/Desktop/cvat_update:/workspace:ro --workdir /workspace \
  --entrypoint python3 cvat/server:autofit-fast \
  -c 'import django, unittest; django.setup(); unittest.main(module="cvat.apps.engine.tests.test_auto_fit", verbosity=2)'

node --test cvat-canvas3d/tests/auto-fit.test.cjs
```

The development Django runner lacked django_extensions in the installed image.
Production settings plus SimpleTestCase/unittest provided the focused adapter
test without a database or service connection. The model image supplied SciPy
absent on the host. No dependency downloads were required.

Replay uses that same model image with repository mounted read-only, baseline
snapshot at `/baseline`, data mounted at their manifest paths and the output
directory mounted writable. Entrypoint commands were:

```bash
python3 -m vehicle_fit.evaluation.replay \
  --manifest components/vehicle_fit/evidence/object-manifest.json --split tune \
  --inputs components/vehicle_fit/evidence/fast-final-tune/cases.jsonl \
  --baseline /baseline/components/vehicle_fit/geometry.py --output /output/acceptance-tune

python3 -m vehicle_fit.evaluation.replay \
  --manifest components/vehicle_fit/evidence/object-manifest.json --split eval \
  --inputs components/vehicle_fit/evidence/fast-final-heldout/cases.jsonl \
  --baseline /baseline/components/vehicle_fit/geometry.py --output /output/acceptance-eval

python3 -m vehicle_fit.evaluation.replay \
  --manifest components/vehicle_fit/evidence/raw50-manifest.json \
  --inputs components/vehicle_fit/evidence/fast-deployed-raw50/cases.jsonl \
  --baseline /baseline/components/vehicle_fit/geometry.py \
  --tracklet-zip /home/quynhhn/Datasets/2011_09_26_drive_0005_tracklets.zip \
  --output /output/acceptance-raw50
```

## Rollback and direct CVAT check

Set `CVAT_AUTOFIT_ADAPTIVE_SIZE_REFINEMENT=false` (service env
`VEHICLE_FIT_ADAPTIVE_SIZE_REFINEMENT=false`) to disable length refinement.
Safety and quality fixes remain active. Full code rollback requires restoring
the recorded baseline; the feature flag does not restore legacy placement.

The existing localhost:8080 runtime was not redeployed, so it cannot validate
these source changes yet. After a reviewed local test deployment, verify health
reports the intended flag/priors, use a separate 3D Car task, select Draw Cuboid
and Auto Fit, and double-click a genuine point. Check one editable annotation,
review hint for estimates, annotation preservation on no-fit, duplicate-click
suppression, cancellation/frame change, Manual Draw, Undo/Redo and Save/Reload.
Measure gesture-to-result latency separately. No production stack rebuild is
part of this delivery.


## Completed results and decision

Final focused verification: **51/51 vehicle-fit tests PASS**, **13/13 adapter tests PASS**, and **canvas Node tests PASS**. Adversarial cases cover detached and touching neighbors, isolated seeds, both return policies, proposed/fallback paths, merged compact cars, final containment, cancellation and late-publication suppression. Refinement-on tests assert actual growth, correct direction and anchor preservation; disabling the feature cannot satisfy them.

Completed frozen geometry replay: **550 cases** across Object tune 43, Object eval 259 and RAW tracklet 248. No missing masks or transport failures occurred in these offline inputs; Object has three unseedable eval cases. Every final run identity matches the delivered geometry source hash.

| Corpus | Variant | Cases | Accepted | Confident / estimated | No-fit / unseedable | Wrong / seed-outside | Refined | IoU ≥.70 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tune | baseline | 43 | 32 | 9 / 23 | 11 / 0 | 0 / 0 | 0 | 16 |
| tune | safety | 43 | 22 | 0 / 22 | 21 / 0 | 0 / 0 | 0 | 16 |
| tune | refined | 43 | 22 | 0 / 22 | 21 / 0 | 0 / 0 | 0 | 16 |
| eval | baseline | 259 | 212 | 87 / 125 | 44 / 3 | 0 / 0 | 0 | 114 |
| eval | safety | 259 | 171 | 0 / 171 | 85 / 3 | 0 / 0 | 0 | 99 |
| eval | refined | 259 | 171 | 0 / 171 | 85 / 3 | 0 / 0 | 5 | 99 |
| raw50 | baseline | 248 | 229 | 105 / 124 | 19 / 0 | 0 / 0 | 0 | 183 |
| raw50 | safety | 248 | 191 | 0 / 191 | 57 / 0 | 0 / 0 | 0 | 154 |
| raw50 | refined | 248 | 191 | 0 / 191 | 57 / 0 | 0 / 0 | 1 | 154 |

Seed containment counts use the documented 0.05 m tolerance. GT matching uses the disclosed maximum-IoU heuristic. These zero counts supplement the deterministic regression assertions; they do not prove universal instance safety.

| Corpus | Variant | Length MAE / median (m) | Center mean (m) | Yaw mean (degrees) | Accepted mean IoU | All-case mean IoU, zeros retained | CPU geometry p50 / p95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| tune | baseline | 0.3773 / 0.3300 | 0.4509 | 16.20 | 0.6454 | 0.4803 | 42.6 / 53.8 |
| tune | safety | 0.4024 / 0.2900 | 0.2735 | 7.28 | 0.7264 | 0.3717 | 49.6 / 165.0 |
| tune | refined | 0.4024 / 0.2900 | 0.2735 | 7.28 | 0.7264 | 0.3717 | 50.7 / 166.2 |
| eval | baseline | 0.3974 / 0.3050 | 0.4202 | 13.11 | 0.6558 | 0.5368 | 35.4 / 50.2 |
| eval | safety | 0.5084 / 0.3800 | 0.3479 | 6.81 | 0.6887 | 0.4547 | 53.0 / 123.9 |
| eval | refined | 0.5064 / 0.3800 | 0.3469 | 6.81 | 0.6892 | 0.4550 | 53.7 / 124.8 |
| raw50 | baseline | 0.3734 / 0.2347 | 0.2664 | 3.20 | 0.7562 | 0.6983 | 34.4 / 51.7 |
| raw50 | safety | 0.4764 / 0.3267 | 0.2725 | 4.99 | 0.7493 | 0.5771 | 58.4 / 232.1 |
| raw50 | refined | 0.4760 / 0.3267 | 0.2723 | 4.99 | 0.7494 | 0.5772 | 58.7 / 232.0 |

Accepted cohorts differ. The table below uses the intersection of accepted case IDs across all variants; width/height errors, adequate cohorts and all counts are in the linked JSON summaries.

| Corpus | Common accepted | Baseline length MAE | Safety length MAE | Refined length MAE |
|---|---:|---:|---:|---:|
| tune | 22 | 0.3929 | 0.4024 | 0.4024 |
| eval | 171 | 0.4137 | 0.5084 | 0.5064 |
| raw50 | 191 | 0.3809 | 0.4764 | 0.4760 |

### Cases where refinement actually changed length

| Corpus / case | Safety length (m) | Refined length (m) | GT length (m) | Length-error change (m) | Direction |
|---|---:|---:|---:|---:|---|
| eval / object-000031-car-0 | 3.4000 | 3.4854 | 3.8100 | -0.0854 | positive-local-x |
| eval / object-000036-car-5 | 3.4000 | 3.4838 | 3.5900 | -0.0838 | negative-local-x |
| eval / object-000047-car-4 | 3.4000 | 3.4902 | 3.8800 | -0.0902 | positive-local-x |
| eval / object-000050-car-0 | 3.4000 | 3.4808 | 4.3400 | -0.0808 | positive-local-x |
| eval / object-000050-car-2 | 3.6693 | 3.6750 | 3.8900 | -0.0057 | positive-local-x |
| raw50 / raw-0000000112-track-5 | 3.4000 | 3.4890 | 4.0411 | -0.0890 | positive-local-x |

Actual synthetic replay of the 4.20 m vehicle shortened by observation: positive-end crop 3.40→3.55 m (absolute length error 0.80→0.65 m); negative-end crop 3.40→3.52857 m (error 0.80→0.67143 m). Both retain the same supported fitted face, width/height/yaw, and estimated quality. See [synthetic results](synthetic-results.json).

### Acceptance

- **P0 PASS for the specified adversarial regression set:** no wrong-instance or seed-outside acceptance; common validation covers direct estimates, proposed candidates and geometry-only paths.
- **P1 FAIL for broad geometry acceptance:** refinement improves a few eligible cases relative to safety-only, but revised placement/estimation loses coverage and increases length error against baseline, including on common accepted cases. Some baseline fits with IoU≥.70 are lost. Strict face evidence often cannot anchor a real sparse single-frame car; removing unconstrained prior completion also leaves shorter estimates. Synthetic gains are insufficient for rollout.
- **P2 PARTIAL:** focused tests, API validation and complete frozen-mask before/after replay pass. Fresh SNAP end-to-end, browser lifecycle on the revised backend, gesture deadlines and memory changes are NOT MEASURED. The live runtime still uses the earlier image.

**Final decision: PARTIAL.** Keep adaptive refinement disabled. This code is reviewable experimental implementation and safety work, not a production geometry-quality acceptance. No thresholds or priors were retuned on eval to make the numbers pass. Restoring only the flag does not restore baseline coverage; candidate placement and safety remain changed.

### Evidence files

- [Tune summary](tune-summary.json), [eval summary](eval-summary.json), [RAW summary](raw50-summary.json): all metrics, adequate subsets and common-accepted comparisons.
- [Tune identity](tune-run.json), [eval identity](eval-run.json), [RAW identity](raw50-run.json): manifest/input/source/protocol hashes and completion state.
- [Changed case details](changed-cases.json), [focused test log](focused-tests.log), [synthetic outputs](synthetic-results.json).
- Full flushed ledgers remain locally in `components/vehicle_fit/evidence/vehicle-size-refinement/acceptance-{tune,eval,raw50}/cases.jsonl`, intentionally Git-ignored.

### Changed files

- Geometry/service and their README; geometry/service tests plus new test_refinement regression.
- CVAT Auto Fit adapter and its focused tests; optional timeout field and independent policy.
- Compose refinement flag; .gitignore exception only for this compact report directory.
- Interactive/CLI evaluation accounting; new replay evaluator and evaluator regression tests.
- This report, compact JSON evidence and test/synthetic artifacts. SNAP adapter, model weights and frontend source are unchanged.
