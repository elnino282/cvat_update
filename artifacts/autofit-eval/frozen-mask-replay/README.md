# PR0 frozen observed-mask replay

Final pair: `run-3/` and `run-4/`. Intermediate `run-1/` and `run-2/` were removed from the repository during artifact cleanup; only the final verified source scope is retained. Both final runs use the same 100 Car cases, primary seeds, saved observed masks and configuration from `baseline-snap/`. No production fitter change or GT mask substitution occurred. Restore ignored raw ledgers from `../raw-evidence.tar.gz` as documented in [the evidence README](../README.md) before replay.

The final pair has **exactly identical results for all 100 cases**, including geometry, diagnostics, masks and accuracy metrics. Only measured runtime is excluded from the semantic comparison. Both have SHA256 `e8b8b3a724291af4c36cca50209cd067a70b20acdfd53335cc71c3c96b756dff`.

| Measurement | run-3 | run-4 |
|---|---:|---:|
| Estimated / confident / rejected | 6 / 0 / 94 | 6 / 0 / 94 |
| Rejection: insufficient-evidence / ambiguous | 53 / 41 | 53 / 41 |
| Missing mask / replay error | 0 / 0 | 0 / 0 |
| Median 3D IoU, accepted only (n=6) | 0.518875 | 0.518875 |
| Median BEV IoU, accepted only (n=6) | 0.582686 | 0.582686 |
| Median center error, accepted only | 0.516964 m | 0.516964 m |
| Seed containment / instance association proxy | 6/6 / 6/6 | 6/6 / 6/6 |
| CPU fitting p50 / p95 / p99 | 12.496 / 142.979 / 224.026 ms | 12.526 / 140.478 / 221.843 ms |

Coverage is 6/100; false-confident is undefined (zero confident outputs). Counting unreturned boxes as zero gives median 3D IoU 0. This isolated result is not multi-class acceptance and does not measure interactive latency.

Comparison against the original service ledger preserves all 100 outcomes. Five accepted boxes differ numerically at floating point precision: maximum center component difference `3.11e-15 m`, size difference `8.88e-16 m`, rotation difference zero. Maximum 3D IoU difference is `6.66e-16`. Raw exact-difference case IDs and magnitude are retained in each summary.

**Mask provenance matters:** the service replaces success `maskIndices` with fitter support indices when available, so those six saved masks may already be ground/component filtered. Failure masks retain observed SNAP segmentation. This replays the saved observations, and cannot claim that every input is the original pre-fit SNAP mask. No missing mask is reconstructed from GT. The independent end-to-end SNAP variability evidence remains in `../original-repeatability-check.json`; frozen-mask determinism alone cannot resolve it.

Validation covers the original manifest and selection hashes, complete baseline ledger, dataset JSON source hashes, LiDAR asset and uploaded four-column byte hashes, coordinate convention and rigid transform, request/response frame identity, raw cloud index mapping, and first seed. Only `geometry.py` and `cloud.py` execute and must match the original production hashes; original SNAP/service hashes are retained as provenance. Config comes from the original identity: priors 3.70/1.61/1.49 m, max surface score 0.10, adaptive refinement false, estimated allowed. Every output records source hashes, software versions, all case tokens and failure tokens. Dataset files are read-only.

Reproduce from repository root using new empty output directories:

```bash
PYTHONPATH=components python3 -m unittest vehicle_fit.evaluation.test_nuscenes_replay -v

python3 -m components.vehicle_fit.evaluation.nuscenes_replay \
  --manifest artifacts/autofit-eval/nuscenes_manifest.json.gz \
  --selection artifacts/autofit-eval/held-out-selection.json \
  --inputs artifacts/autofit-eval/baseline-snap/cases.jsonl \
  --identity artifacts/autofit-eval/baseline-snap/identity.json \
  --output /tmp/pr0-frozen-replay-a

python3 -m components.vehicle_fit.evaluation.nuscenes_replay \
  --manifest artifacts/autofit-eval/nuscenes_manifest.json.gz \
  --selection artifacts/autofit-eval/held-out-selection.json \
  --inputs artifacts/autofit-eval/baseline-snap/cases.jsonl \
  --identity artifacts/autofit-eval/baseline-snap/identity.json \
  --output /tmp/pr0-frozen-replay-b \
  --reference /tmp/pr0-frozen-replay-a
```

Seven tests pass, including NaN/singular/reflected/wrong-frame transforms, cloud hash mutation, nonfinite-point raw index preservation, seed/request identity mismatch, invalid/missing mask handling, source/config mismatch and comparison cohort checks. Reference semantic mismatch exits nonzero after saving evidence. Mini repeated-track dependence, GT-assisted seed selection, association-proxy identity and GT-bottom rather than road-plane comparison remain limitations.
