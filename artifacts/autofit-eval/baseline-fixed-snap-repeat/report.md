# PR0 nuScenes service baseline

Measured private SNAP/fitter endpoint. Not end-to-end UI acceptance.

| Route | Cases | Accepted | median 3D IoU (accepted) | warm HTTP p95 s |
|---|---:|---:|---:|---:|
| vehicles/car | 100 | 7 | 0.5355692648916219 | 0.0986 |
| vehicles/van | 0 | — | insufficient-ground-truth | — |

See identity.json for exact source/checkpoint/config hashes and limitations; cases.jsonl preserves every response, seed and request.
summary.json includes strata, coverage, confidence intervals, containment, identity proxy, false-confident rate, errors and cold/warm timings.
No thresholds were tuned. This isolated run uses the documented PR0 nested-pooling shuffle correction; original pre-correction runs are preserved separately. Existing deployed service was not changed. Failure token list is failure-tokens.json.
