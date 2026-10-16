# PR0 nuScenes service baseline

Measured private SNAP/fitter endpoint. Not end-to-end UI acceptance.

| Route | Cases | Accepted | median 3D IoU (accepted) | warm HTTP p95 s |
|---|---:|---:|---:|---:|
| vehicles/car | 100 | 8 | 0.5317697188816476 | 0.1007 |
| vehicles/van | 0 | — | insufficient-ground-truth | — |

See identity.json for exact source/checkpoint/config hashes and limitations; cases.jsonl preserves every response, seed and request.
summary.json includes strata, coverage, confidence intervals, containment, identity proxy, false-confident rate, errors and cold/warm timings.
No threshold tuning or production changes were made. Failure token list is failure-tokens.json.
