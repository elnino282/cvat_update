# PR0 nuScenes service baseline

Measured private SNAP/fitter endpoint. Not end-to-end UI acceptance.

| Route | Cases | Accepted | median 3D IoU (accepted) | warm HTTP p95 s |
|---|---:|---:|---:|---:|
| pedestrian | 100 | 0 | None | 0.0108 |
| two-wheels | 97 | 0 | None | 0.0109 |
| vehicles/bus | 100 | 0 | None | 0.0112 |
| vehicles/car | 100 | 6 | 0.5188747186999576 | 0.0990 |
| vehicles/truck | 57 | 0 | None | 0.0117 |
| vehicles/van | 0 | — | insufficient-ground-truth | — |

See identity.json for exact source/checkpoint/config hashes and limitations; cases.jsonl preserves every response, seed and request.
summary.json includes strata, coverage, confidence intervals, containment, identity proxy, false-confident rate, errors and cold/warm timings.
No threshold tuning or production changes were made. Failure token list is failure-tokens.json.
