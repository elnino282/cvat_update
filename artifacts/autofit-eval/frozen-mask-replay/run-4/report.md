# Frozen observed-mask CPU replay

Cases: 100; accepted: 6.
Outcomes: {'insufficient-evidence': 53, 'estimated': 6, 'ambiguous': 41}.
Accepted 3D IoU: {'count': 6, 'p50': 0.5188747186999576, 'p95': 0.7384719253188641, 'p99': 0.7582016436758587}.
CPU geometry seconds: {'count': 100, 'p50': 0.012526249499842379, 'p95': 0.14047761319980054, 'p99': 0.22184259040028392}.
Semantic SHA256 (excluding runtime only): e8b8b3a724291af4c36cca50209cd067a70b20acdfd53335cc71c3c96b756dff.
Original service comparison: {'changed_outcomes': [], 'changed_geometry': ['29724345dcd24e39a3e23e4004c85645', '5fa0a7e3bdf3451b9e0cbb32b099876d', '13d99250a6554a0a9202e96ac6ae2077', '0aa17afa86354013a7531a1b7382e075', '0e4b9792818148449c5a69840fc593b9'], 'changed_metrics': ['29724345dcd24e39a3e23e4004c85645', '5fa0a7e3bdf3451b9e0cbb32b099876d', '13d99250a6554a0a9202e96ac6ae2077', '0aa17afa86354013a7531a1b7382e075', '0e4b9792818148449c5a69840fc593b9'], 'max_absolute_geometry_delta': {'center': 3.1086244689504383e-15, 'size': 8.881784197001252e-16, 'rotation': 0.0}}.
Repeat comparison: {'changed_outcomes': [], 'changed_geometry': [], 'changed_metrics': [], 'max_absolute_geometry_delta': {'center': 0.0, 'size': 0.0, 'rotation': 0.0}, 'semantic_exact_match': True, 'reference_cases_sha256': '6b81763eef7c5b74f522348be72bebf91da86553a16b05b03c6e771fcbc79eb1'}.

CPU geometry only; no SNAP inference, deadline, HTTP, engine or UI latency.
Saved success maskIndices may be post-filtered fitter support, not original SNAP input; saved failure maskIndices are SNAP output. The original segmentation cannot be recovered for all cases.
Missing masks remain explicit missing-mask rows; no GT-derived masks or geometry-only replacements.
GT-assisted observed-point seeds; correct instance is max exact 3D-IoU association plus containment.
Ground-contact error compares box lowest Z against GT, not independently labelled road plane.
Mini local scene holdout; repeated tracks invalidate independent-annotation confidence assumptions.
This repeatable isolated baseline does not clear the observed real-SNAP repeatability blocker.
