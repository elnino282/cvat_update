"""Replay frozen *observed* service masks on the original nuScenes Car cohort.

Success masks may already be fitter-filtered support; failure masks are the
saved SNAP output. This CPU-only experiment is not an end-to-end SNAP replay.
Ground truth is used only after fitting, never to reconstruct a missing mask.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .. import geometry
from ..cloud import decode_cloud
from .manifest import sha256
from .metrics import errors, oriented_iou, result_cuboid
from .nuscenes_baseline import bev_iou, gt_box, percentiles, rate, read_json, route
from .nuscenes_dataset import load_cloud, source_path


LIMITATIONS = [
    'CPU geometry only; no SNAP inference, deadline, HTTP, engine or UI latency.',
    'Saved success maskIndices may be post-filtered fitter support, not original SNAP input; '
    'saved failure maskIndices are SNAP output. The original segmentation cannot be recovered for all cases.',
    'Missing masks remain explicit missing-mask rows; no GT-derived masks or geometry-only replacements.',
    'GT-assisted observed-point seeds; correct instance is max exact 3D-IoU association plus containment.',
    'Ground-contact error compares box lowest Z against GT, not independently labelled road plane.',
    'Mini local scene holdout; repeated tracks invalidate independent-annotation confidence assumptions.',
    'This repeatable isolated baseline does not clear the observed real-SNAP repeatability blocker.',
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verified_config(identity, source_dir=None):
    source_dir = Path(source_dir or Path(geometry.__file__).parent)
    # SNAP and HTTP service code are not executed in this experiment.
    for name in ('geometry.py', 'cloud.py'):
        expected = identity['source_hashes'].get(name)
        if not expected or sha256(source_dir / name) != expected:
            raise ValueError(f'Baseline production source hash mismatch: {name}')
    health = identity['health']
    frozen_config = {'health': health, 'timeout_ms': identity['timeout_ms'], 'allowEstimated': True}
    if digest(json.dumps(frozen_config, sort_keys=True).encode()) != identity['config_sha256']:
        raise ValueError('Baseline config hash mismatch')
    priors = health['vehiclePriorsMetres']
    values = [priors[name] for name in ('length', 'width', 'height')] + [health['maxSurfaceScore']]
    if not np.isfinite(values).all() or min(values) <= 0 or not isinstance(health['adaptiveSizeRefinement'], bool):
        raise ValueError('Invalid baseline fitter config')
    if identity['seed_tolerance_m'] != geometry.SEED_TOLERANCE:
        raise ValueError('Baseline seed tolerance changed')
    return {'priors': geometry.VehiclePriors(**priors), 'max_surface_score': health['maxSurfaceScore'],
            'adaptive_refinement': health['adaptiveSizeRefinement'], 'allow_estimated': True}


def verify_coordinates(manifest, frame):
    if manifest['coordinate_frame'] != 'LIDAR_TOP' or manifest['dimension_order'] != 'length,width,height':
        raise ValueError('Unsupported coordinate frame or dimension order')
    if manifest['rotation_convention'] != 'Three.js Euler XYZ (Rx @ Ry @ Rz)':
        raise ValueError('Unsupported rotation convention')
    transform = np.asarray(frame['lidar_to_global'], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Invalid coordinate transform')
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=2e-6)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=2e-6)):
        raise ValueError('Invalid rigid coordinate transform')


def verified_frame(manifest, sample, dataroot):
    frame = manifest['frames'][sample]
    verify_coordinates(manifest, frame)
    raw = load_cloud(manifest, sample, dataroot)
    if len(raw) != frame['point_count'] or frame['point_dimensions'] != 5:
        raise ValueError('Cloud index convention mismatch')
    service_bytes = np.asarray(raw[:, :4], dtype='<f4').tobytes()
    cloud, original_indices = decode_cloud(service_bytes, 'bin')
    return cloud, original_indices, digest(service_bytes)


def verified_seed(case, item, cloud, original_indices, manifest_hash):
    request, response = item['request'], item['response']
    if item['sample_token'] != case['sample_token'] or item['route'] != route(case):
        raise ValueError('Baseline case/frame/route mismatch')
    if (request['dataVersion'] != manifest_hash or request['cloud_format'] != 'bin'
            or request['labelName'] != case['target_label'] or request['vehicleType'] != case['vehicles_type']
            or request['allowEstimated'] is not True):
        raise ValueError('Baseline request identity mismatch')
    if response.get('requestId') != request['requestId'] or response.get('frame') != request['frame']:
        raise ValueError('Baseline response request/frame identity mismatch')
    primary = case['seeds'][0]
    seed = np.asarray([request['seed'][axis] for axis in ('x', 'y', 'z')], dtype=float)
    index = request['seed']['pointIndex']
    if (type(index) is not int or index != primary['pointIndex'] or seed.shape != (3,)
            or not np.isfinite(seed).all() or not np.allclose(seed, primary['point'], rtol=0, atol=1e-6)):
        raise ValueError('Frozen primary seed mismatch')
    positions = np.flatnonzero(original_indices == index)
    if len(positions) != 1 or not np.allclose(cloud[positions[0], :3], seed, rtol=0, atol=1e-6):
        raise ValueError('Frozen seed/cloud point index mismatch')
    return seed


def observed_mask(item, original_indices, cloud_hash):
    diagnostics = item['response'].get('diagnostics', {})
    if 'maskIndices' not in diagnostics:
        return None, None
    if diagnostics.get('cloudSHA256') != cloud_hash:
        raise ValueError('Historical mask/cloud hash mismatch')
    indices = diagnostics['maskIndices']
    if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
            or len(indices) != len(set(indices)) or not set(indices).issubset(set(original_indices.tolist()))):
        raise ValueError('Invalid observed mask indices')
    # Hash the canonical ascending raw-cloud indices, independent of native endianness.
    return np.isin(original_indices, indices), digest(canonical(sorted(indices)))


def measure(result, case, frame, seed, tolerance):
    predicted, truth = result_cuboid(result), gt_box(case)
    metrics = errors(predicted, truth)
    metrics['iou_bev'] = bev_iou(predicted, truth)
    metrics['seed_contained'] = bool(predicted.contains(seed[None], tolerance=tolerance)[0])
    best = max((oriented_iou(predicted, gt_box(other)), other['annotation_token']) for other in frame['annotations'])
    metrics['matched_annotation_token'] = best[1] if best[0] > 0 else None
    metrics['correct_instance_proxy'] = best[0] > 0 and best[1] == case['case_id'] and metrics['seed_contained']
    metrics['ground_contact_error_m'] = abs(float(predicted.corners[:, 2].min() - truth.corners[:, 2].min()))
    metrics['quality_pass'] = metrics['correct_instance_proxy'] and metrics['iou_3d'] >= .7
    return metrics


def replay_case(case, item, frame, cloud, seed, mask, mask_hash, config, tolerance):
    row = {'case_id': case['case_id'], 'sample_token': case['sample_token'], 'route': route(case),
           'bins': case['bins'], 'baseline_outcome': item['outcome'], 'mask_sha256': mask_hash,
           'seed_sha256': digest(canonical(item['request']['seed'])),
           'mask_provenance': 'success-support-may-be-filtered' if item['response']['status'] == 'ok' else 'failure-observed-snap'}
    if mask is None:
        row.update(outcome='missing-mask', response={'status': 'missing-mask', 'reason': 'missing-mask'})
        return row
    row['mask_point_count'] = int(mask.sum())
    started = time.perf_counter()
    try:
        result = geometry.fit_vehicle(cloud, seed, mask=mask, **config)
        row.update(outcome=result['quality'], response={'status': 'ok', **result})
    except geometry.NoFit as exc:
        row.update(outcome=exc.reason, response={'status': 'no-fit', 'reason': exc.reason,
                                               'detail': exc.detail, 'diagnostics': exc.diagnostics})
    except Exception as exc:  # A replay crash is preserved separately from a fitter rejection.
        row.update(outcome='replay-error', response={'status': 'replay-error', 'reason': type(exc).__name__, 'detail': str(exc)})
    row['geometry_seconds'] = time.perf_counter() - started
    if row['response']['status'] == 'ok':
        row['metrics'] = measure(row['response'], case, frame, seed, tolerance)
    return row


def summarize(rows):
    accepted = [r for r in rows if r.get('metrics')]
    confident = [r for r in accepted if r['outcome'] == 'confident']
    return {'cases': len(rows), 'accepted': len(accepted), 'outcomes': dict(Counter(r['outcome'] for r in rows)),
            'coverage': rate(len(accepted), len(rows)),
            'correct_instance_proxy': rate(sum(r['metrics']['correct_instance_proxy'] for r in accepted), len(accepted)),
            'seed_containment': rate(sum(r['metrics']['seed_contained'] for r in accepted), len(accepted)),
            'false_confident': rate(sum(not r['metrics']['quality_pass'] for r in confident), len(confident)),
            'metrics_accepted': {key: percentiles([r['metrics'][key] for r in accepted]) for key in
                                 ('iou_3d', 'iou_bev', 'center_error_m', 'yaw_error_rad_mod_pi', 'ground_contact_error_m')},
            'iou_all_cases_zero_for_unreturned': percentiles([r.get('metrics', {}).get('iou_3d', 0) for r in rows]),
            'dimension_relative_error_lwh': [percentiles([r['metrics']['dimension_relative_error_lwh'][i] for r in accepted]) for i in range(3)],
            'geometry_seconds': percentiles([r['geometry_seconds'] for r in rows if 'geometry_seconds' in r])}


def semantic_rows(rows):
    """Only runtime is excluded; geometry, diagnostics, masks, metrics and failures stay."""
    return [{key: value for key, value in row.items() if key != 'geometry_seconds'} for row in rows]


def compare(rows, previous):
    previous = {r['case_id']: r for r in previous}
    if set(previous) != {r['case_id'] for r in rows}:
        raise ValueError('Comparison cohort mismatch')
    deltas = {key: [float(np.max(np.abs(np.asarray(r['response'][key]) - previous[r['case_id']]['response'][key])))
                    for r in rows if key in r['response'] and key in previous[r['case_id']]['response']]
              for key in ('center', 'size', 'rotation')}
    return {'changed_outcomes': [r['case_id'] for r in rows if r['outcome'] != previous[r['case_id']]['outcome']],
            'changed_geometry': [r['case_id'] for r in rows if any(r['response'].get(k) != previous[r['case_id']]['response'].get(k)
                                                                  for k in ('center', 'size', 'rotation', 'quality'))],
            'changed_metrics': [r['case_id'] for r in rows if r.get('metrics') != previous[r['case_id']].get('metrics')],
            'max_absolute_geometry_delta': {key: max(values) if values else None for key, values in deltas.items()}}


def run(args):
    manifest, selection, baseline_identity = map(read_json, (args.manifest, args.selection, args.identity))
    manifest_hash = sha256(args.manifest)
    if manifest_hash != baseline_identity['manifest_sha256'] or manifest_hash != selection['manifest_sha256']:
        raise ValueError('Manifest identity hash mismatch')
    if sha256(args.selection) != baseline_identity['selection_sha256']:
        raise ValueError('Baseline selection hash mismatch')
    config = verified_config(baseline_identity)
    root = args.dataroot or manifest['dataset_root']
    for relative, expected in manifest['source_hashes'].items():
        if sha256(source_path(root, relative)) != expected:
            raise ValueError(f'Dataset source hash mismatch: {relative}')
    inputs = [json.loads(line) for line in Path(args.inputs).read_text().splitlines()]
    by_id = {r['case_id']: r for r in inputs}
    if len(by_id) != len(inputs) or set(by_id) != set(selection['case_ids']):
        raise ValueError('Baseline ledger duplicate/missing/unselected cases')
    cases = [c for c in manifest['cases'] if c['case_id'] in by_id and route(c) == 'vehicles/car']
    cases.sort(key=lambda c: (c['sample_token'], c['case_id']))
    if not cases or len(cases) != sum(r['route'] == 'vehicles/car' for r in inputs):
        raise ValueError('Car cohort identity mismatch')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Refusing to overwrite replay artifacts; use an empty output directory')
    identity = {'started_utc': datetime.now(timezone.utc).isoformat(), 'completed': False,
                'manifest_sha256': manifest_hash, 'selection_sha256': sha256(args.selection),
                'baseline_inputs_sha256': sha256(args.inputs), 'baseline_identity_sha256': sha256(args.identity),
                'production_source_hashes': baseline_identity['source_hashes'], 'config_sha256': baseline_identity['config_sha256'],
                'verified_executed_production_source_hashes': {name: sha256(Path(geometry.__file__).parent / name)
                                                              for name in ('geometry.py', 'cloud.py')},
                'source_verification_scope': 'geometry.py and cloud.py execute and must match baseline; '
                                             'original SNAP/service hashes are provenance only; those modules are not executed',
                'config': {key: vars(value) if key == 'priors' else value for key, value in config.items()},
                'evaluation_source_hashes': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')},
                'case_ids': [c['case_id'] for c in cases], 'python': platform.python_version(),
                'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(),
                'limitations': LIMITATIONS}
    write_json(out / 'identity.json', identity)
    rows, cached_sample = [], None
    with (out / 'cases.jsonl').open('x') as stream:
        for case in cases:
            sample = case['sample_token']
            if sample != cached_sample:
                cloud, original_indices, cloud_hash = verified_frame(manifest, sample, root)
                cached_sample = sample
            item = by_id[case['case_id']]
            seed = verified_seed(case, item, cloud, original_indices, manifest_hash)
            mask, mask_hash = observed_mask(item, original_indices, cloud_hash)
            row = replay_case(case, item, manifest['frames'][sample], cloud, seed, mask, mask_hash,
                              config, baseline_identity['seed_tolerance_m'])
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            print(f'{len(rows)}/{len(cases)} {case["case_id"]} {row["outcome"]}', flush=True)
    summary = summarize(rows)
    summary['by_stratum'] = {f'{key}/{value}': summarize([r for r in rows if str(r['bins'].get(key)) == value])
                              for key in sorted({k for r in rows for k in r['bins']})
                              for value in sorted({str(r['bins'].get(key)) for r in rows})}
    summary['semantic_sha256'] = digest(canonical(semantic_rows(rows)))
    summary['original_service_comparison'] = compare(rows, [by_id[c['case_id']] for c in cases])
    if args.reference:
        reference = Path(args.reference)
        old_identity = read_json(reference / 'identity.json')
        for key in ('manifest_sha256', 'selection_sha256', 'baseline_inputs_sha256', 'baseline_identity_sha256',
                    'production_source_hashes', 'config_sha256', 'evaluation_source_hashes', 'case_ids'):
            if identity[key] != old_identity[key]:
                raise ValueError(f'Replay reference identity mismatch: {key}')
        previous = [json.loads(line) for line in (reference / 'cases.jsonl').read_text().splitlines()]
        summary['repeat_comparison'] = {**compare(rows, previous),
                                        'semantic_exact_match': semantic_rows(rows) == semantic_rows(previous),
                                        'reference_cases_sha256': sha256(reference / 'cases.jsonl')}
    write_json(out / 'summary.json', summary)
    write_json(out / 'failure-tokens.json', [r['case_id'] for r in rows if not r.get('metrics', {}).get('quality_pass')])
    identity.update(completed=True, semantic_sha256=summary['semantic_sha256'], cases_sha256=sha256(out / 'cases.jsonl'))
    write_json(out / 'identity.json', identity)
    (out / 'report.md').write_text('\n'.join([
        '# Frozen observed-mask CPU replay', '', f'Cases: {len(rows)}; accepted: {summary["accepted"]}.',
        f'Outcomes: {summary["outcomes"]}.', f'Accepted 3D IoU: {summary["metrics_accepted"]["iou_3d"]}.',
        f'CPU geometry seconds: {summary["geometry_seconds"]}.',
        f'Semantic SHA256 (excluding runtime only): {summary["semantic_sha256"]}.',
        f'Original service comparison: {summary["original_service_comparison"]}.',
        f'Repeat comparison: {summary.get("repeat_comparison", "not requested")}.', '', *LIMITATIONS, '']))
    if args.reference and not summary['repeat_comparison']['semantic_exact_match']:
        raise RuntimeError('Frozen replay semantic mismatch; complete evidence saved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'selection', 'inputs', 'identity', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--dataroot')
    parser.add_argument('--reference', help='Prior replay output directory for strict identity and exact semantic comparison')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
