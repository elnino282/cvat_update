"""Frozen, single-frame PR0 service baseline. No GT geometry enters requests.

python3 -m components.vehicle_fit.evaluation.nuscenes_baseline freeze|run --help
This measures the private service, not CVAT UI/network end-to-end acceptance.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np
from scipy.spatial import ConvexHull

from .manifest import sha256, write_frozen
from .metrics import Cuboid, errors, oriented_iou, result_cuboid
from ..geometry import SEED_TOLERANCE


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def route(case):
    return 'vehicles/' + case['vehicles_type'] if case['target_label'] == 'vehicles' else case['target_label']


def select_cases(manifest, split, per_class):
    """Round robin populated strata; hash rank breaks ties, never fit outcomes."""
    groups = defaultdict(lambda: defaultdict(list))
    eligible = Counter()
    excluded = Counter()
    for case in manifest['cases']:
        if case['split'] != split:
            continue
        key = route(case)
        if not case.get('seeds'):
            excluded[key] += 1
            continue
        eligible[key] += 1
        bins = json.dumps(case.get('bins', {}), sort_keys=True)
        groups[key][bins].append(case)
    selected = []
    for key, strata in sorted(groups.items()):
        for bucket in strata.values():
            bucket.sort(key=lambda c: hashlib.sha256((str(manifest['random_seed']) + c['case_id']).encode()).hexdigest())
        count = 0
        while any(strata.values()) and count < per_class:
            for name in sorted(strata):
                if strata[name] and count < per_class:
                    selected.append(strata[name].pop(0)['case_id'])
                    count += 1
    return {'schema_version': 1, 'split': split, 'per_class_cap': per_class,
            'case_ids': selected, 'eligible_by_route': dict(eligible),
            'unseedable_by_route': dict(excluded), 'random_seed': manifest['random_seed'],
            'selection_policy': 'one median-nearest observed GT-contained seed; round robin manifest strata then SHA256 rank; no retries or fit-dependent selection'}


def request(endpoint, path, metadata=None, cloud=None, timeout=35):
    headers = {}
    body = None
    if cloud is not None:
        boundary = '----pr0-' + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n'.encode()
                + json.dumps(metadata, allow_nan=False).encode()
                + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="cloud"; filename="frame.bin"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
                + cloud + f'\r\n--{boundary}--\r\n'.encode())
        headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
    elif metadata is not None:
        body = json.dumps(metadata, allow_nan=False).encode()
        headers['Content-Type'] = 'application/json'
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(endpoint.rstrip('/') + path, data=body, headers=headers), timeout=timeout) as response:
            result, status = json.load(response), response.status
        return result, status, time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors='replace')
        try:
            result = json.loads(raw)
        except ValueError:
            result = {'detail': raw}
        return result, exc.code, time.perf_counter() - start


def gt_box(case):
    return Cuboid(np.array(case['gt_center']), np.array(case['gt_size']), np.array(case['gt_rotation_matrix']))


def valid_success(response, metadata):
    if response.get('requestId') != metadata['requestId'] or response.get('frame') != metadata['frame']:
        return False
    quality = response.get('quality', 'confident')
    provider_ok = response.get('provider') == 'snap' or (
        response.get('provider') == 'geometry-only' and quality == 'estimated' and metadata.get('allowEstimated') is True)
    if not provider_ok or quality not in ('confident', 'estimated'):
        return False
    try:
        result_cuboid(response)
    except (ValueError, KeyError, TypeError):
        return False
    return True


def bev_iou(left, right):
    """Convex projected footprint overlap, including roll/pitch in nuScenes GT."""
    def polygon(box):
        points = np.unique(box.corners[:, :2], axis=0)
        return points[ConvexHull(points).vertices]
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    def area(points):
        if len(points) < 3:
            return 0.
        return abs(sum(cross(a, b) for a, b in zip(points, np.roll(points, -1, axis=0)))) / 2
    a, b = polygon(left), polygon(right)
    clipped = list(a)
    for origin, end in zip(b, np.roll(b, -1, axis=0)):
        edge = end - origin
        output = []
        if not clipped:
            break
        previous = clipped[-1]
        for current in clipped:
            prev_d, curr_d = cross(edge, previous - origin), cross(edge, current - origin)
            if (curr_d >= -1e-10) != (prev_d >= -1e-10):
                output.append(previous + (current - previous) * prev_d / (prev_d - curr_d))
            if curr_d >= -1e-10:
                output.append(current)
            previous = current
        clipped = output
    intersection = area(clipped)
    return intersection / (area(a) + area(b) - intersection)


def percentiles(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return {'count': len(values), **{f'p{p}': float(np.percentile(values, p)) if values else None for p in (50, 95, 99)}}


def rate(successes, count):
    if not count:
        return {'numerator': 0, 'denominator': 0, 'rate': None, 'wilson_95': None}
    p, z = successes / count, 1.95996398454
    center = (p + z*z/(2*count))/(1 + z*z/count)
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/(1 + z*z/count)
    return {'numerator': successes, 'denominator': count, 'rate': p, 'wilson_95': [center-half, center+half]}


def summarize(records):
    def cohort(rows):
        accepted = [r for r in rows if r.get('metrics')]
        confident = [r for r in accepted if r['response'].get('quality', 'confident') == 'confident']
        return {'cases': len(rows), 'accepted': len(accepted),
                'coverage': rate(len(accepted), len(rows)),
                'status': dict(Counter(r['outcome'] for r in rows)),
                'reasons': dict(Counter(r['response'].get('reason', r['response'].get('detail', '')) for r in rows if not r.get('metrics'))),
                'correct_instance_proxy': rate(sum(r['metrics']['correct_instance_proxy'] for r in accepted), len(accepted)),
                'seed_containment': rate(sum(r['metrics']['seed_contained'] for r in accepted), len(accepted)),
                'false_confident': rate(sum(not r['metrics']['quality_pass'] for r in confident), len(confident)),
                'metrics': {name: percentiles([r['metrics'][name] for r in accepted]) for name in
                            ('iou_3d', 'iou_bev', 'center_error_m', 'yaw_error_rad_mod_pi', 'ground_contact_error_m')},
                'dimension_relative_error_lwh': [percentiles([r['metrics']['dimension_relative_error_lwh'][i] for r in accepted]) for i in range(3)],
                'warm_http_seconds': percentiles([r['http_seconds'] for r in rows]),
                'warm_snap_seconds': percentiles([sum(r['response'].get('diagnostics', {}).get('stageMilliseconds', {}).values()) / 1000
                                                  for r in rows if r['response'].get('diagnostics', {}).get('stageMilliseconds')]),
                'cache_hits': sum(r['response'].get('diagnostics', {}).get('cacheHit') is True for r in rows),
                'providers': dict(Counter(r['response'].get('provider', 'none') for r in rows))}
    result = {'all': cohort(records), 'by_route': {}, 'by_stratum': {}}
    for key in sorted({r['route'] for r in records}):
        rows = [r for r in records if r['route'] == key]
        result['by_route'][key] = cohort(rows)
        for bin_name in sorted({k for r in rows for k in r['bins']}):
            for value in sorted({str(r['bins'].get(bin_name)) for r in rows}):
                result['by_stratum'][f'{key}/{bin_name}/{value}'] = cohort([r for r in rows if str(r['bins'].get(bin_name)) == value])
    result['by_route']['vehicles/van'] = {'status': 'insufficient-ground-truth', 'cases': 0}
    return result


def run(args):
    manifest, selection = read_json(args.manifest), read_json(args.selection)
    if selection['manifest_sha256'] != sha256(args.manifest):
        raise ValueError('Frozen selection does not match manifest hash')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'cases.jsonl').exists():
        raise ValueError('Refusing to overwrite a measured run; choose a new output directory')
    health, status, _ = request(args.endpoint, '/health')
    if status != 200 or health.get('backend') != 'snap' or not health.get('snapAvailable'):
        raise RuntimeError(f'Real SNAP unavailable: {health}')
    selected_ids = set(selection['case_ids'])
    cases = sorted((c for c in manifest['cases'] if c['case_id'] in selected_ids), key=lambda c: (c['sample_token'], c['case_id']))
    if len(cases) != len(selected_ids):
        raise ValueError('Selection references missing or duplicate cases')
    by_frame = defaultdict(list)
    for case in cases:
        by_frame[case['sample_token']].append(case)
    scope = int(uuid.uuid4().hex[:7], 16)
    identity = {'started_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha256(args.manifest),
                'selection_sha256': sha256(args.selection), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'script_sha256': sha256(__file__), 'health': health, 'dataset_version': manifest['dataset_version'],
                'random_seed': manifest['random_seed'], 'timeout_ms': args.timeout_ms, 'seed_tolerance_m': SEED_TOLERANCE,
                'source_hashes': {p.name: sha256(p) for p in Path(__file__).parents[1].glob('*.py')},
                'scope': 'Private service HTTP; prepared warm prompts; one frame only; first-frame unprepared cold prompt measured separately; no model reload',
                'limitations': ['GT-assisted observed-point seed, no GT geometry sent to service',
                                'Correct-instance is max 3D-IoU GT association plus seed containment, not independently curated identity truth',
                                'Ground-contact compares box lowest Z to GT lowest Z, not independently labelled road plane',
                                'Yaw modulo pi only; no semantic front/back evidence',
                                'No UI/engine latency. Service does not expose separate fitting/serialization timing.',
                                'Wilson intervals assume independent annotations; repeated tracks in two scenes violate independence and intervals are optimistic']}
    identity['config_sha256'] = hashlib.sha256(json.dumps({'health': health, 'timeout_ms': args.timeout_ms, 'allowEstimated': True}, sort_keys=True).encode()).hexdigest()
    (out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    records, preparation, cold = [], [], []
    first_car = next((c['case_id'] for c in cases if route(c) == 'vehicles/car'), None)
    for frame_index, (sample, frame_cases) in enumerate(by_frame.items()):
        frame = manifest['frames'][sample]
        path = Path(args.dataroot or manifest['dataset_root']) / frame['lidar_path']
        started = time.perf_counter()
        if sha256(path) != frame['sha256']:
            raise ValueError(f'Cloud hash mismatch: {path}')
        cloud = np.fromfile(path, dtype='<f4').reshape(-1, 5)[:, :4].copy()
        cloud_bytes = cloud.tobytes()
        load_seconds = time.perf_counter() - started
        base = {'userId': scope, 'taskId': scope, 'jobId': scope, 'frame': frame_index,
                'dataVersion': selection['manifest_sha256'], 'cloud_format': 'bin'}
        def metadata(case, timeout):
            seed = case['seeds'][0]
            return {**base, 'requestId': str(uuid.uuid4()), 'labelName': case['target_label'],
                    'vehicleType': case.get('vehicles_type'), 'allowEstimated': True, 'timeout_ms': timeout,
                    'seed': dict(zip(('x', 'y', 'z'), seed['point']), pointIndex=seed['pointIndex'])}
        if any(c['case_id'] == first_car for c in frame_cases):
            case = next(c for c in frame_cases if c['case_id'] == first_car)
            response, code, seconds = request(args.endpoint, '/v1/fit', metadata(case, 16000), cloud_bytes)
            cold.append({'case_id': first_car, 'http_status': code, 'http_seconds': seconds, 'response': response})
        # Unsupported routes need no GPU preparation. One supported Car per frame warrants warm-up.
        token = None
        if any(route(c) == 'vehicles/car' for c in frame_cases):
            started = time.perf_counter()
            prepared, code, upload_seconds = request(args.endpoint, '/v1/prepare', base, cloud_bytes)
            token = prepared.get('frameToken')
            if not token:
                raise RuntimeError(f'Preparation failed: {prepared}')
            while prepared.get('state') == 'preparing' and time.perf_counter() - started < 90:
                time.sleep(.1)
                prepared, code, _ = request(args.endpoint, '/v1/prepare?' + urllib.parse.urlencode({**base, 'frameToken': token}))
            if prepared.get('state') != 'ready':
                raise RuntimeError(f'Frame not ready: {prepared}')
            preparation.append({'sample_token': sample, 'data_load_seconds': load_seconds,
                                'upload_http_seconds': upload_seconds, 'prepare_ready_seconds': time.perf_counter() - started})
        for case in frame_cases:
            meta = metadata(case, args.timeout_ms)
            if token:
                meta['frameToken'] = token
            response, code, seconds = request(args.endpoint, '/v1/fit', meta, None if token else cloud_bytes)
            row = {'case_id': case['case_id'], 'sample_token': sample, 'route': route(case), 'bins': case.get('bins', {}),
                   'request': meta, 'http_status': code, 'http_seconds': seconds, 'response': response,
                   'outcome': response.get('quality', 'confident') if response.get('status') == 'ok' else response.get('reason', 'http-error')}
            if response.get('status') == 'ok' and not valid_success(response, meta):
                row['outcome'] = 'invalid-response'
            elif response.get('status') == 'ok':
                predicted, ground_truth = result_cuboid(response), gt_box(case)
                metrics = errors(predicted, ground_truth)
                metrics['iou_bev'] = bev_iou(predicted, ground_truth)
                metrics['seed_contained'] = bool(predicted.contains(np.array([case['seeds'][0]['point']]), tolerance=SEED_TOLERANCE)[0])
                candidates = frame['annotations']  # Include unsupported neighboring categories too.
                overlaps = [(oriented_iou(predicted, gt_box(n)), n['annotation_token']) for n in candidates]
                best = max(overlaps)
                metrics['matched_annotation_token'] = best[1] if best[0] > 0 else None
                metrics['correct_instance_proxy'] = best[0] > 0 and best[1] == case['case_id'] and metrics['seed_contained']
                metrics['ground_contact_error_m'] = abs(float(predicted.corners[:, 2].min() - ground_truth.corners[:, 2].min()))
                threshold = .7 if case['target_label'] == 'vehicles' else .5
                metrics['quality_pass'] = metrics['correct_instance_proxy'] and metrics['iou_3d'] >= threshold
                row['metrics'] = metrics
            records.append(row)
            with (out / 'cases.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f'frames {frame_index+1}/{len(by_frame)} cases {len(records)}/{len(cases)}', flush=True)
    probes = []
    # Contract probes share an observed point, but carry no class-specific GT claim.
    if cases:
        for label, vehicle_type in [('vehicles', 'truck'), ('vehicles', 'bus'), ('vehicles', 'van'),
                                    ('vehicles', 'other'), ('pedestrian', None), ('two-wheels', None),
                                    ('Animal', None), ('Obstacle', None)]:
            meta = metadata(frame_cases[0], args.timeout_ms)
            meta.update(labelName=label, vehicleType=vehicle_type)
            response, code, seconds = request(args.endpoint, '/v1/fit', meta, cloud_bytes)
            probes.append({'label': label, 'vehicle_type': vehicle_type, 'http_status': code,
                           'http_seconds': seconds, 'response': response,
                           'scope': 'routing contract only; not class accuracy or Van GT'})
    summary = summarize(records)
    summary['cold_unprepared_http_seconds'] = percentiles([r['http_seconds'] for r in cold])
    summary['preparation_seconds'] = percentiles([r['prepare_ready_seconds'] for r in preparation])
    summary['data_load_seconds'] = percentiles([r['data_load_seconds'] for r in preparation])
    for name, obj in [('summary', summary), ('preparation', preparation), ('cold', cold), ('route-probes', probes)]:
        (out / f'{name}.json').write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    failures = [r['case_id'] for r in records if not r.get('metrics', {}).get('quality_pass')]
    (out / 'failure-tokens.json').write_text(json.dumps(failures, indent=2) + '\n')
    lines = ['# PR0 nuScenes service baseline', '', 'Measured private SNAP/fitter endpoint. Not end-to-end UI acceptance.', '',
             '| Route | Cases | Accepted | median 3D IoU (accepted) | warm HTTP p95 s |', '|---|---:|---:|---:|---:|']
    for key, value in summary['by_route'].items():
        if value.get('cases'):
            lines.append(f"| {key} | {value['cases']} | {value['accepted']} | {value['metrics']['iou_3d']['p50']} | {value['warm_http_seconds']['p95']:.4f} |")
        else:
            lines.append(f'| {key} | 0 | — | insufficient-ground-truth | — |')
    lines += ['', 'See identity.json for exact source/checkpoint/config hashes and limitations; cases.jsonl preserves every response, seed and request.',
              'summary.json includes strata, coverage, confidence intervals, containment, identity proxy, false-confident rate, errors and cold/warm timings.',
              'No threshold tuning or production changes were made. Failure token list is failure-tokens.json.', '']
    (out / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--manifest', required=True)
    freeze.add_argument('--output', required=True)
    freeze.add_argument('--split', choices=('dev', 'validation', 'held_out'), default='held_out')
    freeze.add_argument('--per-class', type=int, default=100)
    execute = commands.add_parser('run')
    execute.add_argument('--manifest', required=True)
    execute.add_argument('--selection', required=True)
    execute.add_argument('--output', required=True)
    execute.add_argument('--endpoint', default='http://127.0.0.1:8093')
    execute.add_argument('--dataroot')
    execute.add_argument('--timeout-ms', type=int, default=1700)
    args = parser.parse_args()
    if args.command == 'freeze':
        selection = select_cases(read_json(args.manifest), args.split, args.per_class)
        selection['manifest_sha256'] = sha256(args.manifest)
        write_frozen(selection, args.output)
    else:
        run(args)


if __name__ == '__main__':
    main()
