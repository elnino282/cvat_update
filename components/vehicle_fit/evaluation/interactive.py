"""Non-browser, first-click replay of prepared Auto Fit on frozen Object/RAW clouds."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

import numpy as np

from .metrics import Cuboid, calibration, errors, euler_xyz_matrix, label_cuboid, oriented_iou, result_cuboid


def http(endpoint, method, path, metadata, cloud=None):
    if cloud is not None:
        boundary = 'autofit-' + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n' +
                json.dumps(metadata) + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="cloud"; filename="frame.bin"\r\n\r\n').encode()
        body += cloud + f'\r\n--{boundary}--\r\n'.encode()
        headers = {'Content-Type': 'multipart/form-data; boundary=' + boundary}
    elif method == 'GET':
        path += '?' + urllib.parse.urlencode(metadata)
        body, headers = None, {}
    else:
        body, headers = json.dumps(metadata).encode(), {'Content-Type': 'application/json'}
    started = time.perf_counter()
    request = urllib.request.Request(endpoint + path, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=6) as response:
        return json.load(response), (time.perf_counter() - started) * 1000


def quantiles(values):
    return {'n': len(values), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': max(values)} if values else {'n': 0}


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    raw = 'tracklet_sha256' in manifest
    cases = [c for c in manifest['cases'] if raw or c['split'] == args.split]
    by_frame = {}
    for case in cases:
        by_frame.setdefault(case['frame'], []).append(case)
    records, prepare_times, cold = [], [], None
    for number, (frame, cohort) in enumerate(by_frame.items()):
        assets = manifest['frames'][frame]
        if raw:
            with zipfile.ZipFile(assets['archive']) as archive:
                cloud = archive.read(assets['assets']['velodyne_points'])
            expected_hash = assets['sha256']['velodyne_points']
            truths = [Cuboid(np.array(c['gt_center']), np.array(c['gt_size']), euler_xyz_matrix(c['gt_rotation'])) for c in cohort]
        else:
            cloud = Path(assets['assets']['velodyne']).read_bytes()
            expected_hash = assets['sha256']['velodyne']
            transform, _ = calibration(assets['assets']['calib'])
            truths = [label_cuboid(c['label'], transform) for c in cohort]
        if hashlib.sha256(cloud).hexdigest() != expected_hash:
            raise ValueError('Frozen point cloud changed: ' + frame)
        meta = {'userId': 700001, 'jobId': 700001 if not raw else 700002, 'taskId': 700001,
                'frame': int(frame), 'cloud_format': 'bin', 'dataVersion': expected_hash}
        prepare_started = time.perf_counter()
        prepared, upload_ms = http(args.endpoint, 'POST', '/v1/prepare', meta, cloud)
        if number == 0:
            candidate = next((c for c in cohort if c.get('adequate') and c['seeds']), next((c for c in cohort if c['seeds']), None))
            if candidate:
                seed = candidate['seeds'][0]
                request = {**meta, 'frameToken': prepared['frameToken'], 'requestId': str(uuid.uuid4()),
                           'labelName': 'Car', 'timeout_ms': 1700, 'allowEstimated': True,
                           'seed': dict(zip(('x', 'y', 'z'), seed['point']), pointIndex=seed['pointIndex'])}
                response, latency = http(args.endpoint, 'POST', '/v1/fit', request)
                cold = {'preparation_state_at_click': prepared['state'], 'http_ms': latency,
                        'case_id': candidate['case_id'], 'status': response['status'],
                        'quality': response.get('quality'), 'provider': response.get('provider')}
        while prepared['state'] == 'preparing':
            if time.perf_counter() - prepare_started > 30:
                raise TimeoutError('Preparation exceeded 30 seconds for frame ' + frame)
            time.sleep(.1)
            prepared, _ = http(args.endpoint, 'GET', '/v1/prepare', {**meta, 'frameToken': prepared['frameToken']})
        if prepared['state'] != 'ready':
            raise RuntimeError('SNAP unavailable: ' + str(prepared))
        prep_ms = (time.perf_counter() - prepare_started) * 1000
        prepare_times.append(prep_ms)
        for index, case in enumerate(cohort):
            adequate = case.get('adequate', case['gt_point_count'] >= 100 and int(case.get('occlusion', 0)) <= 1)
            row = {'case_id': case['case_id'], 'frame': frame, 'adequate': adequate, 'status': 'unseedable'}
            if case['seeds']:
                seed = case['seeds'][0]
                request = {**meta, 'frameToken': prepared['frameToken'], 'requestId': str(uuid.uuid4()),
                           'labelName': 'Car', 'timeout_ms': 1700, 'allowEstimated': True,
                           'seed': dict(zip(('x', 'y', 'z'), seed['point']), pointIndex=seed['pointIndex'])}
                response, latency = http(args.endpoint, 'POST', '/v1/fit', request)
                row.update(status=response['status'], quality=response.get('quality'), http_ms=latency,
                           provider=response.get('provider'), response=response)
                if response['status'] == 'ok':
                    box = result_cuboid(response)
                    row['metrics'] = errors(box, truths[index])
                    scores = [oriented_iou(box, truth) for truth in truths]
                    row['correct_vehicle'] = max(scores) > 0 and int(np.argmax(scores)) == index
            records.append(row)
        print(json.dumps({'frame': frame, 'completed_cases': len(records), 'prepare_ms': round(prep_ms, 1),
                          'boxes': sum(r['status'] == 'ok' for r in records)}), flush=True)
    summary = {'first_click': cold, 'prepared_http': quantiles([r['http_ms'] for r in records if 'http_ms' in r]),
               'prepare': quantiles(prepare_times), 'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
               'split': args.split, 'endpoint': args.endpoint, 'retries': 0}
    for name, cohort in [('all', records), ('adequate', [r for r in records if r['adequate']])]:
        accepted = [r for r in cohort if r['status'] == 'ok']
        confident = [r for r in accepted if r['quality'] == 'confident']
        summary[name] = {'cases': len(cohort), 'boxes': len(accepted), 'confident': len(confident),
                         'estimated': len(accepted) - len(confident), 'no_fit': len(cohort) - len(accepted),
                         'wrong_vehicle': sum(not r['correct_vehicle'] for r in accepted),
                         'iou_ge_070': sum(r['correct_vehicle'] and r['metrics']['iou_3d'] >= .7 for r in accepted),
                         'confident_iou_ge_070': sum(r['correct_vehicle'] and r['metrics']['iou_3d'] >= .7 for r in confident)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['tune', 'eval'], default='eval')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8094')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
