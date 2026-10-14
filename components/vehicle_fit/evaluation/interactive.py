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

from .metrics import errors, oriented_iou, result_cuboid
from .replay import verified_frame, truth_boxes, summarize


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
    if raw:
        if not args.tracklet_zip:
            raise ValueError('RAW evaluation requires --tracklet-zip')
        with zipfile.ZipFile(args.tracklet_zip) as archive:
            name, = [n for n in archive.namelist() if n.endswith('tracklet_labels.xml')]
            if hashlib.sha256(archive.read(name)).hexdigest() != manifest['tracklet_sha256']:
                raise ValueError('Frozen tracklet labels changed')
    cases = [c for c in manifest['cases'] if raw or c['split'] == args.split]
    by_frame = {}
    for case in cases:
        by_frame.setdefault(case['frame'], []).append(case)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                'split': args.split, 'endpoint': args.endpoint, 'retries': 0,
                'completed': False, 'expected_cases': len(cases), 'completed_cases': 0}
    records, prepare_times, cold = [], [], None
    (output/'run.json').write_text(json.dumps(identity, indent=2)+'\n')
    with (output/'cases.jsonl').open('w') as log:
        for frame, cohort in by_frame.items():
            points = verified_frame(manifest, frame)  # Includes calibration/label/image hashes before IoU.
            cloud = points.astype('<f4').tobytes()
            cloud_key = 'velodyne_points' if raw else 'velodyne'
            expected_hash = manifest['frames'][frame]['sha256'][cloud_key]
            truths, targets = truth_boxes(manifest, frame, cohort)
            meta = {'userId': 700001, 'jobId': 700002 if raw else 700001, 'taskId': 700001,
                    'frame': int(frame), 'cloud_format': 'bin', 'dataVersion': expected_hash}
            prepare_started = time.perf_counter()
            frame_error = None
            try:
                prepared, _ = http(args.endpoint, 'POST', '/v1/prepare', meta, cloud)
                if cold is None:
                    candidate = next((c for c in cohort if c['seeds']), None)
                    if candidate:
                        seed = candidate['seeds'][0]
                        request = {**meta, 'frameToken': prepared['frameToken'], 'requestId': str(uuid.uuid4()),
                                   'labelName': 'Car', 'timeout_ms': 1700, 'allowEstimated': True,
                                   'seed': dict(zip(('x', 'y', 'z'), seed['point']), pointIndex=seed['pointIndex'])}
                        cold = {'case_id': candidate['case_id'], 'preparation_state_at_click': prepared['state']}
                        try:
                            response, latency = http(args.endpoint, 'POST', '/v1/fit', request)
                            cold.update(status=response['status'], quality=response.get('quality'),
                                        provider=response.get('provider'), http_ms=latency)
                        except (urllib.error.URLError, TimeoutError) as exc:
                            cold.update(status='transport-error', transport_error=str(exc))
                while prepared['state'] == 'preparing':
                    if time.perf_counter()-prepare_started > 30:
                        raise TimeoutError('Preparation exceeded 30 seconds')
                    time.sleep(.1)
                    prepared, _ = http(args.endpoint, 'GET', '/v1/prepare', {**meta, 'frameToken': prepared['frameToken']})
                if prepared['state'] != 'ready':
                    raise RuntimeError('SNAP unavailable: '+str(prepared))
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                frame_error = f'{type(exc).__name__}: {exc}'
            prepare_times.append((time.perf_counter()-prepare_started)*1000)
            for index, case in enumerate(cohort):
                row = {'case_id': case['case_id'], 'frame': frame,
                       'adequate': case.get('adequate', case['gt_point_count'] >= 100 and int(case.get('occlusion', 0)) <= 1),
                       'status': 'unseedable'}
                if case['seeds']:
                    seed = case['seeds'][0]
                    if frame_error:
                        row.update(status='transport-error', transport_error=frame_error, stage='prepare')
                    else:
                        request = {**meta, 'frameToken': prepared['frameToken'], 'requestId': str(uuid.uuid4()),
                                   'labelName': 'Car', 'timeout_ms': 1700, 'allowEstimated': True,
                                   'seed': dict(zip(('x', 'y', 'z'), seed['point']), pointIndex=seed['pointIndex'])}
                        try:
                            response, latency = http(args.endpoint, 'POST', '/v1/fit', request)
                            row.update(status=response['status'], quality=response.get('quality'), http_ms=latency,
                                       provider=response.get('provider'), response=response)
                            if response['status'] == 'ok':
                                box = result_cuboid(response)
                                row['metrics'] = errors(box, truths[targets[index]])
                                scores = [oriented_iou(box, truth) for truth in truths]
                                row['correct_vehicle'] = max(scores) > 0 and int(np.argmax(scores)) == targets[index]
                                row['seed_inside'] = bool(box.contains(np.array(seed['point'])[None], tolerance=.05)[0])
                        except (urllib.error.URLError, TimeoutError) as exc:
                            row.update(status='transport-error', transport_error=f'{type(exc).__name__}: {exc}', stage='fit')
                records.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                summary = {'all': summarize(records), 'adequate': summarize([r for r in records if r['adequate']]),
                           'first_click': cold,
                           'prepared_http': quantiles([r['http_ms'] for r in records if 'http_ms' in r]),
                           'prepare': quantiles(prepare_times), **identity}
                summary['completed_cases'] = len(records)
                (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(json.dumps({'frame': frame, 'completed_cases': len(records)}), flush=True)
    identity.update(completed=True, completed_cases=len(records))
    summary.update(identity)
    (output/'run.json').write_text(json.dumps(identity, indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['tune', 'eval'], default='eval')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8094')
    parser.add_argument('--tracklet-zip', help='Required for verified RAW tracklet labels')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
