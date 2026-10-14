"""Frozen-mask A/B geometry replay; no GT is passed to either fitter.

Historical masks are inputs, not historical performance claims. Every variant
is freshly executed, including baseline source supplied explicitly by the caller.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from .. import geometry
from .metrics import Cuboid, calibration, errors, euler_xyz_matrix, kitti_labels, label_cuboid, oriented_iou, result_cuboid


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified_frame(manifest, frame):
    assets = manifest['frames'][frame]
    payloads = {}
    if 'archive' in assets:
        with zipfile.ZipFile(assets['archive']) as archive:
            for kind, name in assets['assets'].items():
                payloads[kind] = archive.read(name)
        cloud_key = 'velodyne_points'
    else:
        payloads = {kind: Path(path).read_bytes() for kind, path in assets['assets'].items()}
        cloud_key = 'velodyne'
    for kind, data in payloads.items():
        if digest(data) != assets['sha256'][kind]:
            raise ValueError('Frozen asset changed: ' + frame + '/' + kind)
    cloud = np.frombuffer(payloads[cloud_key], dtype='<f4').reshape(-1, 4).copy()
    if not np.isfinite(cloud).all():
        raise ValueError('Replay requires the frozen finite raw-cloud index convention')
    return cloud


def truth_boxes(manifest, frame, cohort):
    if 'tracklet_sha256' in manifest:
        boxes = [Cuboid(np.array(c['gt_center']), np.array(c['gt_size']), euler_xyz_matrix(c['gt_rotation'])) for c in cohort]
        return boxes, list(range(len(cohort)))
    assets = manifest['frames'][frame]['assets']
    transform, _ = calibration(assets['calib'])
    labels = kitti_labels(assets['label_2'])
    boxes = [label_cuboid(label, transform) for label in labels]
    targets = [next(i for i, label in enumerate(labels) if label['line_index'] == c['label']['line_index']) for c in cohort]
    return boxes, targets


def stats(values):
    return {'n': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))} if values else {'n': 0}


def summarize(rows):
    accepted = [r for r in rows if r['status'] == 'ok']
    summary = {'cases': len(rows), 'accepted': len(accepted),
               'confident': sum(r.get('quality') == 'confident' for r in accepted),
               'estimated': sum(r.get('quality') == 'estimated' for r in accepted),
               'assessed_quality_counts': {quality: sum(r.get('response', {}).get('diagnostics', r.get('diagnostics', {})).get('assessedQuality', r.get('quality')) == quality for r in rows)
                                           for quality in ('confident', 'estimated')},
               'no_fit': sum(r['status'] == 'no-fit' for r in rows),
               'unseedable': sum(r['status'] == 'unseedable' for r in rows),
               'transport_error': sum(r['status'] == 'transport-error' for r in rows),
               'missing_mask': sum(r['status'] == 'missing-mask' for r in rows),
               'wrong_instance_accepted': sum(not r['correct_vehicle'] for r in accepted),
               'seed_outside_accepted': sum(not r['seed_inside'] for r in accepted),
               'correct_instance_coverage': sum(r['correct_vehicle'] for r in accepted)/len(rows) if rows else None,
               'iou_ge_070': sum(r['correct_vehicle'] and r['metrics']['iou_3d'] >= .70 for r in accepted),
               'iou_3d': stats([r['metrics']['iou_3d'] for r in accepted]),
               'iou_all_cases_zero_for_unreturned': stats([r.get('metrics', {}).get('iou_3d', 0) for r in rows]),
               'dimension_error_m_lwh': [stats([r['metrics']['dimension_error_m_lwh'][i] for r in accepted]) for i in range(3)],
               'center_error_m': stats([r['metrics']['center_error_m'] for r in accepted]),
               'yaw_error_rad_mod_pi': stats([r['metrics']['yaw_error_rad_mod_pi'] for r in accepted]),
               'geometry_ms': stats([r['geometry_ms'] for r in rows if 'geometry_ms' in r]),
               'refined': sum(r.get('response', {}).get('diagnostics', {}).get('sizeRefinement', {}).get('applied', False) for r in rows)}
    return summary


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('vehicle_size_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    if 'tracklet_sha256' in manifest:
        if not args.tracklet_zip:
            raise ValueError('RAW replay requires --tracklet-zip to verify label provenance')
        with zipfile.ZipFile(args.tracklet_zip) as archive:
            name, = [n for n in archive.namelist() if n.endswith('tracklet_labels.xml')]
            if digest(archive.read(name)) != manifest['tracklet_sha256']:
                raise ValueError('Frozen tracklet labels changed')
    selected = [c for c in manifest['cases'] if 'tracklet_sha256' in manifest or c['split'] == args.split]
    inputs = [json.loads(line) for line in Path(args.inputs).read_text().splitlines()]
    by_id = {r['case_id']: r for r in inputs}
    if len(by_id) != len(inputs):
        raise ValueError('Duplicate input cases')
    missing = {c['case_id'] for c in selected} - by_id.keys()
    if missing:
        raise ValueError('Input ledger incomplete: ' + str(sorted(missing)))
    baseline = load_baseline(args.baseline)
    variants = [('baseline', baseline, {}), ('safety', geometry, {'adaptive_refinement': False}),
                ('refined', geometry, {'adaptive_refinement': True})]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': digest(manifest_bytes), 'inputs_sha256': digest(Path(args.inputs).read_bytes()),
                'replay_sha256': digest(Path(__file__).read_bytes()),
                'baseline_geometry_sha256': digest(Path(args.baseline).read_bytes()),
                'new_geometry_sha256': digest(Path(geometry.__file__).read_bytes()), 'split': args.split,
                'protocol': 'first frozen seed, historical mask held fixed (may be post ground-filter); fresh geometry; no retries; allowEstimated=true; explicit geometry-only rows replay estimator',
                'latency': 'CPU geometry only, no model/HTTP/UI latency', 'priors_lwh': [3.7, 1.61, 1.49],
                'completed': False, 'expected_cases': len(selected)}
    (output/'run.json').write_text(json.dumps(identity, indent=2)+'\n')
    records = {name: [] for name, _, _ in variants}
    cached_frame = None
    with (output/'cases.jsonl').open('w') as log:
        for case in selected:
            frame = case['frame']
            if frame != cached_frame:
                cloud = verified_frame(manifest, frame)
                cohort = [c for c in selected if c['frame'] == frame]
                truths, targets = truth_boxes(manifest, frame, cohort)
                target_by_id = dict(zip((c['case_id'] for c in cohort), targets))
                cached_frame = frame
            item = by_id[case['case_id']]
            original = item.get('response', {})
            diagnostics = original.get('diagnostics', {})
            cloud_key = 'velodyne_points' if 'tracklet_sha256' in manifest else 'velodyne'
            if diagnostics.get('cloudSHA256') and diagnostics['cloudSHA256'] != manifest['frames'][frame]['sha256'][cloud_key]:
                raise ValueError('Historical mask/cloud identity mismatch: ' + case['case_id'])
            for name, module, kwargs in variants:
                row = {'case_id': case['case_id'], 'frame': frame, 'variant': name,
                       'adequate': case.get('adequate', case['gt_point_count'] >= 100 and int(case.get('occlusion', 0)) <= 1)}
                if not case['seeds']:
                    row['status'] = 'unseedable'
                elif item.get('status') == 'transport-error':
                    row['status'] = 'transport-error'
                elif item.get('provider') != 'geometry-only' and 'maskIndices' not in diagnostics:
                    row['status'] = 'missing-mask'
                else:
                    seed = np.array(case['seeds'][0]['point'])
                    if not np.allclose(cloud[case['seeds'][0]['pointIndex'], :3], seed, atol=1e-6):
                        raise ValueError('Frozen seed/index mismatch')
                    mask = None
                    if item.get('provider') != 'geometry-only':
                        indices = np.asarray(diagnostics['maskIndices'], dtype=int)
                        if np.any(indices < 0) or np.any(indices >= len(cloud)):
                            raise ValueError('Mask index out of range')
                        mask = np.zeros(len(cloud), dtype=bool)
                        mask[indices] = True
                        row['mask_sha256'] = digest(indices.tobytes())
                    start = time.perf_counter()
                    try:
                        priors = module.VehiclePriors(3.7, 1.61, 1.49)
                        if item.get('provider') == 'geometry-only':
                            row['replay_path'] = 'estimate'
                            result = module.estimate_vehicle(cloud, seed, priors=priors, **kwargs)
                        else:
                            row['replay_path'] = 'fit'
                            result = module.fit_vehicle(cloud, seed, mask=mask, priors=priors,
                                                        allow_estimated=True, max_surface_score=.10, **kwargs)
                        row['geometry_ms'] = (time.perf_counter()-start)*1000
                        box = result_cuboid(result)
                        target = target_by_id[case['case_id']]
                        scores = [oriented_iou(box, truth) for truth in truths]
                        row.update(status='ok', quality=result['quality'], response=result,
                                   metrics=errors(box, truths[target]), seed_inside=bool(box.contains(seed[None], tolerance=.05)[0]),
                                   correct_vehicle=max(scores) > 0 and int(np.argmax(scores)) == target)
                    except module.NoFit as exc:
                        row.update(status='no-fit', reason=exc.reason, diagnostics=exc.diagnostics,
                                   geometry_ms=(time.perf_counter()-start)*1000)
                records[name].append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
            summaries = {name: {'all': summarize(rows), 'adequate': summarize([r for r in rows if r['adequate']])} for name, rows in records.items()}
            summaries['progress'] = {'completed_cases': len(records['baseline']), 'expected_cases': len(selected)}
            (output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
            print(case['case_id'], ' '.join(f'{n}:{records[n][-1]["status"]}' for n, _, _ in variants), flush=True)
    common_ids = set.intersection(*({r['case_id'] for r in rows if r['status']=='ok'} for rows in records.values()))
    summaries['common_accepted'] = {name: summarize([r for r in rows if r['case_id'] in common_ids]) for name, rows in records.items()}
    summaries['refinement_changed'] = [r['case_id'] for r in records['refined'] if r.get('response', {}).get('diagnostics', {}).get('sizeRefinement', {}).get('applied')]
    (output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    identity['completed'] = True
    (output/'run.json').write_text(json.dumps(identity, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--split', choices=['tune', 'eval'], default='tune')
    parser.add_argument('--tracklet-zip')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
