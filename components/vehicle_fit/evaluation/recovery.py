"""Versioned recovery replay. GT is used only after geometry returns.

Controls never patch the baseline generator, score or placement. Policy and
protected cases are immutable and checked before any recovery replay.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .metrics import calibration, errors, label_cuboid, oriented_iou, result_cuboid
from .replay import digest, stats, summarize, verified_frame


def load_geometry(path):
    name = 'recovery_geometry_' + digest(Path(path).read_bytes())[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def annotated_neighbors(manifest, frame):
    assets = manifest['frames'][frame]['assets']
    transform, _ = calibration(assets['calib'])
    neighbors = []
    for index, line in enumerate(Path(assets['label_2']).read_text().splitlines()):
        parts = line.split()
        if len(parts) < 15 or parts[0] == 'DontCare':
            continue
        dimensions = list(map(float, parts[8:11]))
        if min(dimensions) <= 0:
            continue
        box = label_cuboid({'hwl': dimensions, 'location': list(map(float, parts[11:14])),
                            'ry': float(parts[14])}, transform)
        neighbors.append({'id': index, 'class': parts[0], 'box': box})
    return neighbors


def associate(result, seed, case, neighbors, epsilon):
    box = result_cuboid(result)
    target = next(n for n in neighbors if n['id'] == case['label']['line_index'])
    if not target['box'].contains(seed[None])[0]:
        raise ValueError('Frozen seed does not belong to target GT')
    matches = [{'id': n['id'], 'class': n['class'], 'iou': oriented_iou(box, n['box'])} for n in neighbors]
    target_iou = next(n['iou'] for n in matches if n['id'] == target['id'])
    others = [n for n in matches if n['id'] != target['id']]
    best = max(others, key=lambda n: n['iou'], default=None)
    margin = target_iou - (best['iou'] if best else 0)
    correct = target_iou > 0 and margin > epsilon
    return {'metrics': errors(box, target['box']), 'correct_vehicle': bool(correct),
            'seed_inside': bool(box.contains(seed[None], tolerance=.05)[0]),
            'association': {'target_id': target['id'], 'target_iou': target_iou,
                            'best_neighbor': best, 'margin': margin,
                            'classification': 'target' if correct else 'tie' if abs(margin) <= epsilon else 'neighbor',
                            'neighbors': matches}}


def compact(result):
    result = deepcopy(result)
    for key in ('mask_indices', 'maskIndices'):
        result.get('diagnostics', {}).pop(key, None)
    return result


def observed_calls(function):
    """External observer only: frozen generators are not rewritten for tracing."""
    captured = {}
    def observer(frame, event, _arg):
        if event == 'return' and frame.f_code.co_name in ('_fit_vehicle', '_estimate_candidate', 'estimate_vehicle'):
            values = frame.f_locals
            if 'hypotheses' in values:
                captured['hypotheses'] = list(values['hypotheses'])
            if 'candidates' in values:
                captured['candidates'] = list(values['candidates'])
    previous = sys.getprofile()
    sys.setprofile(observer)
    try:
        return function(), captured
    finally:
        sys.setprofile(previous)


def verify_policy(path):
    data = Path(path).read_bytes()
    expected = Path(path).with_suffix('.sha256').read_text().strip()
    if digest(data) != expected:
        raise ValueError('Pre-registered policy changed')
    return json.loads(data), expected


def verify_protected(path, policy_hash, input_hash):
    data = Path(path).read_bytes()
    if digest(data) != Path(path).with_suffix('.sha256').read_text().strip():
        raise ValueError('Protected case manifest changed')
    protected = json.loads(data)
    if protected['policy_sha256'] != policy_hash or protected['control_run']['inputs_sha256'] != input_hash:
        raise ValueError('Protected cases do not match policy/input')
    return protected


def assess_stage(rows, baseline_p0, protected, policy):
    by_id = {r['case_id']: r for r in rows}
    baseline = {r['case_id']: r for r in baseline_p0 if r['status']=='ok' and r['correct_vehicle']}
    epsilon = policy['numeric_epsilon']
    protected_rows = []
    for case_id, reference in protected['cases'].items():
        row = by_id.get(case_id)
        returned = bool(row and row['status']=='ok')
        iou = row['metrics']['iou_3d'] if returned else None
        floor = max(policy['minimum_iou'],reference['reference_iou']-policy['maximum_protected_iou_drop'])
        passed = bool(returned and row['correct_vehicle'] and iou+epsilon >= floor)
        protected_rows.append({'case_id':case_id,'returned':returned,'correct_instance':row.get('correct_vehicle') if row else False,
                               'reference_iou':reference['reference_iou'],'minimum_allowed_iou':floor,'iou':iou,
                               'iou_drop':reference['reference_iou']-iou if returned else None,'pass':passed,
                               'reason':row.get('reason') if row and not returned else None})
    common = sorted(case_id for case_id in baseline if case_id in by_id and by_id[case_id]['status']=='ok' and by_id[case_id]['correct_vehicle'])
    lost = sorted(set(baseline)-set(common))
    paired = {name:stats([group[c]['metrics']['dimension_error_m_lwh'][0] for c in common])
              for name,group in [('baseline_p0',baseline),('stage',by_id)]}
    paired_pass = bool(common and not lost and paired['stage']['mean'] <= paired['baseline_p0']['mean']+epsilon
                       and paired['stage']['median'] <= paired['baseline_p0']['median']+epsilon)
    accepted = [r for r in rows if r['status']=='ok']
    correct = [r for r in accepted if r['correct_vehicle']]
    good = [r for r in correct if r['metrics']['iou_3d'] >= policy['minimum_iou']]
    changes = []
    for case_id, old in baseline.items():
        new = by_id.get(case_id)
        if not new or new['status']!='ok':
            changes.append({'case_id':case_id,'kind':'lost','reason':new.get('reason') if new else 'missing'})
            continue
        delta_iou = new['metrics']['iou_3d']-old['metrics']['iou_3d']
        delta_length_error = new['metrics']['dimension_error_m_lwh'][0]-old['metrics']['dimension_error_m_lwh'][0]
        delta_length = new['response']['size'][0]-old['response']['size'][0]
        if delta_iou < -epsilon or delta_length_error > epsilon or delta_length < -epsilon:
            changes.append({'case_id':case_id,'kind':'geometry-change','iou_delta':delta_iou,
                            'length_error_delta_m':delta_length_error,'length_delta_m':delta_length})
    gates = {'count':len(correct) >= policy['minimum_correct_boxes'],
             'good_count':len(good) >= policy['minimum_good_boxes'],
             'protected_cases':all(r['pass'] for r in protected_rows),'paired_length':paired_pass,
             'wrong_instance':len(correct)==len(accepted),'seed_containment':all(r['seed_inside'] for r in accepted),
             'deadline':not any(r.get('reason')=='timeout' for r in rows),
             'complete_candidate_checks':all(r.get('counts',{}).get('completed',False) for r in accepted)}
    counters = {key:stats([r.get('counts',{}).get(key,0) for r in rows])
                for key in ('generated','evaluated','safe','ambiguityCompared')}
    return {'pass':all(gates.values()),'gates':gates,'protected':protected_rows,'paired_length_error_m':paired,
            'paired_case_ids':common,'lost_reference_case_ids':lost,'changes_vs_baseline_p0':changes,
            'candidate_counts':counters}


def invoke(module, cloud, seed, mask, policy, cancelled, native_trace=None):
    kwargs = dict(priors=module.VehiclePriors(*policy['priors_lwh']), cancelled=cancelled,
                  allow_estimated=policy['allowEstimated'], max_surface_score=policy['surface_threshold'])
    if 'adaptive_refinement' in module.fit_vehicle.__code__.co_varnames:
        kwargs['adaptive_refinement'] = policy['adaptive_refinement']
    if native_trace is not None and 'trace' in module.fit_vehicle.__code__.co_varnames:
        kwargs['trace'] = native_trace
    return module.fit_vehicle(cloud, seed, mask=mask, **kwargs)


def run(args):
    policy, policy_hash = verify_policy(args.policy)
    manifest = json.loads(Path(args.manifest).read_bytes())
    cases = [c for c in manifest['cases'] if c['split'] == args.split]
    inputs = {r['case_id']: r for r in map(json.loads, Path(args.inputs).read_text().splitlines())}
    protected = verify_protected(args.protected,policy_hash,digest(Path(args.inputs).read_bytes())) if args.protected else None
    baseline, reference = load_geometry(args.baseline), load_geometry(args.reference)
    stages = [('baseline', baseline), ('baseline_p0', baseline), ('reference9277', reference)]
    for item in args.stage:
        name, path = item.split('=', 1)
        if name in {n for n, _ in stages}:
            raise ValueError('Duplicate stage name')
        stages.append((name, load_geometry(path)))
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite a recovery run')
    output.mkdir(parents=True)
    identity = {'policy_sha256': policy_hash, 'manifest_sha256': digest(Path(args.manifest).read_bytes()),
                'inputs_sha256': digest(Path(args.inputs).read_bytes()), 'evaluator_sha256': digest(Path(__file__).read_bytes()),
                'stage_sources': {name: digest(Path(module.__file__).read_bytes()) for name, module in stages},
                'split_role': 'tune' if args.split == 'tune' else 'regression', 'completed': False,
                'deadline_ms': policy['deadline_ms'], 'trace_enabled': args.trace,
                'association_protocol': 'all valid annotated 3D instances; unique target maximum IoU with epsilon',
                'latency_scope': 'CPU geometry, excludes cloud I/O and GT evaluation', 'expected_cases': len(cases)}
    if protected:
        identity['protected_sha256'] = digest(Path(args.protected).read_bytes())
    (output/'run.json').write_text(json.dumps(identity, indent=2)+'\n')
    results = {name: [] for name, _ in stages}
    last_frame = None
    with (output/'cases.jsonl').open('w') as log:
        for case in cases:
            frame = case['frame']
            if frame != last_frame:
                cloud = verified_frame(manifest, frame)
                neighbors = annotated_neighbors(manifest, frame)
                last_frame = frame
            original = inputs[case['case_id']]
            diagnostics = original.get('response', {}).get('diagnostics', {})
            seed = np.array(case['seeds'][0]['point']) if case['seeds'] else None
            mask = None
            if original.get('provider') != 'geometry-only' and seed is not None:
                if 'maskIndices' not in diagnostics:
                    raise ValueError('Missing frozen mask: '+case['case_id'])
                indices = np.array(diagnostics['maskIndices'], dtype='<i8')
                if np.any(indices < 0) or np.any(indices >= len(cloud)):
                    raise ValueError('Invalid mask indices')
                mask = np.zeros(len(cloud), bool)
                mask[indices] = True
            for name, module in stages:
                row = {'case_id': case['case_id'], 'frame': frame, 'variant': name, 'adequate': case['adequate'],
                       'input_provider': original.get('provider'), 'mask_sha256': digest(mask.tobytes()) if mask is not None else None,
                       'seed': case['seeds'][0] if seed is not None else None}
                if seed is None:
                    row['status'] = 'unseedable'
                else:
                    if name == 'baseline_p0':
                        row.update(validator_input='not-run', validator_final='not-run', quality_after_validation=None)
                    if not np.allclose(cloud[case['seeds'][0]['pointIndex'], :3], seed, atol=1e-6):
                        raise ValueError('Seed mismatch')
                    started = time.perf_counter()
                    deadline = time.monotonic()+policy['deadline_ms']/1000
                    cancelled = lambda: time.monotonic() >= deadline
                    native_trace = [] if args.trace else None
                    try:
                        captured = {}
                        if args.trace:
                            result, captured = observed_calls(lambda: invoke(module, cloud, seed, mask, policy, cancelled, native_trace))
                        else:
                            result = invoke(module, cloud, seed, mask, policy, cancelled)
                        if name == 'baseline_p0':
                            row['original_candidate'] = compact(result)
                            evidence = reference._prepare_evidence(cloud, seed, mask, cancelled)
                            row['validator_input'] = 'pass'
                            geometry_before = deepcopy({k: result[k] for k in ('center', 'size', 'rotation')})
                            result = reference._finish(deepcopy(result), evidence, reference.VehiclePriors(*policy['priors_lwh']), False, cancelled)
                            assert geometry_before == {k: result[k] for k in geometry_before}
                            row['validator_final'] = 'pass'
                            row['geometry_unchanged'] = True
                            row['quality_after_validation'] = result['quality']
                        elapsed = (time.perf_counter()-started)*1000
                        if cancelled():
                            raise module.NoFit('timeout', 'Replay deadline expired')
                        row.update(status='ok', quality=result['quality'], response=compact(result), geometry_ms=elapsed,
                                   counts=result.get('diagnostics', {}).get('candidateCounts', {}))
                        if args.trace:
                            row['candidate_trace'] = native_trace
                            row['legacy_generated_candidates'] = len(captured.get('hypotheses', captured.get('candidates', [])))
                        # GT never enters the invoked function or candidate tracing.
                        row.update(associate(result, seed, case, neighbors, policy['numeric_epsilon']))
                    except (module.NoFit, reference.NoFit) as exc:
                        if name == 'baseline_p0' and 'original_candidate' in row:
                            failed_validator = 'validator_final' if row['validator_input'] == 'pass' else 'validator_input'
                            if row[failed_validator] != 'pass':
                                row[failed_validator] = 'reject'
                                row['validator_rejection'] = exc.diagnostics.get('safetyGate', exc.reason)
                        row.update(status='no-fit', reason='timeout' if cancelled() else exc.reason,
                                   detail=exc.detail, diagnostics=exc.diagnostics, geometry_ms=(time.perf_counter()-started)*1000,
                                   counts=exc.diagnostics.get('candidateCounts', {}))
                        if args.trace:
                            row['candidate_trace'] = native_trace
                results[name].append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
            (output/'progress.json').write_text(json.dumps({'completed_cases': len(results['baseline']), 'expected_cases': len(cases)})+'\n')
            print(case['case_id'], ' '.join(n+':'+results[n][-1]['status'] for n, _ in stages), flush=True)
    summary = {name: summarize(rows) for name, rows in results.items()}
    common = set.intersection(*({r['case_id'] for r in rows if r['status']=='ok'} for rows in results.values()))
    summary['common_accepted'] = {name: summarize([r for r in rows if r['case_id'] in common]) for name, rows in results.items()}
    summary['rejection_causes'] = {name: {reason: sum(r['status']=='no-fit' and r.get('diagnostics', {}).get('safetyGate', r.get('reason')) == reason for r in rows)
                                        for reason in sorted({r.get('diagnostics', {}).get('safetyGate', r.get('reason')) for r in rows if r['status']=='no-fit'})}
                                   for name, rows in results.items()}
    if protected:
        summary['acceptance'] = {name:assess_stage(rows,results['baseline_p0'],protected,policy)
                                 for name,rows in results.items() if name not in ('baseline','baseline_p0','reference9277')}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if args.freeze_protected:
        if args.stage or args.split != 'tune':
            raise ValueError('Protection must be frozen from controls on tune only')
        if [(summary[n]['accepted'], summary[n]['iou_ge_070']) for n, _ in stages] != [(32,16),(28,15),(22,16)]:
            raise ValueError('Controls differ; explain discrepancy before freezing protected cases')
        good = {name: {r['case_id']: r for r in results[name] if r['status']=='ok' and r['correct_vehicle'] and r['metrics']['iou_3d'] >= policy['minimum_iou']}
                for name in ('baseline_p0','reference9277')}
        ids = sorted(set(good['baseline_p0']) | set(good['reference9277']))
        protected = {'policy_sha256': policy_hash, 'control_run': identity, 'cases': {
            case_id: {'reference_iou': max(v[case_id]['metrics']['iou_3d'] for v in good.values() if case_id in v),
                      'references': {name: compact(rows[case_id]) for name, rows in good.items() if case_id in rows}}
            for case_id in ids}}
        path = Path(args.freeze_protected)
        if path.exists():
            raise ValueError('Protected cases already frozen')
        path.write_text(json.dumps(protected, indent=2)+'\n')
        path.with_suffix('.sha256').write_text(digest(path.read_bytes())+'\n')
    identity['completed'] = True
    (output/'run.json').write_text(json.dumps(identity, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--split', default='tune', choices=['tune','eval'])
    parser.add_argument('--stage', action='append', default=[], help='name=geometry source snapshot')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--freeze-protected')
    parser.add_argument('--protected',help='Previously frozen protected case manifest')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
