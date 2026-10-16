"""Compare repeated Car runs after encoder-cache rebuild, ignoring timing/UUIDs."""
import argparse
import json
from pathlib import Path

from .manifest import sha256, write_frozen


def compare(left_dir, right_dir):
    directories = [Path(left_dir), Path(right_dir)]
    identities = [json.loads((p / 'identity.json').read_text()) for p in directories]
    cohorts = []
    for directory in directories:
        rows = [json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()]
        cars = [r for r in rows if r['route'] == 'vehicles/car']
        cohort = {r['case_id']: r for r in cars}
        if len(cohort) != len(cars):
            raise ValueError('Duplicate Car case IDs')
        cohorts.append(cohort)
    if not cohorts[0] or cohorts[0].keys() != cohorts[1].keys():
        raise ValueError('Repeated cohorts must contain the same nonempty Car case IDs')
    for key in ('manifest_sha256', 'source_hashes', 'config_sha256', 'timeout_ms', 'seed_tolerance_m'):
        if identities[0][key] != identities[1][key]:
            raise ValueError(f'Repeatability requires identical {key}')
    fields = ('status', 'quality', 'reason', 'provider', 'center', 'size', 'rotation')
    differences, changed_masks, changed_outcomes = [], [], []
    for key, left in cohorts[0].items():
        right = cohorts[1][key]
        if left['request']['seed'] != right['request']['seed'] or left['sample_token'] != right['sample_token']:
            raise ValueError(f'Case input changed: {key}')
        a, b = left['response'], right['response']
        changed = [field for field in fields if a.get(field) != b.get(field)]
        if changed:
            differences.append({'case_id': key, 'fields': changed,
                                'first': {field: a.get(field) for field in changed},
                                'repeat': {field: b.get(field) for field in changed}})
        if any(field in changed for field in ('status', 'quality', 'reason')):
            changed_outcomes.append(key)
        a_mask = a.get('diagnostics', {}).get('maskIndices')
        b_mask = b.get('diagnostics', {}).get('maskIndices')
        if a_mask is None or b_mask is None:
            raise ValueError(f'Missing observed mask: {key}')
        if a_mask != b_mask:
            changed_masks.append({'case_id': key, 'first_count': len(a_mask), 'repeat_count': len(b_mask),
                                  'symmetric_difference': len(set(a_mask) ^ set(b_mask))})
    return {'schema_version': 1, 'car_cases': len(cohorts[0]), 'exact_match': not differences and not changed_masks,
            'compared_fields': list(fields), 'changed_outcomes': changed_outcomes,
            'response_differences': differences, 'changed_masks': changed_masks,
            'input_sha256': [{name: sha256(p / name) for name in ('identity.json', 'cases.jsonl')} for p in directories],
            'scope': 'Same manifest/config/source/primary seeds; compare response semantics and saved mask indices after full cache rebuild. Timing/UUIDs excluded; saved successful masks may be fitter-filtered.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True)
    parser.add_argument('--repeat', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.first, args.repeat)
    write_frozen(result, args.output)
    print(json.dumps({key: result[key] for key in ('car_cases', 'exact_match')}))
    raise SystemExit(0 if result['exact_match'] else 1)


if __name__ == '__main__':
    main()
