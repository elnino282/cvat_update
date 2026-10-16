"""Exercise real GT import/migration/export/rollback only in cvat-vehicles-qa."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import time

import requests
import yaml


URL = 'http://localhost:18080'
COMPOSE = '/tmp/cvat-vehicles-qa.yml'


def source_import_audit(dataset, data, labels):
    names = {label['id']: label['name'] for label in labels}
    specs = {spec['id']: spec['name'] for label in labels for spec in label['attributes']}
    actual = {}
    for track in data['tracks']:
        immutable = {specs[a['spec_id']]: a['value'] for a in track['attributes']}
        for shape in track['shapes']:
            attributes = {**immutable, **{specs[a['spec_id']]: a['value'] for a in shape['attributes']}}
            actual[(immutable['kitti_track_id'], shape['frame'])] = (names[track['label_id']], shape, attributes)
    source = json.loads((dataset / 'annotations/annotations/default.json').read_text())
    maximum, count = 0., 0
    for item in source['items']:
        frame = int(Path(item['id']).name)
        for annotation in item['annotations']:
            expected = annotation['attributes']
            name, shape, attributes = actual[(expected['kitti_track_id'], frame)]
            assert name == source['categories']['label']['labels'][annotation['label_id']]['name']
            assert shape['outside'] == expected['outside'] and shape['occluded'] == expected['occluded']
            assert all(str(expected[key]) == value for key, value in attributes.items())
            source_points = annotation['position'] + annotation['rotation'] + annotation['scale']
            maximum = max(maximum, max(abs(a - b) for a, b in zip(shape['points'][:9], source_points)))
            count += 1
    assert count == len(actual) == 722
    assert maximum <= .00500001, 'Import geometry exceeds the observed two-decimal rounding'
    return {'keyframes': count, 'custom_attributes_exact': True, 'flags_exact': True,
            'max_geometry_coordinate_difference': maximum,
            'note': 'CVAT import rounds geometry before migration; migration/rollback preserve that snapshot exactly.'}


def normalized(data, labels, *, preserve_ids):
    label_names = {label['id']: label['name'] for label in labels}
    specs = {spec['id']: spec['name'] for label in labels for spec in label['attributes']}

    def visit(value, key=''):
        if isinstance(value, dict):
            result = {}
            vehicle = label_names.get(value.get('label_id')) == 'Vehicles'
            if vehicle:
                assert any(specs[a['spec_id']] == 'type' and a['value'] == 'car'
                           for a in value['attributes']), 'Missing immutable type=car'
            for name, item in value.items():
                if name == 'version' or (name == 'id' and not preserve_ids):
                    continue
                if name == 'label_id':
                    result[name] = 'Car' if vehicle else label_names[item]
                elif name == 'spec_id':
                    result[name] = specs[item]
                elif name == 'attributes' and vehicle:
                    result[name] = visit([a for a in item if specs[a['spec_id']] != 'type'], name)
                else:
                    result[name] = visit(item, name)
            return result
        if isinstance(value, list):
            values = [visit(item) for item in value]
            if key in {'attributes', 'tracks', 'shapes', 'tags', 'elements'}:
                values.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return values
        return value
    return visit(data)


def equivalent(left, right, path='root'):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), f'Fields changed at {path}'
        for key in left:
            equivalent(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, list):
        assert len(left) == len(right), f'Count changed at {path}'
        for index, (a, b) in enumerate(zip(left, right)):
            equivalent(a, b, f'{path}[{index}]')
    elif isinstance(left, float):
        assert math.isclose(left, right, abs_tol=1e-6, rel_tol=0), f'Geometry changed at {path}'
    else:
        assert left == right, f'Value changed at {path}: {left!r} != {right!r}'


def run(dataset, output):
    config = yaml.safe_load(Path(COMPOSE).read_text())
    assert config['name'] == 'cvat-vehicles-qa'
    assert all(not v.get('external') and not v.get('name') for v in config['volumes'].values())
    container_id = subprocess.check_output(
        ['docker', 'compose', '-f', COMPOSE, 'ps', '-q', 'cvat_server'], text=True,
    ).strip()
    assert container_id, 'Isolated QA server is not running'
    container = json.loads(subprocess.check_output(['docker', 'inspect', container_id], text=True))[0]
    assert container['Config']['Labels']['com.docker.compose.project'] == 'cvat-vehicles-qa'
    assert container['State']['Running']
    assert {'HostIp': '127.0.0.1', 'HostPort': '18080'} in container['NetworkSettings']['Ports']['8080/tcp']
    assert all(mount['Name'].startswith('cvat-vehicles-qa_')
               for mount in container['Mounts'] if mount['Type'] == 'volume')
    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.auth = ('admin', '12qwaszx')  # Disposable local QA account only.
    report = {'gates': {}, 'task_ids': []}

    def api(method, path, **kwargs):
        response = session.request(method, URL + path, timeout=300, **kwargs)
        if not response.ok:
            raise RuntimeError(f'{method} {path}: {response.status_code} {response.text[:1500]}')
        return response

    def wait(rq_id):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            result = api('GET', f'/api/requests/{rq_id}').json()
            if result['status'] == 'finished':
                return result
            if result['status'] == 'failed':
                raise RuntimeError(result.get('message', str(result)))
            time.sleep(.5)
        raise TimeoutError(f'Background operation did not finish: {rq_id}')

    def labels_for(task_id):
        return api('GET', '/api/labels', params={'task_id': task_id, 'page_size': 500}).json()['results']

    def create_task(labels, suffix):
        clean = [{key: label[key] for key in ('name', 'type', 'color', 'attributes') if key in label}
                 for label in copy.deepcopy(labels)]
        for label in clean:
            for spec in label['attributes']:
                spec.pop('id', None)
        task = api('POST', '/api/tasks', json={
            'name': f'Vehicles GT regression {suffix}', 'labels': clean, 'segment_size': 154, 'overlap': 0,
        }).json()
        task_id = task['id']
        report['task_ids'].append(task_id)
        with (dataset / 'cvat_data_3d.zip').open('rb') as stream:
            result = api('POST', f'/api/tasks/{task_id}/data',
                         data={'image_quality': 100, 'sorting_method': 'lexicographical', 'use_zip_chunks': 'true'},
                         files={'client_files[0]': ('cvat_data_3d.zip', stream, 'application/zip')}).json()
        wait(result['rq_id'])
        task = api('GET', f'/api/tasks/{task_id}').json()
        assert task['dimension'] == '3d' and task['size'] == 154
        return task_id

    def import_annotations(task_id, archive):
        with archive.open('rb') as stream:
            result = api('POST', f'/api/tasks/{task_id}/annotations',
                         params={'format': 'Datumaro 3D 1.0'},
                         files={'annotation_file': (archive.name, stream, 'application/zip')}).json()
        wait(result['rq_id'])

    def migration(task_id, mode):
        backup = f'/tmp/vehicles-gt-{task_id}-backup.json'
        command = ['docker', 'compose', '-f', COMPOSE, 'exec', '-T', 'cvat_server',
                   'python', 'manage.py', 'migrate_autofit_vehicles', '--task-id', str(task_id), '--mode', mode]
        if mode != 'dry-run':
            command += ['--backup', backup]
        if mode == 'apply':
            command += ['--confirm-scope', f'task:{task_id}']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        (output / f'{mode}.log').write_text(result.stdout + result.stderr)
        if mode in {'backup', 'apply'}:
            snapshot = subprocess.run(['docker', 'compose', '-f', COMPOSE, 'exec', '-T',
                                       'cvat_server', 'cat', backup], check=True, capture_output=True)
            (output / f'{mode}.json').write_bytes(snapshot.stdout)
            (output / f'{mode}.json').chmod(0o600)

    try:
        task_id = create_task(json.loads((dataset / 'labels.json').read_text()), 'source')
        print(f'Created isolated GT task {task_id}', flush=True)
        import_annotations(task_id, dataset / 'cvat_annotations_datumaro3d.zip')
        before = api('GET', f'/api/tasks/{task_id}/annotations').json()
        before_labels = labels_for(task_id)
        (output / 'before-labels.json').write_text(json.dumps(before_labels, indent=2))
        (output / 'before.json').write_text(json.dumps(before, indent=2))
        assert len(before['tracks']) == 15, f"Unexpected tracks: {len(before['tracks'])}"
        report['source_import_audit'] = source_import_audit(dataset, before, before_labels)
        report['gates']['import'] = 'PASS'
        for mode in ('dry-run', 'backup', 'apply'):
            migration(task_id, mode)
        after = api('GET', f'/api/tasks/{task_id}/annotations').json()
        after_labels = labels_for(task_id)
        (output / 'after-labels.json').write_text(json.dumps(after_labels, indent=2))
        (output / 'after.json').write_text(json.dumps(after, indent=2))
        assert normalized(before, before_labels, preserve_ids=True) == normalized(
            after, after_labels, preserve_ids=True), 'Migration changed annotation content'
        report['gates']['migration_preserves_ids_geometry_attributes'] = 'PASS'
        print('GT migration preservation PASS', flush=True)
        export = api('POST', f'/api/tasks/{task_id}/dataset/export',
                     params={'format': 'Datumaro 3D 1.0', 'save_images': 'false'}).json()
        result = wait(export['rq_id'])
        download = session.get(result['result_url'], timeout=120)
        download.raise_for_status()
        archive = output / 'vehicles-datumaro3d.zip'
        archive.write_bytes(download.content)
        clone_id = create_task(after_labels, 'roundtrip')
        import_annotations(clone_id, archive)
        roundtrip = api('GET', f'/api/tasks/{clone_id}/annotations').json()
        (output / 'roundtrip.json').write_text(json.dumps(roundtrip, indent=2))
        equivalent(normalized(after, after_labels, preserve_ids=False),
                   normalized(roundtrip, labels_for(clone_id), preserve_ids=False))
        report['gates']['export_import'] = 'PASS'
        migration(task_id, 'rollback')
        restored = api('GET', f'/api/tasks/{task_id}/annotations').json()
        assert normalized(before, before_labels, preserve_ids=True) == normalized(
            restored, labels_for(task_id), preserve_ids=True), 'Rollback did not restore exact content'
        report['gates']['rollback'] = 'PASS'
        report['complete'] = True
    except Exception as exc:
        report.update(complete=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset.expanduser(), args.output)
