"""Generate a fully isolated local CVAT backend for Vehicles browser tests."""
import argparse
import copy
from pathlib import Path

import yaml


def configuration(root):
    base = yaml.safe_load((root / 'docker-compose.yml').read_text())
    names = ('cvat_db', 'cvat_redis_inmem', 'cvat_redis_ondisk', 'cvat_opa',
             'cvat_clickhouse', 'cvat_vector', 'cvat_server', 'cvat_worker_import',
             'cvat_worker_export', 'cvat_worker_annotation', 'cvat_worker_chunks')
    services = {name: copy.deepcopy(base['services'][name]) for name in names}
    for name, service in services.items():
        service.pop('container_name', None)
        service.pop('ports', None)
        service['labels'] = {'traefik.enable': 'false'}
        service['restart'] = 'no'
        if name == 'cvat_server' or name.startswith('cvat_worker_'):
            service['image'] = 'cvat/server:autofit-pr3'
            service['environment']['NUMPROCS'] = 1
            service['environment']['CVAT_AUTOFIT_ENABLED'] = 'false'
        for volume in service.get('volumes', []):
            if isinstance(volume, dict) and volume.get('type') == 'bind':
                volume['source'] = str((root / volume['source']).resolve())
    services['cvat_server']['ports'] = ['127.0.0.1:18080:8080']
    return {'name': 'cvat-vehicles-qa', 'services': services,
            'volumes': {name: {} for name in base['volumes']}, 'networks': {'cvat': {}}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    with output.open('x') as stream:
        yaml.safe_dump(configuration(Path(__file__).resolve().parents[1]), stream, sort_keys=False)
