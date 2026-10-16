"""Exclusive GPU regression window; always restore the pre-existing fitter."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


ORIGINAL = 'cvat-vehicle_fit-1'
QA = 'cvat-vehicles-gpu-qa'
ENDPOINT = 'http://127.0.0.1:18093'


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


def run(args):
    if not args.allow_service_interruption:
        raise ValueError('Explicit service-interruption authorization is required')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads(docker('inspect', ORIGINAL))[0]
    assert original['State']['Running'], 'Original fitter must be running before the test window'
    provenance = {'original_image_id': original['Image'], 'qa_image': 'cvat/vehicle-fit:vehicles-qa'}
    for filename in ('geometry.py', 'snap_backend.py'):
        path = f'components/vehicle_fit/{filename}'
        baseline = subprocess.check_output(['git', 'show', f'HEAD:{path}'], cwd=root)
        current = (root / path).read_bytes()
        assert current == baseline, f'Protected fitter source changed: {filename}'
        provenance[filename] = hashlib.sha256(current).hexdigest()
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        code = cuda.cuInit(0)
        error = ctypes.c_char_p()
        cuda.cuGetErrorString(code, ctypes.byref(error))
        provenance['host_cuda'] = {'code': code, 'detail': error.value.decode() if error.value else ''}
    except OSError as exc:
        provenance['host_cuda'] = {'code': None, 'detail': str(exc)}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    if provenance['host_cuda']['code'] != 0:
        report = {'status': 'BLOCKED', 'stage': 'host-cuda-preflight',
                  'host_cuda': provenance['host_cuda'], 'original_service_interrupted': False}
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return 2
    stopped = False
    qa_started = False
    try:
        docker('stop', '--time', '20', ORIGINAL)
        stopped = True
        docker('run', '--rm', '-d', '--name', QA, '--gpus', 'all',
               '-p', '127.0.0.1:18093:8093',
               '--mount', f'type=bind,src={root / "components/vehicle_fit/weights/SNAP_Outdoor.pth"},dst=/weights/SNAP_Outdoor.pth,readonly',
               '-e', 'VEHICLE_FIT_BACKEND=snap', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=2',
               '-e', 'VEHICLE_FIT_PRIOR_LENGTH=3.7', '-e', 'VEHICLE_FIT_PRIOR_WIDTH=1.61',
               '-e', 'VEHICLE_FIT_PRIOR_HEIGHT=1.49', '-e', 'VEHICLE_FIT_MAX_SURFACE_SCORE=0.10',
               '-e', 'VEHICLE_FIT_ADAPTIVE_SIZE_REFINEMENT=false', 'cvat/vehicle-fit:vehicles-qa')
        qa_started = True
        provenance['qa_image_id'] = json.loads(docker('inspect', QA))[0]['Image']
        for _ in range(120):
            try:
                with urllib.request.urlopen(ENDPOINT + '/health', timeout=3) as response:
                    health = json.load(response)
                if health['status'] in {'model-ready', 'ready'}:
                    provenance['health'] = health
                    break
            except urllib.error.HTTPError as exc:
                health = json.load(exc)
                if health.get('status') == 'unavailable':
                    provenance['health'] = health
                    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
                    raise RuntimeError(f"SNAP unavailable: {health.get('error')}") from exc
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(.5)
        else:
            raise RuntimeError('Isolated SNAP model did not initialize')
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        print('Isolated SNAP model ready; starting one-case smoke', flush=True)
        environment = dict(os.environ, PYTHONPATH=str(root / 'components'))
        common = [sys.executable, '-m', 'vehicle_fit.evaluation.drive_0005',
                  '--dataset', str(args.dataset.expanduser()), '--endpoint', ENDPOINT]
        for name, extra in [('smoke', ['--limit', '1', '--paired-legacy', '1']),
                            ('full', ['--paired-legacy', '248'])]:
            with (output / f'{name}.log').open('w') as log:
                result = subprocess.run(common + ['--output', str(output / name)] + extra,
                                        env=environment, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            print(f'{name} evaluation exit={result.returncode}', flush=True)
            if result.returncode:
                (output / 'report.json').write_text(json.dumps({
                    'status': 'BLOCKED', 'stage': name, 'exit_code': result.returncode,
                }, indent=2))
                return result.returncode
        (output / 'report.json').write_text(json.dumps({'status': 'completed', 'stage': 'full'}, indent=2))
        return 0
    except Exception as exc:
        (output / 'report.json').write_text(json.dumps({
            'status': 'BLOCKED', 'stage': 'startup', 'error': f'{type(exc).__name__}: {exc}',
        }, indent=2))
        raise
    finally:
        if qa_started:
            logs = subprocess.run(['docker', 'logs', QA], capture_output=True, text=True)
            (output / 'service.log').write_text(logs.stdout + logs.stderr)
            try:
                docker('stop', '--time', '20', QA)
            finally:
                if stopped:
                    docker('start', ORIGINAL)
                    stopped = False
        if stopped:
            docker('start', ORIGINAL)
        print('Original fitter restored', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-service-interruption', action='store_true')
    raise SystemExit(run(parser.parse_args()))
