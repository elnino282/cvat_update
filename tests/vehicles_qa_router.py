"""Run via manage.py shell in cvat-vehicles-qa after vehicles_gt_roundtrip.py.

Exercises HTTP view permissions and schema routing against imported GT rows.
The car service response is stubbed: this proves routing, not SNAP quality.
"""
import json
import uuid
from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework.test import APIClient

from cvat.apps.dataset_manager.task import get_task_data
from cvat.apps.engine.models import Task, Job


task = Task.objects.filter(name='Vehicles GT regression roundtrip').latest('id')
assert task.dimension == '3d'
job = Job.objects.get(segment__task=task)
label = task.label_set.get(name='Vehicles')
spec = label.attributespec_set.get(name='type')
before = get_task_data(task.id)
client = APIClient()
client.force_authenticate(user=get_user_model().objects.get(username='admin'))
payload = {
    'requestId': str(uuid.uuid4()), 'requestedAt': 1000,
    'taskId': task.id, 'jobId': job.id, 'frame': 109, 'labelId': label.id,
    'typeAttributeId': spec.id, 'vehicleType': 'car', 'objectType': 'shape',
    'seed': {'x': 0, 'y': 0, 'z': 0, 'pointIndex': 0},
}
checks = []
with override_settings(CVAT_AUTOFIT_ENABLED=True, CVAT_AUTOFIT_BACKEND='snap'):
    with patch('cvat.apps.engine.auto_fit.JobFrameProvider', side_effect=AssertionError('Unexpected frame read')), \
         patch('cvat.apps.engine.auto_fit.requests.post', side_effect=AssertionError('Unexpected fitter call')):
        for kind in ('truck', 'bus', 'van', 'other'):
            response = client.post(f'/api/jobs/{job.id}/auto-fit', dict(payload, vehicleType=kind), format='json')
            assert response.status_code == 200, response.content
            assert response.data['status'] == 'no-fit' and response.data['reason'] == 'unsupported-type'
            checks.append(f'{kind}: no-fit without frame read or fitter')
        for changes in ({'vehicleType': None}, {'vehicleType': 'plane'}, {'typeAttributeId': None},
                        {'typeAttributeId': spec.id + 100000}):
            response = client.post(f'/api/jobs/{job.id}/auto-fit', dict(payload, **changes), format='json')
            assert response.status_code == 400, response.content
            checks.append(f'invalid selection rejected: {changes}')
    fitted = {'status': 'ok', 'requestId': payload['requestId'], 'frame': 109, 'provider': 'snap',
              'center': [1, 2, 3], 'size': [4, 1.8, 1.5], 'rotation': [0, 0, 0], 'quality': 'confident'}
    with patch('cvat.apps.engine.auto_fit.requests.post', return_value=Mock(status_code=200)) as post:
        post.return_value.json.return_value = fitted
        response = client.post(f'/api/jobs/{job.id}/auto-fit', payload, format='json')
        assert response.status_code == 200 and response.data == fitted, response.content
        post.assert_called_once()
        metadata = json.loads(post.call_args.kwargs['data']['metadata'])
        assert metadata['vehicleType'] == 'car' and metadata['typeAttributeId'] == spec.id
        assert len(post.call_args.kwargs['files']['cloud'][1]) > 1000
        checks.append('car: real PCD forwarded once to stubbed fitter')
assert get_task_data(task.id) == before, 'Auto Fit endpoint mutated stored annotations'
print(json.dumps({'status': 'PASS', 'checks': checks, 'annotation_mutations': 0}, indent=2))
