"""Versioned PR0 dev prototypes. Descriptive ranges remain experimental."""
from pathlib import Path
import json

import numpy as np


class PriorRegistry:
    def __init__(self, path=None):
        path = path or Path(__file__).with_name('data') / 'nuscenes_pr0_priors.json'
        self.document = json.loads(Path(path).read_text())
        self.version = self.document['prior_version']
        self.records = self.document['records']
        for record in self.records:
            values = np.array([record[key] for key in ('p10_lwh', 'median_lwh', 'p90_lwh')])
            if values.shape != (3, 3) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('Prior dimensions must be finite positive L/W/H')
            if np.any(np.diff(values, axis=0) < 0):
                raise ValueError('Prior quantiles must be ordered')

    def pool(self, class_key, ride_state=None, *, reliable=True):
        label, kind = ('vehicles', class_key.split('/')[1]) if class_key.startswith('vehicles/') else (class_key, None)
        records = [r for r in self.records if r['target_label'] == label and r['vehicles_type'] == kind
                   and (label != 'two-wheels' or r['ride_state'] is ride_state)]
        if reliable:
            records = [r for r in records if r['split'] == 'dev' and r['status'] == 'development-prototype'
                       and r['instance_count'] >= 8 and r['independent_instance_count'] >= 8]
        return records


REGISTRY = PriorRegistry()
