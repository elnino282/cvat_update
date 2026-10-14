import http.client
import json
import threading
import time
import unittest
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer

import numpy as np

from vehicle_fit.geometry import NoFit
from vehicle_fit.service import FitRuntime, Handler


def metadata(request_id="1ab2c3d4-e5f6-4789-8123-456789abcdef"):
    return {
        "requestId": request_id, "frame": 7, "labelName": "Car", "cloud_format": "bin",
        "userId": 5, "jobId": 6, "taskId": 4, "timeout_ms": 2000,
        "seed": {"x": 1, "y": 2, "z": 3, "pointIndex": 0},
    }


def fake_fit(_cloud, _seed, **_kwargs):
    return {"center": [1.0, 2.0, 3.0], "size": [4.0, 1.8, 1.5],
            "rotation": [0.0, 0.0, .2], "confidence": .8,
            "diagnostics": {"mask_indices": np.array([0], dtype=int), "point_count": 1}}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FitRuntime()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), type("TestHandler", (Handler,),
                                                                    {"runtime": self.runtime}))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_private_http_fit_returns_single_valid_geometry_and_provenance(self):
        raw = np.array([[1, 2, 3, .5]], dtype="<f4").tobytes()
        boundary = "auto-fit-test-boundary"
        meta = metadata()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n"
            "Content-Type: application/json\r\n\r\n"
            + json.dumps(meta)
            + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"cloud\"; filename=\"frame.bin\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
        with patch("vehicle_fit.service.decode_cloud", return_value=(
            np.array([[1, 2, 3, .5]], dtype=np.float32), np.array([0])),
        ), patch("vehicle_fit.service.fit_vehicle", side_effect=fake_fit):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
            connection.request("POST", "/v1/fit", body=body,
                               headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            response = connection.getresponse()
            result = json.loads(response.read())
            connection.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "geometry-only")
        self.assertEqual(result["size"], [4.0, 1.8, 1.5])
        self.assertEqual(result["diagnostics"]["maskIndices"], [0])
        self.assertEqual(len(result["diagnostics"]["cloudSHA256"]), 64)

    def test_duplicate_request_is_single_flight_and_payload_mismatch_conflicts(self):
        raw = np.array([[1, 2, 3, .5]], dtype="<f4").tobytes()
        meta = metadata()
        calls = []

        def slow_fit(*args, **kwargs):
            calls.append(1)
            time.sleep(.1)
            return fake_fit(*args, **kwargs)

        cloud = np.array([[1, 2, 3, .5]], dtype=np.float32)
        output = []
        with patch("vehicle_fit.service.decode_cloud", return_value=(cloud, np.array([0]))), \
             patch("vehicle_fit.service.fit_vehicle", side_effect=slow_fit):
            first = threading.Thread(target=lambda: output.append(self.runtime.fit_once(meta, raw)))
            second = threading.Thread(target=lambda: output.append(self.runtime.fit_once(meta, raw)))
            first.start()
            time.sleep(.02)
            second.start()
            first.join(2)
            second.join(2)
            self.assertEqual(len(output), 2)
            self.assertEqual(len(calls), 1)
            conflict = dict(meta, **{"seed": dict(meta["seed"], x=9)})
            with self.assertRaises(LookupError):
                self.runtime.fit_once(conflict, raw)

    def test_no_fit_retains_snap_mask_diagnostics_and_quality_gate_measurements(self):
        raw = np.array([[1, 2, 3, .5]], dtype="<f4").tobytes()
        cloud = np.array([[1, 2, 3, .5]], dtype=np.float32)
        details = {"backend": "snap-outdoor", "maskPoints": 12, "stageMilliseconds": {"backbone": 10}}
        error = NoFit("ambiguous", "surface evidence does not support a car", {"surfaceScore": .22})
        self.runtime.backend_name = "snap"
        self.runtime.backend = Mock()
        self.runtime.backend.segment.return_value = (np.array([True]), details)
        with patch("vehicle_fit.service.decode_cloud", return_value=(cloud, np.array([0]))), \
             patch("vehicle_fit.service.fit_vehicle", side_effect=error):
            result = self.runtime.fit_once(dict(metadata(), timeout_ms=8000), raw)
        self.assertEqual(result["status"], "no-fit")
        self.assertEqual(result["diagnostics"]["surfaceScore"], .22)
        self.assertEqual(result["diagnostics"]["maskPoints"], 12)
        self.assertEqual(result["diagnostics"]["maskIndices"], [0])
        self.assertEqual(len(result["diagnostics"]["cloudSHA256"]), 64)


if __name__ == "__main__":
    unittest.main()

class RuntimeLifecycleTests(unittest.TestCase):
    def setUp(self):
        with patch.dict('os.environ', {'VEHICLE_FIT_BACKEND': 'geometry-only'}):
            self.runtime = FitRuntime()
        self.raw = np.array([[1, 2, 3, .5]], dtype='<f4').tobytes()

    def test_prepared_scope_expiry_and_cache_skip_decode(self):
        meta = metadata()
        prepared = self.runtime.prepare(meta, self.raw)
        self.assertEqual(prepared['state'], 'ready')
        duplicate = self.runtime.prepare(meta, self.raw)
        self.assertEqual(duplicate['frameToken'], prepared['frameToken'])
        request = dict(meta, frameToken=prepared['frameToken'])
        with patch('vehicle_fit.service.decode_cloud', side_effect=AssertionError('decoded twice')), \
             patch('vehicle_fit.service.fit_vehicle', side_effect=fake_fit):
            self.assertEqual(self.runtime.fit_once(request)['status'], 'ok')
        with self.assertRaises(ValueError):
            self.runtime.prepared_status(dict(request, userId=99))
        self.runtime._frames[prepared['frameToken']]['expires'] = 0
        from vehicle_fit.service import PreparedFrameMissing
        with self.assertRaises(PreparedFrameMissing):
            self.runtime.prepared_status(request)

    def test_owner_deadline_returns_without_waiting_for_running_work(self):
        started = threading.Event()
        release = threading.Event()
        def slow(*args, **kwargs):
            started.set()
            release.wait(2)
            return fake_fit(*args, **kwargs)
        meta = dict(metadata(), timeout_ms=40)
        with patch('vehicle_fit.service.fit_vehicle', side_effect=slow):
            before = time.monotonic()
            result = self.runtime.fit_once(meta, self.raw)
            self.assertEqual(result['reason'], 'timeout')
            self.assertLess(time.monotonic() - before, .3)
            self.assertTrue(started.is_set())
            release.set()
            self.runtime._requests[self.runtime._key(meta)]['done'].wait(1)
            self.assertEqual(self.runtime._requests[self.runtime._key(meta)]['result']['reason'], 'timeout')

    def test_geometry_deadline_is_timeout_not_explicit_cancellation(self):
        meta = metadata()
        def expired(*args, **kwargs):
            self.runtime._requests[self.runtime._key(meta)]['deadline'] = time.monotonic()-1
            raise NoFit('cancelled', 'Geometry observed expired deadline')
        with patch('vehicle_fit.service.fit_vehicle', side_effect=expired):
            result = self.runtime.fit_once(meta, self.raw)
        self.assertEqual(result['reason'], 'timeout')

    def test_cancelled_gpu_queue_never_calls_segment(self):
        self.runtime.backend_name = 'snap'
        self.runtime.backend = Mock()
        self.runtime._gpu.acquire()
        meta = dict(metadata(), timeout_ms=8000)
        output = []
        with patch('vehicle_fit.service.fit_vehicle', side_effect=fake_fit):
            thread = threading.Thread(target=lambda: output.append(self.runtime.fit_once(meta, self.raw)))
            thread.start()
            for _ in range(100):
                if self.runtime._key(meta) in self.runtime._requests:
                    break
                time.sleep(.001)
            self.assertTrue(self.runtime.cancel(self.runtime._key(meta)))
            self.runtime._gpu.release()
            thread.join(2)
        self.runtime.backend.segment.assert_not_called()
        self.assertEqual(output[0]['reason'], 'cancelled')

    def test_cold_inference_and_quality_are_independent_of_estimated_policy(self):
        from vehicle_fit.tests.test_geometry import road_scene
        cloud = road_scene().astype('<f4')
        seed = {'x': 10.05, 'y': .9, 'z': .75, 'pointIndex': 0}
        self.runtime.backend_name = 'snap'
        self.runtime.backend = None
        responses = []
        for allow in (False, True):
            meta = dict(metadata('1ab2c3d4-e5f6-4789-8123-456789abcde' + str(int(allow))),
                        seed=seed, allowEstimated=allow)
            responses.append(self.runtime.fit_once(meta, cloud.tobytes()))
        strict, relaxed = responses
        self.assertEqual(strict['status'], 'no-fit')
        self.assertEqual(strict['provider'], 'snap')
        self.assertEqual(strict['diagnostics']['assessedQuality'], 'estimated')
        self.assertEqual(strict['diagnostics']['attemptedProvider'], 'geometry-only')
        self.assertEqual(relaxed['quality'], 'estimated')
        self.assertEqual(relaxed['provider'], 'geometry-only')

    def test_warm_snap_is_selected_for_both_quality_policies(self):
        from vehicle_fit.tests.test_geometry import road_scene
        cloud = road_scene().astype('<f4')
        token = self.runtime.prepare(metadata(), cloud.tobytes())['frameToken']
        entry = self.runtime._frames[token]
        entry['prepared'] = True
        self.runtime.backend_name = 'snap'
        self.runtime.backend = Mock(_cached_key=entry['feature_key'])
        self.runtime.backend.segment.return_value = ((cloud[:, 0] <= 11.3) & (cloud[:, 2] > .1), {})
        responses = []
        for allow in (False, True):
            meta = dict(metadata('1ab2c3d4-e5f6-4789-8123-456789abcde' + str(int(allow))),
                        seed={'x': 10.05, 'y': .9, 'z': .75, 'pointIndex': 0},
                        frameToken=token, allowEstimated=allow)
            responses.append(self.runtime.fit_once(meta))
        self.assertEqual(self.runtime.backend.segment.call_count, 2)
        self.assertEqual(responses[0]['status'], 'no-fit')
        self.assertEqual(responses[0]['diagnostics']['assessedQuality'], responses[1]['quality'])
        self.assertEqual(responses[1]['provider'], 'snap')

    def test_cold_snap_estimate_is_explicit_and_does_not_wait_for_gpu(self):
        self.runtime.backend_name = 'snap'
        self.runtime.backend = None
        self.runtime._gpu.acquire()
        try:
            with patch('vehicle_fit.service.estimate_vehicle', return_value=dict(fake_fit(None, None), quality='estimated')):
                result = self.runtime.fit_once(dict(metadata(), allowEstimated=True), self.raw)
        finally:
            self.runtime._gpu.release()
        self.assertEqual(result['provider'], 'geometry-only')
        self.assertEqual(result['quality'], 'estimated')
        self.assertEqual(result['diagnostics']['fallbackFrom'], 'snap')

    def test_prepared_token_is_not_a_new_identity_and_expired_request_can_recover(self):
        expired = dict(metadata(), frameToken='0' * 32)
        self.assertEqual(self.runtime.fit_once(expired)['reason'], 'frame-expired')
        prepared = self.runtime.prepare(metadata(), self.raw)
        with patch('vehicle_fit.service.fit_vehicle', side_effect=fake_fit):
            result = self.runtime.fit_once(dict(expired, frameToken=prepared['frameToken']))
        self.assertEqual(result['status'], 'ok')

    def test_prompt_gpu_failure_recovers_to_explicit_estimate_and_schedules_prepare(self):
        meta = metadata()
        token = self.runtime.prepare(meta, self.raw)['frameToken']
        entry = self.runtime._frames[token]
        entry['prepared'] = True
        self.runtime.backend_name = 'snap'
        self.runtime.backend = Mock(_cached_key=entry['feature_key'])
        self.runtime.backend.segment.side_effect = RuntimeError('CUDA temporary failure')
        with patch('vehicle_fit.service.estimate_vehicle', return_value=dict(fake_fit(None, None), quality='estimated')), \
             patch.object(self.runtime, '_retry_prepare') as retry:
            result = self.runtime.fit_once(dict(meta, frameToken=token, allowEstimated=True))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['provider'], 'geometry-only')
        self.assertEqual(result['quality'], 'estimated')
        self.assertFalse(entry['prepared'])
        retry.assert_called_once_with(token, entry)
