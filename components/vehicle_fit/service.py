"""Bounded private Auto Fit service. Prepared clouds never write annotations."""
from __future__ import annotations

from collections import OrderedDict
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

import numpy as np

from .cloud import decode_cloud
from .geometry import NoFit, VehiclePriors, apply_quality_policy, estimate_vehicle, fit_vehicle

LOGGER = logging.getLogger("vehicle_fit")
MAX_BODY_BYTES = 128 * 1024 * 1024
MAX_CACHE_ENTRIES = 512
MAX_ACTIVE_REQUESTS = 16
REQUEST_ID = re.compile(r"^[0-9a-fA-F-]{36}$")


class PreparedFrameMissing(LookupError):
    pass


class FitRuntime:
    def __init__(self):
        self.backend_name = os.getenv("VEHICLE_FIT_BACKEND", "geometry-only")
        if self.backend_name not in {"snap", "geometry-only"}:
            raise RuntimeError("VEHICLE_FIT_BACKEND must be snap or geometry-only")
        self.backend = None
        self.backend_error = None
        self.model_loading = self.backend_name == "snap"
        self.warmed = False
        defaults = VehiclePriors()
        self.priors = VehiclePriors(
            length=float(os.getenv("VEHICLE_FIT_PRIOR_LENGTH", str(defaults.length))),
            width=float(os.getenv("VEHICLE_FIT_PRIOR_WIDTH", str(defaults.width))),
            height=float(os.getenv("VEHICLE_FIT_PRIOR_HEIGHT", str(defaults.height))),
        )
        flag = os.getenv("VEHICLE_FIT_ADAPTIVE_SIZE_REFINEMENT", "false").lower()
        if flag not in {"true", "false"}:
            raise RuntimeError("VEHICLE_FIT_ADAPTIVE_SIZE_REFINEMENT must be true or false")
        self.adaptive_refinement = flag == "true"
        if not np.isfinite([self.priors.length, self.priors.width, self.priors.height]).all() or min(self.priors.length, self.priors.width, self.priors.height) <= 0:
            raise RuntimeError("Vehicle priors must be finite and positive")
        self.max_surface_score = float(os.getenv("VEHICLE_FIT_MAX_SURFACE_SCORE", "0.10"))
        if not np.isfinite(self.max_surface_score) or self.max_surface_score <= 0:
            raise RuntimeError("VEHICLE_FIT_MAX_SURFACE_SCORE must be finite and positive")
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._requests: OrderedDict[tuple, dict] = OrderedDict()
        self._frames: OrderedDict[str, dict] = OrderedDict()
        self._gpu = threading.Lock()
        self._pending_prepare = None
        self._latest_prepare = None
        if self.backend_name == "snap":
            threading.Thread(target=self._prepare_worker, daemon=True, name="auto-fit-prepare").start()

    def _prepare_worker(self):
        try:
            from .snap_backend import SNAPBackend
            self.backend = SNAPBackend()
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("SNAP initialization failed")
        finally:
            self.model_loading = False
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending_prepare is not None)
                token = self._pending_prepare
                self._pending_prepare = None
                entry = self._frames.get(token)
                if entry is not None:
                    entry["working"] = True
            if entry is None or entry["expires"] <= time.monotonic():
                continue
            if self.backend is None:
                entry["error"] = self.backend_error
                entry["working"] = False
                continue
            try:
                with self._gpu:
                    with self._lock:
                        if token != self._latest_prepare:
                            continue
                    # A real point warms both the full scene encoder and prompt decoder.
                    self.backend.segment(entry["cloud"], entry["cloud"][0, :3], frame_key=entry["feature_key"])
                    self.warmed = True
                    entry["prepared"] = True
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                if "out of memory" in str(exc).lower():
                    self.backend.clear_cache()
                self._retry_prepare(token, entry)
                LOGGER.exception("Frame preparation failed frame=%s", entry["scope"][3])
            finally:
                entry["working"] = False

    def _retry_prepare(self, token, entry):
        if entry.get("retries", 0) >= 2:
            return
        entry["retries"] = entry.get("retries", 0) + 1

        def retry():
            with self._condition:
                if self._frames.get(token) is not entry or self._latest_prepare != token or entry["expires"] <= time.monotonic():
                    return
                entry["error"] = None
                entry["prepared"] = False
                self._pending_prepare = token
                self._condition.notify()
        timer = threading.Timer(5, retry)
        timer.daemon = True
        timer.start()

    def health(self):
        state = "ready" if self.backend_name == "geometry-only" or self.warmed else (
            "initializing" if self.model_loading else "model-ready" if self.backend else "unavailable")
        return {
            "status": state, "backend": self.backend_name,
            "snapAvailable": self.backend is not None if self.backend_name == "snap" else None,
            "geometryAvailable": True, "error": self.backend_error,
            "vehiclePriorsMetres": {"length": self.priors.length, "width": self.priors.width,
                                    "height": self.priors.height},
            "maxSurfaceScore": self.max_surface_score,
            "adaptiveSizeRefinement": self.adaptive_refinement,
            "provenance": getattr(self.backend, "provenance", {}),
        }

    @staticmethod
    def _scope(meta):
        return (meta["userId"], meta["taskId"], meta["jobId"], meta["frame"], str(meta.get("dataVersion", "")))

    @staticmethod
    def _key(meta):
        return (meta["userId"], meta["taskId"], meta["jobId"], meta["frame"], meta["requestId"])

    def _evict_frames(self):
        now = time.monotonic()
        for token in list(self._frames):
            if self._frames[token]["expires"] <= now:
                del self._frames[token]
        while len(self._frames) > 3 or sum(e["bytes"] for e in self._frames.values()) > 128 * 1024 * 1024:
            self._frames.popitem(last=False)

    def _frame(self, meta):
        with self._lock:
            self._evict_frames()
            token = meta.get("frameToken")
            entry = self._frames.get(token)
            if entry is None:
                raise PreparedFrameMissing("Prepared frame expired; prepare this frame again")
            if entry["scope"] != self._scope(meta):
                raise ValueError("Prepared frame scope mismatch")
            entry["expires"] = time.monotonic() + 120
            self._frames.move_to_end(token)
            return entry

    def _frame_status(self, token, entry):
        ready = self.backend_name == "geometry-only" or (
            entry["prepared"] and getattr(self.backend, "_cached_key", None) == entry["feature_key"])
        return {"frameToken": token, "frame": entry["scope"][3],
                "state": "ready" if ready else "unavailable" if entry["error"] else "preparing",
                "geometryReady": True, "backend": self.backend_name}

    def prepare(self, meta, cloud_bytes):
        if meta["cloud_format"] not in {"bin", "pcd"} or not 0 < len(cloud_bytes) <= MAX_BODY_BYTES:
            raise ValueError("Invalid point cloud")
        cloud_hash = hashlib.sha256(cloud_bytes).hexdigest()
        scope = self._scope(meta)
        with self._condition:
            self._evict_frames()
            for token, entry in list(self._frames.items()):
                if entry["scope"] == scope and entry["hash"] == cloud_hash:
                    entry["expires"] = time.monotonic() + 120
                    self._frames.move_to_end(token)
                    if self._frame_status(token, entry)["state"] != "ready" and not entry["error"] and not entry["working"]:
                        self._pending_prepare = self._latest_prepare = token
                        self._condition.notify()
                    return self._frame_status(token, entry)
        cloud, indices = decode_cloud(cloud_bytes, meta["cloud_format"])
        if cloud.nbytes + indices.nbytes > 128 * 1024 * 1024:
            raise ValueError("Decoded frame exceeds preparation cache")
        with self._condition:
            # Resolve uploads of the same frame that completed decoding concurrently.
            for token, entry in self._frames.items():
                if entry["scope"] == scope and entry["hash"] == cloud_hash:
                    return self._frame_status(token, entry)
            token = uuid.uuid4().hex
            entry = {"scope": scope, "cloud": cloud, "indices": indices, "hash": cloud_hash,
                     "feature_key": hashlib.sha256(cloud.tobytes()).hexdigest(),
                     "bytes": cloud.nbytes + indices.nbytes, "expires": time.monotonic() + 120,
                     "prepared": False, "working": False, "error": self.backend_error}
            self._frames[token] = entry
            self._evict_frames()
            self._pending_prepare = self._latest_prepare = token
            self._condition.notify()
            return self._frame_status(token, entry)

    def prepared_status(self, meta):
        entry = self._frame(meta)
        with self._condition:
            result = self._frame_status(meta["frameToken"], entry)
            if result["state"] == "preparing" and not entry["working"]:
                self._pending_prepare = self._latest_prepare = meta["frameToken"]
                self._condition.notify()
            return result

    def cancel(self, key):
        with self._lock:
            entry = self._requests.get(key)
            if entry is not None:
                entry["cancelled"] = True
                return True
        return False

    def _fit(self, meta, cloud_bytes):
        result_base = {"requestId": str(meta["requestId"]), "frame": int(meta["frame"]),
                       "provider": self.backend_name}
        diagnostics = {}
        mask_indices = None
        cloud_hash = ""
        started = time.perf_counter()
        try:
            if meta.get("frameToken"):
                prepared = self._frame(meta)
                cloud, original_indices, cloud_hash = prepared["cloud"], prepared["indices"], prepared["hash"]
            else:
                prepared = None
                cloud_hash = hashlib.sha256(cloud_bytes).hexdigest()
                cloud, original_indices = decode_cloud(cloud_bytes, meta["cloud_format"])
            seed = np.asarray([meta["seed"][axis] for axis in ("x", "y", "z")], dtype=np.float64)
            if not np.isfinite(seed).all():
                raise ValueError("seed contains a non-finite coordinate")
            entry = self._requests[self._key(meta)]
            allow_estimated = meta.get("allowEstimated") is True
            interactive = float(meta.get("timeout_ms", 16000)) <= 2000

            def cancelled():
                return entry["cancelled"] or time.monotonic() >= entry["deadline"]

            def check():
                if cancelled():
                    raise NoFit("cancelled" if entry["cancelled"] else "timeout", "Operation ended")

            check()
            mask = None
            use_snap = self.backend_name == "snap" and self.backend is not None
            # Interactive clients never pay the first full-scene CUDA inference.
            if interactive and (not prepared or self._frame_status(meta["frameToken"], prepared)["state"] != "ready"):
                use_snap = False
            if use_snap:
                remaining = max(0, entry["deadline"] - time.monotonic())
                acquired = self._gpu.acquire(timeout=min(remaining, .2 if interactive else remaining))
                if acquired:
                    try:
                        check()
                        if interactive and getattr(self.backend, "_cached_key", None) != prepared["feature_key"]:
                            use_snap = False
                        else:
                            kwargs = {"frame_key": prepared["feature_key"]} if prepared else {}
                            try:
                                mask, diagnostics = self.backend.segment(cloud, seed, **kwargs)
                                mask_indices = np.flatnonzero(mask)
                            except Exception as exc:
                                use_snap = False
                                if prepared:
                                    prepared["prepared"] = False
                                    prepared["error"] = f"{type(exc).__name__}: {exc}"
                                if "out of memory" in str(exc).lower():
                                    self.backend.clear_cache()
                                if prepared:
                                    self._retry_prepare(meta["frameToken"], prepared)
                                LOGGER.exception("SNAP prompt failed; using explicit geometry estimate")
                            check()
                    finally:
                        self._gpu.release()
                else:
                    use_snap = False
            if self.backend_name == "snap" and not use_snap:
                diagnostics = {"fallbackFrom": "snap", "fallbackReason": "model-or-frame-not-ready"}
                fit = estimate_vehicle(cloud, seed, priors=self.priors, cancelled=cancelled,
                                       adaptive_refinement=self.adaptive_refinement)
                diagnostics["attemptedProvider"] = "geometry-only"
                fit = apply_quality_policy(fit, allow_estimated)
                result_base["provider"] = "geometry-only"
            else:
                fit = fit_vehicle(cloud, seed, mask=mask, priors=self.priors, cancelled=cancelled,
                                  max_surface_score=self.max_surface_score, allow_estimated=allow_estimated,
                                  adaptive_refinement=self.adaptive_refinement)
            check()
            mask_indices = fit["diagnostics"].pop("mask_indices", mask_indices)
            diagnostics.update(fit.pop("diagnostics"))
            if self.backend_name == "snap" and diagnostics.get("maskSource") == "seed-connected":
                result_base["provider"] = "geometry-only"
                diagnostics["fallbackFrom"] = "snap"
            if mask_indices is not None:
                diagnostics["maskIndices"] = original_indices[np.asarray(mask_indices, dtype=int)].tolist()
            diagnostics.update({"rawPoints": len(cloud), "cloudSHA256": cloud_hash,
                                "latencyMilliseconds": (time.perf_counter() - started) * 1000})
            check()  # Includes diagnostic/index construction before publication.
            return {"status": "ok", **result_base, **fit, "diagnostics": diagnostics}
        except PreparedFrameMissing:
            return {"status": "no-fit", **result_base, "reason": "frame-expired"}
        except NoFit as exc:
            diagnostics.update(exc.diagnostics)
            diagnostics.update(detail=exc.detail, cloudSHA256=cloud_hash)
            if mask_indices is not None:
                diagnostics["maskIndices"] = original_indices[np.asarray(mask_indices, dtype=int)].tolist()
            reason = exc.reason
            request = self._requests.get(self._key(meta))
            if reason == "cancelled" and request and not request["cancelled"] and time.monotonic() >= request["deadline"]:
                reason = "timeout"
            return {"status": "no-fit", **result_base, "reason": reason, "diagnostics": diagnostics}
        except ValueError as exc:
            return {"status": "no-fit", **result_base, "reason": "invalid-seed", "diagnostics": {"detail": str(exc)}}
        except Exception as exc:
            LOGGER.exception("Fit failed frame=%s", meta["frame"])
            return {"status": "no-fit", **result_base, "reason": "unavailable", "diagnostics": {"detail": str(exc)}}

    def fit_once(self, meta, cloud_bytes=b""):
        if meta["labelName"].strip().lower() != "car":
            raise ValueError("Only the Car label is supported")
        if meta["cloud_format"] not in {"bin", "pcd"} or len(cloud_bytes) > MAX_BODY_BYTES:
            raise ValueError("Invalid cloud format or request size")
        budget = min(16.0, max(.001, float(meta.get("timeout_ms", 16000)) / 1000))
        key = self._key(meta)
        payload_hash = hashlib.sha256(json.dumps(
            {"metadata": meta, "cloud": hashlib.sha256(cloud_bytes).hexdigest()},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self._lock:
            entry = self._requests.get(key)
            if entry:
                if entry["payload_hash"] != payload_hash:
                    raise LookupError("requestId already used for another payload")
                self._requests.move_to_end(key)
            else:
                completed = [k for k, v in self._requests.items() if v["done"].is_set()]
                for old in completed[:max(0, len(self._requests) - MAX_CACHE_ENTRIES + 1)]:
                    del self._requests[old]
                if sum(not v["done"].is_set() for v in self._requests.values()) >= MAX_ACTIVE_REQUESTS:
                    return {"status": "no-fit", "requestId": str(meta["requestId"]), "frame": meta["frame"],
                            "provider": self.backend_name, "reason": "unavailable"}
                entry = {"payload_hash": payload_hash, "done": threading.Event(), "cancelled": False,
                         "result": None, "deadline": time.monotonic() + budget}
                self._requests[key] = entry

                def run():
                    try:
                        entry["result"] = self._fit(meta, cloud_bytes)
                    finally:
                        entry["done"].set()
                threading.Thread(target=run, daemon=True, name="auto-fit-request").start()
        entry["done"].wait(max(0, min(budget, entry["deadline"] - time.monotonic())))
        with self._lock:
            reason = "cancelled" if entry["cancelled"] else "timeout" if not entry["done"].is_set() else None
            if reason:
                return {"status": "no-fit", "requestId": str(meta["requestId"]), "frame": int(meta["frame"]),
                        "provider": self.backend_name, "reason": reason}
            result = dict(entry["result"])
            if result.get("reason") == "frame-expired":
                self._requests.pop(key, None)
            return result


RUNTIME = FitRuntime()


class Handler(BaseHTTPRequestHandler):
    server_version = "vehicle-fit/2"
    runtime = RUNTIME

    def log_message(self, format_string, *args):
        LOGGER.info("%s - %s", self.address_string(), format_string % args)

    def _json(self, code, value):
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            health = self.runtime.health()
            self._json(503 if health["status"] == "unavailable" else 200, health)
        elif parsed.path == "/v1/prepare":
            try:
                params = parse_qs(parsed.query)
                meta = {name: int(params[name][0]) for name in ("userId", "taskId", "jobId", "frame")}
                meta.update(frameToken=params["frameToken"][0], dataVersion=params.get("dataVersion", [""])[0])
                self._json(200, self.runtime.prepared_status(meta))
            except PreparedFrameMissing as exc:
                self._json(410, {"detail": str(exc)})
            except (ValueError, KeyError) as exc:
                self._json(400, {"detail": str(exc)})
        else:
            self._json(404, {"detail": "not found"})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        match = re.fullmatch(r"/v1/requests/([0-9a-fA-F-]{36})", parsed.path)
        params = parse_qs(parsed.query)
        try:
            if not match:
                raise ValueError("invalid request ID")
            scope = tuple(int(params[name][0]) for name in ("user_id", "task_id", "job_id"))
            with self.runtime._lock:
                matches = [k for k in self.runtime._requests if k[:3] == scope and k[4] == match.group(1)]
            self._json(200, {"cancelled": any(self.runtime.cancel(k) for k in matches)})
        except (KeyError, ValueError) as exc:
            self._json(400, {"detail": str(exc)})

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in {"/v1/fit", "/v1/prepare"}:
            self._json(404, {"detail": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES + 1024 * 1024:
                self._json(413, {"detail": "invalid request size"})
                return
            content_type = self.headers.get("Content-Type", "")
            raw = self.rfile.read(length)
            if content_type.startswith("application/json") and path == "/v1/fit":
                metadata = json.loads(raw)
                cloud = b""
                if not metadata.get("frameToken"):
                    raise ValueError("A prepared frame token is required for JSON fitting")
            elif content_type.lower().startswith("multipart/form-data;"):
                message = BytesParser(policy=policy.default).parsebytes(
                    b"MIME-Version: 1.0\r\nContent-Type: " + content_type.encode() + b"\r\n\r\n" + raw)
                fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                          for part in message.iter_parts()}
                metadata = json.loads(fields["metadata"].decode("utf-8"))
                cloud = fields["cloud"]
            else:
                self._json(415, {"detail": "expected multipart cloud or prepared JSON"})
                return
            required = {"frame", "cloud_format", "userId", "jobId", "taskId"}
            if path == "/v1/fit":
                required |= {"requestId", "seed", "labelName", "timeout_ms"}
            if not required.issubset(metadata):
                raise ValueError("missing request identity")
            if path == "/v1/fit" and not REQUEST_ID.fullmatch(str(metadata["requestId"])):
                raise ValueError("invalid request ID")
            result = self.runtime.prepare(metadata, cloud) if path == "/v1/prepare" else self.runtime.fit_once(metadata, cloud)
            self._json(202 if path == "/v1/prepare" and result["state"] == "preparing" else 200, result)
        except PreparedFrameMissing as exc:
            self._json(410, {"detail": str(exc)})
        except LookupError as exc:
            self._json(409, {"detail": str(exc)})
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            self._json(400, {"detail": str(exc)})


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    server = ThreadingHTTPServer((os.getenv("VEHICLE_FIT_HOST", "0.0.0.0"), int(os.getenv("VEHICLE_FIT_PORT", "8093"))), Handler)
    LOGGER.info("starting %s", json.dumps(RUNTIME.health(), sort_keys=True))
    server.serve_forever(poll_interval=.25)


if __name__ == "__main__":
    main()
