#!/usr/bin/env python3
"""Create retained, small nuScenes tasks and probe the real local CVAT Auto Fit API.

Selections are fixed before inference. This is interaction testing, not a benchmark.
The dataset is only read; no existing task or annotation is changed or deleted.
Credentials are read once from the explicitly supplied JSON file and never logged.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile
import time
import uuid

import requests


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "artifacts/autofit-eval"
ROUTES = ("vehicles/car", "pedestrian", "vehicles/truck", "vehicles/bus", "vehicles/van", "two-wheels")
LABELS = [
    {"name": "vehicles", "type": "cuboid", "attributes": [{
        "name": "type", "mutable": False, "input_type": "select", "default_value": "other",
        "values": ["car", "truck", "bus", "van", "other"],
    }]},
    {"name": "pedestrian", "type": "cuboid", "attributes": [{
        "name": name, "mutable": False, "input_type": "text", "default_value": "", "values": [],
    } for name in ("age", "pose")]},
    {"name": "two-wheels", "type": "cuboid", "attributes": [{
        "name": "ride_state", "mutable": False, "input_type": "checkbox",
        "default_value": "false", "values": ["false"],
    }]},
    {"name": "Animal", "type": "cuboid", "attributes": []},
    {"name": "Obstacle", "type": "cuboid", "attributes": []},
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest(path):
    index = json.loads((EVIDENCE / "evidence-index.json").read_bytes())
    expected = index["regenerated_manifest"]
    raw = Path(path).read_bytes()
    if digest(raw) != expected["sha256"] or len(raw) != expected["bytes"]:
        raise ValueError("Manifest does not match the committed PR0 evidence index")
    return json.loads(gzip.decompress(raw)), expected["sha256"], index


def read_fixed_ledger(index):
    raw = (EVIDENCE / index["archive"]).read_bytes()
    if digest(raw) != index["archive_sha256"]:
        raise ValueError("Archived PR0 ledger checksum mismatch")
    name = "baseline-fixed-snap/cases.jsonl"
    expected = next(item for item in index["entries"] if item["path"] == name)
    with tarfile.open(EVIDENCE / index["archive"], "r:gz") as archive:
        raw = archive.extractfile(name).read()
    if digest(raw) != expected["sha256"] or len(raw) != expected["bytes"]:
        raise ValueError("Fixed Car ledger checksum mismatch")
    return [json.loads(line) for line in raw.splitlines() if line], expected["sha256"]


def route(case):
    return "vehicles/" + case["vehicles_type"] if case["target_label"] == "vehicles" else case["target_label"]


def select_cases(manifest, class_key, count, ledger=()):
    if not 1 <= count <= 5:
        raise ValueError("Handoff count must be between 1 and 5")
    if class_key not in ROUTES:
        raise ValueError("Unsupported handoff route")
    all_cases = {case["case_id"]: case for case in manifest["cases"]}
    selected = []
    if class_key == "vehicles/car":
        rows = sorted((row for row in ledger if row["route"] == class_key), key=lambda row: row["case_id"])
        accepted = [row for row in rows if row["response"].get("status") == "ok"]
        rejected = [row for row in rows if row["response"].get("status") == "no-fit"]
        if len(accepted) < 2 or not rejected:
            raise ValueError("PR0 fixed ledger lacks the required 2 accepted + 1 no-fit Car examples")
        ordered = accepted[:2] + rejected[:1] + accepted[2:]
        for row in ordered[:count]:
            case = all_cases[row["case_id"]]
            if not case.get("seeds"):
                raise ValueError("Fixed Car reference has no seed in the locked manifest")
            selected.append((case, "pr0-accepted" if row["response"]["status"] == "ok" else "pr0-no-fit", row))
        return selected

    source_route = "vehicles/car" if class_key == "vehicles/van" else class_key
    cases = sorted((case for case in manifest["cases"] if case["split"] == "dev" and
        route(case) == source_route and case.get("seeds")), key=lambda case: case["annotation_token"])
    if not cases:
        raise ValueError("No seeded development cases for requested route")
    strata = [
        ("dense", lambda case: case["gt_point_count"] > 50),
        ("5-15-points", lambda case: 5 <= case["gt_point_count"] <= 15),
        ("sparse-1-4", lambda case: 1 <= case["gt_point_count"] <= 4),
        ("adjacent", lambda case: bool(case["adjacent_annotation_tokens"]) and case["gt_point_count"] >= 16),
        ("16-50-points", lambda case: 16 <= case["gt_point_count"] <= 50),
    ]
    used = set()
    for index, (name, predicate) in enumerate(strata[:count]):
        candidates = [case for case in cases if case["case_id"] not in used and predicate(case)]
        if class_key == "two-wheels" and index < 2:
            # Preserve the support strata while explicitly covering both ride states.
            wanted_ride_state = index == 1
            with_state = [case for case in candidates if case["ride_state"] is wanted_ride_state]
            candidates = with_state or candidates
        if candidates:
            selected.append((candidates[0], name, None))
            used.add(candidates[0]["case_id"])
    if class_key == "two-wheels" and count >= 2:
        for wanted in (False, True):
            if any(case["ride_state"] is wanted for case, _, _ in selected):
                continue
            candidate = next((case for case in cases if case["ride_state"] is wanted and case["case_id"] not in used), None)
            if candidate is not None:
                selected[-1] = (candidate, "ride-state-coverage", None)
    return selected


def cloud_to_pcd(raw):
    """Drop nuScenes ring only. Keep XYZI bytes and every original vertex in order."""
    if not raw or len(raw) % 20:
        raise ValueError("nuScenes point cloud must have 5 float32 channels")
    for offset in range(0, len(raw), 20):
        if not all(math.isfinite(value) for value in struct.unpack_from("<4f", raw, offset)):
            raise ValueError("Nonfinite XYZI: refusing to filter or silently reindex cloud")
    count = len(raw) // 20
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\n" + f"WIDTH {count}\nHEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {count}\nDATA binary\n").encode()
    return header + b"".join(raw[offset:offset + 16] for offset in range(0, len(raw), 20))


def load_cloud(dataroot, frame, seed):
    root = Path(dataroot).expanduser().resolve(strict=True)
    cloud_path = (root / frame["lidar_path"]).resolve(strict=True)
    if not cloud_path.is_relative_to(root):
        raise ValueError("Manifest cloud path escapes dataset root")
    raw = cloud_path.read_bytes()
    if digest(raw) != frame["sha256"] or frame["point_dimensions"] != 5:
        raise ValueError("Source cloud checksum or point stride differs from PR0")
    index = seed["pointIndex"]
    if type(index) is not int or not 0 <= index < len(raw) // 20:
        raise ValueError("Seed index is outside original cloud")
    point = struct.unpack_from("<3f", raw, index * 20)
    if not all(math.isfinite(seed[key]) and abs(point[i] - seed[key]) <= 1e-5
        for i, key in enumerate(("x", "y", "z"))):
        raise ValueError("Seed coordinates do not match original vertex")
    return cloud_to_pcd(raw)


class API:
    def __init__(self, base_url, credentials_path):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        credentials = json.loads(Path(credentials_path).read_bytes())
        response = self.session.post(self.base + "/api/auth/login", json={
            "username": credentials["username"], "password": credentials["password"],
        }, timeout=20)
        if not response.ok:
            raise RuntimeError(f"CVAT login failed: HTTP {response.status_code}")
        token = response.json().get("key")
        if token:
            self.session.headers["Authorization"] = "Token " + token
        csrf = self.session.cookies.get("csrftoken")
        if csrf:
            self.session.headers["X-CSRFToken"] = csrf

    def call(self, method, path, *, check=True, **kwargs):
        response = self.session.request(method, self.base + path, timeout=kwargs.pop("timeout", 30), **kwargs)
        if check and not response.ok:
            raise RuntimeError(f"CVAT {method} {path}: HTTP {response.status_code}")
        return response


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def resolve_schema(labels):
    by_name = {label["name"]: label for label in labels}
    if set(by_name) != {label["name"] for label in LABELS}:
        raise ValueError("Created task taxonomy differs from locked label names")
    for expected in LABELS:
        label = by_name[expected["name"]]
        if label["type"] != "cuboid" or type(label.get("id")) is not int:
            raise ValueError("Runtime label schema is invalid")
        actual_attrs = {item["name"]: item for item in label["attributes"]}
        if set(actual_attrs) != {item["name"] for item in expected["attributes"]}:
            raise ValueError("Runtime attributes differ from locked taxonomy")
        for attr in expected["attributes"]:
            actual = actual_attrs[attr["name"]]
            values = actual["values"]
            # CVAT stores unconstrained text values as one empty string.
            if attr["input_type"] == "text" and values == [""]:
                values = []
            if any(actual[key] != attr[key] for key in ("mutable", "input_type")) or values != attr["values"]:
                raise ValueError("Runtime attribute semantics differ from locked taxonomy")
    vehicle = by_name["vehicles"]
    two_wheels = by_name["two-wheels"]
    return {"vehicleLabelId": vehicle["id"], "typeAttributeId": vehicle["attributes"][0]["id"],
        "pedestrianLabelId": by_name["pedestrian"]["id"], "twoWheelsLabelId": two_wheels["id"],
        "rideStateAttributeId": two_wheels["attributes"][0]["id"]}


def create(args):
    output = Path(args.output).resolve()
    previous = json.loads(output.read_bytes()) if output.exists() else None
    if previous and previous.get("status") not in ("selected", "created"):
        raise ValueError("Fixture already ready; choose a fresh stage path")
    if previous and "taskId" in previous and "upload_request_id" not in previous:
        raise ValueError("Task exists with interrupted upload; inspect it before retrying")
    root = Path(args.dataroot).expanduser().resolve(strict=True)
    if output.is_relative_to(root):
        raise ValueError("Output must not write into the read-only dataset")
    manifest, manifest_hash, index = read_manifest(args.manifest)
    ledger, ledger_hash = read_fixed_ledger(index) if args.class_key == "vehicles/car" else ([], None)
    selected = select_cases(manifest, args.class_key, args.count, ledger)
    files = []
    frames = []
    for frame_number, (case, stratum, baseline) in enumerate(selected):
        primary = case["seeds"][0]
        seed = {**dict(zip(("x", "y", "z"), primary["point"])), "pointIndex": primary["pointIndex"]}
        if baseline is not None:
            seed = dict(baseline["request"]["seed"])
        source = manifest["frames"][case["sample_token"]]
        pcd = load_cloud(root, source, seed)
        filename = f"{frame_number:06d}_{case['sample_token']}.pcd"
        files.append((f"client_files[{frame_number}]", (filename, pcd, "application/octet-stream")))
        reference = None if baseline is None else {key: baseline["response"].get(key) for key in
            ("status", "quality", "reason", "center", "size", "rotation")}
        frames.append({"frame": frame_number, "filename": filename, "case_id": case["case_id"],
            "annotation_token": case["annotation_token"], "sample_token": case["sample_token"],
            "split": case["split"], "source_category": case["source_category"], "route": args.class_key,
            "seed": seed, "seeds": [{"point": [seed[key] for key in ("x", "y", "z")], "pointIndex": seed["pointIndex"]}],
            "rideState": case["ride_state"], "selection_stratum": stratum,
            "gt_point_count": case["gt_point_count"], "bins": case["bins"],
            "adjacent_annotation_tokens": case["adjacent_annotation_tokens"],
            "source_sha256": source["sha256"], "pcd_sha256": digest(pcd), "lidar_path": source["lidar_path"],
            "point_count": source["point_count"], "baseline_reference": reference})
    fixture = {"schema_version": 1, "class_key": args.class_key, "baseURL": args.base_url.rstrip("/"),
        "created_at": datetime.now(timezone.utc).isoformat(), "manifestHash": manifest_hash,
        "manifestPath": str(Path(args.manifest).resolve()), "fixed_ledger_sha256": ledger_hash,
        "selection_policy": "Before inference: fixed Car ledger first 2 accepted + 1 no-fit by token; other routes dev support/adjacency strata by annotation token",
        "limitations": ["Experimental interaction fixture; box creation does not establish accuracy"] +
            (["Van insufficient-ground-truth: source vehicle.car is a merged Car/Van/SUV pool"] if args.class_key == "vehicles/van" else []),
        "status": "selected", "frames": frames}
    if previous:
        if any(previous[key] != fixture[key] for key in ("class_key", "manifestHash", "frames", "baseURL")):
            raise ValueError("Refusing to change a previously frozen fixture selection")
        fixture = previous
    # Commit the selection locally before the first API call; never change it based on fit results.
    write_json(output, fixture)
    api = API(args.base_url, args.credentials)
    if "taskId" in fixture:
        finish_task(api, fixture, output)
        return
    name = "autofit-multiclass-" + args.class_key.replace("/", "-") + "-" + str(uuid.uuid4())
    created = api.call("POST", "/api/tasks", json={"name": name, "labels": LABELS, "segment_size": 0}).json()
    task_id = created["id"]
    fixture.update(taskId=task_id, taskName=created["name"], status="created",
        taskUrl=f"{api.base}/tasks/{task_id}")
    write_json(output, fixture)
    print(json.dumps({"event": "task-created", "taskId": task_id, "fixture": str(output)}), flush=True)
    path = f"/api/tasks/{task_id}/data"
    api.call("POST", path, headers={"Upload-Start": "true"}, json={})
    fields = {"image_quality": "100", "use_zip_chunks": "true", "sorting_method": "lexicographical"}
    api.call("POST", path, headers={"Upload-Multiple": "true"}, data=fields, files=files, timeout=60)
    finished = api.call("POST", path, headers={"Upload-Finish": "true"}, json={
        "image_quality": 100, "use_zip_chunks": True, "sorting_method": "lexicographical",
    }).json()
    fixture["upload_request_id"] = finished["rq_id"]
    write_json(output, fixture)
    finish_task(api, fixture, output)


def finish_task(api, fixture, output):
    """Resolve a completed upload without creating another task after interruption."""
    task_id = fixture["taskId"]
    actual = api.call("GET", f"/api/tasks/{task_id}").json()
    if actual["name"] != fixture["taskName"] or not actual["name"].startswith("autofit-multiclass-"):
        raise ValueError("Fixture ownership mismatch")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        request = api.call("GET", "/api/requests/" + fixture["upload_request_id"]).json()
        if request["status"] == "failed":
            raise RuntimeError("New task import failed; retained ownership is recorded in fixture")
        if request["status"] == "finished":
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("New task import has not completed; task is retained")
    jobs = api.call("GET", "/api/jobs", params={"task_id": task_id, "page_size": 100}).json()["results"]
    jobs = [job for job in jobs if job["task_id"] == task_id and job["type"] == "annotation"]
    if len(jobs) != 1:
        raise ValueError("Expected one isolated annotation job covering all fixture frames")
    labels = api.call("GET", "/api/labels", params={"task_id": task_id, "page_size": 100}).json()["results"]
    fixture.update(resolve_schema(labels))
    fixture.update(jobId=jobs[0]["id"], jobUrl=f"{api.base}/tasks/{task_id}/jobs/{jobs[0]['id']}", status="ready")
    for frame in fixture["frames"]:
        frame["url"] = fixture["jobUrl"] + f"?frame={frame['frame']}"
    write_json(output, fixture)
    print(json.dumps({"event": "fixture-ready", "taskId": task_id, "jobId": fixture["jobId"],
        "jobUrl": fixture["jobUrl"], "frames": [{"frame": f["frame"], "case": f["case_id"],
        "stratum": f["selection_stratum"]} for f in fixture["frames"]]}), flush=True)


def probe(args):
    fixture = json.loads(Path(args.fixture).read_bytes())
    api = API(args.base_url, args.credentials)
    actual = api.call("GET", f"/api/tasks/{fixture['taskId']}").json()
    if actual["name"] != fixture["taskName"] or not actual["name"].startswith("autofit-multiclass-"):
        raise ValueError("Fixture ownership mismatch")
    job_id = fixture["jobId"]
    probes = []
    preparations = []
    preparation_path = Path(args.output).with_name(Path(args.output).stem + "-preparation.json")

    def prepare_call(method, scope, **kwargs):
        response = api.call(method, f"/api/jobs/{job_id}/auto-fit/prepare", check=False, **kwargs)
        body = response.json()
        preparations.append({"method": method, "scope": scope, "http_status": response.status_code, "response": body})
        write_json(preparation_path, preparations)
        if not response.ok:
            raise RuntimeError(f"Frame preparation HTTP {response.status_code}; recorded {preparation_path}")
        return body

    for frame in fixture["frames"]:
        scope = {"taskId": fixture["taskId"], "jobId": job_id, "frame": frame["frame"]}
        started = time.monotonic()
        prepare = prepare_call("POST", scope, json=scope)
        params = {**scope, "frameToken": prepare["frameToken"]}
        while prepare.get("state") != "ready" and time.monotonic() - started < 120:
            if prepare.get("state") in ("failed", "error"):
                raise RuntimeError("SNAP frame preparation failed")
            time.sleep(0.5)
            prepare = prepare_call("GET", scope, params=params)
        if prepare.get("state") != "ready":
            raise TimeoutError("SNAP preparation did not become ready")
        payload = {**params, "requestId": str(uuid.uuid4()), "requestedAt": time.time() * 1000,
            "objectType": args.object_type, "seed": frame["seed"], "allowEstimated": True,
            "timeout_ms": 1700, "coordinateSystem": "lidar", "units": "metres"}
        if frame["route"].startswith("vehicles/"):
            payload.update(labelId=fixture["vehicleLabelId"], vehicleType=frame["route"].split("/")[1],
                typeAttributeId=fixture["typeAttributeId"])
        elif frame["route"] == "pedestrian":
            payload["labelId"] = fixture["pedestrianLabelId"]
        else:
            if type(frame["rideState"]) is not bool:
                raise ValueError("Two-Wheels probe needs a known manifest ride_state")
            payload.update(labelId=fixture["twoWheelsLabelId"], rideState=frame["rideState"],
                rideStateAttributeId=fixture["rideStateAttributeId"])
        started = time.monotonic()
        response = api.call("POST", f"/api/jobs/{job_id}/auto-fit", json=payload, check=False, timeout=5)
        elapsed = time.monotonic() - started
        result = response.json()
        probes.append({"frame": frame["frame"], "case_id": frame["case_id"], "request": payload,
            "http_status": response.status_code, "http_seconds": elapsed, "response": result})
        write_json(args.output, {"fixture": str(Path(args.fixture).resolve()), "probes": probes,
            "note": "Few fixed cases via real API; no annotation writes, no accuracy claim"})
        print(json.dumps({"frame": frame["frame"], "status": result.get("status"),
            "quality": result.get("quality"), "reason": result.get("reason"), "http_seconds": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--class-key", choices=ROUTES, required=True)
    create_parser.add_argument("--manifest", default=str(EVIDENCE / "nuscenes_manifest.json.gz"))
    create_parser.add_argument("--dataroot", default=str(Path.home() / "Datasets/nuScenes"))
    create_parser.add_argument("--count", type=int, default=4)
    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("--fixture", required=True)
    probe_parser.add_argument("--object-type", choices=("shape", "track"), default="shape")
    for command in (create_parser, probe_parser):
        command.add_argument("--credentials", required=True)
        command.add_argument("--base-url", default="http://localhost:8080")
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    (create if args.command == "create" else probe)(args)


if __name__ == "__main__":
    main()
