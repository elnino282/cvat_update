"""python -m components.vehicle_fit.evaluation.cli freeze|run (focused only)."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

from .manifest import freeze_object, freeze_raw, sha256, write_frozen
from .metrics import calibration, errors, kitti_labels, label_cuboid, oriented_iou, result_cuboid
from .overlay import overlays


def request_fit(endpoint, frame, seed, cloud_path, timeout, request_id):
    metadata = {"requestId": request_id, "frame": int(frame),
                "seed": dict(zip(("x", "y", "z"), seed["point"]), pointIndex=seed["pointIndex"]),
                "labelName": "Car", "cloud_format": "bin", "userId": "evaluation",
                "jobId": "kitti-object", "taskId": "frozen-evaluation", "timeout_ms": 8000,
                "coordinateSystem": "lidar", "units": "metres"}
    boundary = "----vehicle-fit-" + uuid.uuid4().hex
    body = bytearray()
    def field(name, content, filename=None):
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"".encode())
        if filename:
            body.extend(f"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream".encode())
        body.extend(b"\r\n\r\n" + content + b"\r\n")
    field("metadata", json.dumps(metadata).encode())
    field("cloud_format", b"bin")
    field("cloud", Path(cloud_path).read_bytes(), Path(cloud_path).name)
    body.extend(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/fit", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        return result, time.perf_counter() - start, None
    except urllib.error.HTTPError as error:
        payload = error.read().decode(errors="replace")
        return None, time.perf_counter() - start, f"HTTP {error.code}: {payload}"
    except Exception as error:
        return None, time.perf_counter() - start, f"{type(error).__name__}: {error}"


class VramSampler:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            sample = {"utc": datetime.now(timezone.utc).isoformat()}
            for kind, query in (("device", "--query-gpu=uuid,memory.used,memory.total,utilization.gpu"),
                                ("process", "--query-compute-apps=pid,process_name,used_gpu_memory")):
                try:
                    proc = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True, timeout=2, check=False)
                    sample[kind] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
                except Exception as error:
                    sample[kind] = str(error)
            self.samples.append(sample)
            self.stop.wait(.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def percentiles(values):
    if not values:
        return {"count": 0, "p50": None, "p95": None, "mean": None}
    return {"count": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "mean": float(np.mean(values))}


def vram_summary(samples, records):
    device_peaks, process_peaks = {}, {}
    for sample in samples:
        for fields in csv.reader(sample.get("device", "").splitlines()):
            try:
                device_peaks[fields[0]] = max(device_peaks.get(fields[0], 0), float(fields[1]))
            except (ValueError, IndexError):
                pass
        for fields in csv.reader(sample.get("process", "").splitlines()):
            try:
                key = f"{fields[0].strip()}:{fields[1].strip()}"
                process_peaks[key] = max(process_peaks.get(key, 0), float(fields[2]))
            except (ValueError, IndexError):
                pass
    diagnostics = [a.get("response", {}).get("diagnostics", {}) for r in records for a in r["attempts"] if a.get("response")]
    service_counters = {}
    for item in diagnostics:
        for key, value in item.items():
            if any(token in key.lower() for token in ("vram", "allocated", "reserved")) and isinstance(value, (int, float)):
                service_counters[key] = max(service_counters.get(key, 0), value)
    return {"device_peak_used_mib": device_peaks, "process_peak_used_mib": process_peaks,
            "service_peak_counters_original_units": service_counters,
            "sampling_interval_seconds": .2,
            "limitation": "Device includes all GPU clients; process numbers identify observed PIDs. Sampling can miss sub-200ms peaks; service counters are authoritative for allocator peaks."}


def write_report(output, summary, identity, records):
    from PIL import Image, ImageDraw
    all_cases, adequate = summary["all"], summary["adequate"]
    latency = summary["http_latency_seconds"]["all"]
    text = [f"# KITTI Object {identity['split']} evaluation: {identity['expected_backend']}", "",
            f"Frozen manifest SHA-256: `{identity['manifest_sha256']}`. Started {identity['started_utc']}.", "",
            "| Cohort | Cars | Accepted | IoU >=0.70 and correct vehicle | No fit | Unseedable | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, cohort in (("All", all_cases), ("Adequate", adequate)):
        text.append(f"| {name} | {cohort['cases']} | {cohort['accepted']} | {cohort['success_iou_ge_0_70']} | {cohort['no_fit']} | {cohort['unseedable']} | {(cohort['coverage'] or 0):.1%} |")
    def fmt(value):
        return f"{value:.3f}" if value is not None else "not measured"
    text.extend(["", f"Accepted-box mean IoU: {fmt(all_cases['iou_accepted']['mean'])}; adequate mean: {fmt(adequate['iou_accepted']['mean'])}. "
                 f"Including zero for no-fit/unseedable, all-case mean IoU: {fmt(all_cases['iou_all_cases_zero_for_no_fit']['mean'])}.", "",
                 f"HTTP p50/p95: {fmt(latency['p50'])}/{fmt(latency['p95'])} s across {latency['count']} attempts, including rejected/error attempts. "
                 "UI latency is not measured by this service replay.", "",
                 f"Correct vehicle among accepted: {all_cases['correct_vehicle']}/{all_cases['accepted']}; "
                 f"retry: {all_cases['retry_cases']}/{all_cases['primary_attempted']} primary-attempted Cars.", "",
                 "Targets are IoU >=0.70 on adequate observations and p95 <=5 seconds. Coverage and all-case success rates remain explicit; "
                 "functional execution does not establish quality PASS.", "",
                 "Each case and attempt, including service rejection details, is recorded in [cases.jsonl](cases.jsonl). "
                 "Full quantitative errors, latency groups, service timings and VRAM evidence are in [summary.json](summary.json) and [vram-samples.json](vram-samples.json).", "",
                 "## Outcomes", "", "| Outcome | Cars |", "|---|---:|"])
    for cause, count in summary["fail_causes"].items():
        text.append(f"| {cause} | {count} |")
    text.extend(["", "## Representative evidence", ""])
    for cause in sorted(summary["fail_causes"]):
        case = next(r for r in records if r["outcome"] == cause)
        if case["attempts"]:
            stem = f"{case['case_id']}-attempt{len(case['attempts']) - 1}"
            text.append(f"{cause}: [camera/mask/box](overlays/{stem}-camera.png), [LiDAR/mask/box](overlays/{stem}-cloud.png).")
    text.extend(["", "GT-assisted observed seeds are disclosed in the manifest; no GT geometry or mask is transmitted to inference. "
                 "Accepted poor fits are never retried based on evaluator IoU. No manual box changes enter these results.", ""])
    (output / "report.md").write_text("\n".join(text))
    chart = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((25, 20), f"{identity['expected_backend']} / frozen {identity['split']} / {all_cases['cases']} Cars", fill="black")
    draw.text((25, 45), "All accepted + no-fit/unseedable, no repaired annotations", fill="black")
    counts, max_count = summary["fail_causes"], max(summary["fail_causes"].values(), default=1)
    for i, (cause, count) in enumerate(counts.items()):
        y = 90 + i * 45
        draw.text((25, y + 4), cause, fill="black")
        draw.rectangle((190, y, 190 + 700 * count / max_count, y + 25), fill="#2da44e" if cause == "PASS" else "#cf222e")
        draw.text((905, y + 4), str(count), fill="black")
    draw.text((25, 555), f"Adequate: {adequate['success_iou_ge_0_70']}/{adequate['cases']} achieve >=0.70 IoU; all: {all_cases['success_iou_ge_0_70']}/{all_cases['cases']}", fill="black")
    chart.save(output / "outcomes.png")


def summarize(records):
    summary = {}
    for name, cohort in (("all", records), ("adequate", [r for r in records if r["adequate"]])):
        accepted = [r for r in cohort if r["accepted"]]
        submitted = [r for r in cohort if r["attempts"]]
        correct = [r for r in accepted if r["correct_vehicle"]]
        successes = [r for r in accepted if r["correct_vehicle"] and r["metrics"]["iou_3d"] >= .7]
        summary[name] = {"cases": len(cohort), "primary_attempted": len(submitted),
                         "accepted": len(accepted), "unseedable": sum(not r["attempts"] for r in cohort),
                         "no_fit": sum(r["outcome"] == "NO_FIT" for r in cohort),
                         "transport_error": sum(r["outcome"] == "TRANSPORT_ERROR" for r in cohort),
                         "invalid_geometry": sum(r["outcome"] == "INVALID_GEOMETRY" for r in cohort),
                         "correct_vehicle": len(correct), "wrong_vehicle": len(accepted) - len(correct),
                         "coverage": len(accepted) / len(cohort) if cohort else None,
                         "correct_vehicle_rate_of_accepted": len(correct) / len(accepted) if accepted else None,
                         "success_iou_ge_0_70": len(successes),
                         "success_rate_all_cases": len(successes) / len(cohort) if cohort else None,
                         "retry_cases": sum(len(r["attempts"]) >= 2 for r in cohort),
                         "retry_rate": sum(len(r["attempts"]) >= 2 for r in cohort) / len(submitted) if submitted else None,
                         "iou_accepted": percentiles([r["metrics"]["iou_3d"] for r in accepted]),
                         "iou_all_cases_zero_for_no_fit": percentiles([r["metrics"]["iou_3d"] if r["accepted"] else 0 for r in cohort]),
                         "center_error_m": percentiles([r["metrics"]["center_error_m"] for r in accepted]),
                         "yaw_error_deg_mod_pi": percentiles([np.degrees(r["metrics"]["yaw_error_rad_mod_pi"]) for r in accepted]),
                         "dimension_error_m_lwh": [percentiles([r["metrics"]["dimension_error_m_lwh"][i] for r in accepted]) for i in range(3)],
                         "dimension_relative_error_lwh": [percentiles([r["metrics"]["dimension_relative_error_lwh"][i] for r in accepted]) for i in range(3)]}
    attempts = [attempt for record in records for attempt in record["attempts"]]
    summary["http_latency_seconds"] = {tag: percentiles([a["http_latency_s"] for a in attempts
                                                         if tag == "all" or a["cache_group"] == tag])
                                        for tag in ("all", "first_observed_request", "cache_miss", "warm_frame")}
    stage_values = {}
    for attempt in attempts:
        diagnostics = (attempt.get("response") or {}).get("diagnostics", {})
        timings = dict(diagnostics.get("stageMilliseconds", diagnostics.get("timings_ms", {})))
        if isinstance(diagnostics.get("latencyMilliseconds"), (int, float)):
            timings["inference_and_geometry"] = diagnostics["latencyMilliseconds"]
        for name, value in timings.items():
            if isinstance(value, (int, float)):
                stage_values.setdefault(name, []).append(value)
    summary["service_stage_timings_ms"] = {name: percentiles(values) for name, values in stage_values.items()}
    summary["fail_causes"] = {cause: sum(r["outcome"] == cause for r in records) for cause in sorted({r["outcome"] for r in records})}
    summary["gate"] = {"functional": "MEASURED; review case ledger and runtime evidence separately",
                       "quality_target": "REPORT_ONLY; IoU>=0.70 on adequate observations, failures retained",
                       "http_p95_le_5s": summary["http_latency_seconds"]["all"]["p95"] is not None and summary["http_latency_seconds"]["all"]["p95"] <= 5,
                       "ui_p95_le_5s": "NOT_MEASURED_BY_SERVICE_REPLAY"}
    return summary


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = [case for case in manifest["cases"] if case["split"] == args.split]
    if args.case_id:
        selected = [case for case in selected if case["case_id"] == args.case_id]
        if len(selected) != 1:
            raise ValueError(f"The frozen manifest has no unique {args.split} case {args.case_id}")
    if args.limit:
        selected = selected[:args.limit]
    used_frames = {case["frame"] for case in selected}
    for frame in used_frames:
        info = manifest["frames"][frame]
        for kind, path in info["assets"].items():
            if sha256(path) != info["sha256"][kind]:
                raise ValueError(f"Frozen asset changed: {frame}/{kind}")
    run_identity = {"started_utc": datetime.now(timezone.utc).isoformat(),
                    "manifest_sha256": sha256(args.manifest), "split": args.split,
                    "endpoint": args.endpoint, "expected_backend": args.backend,
                    "limit": args.limit, "timeout_seconds": args.timeout,
                    "retry_policy": "retry only no-fit, invalid geometry, transport failure; max two frozen retries",
                    "calibration": "T=R0_rect@Tr_velo_to_cam; camera y-=h/2; exact transformed corners; P2@T once",
                    "coordinate_units": "metres, LiDAR, geometric centre, length/width/height, EulerXYZ",
                    "seed_disclosure": "Labels assist seed choice only; inference requests contain full cloud, observed point and frame identity"}
    try:
        with urllib.request.urlopen(args.endpoint.rstrip("/") + "/health", timeout=5) as response:
            run_identity["health"] = json.load(response)
    except Exception as error:
        run_identity["health_error"] = str(error)
    (output / "run.json").write_text(json.dumps(run_identity, indent=2) + "\n")
    records, last_frame, request_number = [], None, 0
    with VramSampler() as sampler, (output / "cases.jsonl").open("w") as case_log:
        for case in selected:
            frame = case["frame"]
            assets = manifest["frames"][frame]["assets"]
            cloud = np.fromfile(assets["velodyne"], dtype="<f4").reshape(-1, 4)
            transform, p2 = calibration(assets["calib"])
            gt = label_cuboid(case["label"], transform)
            all_labels = kitti_labels(assets["label_2"])
            all_gt = [label_cuboid(label, transform) for label in all_labels]
            record = {"case_id": case["case_id"], "frame": frame, "adequate": case["adequate"],
                      "gt_point_count": case["gt_point_count"], "attempts": [], "accepted": False,
                      "outcome": "UNSEEDABLE", "correct_vehicle": False}
            for attempt_index, seed in enumerate(case["seeds"][:3]):
                cache_group = "first_observed_request" if request_number == 0 else ("warm_frame" if last_frame == frame else "cache_miss")
                request_number += 1
                last_frame = frame
                request_id = str(uuid.uuid4())
                response, latency, transport_error = request_fit(args.endpoint, frame, seed, assets["velodyne"], args.timeout, request_id)
                if request_number > 1 and response and "cacheHit" in response.get("diagnostics", {}):
                    cache_group = "warm_frame" if response["diagnostics"]["cacheHit"] else "cache_miss"
                attempt = {"attempt_index": attempt_index, "seed": seed, "request_id": request_id,
                           "http_latency_s": latency, "cache_group": cache_group,
                           "response": response, "transport_error": transport_error}
                record["attempts"].append(attempt)
                prediction = None
                overlay_prediction = None
                if transport_error:
                    record["outcome"] = "TRANSPORT_ERROR"
                elif response.get("status") == "no-fit":
                    record["outcome"] = "NO_FIT"
                    record["reason"] = response.get("reason")
                    diagnostics = response.get("diagnostics", {})
                    proposed = diagnostics.get("proposedCuboid")
                    if proposed:
                        try:
                            overlay_prediction = result_cuboid(proposed)
                            attempt["rejectedCandidateMetrics"] = errors(overlay_prediction, gt)
                        except (KeyError, TypeError, ValueError) as error:
                            attempt["rejectedCandidateGeometryError"] = str(error)
                else:
                    backend = response.get("provider") or response.get("backend") or response.get("diagnostics", {}).get("backend")
                    if backend != args.backend:
                        raise ValueError(f"Backend mismatch in {case['case_id']}: {backend} != {args.backend}")
                    try:
                        prediction = result_cuboid(response)
                    except (KeyError, TypeError, ValueError) as error:
                        record["outcome"] = "NO_FIT" if not (response.get("cuboid") or response.get("geometry") or response.get("center")) else "INVALID_GEOMETRY"
                        attempt["geometry_error"] = str(error)
                    if prediction is not None:
                        overlay_prediction = prediction
                        record["accepted"] = True
                        record["metrics"] = errors(prediction, gt)
                        scores = [oriented_iou(prediction, other) for other in all_gt]
                        best = int(np.argmax(scores))
                        correct = scores[best] > 0 and all_labels[best]["line_index"] == case["label"]["line_index"]
                        record["matched_label_line_index"] = all_labels[best]["line_index"] if scores[best] > 0 else None
                        record["correct_vehicle"] = correct
                        record["outcome"] = "PASS" if correct and record["metrics"]["iou_3d"] >= .7 else ("WRONG_VEHICLE" if not correct else "LOW_IOU")
                diagnostics = (response or {}).get("diagnostics", {})
                mask = diagnostics.get("mask_indices", diagnostics.get("maskIndices", []))
                if mask:
                    indices = np.asarray(mask, dtype=int)
                    indices = indices[(indices >= 0) & (indices < len(cloud))]
                    attempt["mask_gt_purity"] = float(gt.contains(cloud[indices]).mean()) if len(indices) else None
                    attempt["mask_gt_coverage"] = float(gt.contains(cloud[indices]).sum() / case["gt_point_count"]) if case["gt_point_count"] else None
                overlays(output / "overlays" / f"{case['case_id']}-attempt{attempt_index}", assets["image_2"], cloud,
                         gt, overlay_prediction, mask, seed["point"], transform, p2,
                         f"{case['case_id']} / {args.backend} / {record['outcome']}" + (f" IoU={record['metrics']['iou_3d']:.3f}" if prediction else ""))
                if record["accepted"]:
                    break
            records.append(record)
            case_log.write(json.dumps(record) + "\n")
            case_log.flush()
            print(f"{len(records)}/{len(selected)} {case['case_id']} {record['outcome']}", flush=True)
    (output / "vram-samples.json").write_text(json.dumps(sampler.samples, indent=2) + "\n")
    summary = summarize(records)
    summary["vram"] = vram_summary(sampler.samples, records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(output, summary, run_identity, records)
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "frame", "adequate", "gt_point_count", "accepted", "outcome", "correct_vehicle", "attempt_count", "iou_3d", "center_error_m", "yaw_error_rad_mod_pi"])
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in ("case_id", "frame", "adequate", "gt_point_count", "accepted", "outcome", "correct_vehicle")}
            row["attempt_count"] = len(record["attempts"])
            for key in ("iou_3d", "center_error_m", "yaw_error_rad_mod_pi"):
                row[key] = record.get("metrics", {}).get(key)
            writer.writerow(row)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--object-root", default="/home/quynhhn/Downloads/100_training/training")
    freeze.add_argument("--output", default="docs/auto-fit-pr4-evidence/object-manifest.json")
    freeze.add_argument("--raw-zip")
    freeze.add_argument("--tracklet-zip")
    freeze.add_argument("--raw-output", default="docs/auto-fit-pr4-evidence/raw50-manifest.json")
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--manifest", default="docs/auto-fit-pr4-evidence/object-manifest.json")
    evaluate.add_argument("--endpoint", required=True)
    evaluate.add_argument("--backend", choices=("snap", "geometry-only"), required=True)
    evaluate.add_argument("--split", choices=("tune", "eval"), required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--timeout", type=float, default=12)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--case-id", help="Run one exact frozen case for a focused service smoke test")
    args = parser.parse_args()
    if args.command == "freeze":
        manifest = freeze_object(args.object_root)
        digest = write_frozen(manifest, args.output)
        print(json.dumps({"sha256": digest, "tune_cars": sum(c["split"] == "tune" for c in manifest["cases"]),
                          "eval_cars": sum(c["split"] == "eval" for c in manifest["cases"]),
                          "eval_adequate": sum(c["split"] == "eval" and c["adequate"] for c in manifest["cases"])}, indent=2))
        if args.raw_zip and args.tracklet_zip:
            raw = freeze_raw(args.raw_zip, args.tracklet_zip)
            print(json.dumps({"raw_sha256": write_frozen(raw, args.raw_output), "raw_frames": len(raw["frames"]),
                              "raw_cars": len(raw["cases"]), "raw_tracks": len({c["track_id"] for c in raw["cases"]})}, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
