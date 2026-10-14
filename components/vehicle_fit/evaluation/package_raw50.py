"""Package the frozen RAW 104-153 cloud frames into a small CVAT task ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(value):
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def package(source_zip, frozen_manifest, output_zip, report_path, raw_frame_start=104,
            frame_count=50, include_context_images=False):
    source_zip, frozen_manifest = Path(source_zip), Path(frozen_manifest)
    output_zip, report_path = Path(output_zip), Path(report_path)
    manifest = json.loads(frozen_manifest.read_text())
    frames = sorted(manifest["frames"])
    requested = [f"{index:010d}" for index in range(raw_frame_start, raw_frame_start + frame_count)]
    if any(frame not in manifest["frames"] for frame in requested):
        raise ValueError("Requested RAW frame range is outside the frozen drive_0005 manifest")
    records = []
    with zipfile.ZipFile(source_zip) as source, zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
    ) as output:
        for task_frame, raw_frame in enumerate(requested):
            raw_index = int(raw_frame)
            cloud_member = f"pointcloud/{raw_frame}.pcd"
            pcd = source.read(cloud_member)
            marker = b"DATA binary\n"
            split = pcd.find(marker)
            if split < 0 or pcd[split + len(marker):] == b"":
                raise ValueError(f"Invalid binary PCD payload in frame {raw_frame}")
            raw_cloud_hash = manifest["frames"][raw_frame]["sha256"]["velodyne_points"]
            if sha256(pcd[split + len(marker):]) != raw_cloud_hash:
                raise ValueError(f"PCD does not preserve the frozen KITTI .bin bytes for {raw_frame}")
            expected_image_hash = manifest["frames"][raw_frame]["sha256"]["image_02"]
            cloud_output = f"pointcloud/{task_frame:010d}.pcd"
            members = [(cloud_output, pcd)]
            camera_hashes = {}
            for camera in range(4):
                image_name = f"image_{camera:02d}"
                image_member = f"related_images/{raw_frame}_pcd/{image_name}.png"
                image = source.read(image_member)
                image_hash = sha256(image)
                if camera == 2 and image_hash != expected_image_hash:
                    raise ValueError(f"Image_02 does not match the frozen KITTI frame {raw_frame}")
                output_member = f"related_images/{task_frame:010d}_pcd/{image_name}.png"
                if include_context_images:
                    members.append((output_member, image))
                camera_hashes[image_name] = image_hash
            for name, contents in members:
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                output.writestr(entry, contents)
            records.append({"task_frame": task_frame, "raw_frame": raw_index,
                            "source_cloud_sha256": raw_cloud_hash,
                            "camera_sha256": camera_hashes,
                            "cloud_member": cloud_output,
                            "image_members": [name for name, _ in members[1:]]})
    output_sha = hashlib.sha256(output_zip.read_bytes()).hexdigest()
    report = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_frame_start": raw_frame_start,
        "raw_frame_stop_exclusive": raw_frame_start + frame_count,
        "source_zip": str(source_zip.resolve()),
        "source_zip_sha256": hashlib.sha256(source_zip.read_bytes()).hexdigest(),
        "frozen_manifest_sha256": hashlib.sha256(frozen_manifest.read_bytes()).hexdigest(),
        "task_zip": str(output_zip.resolve()), "task_zip_sha256": output_sha,
        "task_zip_bytes": output_zip.stat().st_size,
        "context_images_included": include_context_images,
        "frames": records,
        "validation": "Every CVAT PCD binary payload equals its original .bin; all four related camera images are byte-hashed and image_02 matches the frozen RAW manifest.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raw-frame-start", type=int, default=104)
    parser.add_argument("--frame-count", type=int, default=50)
    parser.add_argument("--include-context-images", action="store_true")
    args = parser.parse_args()
    result = package(args.source_zip, args.manifest, args.output_zip, args.report,
                     args.raw_frame_start, args.frame_count, args.include_context_images)
    print(json.dumps({"task_zip_sha256": result["task_zip_sha256"],
                      "task_zip_bytes": result["task_zip_bytes"],
                      "frames": len(result["frames"])}, indent=2))


if __name__ == "__main__":
    main()
