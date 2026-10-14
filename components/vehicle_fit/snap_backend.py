"""Pinned SNAP-Outdoor mask inference, with a raw-cloud inverse voxel map.

The core decoder does not require the demo's GUI or CLIP text-label model.
This adapter never initializes random weights, crops a scene, or silently uses
geometry when SNAP cannot run. A service must choose its backend explicitly.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np


SNAP_REVISION = "59fa79b3948e2a9aec4e69efccb1f0677bdb4295"
SNAP_OUTDOOR_DOWNLOAD = (
    "https://drive.google.com/file/d/128PoQvnmupWmWWLWNfzHyyQrv2nUpukq/view"
)
SNAP_OUTDOOR_SHA256 = "266b00df105b494e9c86d416f4715bf615baf2ba6c830c298f02e3dbebedd8bb"


class SNAPUnavailable(RuntimeError):
    """A model, dependency, checkpoint or hardware verification failed."""


def deterministic_voxels(
    cloud: np.ndarray, grid_size: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the first original point in each voxel and map every raw point back.

    Grid coordinates use the upstream Outdoor convention: floor in metres,
    followed by subtraction of the grid minimum. Coordinates and reflectance
    remain those of a real LiDAR return; we do not introduce averaged points.
    """
    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] not in (3, 4) or not len(cloud):
        raise ValueError("cloud must be a nonempty N x 3 or N x 4 array")
    if not np.isfinite(cloud).all() or not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError("cloud and positive grid_size must be finite")
    grid = np.floor(cloud[:, :3] / grid_size).astype(np.int64)
    unique, representative, inverse = np.unique(
        grid, axis=0, return_index=True, return_inverse=True,
    )
    shifted = unique - unique.min(axis=0)
    if shifted.max() >= 2**16:
        raise ValueError("cloud exceeds SNAP serialization grid extent")
    return representative, inverse, shifted.astype(np.int32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SNAPBackend:
    """One-worker, one-frame-cache GPU instance segmentation adapter.

    ``segment(cloud, seed)`` returns ``(raw_boolean_mask, diagnostics)``. The
    optional ``attention='nonflash'`` uses the upstream non-flash implementation
    with the original 1024-point patch sizes; it does not shrink neighborhoods
    to make a failing memory test appear successful.
    """

    def __init__(
        self,
        source: str | Path | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        grid_size: float = 0.05,
        attention: str | None = None,
    ) -> None:
        self.source = Path(source or os.environ.get("SNAP_SOURCE", "/opt/SNAP")).resolve()
        self.checkpoint = Path(
            checkpoint or os.environ.get("SNAP_CHECKPOINT", "/weights/SNAP_Outdoor.pth")
        ).resolve()
        self.expected_sha256 = checkpoint_sha256 or os.environ.get("SNAP_CHECKPOINT_SHA256", SNAP_OUTDOOR_SHA256)
        self.grid_size = grid_size
        self.attention = attention or os.environ.get("SNAP_ATTENTION", "flash")
        self._lock = threading.Lock()
        self._cached_key: str | None = None
        self._cached: tuple[Any, Any, np.ndarray] | None = None
        self.model: Any = None
        self.torch: Any = None
        self.provenance: dict[str, Any] = {"backend": "snap-outdoor"}
        self.load()

    def load(self) -> None:
        if self.attention not in {"flash", "nonflash"}:
            raise SNAPUnavailable("SNAP_ATTENTION must be flash or nonflash")
        if self.grid_size != 0.05:
            raise SNAPUnavailable("SNAP Outdoor checkpoint requires the verified 0.05m grid")
        if not self.source.is_dir() or not self.checkpoint.is_file():
            raise SNAPUnavailable("pinned SNAP source or Outdoor checkpoint is missing")
        try:
            revision = subprocess.check_output(
                ["git", "-c", f"safe.directory={self.source}", "-C", str(self.source), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SNAPUnavailable("SNAP source revision cannot be verified") from exc
        if revision != SNAP_REVISION:
            raise SNAPUnavailable(f"unverified SNAP source revision: {revision}")
        checksum = sha256_file(self.checkpoint)
        if not self.expected_sha256 or checksum != self.expected_sha256:
            raise SNAPUnavailable("SNAP_CHECKPOINT_SHA256 must match the official Outdoor checkpoint")
        try:
            import torch

            if not torch.cuda.is_available():
                raise SNAPUnavailable("CUDA is unavailable; CPU inference is not an accepted fallback")
            sys.path.insert(0, str(self.source))
            snap_module = importlib.import_module("src.snap")
            if Path(snap_module.__file__).resolve() != self.source / "src" / "snap.py":
                raise SNAPUnavailable("src.snap resolved to a different source tree")
            original_backbone = snap_module.PointTransformerV3

            def configured_backbone(*args: Any, **kwargs: Any) -> Any:
                kwargs["enable_flash"] = self.attention == "flash"
                # The trained parameters do not depend on serialization order.
                # A fixed order makes replay and cached/uncached masks reproducible.
                kwargs["shuffle_orders"] = False
                return original_backbone(*args, **kwargs)

            # SNAP does not expose enable_flash in its constructor. Override only
            # during construction, then restore the imported module unchanged.
            snap_module.PointTransformerV3 = configured_backbone
            try:
                model = snap_module.SNAP(
                    num_points=1, num_merge_blocks=1, use_pdnorm=True,
                    return_mid_points=True,
                )
            finally:
                snap_module.PointTransformerV3 = original_backbone
            # The official checkpoint contains training metadata. Loading pickle
            # is restricted to the explicitly verified hash supplied above.
            saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(saved, dict) or "model" not in saved:
                raise SNAPUnavailable("Outdoor checkpoint has no model state_dict")
            incompatible = model.load_state_dict(saved["model"], strict=False)
            # The official Outdoor artifact has 30 training-era text/confidence
            # parameters absent from this pinned upstream inference model. The
            # upstream demo ignores them with strict=False. Permit only these
            # known inactive heads; missing active parameters always block.
            inactive = [
                key for key in incompatible.unexpected_keys
                if key.startswith(("merge_text.", "confidence_head."))
                or key in {"iou_token_queries.weight", "bg_query_feat.weight"}
            ]
            if incompatible.missing_keys or set(inactive) != set(incompatible.unexpected_keys):
                raise SNAPUnavailable(
                    f"checkpoint mismatch: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            self.model = model.eval().cuda()
            self.torch = torch
            self.provenance.update({
                "modelRevision": revision,
                "checkpointSHA256": checksum,
                "checkpointSource": SNAP_OUTDOOR_DOWNLOAD,
                "missingWeights": [], "unexpectedWeights": inactive,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "computeCapability": list(torch.cuda.get_device_capability()),
                "attention": self.attention,
                "gridSizeMetres": self.grid_size,
                "serializationShuffle": False,
            })
        except SNAPUnavailable:
            raise
        except Exception as exc:
            raise SNAPUnavailable(f"SNAP initialization failed: {type(exc).__name__}: {exc}") from exc

    def segment(
        self, cloud: np.ndarray, seed: np.ndarray, *, frame_key: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cloud = np.ascontiguousarray(cloud, dtype=np.float32)
        seed = np.asarray(seed, dtype=np.float32)
        if seed.shape != (3,) or not np.isfinite(seed).all():
            raise ValueError("seed must contain three finite LiDAR coordinates")
        nearest_distance = float(np.linalg.norm(cloud[:, :3] - seed, axis=1).min())
        if nearest_distance > 0.075:
            raise ValueError("seed is not supported by an original LiDAR return")
        frame_key = frame_key or hashlib.sha256(cloud.tobytes()).hexdigest()
        with self._lock:
            torch = self.torch
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            cache_hit = self._cached_key == frame_key and self._cached is not None
            backbone_ms = 0.0
            with torch.inference_mode():
                if cache_hit:
                    data, features, inverse = self._cached
                else:
                    # Free the previous frame before allocating a new scene.
                    self._cached = None
                    self._cached_key = None
                    representative, inverse, grid = deterministic_voxels(cloud, self.grid_size)
                    sampled = cloud[representative]
                    strength = sampled[:, 3:4] if cloud.shape[1] == 4 else np.zeros((len(sampled), 1), np.float32)
                    data = {
                        "coord": torch.as_tensor(sampled[:, :3], device="cuda"),
                        "grid_coord": torch.as_tensor(grid, device="cuda"),
                        "strength": torch.as_tensor(strength, device="cuda"),
                        "offset": torch.tensor([len(sampled)], device="cuda", dtype=torch.long),
                        "condition": "SemanticKITTI", "domain": "Outdoor",
                    }
                    backbone_started = time.perf_counter()
                    data = self.model.run_input_encoder(data)
                    features, _ = self.model.run_backbone(data)
                    torch.cuda.synchronize()
                    backbone_ms = (time.perf_counter() - backbone_started) * 1000
                    self._cached = (data, features, inverse)
                    self._cached_key = frame_key
                decoder_data = dict(data)
                decoder_data["point"] = torch.as_tensor(seed.reshape(1, 1, 3), device="cuda")
                decoder_data["point_offset"] = [1]
                decoder_started = time.perf_counter()
                result = self.model.run_mask_decoder(features, decoder_data)
                logits = result[0][0].reshape(-1)
                if logits.numel() != len(data["coord"]) or not torch.isfinite(logits).all():
                    raise SNAPUnavailable("decoder returned a malformed or nonfinite mask")
                voxel_mask = (logits.sigmoid() > 0.5).cpu().numpy()
                raw_mask = np.asarray(voxel_mask[inverse], dtype=bool)
                score = float(result[2][0].reshape(-1)[0].cpu())
                torch.cuda.synchronize()
                decoder_ms = (time.perf_counter() - decoder_started) * 1000
            diagnostics = dict(self.provenance)
            diagnostics.update({
                "rawPoints": len(cloud), "voxelPoints": len(data["coord"]),
                "maskPoints": int(raw_mask.sum()), "cacheHit": cache_hit,
                "maskPredictedIoU": score,
                "stageMilliseconds": {"backbone": backbone_ms, "decoder": decoder_ms},
                "latencyMilliseconds": (time.perf_counter() - started) * 1000,
                "peakVRAMAllocatedBytes": torch.cuda.max_memory_allocated(),
                "peakVRAMReservedBytes": torch.cuda.max_memory_reserved(),
            })
            return raw_mask, diagnostics

    def clear_cache(self) -> None:
        """Release failed frame allocations only after a CUDA memory error."""
        with self._lock:
            self._cached = None
            self._cached_key = None
            self.torch.cuda.empty_cache()
