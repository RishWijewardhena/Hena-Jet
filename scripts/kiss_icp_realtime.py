#!/usr/bin/env python3
"""Reliable close-range ZED Mini + KISS-ICP mapping.

The camera streams continuously. Every frame is:
    1. Grabbed from the ZED SDK
    2. Confidence, range, boundary, and depth-edge filtered
    3. Quality-gated and deterministically sampled for KISS-ICP
    4. Fused into a frame-weighted global voxel map
    5. Displayed live and summarized in a JSON quality report

Usage
-----
python scripts/kiss_icp_realtime.py \
    --out outputs/kiss_icp_map.ply
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyzed.sl as sl

from kiss_icp_quality import (
    FrameQuality,
    GlobalVoxelAccumulator,
    deterministic_voxel_sample,
    evaluate_frame_quality,
    evaluate_pose_quality,
    filter_organized_cloud,
    pose_step,
)

# ── KISS-ICP import ────────────────────────────────────────────────────────────
try:
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
except ImportError as exc:
    raise SystemExit(
        "KISS-ICP is required. Install it with:\n"
        "  pip install kiss-icp"
    ) from exc

# ── Open3D import ──────────────────────────────────────────────────────────────
try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit(
        "Open3D is required. Install it with:\n"
        "  pip install open3d"
    ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time ZED + KISS-ICP mapping with live viewer."
    )
    parser.add_argument("--camera-min-depth-m", type=float, default=0.10,
                        help="ZED SDK near depth limit (default: 0.10)")
    parser.add_argument("--camera-max-depth-m", type=float, default=0.25,
                        help="ZED SDK far depth limit (default: 0.25)")
    parser.add_argument("--min-depth-m", type=float, default=0.11,
                        help="Reliable software near cutoff (default: 0.11)")
    parser.add_argument("--max-depth-m", type=float, default=0.22,
                        help="Reliable software far cutoff (default: 0.22)")
    parser.add_argument(
        "--resolution",
        choices=["HD2K", "HD1080", "HD720", "VGA"],
        default="HD720",
    )
    available_depth_modes = [
        name
        for name in (
            "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL",
            "NEURAL_LIGHT", "NEURAL_PLUS",
        )
        if hasattr(sl.DEPTH_MODE, name)
    ]
    parser.add_argument(
        "--depth-mode",
        choices=available_depth_modes,
        default="NEURAL_PLUS" if "NEURAL_PLUS" in available_depth_modes else "NEURAL",
        help="ZED depth mode (default: NEURAL_PLUS when available)",
    )
    parser.add_argument(
        "--coordinate-system",
        choices=["IMAGE", "RIGHT_HANDED_Z_UP_X_FWD"],
        default="RIGHT_HANDED_Z_UP_X_FWD",
    )
    parser.add_argument("--voxel-m", type=float, default=0.002,
                        help="Map voxel size in metres (default: 0.002)")
    parser.add_argument("--max-map-points", type=int, default=5_000_000,
                        help="Safety cap on accumulated map points")
    parser.add_argument("--max-points-per-frame", type=int, default=80_000,
                        help="Subsample each frame to this many points before KISS-ICP. "
                             "Reduces CPU load and prevents map compaction every few frames. "
                             "KISS-ICP voxelises internally so extra points don't help. "
                             "Default 80000 is good for HD720; use 50000 for HD1080.")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Frames to skip before mapping starts")
    parser.add_argument("--out", type=Path, default=Path("outputs/kiss_icp_map.ply"),
                        help="Output PLY path when you press Q to quit")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable Open3D live viewer (headless mode)")
    parser.add_argument("--min-map-observations", type=int, default=2,
                        help="Remove final voxels seen by fewer frames (default: 2)")
    parser.add_argument("--confidence-threshold", type=int, default=60,
                        help="Reject ZED confidence errors above this value (default: 60)")
    parser.add_argument("--texture-confidence-threshold", type=int, default=100,
                        help="Preserve low-texture skin by default (default: 100)")
    parser.add_argument("--edge-threshold-m", type=float, default=0.008,
                        help="Reject depth jumps larger than this (default: 0.008)")
    parser.add_argument("--no-erode-invalid-boundary", action="store_true",
                        help="Keep pixels directly adjacent to invalid depth")
    parser.add_argument("--min-reliable-points", type=int, default=2_000,
                        help="Minimum filtered points required for ICP (default: 2000)")
    parser.add_argument("--min-valid-coverage", type=float, default=0.05,
                        help="Minimum filtered image coverage (default: 0.05)")
    parser.add_argument("--max-jump-m", type=float, default=0.02,
                        help="Discard frames where pose jumps more than this many metres "
                             "in one step — guards against tracking loss after dropouts. "
                             "Default 0.02m. Increase only for verified fast motion.")
    parser.add_argument("--max-jump-deg", type=float, default=5.0,
                        help="Discard frames where pose jumps more than this many degrees "
                             "in one step. Default 5.0 deg.")
    parser.add_argument("--quality-report", type=Path, default=None,
                        help="JSON report path (default: <out-stem>_quality.json)")
    # KISS-ICP tuning
    parser.add_argument("--kiss-voxel-m", type=float, default=0.003,
                        help="KISS-ICP internal voxel size (default: 0.003)")
    parser.add_argument("--kiss-max-range-m", type=float, default=0.25,
                        help="KISS-ICP radial maximum range (default: 0.25)")
    parser.add_argument("--kiss-min-range-m", type=float, default=0.10,
                        help="KISS-ICP radial minimum range (default: 0.10)")
    parser.add_argument("--kiss-min-motion-m", type=float, default=0.005,
                        help="Motion floor for adaptive-threshold updates (default: 0.005)")
    parser.add_argument("--kiss-initial-threshold-m", type=float, default=0.02,
                        help="Initial correspondence distance (default: 0.02)")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.camera_min_depth_m < args.camera_max_depth_m:
        raise ValueError("camera depth limits must be positive and increasing")
    if not args.camera_min_depth_m <= args.min_depth_m < args.max_depth_m <= args.camera_max_depth_m:
        raise ValueError("software depth limits must be inside the camera depth limits")
    if args.voxel_m <= 0 or args.kiss_voxel_m <= 0:
        raise ValueError("voxel sizes must be positive")
    if args.max_points_per_frame <= 0 or args.min_reliable_points <= 0:
        raise ValueError("point-count limits must be positive")
    if args.max_map_points <= 0 or args.min_map_observations <= 0:
        raise ValueError("map limits must be positive")
    if not 0 <= args.confidence_threshold <= 100:
        raise ValueError("--confidence-threshold must be in [0, 100]")
    if not 0 <= args.texture_confidence_threshold <= 100:
        raise ValueError("--texture-confidence-threshold must be in [0, 100]")
    if args.edge_threshold_m < 0:
        raise ValueError("--edge-threshold-m must be non-negative")
    if not 0 <= args.min_valid_coverage <= 1:
        raise ValueError("--min-valid-coverage must be in [0, 1]")


# ──────────────────────────────────────────────────────────────────────────────
# ZED helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolution_enum(name: str) -> sl.RESOLUTION:
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }[name]


def coordinate_system_enum(name: str) -> sl.COORDINATE_SYSTEM:
    return {
        "IMAGE": sl.COORDINATE_SYSTEM.IMAGE,
        "RIGHT_HANDED_Z_UP_X_FWD": sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
    }[name]


def extract_points_and_colors(
    point_cloud: sl.Mat,
    confidence_map: sl.Mat,
    args: argparse.Namespace,
):
    """Apply organized close-range filtering to a ZED XYZRGBA measure."""
    return filter_organized_cloud(
        point_cloud.get_data(),
        confidence_map.get_data(),
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        coordinate_system=args.coordinate_system,
        confidence_threshold=args.confidence_threshold,
        edge_threshold_m=args.edge_threshold_m,
        erode_invalid_boundary=not args.no_erode_invalid_boundary,
    )


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4×4 SE3 pose matrix to an (N,3) point array."""
    R = pose[:3, :3].astype(np.float32)
    t = pose[:3, 3].astype(np.float32)
    return points @ R.T + t


# ──────────────────────────────────────────────────────────────────────────────
# Binary PLY writer
# ──────────────────────────────────────────────────────────────────────────────

def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a colored point cloud to a binary little-endian PLY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
    ])
    records = np.empty(n, dtype=dtype)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["r"] = colors[:, 0]
    records["g"] = colors[:, 1]
    records["b"] = colors[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(records.tobytes())


# ──────────────────────────────────────────────────────────────────────────────
# Open3D live visualiser
# ──────────────────────────────────────────────────────────────────────────────

class LiveVisualiser:
    """Thin wrapper around an Open3D non-blocking visualiser window.

    FIX: The original code called vis.add_geometry() on an empty PointCloud
    in __init__, before any frames were captured. Open3D internally computes
    an axis-aligned bounding box when adding geometry; on an empty cloud this
    triggers a Sophus SO3::exp assertion deep in Open3D's C++ layer and causes
    an immediate core dump — even before KISS-ICP processes a single frame.

    Fix: do NOT add geometry in __init__. Instead, add it lazily on the first
    update() call when we have real points. This is the only safe approach.
    """

    UPDATE_EVERY_N_FRAMES = 5

    def __init__(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="KISS-ICP Real-time Map  [Q = quit & save]",
            width=1280,
            height=720,
        )
        self.cloud = o3d.geometry.PointCloud()
        # FIX: do NOT add empty geometry here — deferred to first update()
        self._geometry_added = False
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 1.5

    def update(self, points: np.ndarray, colors: np.ndarray) -> bool:
        """Push new map data to the viewer. Returns False if window was closed."""
        self.cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        self.cloud.colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0
        )

        if not self._geometry_added:
            # FIX: add geometry only when we have real points — avoids the
            # empty-cloud bounding-box crash in Open3D/Sophus
            self.vis.add_geometry(self.cloud)
            self._geometry_added = True
            self.vis.reset_view_point(True)
        else:
            self.vis.update_geometry(self.cloud)

        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def is_open(self) -> bool:
        return self.vis.poll_events()

    def destroy(self) -> None:
        self.vis.destroy_window()


# ──────────────────────────────────────────────────────────────────────────────
# KISS-ICP setup
# ──────────────────────────────────────────────────────────────────────────────

def _set_if_exists(obj: object, field: str, value) -> bool:
    """Set obj.field = value only if the field exists. Returns True on success."""
    if hasattr(obj, field):
        setattr(obj, field, value)
        return True
    return False


def build_kiss_icp(args: argparse.Namespace) -> KissICP:
    """Construct a KissICP instance tuned for close-range ZED scanning.

    KISSConfig's internal structure changed between versions:
        v0.x : flat fields directly on KISSConfig  (voxel_size, max_range, …)
        v1.0 : nested sub-models                   (config.mapping.voxel_size, …)
        v1.1+: nested sub-models with different names or removed fields

    Rather than hard-coding one version's layout, this function inspects the
    actual config object at runtime and sets whatever fields exist.  Fields
    that have been removed in a newer version are simply skipped with a warning
    so the script keeps running with KISS-ICP defaults for those parameters.
    """
    config = KISSConfig()

    # Print what the installed version actually exposes so debugging is easy
    model_fields = getattr(KISSConfig, "model_fields", {})
    print(f"KISSConfig fields: {list(model_fields.keys())}")

    # ── helper: try nested then flat ─────────────────────────────────────────
    def apply(nested_path: str, flat_name: str, value) -> None:
        """
        Try config.<sub>.<field> first (v1.x nested style).
        Fall back to config.<flat_name> (v0.x flat style).
        Warn if neither exists.
        """
        parts = nested_path.split(".")           # e.g. ["mapping", "voxel_size"]
        sub_name, field_name = parts[0], parts[1]

        sub = getattr(config, sub_name, None)
        if sub is not None and hasattr(sub, field_name):
            setattr(sub, field_name, value)
            return

        if _set_if_exists(config, flat_name, value):
            return

        print(f"  Warning: config field '{nested_path}' / '{flat_name}' "
              f"not found in this KISS-ICP version — using default.")

    # ── apply parameters ──────────────────────────────────────────────────────
    # Confirmed fields in this version:
    #   data, registration, mapping, adaptive_threshold
    apply("mapping.voxel_size",              "voxel_size",         args.kiss_voxel_m)
    apply("data.max_range",                  "max_range",          args.kiss_max_range_m)
    apply("data.min_range",                  "min_range",          args.kiss_min_range_m)
    apply("data.deskew",                     "deskew",             False)  # ZED captures both eyes simultaneously
    apply("registration.max_num_iterations", "max_num_iterations", 500)
    apply("registration.convergence_criterion", "convergence_criterion", 0.0001)

    # min_motion_th controls which model deviations update KISS-ICP's adaptive
    # correspondence threshold. It does not decide whether a frame is registered.
    # A 5 mm floor prevents close-range depth noise from collapsing the threshold.
    apply("adaptive_threshold.min_motion_th",  "min_motion_th",   args.kiss_min_motion_m)
    # The LiDAR-scale default initial threshold is too permissive at 10-20 cm.
    apply("adaptive_threshold.initial_threshold", "initial_threshold", args.kiss_initial_threshold_m)
    # convergence_criterion — also present in registration in this version
    apply("odometry.convergence_criterion",    "convergence_criterion", 0.0001)

    return KissICP(config=config)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def quality_to_dict(quality: FrameQuality) -> dict[str, object]:
    return {
        "accepted": quality.accepted,
        "reason": quality.reason,
        "point_count": quality.point_count,
        "valid_coverage": quality.valid_coverage,
        "extents_m": list(quality.extents_m),
        "eigenvalues": list(quality.eigenvalues),
    }


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_quality_report(path: Path, report: dict[str, object]) -> None:
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(report), file, indent=2, allow_nan=False)
        file.write("\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)
    report_path = args.quality_report or args.out.with_name(
        f"{args.out.stem}_quality.json"
    )

    # ── open ZED ──────────────────────────────────────────────────────────────
    init = sl.InitParameters()
    init.camera_resolution = resolution_enum(args.resolution)
    if args.depth_mode == "NEURAL_PLUS" and not hasattr(sl.DEPTH_MODE, "NEURAL_PLUS"):
        print("Warning: NEURAL_PLUS is unavailable; falling back to NEURAL.")
        args.depth_mode = "NEURAL"
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = coordinate_system_enum(args.coordinate_system)
    init.depth_minimum_distance = args.camera_min_depth_m
    init.depth_maximum_distance = args.camera_max_depth_m

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")
    zed_sdk_version = str(zed.get_sdk_version())

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = args.confidence_threshold
    runtime.texture_confidence_threshold = args.texture_confidence_threshold
    point_cloud_mat = sl.Mat()
    confidence_mat = sl.Mat()

    # ── warmup ────────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} frames)…")
    for i in range(args.warmup):
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            zed.close()
            raise RuntimeError(f"Warmup frame {i + 1} failed")
    print("Warmup done. Starting mapping. Press Q in the viewer to stop.")

    # ── KISS-ICP ──────────────────────────────────────────────────────────────
    try:
        kiss = build_kiss_icp(args)
    except Exception:
        zed.close()
        raise

    # ── global map and diagnostics ────────────────────────────────────────────
    global_map = GlobalVoxelAccumulator(args.voxel_m)
    all_poses: list[np.ndarray] = []
    frame_records: list[dict[str, object]] = []
    rejected_reasons: Counter[str] = Counter()

    # ── visualiser ────────────────────────────────────────────────────────────
    viz: LiveVisualiser | None = None
    if not args.no_viz:
        viz = LiveVisualiser()

    # ── timing ────────────────────────────────────────────────────────────────
    frame_idx = 0
    t_start = time.perf_counter()
    stop_reason = "unknown"
    fatal_error: Exception | None = None

    try:
        while True:
            # ── check if viewer was closed ─────────────────────────────────
            if viz is not None and not viz.is_open():
                print("Viewer closed — stopping.")
                stop_reason = "viewer_closed"
                break

            # ── grab frame ────────────────────────────────────────────────
            grab_status = zed.grab(runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                print(f"Frame grab failed: {grab_status} — skipping")
                rejected_reasons["grab_failed"] += 1
                frame_records.append({
                    "frame": frame_idx,
                    "accepted": False,
                    "reason": "grab_failed",
                    "status": str(grab_status),
                })
                frame_idx += 1
                continue

            zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(confidence_mat, sl.MEASURE.CONFIDENCE)

            # ── extract points ────────────────────────────────────────────
            filtered = extract_points_and_colors(
                point_cloud_mat, confidence_mat, args
            )
            frame_pts = filtered.points
            frame_col = filtered.colors
            coverage = float(filtered.metrics["filtered_valid_coverage"])

            quality = evaluate_frame_quality(
                frame_pts,
                coverage,
                min_points=args.min_reliable_points,
                min_coverage=args.min_valid_coverage,
            )
            if not quality.accepted:
                rejected_reasons[quality.reason] += 1
                frame_records.append({
                    "frame": frame_idx,
                    **quality_to_dict(quality),
                    "depth_filter": filtered.metrics,
                })
                print(
                    f"Frame {frame_idx}: rejected {quality.reason} "
                    f"({frame_pts.shape[0]} points, {coverage:.1%} coverage)"
                )
                frame_idx += 1
                continue

            # Diagnostic on first good frame — confirm points look sane
            if not all_poses:
                print(f"First frame: {frame_pts.shape[0]} points")
                print(f"  XYZ min:  {frame_pts.min(axis=0)}")
                print(f"  XYZ max:  {frame_pts.max(axis=0)}")
                print(f"  XYZ mean: {frame_pts.mean(axis=0)}")

            # ── per-frame subsampling for ICP ─────────────────────────────
            icp_pts = deterministic_voxel_sample(
                frame_pts,
                voxel_m=args.kiss_voxel_m,
                max_points=args.max_points_per_frame,
            )

            # ── KISS-ICP register ─────────────────────────────────────────
            try:
                kiss.register_frame(
                    icp_pts.astype(np.float64),
                    timestamps=np.array([], dtype=np.float64),
                )
            except Exception as exc:
                print(f"\nFrame {frame_idx}: KISS-ICP register failed: {exc} — skipping")
                rejected_reasons["registration_failed"] += 1
                frame_records.append({
                    "frame": frame_idx,
                    "accepted": False,
                    "reason": "registration_failed",
                    "error": str(exc),
                    "depth_filter": filtered.metrics,
                    "quality": quality_to_dict(quality),
                })
                frame_idx += 1
                continue

            # ── get latest pose ───────────────────────────────────────────
            pose = kiss.last_pose.astype(np.float64)
            if not np.isfinite(pose).all():
                rejected_reasons["nonfinite_pose"] += 1
                frame_records.append({
                    "frame": frame_idx,
                    "accepted": False,
                    "reason": "nonfinite_pose",
                })
                stop_reason = "nonfinite_pose"
                print(f"\nFrame {frame_idx}: non-finite pose — stopping")
                frame_idx += 1
                break

            # ── tracking loss detection ───────────────────────────────────
            jump_m, jump_deg = (0.0, 0.0)
            pose_quality = None
            if all_poses:
                pose_quality = evaluate_pose_quality(
                    all_poses[-1],
                    pose,
                    max_translation_m=args.max_jump_m,
                    max_rotation_deg=args.max_jump_deg,
                )
                jump_m = pose_quality.translation_m
                jump_deg = pose_quality.rotation_deg
            if pose_quality is not None and not pose_quality.accepted:
                rejected_reasons["tracking_jump"] += 1
                frame_records.append({
                    "frame": frame_idx,
                    "accepted": False,
                    "reason": "tracking_jump",
                    "translation_m": jump_m,
                    "rotation_deg": jump_deg,
                    "depth_filter": filtered.metrics,
                    "quality": quality_to_dict(quality),
                })
                stop_reason = "tracking_jump"
                print(
                    f"\nFrame {frame_idx}: tracking jump {jump_m:.3f}m / "
                    f"{jump_deg:.2f}deg > threshold — stopping"
                )
                frame_idx += 1
                break

            all_poses.append(pose.copy())

            # ── globally fuse this accepted frame ─────────────────────────
            world_pts = transform_points(frame_pts, pose)
            global_map.update(world_pts, frame_col)
            frame_records.append({
                "frame": frame_idx,
                "accepted": True,
                "reason": "accepted",
                "translation_m": jump_m,
                "rotation_deg": jump_deg,
                "icp_points": int(icp_pts.shape[0]),
                "map_voxels": len(global_map),
                "depth_filter": filtered.metrics,
                "quality": quality_to_dict(quality),
            })

            # ── fps display ───────────────────────────────────────────────
            frame_idx += 1
            elapsed = time.perf_counter() - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            t_xyz = pose[:3, 3]
            print(
                f"Frame {frame_idx:5d} | "
                f"icp={icp_pts.shape[0]:5d} valid={frame_pts.shape[0]:6d} | "
                f"voxels={len(global_map):7d} coverage={coverage:5.1%} | "
                f"pos=({t_xyz[0]:.3f},{t_xyz[1]:.3f},{t_xyz[2]:.3f}) | "
                f"fps={fps:.1f}",
                end="\r",
            )

            # ── update viewer ─────────────────────────────────────────────
            if (
                viz is not None
                and len(all_poses) % viz.UPDATE_EVERY_N_FRAMES == 0
            ):
                display_pts, display_col, _ = global_map.to_arrays(
                    min_observations=1
                )
                viz.update(display_pts, display_col)

            if len(global_map) > args.max_map_points:
                stop_reason = "map_voxel_cap"
                print(
                    f"\nMap reached safety cap ({args.max_map_points} voxels) — stopping"
                )
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt — saving map…")
        stop_reason = "keyboard_interrupt"
    except Exception as exc:
        fatal_error = exc
        stop_reason = "fatal_error"
        print(f"\nFatal error — preserving accepted output: {exc}")
    finally:
        zed.close()
        if viz is not None:
            viz.destroy()

    # ── final map, trajectory, and quality report ─────────────────────────────
    elapsed = time.perf_counter() - t_start
    final_pts, final_col, final_counts = global_map.to_arrays(
        min_observations=args.min_map_observations
    )
    print(f"\nTotal frames captured: {frame_idx}")
    print(f"Accepted poses: {len(all_poses)}")
    print(f"Final map: {final_pts.shape[0]} voxels")

    if final_pts.shape[0] > 0:
        write_binary_ply(args.out, final_pts, final_col)
        print(f"Saved → {args.out}")
    else:
        print("No voxels met the final observation threshold; no PLY was written.")

    trajectory_path = args.out.with_name(
        f"{args.out.stem}_trajectory.npy"
    )
    if all_poses:
        np.save(trajectory_path, np.stack(all_poses, axis=0))
        print(f"Trajectory saved → {trajectory_path} ({len(all_poses)} poses)")

    closure_m, closure_deg = (0.0, 0.0)
    if len(all_poses) > 1:
        closure_m, closure_deg = pose_step(all_poses[0], all_poses[-1])

    report = {
        "output": str(args.out),
        "trajectory": str(trajectory_path) if all_poses else None,
        "stop_reason": stop_reason,
        "versions": {
            "zed_sdk": zed_sdk_version,
            "kiss_icp": package_version("kiss-icp"),
            "open3d": package_version("open3d"),
            "numpy": np.__version__,
        },
        "config": serializable_args(args),
        "summary": {
            "captured_frames": frame_idx,
            "accepted_frames": len(all_poses),
            "rejected_frames": int(sum(rejected_reasons.values())),
            "rejection_reasons": dict(rejected_reasons),
            "elapsed_s": elapsed,
            "capture_fps": frame_idx / elapsed if elapsed > 0 else 0.0,
            "global_voxels_before_observation_filter": len(global_map),
            "final_voxels": int(final_pts.shape[0]),
            "final_observation_count_min": (
                int(final_counts.min()) if final_counts.size else None
            ),
            "final_observation_count_median": (
                float(np.median(final_counts)) if final_counts.size else None
            ),
            "closure_translation_m": closure_m,
            "closure_rotation_deg": closure_deg,
        },
        "frames": frame_records,
    }
    write_quality_report(report_path, report)
    print(f"Quality report saved → {report_path}")

    if fatal_error is not None:
        raise RuntimeError("Scan stopped after a fatal error") from fatal_error


if __name__ == "__main__":
    main()
