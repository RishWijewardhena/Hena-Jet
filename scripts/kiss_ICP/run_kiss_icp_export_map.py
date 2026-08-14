#!/usr/bin/env python3
"""Run KISS-ICP from a live Gemini 305 stream and export its local map."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from pyorbbecsdk import (
    Config,
    Context,
    DepthFrame,
    OBFormat,
    OBSensorType,
    Pipeline,
    PointCloudFilter,
    VideoStreamProfile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KISS-ICP on a live Gemini 305 depth stream and save its local map."
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/kiss_icp_local_map.ply"))
    parser.add_argument("--poses-out", type=Path, default=Path("outputs/kiss_icp_local_map_poses.npy"))
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=530)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1_000)
    parser.add_argument("--max-consecutive-timeouts", type=int, default=10)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument(
        "--point-unit-m",
        type=float,
        default=0.001,
        help=(
            "Metres represented by one SDK-normalized PointCloudFilter XYZ unit. "
            "The stream is normalized to millimetres, so the default is 0.001."
        ),
    )
    parser.add_argument("--min-range", type=float, default=0.04)
    parser.add_argument("--max-range", type=float, default=0.45)
    parser.add_argument("--voxel-size", type=float, default=0.002)
    parser.add_argument("--max-points-per-voxel", type=int, default=20)
    parser.add_argument("--initial-threshold", type=float, default=0.02)
    parser.add_argument("--min-motion-th", type=float, default=0.002)
    parser.add_argument("--n-scans", type=int, default=-1)
    args = parser.parse_args()
    if args.min_range < 0 or args.max_range <= args.min_range:
        parser.error("--max-range must be greater than a non-negative --min-range")
    if args.n_scans == 0 or args.n_scans < -1:
        parser.error("--n-scans must be -1 or a positive integer")
    for name in (
        "width",
        "height",
        "fps",
        "frame_timeout_ms",
        "max_consecutive_timeouts",
        "point_unit_m",
        "voxel_size",
        "max_points_per_voxel",
        "initial_threshold",
        "min_motion_th",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must not be negative")
    return args


def make_config(args: argparse.Namespace) -> KISSConfig:
    config = KISSConfig()
    config.data.min_range = args.min_range
    config.data.max_range = args.max_range
    config.data.deskew = False
    config.mapping.voxel_size = args.voxel_size
    config.mapping.max_points_per_voxel = args.max_points_per_voxel
    config.adaptive_threshold.initial_threshold = args.initial_threshold
    config.adaptive_threshold.min_motion_th = args.min_motion_th
    return config


def pose_step(previous: np.ndarray | None, current: np.ndarray) -> tuple[float, float]:
    if previous is None:
        return 0.0, 0.0

    delta = np.linalg.inv(previous) @ current
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cos_angle = (float(np.trace(delta[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    rotation_deg = math.degrees(math.acos(cos_angle))
    return translation_m, rotation_deg


def prepare_points_meters(points: np.ndarray, point_unit_m: float) -> np.ndarray:
    """Convert SDK XYZ coordinates to metres and discard invalid points."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if point_unit_m <= 0:
        raise ValueError("point_unit_m must be positive")

    valid = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    return points[valid].astype(np.float64, copy=True) * point_unit_m


def point_cloud_from_depth_frame(
    point_cloud_filter: PointCloudFilter,
    depth_frame: DepthFrame,
    point_unit_m: float,
) -> np.ndarray:
    """Generate an XYZ cloud from one Orbbec depth frame in metres."""
    # Official SDK pattern:
    # https://github.com/orbbec/pyorbbecsdk/blob/v2-main/examples/beginner/05_point_cloud.py
    point_cloud_frame = point_cloud_filter.process(depth_frame)
    if point_cloud_frame is None:
        raise RuntimeError("Orbbec PointCloudFilter returned no point-cloud frame")

    data = np.frombuffer(point_cloud_frame.as_points_frame().get_data(), dtype=np.float32)
    if data.size % 3 != 0:
        raise RuntimeError("Orbbec point-cloud buffer size is not a multiple of three")
    return prepare_points_meters(data.reshape((-1, 3)), point_unit_m)


def colored_point_cloud_from_frame(
    point_cloud_filter: PointCloudFilter,
    frame: object,
    point_unit_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate corresponding XYZ metre and RGB uint8 arrays."""
    # RGB_POINT layout is documented by the official Orbbec point-cloud example:
    # https://github.com/orbbec/pyorbbecsdk/blob/v2-main/examples/beginner/05_point_cloud.py
    point_cloud_frame = point_cloud_filter.process(frame)
    if point_cloud_frame is None:
        raise RuntimeError("Orbbec PointCloudFilter returned no colored point cloud")

    data = np.frombuffer(point_cloud_frame.as_points_frame().get_data(), dtype=np.float32)
    if data.size % 6 != 0:
        raise RuntimeError("Orbbec RGB point-cloud buffer size is not a multiple of six")
    records = data.reshape((-1, 6))
    xyz = records[:, :3]
    valid = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    points = xyz[valid].astype(np.float64, copy=True) * point_unit_m
    colors = np.clip(np.rint(records[valid, 3:6]), 0, 255).astype(np.uint8)
    return points, colors


def configure_point_cloud_scale(
    point_cloud_filter: PointCloudFilter,
    depth_frame: DepthFrame,
) -> float:
    """Normalize device-specific XYZ units to millimetres."""
    depth_scale_mm = float(depth_frame.get_depth_scale())
    if not math.isfinite(depth_scale_mm) or depth_scale_mm <= 0:
        raise RuntimeError(f"Invalid Gemini depth scale: {depth_scale_mm}")
    point_cloud_filter.set_position_data_scaled(depth_scale_mm)
    return depth_scale_mm


def write_xyz_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in points:
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def write_xyzrgb_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write corresponding XYZ and RGB arrays as an ASCII PLY."""
    points = np.asarray(points)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must have the same (N, 3) shape as points")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def wait_for_depth_frame(pipeline: Pipeline, timeout_ms: int) -> DepthFrame | None:
    frames = pipeline.wait_for_frames(timeout_ms)
    if frames is None:
        return None
    return frames.get_depth_frame()


def select_depth_profile(
    pipeline: Pipeline,
    args: argparse.Namespace,
) -> VideoStreamProfile:
    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if profiles is None:
        raise RuntimeError("The connected Orbbec device has no depth profiles")
    try:
        return profiles.get_video_stream_profile(
            args.width,
            args.height,
            OBFormat.Y16,
            args.fps,
        )
    except Exception as error:
        raise RuntimeError(
            "Gemini depth profile not available: "
            f"{args.width}x{args.height} @ {args.fps} fps, Y16"
        ) from error


def save_results(args: argparse.Namespace, kiss: KissICP, poses: list[np.ndarray]) -> None:
    local_map = kiss.local_map.point_cloud()
    if local_map.size == 0 or not poses:
        raise RuntimeError("KISS-ICP local map is empty; no output was written.")

    write_xyz_ply(args.out, local_map)
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, np.stack(poses, axis=0))

    print(f"Saved local map: {args.out} ({local_map.shape[0]} points)")
    print(f"Saved poses: {args.poses_out}")


def main() -> None:
    args = parse_args()
    context = Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        raise RuntimeError("No Orbbec camera detected")

    device = devices.get_device_by_index(0)
    device_info = device.get_device_info()
    pipeline = Pipeline(device)
    sdk_config = Config()
    depth_profile = select_depth_profile(pipeline, args)
    sdk_config.enable_stream(depth_profile)

    kiss_config = make_config(args)
    kiss = KissICP(kiss_config)
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.POINT)

    print(
        f"Camera: {device_info.get_name()} | serial={device_info.get_serial_number()} | "
        f"firmware={device_info.get_firmware_version()}"
    )
    if "Gemini 305" not in device_info.get_name():
        print("WARNING: The first connected Orbbec device is not identified as Gemini 305.")
    print(f"Depth stream: {args.width}x{args.height} @ {args.fps} fps, Y16")
    print(
        "KISS config: "
        f"range=[{kiss_config.data.min_range}, {kiss_config.data.max_range}]m, "
        f"voxel={kiss_config.mapping.voxel_size}m, "
        f"threshold={kiss_config.adaptive_threshold.initial_threshold}m"
    )
    print("Press Ctrl+C to stop and save the map.")

    poses: list[np.ndarray] = []
    previous_pose: np.ndarray | None = None
    registered_scans = 0
    consecutive_timeouts = 0
    pipeline_started = False
    empty_timestamps = np.array([], dtype=np.float64)

    try:
        pipeline.start(sdk_config)
        pipeline_started = True

        first_depth_frame = None
        warmup_frames_received = 0
        required_warmup_frames = max(args.warmup_frames, 1)
        while warmup_frames_received < required_warmup_frames:
            first_depth_frame = wait_for_depth_frame(pipeline, args.frame_timeout_ms)
            if first_depth_frame is None:
                consecutive_timeouts += 1
                if consecutive_timeouts >= args.max_consecutive_timeouts:
                    raise RuntimeError("Timed out while warming up the Gemini depth stream")
            else:
                consecutive_timeouts = 0
                warmup_frames_received += 1

        if first_depth_frame is None:
            raise RuntimeError("Gemini warmup completed without a depth frame")
        depth_scale_mm = configure_point_cloud_scale(
            point_cloud_filter,
            first_depth_frame,
        )
        print(f"Raw depth-image scale: {depth_scale_mm:.6g} mm per depth unit")
        print(
            "Point-cloud conversion: SDK-normalized millimetres, then "
            f"XYZ * {args.point_unit_m:g} = metres"
        )

        while args.n_scans < 0 or registered_scans < args.n_scans:
            depth_frame = wait_for_depth_frame(pipeline, args.frame_timeout_ms)
            if depth_frame is None:
                consecutive_timeouts += 1
                print(
                    f"Waiting for depth frame "
                    f"({consecutive_timeouts}/{args.max_consecutive_timeouts})..."
                )
                if consecutive_timeouts >= args.max_consecutive_timeouts:
                    raise RuntimeError("Gemini depth stream stopped producing frames")
                continue
            consecutive_timeouts = 0

            points = point_cloud_from_depth_frame(
                point_cloud_filter,
                depth_frame,
                point_unit_m=args.point_unit_m,
            )
            ranges = np.linalg.norm(points, axis=1)
            usable = points[(ranges >= args.min_range) & (ranges <= args.max_range)]
            if usable.shape[0] < 3:
                median_z = float(np.median(points[:, 2])) if points.size else math.nan
                print(
                    "Skipping frame: no usable close-range cloud "
                    f"(valid={points.shape[0]}, median_z={median_z:.3f}m)"
                )
                continue

            kiss.register_frame(usable, empty_timestamps)
            pose = kiss.last_pose.copy()
            poses.append(pose)
            registered_scans += 1

            step_m, step_deg = pose_step(previous_pose, pose)
            previous_pose = pose
            map_points = kiss.local_map.point_cloud()
            scan_total = str(args.n_scans) if args.n_scans > 0 else "inf"
            print(
                f"{registered_scans:04d}/{scan_total}: "
                f"valid={points.shape[0]} usable={usable.shape[0]} "
                f"map={map_points.shape[0]} step={step_m:.4f}m rot={step_deg:.2f}deg"
            )
    except KeyboardInterrupt:
        print("\nStopping capture and saving results...")
    finally:
        if pipeline_started:
            pipeline.stop()

    save_results(args, kiss, poses)


if __name__ == "__main__":
    main()
