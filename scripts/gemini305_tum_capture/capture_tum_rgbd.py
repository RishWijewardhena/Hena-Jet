#!/usr/bin/env python3
"""Capture filtered Gemini 305 D2C RGB-D frames and KISS poses in TUM format.

Format references:
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
https://github.com/puzzlepaint/surfelmeshing/blob/master/README.md#running
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    Frame,
    OBFormat,
    OBStreamType,
    Pipeline,
    PointCloudFilter,
)


KISS_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "kiss_ICP"
sys.path.insert(0, str(KISS_SCRIPT_DIR))

from run_kiss_icp_export_map import (  # noqa: E402
    apply_depth_filter_chain,
    colored_point_cloud_from_frame,
    configure_point_cloud_scale,
    create_depth_filter_chain,
    make_rgbd_frame_set,
    select_color_profile,
    select_depth_profile,
    wait_for_rgbd_frames,
)


def rotation_matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized TUM-order quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")

    quaternion = np.array(
        [
            math.copysign(
                math.sqrt(
                    max(
                        0.0,
                        1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
                    )
                )
                / 2.0,
                rotation[2, 1] - rotation[1, 2],
            ),
            math.copysign(
                math.sqrt(
                    max(
                        0.0,
                        1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2],
                    )
                )
                / 2.0,
                rotation[0, 2] - rotation[2, 0],
            ),
            math.copysign(
                math.sqrt(
                    max(
                        0.0,
                        1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2],
                    )
                )
                / 2.0,
                rotation[1, 0] - rotation[0, 1],
            ),
            math.sqrt(max(0.0, 1.0 + float(np.trace(rotation)))) / 2.0,
        ]
    )
    norm = float(np.linalg.norm(quaternion))
    if norm == 0:
        raise ValueError("rotation matrix produced a zero quaternion")
    return quaternion / norm


def depth_frame_to_uint16(depth_frame: Frame) -> np.ndarray:
    """Copy an aligned Y16 depth frame into a 16-bit image."""
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    values = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    if values.size != width * height:
        raise RuntimeError("Aligned depth buffer does not match its dimensions")
    return values.reshape((height, width)).copy()


def color_frame_to_rgb(color_frame: Frame) -> np.ndarray:
    """Copy an aligned RGB frame into an 8-bit RGB image."""
    width = color_frame.get_width()
    height = color_frame.get_height()
    values = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    if values.size != width * height * 3:
        raise RuntimeError("Aligned RGB buffer does not match its dimensions")
    return values.reshape((height, width, 3)).copy()


class TumRgbdCapture:
    """Write D2C RGB-D frames and KISS camera poses for SurfelMeshing."""

    def __init__(
        self,
        root: Path,
        intrinsics: tuple[float, float, float, float],
        depth_scale_mm: float,
        metadata: dict[str, str],
    ) -> None:
        self.root = Path(root)
        if self.root.exists():
            if not self.root.is_dir():
                raise RuntimeError(f"Capture output is not a directory: {self.root}")
            if any(self.root.iterdir()):
                raise RuntimeError(f"Capture directory is not empty: {self.root}")
        if len(intrinsics) != 4 or not np.isfinite(intrinsics).all():
            raise ValueError("intrinsics must contain finite fx, fy, cx, cy")
        if intrinsics[0] <= 0 or intrinsics[1] <= 0:
            raise ValueError("focal lengths must be positive")
        if not math.isfinite(depth_scale_mm) or depth_scale_mm <= 0:
            raise ValueError("depth_scale_mm must be positive")

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rgb").mkdir()
        (self.root / "depth").mkdir()
        self.depth_scaling = 1000.0 / depth_scale_mm
        self._depth_scale_mm = depth_scale_mm
        self._metadata = dict(metadata)
        self._records: list[tuple[float, str, str, np.ndarray]] = []

        calibration = " ".join(f"{value:.9g}" for value in intrinsics)
        (self.root / "calibration.txt").write_text(
            f"{calibration}\n",
            encoding="utf-8",
        )
        self._write_capture_info()

    def __len__(self) -> int:
        return len(self._records)

    def _write_capture_info(self) -> None:
        lines = [
            f"depth_scale_mm={self._depth_scale_mm:.9g}\n",
            f"surfelmeshing_depth_scaling={self.depth_scaling:.9g}\n",
            f"frame_count={len(self)}\n",
        ]
        lines.extend(f"{key}={value}\n" for key, value in sorted(self._metadata.items()))
        (self.root / "capture_info.txt").write_text(
            "".join(lines),
            encoding="utf-8",
        )

    def record(
        self,
        timestamp_s: float,
        depth: np.ndarray,
        color_rgb: np.ndarray,
        pose: np.ndarray,
    ) -> None:
        depth = np.asarray(depth)
        color_rgb = np.asarray(color_rgb)
        pose = np.asarray(pose, dtype=np.float64)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._records and timestamp_s <= self._records[-1][0]:
            raise ValueError("capture timestamps must be strictly increasing")
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError("depth must be a uint16 image")
        if color_rgb.shape != (*depth.shape, 3) or color_rgb.dtype != np.uint8:
            raise ValueError("color_rgb must be a matching uint8 RGB image")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be a finite 4x4 matrix")

        timestamp = f"{timestamp_s:.6f}"
        rgb_relative = f"rgb/{timestamp}.png"
        depth_relative = f"depth/{timestamp}.png"
        color_bgr = np.ascontiguousarray(color_rgb[..., ::-1])
        if not cv2.imwrite(str(self.root / depth_relative), depth):
            raise RuntimeError("Failed to write TUM depth PNG")
        if not cv2.imwrite(str(self.root / rgb_relative), color_bgr):
            raise RuntimeError("Failed to write TUM RGB PNG")

        quaternion = rotation_matrix_to_xyzw(pose[:3, :3])
        pose_values = np.concatenate((pose[:3, 3], quaternion))
        self._records.append((timestamp_s, rgb_relative, depth_relative, pose_values))

    def finalize(self) -> None:
        """Write the TUM indexes after all image pairs have been recorded."""
        rgb_lines = ["# timestamp rgb_path\n"]
        depth_lines = ["# timestamp depth_path\n"]
        associated_lines = ["# rgb_timestamp rgb_path depth_timestamp depth_path\n"]
        trajectory_lines = ["# timestamp tx ty tz qx qy qz qw\n"]
        for timestamp_s, rgb_path, depth_path, pose_values in self._records:
            timestamp = f"{timestamp_s:.6f}"
            rgb_lines.append(f"{timestamp} {rgb_path}\n")
            depth_lines.append(f"{timestamp} {depth_path}\n")
            associated_lines.append(
                f"{timestamp} {rgb_path} {timestamp} {depth_path}\n"
            )
            pose_text = " ".join(f"{value:.9g}" for value in pose_values)
            trajectory_lines.append(f"{timestamp} {pose_text}\n")

        for filename, lines in (
            ("rgb.txt", rgb_lines),
            ("depth.txt", depth_lines),
            ("associated.txt", associated_lines),
            ("trajectory.txt", trajectory_lines),
        ):
            (self.root / filename).write_text("".join(lines), encoding="utf-8")
        self._write_capture_info()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Gemini 305 SDK-filtered D2C RGB-D and KISS poses for "
            "offline SurfelMeshing."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=530)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1_000)
    parser.add_argument("--max-consecutive-timeouts", type=int, default=10)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--point-unit-m", type=float, default=0.001)
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
    if args.out.exists():
        if not args.out.is_dir():
            parser.error(f"--out is not a directory: {args.out}")
        if any(args.out.iterdir()):
            parser.error(f"--out directory is not empty: {args.out}")
    return args


def make_kiss_config(args: argparse.Namespace) -> KISSConfig:
    config = KISSConfig()
    config.data.min_range = args.min_range
    config.data.max_range = args.max_range
    config.data.deskew = False
    config.mapping.voxel_size = args.voxel_size
    config.mapping.max_points_per_voxel = args.max_points_per_voxel
    config.adaptive_threshold.initial_threshold = args.initial_threshold
    config.adaptive_threshold.min_motion_th = args.min_motion_th
    return config


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
    color_profile = select_color_profile(pipeline, args)
    sdk_config.enable_stream(depth_profile)
    sdk_config.enable_stream(color_profile)

    kiss = KissICP(make_kiss_config(args))
    depth_filters = create_depth_filter_chain(
        device,
        width=args.width,
        height=args.height,
    )
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

    print(
        f"Camera: {device_info.get_name()} | serial={device_info.get_serial_number()} | "
        f"firmware={device_info.get_firmware_version()}"
    )
    print(f"RGB-D: {args.width}x{args.height} @ {args.fps} fps")
    print("SDK filters: EdgeNoiseRemoval -> HoleFilling (no decimation)")
    print("Alignment: filtered depth -> color camera (D2C)")
    print(f"Capture output: {args.out}")
    print("Press Ctrl+C to stop and finalize the capture.")

    capture: TumRgbdCapture | None = None
    pipeline_started = False
    registered_scans = 0
    consecutive_timeouts = 0
    timestamp_epoch_offset_s: float | None = None
    empty_timestamps = np.array([], dtype=np.float64)

    try:
        pipeline.enable_frame_sync()
        pipeline.start(sdk_config)
        pipeline_started = True

        first_frames = None
        warmup_frames_received = 0
        required_warmup_frames = max(args.warmup_frames, 1)
        while warmup_frames_received < required_warmup_frames:
            first_frames = wait_for_rgbd_frames(pipeline, args.frame_timeout_ms)
            if first_frames is None:
                consecutive_timeouts += 1
                if consecutive_timeouts >= args.max_consecutive_timeouts:
                    raise RuntimeError("Timed out while warming up Gemini RGB-D streams")
            else:
                consecutive_timeouts = 0
                warmup_frames_received += 1

        if first_frames is None:
            raise RuntimeError("Gemini warmup completed without an RGB-D frame set")
        raw_depth_scale_mm = configure_point_cloud_scale(
            point_cloud_filter,
            first_frames.get_depth_frame(),
        )
        intrinsics = color_profile.get_intrinsic()
        print(
            f"Raw depth scale: {raw_depth_scale_mm:g} mm/unit; "
            "the aligned scale will be recorded with the capture"
        )

        while args.n_scans < 0 or registered_scans < args.n_scans:
            frames = wait_for_rgbd_frames(pipeline, args.frame_timeout_ms)
            if frames is None:
                consecutive_timeouts += 1
                if consecutive_timeouts >= args.max_consecutive_timeouts:
                    raise RuntimeError("Gemini RGB-D streams stopped producing frames")
                continue
            consecutive_timeouts = 0

            filtered_depth = apply_depth_filter_chain(
                frames.get_depth_frame(),
                depth_filters,
            )
            filtered_frames = make_rgbd_frame_set(
                filtered_depth,
                frames.get_color_frame(),
            )
            aligned_frames = align_filter.process(filtered_frames)
            if aligned_frames is None:
                print("Skipping frame: D2C alignment failed")
                continue
            aligned_depth = aligned_frames.get_depth_frame()
            aligned_color = aligned_frames.get_color_frame()
            if aligned_depth is None or aligned_color is None:
                print("Skipping frame: D2C did not return both RGB and depth")
                continue
            if capture is None:
                aligned_depth_scale_mm = float(aligned_depth.get_depth_scale())
                capture = TumRgbdCapture(
                    args.out,
                    intrinsics=(
                        intrinsics.fx,
                        intrinsics.fy,
                        intrinsics.cx,
                        intrinsics.cy,
                    ),
                    depth_scale_mm=aligned_depth_scale_mm,
                    metadata={
                        "camera": device_info.get_name(),
                        "serial": device_info.get_serial_number(),
                        "firmware": device_info.get_firmware_version(),
                        "resolution": f"{args.width}x{args.height}",
                        "fps": str(args.fps),
                        "filter_chain": (
                            "EdgeNoiseRemovalFilter,HoleFillingFilter"
                        ),
                        "decimation": "disabled",
                        "alignment": "D2C",
                        "pose_source": "KISS-ICP",
                    },
                )
                print(
                    "Aligned depth units: "
                    f"{aligned_depth_scale_mm:g} mm/unit; "
                    "SurfelMeshing --depth_scaling "
                    f"{capture.depth_scaling:g}"
                )

            points, _ = colored_point_cloud_from_frame(
                point_cloud_filter,
                aligned_frames,
                point_unit_m=args.point_unit_m,
            )
            ranges = np.linalg.norm(points, axis=1)
            usable = points[(ranges >= args.min_range) & (ranges <= args.max_range)]
            if usable.shape[0] < 3:
                print(f"Skipping frame: only {usable.shape[0]} usable close-range points")
                continue

            kiss.register_frame(usable, empty_timestamps)
            pose = kiss.last_pose.copy()
            hardware_timestamp_s = aligned_depth.get_timestamp_us() / 1_000_000.0
            if timestamp_epoch_offset_s is None:
                timestamp_epoch_offset_s = time.time() - hardware_timestamp_s
            timestamp_s = timestamp_epoch_offset_s + hardware_timestamp_s

            capture.record(
                timestamp_s,
                depth_frame_to_uint16(aligned_depth),
                color_frame_to_rgb(aligned_color),
                pose,
            )
            registered_scans += 1
            scan_total = str(args.n_scans) if args.n_scans > 0 else "inf"
            print(
                f"{registered_scans:04d}/{scan_total}: "
                f"valid={points.shape[0]} usable={usable.shape[0]} saved"
            )
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        if pipeline_started:
            pipeline.stop()
        if capture is not None:
            capture.finalize()
            print(f"Saved {len(capture)} synchronized RGB-D frames: {capture.root}")
            print(
                "SurfelMeshing input: "
                f"{capture.root} trajectory.txt --depth_scaling "
                f"{capture.depth_scaling:g} --max_depth {args.max_range:g}"
            )


if __name__ == "__main__":
    main()
