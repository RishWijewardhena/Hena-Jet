#!/usr/bin/env python3
"""Capture raw Gemini 305 RGB-D frames without an external pose estimator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def depth_frame_to_uint16(depth_frame: object) -> np.ndarray:
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    values = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    if values.size != width * height:
        raise RuntimeError("depth buffer does not match frame dimensions")
    return values.reshape(height, width).copy()


def color_frame_to_rgb(color_frame: object) -> np.ndarray:
    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    values = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    if values.size != width * height * 3:
        raise RuntimeError("RGB buffer does not match frame dimensions")
    return values.reshape(height, width, 3).copy()


class RawRgbdWriter:
    """Write a trajectory-free RGB-D sequence for offline DynamicFusion."""

    def __init__(
        self,
        root: Path,
        depth_intrinsics: tuple[float, float, float, float],
        depth_scale_mm: float,
        camera_metadata: dict[str, str],
    ) -> None:
        self.root = Path(root)
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise RuntimeError(f"capture directory is not empty: {self.root}")
        intrinsics = tuple(float(value) for value in depth_intrinsics)
        if len(intrinsics) != 4 or not np.isfinite(intrinsics).all():
            raise ValueError("depth intrinsics must contain finite fx, fy, cx, cy")
        if intrinsics[0] <= 0 or intrinsics[1] <= 0:
            raise ValueError("focal lengths must be positive")
        if not math.isfinite(depth_scale_mm) or depth_scale_mm <= 0:
            raise ValueError("depth_scale_mm must be positive")

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "depth").mkdir()
        (self.root / "rgb").mkdir()
        self.depth_intrinsics = intrinsics
        self.depth_scale_mm = float(depth_scale_mm)
        self.camera_metadata = dict(camera_metadata)
        self._records: list[tuple[float, str, str]] = []

        calibration = " ".join(f"{value:.9g}" for value in intrinsics)
        (self.root / "calibration.txt").write_text(calibration + "\n", encoding="utf-8")
        (self.root / "camera.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "depthIntrinsics": {
                        "fx": intrinsics[0],
                        "fy": intrinsics[1],
                        "cx": intrinsics[2],
                        "cy": intrinsics[3],
                    },
                    "depthScaleMm": self.depth_scale_mm,
                    "metadata": self.camera_metadata,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_indexes()

    def __len__(self) -> int:
        return len(self._records)

    def record(
        self,
        timestamp_s: float,
        raw_depth: np.ndarray,
        color_rgb: np.ndarray,
    ) -> None:
        raw_depth = np.asarray(raw_depth)
        color_rgb = np.asarray(color_rgb)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp must be finite")
        if self._records and timestamp_s <= self._records[-1][0]:
            raise ValueError("capture timestamps must be strictly increasing")
        if raw_depth.ndim != 2 or raw_depth.dtype != np.uint16:
            raise ValueError("raw_depth must be a uint16 image")
        if color_rgb.ndim != 3 or color_rgb.shape[2] != 3 or color_rgb.dtype != np.uint8:
            raise ValueError("color_rgb must be an HxWx3 uint8 image")

        timestamp = f"{timestamp_s:.6f}"
        depth_path = f"depth/{timestamp}.png"
        rgb_path = f"rgb/{timestamp}.png"
        if not cv2.imwrite(str(self.root / depth_path), raw_depth):
            raise RuntimeError("failed to write raw depth PNG")
        if not cv2.imwrite(
            str(self.root / rgb_path), np.ascontiguousarray(color_rgb[..., ::-1])
        ):
            raise RuntimeError("failed to write RGB PNG")
        self._records.append((timestamp_s, rgb_path, depth_path))

    def _write_indexes(self) -> None:
        associations = ["# rgb_timestamp rgb_path depth_timestamp depth_path\n"]
        for timestamp_s, rgb_path, depth_path in self._records:
            timestamp = f"{timestamp_s:.6f}"
            associations.append(
                f"{timestamp} {rgb_path} {timestamp} {depth_path}\n"
            )
        (self.root / "associated.txt").write_text(
            "".join(associations), encoding="utf-8"
        )
        info = [
            f"depth_scale_mm={self.depth_scale_mm:.9g}\n",
            "depth_processing=raw\n",
            "pose_source=none\n",
            f"frame_count={len(self)}\n",
        ]
        info.extend(
            f"{key}={value}\n" for key, value in sorted(self.camera_metadata.items())
        )
        (self.root / "capture_info.txt").write_text("".join(info), encoding="utf-8")

    def finalize(self) -> None:
        self._write_indexes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture raw Gemini 305 RGB-D for offline DynamicFusion"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=530)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n-frames", type=int, default=-1)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    args = parser.parse_args()
    if args.n_frames == 0 or args.n_frames < -1:
        parser.error("--n-frames must be -1 or a positive integer")
    if min(args.width, args.height, args.fps, args.frame_timeout_ms) <= 0:
        parser.error("stream dimensions, FPS, and timeout must be positive")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    try:
        from pyorbbecsdk import (  # type: ignore
            Config,
            Context,
            OBFormat,
            OBFrameAggregateOutputMode,
            OBSensorType,
            Pipeline,
        )
    except ImportError as exc:
        raise SystemExit(
            "pyorbbecsdk2 is required; run this command in the hena_jet environment"
        ) from exc

    context = Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        raise SystemExit("No Orbbec camera detected")

    pipeline = Pipeline()
    config = Config()
    try:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(
            args.width, args.height, OBFormat.Y16, args.fps
        )
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            args.width, args.height, OBFormat.RGB, args.fps
        )
    except Exception as exc:
        raise SystemExit(
            f"Gemini does not provide {args.width}x{args.height}@{args.fps} "
            f"raw depth and RGB profiles: {exc}"
        ) from exc

    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    try:
        pipeline.enable_frame_sync()
    except Exception as exc:
        print(f"Warning: hardware frame synchronization is unavailable: {exc}")

    writer: RawRgbdWriter | None = None
    pipeline_started = False
    last_timestamp_s: float | None = None
    try:
        pipeline.start(config)
        pipeline_started = True
        device_info = pipeline.get_device().get_device_info()
        for _ in range(args.warmup_frames):
            if pipeline.wait_for_frames(args.frame_timeout_ms) is None:
                raise RuntimeError("Gemini timed out during warmup")

        camera = pipeline.get_camera_param().depth_intrinsic
        print(
            "Capturing raw depth without KISS-ICP, alignment, hole filling, "
            "or any saved trajectory"
        )
        while args.n_frames < 0 or writer is None or len(writer) < args.n_frames:
            frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            if frames is None:
                print("Skipping timeout")
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame is None or color_frame is None:
                print("Skipping incomplete RGB-D frame set")
                continue
            timestamp_s = float(depth_frame.get_timestamp_us()) / 1_000_000.0
            if last_timestamp_s is not None and timestamp_s <= last_timestamp_s:
                print("Skipping non-monotonic camera timestamp")
                continue

            if writer is None:
                writer = RawRgbdWriter(
                    args.out,
                    depth_intrinsics=(camera.fx, camera.fy, camera.cx, camera.cy),
                    depth_scale_mm=float(depth_frame.get_depth_scale()),
                    camera_metadata={
                        "camera": device_info.get_name(),
                        "serial": device_info.get_serial_number(),
                        "firmware": device_info.get_firmware_version(),
                        "depth_resolution": f"{depth_frame.get_width()}x{depth_frame.get_height()}",
                        "rgb_resolution": f"{color_frame.get_width()}x{color_frame.get_height()}",
                        "fps": str(args.fps),
                    },
                )

            writer.record(
                timestamp_s,
                depth_frame_to_uint16(depth_frame),
                color_frame_to_rgb(color_frame),
            )
            last_timestamp_s = timestamp_s
            target = str(args.n_frames) if args.n_frames > 0 else "inf"
            print(f"{len(writer):05d}/{target}: {timestamp_s:.6f}")
    except KeyboardInterrupt:
        print("Stopping capture")
    finally:
        if pipeline_started:
            pipeline.stop()
        if writer is not None:
            writer.finalize()
            print(f"Saved {len(writer)} raw RGB-D frames to {writer.root}")


if __name__ == "__main__":
    main()
