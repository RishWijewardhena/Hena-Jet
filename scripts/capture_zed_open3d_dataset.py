#!/usr/bin/env python3
"""Capture a ZED-M RGB-D sequence in Open3D reconstruction-system format."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pyzed.sl as sl


RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}
DEPTH_MODES = {
    name: getattr(sl.DEPTH_MODE, name)
    for name in ("PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT")
    if hasattr(sl.DEPTH_MODE, name)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture ZED-M frames as image/*.png and depth/*.png for Open3D's "
            "reconstruction_system scripts."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/zed_m_open3d"),
        help="Dataset output folder. Creates image/ and depth/ inside it.",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.15,
        help="Delay between saved frames. Increase this when moving by hand.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=1.00)
    parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTIONS),
        default="HD720",
    )
    parser.add_argument(
        "--depth-mode",
        choices=sorted(DEPTH_MODES),
        default="NEURAL_PLUS" if "NEURAL_PLUS" in DEPTH_MODES else "QUALITY",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.interval_s < 0:
        raise ValueError("--interval-s must be zero or positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or positive")
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m must be greater than --min-depth-m")


def color_image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        return image[:, :, :3][:, :, ::-1].astype(np.uint8)
    raise ValueError(f"Unsupported ZED color image shape: {image.shape}")


def depth_meters_to_png_mm(
    depth_image: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(depth[valid] * 1000.0, 0, np.iinfo(np.uint16).max).astype(
        np.uint16
    )
    return depth_mm


def write_open3d_intrinsic(path: Path, zed: sl.Camera, width: int, height: int) -> None:
    calibration = zed.get_camera_information().camera_configuration.calibration_parameters
    left = calibration.left_cam
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(left.fx),
        float(left.fy),
        float(left.cx),
        float(left.cy),
    )
    o3d.io.write_pinhole_camera_intrinsic(str(path), intrinsic)


def write_reconstruction_config(args: argparse.Namespace, width: int, height: int) -> None:
    config = {
        "name": "zed_m_open3d_capture",
        "path_dataset": str(args.out_dir),
        "path_intrinsic": str(args.out_dir / "intrinsic.json"),
        "depth_min": args.min_depth_m,
        "depth_max": args.max_depth_m,
        "depth_scale": 1000.0,
        "width": width,
        "height": height,
    }
    with (args.out_dir / "capture_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)

    image_dir = args.out_dir / "image"
    depth_dir = args.out_dir / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[args.resolution]
    init.depth_mode = DEPTH_MODES[args.depth_mode]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m

    runtime = sl.RuntimeParameters()
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    color_mat = sl.Mat()
    depth_mat = sl.Mat()

    try:
        for _ in range(args.warmup):
            zed.grab(runtime)

        saved = 0
        width = height = None
        while saved < args.frames:
            status = zed.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"Skipping frame {saved:06d}: grab failed with {status}")
                continue

            zed.retrieve_image(color_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

            color = color_image_to_rgb(color_mat.get_data())
            depth = depth_meters_to_png_mm(
                depth_mat.get_data(),
                args.min_depth_m,
                args.max_depth_m,
            )

            if width is None or height is None:
                height, width = color.shape[:2]
                write_open3d_intrinsic(args.out_dir / "intrinsic.json", zed, width, height)
                write_reconstruction_config(args, width, height)

            frame_name = f"{saved:06d}.png"
            o3d.io.write_image(str(image_dir / frame_name), o3d.geometry.Image(color))
            o3d.io.write_image(str(depth_dir / frame_name), o3d.geometry.Image(depth))
            print(f"saved {frame_name}")

            saved += 1
            if args.interval_s:
                time.sleep(args.interval_s)
    finally:
        zed.close()

    print(f"\nOpen3D dataset saved to: {args.out_dir}")
    print(f"Color frames: {image_dir}")
    print(f"Depth frames: {depth_dir}")
    print(f"Intrinsics: {args.out_dir / 'intrinsic.json'}")
    print(f"Capture config: {args.out_dir / 'capture_config.json'}")


if __name__ == "__main__":
    main()
