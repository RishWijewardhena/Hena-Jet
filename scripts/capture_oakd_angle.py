#!/usr/bin/env python3
"""Capture one or more OAK-D RGB-D frames as point clouds at known scanner angles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import depthai as dai
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture RGB/depth point clouds from an OAK-D camera."
    )
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--height-m", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("captures/oakd_scan"))
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=0.80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--no-left-right-check", action="store_true")
    parser.add_argument("--no-subpixel", action="store_true")
    parser.add_argument("--extended-disparity", action="store_true")
    return parser.parse_args()


def mono_resolution_from_size(width: int, height: int) -> dai.MonoCameraProperties.SensorResolution:
    if width <= 640 and height <= 480:
        return dai.MonoCameraProperties.SensorResolution.THE_400_P
    if width <= 1280 and height <= 800:
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    return dai.MonoCameraProperties.SensorResolution.THE_1200_P


def create_pipeline(args: argparse.Namespace) -> tuple[dai.Pipeline, object, object]:
    pipeline = dai.Pipeline()

    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb.setPreviewSize(args.width, args.height)
    rgb.setPreviewKeepAspectRatio(False)
    rgb.setInterleaved(False)
    rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    rgb.setFps(args.fps)

    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_resolution = mono_resolution_from_size(args.width, args.height)
    left.setResolution(mono_resolution)
    right.setResolution(mono_resolution)
    left.setFps(args.fps)
    right.setFps(args.fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setLeftRightCheck(not args.no_left_right_check)
    stereo.setSubpixel(not args.no_subpixel)
    stereo.setExtendedDisparity(args.extended_disparity)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(args.width, args.height)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    rgb_queue = rgb.preview.createOutputQueue(maxSize=4, blocking=False)
    depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    return pipeline, rgb_queue, depth_queue


def camera_intrinsics_from_device(device: dai.Device, width: int, height: int) -> dict:
    calibration = device.readCalibration2()
    matrix = np.asarray(
        calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height),
        dtype=np.float64,
    )
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def resize_color_nearest(color: np.ndarray, height: int, width: int) -> np.ndarray:
    if color.shape[:2] == (height, width):
        return color
    row_indices = np.minimum(
        (np.arange(height) * color.shape[0] / height).astype(np.int64),
        color.shape[0] - 1,
    )
    col_indices = np.minimum(
        (np.arange(width) * color.shape[1] / width).astype(np.int64),
        color.shape[1] - 1,
    )
    return color[row_indices[:, None], col_indices]


def color_bgr_to_rgb(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    color = np.asarray(frame)
    if color.ndim == 2:
        color = np.repeat(color[:, :, None], 3, axis=2)
    if color.shape[2] > 3:
        color = color[:, :, :3]
    color = resize_color_nearest(color, height, width)
    return color[:, :, ::-1].astype(np.uint8)


def point_cloud_from_depth(
    depth_m: np.ndarray,
    color_rgb: np.ndarray,
    intrinsics: dict,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    rows, cols = np.indices(depth_m.shape, dtype=np.float32)

    z = depth_m
    x = (cols - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (rows - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points_image = np.stack([x, y, z], axis=-1)

    return points_image[valid].astype(np.float32), color_rgb[valid].astype(np.uint8)


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_capture(
    args: argparse.Namespace,
    angle_deg: float,
    depth_msg,
    rgb_msg,
    intrinsics: dict,
) -> None:
    depth_mm = np.asarray(depth_msg.getFrame(), dtype=np.float32)
    depth_m = depth_mm / 1000.0
    color_rgb = color_bgr_to_rgb(rgb_msg.getCvFrame(), depth_m.shape[0], depth_m.shape[1])
    points, colors = point_cloud_from_depth(
        depth_m,
        color_rgb,
        intrinsics,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )

    stem = f"angle_{angle_deg:07.2f}".replace(".", "p")
    npz_path = args.out_dir / f"{stem}.npz"
    meta_path = args.out_dir / f"{stem}.json"
    ply_path = args.out_dir / f"{stem}.ply"
    if npz_path.exists() or meta_path.exists() or ply_path.exists():
        print(f"Warning: overwriting existing files for angle {angle_deg:g}")

    np.savez_compressed(
        npz_path,
        points=points,
        colors=colors,
        depth_image_m=depth_m.astype(np.float32),
        color_image=color_rgb,
    )
    write_ascii_ply(ply_path, points, colors)
    meta_path.write_text(
        json.dumps(
            {
                "angle_deg": angle_deg,
                "radius_m": args.radius_m,
                "height_m": args.height_m,
                "min_depth_m": args.min_depth_m,
                "max_depth_m": args.max_depth_m,
                "resolution": f"{args.width}x{args.height}",
                "coordinate_system": "IMAGE",
                "camera_model": "OAK-D",
                "camera_intrinsics": intrinsics,
                "point_count": int(points.shape[0]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved angle {angle_deg:g} with {points.shape[0]} points")
    print(npz_path)
    print(meta_path)
    print(ply_path)


def prompt_for_angle() -> float | None:
    raw_value = input("Angle degrees to capture, or q to quit: ").strip()
    if raw_value.lower() in {"q", "quit", "exit"}:
        return None
    if not raw_value:
        return prompt_for_angle()
    return float(raw_value)


def latest_frames(rgb_queue, depth_queue, warmup: int):
    rgb_msg = None
    depth_msg = None
    for _ in range(max(warmup, 1)):
        rgb_msg = rgb_queue.get()
        depth_msg = depth_queue.get()
    return rgb_msg, depth_msg


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipeline, rgb_queue, depth_queue = create_pipeline(args)
    with pipeline:
        device = pipeline.getDefaultDevice()
        intrinsics = camera_intrinsics_from_device(device, args.width, args.height)
        pipeline.start()
        print("OAK-D pipeline started.")
        print(f"Depth/RGB output: {args.width}x{args.height}, coordinate_system=IMAGE")

        if args.angle_deg is not None:
            rgb_msg, depth_msg = latest_frames(rgb_queue, depth_queue, args.warmup)
            save_capture(args, args.angle_deg, depth_msg, rgb_msg, intrinsics)
            return

        print("Interactive capture mode. Type an angle and press Enter.")
        print("Type q to quit.")
        latest_frames(rgb_queue, depth_queue, args.warmup)
        while True:
            try:
                angle_deg = prompt_for_angle()
            except ValueError as exc:
                print(f"Invalid angle input: {exc}")
                continue
            if angle_deg is None:
                print("Capture session finished.")
                break
            rgb_msg, depth_msg = latest_frames(rgb_queue, depth_queue, 1)
            save_capture(args, angle_deg, depth_msg, rgb_msg, intrinsics)


if __name__ == "__main__":
    main()
