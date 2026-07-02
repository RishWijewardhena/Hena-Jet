#!/usr/bin/env python3
"""Capture a fixed number of colored point clouds from a ZED Mini camera."""

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
        description="Save colored ZED Mini XYZRGB point clouds as numbered PLY files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("captures/zed_mini_pointclouds"),
        help="Folder where 000000.ply, 000001.ply, ... will be saved.",
    )
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.15,
        help="Delay between saved point clouds.",
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
        default="NEURAL_LIGHT" if "NEURAL_LIGHT" in DEPTH_MODES else "QUALITY",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write binary PLY files. Smaller and faster, but less human-readable.",
    )
    parser.add_argument(
        "--max-grab-failures",
        type=int,
        default=20,
        help="Stop if this many capture grabs fail before saving all frames.",
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
    if args.max_grab_failures <= 0:
        raise ValueError("--max-grab-failures must be positive")


def rgba_float_to_rgb(rgba_values: np.ndarray) -> np.ndarray:
    rgba_uint32 = rgba_values.view(np.uint32)
    blue = rgba_uint32 & 0xFF
    green = (rgba_uint32 >> 8) & 0xFF
    red = (rgba_uint32 >> 16) & 0xFF
    return np.stack([red, green, blue], axis=1).astype(np.float64) / 255.0


def pointcloud_from_zed_mat(
    cloud_mat: sl.Mat,
    min_depth_m: float,
    max_depth_m: float,
) -> o3d.geometry.PointCloud:
    cloud = cloud_mat.get_data()
    xyz = np.asarray(cloud[:, :, :3], dtype=np.float32).reshape(-1, 3)
    rgba = np.asarray(cloud[:, :, 3], dtype=np.float32).reshape(-1)

    depth = xyz[:, 2]
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(rgba)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[valid].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgba_float_to_rgb(rgba[valid]))
    return pcd


def write_session_metadata(path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "camera": "ZED Mini",
        "frames": args.frames,
        "interval_s": args.interval_s,
        "warmup": args.warmup,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "resolution": args.resolution,
        "depth_mode": args.depth_mode,
        "max_grab_failures": args.max_grab_failures,
        "coordinate_system": "IMAGE",
        "coordinate_units": "METER",
        "file_pattern": "000000.ply",
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[args.resolution]
    init.depth_mode = DEPTH_MODES[args.depth_mode]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED Mini camera: {status}")

    runtime = sl.RuntimeParameters()
    cloud_mat = sl.Mat()

    try:
        for index in range(args.warmup):
            status = zed.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Warmup frame {index + 1} failed: {status}")

        write_session_metadata(args.out_dir / "session.json", args)

        saved = 0
        grab_failures = 0
        while saved < args.frames:
            status = zed.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                grab_failures += 1
                print(f"Grab failed {grab_failures}/{args.max_grab_failures}: {status}")
                if grab_failures >= args.max_grab_failures:
                    raise RuntimeError("Too many ZED grab failures")
                continue

            grab_failures = 0
            zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
            pcd = pointcloud_from_zed_mat(
                cloud_mat,
                args.min_depth_m,
                args.max_depth_m,
            )

            ply_path = args.out_dir / f"{saved:06d}.ply"
            o3d.io.write_point_cloud(
                str(ply_path),
                pcd,
                write_ascii=not args.binary,
                compressed=False,
            )
            print(f"saved {ply_path} ({len(pcd.points)} points)")

            saved += 1
            if args.interval_s:
                time.sleep(args.interval_s)
    finally:
        zed.close()

    print(f"\nSaved point clouds to: {args.out_dir}")
    print(f"Session metadata: {args.out_dir / 'session.json'}")


if __name__ == "__main__":
    main()
