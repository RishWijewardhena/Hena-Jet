#!/usr/bin/env python3
"""Capture one ZED RGB-D and point cloud frame at a known scanner angle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyzed.sl as sl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one ZED point-cloud capture.

    --angle-deg: Required camera angle around the scanner center, in degrees.
        This tells the merge step where this capture sits on the circular path.
    --radius-m: Required distance from the scanner center to the camera, in meters.
        This is used to place the camera around the object during merging.
    --height-m: Optional vertical camera offset, in meters. The default is 0.0.
        Use this only if the camera moves up or down between captures.
    --out-dir: Optional output folder for the .npz point cloud and .json metadata.
        The default is captures/zed_m_first_scan.
    --max-depth-m: Optional far depth cutoff from the camera, in meters.
        Points farther than this are discarded before saving.
    --min-depth-m: Optional near depth cutoff from the camera, in meters.
        The ZED SDK may clamp this upward if the requested value is too close.
    --warmup: Optional number of camera frames to skip before saving.
        This helps the camera stabilize exposure and depth.
    --resolution: Optional ZED camera resolution. Allowed values are HD2K,
        HD1080, HD720, and VGA. The default is HD720.
    """
    parser = argparse.ArgumentParser(
        description="Capture one RGB/depth point cloud from a ZED camera."
    )
    parser.add_argument("--angle-deg", type=float, required=True)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--height-m", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("captures/zed_m_first_scan"))
    parser.add_argument("--max-depth-m", type=float, default=1.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD720")
    return parser.parse_args()


def resolution_from_name(name: str) -> sl.RESOLUTION:
    """Convert the user-facing resolution name into the ZED SDK enum value.

    argparse restricts the input to HD2K, HD1080, HD720, or VGA. The ZED SDK
    needs the matching sl.RESOLUTION enum instead of the string, so this helper
    performs that translation before camera initialization.
    """
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }[name]


def rgba_float_to_rgb(rgba_values: np.ndarray) -> np.ndarray:
    """Decode ZED XYZRGBA color values stored in the fourth float channel."""
    rgba_uint32 = rgba_values.view(np.uint32)
    r = rgba_uint32 & 0xFF
    g = (rgba_uint32 >> 8) & 0xFF
    b = (rgba_uint32 >> 16) & 0xFF
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def camera_intrinsics_from_zed(zed: sl.Camera, image_shape: tuple[int, int]) -> dict:
    """Read left-camera pinhole intrinsics from the ZED SDK.

    Open3D TSDF fusion needs fx, fy, cx, cy, width, and height so it can
    unproject every depth pixel into a 3D point. The width and height are taken
    from the actual retrieved image shape because that is the safest match for
    the arrays saved in the .npz file.
    """
    camera_info = zed.get_camera_information()
    calibration = camera_info.camera_configuration.calibration_parameters
    left_camera = calibration.left_cam
    height, width = image_shape
    return {
        "fx": float(left_camera.fx),
        "fy": float(left_camera.fy),
        "cx": float(left_camera.cx),
        "cy": float(left_camera.cy),
        "width": int(width),
        "height": int(height),
    }


def color_image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a ZED left camera image array into an RGB uint8 image.

    The ZED SDK usually returns a 4-channel color image. TSDF fusion only needs
    three 8-bit color channels, so this function drops alpha if present and
    keeps the image-shaped layout instead of flattening it.
    """
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        return image[:, :, :3].astype(np.uint8)
    raise ValueError(f"Unsupported color image shape: {image.shape}")


def clean_depth_image(
    depth_image: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Prepare a meter-scale depth image for Open3D RGB-D integration.

    Open3D treats zero depth as invalid. This function keeps finite depth
    values inside the requested ZED range and changes all invalid/background
    pixels to 0.0 while preserving the original image shape.
    """
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = depth[valid]
    return cleaned


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write one captured colored point cloud to an ASCII PLY preview file.

    The .npz file remains the raw processing format for merging, while this .ply
    file is a viewer-friendly copy that can be opened directly in MeshLab or
    CloudCompare to inspect one capture before running the merge step.
    """
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


def main() -> None:
    """Capture one filtered XYZRGBA point cloud and save RGB-D data plus metadata.

    The function opens the ZED camera with the requested resolution and depth
    limits, skips warmup frames, grabs one synchronized frame, saves the
    image-shaped depth/color arrays needed by Open3D TSDF fusion, and also
    keeps the older filtered point/color arrays plus a .ply preview for quick
    MeshLab inspection.
    """
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.camera_resolution = resolution_from_name(args.resolution)
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    runtime = sl.RuntimeParameters()
    point_cloud = sl.Mat()
    depth_mat = sl.Mat()
    color_mat = sl.Mat()

    try:
        for _ in range(args.warmup):
            zed.grab(runtime)

        status = zed.grab(runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Could not grab frame: {status}")

        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        zed.retrieve_image(color_mat, sl.VIEW.LEFT)
        cloud = point_cloud.get_data()
        depth_image = clean_depth_image(
            depth_mat.get_data(),
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        color_image = color_image_to_rgb(color_mat.get_data())
        intrinsics = camera_intrinsics_from_zed(zed, depth_image.shape)

        xyz = cloud[:, :, :3].reshape(-1, 3)
        rgba = cloud[:, :, 3].reshape(-1)

        finite = np.isfinite(xyz).all(axis=1)
        depth = xyz[:, 0]
        in_range = (depth >= args.min_depth_m) & (depth <= args.max_depth_m)
        valid = finite & in_range

        points = xyz[valid].astype(np.float32)
        colors = rgba_float_to_rgb(rgba[valid])

        stem = f"angle_{args.angle_deg:07.2f}".replace(".", "p")
        npz_path = args.out_dir / f"{stem}.npz"
        meta_path = args.out_dir / f"{stem}.json"
        ply_path = args.out_dir / f"{stem}.ply"

        np.savez_compressed(
            npz_path,
            points=points,
            colors=colors,
            depth_image_m=depth_image,
            color_image=color_image,
        )
        write_ascii_ply(ply_path, points, colors)
        meta_path.write_text(
            json.dumps(
                {
                    "angle_deg": args.angle_deg,
                    "radius_m": args.radius_m,
                    "height_m": args.height_m,
                    "min_depth_m": args.min_depth_m,
                    "max_depth_m": args.max_depth_m,
                    "resolution": args.resolution,
                    "coordinate_system": "RIGHT_HANDED_Z_UP_X_FWD",
                    "camera_intrinsics": intrinsics,
                    "point_count": int(points.shape[0]),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(f"Saved {points.shape[0]} points")
        print(npz_path)
        print(meta_path)
        print(ply_path)
    finally:
        zed.close()


if __name__ == "__main__":
    main()
