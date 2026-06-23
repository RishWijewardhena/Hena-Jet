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

    --angle-deg: Optional camera angle around the scanner center, in degrees.
        If omitted, the script stays open and prompts for angles until you type q.
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
    --angle-warmup: Optional number of extra frames to grab between interactive
        captures. This flushes stale buffered frames before each new angle.
    --resolution: Optional ZED camera resolution. Allowed values are HD2K,
        HD1080, HD720, and VGA. The default is HD720.
    --coordinate-system: ZED point-cloud coordinate system. IMAGE is the ZED
        image/depth convention: +X right, +Y down, +Z forward.
    """
    parser = argparse.ArgumentParser(
        description="Capture one RGB/depth point cloud from a ZED camera."
    )
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--height-m", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("captures/zed_m_first_scan"))
    parser.add_argument("--max-depth-m", type=float, default=1.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--angle-warmup", type=int, default=3)
    parser.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD720")
    parser.add_argument(
        "--coordinate-system",
        choices=["IMAGE", "RIGHT_HANDED_Z_UP_X_FWD"],
        default="IMAGE",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments and raise early with a clear message on errors.

    FIX: Added validation for radius_m. A zero or negative radius would silently
    produce garbage camera transforms in the merge step.
    """
    if args.radius_m <= 0:
        raise ValueError(f"--radius-m must be positive, got {args.radius_m}")
    if args.min_depth_m <= 0:
        raise ValueError(f"--min-depth-m must be positive, got {args.min_depth_m}")
    if args.max_depth_m <= args.min_depth_m:
        raise ValueError(
            f"--max-depth-m ({args.max_depth_m}) must be greater than "
            f"--min-depth-m ({args.min_depth_m})"
        )


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


def coordinate_system_from_name(name: str) -> sl.COORDINATE_SYSTEM:
    """Convert the user-facing coordinate-system name into a ZED SDK enum."""
    return {
        "IMAGE": sl.COORDINATE_SYSTEM.IMAGE,
        "RIGHT_HANDED_Z_UP_X_FWD": sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
    }[name]


def forward_depth_from_points(points: np.ndarray, coordinate_system: str) -> np.ndarray:
    """Return camera-forward depth from XYZ points for the selected ZED axes."""
    if coordinate_system == "IMAGE":
        return points[:, 2]
    if coordinate_system == "RIGHT_HANDED_Z_UP_X_FWD":
        return points[:, 0]
    raise ValueError(f"Unsupported coordinate system: {coordinate_system}")


def rgba_float_to_rgb(rgba_values: np.ndarray) -> np.ndarray:
    """Decode ZED XYZRGBA color values stored in the fourth float channel.

    FIX: The ZED SDK packs color as BGRA, not RGBA. The original code labelled
    the extracted bytes as R, G, B but was actually reading B, G, R order,
    producing swapped red and blue channels in all saved point clouds.
    Corrected byte extraction:
        byte 0 (& 0xFF)        → Blue
        byte 1 (>> 8  & 0xFF)  → Green
        byte 2 (>> 16 & 0xFF)  → Red
    """
    rgba_uint32 = rgba_values.view(np.uint32)
    b = rgba_uint32 & 0xFF
    g = (rgba_uint32 >> 8) & 0xFF
    r = (rgba_uint32 >> 16) & 0xFF
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

    FIX: The ZED SDK returns images in BGRA channel order. The original code
    took the first three channels directly, which produced a BGR image instead
    of RGB. The channel order is now reversed (::-1) to give correct RGB output
    for Open3D TSDF fusion and any downstream viewer.
    """
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        # ZED returns BGRA — take first 3 channels then reverse to get RGB
        return image[:, :, :3][:, :, ::-1].astype(np.uint8)
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


def flush_frames(zed: sl.Camera, runtime: sl.RuntimeParameters, count: int) -> None:
    """Grab and discard a number of frames to flush the ZED buffer.

    FIX: In interactive capture mode the camera runs continuously between angle
    inputs. Without flushing, the grabbed frame may be stale (captured before
    the user finished positioning the scanner). Discarding a small number of
    frames gives the sensor time to settle at each new angle.
    """
    for i in range(count):
        status = zed.grab(runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Frame flush {i + 1}/{count} failed: {status}")


def capture_angle(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    point_cloud: sl.Mat,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    args: argparse.Namespace,
    angle_deg: float,
    *,
    flush: bool = False,
) -> None:
    """Capture and save one RGB-D point cloud at the requested scanner angle.

    Parameters
    ----------
    flush:
        When True, grab and discard ``args.angle_warmup`` frames before the
        real capture. Set this in interactive mode so stale buffered frames
        accumulated while the user was repositioning the scanner are discarded.
    """
    if flush:
        flush_frames(zed, runtime, args.angle_warmup)

    status = zed.grab(runtime)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not grab frame at angle {angle_deg}: {status}")

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
    depth = forward_depth_from_points(xyz, args.coordinate_system)
    in_range = (depth >= args.min_depth_m) & (depth <= args.max_depth_m)
    valid = finite & in_range

    points = xyz[valid].astype(np.float32)
    colors = rgba_float_to_rgb(rgba[valid])

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
        depth_image_m=depth_image,
        color_image=color_image,
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
                "resolution": args.resolution,
                "coordinate_system": args.coordinate_system,
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
    """Read one angle from stdin; q exits the interactive loop.

    FIX: The original implementation called itself recursively on empty input,
    which would eventually hit Python's recursion limit if the user kept pressing
    Enter. Replaced with an explicit while loop that retries until valid input or
    a quit command is received.
    """
    while True:
        raw_value = input("Angle degrees to capture, or q to quit: ").strip()
        if raw_value.lower() in {"q", "quit", "exit"}:
            return None
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except ValueError:
            print(f"Invalid input '{raw_value}'. Please enter a number or q.")


def main() -> None:
    """Capture one filtered XYZRGBA point cloud and save RGB-D data plus metadata.

    The function opens the ZED camera with the requested resolution and depth
    limits, skips warmup frames, grabs one synchronized frame, saves the
    image-shaped depth/color arrays needed by Open3D TSDF fusion, and also
    keeps the older filtered point/color arrays plus a .ply preview for quick
    MeshLab inspection.
    """
    args = parse_args()
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.camera_resolution = resolution_from_name(args.resolution)
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = coordinate_system_from_name(args.coordinate_system)
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
        for index in range(args.warmup):
            status = zed.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Warmup frame {index + 1} failed: {status}")

        if args.angle_deg is not None:
            # Single-angle mode: warmup already done above, no extra flush needed
            capture_angle(
                zed,
                runtime,
                point_cloud,
                depth_mat,
                color_mat,
                args,
                args.angle_deg,
                flush=False,
            )
            return

        print("Interactive capture mode. Type an angle and press Enter.")
        print("Type q to quit.")
        while True:
            angle_deg = prompt_for_angle()
            if angle_deg is None:
                print("Capture session finished.")
                break
            # FIX: flush stale frames accumulated while the user was positioning
            capture_angle(
                zed,
                runtime,
                point_cloud,
                depth_mat,
                color_mat,
                args,
                angle_deg,
                flush=True,
            )
    finally:
        zed.close()


if __name__ == "__main__":
    main()