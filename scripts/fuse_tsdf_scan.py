#!/usr/bin/env python3
"""Fuse orbiting-camera RGB-D captures into a mesh with Open3D TSDF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Open3D ScalableTSDFVolume fusion.

    --capture-dir: Folder containing angle_*.json and matching angle_*.npz
        files from capture_zed_angle.py. The .npz files must include
        depth_image_m, color_image, points, and colors.
    --mesh-out: Output triangle mesh path. This is the main TSDF result.
    --cloud-out: Optional output point cloud extracted from the TSDF volume.
    --voxel-length-m: TSDF voxel size in meters. Smaller values keep more
        detail but increase noise and memory. Start around 0.002 for a small
        desktop object, then reduce only if the pose is already correct.
    --sdf-trunc-m: TSDF truncation distance in meters. A common value is about
        4 to 8 times voxel-length-m.
    --depth-trunc-m: Maximum depth Open3D integrates from each depth image.
    --max-depth-over-radius-m: Extra depth beyond the scanner radius. For
        radius 0.20 and value 0.08, pixels deeper than 0.28 m are ignored.
    --min-world-z and --max-world-z: Optional world-height crop before fusion.
        Use these to remove table/base points from the depth images.
    --max-world-radius-m: Optional XY crop around the scanner center before
        fusion. Use this to remove background outside the object area.
    --invert-angles and --angle-offset-deg: Correct the scanner angle
        convention without recapturing.
    --center-offset-x-m and --center-offset-y-m: Move the assumed scanner
        center if the object is not exactly on the rotation origin.
    --override-radius-m and --override-height-m: Replace the saved capture
        radius/height metadata during fusion. Use these when the physical
        measurement was taken from the camera body instead of the optical center.
    --camera-yaw-deg, --camera-pitch-deg, and --camera-roll-deg: Small camera
        mounting corrections applied after the ideal look-at-center pose.
    --heading-mode: Use look-at for a camera that turns to face the scanner
        center at every angle, or fixed for a camera whose heading does not
        rotate with the circular position.
    --fixed-heading-deg: World angle used by fixed heading mode.
    --open3d-camera-coords: Select the camera-axis convention used while
        converting Open3D RGB-D coordinates into the ZED scanner coordinates.
        image uses X right, Y down, Z forward. y-up uses X right, Y up,
        Z backward.
    """
    parser = argparse.ArgumentParser(
        description="Fuse ZED RGB-D captures into a mesh using Open3D TSDF."
    )
    parser.add_argument("--capture-dir", type=Path, default=Path("captures/zed_m_first_scan"))
    parser.add_argument("--mesh-out", type=Path, default=Path("outputs/zed_m_tsdf_mesh.ply"))
    parser.add_argument("--cloud-out", type=Path, default=None)
    parser.add_argument("--voxel-length-m", type=float, default=0.002)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.012)
    parser.add_argument("--depth-trunc-m", type=float, default=None)
    parser.add_argument("--max-depth-over-radius-m", type=float, default=0.08)
    parser.add_argument("--min-world-z", type=float, default=None)
    parser.add_argument("--max-world-z", type=float, default=None)
    parser.add_argument("--max-world-radius-m", type=float, default=None)
    parser.add_argument("--invert-angles", action="store_true")
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--center-offset-x-m", type=float, default=0.0)
    parser.add_argument("--center-offset-y-m", type=float, default=0.0)
    parser.add_argument("--override-radius-m", type=float, default=None)
    parser.add_argument("--override-height-m", type=float, default=None)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-roll-deg", type=float, default=0.0)
    parser.add_argument("--heading-mode", choices=["look-at", "fixed"], default="look-at")
    parser.add_argument("--fixed-heading-deg", type=float, default=0.0)
    parser.add_argument(
        "--open3d-camera-coords",
        choices=["image", "y-up"],
        default="image",
    )
    return parser.parse_args()


def rotation_x(degrees: float) -> np.ndarray:
    """Return a 3D rotation matrix around the camera/local X axis."""
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_value, -sin_value],
            [0.0, sin_value, cos_value],
        ],
        dtype=np.float64,
    )


def rotation_y(degrees: float) -> np.ndarray:
    """Return a 3D rotation matrix around the camera/local Y axis."""
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return np.array(
        [
            [cos_value, 0.0, sin_value],
            [0.0, 1.0, 0.0],
            [-sin_value, 0.0, cos_value],
        ],
        dtype=np.float64,
    )


def rotation_z(degrees: float) -> np.ndarray:
    """Return a 3D rotation matrix around the camera/local Z axis."""
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return np.array(
        [
            [cos_value, -sin_value, 0.0],
            [sin_value, cos_value, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def corrected_angle(angle_deg: float, invert_angles: bool, angle_offset_deg: float) -> float:
    """Apply angle-direction and zero-angle corrections to one capture angle."""
    signed_angle = -angle_deg if invert_angles else angle_deg
    return signed_angle + angle_offset_deg


def zed_from_open3d_axes(open3d_camera_coords: str) -> np.ndarray:
    """Convert selected Open3D camera axes into ZED camera-local axes.

    ZED point-cloud coordinates are +X forward, +Y left, +Z up. The image mode
    matches normal RGB-D image unprojection: +X right, +Y down, +Z forward. The
    y-up mode matches the Open3D/OpenGL-style camera convention: +X right,
    +Y up, +Z backward.
    """
    if open3d_camera_coords == "image":
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
    if open3d_camera_coords == "y-up":
        return np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported Open3D camera coordinate mode: {open3d_camera_coords}")


def camera_points_from_depth(depth: np.ndarray, meta: dict, open3d_camera_coords: str) -> np.ndarray:
    """Unproject a depth image into the selected Open3D camera coordinate mode.

    The ZED depth image stores forward distance per pixel. For image mode this
    becomes Open3D RGB-D style +Z forward and +Y down. For y-up mode we convert
    the same pixels into +Y up and +Z backward before applying the matching
    camera-to-world transform.
    """
    intrinsics = meta["camera_intrinsics"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    z_forward = depth
    x_right = (cols - cx) * z_forward / fx
    y_down = (rows - cy) * z_forward / fy

    if open3d_camera_coords == "image":
        camera_points = np.stack([x_right, y_down, z_forward], axis=-1)
    elif open3d_camera_coords == "y-up":
        camera_points = np.stack([x_right, -y_down, -z_forward], axis=-1)
    else:
        raise ValueError(f"Unsupported Open3D camera coordinate mode: {open3d_camera_coords}")
    return camera_points.reshape(-1, 3)


def scanner_radius(meta: dict, override_radius_m: float | None) -> float:
    """Return the radius used for fusion, allowing command-line calibration."""
    if override_radius_m is not None:
        return float(override_radius_m)
    return float(meta["radius_m"])


def scanner_height(meta: dict, override_height_m: float | None) -> float:
    """Return the camera height used for fusion, allowing command-line calibration."""
    if override_height_m is not None:
        return float(override_height_m)
    return float(meta["height_m"])


def camera_to_world_matrix(
    angle_deg: float,
    radius_m: float,
    height_m: float,
    invert_angles: bool = False,
    angle_offset_deg: float = 0.0,
    center_offset_x_m: float = 0.0,
    center_offset_y_m: float = 0.0,
    camera_yaw_deg: float = 0.0,
    camera_pitch_deg: float = 0.0,
    camera_roll_deg: float = 0.0,
    heading_mode: str = "look-at",
    fixed_heading_deg: float = 0.0,
    open3d_camera_coords: str = "image",
) -> np.ndarray:
    """Build the Open3D camera-to-world transform for one scanner angle.

    The scanner setup is a fixed object at the origin and a camera orbiting
    around table/world Z. Open3D RGB-D camera coordinates are +X right, +Y down,
    +Z forward. ZED point-cloud coordinates used elsewhere in this project are
    +X forward, +Y left, +Z up, so this function includes that axis conversion.
    The optional correction parameters let you tune real-world mounting errors
    without capturing the full scan again.
    """
    angle_deg = corrected_angle(angle_deg, invert_angles, angle_offset_deg)
    theta = math.radians(angle_deg)
    center = np.array([center_offset_x_m, center_offset_y_m, 0.0], dtype=np.float64)
    radial_out = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
    if heading_mode == "look-at":
        zed_forward = -radial_out
    elif heading_mode == "fixed":
        heading_theta = math.radians(fixed_heading_deg)
        zed_forward = np.array(
            [-math.cos(heading_theta), -math.sin(heading_theta), 0.0],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported heading mode: {heading_mode}")
    zed_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    zed_left = np.cross(zed_up, zed_forward)

    world_from_zed = np.column_stack([zed_forward, zed_left, zed_up])
    zed_from_open3d = zed_from_open3d_axes(open3d_camera_coords)

    transform = np.eye(4, dtype=np.float64)
    mount_correction = (
        rotation_z(camera_yaw_deg)
        @ rotation_y(camera_pitch_deg)
        @ rotation_x(camera_roll_deg)
    )
    transform[:3, :3] = world_from_zed @ mount_correction @ zed_from_open3d
    transform[:3, 3] = np.array(
        [radius_m * math.cos(theta), radius_m * math.sin(theta), height_m],
        dtype=np.float64,
    ) + center
    return transform


def load_capture(meta_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Load one RGB-D capture and fail clearly if it is an old point-only file."""
    npz_path = meta_path.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing capture data for {meta_path}: {npz_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    required = {"depth_image_m", "color_image"}
    missing = required.difference(data.files)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(
            f"{npz_path} is an old point-cloud-only capture. Missing: {missing_list}. "
            "Capture again with the updated capture_zed_angle.py before TSDF fusion."
        )
    if "camera_intrinsics" not in meta:
        raise RuntimeError(
            f"{meta_path} does not contain camera_intrinsics. "
            "Capture again with the updated capture_zed_angle.py before TSDF fusion."
        )

    return meta, data["depth_image_m"].astype(np.float32), data["color_image"].astype(np.uint8)


def make_intrinsic(meta: dict) -> o3d.camera.PinholeCameraIntrinsic:
    """Create the Open3D pinhole camera intrinsic object from capture metadata."""
    intrinsics = meta["camera_intrinsics"]
    return o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )


def apply_depth_filters(
    depth: np.ndarray,
    meta: dict,
    max_depth_over_radius_m: float,
    min_world_z: float | None,
    max_world_z: float | None,
    max_world_radius_m: float | None,
    invert_angles: bool,
    angle_offset_deg: float,
    center_offset_x_m: float,
    center_offset_y_m: float,
    override_radius_m: float | None,
    override_height_m: float | None,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    camera_roll_deg: float,
    heading_mode: str,
    fixed_heading_deg: float,
    open3d_camera_coords: str,
) -> np.ndarray:
    """Zero invalid depth pixels before they are integrated into the TSDF.

    TSDF fusion integrates every nonzero depth pixel. This function removes
    pixels that are too far from the camera or outside the scanner's world crop
    so background/table surfaces do not become part of the final mesh.
    """
    filtered = depth.copy()
    valid = np.isfinite(filtered) & (filtered > 0.0)

    if max_depth_over_radius_m > 0:
        max_camera_depth_m = scanner_radius(meta, override_radius_m) + max_depth_over_radius_m
        valid &= filtered <= max_camera_depth_m

    if min_world_z is not None or max_world_z is not None or max_world_radius_m is not None:
        camera_points = camera_points_from_depth(
            filtered,
            meta,
            open3d_camera_coords=open3d_camera_coords,
        )

        camera_to_world = camera_to_world_matrix(
            angle_deg=float(meta["angle_deg"]),
            radius_m=scanner_radius(meta, override_radius_m),
            height_m=scanner_height(meta, override_height_m),
            invert_angles=invert_angles,
            angle_offset_deg=angle_offset_deg,
            center_offset_x_m=center_offset_x_m,
            center_offset_y_m=center_offset_y_m,
            camera_yaw_deg=camera_yaw_deg,
            camera_pitch_deg=camera_pitch_deg,
            camera_roll_deg=camera_roll_deg,
            heading_mode=heading_mode,
            fixed_heading_deg=fixed_heading_deg,
            open3d_camera_coords=open3d_camera_coords,
        )
        rotation = camera_to_world[:3, :3]
        translation = camera_to_world[:3, 3]
        world_points = camera_points @ rotation.T + translation
        world_points = world_points.reshape(filtered.shape[0], filtered.shape[1], 3)

        if min_world_z is not None:
            valid &= world_points[:, :, 2] >= min_world_z
        if max_world_z is not None:
            valid &= world_points[:, :, 2] <= max_world_z
        if max_world_radius_m is not None:
            center_xy = np.array([center_offset_x_m, center_offset_y_m], dtype=np.float64)
            valid &= np.linalg.norm(world_points[:, :, :2] - center_xy, axis=2) <= max_world_radius_m

    filtered[~valid] = 0.0
    return filtered.astype(np.float32)


def make_rgbd(color: np.ndarray, depth: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    """Create an Open3D RGBDImage from uint8 color and meter-scale float depth."""
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def main() -> None:
    """Integrate all captures into a scalable TSDF volume and extract a mesh."""
    args = parse_args()
    meta_paths = sorted(args.capture_dir.glob("angle_*.json"))
    if not meta_paths:
        raise RuntimeError(f"No angle_*.json captures found in {args.capture_dir}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for index, meta_path in enumerate(meta_paths):
        meta, depth, color = load_capture(meta_path)
        depth = apply_depth_filters(
            depth,
            meta,
            max_depth_over_radius_m=args.max_depth_over_radius_m,
            min_world_z=args.min_world_z,
            max_world_z=args.max_world_z,
            max_world_radius_m=args.max_world_radius_m,
            invert_angles=args.invert_angles,
            angle_offset_deg=args.angle_offset_deg,
            center_offset_x_m=args.center_offset_x_m,
            center_offset_y_m=args.center_offset_y_m,
            override_radius_m=args.override_radius_m,
            override_height_m=args.override_height_m,
            camera_yaw_deg=args.camera_yaw_deg,
            camera_pitch_deg=args.camera_pitch_deg,
            camera_roll_deg=args.camera_roll_deg,
            heading_mode=args.heading_mode,
            fixed_heading_deg=args.fixed_heading_deg,
            open3d_camera_coords=args.open3d_camera_coords,
        )
        depth_trunc_m = (
            args.depth_trunc_m
            if args.depth_trunc_m is not None
            else scanner_radius(meta, args.override_radius_m)
            + max(args.max_depth_over_radius_m, 0.0)
        )
        rgbd = make_rgbd(color, depth, depth_trunc_m=depth_trunc_m)
        intrinsic = make_intrinsic(meta)
        camera_to_world = camera_to_world_matrix(
            angle_deg=float(meta["angle_deg"]),
            radius_m=scanner_radius(meta, args.override_radius_m),
            height_m=scanner_height(meta, args.override_height_m),
            invert_angles=args.invert_angles,
            angle_offset_deg=args.angle_offset_deg,
            center_offset_x_m=args.center_offset_x_m,
            center_offset_y_m=args.center_offset_y_m,
            camera_yaw_deg=args.camera_yaw_deg,
            camera_pitch_deg=args.camera_pitch_deg,
            camera_roll_deg=args.camera_roll_deg,
            heading_mode=args.heading_mode,
            fixed_heading_deg=args.fixed_heading_deg,
            open3d_camera_coords=args.open3d_camera_coords,
        )
        world_to_camera = np.linalg.inv(camera_to_world)
        volume.integrate(rgbd, intrinsic, world_to_camera)
        valid_pixels = int(np.count_nonzero(depth))
        print(f"Integrated {index + 1}/{len(meta_paths)} {meta_path.name}: {valid_pixels} depth pixels")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
    print(f"Saved TSDF mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    print(args.mesh_out)

    if args.cloud_out is not None:
        cloud = volume.extract_point_cloud()
        args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.cloud_out), cloud)
        print(f"Saved TSDF cloud with {len(cloud.points)} points")
        print(args.cloud_out)


if __name__ == "__main__":
    main()
