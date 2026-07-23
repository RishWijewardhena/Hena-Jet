#!/usr/bin/env python3
"""Fuse orbiting-camera RGB-D captures into a mesh with Open3D TSDF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import open3d as o3d


MULTILEVEL_PASS_DIRECTORY = re.compile(r"pass_(\d+)_height_(\d+)mm$")


def discover_capture_paths(capture_dir: Path) -> list[Path]:
    """Discover legacy flat captures or ordered multilevel pass captures."""
    capture_dir = Path(capture_dir)
    flat_paths = list(capture_dir.glob("angle_*.json"))
    multilevel_paths = list(
        capture_dir.glob("pass_*_height_*mm/angle_*.json")
    )
    if flat_paths and multilevel_paths:
        raise RuntimeError(
            "Capture directory mixes flat and multilevel captures; separate the datasets"
        )
    if flat_paths:
        return sorted(flat_paths)

    def multilevel_key(path: Path) -> tuple[int, int, int, str]:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        directory_match = MULTILEVEL_PASS_DIRECTORY.fullmatch(path.parent.name)
        if directory_match is None:
            raise RuntimeError(f"Invalid multilevel pass directory: {path.parent}")
        return (
            int(metadata.get("multilevel_pass_index", directory_match.group(1))),
            int(metadata.get("multilevel_capture_index", 2**63 - 1)),
            int(metadata.get("timestamp_ns", 0)),
            path.name,
        )

    return sorted(multilevel_paths, key=multilevel_key)


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
    The camera is assumed to stay level and face the scanner center while it
    moves around the pillar. Captures use IMAGE coordinates: +X right, +Y down,
    +Z forward.
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
    parser.add_argument("--pose-source", choices=["orbit", "vslam"], default="orbit")
    parser.add_argument("--max-depth-confidence", type=int, default=100)
    parser.add_argument("--object-center-m", type=float, nargs=3, default=None)
    parser.add_argument("--object-up", type=float, nargs=3, default=None)
    parser.add_argument("--max-object-radius-m", type=float, default=None)
    parser.add_argument("--min-object-height-m", type=float, default=None)
    parser.add_argument("--max-object-height-m", type=float, default=None)
    parser.add_argument("--invert-angles", action="store_true")
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--camera-yaw-deg",
        type=float,
        default=0.0,
        help="Constant camera mounting yaw relative to the inward radial direction.",
    )
    parser.add_argument("--center-offset-x-m", type=float, default=0.0)
    parser.add_argument("--center-offset-y-m", type=float, default=0.0)
    parser.add_argument("--override-radius-m", type=float, default=None)
    parser.add_argument("--override-height-m", type=float, default=None)
    return parser.parse_args()


def corrected_angle(angle_deg: float, invert_angles: bool, angle_offset_deg: float) -> float:
    """Apply angle-direction and zero-angle corrections to one capture angle."""
    signed_angle = -angle_deg if invert_angles else angle_deg
    return signed_angle + angle_offset_deg


def camera_points_from_depth(depth: np.ndarray, meta: dict) -> np.ndarray:
    """Unproject an IMAGE-coordinate depth map into camera-local 3D points."""
    intrinsics = meta["camera_intrinsics"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    z_forward = depth
    x_right = (cols - cx) * z_forward / fx
    y_down = (rows - cy) * z_forward / fy

    return np.stack([x_right, y_down, z_forward], axis=-1).reshape(-1, 3)


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
    camera_yaw_deg: float = 0.0,
    center_offset_x_m: float = 0.0,
    center_offset_y_m: float = 0.0,
) -> np.ndarray:
    """Build a level, inward-facing IMAGE camera pose for one scanner angle.

    The scanner setup is a fixed object at the origin and a camera orbiting
    around world Z. The camera stays horizontal and its IMAGE +Z axis always
    points toward the scanner center. IMAGE +X points right and +Y points down.
    """
    angle_deg = corrected_angle(angle_deg, invert_angles, angle_offset_deg)
    theta = math.radians(angle_deg)
    center = np.array([center_offset_x_m, center_offset_y_m, 0.0], dtype=np.float64)
    radial_out = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
    camera_forward = -radial_out
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_right = np.cross(camera_forward, world_up)
    camera_down = -world_up

    transform = np.eye(4, dtype=np.float64)
    camera_to_world_rotation = np.column_stack(
        [camera_right, camera_down, camera_forward]
    )
    yaw = math.radians(camera_yaw_deg)
    local_yaw = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    transform[:3, :3] = camera_to_world_rotation @ local_yaw
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


def load_confidence_image(meta_path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """Load an optional confidence map while remaining compatible with old captures."""
    data = np.load(meta_path.with_suffix(".npz"))
    if "confidence_image" not in data.files:
        return None
    confidence = data["confidence_image"].astype(np.uint8)
    if confidence.shape != expected_shape:
        raise RuntimeError(
            f"{meta_path}: confidence shape {confidence.shape} does not match depth {expected_shape}"
        )
    return confidence


def depth_confidence_mask(
    confidence: np.ndarray,
    maximum_confidence: int,
) -> np.ndarray:
    """Keep ZED confidence values at or below the 0-best/100-worst limit."""
    if not 0 <= maximum_confidence <= 100:
        raise ValueError("--max-depth-confidence must be between 0 and 100")
    return np.asarray(confidence) <= maximum_confidence


def object_crop_mask(
    world_points: np.ndarray,
    *,
    center: np.ndarray,
    up: np.ndarray,
    min_height_m: float | None,
    max_height_m: float | None,
    max_radius_m: float | None,
) -> np.ndarray:
    """Return an object-relative cylindrical crop mask in any world frame."""
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if center.shape != (3,) or up.shape != (3,) or norm == 0.0:
        raise ValueError("Object center and up must be finite three-vectors")
    up = up / norm
    relative = np.asarray(world_points, dtype=np.float64) - center
    height = relative @ up
    radial = relative - height[..., None] * up
    mask = np.ones(height.shape, dtype=bool)
    if min_height_m is not None:
        mask &= height >= min_height_m
    if max_height_m is not None:
        mask &= height <= max_height_m
    if max_radius_m is not None:
        mask &= np.linalg.norm(radial, axis=-1) <= max_radius_m
    return mask


def apply_depth_filters(
    depth: np.ndarray,
    meta: dict,
    radius_m: float,
    camera_to_world: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Zero invalid depth pixels before they are integrated into the TSDF.

    TSDF fusion integrates every nonzero depth pixel. This function removes
    pixels that are too far from the camera or outside the scanner's world crop
    so background/table surfaces do not become part of the final mesh.
    """
    filtered = depth.copy()
    valid = np.isfinite(filtered) & (filtered > 0.0)

    if args.max_depth_over_radius_m > 0:
        valid &= filtered <= radius_m + args.max_depth_over_radius_m

    world_crop_requested = any(
        value is not None
        for value in (
            args.min_world_z,
            args.max_world_z,
            args.max_world_radius_m,
            getattr(args, "min_object_height_m", None),
            getattr(args, "max_object_height_m", None),
            getattr(args, "max_object_radius_m", None),
        )
    )
    if world_crop_requested:
        camera_points = camera_points_from_depth(filtered, meta)
        rotation = camera_to_world[:3, :3]
        translation = camera_to_world[:3, 3]
        world_points = camera_points @ rotation.T + translation
        world_points = world_points.reshape(*filtered.shape, 3)

        if args.min_world_z is not None:
            valid &= world_points[:, :, 2] >= args.min_world_z
        if args.max_world_z is not None:
            valid &= world_points[:, :, 2] <= args.max_world_z
        if args.max_world_radius_m is not None:
            center_xy = np.array(
                [args.center_offset_x_m, args.center_offset_y_m],
                dtype=np.float64,
            )
            distance_from_center = np.linalg.norm(
                world_points[:, :, :2] - center_xy,
                axis=2,
            )
            valid &= distance_from_center <= args.max_world_radius_m

        object_center = getattr(args, "object_center_m", None)
        object_up = getattr(args, "object_up", None)
        if object_center is not None and object_up is not None:
            valid &= object_crop_mask(
                world_points,
                center=np.asarray(object_center),
                up=np.asarray(object_up),
                min_height_m=getattr(args, "min_object_height_m", None),
                max_height_m=getattr(args, "max_object_height_m", None),
                max_radius_m=getattr(args, "max_object_radius_m", None),
            )

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


def capture_pose(meta: dict, args: argparse.Namespace) -> tuple[float, np.ndarray]:
    """Return the calibrated scanner radius and camera-to-world transform."""
    radius_m = scanner_radius(meta, args.override_radius_m)
    if getattr(args, "pose_source", "orbit") == "vslam":
        if "camera_to_vslam_world" not in meta:
            raise RuntimeError("VSLAM pose source requested but capture metadata has no pose")
        camera_to_world = np.asarray(meta["camera_to_vslam_world"], dtype=np.float64)
        if camera_to_world.shape != (4, 4) or not np.all(np.isfinite(camera_to_world)):
            raise RuntimeError("Invalid camera_to_vslam_world matrix in capture metadata")
        return radius_m, camera_to_world
    camera_to_world = camera_to_world_matrix(
        angle_deg=float(meta["angle_deg"]),
        radius_m=radius_m,
        height_m=scanner_height(meta, args.override_height_m),
        invert_angles=args.invert_angles,
        angle_offset_deg=args.angle_offset_deg,
        camera_yaw_deg=getattr(args, "camera_yaw_deg", 0.0),
        center_offset_x_m=args.center_offset_x_m,
        center_offset_y_m=args.center_offset_y_m,
    )
    return radius_m, camera_to_world


def configure_object_frame(args: argparse.Namespace) -> None:
    """Resolve the object frame from CLI values or the VSLAM session manifest."""
    center = args.object_center_m
    up = args.object_up
    session_path = args.capture_dir / "scan_session.json"
    if (center is None or up is None) and session_path.exists():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        if center is None:
            center = session.get("object_center_vslam_world_m")
        if up is None:
            up = session.get("object_up_vslam_world")
    object_crop_requested = any(
        value is not None
        for value in (
            args.min_object_height_m,
            args.max_object_height_m,
            args.max_object_radius_m,
        )
    )
    if object_crop_requested and (center is None or up is None):
        raise RuntimeError(
            "Object-relative crop requested without --object-center-m/--object-up "
            "or a scan_session.json"
        )
    args.object_center_m = None if center is None else np.asarray(center, dtype=np.float64)
    args.object_up = None if up is None else np.asarray(up, dtype=np.float64)


def integrate_capture(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    meta_path: Path,
    args: argparse.Namespace,
) -> int:
    """Load, filter, and integrate one capture; return its valid pixel count."""
    meta, depth, color = load_capture(meta_path)
    confidence = load_confidence_image(meta_path, depth.shape)
    if confidence is None:
        if args.max_depth_confidence < 100:
            raise RuntimeError(
                f"{meta_path}: --max-depth-confidence requires confidence_image"
            )
    else:
        depth = depth.copy()
        depth[~depth_confidence_mask(confidence, args.max_depth_confidence)] = 0.0
    radius_m, camera_to_world = capture_pose(meta, args)
    depth = apply_depth_filters(depth, meta, radius_m, camera_to_world, args)

    depth_trunc_m = args.depth_trunc_m
    if depth_trunc_m is None:
        depth_trunc_m = radius_m + max(args.max_depth_over_radius_m, 0.0)

    volume.integrate(
        make_rgbd(color, depth, depth_trunc_m),
        make_intrinsic(meta),
        np.linalg.inv(camera_to_world),
    )
    return int(np.count_nonzero(depth))


def save_outputs(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    args: argparse.Namespace,
) -> None:
    """Extract and save the requested mesh and optional point cloud."""

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


def main() -> None:
    """Integrate all captures into a scalable TSDF volume and save the result."""
    args = parse_args()
    configure_object_frame(args)
    if not 0 <= args.max_depth_confidence <= 100:
        raise ValueError("--max-depth-confidence must be between 0 and 100")
    meta_paths = discover_capture_paths(args.capture_dir)
    if not meta_paths:
        raise RuntimeError(f"No angle_*.json captures found in {args.capture_dir}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for index, meta_path in enumerate(meta_paths, start=1):
        valid_pixels = integrate_capture(volume, meta_path, args)
        print(
            f"Integrated {index}/{len(meta_paths)} {meta_path.name}: "
            f"{valid_pixels} depth pixels"
        )

    save_outputs(volume, args)


if __name__ == "__main__":
    main()
