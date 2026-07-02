#!/usr/bin/env python3
"""Fuse saved OAK-D angle captures into a TSDF mesh using circular poses."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_CAPTURE_DIR = Path("captures/oakd_scan")
DEFAULT_MIN_DEPTH_M = 0.10
DEFAULT_MAX_DEPTH_M = 0.35
DEFAULT_VOXEL_LENGTH_M = 0.002
DEFAULT_SDF_TRUNC_M = 0.012
DEFAULT_MIN_VALID_DEPTH_PX = 1000


@dataclass
class Capture:
    npz_path: Path
    meta_path: Path
    meta: dict


@dataclass
class IntegratedCapture:
    index: int
    source: str
    angle_deg: float
    corrected_angle_deg: float
    radius_m: float
    height_m: float
    valid_depth_px: int
    pose: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline TSDF fusion for saved OAK-D circular angle captures."
    )
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--min-depth-m", type=float, default=DEFAULT_MIN_DEPTH_M)
    parser.add_argument("--max-depth-m", type=float, default=DEFAULT_MAX_DEPTH_M)
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=(0.0, 0.0, 1.0, 1.0),
        help="Normalized crop applied to both color and depth.",
    )
    parser.add_argument("--voxel-length-m", type=float, default=DEFAULT_VOXEL_LENGTH_M)
    parser.add_argument("--sdf-trunc-m", type=float, default=DEFAULT_SDF_TRUNC_M)
    parser.add_argument("--min-valid-depth-px", type=int, default=DEFAULT_MIN_VALID_DEPTH_PX)
    parser.add_argument(
        "--override-radius-m",
        type=float,
        default=None,
        help="Replace radius_m from metadata if physical measurement needs correction.",
    )
    parser.add_argument(
        "--override-height-m",
        type=float,
        default=None,
        help="Replace height_m from metadata if physical measurement needs correction.",
    )
    parser.add_argument("--invert-angles", action="store_true")
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--center-offset-x-m", type=float, default=0.0)
    parser.add_argument("--center-offset-y-m", type=float, default=0.0)
    parser.add_argument(
        "--skip-duplicate-360",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip 360-degree captures when a 0-degree capture is present.",
    )
    parser.add_argument(
        "--mesh-out",
        type=Path,
        default=Path("outputs/oakd_offline_circle_tsdf_mesh.ply"),
    )
    parser.add_argument(
        "--cloud-out",
        type=Path,
        default=Path("outputs/oakd_offline_circle_tsdf_cloud.ply"),
    )
    parser.add_argument(
        "--poses-out",
        type=Path,
        default=Path("outputs/oakd_offline_circle_poses.npy"),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("outputs/oakd_offline_circle_keyframes.json"),
    )
    return parser.parse_args()


def validate_roi(roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = roi
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError("--roi must satisfy 0<=X_MIN<X_MAX<=1 and 0<=Y_MIN<Y_MAX<=1")
    return roi


def validate_args(args: argparse.Namespace) -> None:
    args.roi = validate_roi(tuple(args.roi))
    if not args.capture_dir.exists():
        raise FileNotFoundError(f"Capture directory does not exist: {args.capture_dir}")
    if args.min_depth_m <= 0.0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m must be greater than --min-depth-m")
    if args.voxel_length_m <= 0.0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")
    if args.min_valid_depth_px < 0:
        raise ValueError("--min-valid-depth-px must be zero or positive")
    if args.override_radius_m is not None and args.override_radius_m <= 0.0:
        raise ValueError("--override-radius-m must be positive")


def load_captures(capture_dir: Path, skip_duplicate_360: bool) -> list[Capture]:
    captures: list[Capture] = []
    for meta_path in sorted(capture_dir.glob("angle_*.json")):
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
        npz_path = meta_path.with_suffix(".npz")
        if not npz_path.exists():
            print(f"Skipping {meta_path.name}: missing {npz_path.name}")
            continue
        captures.append(Capture(npz_path=npz_path, meta_path=meta_path, meta=meta))

    if skip_duplicate_360 and any(abs(float(c.meta["angle_deg"])) < 1e-6 for c in captures):
        filtered: list[Capture] = []
        for capture in captures:
            angle_mod = float(capture.meta["angle_deg"]) % 360.0
            if abs(angle_mod) < 1e-6 and abs(float(capture.meta["angle_deg"])) > 1e-6:
                print(f"Skipping duplicate full-turn capture: {capture.meta_path.name}")
                continue
            filtered.append(capture)
        captures = filtered

    if not captures:
        raise RuntimeError(f"No angle_*.json/.npz capture pairs found in {capture_dir}")
    return sorted(captures, key=lambda item: float(item.meta["angle_deg"]))


def intrinsic_from_meta(meta: dict) -> o3d.camera.PinholeCameraIntrinsic:
    intrinsics = meta["camera_intrinsics"]
    return o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )


def intrinsic_to_dict(intrinsic: o3d.camera.PinholeCameraIntrinsic) -> dict:
    matrix = intrinsic.intrinsic_matrix
    return {
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
    }


def apply_roi_mask(
    image: np.ndarray,
    roi: tuple[float, float, float, float],
    fill_value: float | int = 0,
) -> np.ndarray:
    if roi == (0.0, 0.0, 1.0, 1.0):
        return image
    x_min, y_min, x_max, y_max = roi
    h, w = image.shape[:2]
    x0 = int(round(x_min * w))
    x1 = int(round(x_max * w))
    y0 = int(round(y_min * h))
    y1 = int(round(y_max * h))
    result = np.full_like(image, fill_value)
    result[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    return result


def clean_depth_image(
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = depth[valid]
    return apply_roi_mask(cleaned, roi, fill_value=0)


def corrected_angle(angle_deg: float, invert_angles: bool, angle_offset_deg: float) -> float:
    signed = -angle_deg if invert_angles else angle_deg
    return signed + angle_offset_deg


def camera_pose_from_circle(
    angle_deg: float,
    radius_m: float,
    height_m: float,
    center_offset_x_m: float,
    center_offset_y_m: float,
) -> np.ndarray:
    """Return camera-to-world pose for Open3D image camera axes.

    Camera-local axes are Open3D pinhole image coordinates:
        +X right, +Y down, +Z forward.
    """
    theta = math.radians(angle_deg)
    center = np.array([center_offset_x_m, center_offset_y_m, 0.0], dtype=np.float64)
    radial_out = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
    forward = -radial_out
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, radial_out)
    right /= np.linalg.norm(right)
    down = -world_up

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = center + np.array(
        [radius_m * math.cos(theta), radius_m * math.sin(theta), height_m],
        dtype=np.float64,
    )
    return pose


def make_rgbd(color: np.ndarray, depth: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def make_tsdf_volume(args: argparse.Namespace) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def radius_for_capture(meta: dict, args: argparse.Namespace) -> float:
    if args.override_radius_m is not None:
        return float(args.override_radius_m)
    return float(meta["radius_m"])


def height_for_capture(meta: dict, args: argparse.Namespace) -> float:
    if args.override_height_m is not None:
        return float(args.override_height_m)
    return float(meta["height_m"])


def integrate_capture(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    capture: Capture,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
    index: int,
) -> IntegratedCapture | None:
    with np.load(capture.npz_path) as data:
        depth = clean_depth_image(
            data["depth_image_m"],
            args.min_depth_m,
            args.max_depth_m,
            args.roi,
        )
        color = apply_roi_mask(data["color_image"].astype(np.uint8), args.roi, fill_value=0)

    valid_depth_px = int(np.count_nonzero(depth))
    if valid_depth_px < args.min_valid_depth_px:
        print(
            f"Skipping {capture.npz_path.name}: valid_depth_px={valid_depth_px} "
            f"< {args.min_valid_depth_px}"
        )
        return None

    angle_deg = float(capture.meta["angle_deg"])
    fixed_angle = corrected_angle(angle_deg, args.invert_angles, args.angle_offset_deg)
    radius_m = radius_for_capture(capture.meta, args)
    height_m = height_for_capture(capture.meta, args)
    pose = camera_pose_from_circle(
        fixed_angle,
        radius_m,
        height_m,
        args.center_offset_x_m,
        args.center_offset_y_m,
    )
    rgbd = make_rgbd(color, depth, args.max_depth_m)
    volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    return IntegratedCapture(
        index=index,
        source=str(capture.npz_path),
        angle_deg=angle_deg,
        corrected_angle_deg=fixed_angle,
        radius_m=radius_m,
        height_m=height_m,
        valid_depth_px=valid_depth_px,
        pose=pose,
    )


def save_outputs(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    integrated: list[IntegratedCapture],
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> None:
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
    print(f"Mesh: {args.mesh_out}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")

    cloud = volume.extract_point_cloud()
    args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.cloud_out), cloud)
    print(f"Cloud: {args.cloud_out}  ({len(cloud.points)} pts)")

    poses = np.stack([item.pose for item in integrated]) if integrated else np.empty((0, 4, 4))
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)
    print(f"Poses: {args.poses_out}  ({len(poses)} poses)")

    metadata = {
        "capture_dir": str(args.capture_dir),
        "coordinate_system": "Open3D image coordinates: +X right, +Y down, +Z forward",
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "voxel_length_m": args.voxel_length_m,
        "sdf_trunc_m": args.sdf_trunc_m,
        "roi": list(args.roi),
        "override_radius_m": args.override_radius_m,
        "override_height_m": args.override_height_m,
        "invert_angles": args.invert_angles,
        "angle_offset_deg": args.angle_offset_deg,
        "center_offset_x_m": args.center_offset_x_m,
        "center_offset_y_m": args.center_offset_y_m,
        "skip_duplicate_360": args.skip_duplicate_360,
        "camera_intrinsics": intrinsic_to_dict(intrinsic),
        "keyframes": [
            {
                "index": item.index,
                "source": item.source,
                "angle_deg": item.angle_deg,
                "corrected_angle_deg": item.corrected_angle_deg,
                "radius_m": item.radius_m,
                "height_m": item.height_m,
                "valid_depth_px": item.valid_depth_px,
                "pose": item.pose.tolist(),
            }
            for item in integrated
        ],
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Metadata: {args.metadata_out}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    captures = load_captures(args.capture_dir, args.skip_duplicate_360)
    intrinsic = intrinsic_from_meta(captures[0].meta)
    volume = make_tsdf_volume(args)
    integrated: list[IntegratedCapture] = []

    print(f"Found {len(captures)} OAK-D captures in {args.capture_dir}")
    print(f"Depth range: {args.min_depth_m:.3f}m to {args.max_depth_m:.3f}m")
    for capture in captures:
        item = integrate_capture(volume, capture, intrinsic, args, len(integrated))
        if item is None:
            continue
        integrated.append(item)
        print(
            f"KF {item.index:03d}  angle={item.angle_deg:.2f}  "
            f"corrected={item.corrected_angle_deg:.2f}  "
            f"radius={item.radius_m:.3f}m  valid_px={item.valid_depth_px}"
        )

    if not integrated:
        print("No captures integrated; no outputs written.")
        return

    save_outputs(volume, integrated, intrinsic, args)


if __name__ == "__main__":
    main()
