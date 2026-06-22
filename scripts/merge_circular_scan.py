#!/usr/bin/env python3
"""Merge ZED point cloud captures from a fixed-object, orbiting-camera scan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the orbiting-camera merge.

    --capture-dir: Folder containing angle_*.json and matching angle_*.npz files.
    --out: Output PLY path for the merged colored point cloud.
    --voxel-m: Final output voxel size in meters. Use 0 to disable downsampling.
    --max-points-per-frame: Safety limit for very dense captures.
    --max-depth-over-radius-m: Camera-local depth crop. For example, if the
        scanner radius is 0.20 and this value is 0.05, points deeper than
        0.25 m from the camera are removed before merging.
    --min-world-z and --max-world-z: Optional height crop after transforming
        each frame into scanner/world coordinates.
    --max-world-radius-m: Optional XY radius crop around the scanner center.
    --icp: Optionally refine each new frame against the accumulated cloud with
        guarded Open3D point-to-plane ICP.
    --icp-* options: Control ICP downsampling, matching distance, iteration
        count, and safety gates for accepting only small corrections.
    """
    parser = argparse.ArgumentParser(
        description="Merge ZED captures for a fixed object with camera orbiting around Z."
    )
    parser.add_argument("--capture-dir", type=Path, default=Path("captures/zed_m_first_scan"))
    parser.add_argument("--out", type=Path, default=Path("outputs/zed_m_merged_scan.ply"))
    parser.add_argument("--voxel-m", type=float, default=0.002)
    parser.add_argument("--max-points-per-frame", type=int, default=250_000)
    parser.add_argument("--max-depth-over-radius-m", type=float, default=0.05)
    parser.add_argument("--min-world-z", type=float, default=None)
    parser.add_argument("--max-world-z", type=float, default=None)
    parser.add_argument("--max-world-radius-m", type=float, default=None)
    parser.add_argument("--icp", action="store_true")
    parser.add_argument("--icp-voxel-m", type=float, default=0.005)
    parser.add_argument("--icp-distance-m", type=float, default=0.02)
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--icp-min-fitness", type=float, default=0.65)
    parser.add_argument("--icp-max-translation-m", type=float, default=0.02)
    parser.add_argument("--icp-max-rotation-deg", type=float, default=8.0)
    return parser.parse_args()


def transform_camera_to_world(
    points: np.ndarray,
    angle_deg: float,
    radius_m: float,
    height_m: float,
) -> np.ndarray:
    """Transform camera-local ZED points into scanner/world coordinates.

    This project now assumes one physical setup: the object stays fixed at the
    scanner center while the ZED camera moves around it on a circular path.

    The ZED capture uses RIGHT_HANDED_Z_UP_X_FWD:
    - camera +X is forward from the camera toward what it sees
    - camera +Y is left
    - camera +Z is up

    angle_deg is the camera position angle around the scanner center. radius_m
    is the distance from scanner center to the camera optical center. height_m
    is the camera height offset from the scanner/world reference plane.
    """
    theta = math.radians(angle_deg)
    radial_out = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float32)
    forward = -radial_out
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    left = np.cross(up, forward).astype(np.float32)
    camera_position = np.array(
        [radius_m * math.cos(theta), radius_m * math.sin(theta), height_m],
        dtype=np.float32,
    )

    return (
        camera_position
        + points[:, 0:1] * forward
        + points[:, 1:2] * left
        + points[:, 2:3] * up
    )


def crop_world_region(
    points: np.ndarray,
    colors: np.ndarray,
    min_world_z: float | None,
    max_world_z: float | None,
    max_world_radius_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop transformed world points to the scanner region containing the object.

    min_world_z removes table/base points below the object. max_world_z removes
    high background clutter. max_world_radius_m removes side/background points
    that are too far from the scanner center in the XY plane.
    """
    keep = np.ones(points.shape[0], dtype=bool)
    if min_world_z is not None:
        keep &= points[:, 2] >= min_world_z
    if max_world_z is not None:
        keep &= points[:, 2] <= max_world_z
    if max_world_radius_m is not None:
        keep &= np.linalg.norm(points[:, :2], axis=1) <= max_world_radius_m
    return points[keep], colors[keep]


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the final point cloud by keeping one point per voxel grid cell."""
    if voxel_m <= 0:
        return points, colors

    keys = np.floor(points / voxel_m).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices], colors[unique_indices]


def make_o3d_cloud(o3d, points: np.ndarray, colors: np.ndarray, voxel_m: float):
    """Build an Open3D cloud with estimated normals for point-to-plane ICP."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    if voxel_m > 0:
        cloud = cloud.voxel_down_sample(voxel_m)

    normal_radius = max(voxel_m * 3.0, 0.01)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    return cloud


def refine_with_point_to_plane_icp(
    o3d,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    target_points: np.ndarray,
    target_colors: np.ndarray,
    voxel_m: float,
    distance_m: float,
    iterations: int,
):
    """Refine one mechanically placed frame against the accumulated cloud.

    ICP starts from identity because the angle/radius/height transform has
    already placed source_points in world coordinates. The returned points are
    the full-resolution source points after the ICP correction.
    """
    source = make_o3d_cloud(o3d, source_points, source_colors, voxel_m)
    target = make_o3d_cloud(o3d, target_points, target_colors, voxel_m)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_m,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
    )

    rotation = result.transformation[:3, :3]
    translation = result.transformation[:3, 3]
    transformed_points = source_points.astype(np.float64) @ rotation.T + translation
    return transformed_points.astype(np.float32), result


def icp_transform_summary(transformation: np.ndarray) -> tuple[float, float]:
    """Return ICP correction size as translation meters and rotation degrees."""
    translation_m = float(np.linalg.norm(transformation[:3, 3]))
    rotation = transformation[:3, :3]
    cos_angle = (float(np.trace(rotation)) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    rotation_deg = math.degrees(math.acos(cos_angle))
    return translation_m, rotation_deg


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write colored XYZ points to an ASCII PLY file for MeshLab/CloudCompare."""
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


def load_capture(meta_path: Path) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Load one capture metadata file and its matching compressed point arrays."""
    npz_path = meta_path.with_suffix(".npz")
    if not npz_path.exists():
        print(f"Skipping {meta_path}: missing {npz_path.name}")
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    return meta, data["points"], data["colors"]


def main() -> None:
    """Merge fixed-object, orbiting-camera captures into one colored PLY."""
    args = parse_args()
    o3d = None
    if args.icp:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "Open3D is required for --icp. Install it with: python -m pip install open3d"
            ) from exc

    frames: list[tuple[np.ndarray, np.ndarray]] = []

    for meta_path in sorted(args.capture_dir.glob("angle_*.json")):
        loaded = load_capture(meta_path)
        if loaded is None:
            continue

        meta, points, colors = loaded
        radius_m = float(meta["radius_m"])
        angle_deg = float(meta["angle_deg"])
        height_m = float(meta["height_m"])

        if args.max_depth_over_radius_m > 0:
            max_camera_depth_m = radius_m + args.max_depth_over_radius_m
            keep_depth = points[:, 0] <= max_camera_depth_m
            points = points[keep_depth]
            colors = colors[keep_depth]

        if points.shape[0] > args.max_points_per_frame:
            stride = math.ceil(points.shape[0] / args.max_points_per_frame)
            points = points[::stride]
            colors = colors[::stride]

        world_points = transform_camera_to_world(
            points,
            angle_deg=angle_deg,
            radius_m=radius_m,
            height_m=height_m,
        ).astype(np.float32)
        colors = colors.astype(np.uint8)

        before_crop = world_points.shape[0]
        world_points, colors = crop_world_region(
            world_points,
            colors,
            min_world_z=args.min_world_z,
            max_world_z=args.max_world_z,
            max_world_radius_m=args.max_world_radius_m,
        )
        if world_points.shape[0] == 0:
            print(f"Skipping {meta_path.name}: crop removed all points")
            continue
        if world_points.shape[0] != before_crop:
            print(f"World crop {meta_path.name}: {before_crop} -> {world_points.shape[0]} points")

        if args.icp and frames:
            target_points = np.concatenate([frame[0] for frame in frames], axis=0)
            target_colors = np.concatenate([frame[1] for frame in frames], axis=0)
            icp_points, icp_result = refine_with_point_to_plane_icp(
                o3d,
                world_points,
                colors,
                target_points,
                target_colors,
                voxel_m=args.icp_voxel_m,
                distance_m=args.icp_distance_m,
                iterations=args.icp_iterations,
            )
            icp_translation_m, icp_rotation_deg = icp_transform_summary(
                icp_result.transformation
            )
            icp_ok = (
                icp_result.fitness >= args.icp_min_fitness
                and icp_translation_m <= args.icp_max_translation_m
                and icp_rotation_deg <= args.icp_max_rotation_deg
            )
            status = "accepted" if icp_ok else "rejected"
            print(
                f"ICP {meta_path.name}: {status}, "
                f"fitness={icp_result.fitness:.3f}, "
                f"rmse={icp_result.inlier_rmse:.4f}, "
                f"move={icp_translation_m:.4f}m, "
                f"rot={icp_rotation_deg:.2f}deg"
            )
            if icp_ok:
                world_points = icp_points

        frames.append((world_points, colors))
        print(f"Loaded {meta_path.name}: {world_points.shape[0]} points")

    if not frames:
        raise RuntimeError(f"No captures found in {args.capture_dir}")

    merged_points = np.concatenate([frame[0] for frame in frames], axis=0)
    merged_colors = np.concatenate([frame[1] for frame in frames], axis=0)
    merged_points, merged_colors = voxel_downsample(
        merged_points,
        merged_colors,
        args.voxel_m,
    )

    write_ascii_ply(args.out, merged_points, merged_colors)
    print(f"Saved merged cloud with {merged_points.shape[0]} points")
    print(args.out)


if __name__ == "__main__":
    main()
