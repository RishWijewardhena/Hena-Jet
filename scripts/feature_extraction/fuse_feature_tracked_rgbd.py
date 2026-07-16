#!/usr/bin/env python3
"""Fuse RGB-D frames using poses from rgbd_feature_ransac_tracker.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_DATASET = Path("datasets/zed_m_open3d")
DEFAULT_REPORT = Path("outputs/feature_tracking/rgbd_feature_report.json")
DEFAULT_POSES = Path("outputs/feature_tracking/rgbd_feature_poses.npy")
DEFAULT_CLOUD_OUT = Path("outputs/feature_tracking/feature_tracked_cloud.ply")
DEFAULT_MESH_OUT = Path("outputs/feature_tracking/feature_tracked_mesh.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate feature-tracked RGB-D frames into a TSDF cloud/mesh."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--intrinsic", type=Path, default=None)
    parser.add_argument("--cloud-out", type=Path, default=DEFAULT_CLOUD_OUT)
    parser.add_argument("--mesh-out", type=Path, default=DEFAULT_MESH_OUT)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--voxel-length-m", type=float, default=0.002)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.008)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--cleanup-outlier-neighbors", type=int, default=20)
    parser.add_argument("--cleanup-outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--cleanup-min-cluster-fraction", type=float, default=0.02)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")
    if not args.report.exists():
        raise FileNotFoundError(f"Tracker report does not exist: {args.report}")
    if not args.poses.exists():
        raise FileNotFoundError(f"Pose file does not exist: {args.poses}")
    if args.depth_scale is not None and args.depth_scale <= 0:
        raise ValueError("--depth-scale must be positive")
    if args.depth_min_m is not None and args.depth_min_m < 0:
        raise ValueError("--depth-min-m must be zero or positive")
    if args.depth_max_m is not None and args.depth_max_m <= 0:
        raise ValueError("--depth-max-m must be positive")
    if args.voxel_length_m <= 0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")
    if args.cleanup_outlier_neighbors <= 0:
        raise ValueError("--cleanup-outlier-neighbors must be positive")
    if args.cleanup_outlier_std_ratio <= 0:
        raise ValueError("--cleanup-outlier-std-ratio must be positive")
    if not 0.0 < args.cleanup_min_cluster_fraction <= 1.0:
        raise ValueError("--cleanup-min-cluster-fraction must be between 0 and 1")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_capture_config(dataset: Path) -> dict:
    path = dataset / "capture_config.json"
    if not path.exists():
        return {}
    return read_json(path)


def resolve_intrinsic_path(args: argparse.Namespace, config: dict, report: dict) -> Path:
    if args.intrinsic is not None:
        return args.intrinsic
    for value in (report.get("intrinsic"), config.get("path_intrinsic")):
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path
        candidate = args.dataset / path.name
        if candidate.exists():
            return candidate
    return args.dataset / "intrinsic.json"


def common_frame_stems(dataset: Path) -> list[str]:
    image_dir = dataset / "image"
    depth_dir = dataset / "depth"
    if not image_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(f"Expected image/ and depth/ folders in {dataset}")
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    depth_stems = {path.stem for path in depth_dir.glob("*.png")}
    stems = sorted(image_stems & depth_stems)
    if not stems:
        raise ValueError(f"No matching RGB/depth PNG pairs found in {dataset}")
    return stems


def accepted_stems_from_report(report: dict, dataset: Path) -> list[str]:
    stems = report.get("accepted_frame_stems")
    if stems:
        return [str(stem) for stem in stems]

    all_stems = common_frame_stems(dataset)
    selected_count = int(report.get("selected_frames", len(all_stems)))
    selected_stems = all_stems[:selected_count]
    indices = report.get("accepted_frame_indices")
    if not indices:
        raise ValueError("Report has no accepted_frame_stems or accepted_frame_indices")
    return [selected_stems[int(index)] for index in indices]


def read_rgbd(
    dataset: Path,
    stem: str,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
) -> o3d.geometry.RGBDImage:
    color = o3d.io.read_image(str(dataset / "image" / f"{stem}.png"))
    depth_raw = np.asarray(o3d.io.read_image(str(dataset / "depth" / f"{stem}.png"))).copy()
    depth_m = depth_raw.astype(np.float32) / depth_scale
    invalid = (depth_m <= 0.0) | (depth_m < depth_min_m) | (depth_m > depth_max_m)
    depth_raw[invalid] = 0
    depth = o3d.geometry.Image(depth_raw)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=depth_scale,
        depth_trunc=depth_max_m,
        convert_rgb_to_intensity=False,
    )


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
    min_cluster_fraction: float,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    threshold = max(1, int(cluster_n_triangles.max() * min_cluster_fraction))
    mesh.remove_triangles_by_mask(cluster_n_triangles[triangle_clusters] < threshold)
    mesh.remove_unreferenced_vertices()
    return mesh


def clean_cloud(
    cloud: o3d.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0:
        return cloud
    cleaned, _ = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return cleaned


def main() -> None:
    args = parse_args()
    validate_args(args)
    report = read_json(args.report)
    config = read_capture_config(args.dataset)

    depth_scale = args.depth_scale or float(
        report.get("depth_scale", config.get("depth_scale", 1000.0))
    )
    depth_min_m = (
        args.depth_min_m
        if args.depth_min_m is not None
        else float(report.get("depth_min_m", config.get("depth_min", 0.0)))
    )
    depth_max_m = (
        args.depth_max_m
        if args.depth_max_m is not None
        else float(report.get("depth_max_m", config.get("depth_max", 1.0)))
    )
    if depth_max_m <= depth_min_m:
        raise ValueError("--depth-max-m must be greater than --depth-min-m")

    intrinsic_path = resolve_intrinsic_path(args, config, report)
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Camera intrinsic file not found: {intrinsic_path}")
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(str(intrinsic_path))

    poses = np.load(args.poses)
    stems = accepted_stems_from_report(report, args.dataset)
    if len(stems) != len(poses):
        raise ValueError(
            f"Accepted frame count ({len(stems)}) does not match poses ({len(poses)})"
        )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Report: {args.report}")
    print(f"Poses: {args.poses}")
    print(f"Accepted frames to fuse: {len(stems)}")
    print(f"Depth range: {depth_min_m:g}m..{depth_max_m:g}m")
    for index, (stem, camera_to_world) in enumerate(zip(stems, poses), start=1):
        rgbd = read_rgbd(args.dataset, stem, depth_scale, depth_min_m, depth_max_m)
        world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
        volume.integrate(rgbd, intrinsic, world_to_camera)
        print(f"  integrated {index}/{len(stems)} {stem}", end="\r")
    print()

    mesh = volume.extract_triangle_mesh()
    if not args.no_cleanup:
        before = len(mesh.triangles)
        mesh = clean_mesh(mesh, args.cleanup_min_cluster_fraction)
        print(f"Mesh cleanup: {before} -> {len(mesh.triangles)} tris")
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)

    cloud = volume.extract_point_cloud()
    if not args.no_cleanup:
        before = len(cloud.points)
        cloud = clean_cloud(
            cloud,
            args.cleanup_outlier_neighbors,
            args.cleanup_outlier_std_ratio,
        )
        print(f"Cloud cleanup: {before} -> {len(cloud.points)} pts")
    args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.cloud_out), cloud)

    print(f"Mesh: {args.mesh_out} ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")
    print(f"Cloud: {args.cloud_out} ({len(cloud.points)} pts)")


if __name__ == "__main__":
    main()
