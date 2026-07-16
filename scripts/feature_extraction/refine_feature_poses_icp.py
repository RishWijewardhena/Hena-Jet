#!/usr/bin/env python3
"""Refine feature-tracked RGB-D poses with guarded local-map ICP."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_DATASET = Path("datasets/zed_m_open3d")
DEFAULT_REPORT = Path("outputs/feature_tracking/lightglue_feature_report_10_20cm.json")
DEFAULT_POSES = Path("outputs/feature_tracking/lightglue_feature_poses_10_20cm.npy")
DEFAULT_POSES_OUT = Path("outputs/feature_tracking/lightglue_icp_refined_poses_10_20cm.npy")
DEFAULT_REPORT_OUT = Path("outputs/feature_tracking/lightglue_icp_refined_report_10_20cm.json")


@dataclass
class RefinementResult:
    index: int
    stem: str
    accepted: bool
    reason: str
    fitness: float
    inlier_rmse: float
    correction_translation_m: float
    correction_rotation_deg: float
    point_count: int
    local_map_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine feature-tracked camera-to-world poses with local-map ICP."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--intrinsic", type=Path, default=None)
    parser.add_argument("--poses-out", type=Path, default=DEFAULT_POSES_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--icp-voxel-m", type=float, default=0.002)
    parser.add_argument("--icp-distance-m", type=float, default=0.008)
    parser.add_argument("--icp-iterations", type=int, default=40)
    parser.add_argument("--icp-min-fitness", type=float, default=0.30)
    parser.add_argument("--icp-max-rmse-m", type=float, default=0.006)
    parser.add_argument("--icp-max-correction-m", type=float, default=0.015)
    parser.add_argument("--icp-max-correction-deg", type=float, default=5.0)
    parser.add_argument("--local-map-frames", type=int, default=8)
    parser.add_argument("--local-map-voxel-m", type=float, default=0.002)
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
    if args.icp_voxel_m <= 0:
        raise ValueError("--icp-voxel-m must be positive")
    if args.icp_distance_m <= 0:
        raise ValueError("--icp-distance-m must be positive")
    if args.icp_iterations <= 0:
        raise ValueError("--icp-iterations must be positive")
    if not 0.0 <= args.icp_min_fitness <= 1.0:
        raise ValueError("--icp-min-fitness must be between 0 and 1")
    if args.icp_max_rmse_m <= 0:
        raise ValueError("--icp-max-rmse-m must be positive")
    if args.icp_max_correction_m <= 0:
        raise ValueError("--icp-max-correction-m must be positive")
    if args.icp_max_correction_deg <= 0:
        raise ValueError("--icp-max-correction-deg must be positive")
    if args.local_map_frames <= 0:
        raise ValueError("--local-map-frames must be positive")
    if args.local_map_voxel_m <= 0:
        raise ValueError("--local-map-voxel-m must be positive")


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


def accepted_stems_from_report(report: dict) -> list[str]:
    stems = report.get("accepted_frame_stems")
    if not stems:
        raise ValueError("Report must contain accepted_frame_stems")
    return [str(stem) for stem in stems]


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


def make_frame_cloud(
    dataset: Path,
    stem: str,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    voxel_m: float,
) -> o3d.geometry.PointCloud:
    rgbd = read_rgbd(dataset, stem, depth_scale, depth_min_m, depth_max_m)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    if len(cloud.points) == 0:
        return cloud
    cloud = cloud.voxel_down_sample(voxel_m)
    if len(cloud.points) > 0:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 3.0, max_nn=30)
        )
    return cloud


def transformed_copy(cloud: o3d.geometry.PointCloud, transform: np.ndarray) -> o3d.geometry.PointCloud:
    copy = o3d.geometry.PointCloud(cloud)
    copy.transform(transform)
    return copy


def build_local_map(
    local_clouds: list[o3d.geometry.PointCloud],
    voxel_m: float,
) -> o3d.geometry.PointCloud:
    local_map = o3d.geometry.PointCloud()
    for cloud in local_clouds:
        local_map += cloud
    if len(local_map.points) == 0:
        return local_map
    local_map = local_map.voxel_down_sample(voxel_m)
    if len(local_map.points) > 0:
        local_map.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 3.0, max_nn=50)
        )
    return local_map


def pose_delta(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(reference) @ candidate
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cos_angle = (float(np.trace(delta[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return translation_m, math.degrees(math.acos(cos_angle))


def refine_pose_with_icp(
    source_cloud: o3d.geometry.PointCloud,
    local_map: o3d.geometry.PointCloud,
    initial_camera_to_world: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, float]:
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        local_map,
        args.icp_distance_m,
        initial_camera_to_world,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=args.icp_iterations
        ),
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def should_accept_icp(
    fitness: float,
    inlier_rmse: float,
    correction_translation_m: float,
    correction_rotation_deg: float,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if fitness < args.icp_min_fitness:
        return False, "low_fitness"
    if inlier_rmse > args.icp_max_rmse_m:
        return False, "high_rmse"
    if correction_translation_m > args.icp_max_correction_m:
        return False, "translation_correction_too_large"
    if correction_rotation_deg > args.icp_max_correction_deg:
        return False, "rotation_correction_too_large"
    return True, "accepted"


def refinement_result_to_dict(result: RefinementResult) -> dict:
    return {
        "index": result.index,
        "stem": result.stem,
        "accepted": result.accepted,
        "reason": result.reason,
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "correction_translation_m": result.correction_translation_m,
        "correction_rotation_deg": result.correction_rotation_deg,
        "point_count": result.point_count,
        "local_map_points": result.local_map_points,
    }


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

    stems = accepted_stems_from_report(report)
    initial_poses = np.load(args.poses)
    if len(stems) != len(initial_poses):
        raise ValueError(
            f"Accepted frame count ({len(stems)}) does not match poses ({len(initial_poses)})"
        )

    print(f"Dataset: {args.dataset}")
    print(f"Report: {args.report}")
    print(f"Input poses: {args.poses}")
    print(f"Frames to refine: {len(stems)}")
    print(f"Depth range: {depth_min_m:g}m..{depth_max_m:g}m")
    print(
        f"ICP: voxel={args.icp_voxel_m:g}m distance={args.icp_distance_m:g}m "
        f"local_map_frames={args.local_map_frames}"
    )

    refined_poses: list[np.ndarray] = []
    local_clouds: list[o3d.geometry.PointCloud] = []
    results: list[RefinementResult] = []

    for index, (stem, initial_pose) in enumerate(zip(stems, initial_poses)):
        initial_pose = np.asarray(initial_pose, dtype=np.float64)
        source_cloud = make_frame_cloud(
            args.dataset,
            stem,
            intrinsic,
            depth_scale,
            depth_min_m,
            depth_max_m,
            args.icp_voxel_m,
        )
        if index == 0:
            refined_pose = initial_pose
            reason = "first_frame"
            accepted = False
            fitness = 0.0
            inlier_rmse = 0.0
            correction_translation_m = 0.0
            correction_rotation_deg = 0.0
            local_map_points = 0
        elif len(source_cloud.points) == 0:
            refined_pose = initial_pose
            reason = "empty_source_cloud"
            accepted = False
            fitness = 0.0
            inlier_rmse = 0.0
            correction_translation_m = 0.0
            correction_rotation_deg = 0.0
            local_map_points = 0
        else:
            local_map = build_local_map(local_clouds, args.local_map_voxel_m)
            local_map_points = len(local_map.points)
            if local_map_points == 0:
                refined_pose = initial_pose
                reason = "empty_local_map"
                accepted = False
                fitness = 0.0
                inlier_rmse = 0.0
                correction_translation_m = 0.0
                correction_rotation_deg = 0.0
            else:
                candidate_pose, fitness, inlier_rmse = refine_pose_with_icp(
                    source_cloud,
                    local_map,
                    initial_pose,
                    args,
                )
                correction_translation_m, correction_rotation_deg = pose_delta(
                    initial_pose, candidate_pose
                )
                accepted, reason = should_accept_icp(
                    fitness,
                    inlier_rmse,
                    correction_translation_m,
                    correction_rotation_deg,
                    args,
                )
                refined_pose = candidate_pose if accepted else initial_pose

        refined_poses.append(refined_pose)
        world_cloud = transformed_copy(source_cloud, refined_pose)
        local_clouds.append(world_cloud)
        if len(local_clouds) > args.local_map_frames:
            local_clouds = local_clouds[-args.local_map_frames :]

        results.append(
            RefinementResult(
                index=index,
                stem=stem,
                accepted=accepted,
                reason=reason,
                fitness=fitness,
                inlier_rmse=inlier_rmse,
                correction_translation_m=correction_translation_m,
                correction_rotation_deg=correction_rotation_deg,
                point_count=len(source_cloud.points),
                local_map_points=local_map_points,
            )
        )
        print(
            f"{stem} {reason} fitness={fitness:.3f} rmse={inlier_rmse:.4f} "
            f"corr={correction_translation_m:.4f}m/{correction_rotation_deg:.2f}deg "
            f"pts={len(source_cloud.points)} map={local_map_points}"
        )

    poses_array = np.stack(refined_poses, axis=0)
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses_array)

    accepted_count = sum(1 for result in results if result.accepted)
    refined_report = dict(report)
    refined_report.update(
        {
            "source_report": str(args.report),
            "source_poses": str(args.poses),
            "method": f"{report.get('method', 'feature_tracking')}_icp_refined",
            "depth_scale": depth_scale,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "accepted_frames": len(stems),
            "accepted_frame_stems": stems,
            "icp": {
                "accepted_corrections": accepted_count,
                "attempted_corrections": max(0, len(stems) - 1),
                "icp_voxel_m": args.icp_voxel_m,
                "icp_distance_m": args.icp_distance_m,
                "icp_iterations": args.icp_iterations,
                "icp_min_fitness": args.icp_min_fitness,
                "icp_max_rmse_m": args.icp_max_rmse_m,
                "icp_max_correction_m": args.icp_max_correction_m,
                "icp_max_correction_deg": args.icp_max_correction_deg,
                "local_map_frames": args.local_map_frames,
                "local_map_voxel_m": args.local_map_voxel_m,
            },
            "icp_refinements": [
                refinement_result_to_dict(result) for result in results
            ],
        }
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", encoding="utf-8") as file:
        json.dump(refined_report, file, indent=2)
        file.write("\n")

    print(f"ICP corrections accepted: {accepted_count}/{max(0, len(stems) - 1)}")
    print(f"Refined poses: {args.poses_out}")
    print(f"Refinement report: {args.report_out}")


if __name__ == "__main__":
    main()
