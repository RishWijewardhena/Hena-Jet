#!/usr/bin/env python3
"""Optimize feature-tracked RGB-D poses with loop-closure pose graph edges."""

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
DEFAULT_POSES_OUT = Path("outputs/feature_tracking/lightglue_posegraph_poses_10_20cm.npy")
DEFAULT_REPORT_OUT = Path("outputs/feature_tracking/lightglue_posegraph_report_10_20cm.json")
DEFAULT_POSEGRAPH_OUT = Path("outputs/feature_tracking/lightglue_posegraph_10_20cm.json")


@dataclass
class LoopCandidate:
    source: int
    target: int
    initial_distance_m: float


@dataclass
class LoopResult:
    source: int
    target: int
    accepted: bool
    reason: str
    initial_distance_m: float
    fitness: float
    inlier_rmse: float
    correction_translation_m: float
    correction_rotation_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add loop closures and optimize feature-tracked camera poses."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--intrinsic", type=Path, default=None)
    parser.add_argument("--poses-out", type=Path, default=DEFAULT_POSES_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--posegraph-out", type=Path, default=DEFAULT_POSEGRAPH_OUT)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--cloud-voxel-m", type=float, default=0.002)
    parser.add_argument("--odometry-distance-m", type=float, default=0.008)
    parser.add_argument("--loop-distance-m", type=float, default=0.008)
    parser.add_argument("--loop-min-gap", type=int, default=20)
    parser.add_argument("--loop-stride", type=int, default=4)
    parser.add_argument("--loop-search-radius-m", type=float, default=0.12)
    parser.add_argument("--loop-min-fitness", type=float, default=0.55)
    parser.add_argument("--loop-max-rmse-m", type=float, default=0.006)
    parser.add_argument("--loop-max-correction-m", type=float, default=0.04)
    parser.add_argument("--loop-max-correction-deg", type=float, default=20.0)
    parser.add_argument("--max-loop-edges", type=int, default=24)
    parser.add_argument("--edge-prune-threshold", type=float, default=0.25)
    parser.add_argument("--no-loop-closures", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
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
    if args.cloud_voxel_m <= 0:
        raise ValueError("--cloud-voxel-m must be positive")
    if args.odometry_distance_m <= 0 or args.loop_distance_m <= 0:
        raise ValueError("ICP correspondence distances must be positive")
    if args.loop_min_gap <= 0:
        raise ValueError("--loop-min-gap must be positive")
    if args.loop_stride <= 0:
        raise ValueError("--loop-stride must be positive")
    if args.loop_search_radius_m <= 0:
        raise ValueError("--loop-search-radius-m must be positive")
    if not 0.0 <= args.loop_min_fitness <= 1.0:
        raise ValueError("--loop-min-fitness must be between 0 and 1")
    if args.loop_max_rmse_m <= 0:
        raise ValueError("--loop-max-rmse-m must be positive")
    if args.loop_max_correction_m <= 0:
        raise ValueError("--loop-max-correction-m must be positive")
    if args.loop_max_correction_deg <= 0:
        raise ValueError("--loop-max-correction-deg must be positive")
    if args.max_loop_edges < 0:
        raise ValueError("--max-loop-edges must be zero or positive")
    if not 0.0 <= args.edge_prune_threshold <= 1.0:
        raise ValueError("--edge-prune-threshold must be between 0 and 1")


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


def make_cloud(
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
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 3.0, max_nn=40)
        )
    return cloud


def target_from_source(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Return transform mapping source camera-local points into target camera-local points."""
    return np.linalg.inv(target_pose) @ source_pose


def pose_delta(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    delta = candidate @ np.linalg.inv(reference)
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cos_angle = (float(np.trace(delta[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return translation_m, math.degrees(math.acos(cos_angle))


def information_matrix(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    distance_m: float,
    transform: np.ndarray,
) -> np.ndarray:
    return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        distance_m,
        transform,
    )


def make_odometry_edge(
    index: int,
    clouds: list[o3d.geometry.PointCloud],
    poses: np.ndarray,
    args: argparse.Namespace,
) -> o3d.pipelines.registration.PoseGraphEdge:
    transform = target_from_source(poses[index], poses[index + 1])
    info = information_matrix(
        clouds[index],
        clouds[index + 1],
        args.odometry_distance_m,
        transform,
    )
    return o3d.pipelines.registration.PoseGraphEdge(
        index,
        index + 1,
        transform,
        info,
        uncertain=False,
    )


def find_loop_candidates(poses: np.ndarray, args: argparse.Namespace) -> list[LoopCandidate]:
    candidates: list[LoopCandidate] = []
    for source in range(0, len(poses), args.loop_stride):
        for target in range(source + args.loop_min_gap, len(poses), args.loop_stride):
            distance = float(np.linalg.norm(poses[source, :3, 3] - poses[target, :3, 3]))
            if distance <= args.loop_search_radius_m:
                candidates.append(LoopCandidate(source, target, distance))
    candidates.sort(key=lambda candidate: candidate.initial_distance_m)
    if args.max_loop_edges:
        candidates = candidates[: args.max_loop_edges]
    return candidates


def validate_loop_edge(
    source: int,
    target: int,
    clouds: list[o3d.geometry.PointCloud],
    poses: np.ndarray,
    args: argparse.Namespace,
) -> tuple[LoopResult, np.ndarray | None, np.ndarray | None]:
    initial_transform = target_from_source(poses[source], poses[target])
    result = o3d.pipelines.registration.registration_icp(
        clouds[source],
        clouds[target],
        args.loop_distance_m,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    correction_translation_m, correction_rotation_deg = pose_delta(
        initial_transform,
        result.transformation,
    )
    initial_distance_m = float(np.linalg.norm(poses[source, :3, 3] - poses[target, :3, 3]))

    accepted = True
    reason = "accepted"
    if result.fitness < args.loop_min_fitness:
        accepted = False
        reason = "low_fitness"
    elif result.inlier_rmse > args.loop_max_rmse_m:
        accepted = False
        reason = "high_rmse"
    elif correction_translation_m > args.loop_max_correction_m:
        accepted = False
        reason = "translation_correction_too_large"
    elif correction_rotation_deg > args.loop_max_correction_deg:
        accepted = False
        reason = "rotation_correction_too_large"

    loop_result = LoopResult(
        source=source,
        target=target,
        accepted=accepted,
        reason=reason,
        initial_distance_m=initial_distance_m,
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        correction_translation_m=correction_translation_m,
        correction_rotation_deg=correction_rotation_deg,
    )
    if not accepted:
        return loop_result, None, None

    info = information_matrix(
        clouds[source],
        clouds[target],
        args.loop_distance_m,
        result.transformation,
    )
    return loop_result, result.transformation, info


def loop_result_to_dict(result: LoopResult) -> dict:
    return {
        "source": result.source,
        "target": result.target,
        "accepted": result.accepted,
        "reason": result.reason,
        "initial_distance_m": result.initial_distance_m,
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "correction_translation_m": result.correction_translation_m,
        "correction_rotation_deg": result.correction_rotation_deg,
    }


def optimize_pose_graph(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    args: argparse.Namespace,
) -> None:
    if args.no_optimize:
        return
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.loop_distance_m,
            edge_prune_threshold=args.edge_prune_threshold,
            reference_node=0,
        ),
    )


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
    poses = np.load(args.poses)
    if len(stems) != len(poses):
        raise ValueError(
            f"Accepted frame count ({len(stems)}) does not match poses ({len(poses)})"
        )

    print(f"Dataset: {args.dataset}")
    print(f"Report: {args.report}")
    print(f"Input poses: {args.poses}")
    print(f"Frames: {len(stems)}")
    print(f"Depth range: {depth_min_m:g}m..{depth_max_m:g}m")

    print("Building frame clouds...")
    clouds = [
        make_cloud(
            args.dataset,
            stem,
            intrinsic,
            depth_scale,
            depth_min_m,
            depth_max_m,
            args.cloud_voxel_m,
        )
        for stem in stems
    ]

    pose_graph = o3d.pipelines.registration.PoseGraph()
    for pose in poses:
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.asarray(pose, dtype=np.float64))
        )

    print("Adding odometry edges...")
    for index in range(len(poses) - 1):
        pose_graph.edges.append(make_odometry_edge(index, clouds, poses, args))

    loop_results: list[LoopResult] = []
    accepted_loop_edges = 0
    if not args.no_loop_closures:
        candidates = find_loop_candidates(poses, args)
        print(f"Loop candidates: {len(candidates)}")
        for candidate in candidates:
            result, transform, info = validate_loop_edge(
                candidate.source,
                candidate.target,
                clouds,
                poses,
                args,
            )
            loop_results.append(result)
            print(
                f"loop {candidate.source}->{candidate.target} {result.reason} "
                f"dist={result.initial_distance_m:.4f}m "
                f"fitness={result.fitness:.3f} rmse={result.inlier_rmse:.4f} "
                f"corr={result.correction_translation_m:.4f}m/"
                f"{result.correction_rotation_deg:.2f}deg"
            )
            if transform is None or info is None:
                continue
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    candidate.source,
                    candidate.target,
                    transform,
                    info,
                    uncertain=True,
                )
            )
            accepted_loop_edges += 1

    print(
        f"Pose graph: {len(pose_graph.nodes)} nodes, "
        f"{len(pose_graph.edges)} edges, {accepted_loop_edges} loop edges"
    )
    optimize_pose_graph(pose_graph, args)

    optimized_poses = np.stack(
        [np.asarray(node.pose, dtype=np.float64) for node in pose_graph.nodes],
        axis=0,
    )
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, optimized_poses)
    args.posegraph_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_pose_graph(str(args.posegraph_out), pose_graph)

    optimized_report = dict(report)
    optimized_report.update(
        {
            "source_report": str(args.report),
            "source_poses": str(args.poses),
            "method": f"{report.get('method', 'feature_tracking')}_posegraph",
            "depth_scale": depth_scale,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "accepted_frames": len(stems),
            "accepted_frame_stems": stems,
            "pose_graph": {
                "nodes": len(pose_graph.nodes),
                "edges": len(pose_graph.edges),
                "odometry_edges": max(0, len(poses) - 1),
                "loop_candidates": len(loop_results),
                "accepted_loop_edges": accepted_loop_edges,
                "cloud_voxel_m": args.cloud_voxel_m,
                "odometry_distance_m": args.odometry_distance_m,
                "loop_distance_m": args.loop_distance_m,
                "loop_min_gap": args.loop_min_gap,
                "loop_stride": args.loop_stride,
                "loop_search_radius_m": args.loop_search_radius_m,
                "loop_min_fitness": args.loop_min_fitness,
                "loop_max_rmse_m": args.loop_max_rmse_m,
                "loop_max_correction_m": args.loop_max_correction_m,
                "loop_max_correction_deg": args.loop_max_correction_deg,
                "edge_prune_threshold": args.edge_prune_threshold,
            },
            "loop_closures": [loop_result_to_dict(result) for result in loop_results],
        }
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", encoding="utf-8") as file:
        json.dump(optimized_report, file, indent=2)
        file.write("\n")

    print(f"Accepted loop edges: {accepted_loop_edges}/{len(loop_results)}")
    print(f"Optimized poses: {args.poses_out}")
    print(f"Pose graph: {args.posegraph_out}")
    print(f"Optimization report: {args.report_out}")


if __name__ == "__main__":
    main()
