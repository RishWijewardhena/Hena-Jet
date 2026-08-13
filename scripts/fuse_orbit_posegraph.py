#!/usr/bin/env python3
"""Fuse motor-angle RGB-D captures with orbit or saved VSLAM pose priors."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from fuse_tsdf_scan import (
    apply_depth_filters,
    configure_object_frame,
    depth_confidence_mask,
    capture_pose,
    load_capture,
    load_confidence_image,
    make_intrinsic,
    make_rgbd,
)


@dataclass
class CaptureFrame:
    """Registration data and selected pose prior for one motor-angle capture."""

    meta_path: Path
    meta: dict
    radius_m: float
    depth_trunc_m: float
    prior_pose: np.ndarray
    cloud: o3d.geometry.PointCloud
    valid_depth_pixels: int


@dataclass
class RegistrationEdge:
    """One guarded source-to-target registration result."""

    source_id: int
    target_id: int
    transform: np.ndarray
    information: np.ndarray
    accepted: bool
    reason: str
    fitness: float
    rmse_m: float
    correction_m: float
    correction_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse angle-indexed ZED RGB-D captures using the measured circular "
            "orbit or saved VSLAM poses as priors, guarded ICP, and pose-graph optimization."
        )
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("captures/zed_m_serial_scan"),
    )
    parser.add_argument(
        "--mesh-out",
        type=Path,
        default=Path("outputs/zed_m_orbit_posegraph_mesh.ply"),
    )
    parser.add_argument(
        "--cloud-out",
        type=Path,
        default=Path("outputs/zed_m_orbit_posegraph_cloud.ply"),
    )
    parser.add_argument(
        "--posegraph-out",
        type=Path,
        default=Path("outputs/zed_m_orbit_posegraph.json"),
    )
    parser.add_argument(
        "--poses-out",
        type=Path,
        default=Path("outputs/zed_m_orbit_optimized_poses.npy"),
    )
    parser.add_argument(
        "--prior-poses-out",
        type=Path,
        default=Path("outputs/zed_m_orbit_prior_poses.npy"),
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=Path("outputs/zed_m_orbit_posegraph_diagnostics.json"),
    )
    parser.add_argument("--feasibility-out", type=Path, default=None)

    parser.add_argument("--voxel-length-m", type=float, default=0.002)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.012)
    parser.add_argument(
        "--sor-neighbors",
        type=int,
        default=30,
        help="Neighbor count used for statistical outlier removal.",
    )
    parser.add_argument(
        "--sor-std-ratio",
        type=float,
        default=1.5,
        help="Remove points whose mean neighbor distance exceeds this standard-deviation ratio.",
    )
    parser.add_argument(
        "--no-sor",
        action="store_true",
        help="Save the extracted TSDF cloud without statistical outlier removal.",
    )
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

    parser.add_argument("--icp-voxel-m", type=float, default=0.004)
    parser.add_argument("--icp-coarse-distance-m", type=float, default=0.020)
    parser.add_argument("--icp-fine-distance-m", type=float, default=0.008)
    parser.add_argument("--icp-max-iterations", type=int, default=60)
    parser.add_argument("--icp-min-fitness", type=float, default=0.30)
    parser.add_argument("--icp-max-rmse-m", type=float, default=0.010)
    parser.add_argument("--icp-max-correction-m", type=float, default=0.010)
    parser.add_argument("--icp-max-correction-deg", type=float, default=4.0)
    parser.add_argument("--icp-min-points", type=int, default=100)
    parser.add_argument(
        "--orbit-prior-weight", "--pose-prior-weight",
        dest="orbit_prior_weight",
        type=float,
        default=20.0,
        help="Information-matrix multiplier for adjacent selected-pose prior edges.",
    )
    parser.add_argument("--edge-prune-threshold", type=float, default=0.25)
    parser.add_argument(
        "--no-loop-closure",
        action="store_true",
        help="Do not register the first and final captures as a loop edge.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "voxel-length-m": args.voxel_length_m,
        "sdf-trunc-m": args.sdf_trunc_m,
        "icp-voxel-m": args.icp_voxel_m,
        "icp-coarse-distance-m": args.icp_coarse_distance_m,
        "icp-fine-distance-m": args.icp_fine_distance_m,
        "icp-max-rmse-m": args.icp_max_rmse_m,
        "icp-max-correction-m": args.icp_max_correction_m,
        "icp-max-correction-deg": args.icp_max_correction_deg,
        "orbit-prior-weight": args.orbit_prior_weight,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} must be greater than zero")
    if args.icp_coarse_distance_m < args.icp_fine_distance_m:
        raise ValueError(
            "--icp-coarse-distance-m must be at least --icp-fine-distance-m"
        )
    if not 0.0 <= args.icp_min_fitness <= 1.0:
        raise ValueError("--icp-min-fitness must be between 0 and 1")
    if args.icp_max_iterations < 1 or args.icp_min_points < 3:
        raise ValueError("ICP iteration and point-count limits are invalid")
    if args.sor_neighbors < 2:
        raise ValueError("--sor-neighbors must be at least 2")
    if args.sor_std_ratio <= 0:
        raise ValueError("--sor-std-ratio must be greater than zero")
    if not 0 <= args.max_depth_confidence <= 100:
        raise ValueError("--max-depth-confidence must be between 0 and 100")
    if args.max_object_radius_m is not None and args.max_object_radius_m <= 0:
        raise ValueError("--max-object-radius-m must be positive")
    if (
        args.min_object_height_m is not None
        and args.max_object_height_m is not None
        and args.min_object_height_m > args.max_object_height_m
    ):
        raise ValueError("Object minimum height cannot exceed maximum height")
    if args.override_radius_m is not None and args.override_radius_m <= 0:
        raise ValueError("--override-radius-m must be greater than zero")


def relative_camera_transform(
    source_camera_to_world: np.ndarray,
    target_camera_to_world: np.ndarray,
) -> np.ndarray:
    """Return the transform that maps source-camera points into target-camera coordinates."""
    return np.linalg.inv(target_camera_to_world) @ source_camera_to_world


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def vslam_orbit_metrics(
    poses: np.ndarray,
    *,
    center: np.ndarray,
    up: np.ndarray,
    initial_pose: np.ndarray,
) -> dict:
    """Measure raw camera trajectory circularity and return-to-start closure."""
    poses = np.asarray(poses, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    relative = poses[:, :3, 3] - center
    height = relative @ up
    radial_vectors = relative - height[:, None] * up
    radii = np.linalg.norm(radial_vectors, axis=1)
    fitted_radius = float(np.mean(radii))
    closure_m, closure_deg = transform_delta(
        np.asarray(initial_pose, dtype=np.float64),
        poses[-1],
    )
    return {
        "fitted_radius_m": fitted_radius,
        "radius_rmse_m": float(np.sqrt(np.mean((radii - fitted_radius) ** 2))),
        "height_std_m": float(np.std(height)),
        "closure_translation_m": closure_m,
        "closure_rotation_deg": closure_deg,
    }


def transform_delta(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Measure the rigid correction from a prior transform to a candidate."""
    correction = candidate @ np.linalg.inv(reference)
    return (
        float(np.linalg.norm(correction[:3, 3])),
        rotation_angle_deg(correction[:3, :3]),
    )


def icp_result_is_acceptable(
    *,
    fitness: float,
    rmse_m: float,
    prior: np.ndarray,
    candidate: np.ndarray,
    min_fitness: float,
    max_rmse_m: float,
    max_correction_m: float,
    max_correction_deg: float,
) -> tuple[bool, str, float, float]:
    correction_m, correction_deg = transform_delta(prior, candidate)
    if not np.isfinite(fitness) or not np.isfinite(rmse_m):
        return False, "non-finite ICP score", correction_m, correction_deg
    if fitness < min_fitness:
        return False, "fitness below threshold", correction_m, correction_deg
    if rmse_m > max_rmse_m:
        return False, "RMSE exceeds threshold", correction_m, correction_deg
    if correction_m > max_correction_m:
        return (
            False,
            "translation correction exceeds prior guard",
            correction_m,
            correction_deg,
        )
    if correction_deg > max_correction_deg:
        return (
            False,
            "rotation correction exceeds prior guard",
            correction_m,
            correction_deg,
        )
    return True, "accepted", correction_m, correction_deg


def depth_truncation(radius_m: float, args: argparse.Namespace) -> float:
    if args.depth_trunc_m is not None:
        return float(args.depth_trunc_m)
    return radius_m + max(args.max_depth_over_radius_m, 0.0)


def registration_cloud(
    color: np.ndarray,
    depth: np.ndarray,
    meta: dict,
    depth_trunc_m: float,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        make_rgbd(color, depth, depth_trunc_m),
        make_intrinsic(meta),
    )
    cloud = cloud.voxel_down_sample(args.icp_voxel_m)
    if len(cloud.points) > 0:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(args.icp_fine_distance_m * 2.0, args.icp_voxel_m * 3.0),
                max_nn=30,
            )
        )
    return cloud


def load_frames(
    meta_paths: list[Path],
    args: argparse.Namespace,
) -> list[CaptureFrame]:
    frames: list[CaptureFrame] = []
    expected_shape: tuple[int, int] | None = None

    for index, meta_path in enumerate(meta_paths):
        meta, depth, color = load_capture(meta_path)
        confidence = load_confidence_image(meta_path, depth.shape)
        if confidence is None:
            if args.max_depth_confidence < 100:
                raise RuntimeError(
                    f"{meta_path}: confidence filtering requires confidence_image"
                )
        else:
            depth = depth.copy()
            depth[~depth_confidence_mask(confidence, args.max_depth_confidence)] = 0.0
        if depth.shape != color.shape[:2]:
            raise RuntimeError(
                f"{meta_path}: depth shape {depth.shape} does not match "
                f"color shape {color.shape[:2]}"
            )
        if expected_shape is None:
            expected_shape = depth.shape
        elif depth.shape != expected_shape:
            raise RuntimeError(
                f"{meta_path}: all captures must use the same image dimensions"
            )

        radius_m, prior_pose = capture_pose(meta, args)
        filtered_depth = apply_depth_filters(depth, meta, radius_m, prior_pose, args)
        truncation_m = depth_truncation(radius_m, args)
        cloud = registration_cloud(
            color,
            filtered_depth,
            meta,
            truncation_m,
            args,
        )
        frame = CaptureFrame(
            meta_path=meta_path,
            meta=meta,
            radius_m=radius_m,
            depth_trunc_m=truncation_m,
            prior_pose=prior_pose,
            cloud=cloud,
            valid_depth_pixels=int(np.count_nonzero(filtered_depth)),
        )
        frames.append(frame)
        print(
            f"Prepared {index + 1}/{len(meta_paths)} {meta_path.name}: "
            f"{frame.valid_depth_pixels} depth pixels, {len(cloud.points)} ICP points"
        )
    return frames


def information_matrix(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if len(source.points) < 3 or len(target.points) < 3:
        return np.eye(6, dtype=np.float64)
    result = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        args.icp_fine_distance_m,
        transform,
    )
    if not np.all(np.isfinite(result)) or np.linalg.norm(result) == 0:
        return np.eye(6, dtype=np.float64)
    return result


def register_pair(
    source_id: int,
    target_id: int,
    frames: list[CaptureFrame],
    args: argparse.Namespace,
) -> RegistrationEdge:
    source = frames[source_id].cloud
    target = frames[target_id].cloud
    prior = relative_camera_transform(
        frames[source_id].prior_pose,
        frames[target_id].prior_pose,
    )
    if (
        len(source.points) < args.icp_min_points
        or len(target.points) < args.icp_min_points
    ):
        return RegistrationEdge(
            source_id,
            target_id,
            prior,
            information_matrix(source, target, prior, args),
            False,
            "registration cloud too small; used pose prior",
            0.0,
            float("inf"),
            0.0,
            0.0,
        )

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=args.icp_max_iterations
    )
    coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        args.icp_coarse_distance_m,
        prior,
        estimation,
        criteria,
    )
    fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        args.icp_fine_distance_m,
        coarse.transformation,
        estimation,
        criteria,
    )
    candidate = np.asarray(fine.transformation, dtype=np.float64)
    accepted, reason, correction_m, correction_deg = icp_result_is_acceptable(
        fitness=float(fine.fitness),
        rmse_m=float(fine.inlier_rmse),
        prior=prior,
        candidate=candidate,
        min_fitness=args.icp_min_fitness,
        max_rmse_m=args.icp_max_rmse_m,
        max_correction_m=args.icp_max_correction_m,
        max_correction_deg=args.icp_max_correction_deg,
    )
    transform = candidate if accepted else prior
    if not accepted:
        reason = f"{reason}; used pose prior"
    return RegistrationEdge(
        source_id=source_id,
        target_id=target_id,
        transform=transform,
        information=information_matrix(source, target, transform, args),
        accepted=accepted,
        reason=reason,
        fitness=float(fine.fitness),
        rmse_m=float(fine.inlier_rmse),
        correction_m=correction_m,
        correction_deg=correction_deg,
    )


def print_edge(edge: RegistrationEdge, label: str) -> None:
    outcome = "accepted" if edge.accepted else "fallback"
    print(
        f"{label} {edge.source_id}->{edge.target_id} {outcome}: "
        f"fitness={edge.fitness:.3f}, rmse={edge.rmse_m:.4f}m, "
        f"correction={edge.correction_m:.4f}m/{edge.correction_deg:.2f}deg "
        f"({edge.reason})"
    )


def build_pose_graph(
    frames: list[CaptureFrame],
    args: argparse.Namespace,
) -> tuple[o3d.pipelines.registration.PoseGraph, list[RegistrationEdge]]:
    graph = o3d.pipelines.registration.PoseGraph()
    for frame in frames:
        graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(frame.prior_pose.copy())
        )

    diagnostics: list[RegistrationEdge] = []
    for target_id in range(1, len(frames)):
        source_id = target_id - 1
        edge = register_pair(source_id, target_id, frames, args)
        diagnostics.append(edge)
        prior = relative_camera_transform(
            frames[source_id].prior_pose,
            frames[target_id].prior_pose,
        )
        prior_information = information_matrix(
            frames[source_id].cloud,
            frames[target_id].cloud,
            prior,
            args,
        ) * args.orbit_prior_weight
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id,
                target_id,
                prior,
                prior_information,
                uncertain=False,
            )
        )
        if edge.accepted:
            graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_id,
                    target_id,
                    edge.transform,
                    edge.information,
                    uncertain=True,
                )
            )
        print_edge(edge, "Sequential")

    if not args.no_loop_closure and len(frames) > 2:
        loop = register_pair(0, len(frames) - 1, frames, args)
        diagnostics.append(loop)
        print_edge(loop, "Loop")
        if loop.accepted:
            graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    loop.source_id,
                    loop.target_id,
                    loop.transform,
                    loop.information,
                    uncertain=True,
                )
            )
        else:
            print("Rejected loop closure was not added to the pose graph.")
    return graph, diagnostics


def optimize_pose_graph(
    graph: o3d.pipelines.registration.PoseGraph,
    args: argparse.Namespace,
) -> None:
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.icp_fine_distance_m,
            edge_prune_threshold=args.edge_prune_threshold,
            reference_node=0,
        ),
    )


def integrate_optimized_tsdf(
    frames: list[CaptureFrame],
    graph: o3d.pipelines.registration.PoseGraph,
    args: argparse.Namespace,
) -> o3d.pipelines.integration.ScalableTSDFVolume:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for index, (frame, node) in enumerate(zip(frames, graph.nodes), start=1):
        meta, depth, color = load_capture(frame.meta_path)
        confidence = load_confidence_image(frame.meta_path, depth.shape)
        if confidence is None:
            if args.max_depth_confidence < 100:
                raise RuntimeError(
                    f"{frame.meta_path}: confidence filtering requires confidence_image"
                )
        else:
            depth = depth.copy()
            depth[~depth_confidence_mask(confidence, args.max_depth_confidence)] = 0.0
        optimized_pose = np.asarray(node.pose, dtype=np.float64)
        filtered_depth = apply_depth_filters(
            depth,
            meta,
            frame.radius_m,
            optimized_pose,
            args,
        )
        volume.integrate(
            make_rgbd(color, filtered_depth, frame.depth_trunc_m),
            make_intrinsic(meta),
            np.linalg.inv(optimized_pose),
        )
        print(
            f"Integrated {index}/{len(frames)} {frame.meta_path.name}: "
            f"{np.count_nonzero(filtered_depth)} depth pixels"
        )
    return volume


def statistical_outlier_filter(
    cloud: o3d.geometry.PointCloud,
    *,
    neighbors: int,
    std_ratio: float,
) -> tuple[o3d.geometry.PointCloud, int]:
    """Remove points with unusually large mean distances to their neighbors."""
    if len(cloud.points) <= neighbors:
        return cloud, 0
    filtered, inlier_indices = cloud.remove_statistical_outlier(
        nb_neighbors=neighbors,
        std_ratio=std_ratio,
    )
    return filtered, len(cloud.points) - len(inlier_indices)


def save_geometry(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    args: argparse.Namespace,
) -> None:
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    if o3d.io.write_triangle_mesh(str(args.mesh_out), mesh) is False:
        raise RuntimeError(f"Could not write mesh to {args.mesh_out}")
    print(
        f"Saved mesh with {len(mesh.vertices)} vertices and "
        f"{len(mesh.triangles)} faces: {args.mesh_out}"
    )

    if args.cloud_out is not None:
        cloud = volume.extract_point_cloud()
        original_count = len(cloud.points)
        if not args.no_sor:
            cloud, removed_count = statistical_outlier_filter(
                cloud,
                neighbors=args.sor_neighbors,
                std_ratio=args.sor_std_ratio,
            )
            print(
                f"Statistical outlier removal kept {len(cloud.points)}/"
                f"{original_count} points and removed {removed_count} "
                f"(neighbors={args.sor_neighbors}, std_ratio={args.sor_std_ratio:g})"
            )
        args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
        if o3d.io.write_point_cloud(str(args.cloud_out), cloud) is False:
            raise RuntimeError(f"Could not write point cloud to {args.cloud_out}")
        print(f"Saved cloud with {len(cloud.points)} points: {args.cloud_out}")


def edge_as_json(edge: RegistrationEdge) -> dict:
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "accepted": edge.accepted,
        "reason": edge.reason,
        "fitness": edge.fitness,
        "rmse_m": edge.rmse_m if np.isfinite(edge.rmse_m) else None,
        "correction_m": edge.correction_m,
        "correction_deg": edge.correction_deg,
        "transform": edge.transform.tolist(),
    }


def build_vslam_feasibility_report(
    frames: list[CaptureFrame],
    args: argparse.Namespace,
) -> dict:
    """Evaluate raw VSLAM capture poses against the selected 2 mm gate."""
    trajectory_path = args.capture_dir / "vslam_trajectory.jsonl"
    samples = []
    if trajectory_path.exists():
        samples = [
            json.loads(line)
            for line in trajectory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    valid_samples = [
        sample
        for sample in samples
        if sample.get("tracking_state") == "OK"
        and sample.get("odometry_status") == "OK"
    ]
    capture_status_ok = all(
        frame.meta.get("tracking_state") == "OK"
        and frame.meta.get("odometry_status") == "OK"
        for frame in frames
    )
    report = {
        "pose_source": "vslam",
        "required_capture_count": 72,
        "capture_count": len(frames),
        "all_capture_poses_ok": capture_status_ok,
        "trajectory_samples": len(samples),
        "valid_tracking_samples": len(valid_samples),
        "valid_tracking_fraction": (
            len(valid_samples) / len(samples) if samples else 0.0
        ),
        "thresholds": {
            "radius_rmse_m": 0.002,
            "radius_bias_m": 0.002,
            "closure_translation_m": 0.002,
            "closure_rotation_deg": 1.0,
        },
    }
    if args.object_center_m is None or args.object_up is None or not valid_samples:
        report["passed"] = False
        report["failure_reason"] = "Missing object frame or valid continuous trajectory"
        return report

    poses = np.stack([frame.prior_pose for frame in frames])
    initial_pose = np.asarray(
        valid_samples[0]["camera_to_vslam_world"],
        dtype=np.float64,
    )
    metrics = vslam_orbit_metrics(
        poses,
        center=np.asarray(args.object_center_m),
        up=np.asarray(args.object_up),
        initial_pose=initial_pose,
    )
    metrics["radius_bias_m"] = abs(metrics["fitted_radius_m"] - frames[0].radius_m)
    report["metrics"] = metrics

    covariance_diagonals = []
    for frame in frames:
        covariance = np.asarray(frame.meta.get("pose_covariance", []), dtype=np.float64)
        if covariance.size == 36:
            covariance_diagonals.append(np.diag(covariance.reshape(6, 6)))
    if covariance_diagonals:
        diagonals = np.stack(covariance_diagonals)
        report["pose_covariance_diagonal_mean"] = np.mean(diagonals, axis=0).tolist()
        report["pose_covariance_diagonal_max"] = np.max(diagonals, axis=0).tolist()

    report["passed"] = bool(
        len(frames) == 72
        and capture_status_ok
        and metrics["radius_rmse_m"] <= 0.002
        and metrics["radius_bias_m"] <= 0.002
        and metrics["closure_translation_m"] <= 0.002
        and metrics["closure_rotation_deg"] <= 1.0
    )
    if not report["passed"]:
        report["failure_reason"] = (
            "Camera-only VSLAM did not meet the 2 mm authoritative-fusion gate"
        )
    return report


def save_pose_outputs(
    frames: list[CaptureFrame],
    graph: o3d.pipelines.registration.PoseGraph,
    diagnostics: list[RegistrationEdge],
    args: argparse.Namespace,
) -> None:
    optimized = np.stack(
        [np.asarray(node.pose, dtype=np.float64) for node in graph.nodes]
    )
    priors = np.stack([frame.prior_pose for frame in frames])

    for path, values in (
        (args.poses_out, optimized),
        (args.prior_poses_out, priors),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)

    args.posegraph_out.parent.mkdir(parents=True, exist_ok=True)
    if o3d.io.write_pose_graph(str(args.posegraph_out), graph) is False:
        raise RuntimeError(f"Could not write pose graph to {args.posegraph_out}")

    pose_corrections = [
        {
            "index": index,
            "capture": frame.meta_path.name,
            "angle_deg": float(frame.meta["angle_deg"]),
            "translation_m": transform_delta(frame.prior_pose, optimized[index])[0],
            "rotation_deg": transform_delta(frame.prior_pose, optimized[index])[1],
        }
        for index, frame in enumerate(frames)
    ]
    payload = {
        "capture_dir": str(args.capture_dir),
        "frame_count": len(frames),
        "radius_m": sorted({frame.radius_m for frame in frames}),
        "settings": {
            "icp_voxel_m": args.icp_voxel_m,
            "pose_source": args.pose_source,
            "max_depth_confidence": args.max_depth_confidence,
            "icp_coarse_distance_m": args.icp_coarse_distance_m,
            "icp_fine_distance_m": args.icp_fine_distance_m,
            "icp_min_fitness": args.icp_min_fitness,
            "icp_max_rmse_m": args.icp_max_rmse_m,
            "icp_max_correction_m": args.icp_max_correction_m,
            "icp_max_correction_deg": args.icp_max_correction_deg,
            "orbit_prior_weight": args.orbit_prior_weight,
            "statistical_outlier_removal": not args.no_sor,
            "sor_neighbors": args.sor_neighbors,
            "sor_std_ratio": args.sor_std_ratio,
            "loop_closure": not args.no_loop_closure,
        },
        "edges": [edge_as_json(edge) for edge in diagnostics],
        "pose_corrections": pose_corrections,
    }
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved pose graph: {args.posegraph_out}")
    print(f"Saved optimized poses: {args.poses_out}")
    print(f"Saved orbit priors: {args.prior_poses_out}")
    print(f"Saved diagnostics: {args.diagnostics_out}")
    if args.pose_source == "vslam":
        report = build_vslam_feasibility_report(frames, args)
        feasibility_out = (
            args.feasibility_out
            if args.feasibility_out is not None
            else args.diagnostics_out.with_name("vslam_feasibility.json")
        )
        feasibility_out.parent.mkdir(parents=True, exist_ok=True)
        feasibility_out.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        verdict = "PASS" if report["passed"] else "FAIL"
        print(f"VSLAM feasibility: {verdict} ({feasibility_out})")


def main() -> None:
    args = parse_args()
    configure_object_frame(args)
    validate_args(args)
    meta_paths = sorted(args.capture_dir.glob("angle_*.json"))
    if len(meta_paths) < 2:
        raise RuntimeError(
            f"Need at least two angle_*.json captures in {args.capture_dir}"
        )

    print(
        ("Using saved camera-only VSLAM poses as graph priors. "
         if args.pose_source == "vslam" else
         "Using one fixed mechanical pivot and analytic orbit poses. ")
        + "ICP is limited to small corrections around that prior."
    )
    frames = load_frames(meta_paths, args)
    graph, diagnostics = build_pose_graph(frames, args)
    print(f"Optimizing {len(graph.nodes)} nodes and {len(graph.edges)} edges...")
    optimize_pose_graph(graph, args)
    save_pose_outputs(frames, graph, diagnostics, args)

    print("Rebuilding TSDF from optimized poses...")
    volume = integrate_optimized_tsdf(frames, graph, args)
    save_geometry(volume, args)


if __name__ == "__main__":
    main()
