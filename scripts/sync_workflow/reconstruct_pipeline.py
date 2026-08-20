#!/usr/bin/env python3
"""CloudCompare-based point-cloud reconstruction pipeline for sync_workflow.

Adapted from scripts/transform_clouds_pipeline.py.  Operates on per-angle
PLY files saved by main_scan.py and produces a merged, cleaned point cloud
plus registration diagnostics.

Stages:
  1. Build orbit pose priors from motor angles
  2. Use motor poses, or optional guarded ICP + pose-graph optimization
  3. CloudCompare: apply optimized transforms, crop, SOR per scan
  4. CloudCompare: merge all scans, dedup, subsample, orient normals
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CloudCompare invocation (flatpak)
# ---------------------------------------------------------------------------
CLOUDCOMPARE_COMMAND = [
    "/usr/bin/flatpak",
    "run",
    "org.cloudcompare.CloudCompare",
]

# ---------------------------------------------------------------------------
# Default registration settings
# ---------------------------------------------------------------------------
REGISTRATION_VOXEL_M = 0.003        # registration cloud (3mm)
REGISTRATION_NORMAL_RADIUS_M = 0.006
ICP_COARSE_DISTANCE_M = 0.004
ICP_FINE_DISTANCE_M = 0.002
ICP_ITERATIONS = 100
ICP_MIN_FITNESS = 0.30
ICP_MAX_RMSE_M = 0.0015
ICP_MAX_CORRECTION_M = 0.005
ICP_MAX_CORRECTION_DEG = 2.0
ICP_MAX_FITNESS_DROP = 0.02
ICP_MIN_FITNESS_GAIN = 0.01
ICP_RMSE_IMPROVEMENT_RATIO = 0.98
ICP_MIN_POINTS = 100
ORBIT_PRIOR_WEIGHT_ACCEPTED = 50.0   # retain a strong motor prior beside accepted ICP
ORBIT_PRIOR_WEIGHT_FALLBACK = 200.0  # strengthen the motor prior when ICP fails
POSE_GRAPH_EDGE_PRUNE_THRESHOLD = 0.25

# Registration cleanup
PRE_ICP_SOR_NEIGHBORS = 20
PRE_ICP_SOR_SIGMA = 1.5

# Full-resolution per-scan cleanup after pose estimation
PER_SCAN_SOR_NEIGHBORS = 10
PER_SCAN_SOR_SIGMA = 2.0

# Final merged-cloud cleanup
FINAL_SOR_NEIGHBORS = 20
FINAL_SOR_SIGMA = 1.5
REMOVE_DUPLICATES_DISTANCE_M = 0.00010
SPATIAL_SUBSAMPLE_M = 0.001
NORMAL_RADIUS_M = 0.0040
NORMAL_MST_NEIGHBORS = 12

# ===================================================================
# Data classes
# ===================================================================

@dataclass
class RegistrationEdge:
    """One pairwise registration constraint."""
    source_id: int
    target_id: int
    kind: str
    transform: np.ndarray
    information: np.ndarray
    accepted: bool
    reason: str
    fitness: float
    rmse_m: float
    correction_m: float
    correction_deg: float
    usable: bool = False
    prior_fitness: float = 0.0
    prior_rmse_m: float = float("inf")


@dataclass
class RegistrationFrame:
    """One raw scan and the poses/cloud used by global registration."""
    path: Path
    angle_deg: float
    prior_pose: np.ndarray
    cloud: object = None


# ===================================================================
# Geometry helpers (from transform_clouds_pipeline.py)
# ===================================================================

def normalized_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"{name} must contain a finite non-zero XYZ vector.")
    return vector / length


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def rodrigues_rotation(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Return a proper 3×3 rotation around an arbitrary unit axis."""
    axis = normalized_vector(axis, name="Rotation axis")
    angle_radians = math.radians(angle_degrees)
    sine = math.sin(angle_radians)
    cosine = math.cos(angle_radians)
    cross = skew_symmetric(axis)
    return (
        np.eye(3, dtype=float) * cosine
        + (1.0 - cosine) * np.outer(axis, axis)
        + sine * cross
    )


def rotation_about_axis(
    angle_degrees: float,
    pivot: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Build a rigid transform rotating around a 3-D line through *pivot*."""
    pivot = np.asarray(pivot, dtype=float)
    rotation = rodrigues_rotation(axis, angle_degrees)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot - rotation @ pivot
    return transform


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def transform_delta(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    correction = candidate @ np.linalg.inv(reference)
    return (
        float(np.linalg.norm(correction[:3, 3])),
        rotation_angle_degrees(correction[:3, :3]),
    )


def relative_camera_transform(
    source_camera_to_reference: np.ndarray,
    target_camera_to_reference: np.ndarray,
) -> np.ndarray:
    """Map source-camera points into target-camera coordinates."""
    return np.linalg.inv(target_camera_to_reference) @ source_camera_to_reference


def transformed_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def points_inside_bounds(
    points: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )


# ===================================================================
# CloudCompare helpers
# ===================================================================

def cc_file_argument(path: Path) -> str:
    """Literal-quote a path for CloudCompare SAVE_CLOUDS FILE."""
    return f'"{path}"'


def cc_crop_argument(
    bounds: tuple[float, float, float, float, float, float],
) -> str:
    return ":".join(f"{v:.10g}" for v in bounds)


def run_cc_command(command: list[str]) -> None:
    logger.info("$ %s", " ".join(command))
    subprocess.run(command, check=True)


# ===================================================================
# File discovery
# ===================================================================

def read_angle_from_filename(filename: str) -> float:
    """Extract angle from filenames like frame_10.0.ply or frame_-20.0.ply."""
    match = re.search(r"frame_([-]?\d+\.?\d*)", filename)
    if match is None:
        raise ValueError(f"Cannot extract angle from: {filename}")
    return float(match.group(1))


def find_ply_files(input_dir: Path) -> list[Path]:
    """Find per-angle PLY files and order them by angle."""
    files = list(input_dir.glob("frame_*.ply"))
    if not files:
        raise RuntimeError(f"No frame_*.ply files found in {input_dir}")
    return sorted(files, key=lambda p: read_angle_from_filename(p.name))


def load_scan_metadata(input_dir: Path) -> dict:
    """Load capture settings when the scan was produced by main_scan.py."""
    metadata_path = input_dir / "scan_metadata.json"
    if not metadata_path.is_file():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}")
    return metadata


def resolve_orbit_radius(cli_radius_m: Optional[float], metadata: dict) -> float:
    """Resolve and validate the effective camera-to-orbit-center radius."""
    value = cli_radius_m
    if value is None:
        value = metadata.get("orbit_radius_m")
    if value is None:
        raise ValueError(
            "Orbit radius is required. Pass --orbit-radius-m or reconstruct "
            "a scan containing scan_metadata.json."
        )
    radius_m = float(value)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("Orbit radius must be a finite positive number of metres.")
    return radius_m


def calculate_auto_radius(o3d, ply_files: list[Path]) -> float:
    """Read frame_0.0.ply (or first frame), find the hand, and return its Z depth."""
    target = next((p for p in ply_files if "frame_0.0.ply" in p.name), ply_files[0])
    cloud = o3d.io.read_point_cloud(str(target))
    if cloud.is_empty():
        raise RuntimeError(f"Could not read points for auto-radius from {target}")
    
    pts = np.asarray(cloud.points, dtype=float)
    finite = pts[np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0.01)]
    if len(finite) < 100:
        raise RuntimeError("Not enough points to calculate auto-radius.")

    # Find center 10%
    cx = (finite[:, 0].max() + finite[:, 0].min()) / 2
    cy = (finite[:, 1].max() + finite[:, 1].min()) / 2
    x_range = finite[:, 0].max() - finite[:, 0].min()
    y_range = finite[:, 1].max() - finite[:, 1].min()
    
    mask = (
        (np.abs(finite[:, 0] - cx) < x_range * 0.10) &
        (np.abs(finite[:, 1] - cy) < y_range * 0.10)
    )
    center_pts = finite[mask]
    if len(center_pts) == 0:
        center_pts = finite # fallback if crop is empty
        
    radius = float(np.median(center_pts[:, 2]))
    logger.warning(
        "Estimated visible-surface depth from %s: %.3fm. This is not a "
        "calibrated optical-center orbit radius.",
        target.name,
        radius,
    )
    return radius


# ===================================================================

# Orbit pose builder
# ===================================================================

def build_orbit_poses(
    ply_files: list[Path],
    orbit_axis: np.ndarray,
    pivot: np.ndarray,
    reference_angle_deg: float,
    angle_sign: float,
) -> list[np.ndarray]:
    """Build motor-angle pose priors."""
    poses = []
    for path in ply_files:
        angle = read_angle_from_filename(path.name)
        relative_angle = angle_sign * (angle - reference_angle_deg)
        # Normalize to [-180, 180)
        relative_angle = (relative_angle + 180.0) % 360.0 - 180.0
        poses.append(rotation_about_axis(relative_angle, pivot, orbit_axis))
    return poses


# ===================================================================
# Registration
# ===================================================================

def prepare_registration_cloud(o3d, cloud, initial_pose, voxel_size=None, crop_bounds=None):
    """Build a downsampled (and optionally cropped) cloud for registration."""
    if voxel_size is None:
        voxel_size = REGISTRATION_VOXEL_M
    points = np.asarray(cloud.points, dtype=float)
    if points.size == 0:
        return o3d.geometry.PointCloud()

    finite = np.all(np.isfinite(points), axis=1)
    keep = finite

    if crop_bounds is not None:
        points_in_reference = transformed_points(points, initial_pose)
        within_crop = points_inside_bounds(points_in_reference, crop_bounds)
        keep = finite & within_crop

    selected_indices = np.flatnonzero(keep)
    selected = cloud.select_by_index(selected_indices.tolist())
    selected = selected.voxel_down_sample(voxel_size)
    if len(selected.points) > PRE_ICP_SOR_NEIGHBORS:
        selected, _ = selected.remove_statistical_outlier(
            nb_neighbors=PRE_ICP_SOR_NEIGHBORS,
            std_ratio=PRE_ICP_SOR_SIGMA,
        )
    if len(selected.points) >= 3:
        selected.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=REGISTRATION_NORMAL_RADIUS_M,
                max_nn=50,
            )
        )
    return selected


def registration_result_is_acceptable(
    *,
    fitness: float,
    rmse_m: float,
    prior_fitness: float,
    prior_rmse_m: float,
    prior: np.ndarray,
    candidate: np.ndarray,
) -> tuple[bool, str, float, float]:
    correction_m, correction_deg = transform_delta(prior, candidate)
    if not np.isfinite(fitness) or not np.isfinite(rmse_m):
        return False, "non-finite ICP score", correction_m, correction_deg
    if fitness < ICP_MIN_FITNESS:
        return False, "fitness below threshold", correction_m, correction_deg
    if rmse_m > ICP_MAX_RMSE_M:
        return False, "RMSE exceeds threshold", correction_m, correction_deg
    if correction_m > ICP_MAX_CORRECTION_M:
        return False, "translation correction exceeds prior guard", correction_m, correction_deg
    if correction_deg > ICP_MAX_CORRECTION_DEG:
        return False, "rotation correction exceeds prior guard", correction_m, correction_deg
    prior_scores_are_finite = (
        np.isfinite(prior_fitness) and np.isfinite(prior_rmse_m)
    )
    if prior_scores_are_finite:
        fitness_not_worse = fitness >= prior_fitness - ICP_MAX_FITNESS_DROP
        fitness_improved = fitness >= prior_fitness + ICP_MIN_FITNESS_GAIN
        rmse_improved = rmse_m <= prior_rmse_m * ICP_RMSE_IMPROVEMENT_RATIO
        if not fitness_not_worse or not (fitness_improved or rmse_improved):
            return False, "ICP did not improve the pose prior", correction_m, correction_deg
    return True, "accepted", correction_m, correction_deg


def information_matrix(o3d, source, target, transform):
    if len(source.points) < 3 or len(target.points) < 3:
        return np.eye(6, dtype=float)
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, ICP_FINE_DISTANCE_M, transform,
    )
    info = np.asarray(info, dtype=float)
    if info.shape != (6, 6) or not np.all(np.isfinite(info)) or np.linalg.norm(info) == 0.0:
        return np.eye(6, dtype=float)
    return info


def register_pair(o3d, source_id, target_id, kind, frames):
    """Guarded coarse (PointToPoint) → fine (PointToPlane) ICP between two frames."""
    source = frames[source_id].cloud
    target = frames[target_id].cloud
    prior = relative_camera_transform(
        frames[source_id].prior_pose,
        frames[target_id].prior_pose,
    )

    if len(source.points) < ICP_MIN_POINTS or len(target.points) < ICP_MIN_POINTS:
        return RegistrationEdge(
            source_id=source_id, target_id=target_id, kind=kind,
            transform=prior,
            information=information_matrix(o3d, source, target, prior),
            accepted=False, reason="registration cloud too small; used pose prior",
            fitness=0.0, rmse_m=float("inf"),
            correction_m=0.0, correction_deg=0.0, usable=False,
        )

    # Evaluate the prior itself
    prior_eval = o3d.pipelines.registration.evaluate_registration(
        source, target, ICP_FINE_DISTANCE_M, prior,
    )
    prior_fitness = float(prior_eval.fitness)
    prior_rmse_m = float(prior_eval.inlier_rmse)
    prior_usable = (
        np.isfinite(prior_fitness)
        and np.isfinite(prior_rmse_m)
        and prior_fitness >= ICP_MIN_FITNESS
        and prior_rmse_m <= ICP_MAX_RMSE_M
    )

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITERATIONS)

    # Coarse pass: PointToPoint with a narrow search around the motor prior
    coarse = o3d.pipelines.registration.registration_icp(
        source, target, ICP_COARSE_DISTANCE_M, prior,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )
    # Fine pass: PointToPlane (uses normals for sub-mm accuracy)
    fine = o3d.pipelines.registration.registration_icp(
        source, target, ICP_FINE_DISTANCE_M, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )

    candidate = np.asarray(fine.transformation, dtype=float)
    accepted, reason, correction_m, correction_deg = registration_result_is_acceptable(
        fitness=float(fine.fitness),
        rmse_m=float(fine.inlier_rmse),
        prior_fitness=prior_fitness,
        prior_rmse_m=prior_rmse_m,
        prior=prior,
        candidate=candidate,
    )
    transform = candidate if accepted else prior
    if not accepted:
        reason = f"{reason}; used pose prior"

    return RegistrationEdge(
        source_id=source_id, target_id=target_id, kind=kind,
        transform=transform,
        information=information_matrix(o3d, source, target, transform),
        accepted=accepted, reason=reason,
        fitness=float(fine.fitness), rmse_m=float(fine.inlier_rmse),
        correction_m=correction_m, correction_deg=correction_deg,
        usable=accepted or prior_usable,
        prior_fitness=prior_fitness, prior_rmse_m=prior_rmse_m,
    )


def registration_pairs(frame_count, include_loop_closure=True):
    """Return sequential pairs and an optional loop-closure pair."""
    pairs = [(i, i + 1, "sequential") for i in range(frame_count - 1)]
    if include_loop_closure and frame_count > 2:
        pairs.append((0, frame_count - 1, "loop"))
    return pairs


def build_pose_graph(o3d, frames):
    """Build and optimize a pose graph with adaptive orbit priors + parallel ICP."""
    graph = o3d.pipelines.registration.PoseGraph()
    for frame in frames:
        graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(frame.prior_pose.copy())
        )

    all_pairs = registration_pairs(len(frames))

    # Run all sequential pairs in parallel (ICP releases the GIL)
    sequential_pairs = [(s, t, k) for s, t, k in all_pairs if k == "sequential"]
    loop_pairs = [(s, t, k) for s, t, k in all_pairs if k != "sequential"]

    edges = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(register_pair, o3d, s, t, k, frames): (s, t, k)
            for s, t, k in sequential_pairs
        }
        seq_edges = {}
        for future in concurrent.futures.as_completed(futures):
            edge = future.result()
            seq_edges[(edge.source_id, edge.target_id)] = edge

    # Keep sequential edges in order
    for s, t, _ in sequential_pairs:
        edges.append(seq_edges[(s, t)])

    # Loop closure runs after (single pair, no parallelism needed)
    for s, t, k in loop_pairs:
        edges.append(register_pair(o3d, s, t, k, frames))

    for edge in edges:
        logger.info(
            "%s %d->%d %s: fitness=%.3f, rmse=%.4f m, "
            "correction=%.4f m/%.2f deg (%s)",
            edge.kind.title(), edge.source_id, edge.target_id,
            "accepted" if edge.accepted else "fallback",
            edge.fitness, edge.rmse_m,
            edge.correction_m, edge.correction_deg, edge.reason,
        )

        # Add orbit prior edge for sequential pairs
        # Adaptive weight: higher when ICP fell back (trust motor angle more)
        if edge.kind == "sequential":
            prior = relative_camera_transform(
                frames[edge.source_id].prior_pose,
                frames[edge.target_id].prior_pose,
            )
            prior_weight = (
                ORBIT_PRIOR_WEIGHT_ACCEPTED if edge.accepted
                else ORBIT_PRIOR_WEIGHT_FALLBACK
            )
            prior_info = information_matrix(
                o3d, frames[edge.source_id].cloud, frames[edge.target_id].cloud, prior,
            ) * prior_weight
            graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    edge.source_id, edge.target_id, prior, prior_info, uncertain=False,
                )
            )
        # Add ICP edge if accepted
        if edge.accepted:
            graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    edge.source_id, edge.target_id, edge.transform, edge.information, uncertain=True,
                )
            )

    return graph, edges


def optimize_pose_graph(o3d, graph):
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=ICP_FINE_DISTANCE_M,
            edge_prune_threshold=POSE_GRAPH_EDGE_PRUNE_THRESHOLD,
            reference_node=0,
        ),
    )
    return [np.asarray(node.pose, dtype=float).copy() for node in graph.nodes]


# ===================================================================
# CloudCompare stages
# ===================================================================

def _transform_single_frame(
    frame: RegistrationFrame,
    optimized_pose: np.ndarray,
    matrix_dir: Path,
    transformed_dir: Path,
    skip_per_scan_sor: bool,
    crop_bounds: Optional[tuple] = None,
) -> Path:
    """Transform a single frame via CloudCompare (used by parallel executor)."""
    matrix_path = matrix_dir / f"{frame.path.stem}_optimized_matrix.txt"
    output_path = transformed_dir / f"{frame.path.stem}_transformed.ply"
    np.savetxt(matrix_path, optimized_pose, fmt="%.10f")

    command = [
        *CLOUDCOMPARE_COMMAND,
        "-VERBOSITY", "2", "-SILENT", "-AUTO_SAVE", "OFF",
        "-O", str(frame.path),
        "-APPLY_TRANS", str(matrix_path),
    ]
    if crop_bounds is not None:
        command.extend(["-CROP", cc_crop_argument(crop_bounds)])
    if not skip_per_scan_sor:
        command.extend(["-SOR", str(PER_SCAN_SOR_NEIGHBORS), str(PER_SCAN_SOR_SIGMA)])
    command.extend([
        "-C_EXPORT_FMT", "PLY",
        "-SAVE_CLOUDS", "FILE", cc_file_argument(output_path),
    ])

    logger.info("Transforming %s (angle=%.2f deg)", frame.path.name, frame.angle_deg)
    run_cc_command(command)
    return output_path


# Max parallel CC processes (flatpak + Qt overhead is ~750ms each;
# running 4-6 in parallel hides the cold-start latency)
CC_MAX_WORKERS = 4


def cc_transform_and_clean(
    frames: list[RegistrationFrame],
    optimized_poses: list[np.ndarray],
    output_dir: Path,
    matrix_dir: Path,
    skip_per_scan_sor: bool = False,
    crop_bounds: Optional[tuple] = None,
) -> list[Path]:
    """Stage 3: Apply optimized poses via parallel CloudCompare calls."""
    transformed_dir = output_dir / "01_transformed"
    transformed_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Transforming %d clouds (%d parallel CC workers)...",
                len(frames), CC_MAX_WORKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=CC_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _transform_single_frame, frame, pose,
                matrix_dir, transformed_dir, skip_per_scan_sor, crop_bounds
            ): i
            for i, (frame, pose) in enumerate(zip(frames, optimized_poses))
        }
        results = {}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    # Return paths in original frame order
    return [results[i] for i in range(len(frames))]


def cc_merge_clouds(
    transformed_paths: list[Path],
    merged_cloud_path: Path,
    log_path: Path,
) -> None:
    """Stage 4: Merge all transformed scans, dedup, subsample, orient normals."""
    command = [
        *CLOUDCOMPARE_COMMAND,
        "-VERBOSITY", "2", "-SILENT", "-AUTO_SAVE", "OFF",
        "-LOG_FILE", str(log_path),
        "-C_EXPORT_FMT", "PLY",
        "-O", str(transformed_paths[0]),
    ]

    for incoming_path in transformed_paths[1:]:
        command.extend(["-O", str(incoming_path), "-MERGE_CLOUDS"])

    command.extend([
        # Subsample FIRST to reduce point count from millions to thousands
        "-SS", "SPATIAL", str(SPATIAL_SUBSAMPLE_M),
        # Remove duplicates
        "-RDP", str(REMOVE_DUPLICATES_DISTANCE_M),
        # Run SOR on the much smaller, subsampled cloud
        "-SOR", str(FINAL_SOR_NEIGHBORS), str(FINAL_SOR_SIGMA),
        # Finally, estimate and orient normals
        "-OCTREE_NORMALS", str(NORMAL_RADIUS_M),
        "-ORIENT", "PLUS_BARYCENTER",
        "-ORIENT_NORMS_MST", str(NORMAL_MST_NEIGHBORS),
        "-SAVE_CLOUDS", "FILE", cc_file_argument(merged_cloud_path),
    ])

    logger.info("Merging %d clouds...", len(transformed_paths))
    run_cc_command(command)




# ===================================================================
# Diagnostics
# ===================================================================

def build_diagnostics(
    frames: list[RegistrationFrame],
    optimized_poses: list[np.ndarray],
    edges: list[RegistrationEdge],
    settings: dict,
) -> dict:
    edge_dicts = []
    for edge in edges:
        edge_dicts.append({
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "kind": edge.kind,
            "accepted": edge.accepted,
            "reason": edge.reason,
            "fitness": edge.fitness,
            "rmse_m": edge.rmse_m if np.isfinite(edge.rmse_m) else None,
            "correction_m": edge.correction_m,
            "correction_deg": edge.correction_deg,
            "usable": edge.accepted or edge.usable,
            "prior_fitness": (
                edge.prior_fitness if np.isfinite(edge.prior_fitness) else None
            ),
            "prior_rmse_m": (
                edge.prior_rmse_m if np.isfinite(edge.prior_rmse_m) else None
            ),
        })

    corrections = []
    for frame, opt_pose in zip(frames, optimized_poses):
        corr_m, corr_deg = transform_delta(frame.prior_pose, opt_pose)
        if corr_m < 1e-12:
            corr_m = 0.0
        if corr_deg < 1e-9:
            corr_deg = 0.0
        corrections.append({
            "filename": frame.path.name,
            "angle_deg": frame.angle_deg,
            "translation_m": corr_m,
            "rotation_deg": corr_deg,
        })

    return {
        "capture_count": len(frames),
        "settings": settings,
        "edges": edge_dicts,
        "optimized_pose_corrections": corrections,
    }


def save_diagnostics(
    output_dir: Path,
    frames: list[RegistrationFrame],
    optimized_poses: list[np.ndarray],
    edges: list[RegistrationEdge],
    settings: dict,
) -> None:
    diagnostics = build_diagnostics(frames, optimized_poses, edges, settings)

    diag_path = output_dir / "registration_diagnostics.json"
    diag_path.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved diagnostics to %s", diag_path)


# ===================================================================
# Main pipeline
# ===================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reconstruct a merged point cloud from per-angle PLY captures."
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing frame_*.ply files (e.g. outputs/debug)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Reconstruction output directory (default: <input-dir>/reconstruction)")
    parser.add_argument(
        "--registration-mode",
        choices=("motor", "guarded-icp"),
        default="motor",
        help="Use deterministic motor poses or guarded ICP residual refinement (default: motor)",
    )
    parser.add_argument("--auto-radius", action="store_true",
                        help="Estimate center-surface depth from the first frame (rough fallback; "
                             "not a physical orbit-radius calibration)")
    parser.add_argument("--orbit-radius-m", type=float, default=None,
                        help="Camera orbit radius in metres; defaults to scan_metadata.json "
                             "(ignored if --auto-radius is used)")
    parser.add_argument("--orbit-axis", type=float, nargs=3, default=None,
                        help="Orbit axis as X Y Z; defaults to scan metadata or 1 0 0")
    parser.add_argument("--pivot", type=float, nargs=3, default=None,
                        help="Pivot point in camera coords as X Y Z metres "
                             "(default: [0, 0, orbit-radius-m] = object centre in front of camera)")
    parser.add_argument("--reference-angle-deg", type=float, default=0.0,
                        help="Angle of the reference frame (default: 0)")
    parser.add_argument("--angle-sign", type=float, default=1.0,
                        help="Sign convention for angle direction (1.0 or -1.0)")
    parser.add_argument("--crop-radius-m", type=float, default=None,
                        help="Half-extent of the object crop cube around the pivot "
                             "(defaults to scan metadata or 0.15m; set <= 0 to disable)")

    parser.add_argument("--skip-per-scan-sor", action="store_true",
                        help="Skip per-scan SOR (much faster; final SOR still runs)")
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / "reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    ply_files = find_ply_files(input_dir)
    logger.info("Found %d PLY files in %s", len(ply_files), input_dir)
    scan_metadata = load_scan_metadata(input_dir)

    orbit_axis_values = args.orbit_axis or scan_metadata.get(
        "orbit_axis", [1.0, 0.0, 0.0]
    )
    orbit_axis = normalized_vector(np.array(orbit_axis_values), name="orbit-axis")

    if args.auto_radius:
        import open3d as o3d

        orbit_radius_m = calculate_auto_radius(o3d, ply_files)
    else:
        orbit_radius_m = resolve_orbit_radius(args.orbit_radius_m, scan_metadata)

    # Pivot = where the hand/object sits in camera space at angle=0.
    # Camera is on the ring, Z+ axis points inward at the object.
    # Object is at [0, 0, orbit_radius_m] in camera coordinates.
    pivot = np.array(args.pivot) if args.pivot else np.array([0.0, 0.0, orbit_radius_m])

    reconstruction_metadata = scan_metadata.get("reconstruction", {})
    if not isinstance(reconstruction_metadata, dict):
        reconstruction_metadata = {}
    crop_radius_m = args.crop_radius_m
    if crop_radius_m is None:
        crop_radius_m = float(reconstruction_metadata.get("crop_radius_m", 0.15))

    r = crop_radius_m
    crop_bounds = (pivot[0] - r, pivot[1] - r, pivot[2] - r,
                   pivot[0] + r, pivot[1] + r, pivot[2] + r) if r > 0 else None


    merged_cloud_path = output_dir / "merged_cloud.ply"
    log_path = output_dir / "cloudcompare.log"

    # Stage 1: Discover files and build orbit poses
    logger.info("Stage 1/4: Discovering PLY files and building orbit poses...")
    logger.info("Found %d point clouds.", len(ply_files))
    logger.info("Orbit radius: %.4f m, axis: %s, pivot: %s", orbit_radius_m, orbit_axis, pivot)

    priors = build_orbit_poses(
        ply_files, orbit_axis, pivot,
        reference_angle_deg=args.reference_angle_deg,
        angle_sign=args.angle_sign,
    )

    frames = [
        RegistrationFrame(
            path=path,
            angle_deg=read_angle_from_filename(path.name),
            prior_pose=prior,
        )
        for path, prior in zip(ply_files, priors)
    ]

    if args.registration_mode == "motor":
        logger.info("Stage 2/4: Using deterministic motor poses (ICP disabled).")
        optimized_poses = [prior.copy() for prior in priors]
        edges = []
    else:
        logger.info("Stage 2/4: Guarded ICP and pose-graph optimization...")
        import open3d as o3d

        for i, frame in enumerate(frames, start=1):
            cloud = o3d.io.read_point_cloud(str(frame.path))
            if cloud.is_empty():
                raise RuntimeError(f"Open3D could not read points from {frame.path}")
            frame.cloud = prepare_registration_cloud(
                o3d,
                cloud,
                frame.prior_pose,
                voxel_size=REGISTRATION_VOXEL_M,
                crop_bounds=crop_bounds,
            )
            logger.info(
                "Loaded %d/%d %s: %d registration points",
                i,
                len(frames),
                frame.path.name,
                len(frame.cloud.points),
            )

        graph, edges = build_pose_graph(o3d, frames)
        optimized_poses = optimize_pose_graph(o3d, graph)

    # Save pose matrices
    np.save(output_dir / "optimized_poses.npy", np.stack(optimized_poses))
    for frame, opt_pose in zip(frames, optimized_poses):
        np.savetxt(matrix_dir / f"{frame.path.stem}_prior.txt", frame.prior_pose, fmt="%.10f")
        np.savetxt(matrix_dir / f"{frame.path.stem}_optimized.txt", opt_pose, fmt="%.10f")

    settings = {
        "registration_mode": args.registration_mode,
        "orbit_radius_m": orbit_radius_m,
        "orbit_radius_source": "auto" if args.auto_radius else (
            "cli" if args.orbit_radius_m is not None else "scan_metadata"
        ),
        "orbit_axis": orbit_axis.tolist(),
        "pivot": pivot.tolist(),
        "reference_angle_deg": args.reference_angle_deg,
        "angle_sign": args.angle_sign,
        "crop_radius_m": crop_radius_m,
        "per_scan_sor_enabled": not args.skip_per_scan_sor,
        "registration_parameters": {
            "voxel_m": REGISTRATION_VOXEL_M,
            "normal_radius_m": REGISTRATION_NORMAL_RADIUS_M,
            "coarse_distance_m": ICP_COARSE_DISTANCE_M,
            "fine_distance_m": ICP_FINE_DISTANCE_M,
            "iterations": ICP_ITERATIONS,
            "min_fitness": ICP_MIN_FITNESS,
            "max_rmse_m": ICP_MAX_RMSE_M,
            "max_correction_m": ICP_MAX_CORRECTION_M,
            "max_correction_deg": ICP_MAX_CORRECTION_DEG,
            "max_fitness_drop": ICP_MAX_FITNESS_DROP,
            "min_fitness_gain": ICP_MIN_FITNESS_GAIN,
            "rmse_improvement_ratio": ICP_RMSE_IMPROVEMENT_RATIO,
            "min_points": ICP_MIN_POINTS,
            "pre_icp_sor_neighbors": PRE_ICP_SOR_NEIGHBORS,
            "pre_icp_sor_sigma": PRE_ICP_SOR_SIGMA,
            "orbit_prior_weight_accepted": ORBIT_PRIOR_WEIGHT_ACCEPTED,
            "orbit_prior_weight_fallback": ORBIT_PRIOR_WEIGHT_FALLBACK,
            "pose_graph_edge_prune_threshold": POSE_GRAPH_EDGE_PRUNE_THRESHOLD,
        },
        "cleanup_parameters": {
            "per_scan_sor_neighbors": PER_SCAN_SOR_NEIGHBORS,
            "per_scan_sor_sigma": PER_SCAN_SOR_SIGMA,
            "final_sor_neighbors": FINAL_SOR_NEIGHBORS,
            "final_sor_sigma": FINAL_SOR_SIGMA,
            "remove_duplicates_distance_m": REMOVE_DUPLICATES_DISTANCE_M,
            "spatial_subsample_m": SPATIAL_SUBSAMPLE_M,
            "normal_radius_m": NORMAL_RADIUS_M,
            "normal_mst_neighbors": NORMAL_MST_NEIGHBORS,
        },
    }
    save_diagnostics(output_dir, frames, optimized_poses, edges, settings)

    # Stage 3: CloudCompare transform (batched single call)
    logger.info("Stage 3/4: CloudCompare batched transform...")
    transformed_paths = cc_transform_and_clean(
        frames, optimized_poses, output_dir, matrix_dir,
        skip_per_scan_sor=args.skip_per_scan_sor, crop_bounds=crop_bounds
    )

    # Stage 4: CloudCompare merge
    logger.info("Stage 4/4: CloudCompare merge + final cleanup...")
    cc_merge_clouds(transformed_paths, merged_cloud_path, log_path)

    logger.info("Finished!")
    logger.info("Merged cloud: %s", merged_cloud_path)
    logger.info("Diagnostics: %s", output_dir / "registration_diagnostics.json")


if __name__ == "__main__":
    main()
