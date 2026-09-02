#!/usr/bin/env python3
"""Open3D/Trimesh point-cloud reconstruction pipeline for sync_workflow.

Adapted from scripts/transform_clouds_pipeline.py.  Operates on per-angle
PLY files saved by main_scan.py and produces a merged, cleaned point cloud
plus registration diagnostics.

Stages:
  1. Build orbit pose priors from motor angles
  2. Use motor poses, or optional guarded ICP + pose-graph optimization
  3. Open3D: apply optimized transforms, crop, SOR per scan
  4. Open3D/Trimesh: merge, dedup, subsample, orient normals, validate
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from calculating_radius.orbit_pose_map import camera_poses_in_reference
from pointcloud_processing import (
    merge_and_finalize_clouds,
    transform_and_clean_clouds,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default registration settings
# ---------------------------------------------------------------------------
REGISTRATION_VOXEL_M = 0.002
REGISTRATION_NORMAL_RADIUS_M = 0.006
ICP_COARSE_DISTANCE_M = 0.006
ICP_FINE_DISTANCE_M = 0.002
ICP_ITERATIONS = 100
ICP_MIN_FITNESS = 0.35
ICP_MAX_RMSE_M = 0.0015
ICP_MAX_CORRECTION_M = 0.005
ICP_MAX_CORRECTION_DEG = 2.5
ICP_MAX_FITNESS_DROP = 0.01
ICP_MIN_FITNESS_GAIN = 0.02
ICP_RMSE_IMPROVEMENT_RATIO = 0.98
ICP_MIN_POINTS = 100
ORBIT_PRIOR_WEIGHT_ACCEPTED = 50.0  # keep accepted ICP anchored to the motor prior
ORBIT_PRIOR_WEIGHT_FALLBACK = 200.0  # strengthen the motor prior when ICP fails
POSE_GRAPH_EDGE_PRUNE_THRESHOLD = 0.25

# Registration cleanup
PRE_ICP_SOR_NEIGHBORS = 20
PRE_ICP_SOR_SIGMA = 1.5

DEFAULT_FINAL_CROP_RADIUS_M = 0.15
DEFAULT_REGISTRATION_CROP_RADIUS_M = 0.10

# Full-resolution per-scan cleanup after pose estimation
PER_SCAN_SOR_NEIGHBORS = 10
PER_SCAN_SOR_SIGMA = 2.0

# Final merged-cloud cleanup
FINAL_SOR_NEIGHBORS = 20
FINAL_SOR_SIGMA = 1.5
REMOVE_DUPLICATES_DISTANCE_M = 0.00010
SPATIAL_SUBSAMPLE_M = 0.001
NORMAL_RADIUS_M = 0.0040
NORMAL_MAX_NEIGHBORS = 50
NORMAL_MST_NEIGHBORS = 12
PROCESSING_MAX_WORKERS = 4

# ===================================================================
# Data classes
# ===================================================================

@dataclass
class CaptureRecord:
    """Capture metadata needed to place one point cloud."""
    path: Path
    angle_deg: float
    station_index: int = 0
    x_position_mm: float = 200.0
    x_offset_m: float = 0.0


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
    station_index: int = 0
    x_position_mm: float = 200.0
    x_offset_m: float = 0.0
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
# File discovery
# ===================================================================

def read_angle_from_filename(filename: str) -> float:
    """Extract the Y angle from legacy or station-aware capture names."""
    match = re.search(r"_y([+-]?\d+\.?\d*)", filename)
    if match is None:
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


def load_orbit_pose_map(path: Path) -> dict:
    """Load a full-pose calibration produced by the orbit-pose capture tool."""
    if not path.is_file():
        raise FileNotFoundError(f"Orbit pose map not found: {path}")
    pose_map = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pose_map, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if int(pose_map.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported orbit pose-map schema in {path}")
    return pose_map


def validate_pose_map_coordinate_frame(input_dir: Path, pose_map: dict) -> None:
    """Prevent applying color-camera poses to depth-camera point clouds or vice versa."""
    intrinsics_path = input_dir / "intrinsics.json"
    if not intrinsics_path.is_file():
        return
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    scan_frame = intrinsics.get("coordinate_frame") if isinstance(intrinsics, dict) else None
    pose_frame = pose_map.get("pointcloud_coordinate_frame")
    if scan_frame and pose_frame and scan_frame != pose_frame:
        raise ValueError(
            f"Point-cloud coordinate-frame mismatch: scan uses {scan_frame!r}, "
            f"pose map uses {pose_frame!r}"
        )


def discover_captures(input_dir: Path, metadata: dict) -> list[CaptureRecord]:
    """Load schema-v2 capture records, with legacy filename compatibility."""
    manifest = metadata.get("captures")
    if manifest is not None:
        if not isinstance(manifest, list):
            raise ValueError("scan_metadata.json captures must be a list")
        if not manifest:
            raise RuntimeError("scan_metadata.json contains no completed captures")
        records = []
        seen_filenames = set()
        for item in manifest:
            if not isinstance(item, dict):
                raise ValueError("Each capture manifest entry must be an object")
            filename = str(item.get("filename", ""))
            if not filename or Path(filename).name != filename:
                raise ValueError(f"Invalid capture filename: {filename!r}")
            if filename in seen_filenames:
                raise ValueError(f"Duplicate capture filename: {filename}")
            path = input_dir / filename
            if not path.is_file():
                raise RuntimeError(f"Capture listed in metadata is missing: {path}")
            seen_filenames.add(filename)
            records.append(CaptureRecord(
                path=path,
                angle_deg=float(item["angle_deg"]),
                station_index=int(item["station_index"]),
                x_position_mm=float(item["x_position_mm"]),
                x_offset_m=float(item["x_offset_m"]),
            ))
        if not all(
            np.isfinite((record.angle_deg, record.x_position_mm, record.x_offset_m)).all()
            for record in records
        ):
            raise ValueError("Capture geometry must contain finite values")
        return sorted(records, key=lambda record: (record.station_index, record.angle_deg))

    return [
        CaptureRecord(path=path, angle_deg=read_angle_from_filename(path.name))
        for path in find_ply_files(input_dir)
    ]


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


def resolve_crop_radii(
    cli_final_crop_m: Optional[float],
    cli_registration_crop_m: Optional[float],
    reconstruction_metadata: dict,
) -> tuple[float, float]:
    """Resolve independent full-resolution and ICP crop half-extents.

    Legacy metadata contains only ``crop_radius_m``. In that case, ICP uses at
    most a 10 cm cube half-extent so enclosure/background geometry cannot
    dominate registration, while the final output retains its requested crop.
    """
    if not isinstance(reconstruction_metadata, dict):
        reconstruction_metadata = {}

    final_crop_m = cli_final_crop_m
    if final_crop_m is None:
        final_crop_m = reconstruction_metadata.get(
            "crop_radius_m", DEFAULT_FINAL_CROP_RADIUS_M,
        )
    final_crop_m = float(final_crop_m)

    registration_crop_m = cli_registration_crop_m
    if registration_crop_m is None:
        registration_crop_m = reconstruction_metadata.get(
            "registration_crop_radius_m"
        )
    if registration_crop_m is None:
        registration_crop_m = DEFAULT_REGISTRATION_CROP_RADIUS_M
        if final_crop_m > 0.0:
            registration_crop_m = min(registration_crop_m, final_crop_m)
    registration_crop_m = float(registration_crop_m)

    if not np.isfinite(final_crop_m):
        raise ValueError("Final crop radius must be finite.")
    if not np.isfinite(registration_crop_m):
        raise ValueError("Registration crop radius must be finite.")
    return final_crop_m, registration_crop_m


def calculate_auto_radius(o3d, ply_files: list[Path]) -> float:
    """Read frame_0.0.ply (or first frame), find the hand, and return its Z depth."""
    target = next(
        (p for p in ply_files if abs(read_angle_from_filename(p.name)) < 1e-9),
        ply_files[0],
    )
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
    captures: list[CaptureRecord],
    orbit_axis: np.ndarray,
    pivot: np.ndarray,
    reference_angle_deg: float,
    angle_sign: float,
) -> list[np.ndarray]:
    """Build motor-angle pose priors."""
    poses = []
    for capture in captures:
        angle = capture.angle_deg
        relative_angle = angle_sign * (angle - reference_angle_deg)
        # Normalize to [-180, 180)
        relative_angle = (relative_angle + 180.0) % 360.0 - 180.0
        pose = rotation_about_axis(relative_angle, pivot, orbit_axis)
        pose[:3, 3] -= orbit_axis * capture.x_offset_m
        poses.append(pose)
    return poses


def build_measured_pose_priors(
    captures: list[CaptureRecord], pose_map: dict
) -> list[np.ndarray]:
    """Resolve ArUco-measured camera-to-reference poses for one X station."""
    calibration_x_mm = float(pose_map["x_position_mm"])
    capture_stations = {capture.station_index for capture in captures}
    if len(capture_stations) != 1:
        raise ValueError(
            "An ArUco pose map currently supports one X station per reconstruction"
        )
    mismatched = [
        capture.x_position_mm
        for capture in captures
        if not np.isclose(capture.x_position_mm, calibration_x_mm, atol=0.05)
    ]
    if mismatched:
        raise ValueError(
            f"Pose map was captured at X={calibration_x_mm:.1f} mm, but the scan "
            f"contains X={mismatched[0]:.1f} mm. Capture the pose map at the same X."
        )
    return camera_poses_in_reference(
        pose_map, [capture.angle_deg for capture in captures]
    )


def crop_bounds_around(center: np.ndarray, half_extent_m: float):
    """Return an axis-aligned crop cube around a station's orbit pivot."""
    if half_extent_m <= 0.0:
        return None
    center = np.asarray(center, dtype=float)
    lower = center - half_extent_m
    upper = center + half_extent_m
    return (*lower.tolist(), *upper.tolist())


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


def registration_pairs(frames, include_loop_closure=True):
    """Return within-station orbit pairs and same-angle station links."""
    if isinstance(frames, int):
        pairs = [(i, i + 1, "sequential") for i in range(frames - 1)]
        if include_loop_closure and frames > 2:
            pairs.append((0, frames - 1, "loop"))
        return pairs

    pairs = []
    station_groups = {}
    for index, frame in enumerate(frames):
        station_groups.setdefault(frame.station_index, []).append(index)
    ordered_stations = sorted(station_groups)
    for station_index in ordered_stations:
        indices = station_groups[station_index]
        pairs.extend(
            (source, target, "sequential")
            for source, target in zip(indices, indices[1:])
        )
        if include_loop_closure and len(indices) > 2:
            pairs.append((indices[0], indices[-1], "loop"))

    for first_station, second_station in zip(ordered_stations, ordered_stations[1:]):
        first_by_angle = {
            round(frames[index].angle_deg, 9): index
            for index in station_groups[first_station]
        }
        for second_index in station_groups[second_station]:
            first_index = first_by_angle.get(round(frames[second_index].angle_deg, 9))
            if first_index is not None:
                pairs.append((first_index, second_index, "station"))
    return pairs


def build_pose_graph(o3d, frames):
    """Build and optimize a pose graph with adaptive orbit priors + parallel ICP."""
    graph = o3d.pipelines.registration.PoseGraph()
    for frame in frames:
        graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(frame.prior_pose.copy())
        )

    all_pairs = registration_pairs(frames)

    # Run sequential and cross-station motion pairs in parallel (ICP releases the GIL)
    motion_pairs = [(s, t, k) for s, t, k in all_pairs if k != "loop"]
    loop_pairs = [(s, t, k) for s, t, k in all_pairs if k == "loop"]

    edges = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(register_pair, o3d, s, t, k, frames): (s, t, k)
            for s, t, k in motion_pairs
        }
        seq_edges = {}
        for future in concurrent.futures.as_completed(futures):
            edge = future.result()
            seq_edges[(edge.source_id, edge.target_id)] = edge

    # Keep motion edges in deterministic order
    for s, t, _ in motion_pairs:
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

        # Add a deterministic prior for orbit and cross-station motor motion.
        # Adaptive weight is higher when ICP falls back.
        if edge.kind in ("sequential", "station"):
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
        source_frame = frames[edge.source_id]
        target_frame = frames[edge.target_id]
        edge_dicts.append({
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "source_filename": source_frame.path.name,
            "target_filename": target_frame.path.name,
            "source_station_index": source_frame.station_index,
            "target_station_index": target_frame.station_index,
            "source_x_position_mm": source_frame.x_position_mm,
            "target_x_position_mm": target_frame.x_position_mm,
            "source_x_offset_m": source_frame.x_offset_m,
            "target_x_offset_m": target_frame.x_offset_m,
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
            "station_index": frame.station_index,
            "x_position_mm": frame.x_position_mm,
            "x_offset_m": frame.x_offset_m,
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
        choices=("motor", "guarded-icp", "aruco", "aruco-guarded-icp"),
        default="motor",
        help=(
            "Pose source: ideal motor orbit, guarded motor+ICP, measured ArUco "
            "poses, or measured ArUco poses with guarded ICP (default: motor)"
        ),
    )
    parser.add_argument(
        "--pose-map",
        type=Path,
        default=None,
        help="orbit_pose_map.json required by the aruco registration modes",
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
    parser.add_argument(
        "--orbit-geometry",
        type=Path,
        default=None,
        help=(
            "radius_calibration.json whose measured orbit_geometry supplies "
            "the pivot and axis instead of [0,0,R] / [1,0,0]"
        ),
    )
    parser.add_argument("--reference-angle-deg", type=float, default=0.0,
                        help="Angle of the reference frame (default: 0)")
    parser.add_argument("--angle-sign", type=float, default=1.0,
                        help="Sign convention for angle direction (1.0 or -1.0)")
    parser.add_argument("--crop-radius-m", type=float, default=None,
                        help="Half-extent of the final output crop cube around the pivot "
                             "(defaults to scan metadata or 0.15m; set <= 0 to disable)")
    parser.add_argument(
        "--registration-crop-radius-m",
        type=float,
        default=None,
        help=(
            "Half-extent of the tighter crop used only by ICP; defaults to "
            "scan metadata or min(final crop, 0.10m); set <= 0 to disable"
        ),
    )

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
    processing_log_path = output_dir / "processing.log"
    file_handler = logging.FileHandler(processing_log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    scan_metadata = load_scan_metadata(input_dir)
    captures = discover_captures(input_dir, scan_metadata)
    ply_files = [capture.path for capture in captures]
    logger.info("Found %d PLY files in %s", len(ply_files), input_dir)

    measured_pose_mode = args.registration_mode in ("aruco", "aruco-guarded-icp")
    pose_map = None
    if measured_pose_mode:
        if args.pose_map is None:
            raise ValueError(
                f"--pose-map is required with --registration-mode {args.registration_mode}"
            )
        if args.auto_radius:
            raise ValueError("--auto-radius cannot be combined with an ArUco pose map")
        pose_map = load_orbit_pose_map(args.pose_map)
        validate_pose_map_coordinate_frame(input_dir, pose_map)

    pose_map_axis = pose_map.get("orbit_axis_reference") if pose_map else None
    orbit_axis_values = args.orbit_axis or pose_map_axis or scan_metadata.get(
        "orbit_axis", [1.0, 0.0, 0.0]
    )
    orbit_axis = normalized_vector(np.array(orbit_axis_values), name="orbit-axis")

    if args.auto_radius:
        import open3d as o3d

        orbit_radius_m = calculate_auto_radius(o3d, ply_files)
    elif measured_pose_mode:
        measured_fit = pose_map.get("orbit_fit_profile_frame") or {}
        measured_radius = measured_fit.get("radius_m")
        if args.orbit_radius_m is not None:
            orbit_radius_m = float(args.orbit_radius_m)
        elif measured_radius is not None:
            orbit_radius_m = float(measured_radius)
        else:
            metadata_radius = scan_metadata.get("orbit_radius_m")
            orbit_radius_m = (
                float(metadata_radius) if metadata_radius is not None else None
            )
    else:
        orbit_radius_m = resolve_orbit_radius(args.orbit_radius_m, scan_metadata)

    measured_geometry = None
    if args.orbit_geometry is not None:
        calibration = json.loads(args.orbit_geometry.read_text(encoding="utf-8"))
        measured_geometry = calibration.get("orbit_geometry")
        if not measured_geometry:
            raise ValueError(
                f"{args.orbit_geometry} has no orbit_geometry block; re-run the "
                "radius calibration with a valid full-orbit result."
            )
        orbit_axis = np.asarray(measured_geometry["axis"], dtype=float)

    if args.pivot:
        pivot = np.array(args.pivot, dtype=float)
    elif measured_geometry is not None:
        pivot = np.asarray(measured_geometry["pivot_m"], dtype=float)
    elif measured_pose_mode:
        pivot = np.asarray(pose_map["profile_origin_in_reference_m"], dtype=float)
    else:
        # Camera is on the ring and Z+ points inward at the object at angle zero.
        pivot = np.array([0.0, 0.0, orbit_radius_m])

    reconstruction_metadata = scan_metadata.get("reconstruction", {})
    if not isinstance(reconstruction_metadata, dict):
        reconstruction_metadata = {}
    crop_radius_m, registration_crop_radius_m = resolve_crop_radii(
        args.crop_radius_m,
        args.registration_crop_radius_m,
        reconstruction_metadata,
    )

    merged_cloud_path = output_dir / "merged_cloud.ply"

    # Stage 1: Discover files and build pose priors
    logger.info("Stage 1/4: Discovering PLY files and building pose priors...")
    logger.info("Found %d point clouds.", len(ply_files))
    if orbit_radius_m is not None:
        logger.info(
            "Orbit radius: %.4f m, axis: %s, pivot: %s",
            orbit_radius_m,
            orbit_axis,
            pivot,
        )
    else:
        logger.info("Measured poses: axis: %s, profile pivot: %s", orbit_axis, pivot)

    if measured_pose_mode:
        priors = build_measured_pose_priors(captures, pose_map)
        logger.info(
            "Using full point-cloud-camera poses measured from %s (reference Y=%+.1f deg).",
            args.pose_map,
            float(pose_map["reference_angle_deg"]),
        )
    else:
        priors = build_orbit_poses(
            captures, orbit_axis, pivot,
            reference_angle_deg=args.reference_angle_deg,
            angle_sign=args.angle_sign,
        )

    frames = [
        RegistrationFrame(
            path=capture.path,
            angle_deg=capture.angle_deg,
            prior_pose=prior,
            station_index=capture.station_index,
            x_position_mm=capture.x_position_mm,
            x_offset_m=capture.x_offset_m,
        )
        for capture, prior in zip(captures, priors)
    ]
    frame_crop_centers = [
        (
            pivot
            if measured_pose_mode
            else pivot - orbit_axis * frame.x_offset_m
        )
        for frame in frames
    ]
    registration_crop_bounds = [
        crop_bounds_around(center, registration_crop_radius_m)
        for center in frame_crop_centers
    ]
    final_crop_bounds = [
        crop_bounds_around(
            center,
            crop_radius_m,
        )
        for center in frame_crop_centers
    ]
    logger.info(
        "Crop half-extents: registration=%s, final=%s",
        (
            "disabled"
            if registration_crop_radius_m <= 0.0
            else f"{registration_crop_radius_m:.3f} m"
        ),
        "disabled" if crop_radius_m <= 0.0 else f"{crop_radius_m:.3f} m",
    )

    if args.registration_mode in ("motor", "aruco"):
        source = "measured ArUco" if measured_pose_mode else "deterministic motor"
        logger.info("Stage 2/4: Using %s poses (ICP disabled).", source)
        optimized_poses = [prior.copy() for prior in priors]
        edges = []
    else:
        logger.info("Stage 2/4: Guarded ICP and pose-graph optimization...")
        import open3d as o3d

        for i, (frame, frame_bounds) in enumerate(
            zip(frames, registration_crop_bounds), start=1,
        ):
            cloud = o3d.io.read_point_cloud(str(frame.path))
            if cloud.is_empty():
                raise RuntimeError(f"Open3D could not read points from {frame.path}")
            frame.cloud = prepare_registration_cloud(
                o3d,
                cloud,
                frame.prior_pose,
                voxel_size=REGISTRATION_VOXEL_M,
                crop_bounds=frame_bounds,
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
        "processing_backend": "open3d+trimesh",
        "registration_mode": args.registration_mode,
        "orbit_radius_m": orbit_radius_m,
        "orbit_radius_source": (
            "auto"
            if args.auto_radius
            else "cli"
            if args.orbit_radius_m is not None
            else "pose_map"
            if measured_pose_mode and pose_map.get("orbit_fit_profile_frame")
            else "scan_metadata"
        ),
        "pose_map": str(args.pose_map) if args.pose_map else None,
        "orbit_geometry_source": (
            str(args.orbit_geometry) if args.orbit_geometry else None
        ),
        "orbit_axis": orbit_axis.tolist(),
        "pivot": pivot.tolist(),
        "reference_angle_deg": (
            float(pose_map["reference_angle_deg"])
            if measured_pose_mode
            else args.reference_angle_deg
        ),
        "angle_sign": args.angle_sign,
        "crop_radius_m": crop_radius_m,
        "registration_crop_radius_m": registration_crop_radius_m,
        "x_stations": [
            {
                "station_index": station_index,
                "x_position_mm": next(
                    frame.x_position_mm for frame in frames
                    if frame.station_index == station_index
                ),
                "x_offset_m": next(
                    frame.x_offset_m for frame in frames
                    if frame.station_index == station_index
                ),
            }
            for station_index in sorted({frame.station_index for frame in frames})
        ],
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
            "normal_max_neighbors": NORMAL_MAX_NEIGHBORS,
            "normal_mst_neighbors": NORMAL_MST_NEIGHBORS,
        },
    }
    save_diagnostics(output_dir, frames, optimized_poses, edges, settings)

    logger.info("Stage 3/4: Open3D full-resolution transform and cleanup...")
    transformed_paths, transform_stats = transform_and_clean_clouds(
        [frame.path for frame in frames],
        optimized_poses,
        output_dir / "01_transformed",
        matrix_dir,
        crop_bounds=None,
        crop_bounds_by_cloud=final_crop_bounds,
        skip_sor=args.skip_per_scan_sor,
        sor_neighbors=PER_SCAN_SOR_NEIGHBORS,
        sor_sigma=PER_SCAN_SOR_SIGMA,
        max_workers=PROCESSING_MAX_WORKERS,
    )

    logger.info("Stage 4/4: Open3D/Trimesh merge and final cleanup...")
    merge_stats = merge_and_finalize_clouds(
        transformed_paths,
        merged_cloud_path,
        pivot=(
            pivot
            if measured_pose_mode
            else pivot
            - orbit_axis * float(np.mean(sorted({frame.x_offset_m for frame in frames})))
        ),
        spatial_subsample_m=SPATIAL_SUBSAMPLE_M,
        duplicate_distance_m=REMOVE_DUPLICATES_DISTANCE_M,
        sor_neighbors=FINAL_SOR_NEIGHBORS,
        sor_sigma=FINAL_SOR_SIGMA,
        normal_radius_m=NORMAL_RADIUS_M,
        normal_max_neighbors=NORMAL_MAX_NEIGHBORS,
        normal_mst_neighbors=NORMAL_MST_NEIGHBORS,
    )
    settings["processing"] = {
        "transformed_frames": transform_stats,
        "merge": merge_stats,
    }
    save_diagnostics(output_dir, frames, optimized_poses, edges, settings)

    logger.info("Finished!")
    logger.info("Merged cloud: %s", merged_cloud_path)
    logger.info("Diagnostics: %s", output_dir / "registration_diagnostics.json")
    logger.info("Processing log: %s", processing_log_path)
    file_handler.flush()
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    main()
