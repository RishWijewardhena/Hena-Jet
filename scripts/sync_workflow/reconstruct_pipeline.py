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

from orbit_geometry import select_orbit_geometry, load_orbit_geometry, validate_camera_frame
import tsdf_fusion
from pointcloud_processing import (
    CylinderCrop,
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

DEFAULT_FINAL_CROP_RADIUS_M = 0.08
DEFAULT_REGISTRATION_CROP_RADIUS_M = 0.10
DEFAULT_CROP_AXIAL_HALF_LENGTH_M = 0.30

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
FINAL_COMPONENT_EPS_M = 0.003
FINAL_COMPONENT_MIN_POINTS = 10
FINAL_COMPONENT_MIN_FRACTION = 0.01
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


def calibration_station_x_mm(calibration: dict, source: Path) -> Optional[float]:
    """Return the X station a calibration was measured at, if it records one.

    The radius and the axis are properties of the mechanism and do not depend on
    X: the camera-to-axis distance is fixed by the mount, and the axis is the X
    direction itself. The pivot is the only part that moves, and it only slides
    along the axis, which leaves the pose priors untouched because rotation
    about a line is invariant to where along that line the pivot sits.

    So one calibration serves every station, provided the pivot is shifted by
    the station difference before it is used as a crop centre.
    """
    motor = calibration.get("motor")
    if not isinstance(motor, dict) or motor.get("x_position_mm") is None:
        logger.warning(
            "%s records no motor X station, so the pivot cannot be shifted to "
            "the scan's station; crop centres assume the scan's own first "
            "station.", source,
        )
        return None
    return float(motor["x_position_mm"])


def resolve_crop_axial_half_length(
    cli_axial_half_length_m: Optional[float],
    reconstruction_metadata: dict,
) -> float:
    """Resolve the along-axis half-length used by the cylindrical crop."""
    value = cli_axial_half_length_m
    if value is None:
        value = reconstruction_metadata.get("crop_axial_half_length_m")
    if value is None:
        value = DEFAULT_CROP_AXIAL_HALF_LENGTH_M
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Crop axial half-length must be a finite positive number of metres."
        )
    return value


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


def crop_bounds_around(
    center: np.ndarray,
    half_extent_m: float,
    *,
    axis: Optional[np.ndarray] = None,
    axial_half_length_m: Optional[float] = None,
):
    """Return a crop region around a station's orbit pivot.

    Without ``axis`` this is the historical axis-aligned cube, where
    ``half_extent_m`` bounds all three axes. With ``axis`` it is a cylinder
    around the orbit axis, where ``half_extent_m`` is the radial limit and
    ``axial_half_length_m`` the extent along the axis.
    """
    if half_extent_m <= 0.0:
        return None
    center = np.asarray(center, dtype=float)
    if axis is not None:
        return CylinderCrop(
            center=tuple(center.tolist()),
            axis=tuple(np.asarray(axis, dtype=float).tolist()),
            radius_m=float(half_extent_m),
            axial_half_length_m=float(
                axial_half_length_m
                if axial_half_length_m is not None
                else DEFAULT_CROP_AXIAL_HALF_LENGTH_M
            ),
        )
    lower = center - half_extent_m
    upper = center + half_extent_m
    return (*lower.tolist(), *upper.tolist())


def points_inside_crop(points: np.ndarray, crop) -> np.ndarray:
    """Return a boolean mask for either crop-region representation."""
    if isinstance(crop, CylinderCrop):
        return crop.mask(points)
    return points_inside_bounds(points, crop)


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
        within_crop = points_inside_crop(points_in_reference, crop_bounds)
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


def high_drift_frame_ids(edges: list[RegistrationEdge]) -> set[int]:
    """Return incoming sequential frames whose ICP candidate exceeded a pose guard.

    Only sequential edges identify one newly arriving capture. Loop-closure and
    cross-station edges constrain two already accepted trajectories, so a large
    correction on either of those edges must not arbitrarily discard a frame.
    """
    return {
        edge.target_id
        for edge in edges
        if edge.kind == "sequential"
        and (
            edge.correction_m > ICP_MAX_CORRECTION_M
            or edge.correction_deg > ICP_MAX_CORRECTION_DEG
        )
    }


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
        choices=("motor", "guarded-icp"),
        default="guarded-icp",
        help=(
            "Pose source: motor orbit or guarded motor+ICP (default: guarded-icp)"
        ),
    )
    parser.add_argument("--orbit-radius-m", type=float, default=None,
                        help="Camera orbit radius in metres; defaults to measured calibration")
    parser.add_argument("--orbit-axis", type=float, nargs=3, default=None,
                        help="Legacy axis option; measured calibration supplies the orbit axis")
    parser.add_argument("--pivot", type=float, nargs=3, default=None,
                        help="Pivot point in camera coords as X Y Z metres "
                             "(default: measured calibration pivot)")
    parser.add_argument(
        "--orbit-geometry",
        type=Path,
        default=None,
        help=(
            "radius_calibration.json whose measured orbit_geometry supplies "
            "the pivot and axis; defaults to scan metadata or the fixed rig calibration"
        ),
    )
    parser.add_argument("--reference-angle-deg", type=float, default=0.0,
                        help="Angle of the reference frame (default: 0)")
    parser.add_argument("--angle-sign", type=float, default=1.0,
                        help="Sign convention for angle direction (1.0 or -1.0)")
    parser.add_argument("--crop-radius-m", type=float, default=None,
                        help="Final output crop limit around the pivot "
                             "(defaults to scan metadata or 0.08m; set <= 0 to disable)")
    parser.add_argument(
        "--registration-crop-radius-m",
        type=float,
        default=None,
        help=(
            "Half-extent of the tighter crop used only by ICP; defaults to "
            "scan metadata or min(final crop, 0.10m); set <= 0 to disable"
        ),
    )
    parser.add_argument(
        "--crop-shape",
        choices=("cube", "cylinder"),
        default="cylinder",
        help=(
            "Crop geometry around the pivot. 'cylinder' treats the crop radii "
            "as radial limits around the orbit axis and bounds the axis "
            "separately, which removes the enclosure ring without clipping the "
            "object along the axis (default: cube)"
        ),
    )
    parser.add_argument(
        "--crop-axial-half-length-m",
        type=float,
        default=None,
        help=(
            "Half-length along the orbit axis for --crop-shape cylinder "
            f"(default: {DEFAULT_CROP_AXIAL_HALF_LENGTH_M} m)"
        ),
    )
    parser.add_argument(
        "--exclude-high-drift-frames",
        action="store_true",
        help=(
            "Do not include an incoming frame in the final merge when its "
            "sequential ICP candidate exceeds the translation or rotation "
            "correction guard (guarded ICP modes only)"
        ),
    )

    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help="Keep detached point clusters in the merged cloud; by default a "
             "component smaller than 1 percent of the largest is discarded, "
             "since statistical outlier removal cannot see a compact blob of "
             "noise that floats clear of the object",
    )
    parser.add_argument("--skip-per-scan-sor", action="store_true",
                        help="Skip per-scan SOR (much faster; final SOR still runs)")
    parser.add_argument(
        "--fusion", choices=("points", "tsdf", "both"), default="both",
        help="How to combine the registered captures. 'points' concatenates the "
             "per-view clouds, which stacks every view's noise into one surface. "
             "'tsdf' averages them into a signed-distance field, cancelling "
             "independent per-view error and producing a mesh. 'both' writes each.",
    )
    parser.add_argument(
        "--tsdf-voxel-m", type=float, default=tsdf_fusion.DEFAULT_VOXEL_M,
        help="TSDF voxel edge length. The default matches the camera's 1 mm depth "
             "quantisation: finer adds no information, coarser trades real detail "
             "for smoothness",
    )
    parser.add_argument(
        "--tsdf-trunc-m", type=float, default=tsdf_fusion.DEFAULT_SDF_TRUNC_M,
        help="TSDF truncation distance; keep it a few voxels wide so the field "
             "can interpolate across a surface",
    )
    parser.add_argument(
        "--tsdf-depth", choices=("sensor", "output"), default="sensor",
        help="Which saved depth to fuse when a scan recorded rgbd/: the raw "
             "sensor depth, or the processed per-capture output",
    )
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args(argv)

    if args.tsdf_voxel_m <= 0 or args.tsdf_trunc_m <= 0:
        raise ValueError("--tsdf-voxel-m and --tsdf-trunc-m must be positive")
    if args.fusion != "points" and args.tsdf_trunc_m < args.tsdf_voxel_m:
        raise ValueError(
            "--tsdf-trunc-m must be at least --tsdf-voxel-m so the signed-distance "
            "field can interpolate across a surface"
        )

    guarded_icp_mode = args.registration_mode == "guarded-icp"
    if args.exclude_high_drift_frames and not guarded_icp_mode:
        raise ValueError(
            "--exclude-high-drift-frames requires --registration-mode "
            "guarded-icp"
        )

    input_dir = args.input_dir
    scan_metadata = load_scan_metadata(input_dir)
    captures = discover_captures(input_dir, scan_metadata)
    ply_files = [capture.path for capture in captures]
    logger.info("Found %d PLY files in %s", len(ply_files), input_dir)

    args.orbit_geometry = select_orbit_geometry(args.orbit_geometry, scan_metadata)
    calibration = load_orbit_geometry(args.orbit_geometry)
    intrinsics_path = input_dir / "intrinsics.json"
    camera_intrinsics = None
    if intrinsics_path.is_file():
        camera_intrinsics = json.loads(intrinsics_path.read_text())
        validate_camera_frame(calibration, camera_intrinsics)
    if args.fusion != "points" and camera_intrinsics is None:
        raise ValueError(
            f"TSDF fusion needs camera intrinsics, but {intrinsics_path} is missing. "
            "Depth is recovered by reprojecting each capture through its pinhole "
            "model, which cannot be done without them."
        )

    # TSDF integrates depth maps, so it needs the range the captures were
    # exported with rather than the merge path's geometric crop alone.
    capture_settings = scan_metadata.get("capture", {}) or {}
    capture_range = capture_settings.get("depth_range_m") or []
    if len(capture_range) == 2 and all(np.isfinite(capture_range)):
        fusion_depth_min_m, fusion_depth_max_m = (float(v) for v in capture_range)
    else:
        fusion_depth_min_m, fusion_depth_max_m = 0.0, 1.0
    measured_geometry = calibration["orbit_geometry"]
    orbit_axis = np.asarray(measured_geometry["axis"], dtype=float)
    calibration_x_mm = (
        None if args.pivot else calibration_station_x_mm(calibration, args.orbit_geometry)
    )
    orbit_radius_m = resolve_orbit_radius(
        args.orbit_radius_m,
        {"orbit_radius_m": calibration["recommended_radius_m"]},
    )
    pivot = np.asarray(args.pivot if args.pivot else measured_geometry["pivot_m"], dtype=float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        raise ValueError("--pivot must contain three finite coordinates")

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

    reconstruction_metadata = scan_metadata.get("reconstruction", {})
    if not isinstance(reconstruction_metadata, dict):
        reconstruction_metadata = {}
    crop_radius_m, registration_crop_radius_m = resolve_crop_radii(
        args.crop_radius_m,
        args.registration_crop_radius_m,
        reconstruction_metadata,
    )
    crop_axial_half_length_m = resolve_crop_axial_half_length(
        args.crop_axial_half_length_m,
        reconstruction_metadata,
    )

    merged_cloud_path = output_dir / "merged_cloud.ply"

    # Stage 1: Discover files and build pose priors
    logger.info("Stage 1/4: Discovering PLY files and building pose priors...")
    logger.info("Found %d point clouds.", len(ply_files))
    logger.info(
        "Orbit calibration: %s; radius %.4f m, axis %s, pivot %s",
        args.orbit_geometry, orbit_radius_m, orbit_axis, pivot,
    )

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
    # Shift the pivot from wherever it was measured to each frame's station.
    # With a calibration station the offset is absolute; without one, fall back
    # to the scan's own first station, which is all the metadata records.
    frame_crop_centers = [
        (
            pivot - orbit_axis * (
                (frame.x_position_mm - calibration_x_mm) / 1000.0
                if calibration_x_mm is not None
                else frame.x_offset_m
            )
        )
        for frame in frames
    ]
    crop_axis = orbit_axis if args.crop_shape == "cylinder" else None
    registration_crop_bounds = [
        crop_bounds_around(
            center,
            registration_crop_radius_m,
            axis=crop_axis,
            axial_half_length_m=crop_axial_half_length_m,
        )
        for center in frame_crop_centers
    ]
    final_crop_bounds = [
        crop_bounds_around(
            center,
            crop_radius_m,
            axis=crop_axis,
            axial_half_length_m=crop_axial_half_length_m,
        )
        for center in frame_crop_centers
    ]
    logger.info(
        "Crop shape: %s, half-extents: registration=%s, final=%s%s",
        args.crop_shape,
        (
            "disabled"
            if registration_crop_radius_m <= 0.0
            else f"{registration_crop_radius_m:.3f} m"
        ),
        "disabled" if crop_radius_m <= 0.0 else f"{crop_radius_m:.3f} m",
        (
            f", axial half-length={crop_axial_half_length_m:.3f} m"
            if crop_axis is not None
            else ""
        ),
    )

    if args.registration_mode == "motor":
        source = "deterministic motor"
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

    excluded_frame_ids = (
        high_drift_frame_ids(edges) if args.exclude_high_drift_frames else set()
    )
    merge_frame_ids = [
        frame_id for frame_id in range(len(frames))
        if frame_id not in excluded_frame_ids
    ]
    excluded_frames = []
    for edge in sorted(edges, key=lambda item: item.target_id):
        if edge.kind != "sequential" or edge.target_id not in excluded_frame_ids:
            continue
        frame = frames[edge.target_id]
        excluded_frames.append({
            "frame_id": edge.target_id,
            "filename": frame.path.name,
            "angle_deg": frame.angle_deg,
            "station_index": frame.station_index,
            "x_position_mm": frame.x_position_mm,
            "correction_m": edge.correction_m,
            "correction_deg": edge.correction_deg,
            "reason": edge.reason,
        })
        logger.warning(
            "Excluded %s from merge: sequential ICP requested %.3f mm / %.2f deg",
            frame.path.name,
            edge.correction_m * 1000.0,
            edge.correction_deg,
        )
    if args.exclude_high_drift_frames:
        logger.info(
            "High-drift frame filter retained %d/%d captures.",
            len(merge_frame_ids),
            len(frames),
        )

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
            "cli" if args.orbit_radius_m is not None else "orbit_geometry"
        ),
        "orbit_geometry_source": (
            str(args.orbit_geometry) if args.orbit_geometry else None
        ),
        "calibration_x_position_mm": calibration_x_mm,
        "orbit_axis": orbit_axis.tolist(),
        "pivot": pivot.tolist(),
        "reference_angle_deg": args.reference_angle_deg,
        "angle_sign": args.angle_sign,
        "crop_radius_m": crop_radius_m,
        "registration_crop_radius_m": registration_crop_radius_m,
        "crop_shape": args.crop_shape,
        "crop_axial_half_length_m": (
            crop_axial_half_length_m if args.crop_shape == "cylinder" else None
        ),
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
        "high_drift_frame_exclusion": {
            "enabled": args.exclude_high_drift_frames,
            "translation_limit_m": ICP_MAX_CORRECTION_M,
            "rotation_limit_deg": ICP_MAX_CORRECTION_DEG,
            "excluded_count": len(excluded_frames),
            "excluded_frames": excluded_frames,
        },
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
            "component_eps_m": FINAL_COMPONENT_EPS_M,
            "component_min_fraction": (
                0.0 if args.keep_all_components else FINAL_COMPONENT_MIN_FRACTION
            ),
        },
    }
    save_diagnostics(output_dir, frames, optimized_poses, edges, settings)

    tsdf_stats = None
    if args.fusion in ("tsdf", "both"):
        logger.info("Stage 3a: TSDF volumetric fusion...")
        tsdf_stats = tsdf_fusion.fuse_captures(
            [frames[frame_id].path for frame_id in merge_frame_ids],
            [optimized_poses[frame_id] for frame_id in merge_frame_ids],
            camera_intrinsics,
            input_dir=input_dir,
            output_dir=output_dir,
            voxel_length_m=args.tsdf_voxel_m,
            sdf_trunc_m=args.tsdf_trunc_m,
            depth_min_m=fusion_depth_min_m,
            depth_max_m=fusion_depth_max_m,
            depth_source=args.tsdf_depth,
            crop_bounds=final_crop_bounds[merge_frame_ids[0]],
        )
        settings["tsdf_fusion"] = tsdf_stats
        save_diagnostics(output_dir, frames, optimized_poses, edges, settings)

    if args.fusion == "tsdf":
        logger.info("Finished!")
        logger.info("TSDF mesh: %s", tsdf_stats["mesh_path"])
        logger.info("TSDF cloud: %s", tsdf_stats["cloud_path"])
        logger.info("Diagnostics: %s", output_dir / "registration_diagnostics.json")
        logger.info("Processing log: %s", processing_log_path)
        file_handler.flush()
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return

    logger.info("Stage 3/4: Open3D full-resolution transform and cleanup...")
    transformed_paths, transform_stats = transform_and_clean_clouds(
        [frames[frame_id].path for frame_id in merge_frame_ids],
        [optimized_poses[frame_id] for frame_id in merge_frame_ids],
        output_dir / "01_transformed",
        matrix_dir,
        crop_bounds=None,
        crop_bounds_by_cloud=[
            final_crop_bounds[frame_id] for frame_id in merge_frame_ids
        ],
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
            - orbit_axis * float(np.mean(sorted({frame.x_offset_m for frame in frames})))
        ),
        spatial_subsample_m=SPATIAL_SUBSAMPLE_M,
        duplicate_distance_m=REMOVE_DUPLICATES_DISTANCE_M,
        sor_neighbors=FINAL_SOR_NEIGHBORS,
        sor_sigma=FINAL_SOR_SIGMA,
        normal_radius_m=NORMAL_RADIUS_M,
        normal_max_neighbors=NORMAL_MAX_NEIGHBORS,
        normal_mst_neighbors=NORMAL_MST_NEIGHBORS,
        component_eps_m=FINAL_COMPONENT_EPS_M,
        component_min_points=FINAL_COMPONENT_MIN_POINTS,
        component_min_fraction=(
            0.0 if args.keep_all_components else FINAL_COMPONENT_MIN_FRACTION
        ),
    )
    settings["processing"] = {
        "transformed_frames": transform_stats,
        "merge": merge_stats,
    }
    save_diagnostics(output_dir, frames, optimized_poses, edges, settings)

    settings["processing"]["tsdf_fusion"] = tsdf_stats
    save_diagnostics(output_dir, frames, optimized_poses, edges, settings)

    logger.info("Finished!")
    logger.info("Merged cloud: %s", merged_cloud_path)
    if tsdf_stats is not None:
        logger.info("TSDF mesh: %s", tsdf_stats["mesh_path"])
        logger.info("TSDF cloud: %s", tsdf_stats["cloud_path"])
    logger.info("Diagnostics: %s", output_dir / "registration_diagnostics.json")
    logger.info("Processing log: %s", processing_log_path)
    file_handler.flush()
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    main()
