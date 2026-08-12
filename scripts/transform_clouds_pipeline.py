"""Internal implementation for the point-cloud reconstruction pipeline.

Use ``transform_clouds.py`` as the public command-line entry point.  Keeping
the numerical implementation here lets the entry point show the workflow
without mixing it with geometry, registration, CloudCompare, and meshing
details.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import subprocess

import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_DIR = Path(
    "/media/rishmika/Shared Data/Idea8/Hena Jet/captures/Plastic_Hand"
)

OUTPUT_DIR = INPUT_DIR / "reconstruction_new_without_platform_alignment"
TRANSFORMED_DIR = OUTPUT_DIR / "01_transformed"
MATRIX_DIR = OUTPUT_DIR / "matrices"

MERGED_CLOUD_PATH = OUTPUT_DIR / "mascot_merged_cleaned.ply"
MESH_PATH = OUTPUT_DIR / "mascot_mesh_poisson.ply"
LOG_PATH = OUTPUT_DIR / "cloudcompare_pipeline.log"
OPTIMIZED_POSES_PATH = OUTPUT_DIR / "optimized_poses.npy"
DIAGNOSTICS_PATH = OUTPUT_DIR / "registration_diagnostics.json"

# First/reference scan: angle_005p_00.ply
REFERENCE_ANGLE_DEG = 5.0

# ZED IMAGE coordinates: X right, Y down, Z forward.
# The physical orbit axis is vertical and is therefore parallel to camera Y.
ORBIT_RADIUS_M = 0.175

# Pivot location in the reference camera frame.
PIVOT_IN_REFERENCE = np.array(
    [0.025, 0.0, ORBIT_RADIUS_M],
    dtype=float,
)

# The calibrated motor orbit is fixed. Registration does not fit or align a
# stage/platform because hand captures may not contain one.
ORBIT_AXIS_IN_REFERENCE = np.array(
    [0.03097, 0.99918, -0.02609],
    dtype=float,
)

# Reverse this if the known-angle transformations move the scans apart.
ANGLE_SIGN = 1.0

# Crop after the known transform and before ICP, expressed in the
# reference camera frame as (Xmin, Ymin, Zmin, Xmax, Ymax, Zmax), in metres.
# Measure these bounds around only the mascot. Keeping the table/background can
# make ICP align those surfaces instead of the mascot.
CROP_BOUNDS_M: tuple[float, float, float, float, float, float] = (
    -0.060, -0.100, 0.095,
     0.110,  0.100, 0.255,
)

# Per-scan cleanup before ICP.
PRE_ICP_SOR_NEIGHBORS = 20
PRE_ICP_SOR_SIGMA = 2.0

# Markerless pose-graph registration settings.
REGISTRATION_VOXEL_M = 0.002
REGISTRATION_NORMAL_RADIUS_M = 0.006
ICP_COARSE_DISTANCE_M = 0.008
ICP_FINE_DISTANCE_M = 0.003
ICP_ITERATIONS = 60
ICP_MIN_FITNESS = 0.30
ICP_MAX_RMSE_M = 0.004
ICP_MAX_CORRECTION_M = 0.010
ICP_MAX_CORRECTION_DEG = 4.0
ICP_MIN_POINTS = 100
REGISTRATION_NEIGHBOR_SPAN = 1
ORBIT_PRIOR_WEIGHT = 50.0
POSE_GRAPH_EDGE_PRUNE_THRESHOLD = 0.25

# Quality gates. Failed runs still write diagnostics and matrices, but do not
# emit a misleading final reconstruction unless explicitly overridden.
MINIMUM_SEQUENTIAL_USABILITY = 0.90
MAXIMUM_LOOP_CORRECTION_M = 0.003
MAXIMUM_LOOP_CORRECTION_DEG = 1.0
ALLOW_LOW_QUALITY_OUTPUT = False

# Final merged-cloud cleanup. All distances are metres.
FINAL_SOR_NEIGHBORS = 30
FINAL_SOR_SIGMA = 1.5
REMOVE_DUPLICATES_DISTANCE_M = 0.00010
SPATIAL_SUBSAMPLE_M = 0.00075
NORMAL_RADIUS_M = 0.0040
NORMAL_MST_NEIGHBORS = 12

# Set to False if you only want the registered point cloud.
CREATE_POISSON_MESH = True
POISSON_DEPTH = 8
POISSON_DENSITY_TRIM_QUANTILE = 0.04
POISSON_SCALE = 1.1

FLATPAK_APP_ID = "org.cloudcompare.CloudCompare"
CLOUDCOMPARE_COMMAND = [
    "/usr/bin/flatpak",
    "run",
    FLATPAK_APP_ID,
]


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Dataset-specific settings selected at program startup."""

    input_dir: Path
    output_dir: Path
    reference_angle_deg: float
    orbit_radius_m: float
    pivot_in_reference: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse dataset and orbit parameters without changing module settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Register angle_*.ply captures and reconstruct a merged cloud "
            "and optional mesh."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=(
            "Directory containing root-level angle_*.ply files "
            f"(default: {INPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Reconstruction output directory (default: "
            "<input-dir>/reconstruction_new_without_platform_alignment)"
        ),
    )
    parser.add_argument(
        "--reference-angle-deg",
        type=float,
        default=REFERENCE_ANGLE_DEG,
        help=(
            "Capture angle used as the reference camera frame "
            f"(default: {REFERENCE_ANGLE_DEG:g})"
        ),
    )
    parser.add_argument(
        "--orbit-radius-m",
        type=float,
        default=ORBIT_RADIUS_M,
        help=f"Camera orbit radius in metres (default: {ORBIT_RADIUS_M:g})",
    )
    return parser.parse_args(argv)


def runtime_configuration(args: argparse.Namespace) -> RuntimeConfiguration:
    """Resolve CLI arguments into one internally consistent configuration."""
    if args.orbit_radius_m <= 0.0:
        raise ValueError("--orbit-radius-m must be greater than zero.")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            args.input_dir
            / "reconstruction_new_without_platform_alignment"
        )
    pivot = np.asarray(PIVOT_IN_REFERENCE, dtype=float).copy()
    pivot[2] = args.orbit_radius_m
    return RuntimeConfiguration(
        input_dir=args.input_dir,
        output_dir=output_dir,
        reference_angle_deg=args.reference_angle_deg,
        orbit_radius_m=args.orbit_radius_m,
        pivot_in_reference=pivot,
    )


def apply_runtime_configuration(configuration: RuntimeConfiguration) -> None:
    """Apply dataset paths and orbit values before the pipeline starts."""
    global INPUT_DIR, OUTPUT_DIR, TRANSFORMED_DIR, MATRIX_DIR
    global MERGED_CLOUD_PATH, MESH_PATH, LOG_PATH
    global OPTIMIZED_POSES_PATH, DIAGNOSTICS_PATH
    global REFERENCE_ANGLE_DEG, ORBIT_RADIUS_M, PIVOT_IN_REFERENCE

    INPUT_DIR = configuration.input_dir
    OUTPUT_DIR = configuration.output_dir
    TRANSFORMED_DIR = OUTPUT_DIR / "01_transformed"
    MATRIX_DIR = OUTPUT_DIR / "matrices"
    MERGED_CLOUD_PATH = OUTPUT_DIR / "mascot_merged_cleaned.ply"
    MESH_PATH = OUTPUT_DIR / "mascot_mesh_poisson.ply"
    LOG_PATH = OUTPUT_DIR / "cloudcompare_pipeline.log"
    OPTIMIZED_POSES_PATH = OUTPUT_DIR / "optimized_poses.npy"
    DIAGNOSTICS_PATH = OUTPUT_DIR / "registration_diagnostics.json"
    REFERENCE_ANGLE_DEG = configuration.reference_angle_deg
    ORBIT_RADIUS_M = configuration.orbit_radius_m
    PIVOT_IN_REFERENCE = configuration.pivot_in_reference.copy()


@dataclass
class RegistrationEdge:
    """One guarded pairwise registration constraint."""

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
    cloud: object


def normalized_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"{name} must contain a finite non-zero XYZ vector.")
    return vector / length


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def rodrigues_rotation(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Return a proper 3x3 rotation around an arbitrary unit axis."""
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
    """Build a rigid transform rotating around a 3D line through ``pivot``."""
    pivot = np.asarray(pivot, dtype=float)
    if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
        raise ValueError("Rotation pivot must contain finite X, Y, Z values.")
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


def registration_result_is_acceptable(
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


def registration_pairs(
    *,
    frame_count: int,
    neighbor_span: int,
    include_loop_closure: bool,
) -> list[tuple[int, int, str]]:
    """Return deterministic sequential, wider-neighbor, and loop pairs."""
    if frame_count < 1:
        raise ValueError("At least one frame is required.")
    if neighbor_span < 1:
        raise ValueError("Neighbor span must be at least one.")
    pairs = [
        (source_id, source_id + 1, "sequential")
        for source_id in range(frame_count - 1)
    ]
    for gap in range(2, neighbor_span + 1):
        pairs.extend(
            (source_id, source_id + gap, "neighbor")
            for source_id in range(frame_count - gap)
        )
    if include_loop_closure and frame_count > 2:
        pairs.append((0, frame_count - 1, "loop"))
    return pairs


def angle_between_vectors_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = normalized_vector(first, name="First vector")
    second = normalized_vector(second, name="Second vector")
    cosine = float(np.clip(first @ second, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def evaluate_registration_quality(
    *,
    optimized_poses: list[np.ndarray],
    initial_poses: list[np.ndarray],
    edges: list[RegistrationEdge],
    minimum_sequential_usable_fraction: float,
    maximum_loop_correction_m: float,
    maximum_loop_correction_deg: float,
) -> dict:
    """Evaluate whether optimized poses are authoritative enough to merge."""
    if len(initial_poses) != len(optimized_poses):
        raise ValueError("Every optimized pose must have one pose prior.")

    sequential_edges = [
        edge for edge in edges if edge.kind == "sequential"
    ]
    sequential_icp_acceptance = (
        sum(edge.accepted for edge in sequential_edges)
        / len(sequential_edges)
        if sequential_edges
        else 0.0
    )
    sequential_usable = (
        sum(edge.accepted or edge.usable for edge in sequential_edges)
        / len(sequential_edges)
        if sequential_edges
        else 0.0
    )
    loop_edges = [edge for edge in edges if edge.kind == "loop"]
    loop_edge = loop_edges[-1] if loop_edges else None
    loop_translation_m = float("inf")
    loop_rotation_deg = float("inf")
    if loop_edge is not None:
        expected_loop = relative_camera_transform(
            initial_poses[loop_edge.source_id],
            initial_poses[loop_edge.target_id],
        )
        optimized_loop = relative_camera_transform(
            optimized_poses[loop_edge.source_id],
            optimized_poses[loop_edge.target_id],
        )
        loop_translation_m, loop_rotation_deg = transform_delta(
            expected_loop,
            optimized_loop,
        )
    loop_passed = bool(
        loop_edge is not None
        and (loop_edge.accepted or loop_edge.usable)
        and loop_translation_m <= maximum_loop_correction_m
        and loop_rotation_deg <= maximum_loop_correction_deg
    )

    failure_reasons = []
    if sequential_usable < minimum_sequential_usable_fraction:
        failure_reasons.append(
            "usable sequential registration is below the quality gate"
        )
    if not loop_passed:
        failure_reasons.append("loop closure did not pass the quality gate")

    return {
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "sequential_usable_fraction": sequential_usable,
        "sequential_icp_acceptance_fraction": sequential_icp_acceptance,
        "loop_closure_passed": loop_passed,
        "loop_closure_translation_m": (
            loop_translation_m if np.isfinite(loop_translation_m) else None
        ),
        "loop_closure_rotation_deg": (
            loop_rotation_deg if np.isfinite(loop_rotation_deg) else None
        ),
    }


def run_command(command: list[str]) -> None:
    """Run a command and raise a readable error if it fails."""
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def read_angle(filename: str) -> float:
    """
    Read angles from names such as:
        angle_005p_00.ply -> 5.00
        angle_0345p00.ply -> 345.00
    """
    match = re.search(r"angle_(\d+)p_?(\d+)", filename)
    if match is None:
        raise ValueError(f"Cannot extract an angle from: {filename}")

    return float(f"{match.group(1)}.{match.group(2)}")


def find_ply_files(input_dir: Path) -> list[Path]:
    """Find input captures and order them by their recorded angle."""
    files = list(input_dir.glob("angle_*p*.ply"))
    return sorted(files, key=lambda path: read_angle(path.name))


def normalize_angle_degrees(angle_degrees: float) -> float:
    """Represent an angle in the interval [-180, 180)."""
    return (angle_degrees + 180.0) % 360.0 - 180.0


def cloudcompare_file_argument(path: Path) -> str:
    """
    CloudCompare parses the value following SAVE_CLOUDS FILE itself.
    Literal quotes preserve output paths containing spaces.
    """
    return f'"{path}"'


def cloudcompare_crop_argument(
    bounds: tuple[float, float, float, float, float, float],
) -> str:
    """Convert XYZ minimum/maximum bounds to CloudCompare's crop syntax."""
    return ":".join(f"{value:.10g}" for value in bounds)


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for markerless plane calibration, guarded "
            "ICP, pose-graph optimization, and meshing. Install it with:\n"
            "python3 -m pip install open3d"
        ) from error
    return o3d


def points_inside_bounds(
    points: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )


def build_orbit_poses(
    ply_files: list[Path],
    orbit_axis: np.ndarray,
) -> list[np.ndarray]:
    """Build motor-angle pose priors without requiring a visible stage."""
    poses = []
    for path in ply_files:
        relative_angle = normalize_angle_degrees(
            ANGLE_SIGN * (read_angle(path.name) - REFERENCE_ANGLE_DEG)
        )
        poses.append(
            rotation_about_axis(
                relative_angle,
                PIVOT_IN_REFERENCE,
                orbit_axis,
            )
        )
    return poses


def transformed_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def prepare_registration_cloud(
    o3d,
    cloud,
    initial_pose: np.ndarray,
):
    """Build a cropped, downsampled object cloud for registration."""
    points = np.asarray(cloud.points, dtype=float)
    if points.size == 0:
        return o3d.geometry.PointCloud()
    finite = np.all(np.isfinite(points), axis=1)
    points_in_reference = transformed_points(
        points,
        initial_pose,
    )
    within_crop = points_inside_bounds(points_in_reference, CROP_BOUNDS_M)
    keep = finite & within_crop
    selected_indices = np.flatnonzero(keep)
    selected = cloud.select_by_index(selected_indices.tolist())
    selected = selected.voxel_down_sample(REGISTRATION_VOXEL_M)
    if len(selected.points) >= 3:
        selected.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=REGISTRATION_NORMAL_RADIUS_M,
                max_nn=50,
            )
        )
    return selected


def load_registration_frames(
    o3d,
    ply_files: list[Path],
    priors: list[np.ndarray],
) -> list[RegistrationFrame]:
    frames = []
    for index, (path, prior) in enumerate(
        zip(ply_files, priors),
        start=1,
    ):
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise RuntimeError(f"Open3D could not read points from {path}")
        registration_cloud = prepare_registration_cloud(
            o3d,
            cloud,
            prior,
        )
        frames.append(
            RegistrationFrame(
                path=path,
                angle_deg=read_angle(path.name),
                prior_pose=prior,
                cloud=registration_cloud,
            )
        )
        print(
            f"Registration {index}/{len(ply_files)} {path.name}: "
            f"{len(registration_cloud.points)} object points"
        )
    return frames


def relative_camera_transform(
    source_camera_to_reference: np.ndarray,
    target_camera_to_reference: np.ndarray,
) -> np.ndarray:
    """Map source-camera points into target-camera coordinates."""
    return (
        np.linalg.inv(target_camera_to_reference)
        @ source_camera_to_reference
    )


def information_matrix(
    o3d,
    source,
    target,
    transform: np.ndarray,
) -> np.ndarray:
    if len(source.points) < 3 or len(target.points) < 3:
        return np.eye(6, dtype=float)
    information = (
        o3d.pipelines.registration
        .get_information_matrix_from_point_clouds(
            source,
            target,
            ICP_FINE_DISTANCE_M,
            transform,
        )
    )
    information = np.asarray(information, dtype=float)
    if (
        information.shape != (6, 6)
        or not np.all(np.isfinite(information))
        or np.linalg.norm(information) == 0.0
    ):
        return np.eye(6, dtype=float)
    return information


def register_pair(
    o3d,
    source_id: int,
    target_id: int,
    kind: str,
    frames: list[RegistrationFrame],
) -> RegistrationEdge:
    source = frames[source_id].cloud
    target = frames[target_id].cloud
    prior = relative_camera_transform(
        frames[source_id].prior_pose,
        frames[target_id].prior_pose,
    )
    if (
        len(source.points) < ICP_MIN_POINTS
        or len(target.points) < ICP_MIN_POINTS
    ):
        return RegistrationEdge(
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            transform=prior,
            information=information_matrix(
                o3d,
                source,
                target,
                prior,
            ),
            accepted=False,
            reason="registration cloud too small; used pose prior",
            fitness=0.0,
            rmse_m=float("inf"),
            correction_m=0.0,
            correction_deg=0.0,
            usable=False,
        )

    prior_evaluation = (
        o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            ICP_FINE_DISTANCE_M,
            prior,
        )
    )
    prior_fitness = float(prior_evaluation.fitness)
    prior_rmse_m = float(prior_evaluation.inlier_rmse)
    prior_usable = (
        np.isfinite(prior_fitness)
        and np.isfinite(prior_rmse_m)
        and prior_fitness >= ICP_MIN_FITNESS
        and prior_rmse_m <= ICP_MAX_RMSE_M
    )

    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=ICP_ITERATIONS,
    )
    coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_COARSE_DISTANCE_M,
        prior,
        estimation,
        criteria,
    )
    fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_FINE_DISTANCE_M,
        coarse.transformation,
        estimation,
        criteria,
    )
    candidate = np.asarray(fine.transformation, dtype=float)
    accepted, reason, correction_m, correction_deg = (
        registration_result_is_acceptable(
            fitness=float(fine.fitness),
            rmse_m=float(fine.inlier_rmse),
            prior=prior,
            candidate=candidate,
            min_fitness=ICP_MIN_FITNESS,
            max_rmse_m=ICP_MAX_RMSE_M,
            max_correction_m=ICP_MAX_CORRECTION_M,
            max_correction_deg=ICP_MAX_CORRECTION_DEG,
        )
    )
    transform = candidate if accepted else prior
    if not accepted:
        reason = f"{reason}; used pose prior"
    return RegistrationEdge(
        source_id=source_id,
        target_id=target_id,
        kind=kind,
        transform=transform,
        information=information_matrix(
            o3d,
            source,
            target,
            transform,
        ),
        accepted=accepted,
        reason=reason,
        fitness=float(fine.fitness),
        rmse_m=float(fine.inlier_rmse),
        correction_m=correction_m,
        correction_deg=correction_deg,
        usable=accepted or prior_usable,
        prior_fitness=prior_fitness,
        prior_rmse_m=prior_rmse_m,
    )


def print_registration_edge(edge: RegistrationEdge) -> None:
    outcome = "accepted" if edge.accepted else "fallback"
    print(
        f"{edge.kind.title()} {edge.source_id}->{edge.target_id} "
        f"{outcome}: fitness={edge.fitness:.3f}, "
        f"rmse={edge.rmse_m:.4f} m, "
        f"correction={edge.correction_m:.4f} m/"
        f"{edge.correction_deg:.2f} deg, "
        f"usable={'yes' if edge.usable else 'no'} ({edge.reason})"
    )


def build_pose_graph(
    o3d,
    frames: list[RegistrationFrame],
) -> tuple[object, list[RegistrationEdge]]:
    graph = o3d.pipelines.registration.PoseGraph()
    for frame in frames:
        graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                frame.prior_pose.copy()
            )
        )

    edges = []
    for source_id, target_id, kind in registration_pairs(
        frame_count=len(frames),
        neighbor_span=REGISTRATION_NEIGHBOR_SPAN,
        include_loop_closure=True,
    ):
        edge = register_pair(
            o3d,
            source_id,
            target_id,
            kind,
            frames,
        )
        edges.append(edge)
        print_registration_edge(edge)

        if kind == "sequential":
            prior = relative_camera_transform(
                frames[source_id].prior_pose,
                frames[target_id].prior_pose,
            )
            prior_information = information_matrix(
                o3d,
                frames[source_id].cloud,
                frames[target_id].cloud,
                prior,
            ) * ORBIT_PRIOR_WEIGHT
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
    return graph, edges


def optimize_pose_graph(o3d, graph) -> list[np.ndarray]:
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
    return [
        np.asarray(node.pose, dtype=float).copy()
        for node in graph.nodes
    ]


def save_pose_matrices(
    frames: list[RegistrationFrame],
    optimized_poses: list[np.ndarray],
) -> None:
    np.save(
        OPTIMIZED_POSES_PATH,
        np.stack(optimized_poses),
    )
    for frame, optimized in zip(frames, optimized_poses):
        stem = frame.path.stem
        matrices = {
            f"{stem}_prior_matrix.txt": frame.prior_pose,
            f"{stem}_optimized_matrix.txt": optimized,
        }
        for filename, matrix in matrices.items():
            np.savetxt(MATRIX_DIR / filename, matrix, fmt="%.10f")


def edge_as_json(edge: RegistrationEdge) -> dict:
    return {
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
        "prior_fitness": edge.prior_fitness,
        "prior_rmse_m": (
            edge.prior_rmse_m
            if np.isfinite(edge.prior_rmse_m)
            else None
        ),
        "transform": edge.transform.tolist(),
    }


def save_registration_diagnostics(
    *,
    frames: list[RegistrationFrame],
    orbit_axis: np.ndarray,
    optimized_poses: list[np.ndarray],
    edges: list[RegistrationEdge],
    quality: dict,
) -> None:
    corrections = []
    for frame, optimized in zip(frames, optimized_poses):
        correction_m, correction_deg = transform_delta(
            frame.prior_pose,
            optimized,
        )
        corrections.append(
            {
                "filename": frame.path.name,
                "angle_deg": frame.angle_deg,
                "translation_m": correction_m,
                "rotation_deg": correction_deg,
            }
        )
    diagnostics = {
        "schema_version": 2,
        "input_dir": str(INPUT_DIR),
        "capture_count": len(frames),
        "orbit_axis_in_reference": orbit_axis.tolist(),
        "orbit_axis_tilt_from_camera_y_deg": (
            angle_between_vectors_degrees(
                orbit_axis,
                np.array([0.0, 1.0, 0.0]),
            )
        ),
        "settings": {
            "alignment_mode": "motor_orbit_and_object_icp",
            "pivot_in_reference": PIVOT_IN_REFERENCE.tolist(),
            "registration_voxel_m": REGISTRATION_VOXEL_M,
            "icp_coarse_distance_m": ICP_COARSE_DISTANCE_M,
            "icp_fine_distance_m": ICP_FINE_DISTANCE_M,
            "neighbor_span": REGISTRATION_NEIGHBOR_SPAN,
            "orbit_prior_weight": ORBIT_PRIOR_WEIGHT,
            "allow_low_quality_output": ALLOW_LOW_QUALITY_OUTPUT,
        },
        "edges": [edge_as_json(edge) for edge in edges],
        "optimized_pose_corrections": corrections,
        "quality": quality,
    }
    DIAGNOSTICS_PATH.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def optimized_transform_and_clean(
    frames: list[RegistrationFrame],
    optimized_poses: list[np.ndarray],
) -> list[Path]:
    """Apply globally optimized poses and per-scan cleanup at full resolution."""
    if len(frames) != len(optimized_poses):
        raise ValueError("Every frame must have one optimized pose.")
    transformed_paths: list[Path] = []

    print("\nStage 3/5: optimized transforms and per-scan cleanup")

    for frame, optimized_pose in zip(frames, optimized_poses):
        ply_path = frame.path
        matrix_path = (
            MATRIX_DIR / f"{ply_path.stem}_optimized_matrix.txt"
        )
        output_path = (
            TRANSFORMED_DIR / f"{ply_path.stem}_transformed.ply"
        )

        np.savetxt(matrix_path, optimized_pose, fmt="%.10f")

        command = [
            *CLOUDCOMPARE_COMMAND,
            "-VERBOSITY",
            "2",
            "-SILENT",
            "-AUTO_SAVE",
            "OFF",
            "-O",
            str(ply_path),
            "-APPLY_TRANS",
            str(matrix_path),
        ]

        command.extend(
            [
                "-CROP",
                cloudcompare_crop_argument(CROP_BOUNDS_M),
            ]
        )

        command.extend(
            [
                "-SOR",
                str(PRE_ICP_SOR_NEIGHBORS),
                str(PRE_ICP_SOR_SIGMA),
                "-C_EXPORT_FMT",
                "PLY",
                "-SAVE_CLOUDS",
                "FILE",
                cloudcompare_file_argument(output_path),
            ]
        )

        print(
            f"\n{ply_path.name}: optimized angle={frame.angle_deg:.2f} deg"
        )
        run_command(command)
        transformed_paths.append(output_path)

    return transformed_paths


def reference_first(
    paths: list[Path],
) -> list[Path]:
    """Order scans forward around the orbit, starting at the reference."""
    reference_candidates = [
        path
        for path in paths
        if math.isclose(
            read_angle(path.name),
            REFERENCE_ANGLE_DEG,
            abs_tol=1.0e-6,
        )
    ]

    if not reference_candidates:
        raise RuntimeError(
            "The configured reference scan was not found. "
            f"REFERENCE_ANGLE_DEG={REFERENCE_ANGLE_DEG}"
        )

    return sorted(
        paths,
        key=lambda path: (
            read_angle(path.name) - REFERENCE_ANGLE_DEG
        ) % 360.0,
    )


def merge_optimized_clouds(
    transformed_paths: list[Path],
) -> None:
    """Merge optimized full-resolution scans without further pose drift."""
    ordered_paths = reference_first(transformed_paths)

    print("\nStage 4/5: optimized merge and final cloud cleanup")

    command = [
        *CLOUDCOMPARE_COMMAND,
        "-VERBOSITY",
        "2",
        "-SILENT",
        "-AUTO_SAVE",
        "OFF",
        "-LOG_FILE",
        str(LOG_PATH),
        "-C_EXPORT_FMT",
        "PLY",
        "-O",
        str(ordered_paths[0]),
    ]

    for incoming_path in ordered_paths[1:]:
        command.extend(
            [
                "-O",
                str(incoming_path),
                "-MERGE_CLOUDS",
            ]
        )

    # Clean, reduce duplicate density, calculate consistently oriented normals,
    # and save one final registered cloud.
    command.extend(
        [
            "-SOR",
            str(FINAL_SOR_NEIGHBORS),
            str(FINAL_SOR_SIGMA),
            "-RDP",
            str(REMOVE_DUPLICATES_DISTANCE_M),
            "-SS",
            "SPATIAL",
            str(SPATIAL_SUBSAMPLE_M),
            "-OCTREE_NORMALS",
            str(NORMAL_RADIUS_M),
            "-ORIENT",
            "PLUS_BARYCENTER",
            "-ORIENT_NORMS_MST",
            str(NORMAL_MST_NEIGHBORS),
            "-SAVE_CLOUDS",
            "FILE",
            cloudcompare_file_argument(MERGED_CLOUD_PATH),
        ]
    )

    run_command(command)


def create_poisson_mesh() -> None:
    """Create and crop a Poisson mesh using Open3D."""
    print("\nStage 5/5: Poisson surface reconstruction")

    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required only for the mesh stage. Install it with:\n"
            "python3 -m pip install open3d\n"
            "Then run this script again, or set CREATE_POISSON_MESH=False "
            "to generate only the registered point cloud."
        ) from error

    cloud = o3d.io.read_point_cloud(str(MERGED_CLOUD_PATH))
    if cloud.is_empty():
        raise RuntimeError(
            f"Open3D could not read points from {MERGED_CLOUD_PATH}"
        )

    # Re-estimate normals here so the meshing stage does not depend on whether
    # the installed PLY importer retains CloudCompare normal properties.
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS_M,
            max_nn=50,
        )
    )
    cloud.orient_normals_consistent_tangent_plane(
        NORMAL_MST_NEIGHBORS
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=POISSON_DEPTH,
            scale=POISSON_SCALE,
            linear_fit=False,
        )
    )

    densities_array = np.asarray(densities)
    if (
        densities_array.size > 0
        and POISSON_DENSITY_TRIM_QUANTILE > 0.0
    ):
        threshold = np.quantile(
            densities_array,
            POISSON_DENSITY_TRIM_QUANTILE,
        )
        mesh.remove_vertices_by_mask(densities_array < threshold)

    # Remove Poisson geometry extending beyond the measured cloud bounds.
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(
        str(MESH_PATH),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    ):
        raise RuntimeError(f"Failed to save mesh to {MESH_PATH}")


def print_summary() -> None:
    print("\nFinished")
    print(f"Registered cloud: {MERGED_CLOUD_PATH}")
    print(f"CloudCompare log: {LOG_PATH}")
    print(f"Optimized poses: {OPTIMIZED_POSES_PATH}")
    print(f"Registration diagnostics: {DIAGNOSTICS_PATH}")
    if CREATE_POISSON_MESH:
        print(f"Poisson mesh: {MESH_PATH}")


def validate_settings() -> None:
    if ORBIT_RADIUS_M <= 0.0:
        raise ValueError("ORBIT_RADIUS_M must be greater than zero.")
    if PIVOT_IN_REFERENCE.shape != (3,):
        raise ValueError("PIVOT_IN_REFERENCE must contain X, Y, Z.")
    normalized_vector(
        ORBIT_AXIS_IN_REFERENCE,
        name="ORBIT_AXIS_IN_REFERENCE",
    )
    if len(CROP_BOUNDS_M) != 6:
        raise ValueError("CROP_BOUNDS_M must contain six values.")
    x_min, y_min, z_min, x_max, y_max, z_max = CROP_BOUNDS_M
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError(
            "Every CROP_BOUNDS_M minimum must be below its maximum."
        )
    positive_values = {
        "REGISTRATION_VOXEL_M": REGISTRATION_VOXEL_M,
        "REGISTRATION_NORMAL_RADIUS_M": REGISTRATION_NORMAL_RADIUS_M,
        "ICP_COARSE_DISTANCE_M": ICP_COARSE_DISTANCE_M,
        "ICP_FINE_DISTANCE_M": ICP_FINE_DISTANCE_M,
        "ICP_MAX_RMSE_M": ICP_MAX_RMSE_M,
        "ICP_MAX_CORRECTION_M": ICP_MAX_CORRECTION_M,
        "ICP_MAX_CORRECTION_DEG": ICP_MAX_CORRECTION_DEG,
        "ORBIT_PRIOR_WEIGHT": ORBIT_PRIOR_WEIGHT,
    }
    for name, value in positive_values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero.")
    if ICP_COARSE_DISTANCE_M < ICP_FINE_DISTANCE_M:
        raise ValueError(
            "ICP_COARSE_DISTANCE_M must be at least ICP_FINE_DISTANCE_M."
        )
    unit_interval_values = {
        "ICP_MIN_FITNESS": ICP_MIN_FITNESS,
        "MINIMUM_SEQUENTIAL_USABILITY": MINIMUM_SEQUENTIAL_USABILITY,
    }
    for name, value in unit_interval_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if REGISTRATION_NEIGHBOR_SPAN < 1:
        raise ValueError("REGISTRATION_NEIGHBOR_SPAN must be at least one.")
    if not 0.0 <= POISSON_DENSITY_TRIM_QUANTILE < 1.0:
        raise ValueError(
            "POISSON_DENSITY_TRIM_QUANTILE must be in [0, 1)."
        )


def main(argv: list[str] | None = None) -> None:
    configuration = runtime_configuration(parse_args(argv))
    apply_runtime_configuration(configuration)
    validate_settings()
    o3d = import_open3d()
    ply_files = reference_first(find_ply_files(INPUT_DIR))

    if not ply_files:
        raise RuntimeError(f"No matching PLY files found in:\n{INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Using CloudCompare command: "
        f"{' '.join(CLOUDCOMPARE_COMMAND)}"
    )
    print(f"Found {len(ply_files)} point clouds.")
    print(f"Orbit radius: {ORBIT_RADIUS_M:.4f} m")
    print(f"Pivot in reference frame: {PIVOT_IN_REFERENCE}")

    print("\nStage 1/5: motor orbit initialization")
    print(
        "Using the fixed calibrated orbit axis and motor angles. "
        "No stage/platform alignment is performed."
    )
    orbit_axis = normalized_vector(
        ORBIT_AXIS_IN_REFERENCE,
        name="ORBIT_AXIS_IN_REFERENCE",
    )
    print(f"Orbit axis: {orbit_axis}")
    priors = build_orbit_poses(ply_files, orbit_axis)

    print("\nStage 2/5: guarded ICP and global pose graph")
    frames = load_registration_frames(
        o3d,
        ply_files,
        priors,
    )
    graph, edges = build_pose_graph(o3d, frames)
    optimized_poses = optimize_pose_graph(o3d, graph)
    save_pose_matrices(frames, optimized_poses)

    quality = evaluate_registration_quality(
        optimized_poses=optimized_poses,
        initial_poses=priors,
        edges=edges,
        minimum_sequential_usable_fraction=(
            MINIMUM_SEQUENTIAL_USABILITY
        ),
        maximum_loop_correction_m=MAXIMUM_LOOP_CORRECTION_M,
        maximum_loop_correction_deg=MAXIMUM_LOOP_CORRECTION_DEG,
    )
    save_registration_diagnostics(
        frames=frames,
        orbit_axis=orbit_axis,
        optimized_poses=optimized_poses,
        edges=edges,
        quality=quality,
    )
    quality_parts = [
        f"passed={quality['passed']}",
        (
            "sequential_usable="
            f"{quality['sequential_usable_fraction']:.1%}"
        ),
        (
            "sequential_ICP="
            f"{quality['sequential_icp_acceptance_fraction']:.1%}"
        ),
        f"loop={quality['loop_closure_passed']}",
    ]
    print("Quality: " + ", ".join(quality_parts))
    if not quality["passed"] and not ALLOW_LOW_QUALITY_OUTPUT:
        reasons = "\n- ".join(quality["failure_reasons"])
        raise RuntimeError(
            "Markerless registration did not pass the quality gates. "
            "Diagnostics and matrices were saved, but final reconstruction "
            f"was stopped:\n- {reasons}\n"
            "Inspect registration_diagnostics.json or set "
            "ALLOW_LOW_QUALITY_OUTPUT=True to override."
        )

    transformed_paths = optimized_transform_and_clean(
        frames,
        optimized_poses,
    )
    merge_optimized_clouds(transformed_paths)

    if CREATE_POISSON_MESH:
        create_poisson_mesh()

    print_summary()


if __name__ == "__main__":
    main()
