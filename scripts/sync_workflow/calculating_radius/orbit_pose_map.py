"""Full aligned-pointcloud pose helpers for a fixed ArUco profile calibration."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from calculating_radius.radius_calibration import fit_orbit_circle, transform_inverse


def _validate_transform(transform: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must have a rigid-transform final row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be proper")
    return matrix


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _rotation_medoid(rotations: np.ndarray) -> np.ndarray:
    costs = [
        np.median([rotation_distance_deg(candidate, other) for other in rotations])
        for candidate in rotations
    ]
    return rotations[int(np.argmin(costs))]


def _mad_inliers(residuals: np.ndarray, threshold: float, floor: float) -> np.ndarray:
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_sigma = 1.4826 * mad
    return residuals <= median + threshold * max(robust_sigma, floor)


def aggregate_world_to_camera_poses(
    world_to_camera_poses: Iterable[np.ndarray],
    *,
    min_inliers: int = 3,
    mad_threshold: float = 3.5,
) -> dict[str, Any]:
    """Robustly combine repeated PnP poses at one stationary motor angle.

    Poses are inverted before aggregation so translations are physical camera
    centers in the fixed profile frame. Rotation and translation outliers are
    rejected independently, then the surviving camera-to-world poses are
    combined and inverted back to a world-to-camera transform.
    """
    poses = [
        _validate_transform(pose, name=f"pose[{index}]")
        for index, pose in enumerate(world_to_camera_poses)
    ]
    if len(poses) < min_inliers:
        raise ValueError(f"At least {min_inliers} poses are required")
    if mad_threshold <= 0.0:
        raise ValueError("mad_threshold must be positive")

    camera_to_world = np.stack([transform_inverse(pose) for pose in poses])
    centers = camera_to_world[:, :3, 3]
    rotations = camera_to_world[:, :3, :3]

    center_seed = np.median(centers, axis=0)
    rotation_seed = _rotation_medoid(rotations)
    translation_residuals = np.linalg.norm(centers - center_seed, axis=1)
    rotation_residuals = np.array(
        [rotation_distance_deg(rotation_seed, rotation) for rotation in rotations]
    )
    inliers = _mad_inliers(translation_residuals, mad_threshold, 1e-6)
    inliers &= _mad_inliers(rotation_residuals, mad_threshold, 1e-3)
    if int(np.count_nonzero(inliers)) < min_inliers:
        raise ValueError(
            f"Only {int(np.count_nonzero(inliers))} mutually consistent poses remain; "
            f"at least {min_inliers} are required"
        )

    combined_camera_to_world = np.eye(4)
    combined_camera_to_world[:3, :3] = _mean_rotation(rotations[inliers])
    combined_camera_to_world[:3, 3] = np.median(centers[inliers], axis=0)
    final_translation_residuals = np.linalg.norm(
        centers - combined_camera_to_world[:3, 3], axis=1
    )
    final_rotation_residuals = np.array(
        [
            rotation_distance_deg(combined_camera_to_world[:3, :3], rotation)
            for rotation in rotations
        ]
    )
    return {
        "world_to_camera": transform_inverse(combined_camera_to_world).tolist(),
        "camera_to_world": combined_camera_to_world.tolist(),
        "inlier_mask": inliers.tolist(),
        "translation_residuals_m": final_translation_residuals.tolist(),
        "rotation_residuals_deg": final_rotation_residuals.tolist(),
        "translation_spread_m": float(np.max(final_translation_residuals[inliers])),
        "rotation_spread_deg": float(np.max(final_rotation_residuals[inliers])),
    }


def _circular_angle_distance(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _find_angle_record(records: list[dict[str, Any]], angle_deg: float) -> dict[str, Any]:
    exact = [record for record in records if abs(float(record["angle_deg"]) - angle_deg) <= 1e-6]
    candidates = exact or [
        record
        for record in records
        if _circular_angle_distance(record["angle_deg"], angle_deg) <= 1e-4
    ]
    if not candidates:
        raise ValueError(f"Pose map has no measured pose for angle {angle_deg:+.1f} deg")
    valid = [record for record in candidates if record.get("pose_valid")]
    if not valid:
        raise ValueError(f"Pose map has no measured pose for angle {angle_deg:+.1f} deg")
    return valid[0]


def build_orbit_pose_map(
    angle_records: list[dict[str, Any]],
    *,
    reference_angle_deg: float,
    x_position_mm: float,
    marker_map_path: str,
    pointcloud_coordinate_frame: str = "color",
) -> dict[str, Any]:
    """Build the portable map consumed by measured-pose reconstruction."""
    if pointcloud_coordinate_frame not in ("color", "depth"):
        raise ValueError("pointcloud_coordinate_frame must be 'color' or 'depth'")
    records = [dict(record) for record in angle_records]
    reference = _find_angle_record(records, float(reference_angle_deg))
    reference_transform = reference.get(
        "world_to_pointcloud", reference.get("world_to_depth")
    )
    reference_world_to_pointcloud = _validate_transform(
        reference_transform, name="reference world_to_pointcloud"
    )

    valid_records = [record for record in records if record.get("pose_valid")]
    centers = []
    for record in valid_records:
        world_to_pointcloud = _validate_transform(
            record.get("world_to_pointcloud", record.get("world_to_depth")),
            name=f"angle {record['angle_deg']} world_to_pointcloud",
        )
        camera_to_reference = (
            reference_world_to_pointcloud @ transform_inverse(world_to_pointcloud)
        )
        record["camera_to_reference"] = camera_to_reference.tolist()
        centers.append(transform_inverse(world_to_pointcloud)[:3, 3])

    orbit_fit = None
    orbit_axis_reference = None
    if len(centers) >= 3:
        try:
            orbit_fit = fit_orbit_circle(np.asarray(centers))
            world_axis = np.asarray(orbit_fit["axis"])
            orbit_axis_reference = (
                reference_world_to_pointcloud[:3, :3] @ world_axis
            ).tolist()
        except (ValueError, np.linalg.LinAlgError):
            orbit_fit = None

    invalid_angles = [
        float(record["angle_deg"]) for record in records if not record.get("pose_valid")
    ]
    return {
        "schema_version": 1,
        "purpose": "measured aligned-pointcloud camera poses from a fixed ArUco profile",
        "pointcloud_coordinate_frame": pointcloud_coordinate_frame,
        "quality_status": "valid" if not invalid_angles else "invalid",
        "quality_reasons": (
            [] if not invalid_angles else [f"missing valid poses at angles {invalid_angles}"]
        ),
        "marker_map": str(marker_map_path),
        "x_position_mm": float(x_position_mm),
        "reference_angle_deg": float(reference["angle_deg"]),
        "reference_world_to_pointcloud": reference_world_to_pointcloud.tolist(),
        "profile_origin_in_reference_m": reference_world_to_pointcloud[:3, 3].tolist(),
        "orbit_fit_profile_frame": orbit_fit,
        "orbit_axis_reference": orbit_axis_reference,
        "angles": records,
    }


def camera_poses_in_reference(
    pose_map: dict[str, Any], angles_deg: Iterable[float]
) -> list[np.ndarray]:
    """Resolve measured raw-camera-to-reference transforms by motor angle."""
    if int(pose_map.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported orbit pose-map schema version")
    records = pose_map.get("angles")
    if not isinstance(records, list):
        raise ValueError("Pose map angles must be a list")
    poses = []
    for angle in angles_deg:
        record = _find_angle_record(records, float(angle))
        poses.append(
            _validate_transform(
                record["camera_to_reference"],
                name=f"angle {angle} camera_to_reference",
            ).copy()
        )
    return poses
