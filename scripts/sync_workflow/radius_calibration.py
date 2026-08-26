"""Pure geometry and pose helpers for ArUco orbit-radius calibration."""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_DICTIONARY = "DICT_5X5_100"
DEFAULT_MARKER_SIZES_M = (0.018, 0.014, 0.018, 0.014)
DEFAULT_PROFILE_WIDTH_M = 0.040
DEFAULT_PROFILE_DEPTH_M = 0.020
DEFAULT_CARRIER_OFFSET_M = 0.0001
DEFAULT_AXIAL_OFFSETS_M = (0.0, 0.0, 0.0, 0.0)


def _marker_corners(
    center: np.ndarray,
    horizontal: np.ndarray,
    marker_size_m: float,
) -> np.ndarray:
    """Return top-left through bottom-left corners as viewed from outside."""
    half = marker_size_m / 2.0
    vertical = np.array([0.0, 0.0, 1.0])
    return np.array(
        [
            center - horizontal * half + vertical * half,
            center + horizontal * half + vertical * half,
            center + horizontal * half - vertical * half,
            center - horizontal * half - vertical * half,
        ],
        dtype=np.float64,
    )


def marker_normal(corners: np.ndarray) -> np.ndarray:
    """Calculate the outward normal from clockwise OpenCV marker corners."""
    points = np.asarray(corners, dtype=np.float64)
    normal = -np.cross(points[1] - points[0], points[2] - points[1])
    length = np.linalg.norm(normal)
    if length == 0.0:
        raise ValueError("Marker corners are degenerate.")
    return normal / length


def build_profile_marker_map(
    *,
    dictionary: str = DEFAULT_DICTIONARY,
    marker_sizes_m: Iterable[float] = DEFAULT_MARKER_SIZES_M,
    profile_width_m: float = DEFAULT_PROFILE_WIDTH_M,
    profile_depth_m: float = DEFAULT_PROFILE_DEPTH_M,
    carrier_offset_m: float = DEFAULT_CARRIER_OFFSET_M,
    axial_offsets_m: Iterable[float] = DEFAULT_AXIAL_OFFSETS_M,
) -> dict[str, Any]:
    """Build the fixed 3D marker map for a rectangular four-face profile."""
    marker_sizes = tuple(float(value) for value in marker_sizes_m)
    if len(marker_sizes) != 4 or any(value <= 0.0 for value in marker_sizes):
        raise ValueError("Exactly four positive marker sizes are required")
    if profile_width_m <= 0.0 or profile_depth_m <= 0.0:
        raise ValueError("Profile dimensions must be positive")
    if carrier_offset_m < 0.0:
        raise ValueError("carrier_offset_m cannot be negative")

    z_offsets = tuple(float(value) for value in axial_offsets_m)
    if len(z_offsets) != 4:
        raise ValueError("Exactly four axial marker offsets are required")

    half_width = profile_width_m / 2.0 + carrier_offset_m
    half_depth = profile_depth_m / 2.0 + carrier_offset_m
    definitions = [
        (0, "front", [0.0, half_depth, z_offsets[0]], [-1.0, 0.0, 0.0]),
        (1, "top", [half_width, 0.0, z_offsets[1]], [0.0, 1.0, 0.0]),
        (2, "back", [0.0, -half_depth, z_offsets[2]], [1.0, 0.0, 0.0]),
        (3, "bottom", [-half_width, 0.0, z_offsets[3]], [0.0, -1.0, 0.0]),
    ]

    markers = []
    for marker_id, face, center_values, horizontal_values in definitions:
        center = np.asarray(center_values, dtype=np.float64)
        horizontal = np.asarray(horizontal_values, dtype=np.float64)
        marker_size_m = marker_sizes[marker_id]
        corners = _marker_corners(center, horizontal, marker_size_m)
        markers.append(
            {
                "id": marker_id,
                "face": face,
                "size_m": marker_size_m,
                "center_m": center.tolist(),
                "corners_m": corners.tolist(),
                "outward_normal": marker_normal(corners).tolist(),
                "bar_up": [0.0, 0.0, 1.0],
            }
        )

    return {
        "schema_version": 2,
        "purpose": "same-position continuous-wrap orbit-radius calibration",
        "dictionary": dictionary,
        "marker_sizes_m": list(marker_sizes),
        "profile_cross_section_m": [float(profile_width_m), float(profile_depth_m)],
        "carrier_offset_m": float(carrier_offset_m),
        "world_frame": {
            "units": "meters",
            "origin": "profile cross-section center on the shared marker centerline",
            "x_axis": "toward the top profile face",
            "y_axis": "toward the front profile face",
            "z_axis": "along the horizontal profile toward its chosen right end",
        },
        "markers": markers,
    }


def transform_inverse(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def convert_world_to_color_to_world_to_depth(
    world_to_color: np.ndarray,
    depth_to_color: np.ndarray,
) -> np.ndarray:
    """Convert T_color_from_world using the SDK's T_color_from_depth."""
    return transform_inverse(depth_to_color) @ np.asarray(world_to_color, dtype=np.float64)


def orbbec_extrinsic_to_matrix(extrinsic: Any) -> np.ndarray:
    """Convert an Orbbec depth-to-color extrinsic object to a 4x4 matrix."""
    rotation = np.asarray(extrinsic.rot, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(extrinsic.transform, dtype=np.float64).reshape(3)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    # The Orbbec SDK exposes extrinsic translation in millimetres.
    matrix[:3, 3] = translation * 0.001
    return matrix


def camera_center_world(world_to_camera: np.ndarray) -> np.ndarray:
    return transform_inverse(world_to_camera)[:3, 3]


def _fit_circle_once(points: np.ndarray) -> dict[str, Any]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = vh[-1]
    basis_u = vh[0]
    basis_v = np.cross(axis, basis_u)
    basis_v /= np.linalg.norm(basis_v)

    coordinates = np.column_stack(
        ((points - centroid) @ basis_u, (points - centroid) @ basis_v)
    )
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points))))
    solution, _, _, _ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    center_2d = solution[:2]
    radius_sq = solution[2] + np.dot(center_2d, center_2d)
    if radius_sq <= 0.0:
        raise ValueError("Circle fit produced a non-positive radius")
    radius = float(np.sqrt(radius_sq))
    center_3d = centroid + center_2d[0] * basis_u + center_2d[1] * basis_v

    plane_distance = (points - center_3d) @ axis
    radial_distance = np.linalg.norm(coordinates - center_2d, axis=1)
    residuals = np.sqrt((radial_distance - radius) ** 2 + plane_distance**2)
    return {
        "center": center_3d,
        "axis": axis,
        "radius": radius,
        "residuals": residuals,
    }


def fit_orbit_circle(
    camera_centers_m: np.ndarray,
    *,
    mad_threshold: float = 3.5,
) -> dict[str, Any]:
    """Robustly fit a plane and circle to 3D camera-center samples."""
    points = np.asarray(camera_centers_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("At least three 3D camera centers are required")
    if not np.isfinite(points).all():
        raise ValueError("Camera centers must be finite")

    first = _fit_circle_once(points)
    residuals = first["residuals"]
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_sigma = 1.4826 * mad
    limit = median + mad_threshold * max(robust_sigma, 1e-9)
    inliers = residuals <= limit
    if np.count_nonzero(inliers) < 3:
        inliers = np.ones(len(points), dtype=bool)

    fitted = _fit_circle_once(points[inliers])
    center = fitted["center"]
    axis = fitted["axis"]
    if axis[2] < 0.0:
        axis = -axis

    centered = points - center
    plane_distance = centered @ axis
    radial_vectors = centered - np.outer(plane_distance, axis)
    radial_distance = np.linalg.norm(radial_vectors, axis=1)
    all_residuals = np.sqrt(
        (radial_distance - fitted["radius"]) ** 2 + plane_distance**2
    )
    inlier_residuals = all_residuals[inliers]
    return {
        "radius_m": float(fitted["radius"]),
        "center_m": center.tolist(),
        "axis": axis.tolist(),
        "residuals_m": all_residuals.tolist(),
        "inlier_mask": inliers.tolist(),
        "rmse_m": float(np.sqrt(np.mean(inlier_residuals**2))),
        "max_residual_m": float(np.max(inlier_residuals)),
    }


def has_sufficient_angular_coverage(
    angles_deg: Iterable[float],
    *,
    min_unique_angles: int = 6,
    max_gap_deg: float = 90.0,
) -> bool:
    canonical = sorted({round(float(angle) % 360.0, 6) for angle in angles_deg})
    if len(canonical) < min_unique_angles:
        return False
    wrapped = canonical + [canonical[0] + 360.0]
    largest_gap = max(b - a for a, b in zip(wrapped, wrapped[1:]))
    return largest_gap <= max_gap_deg + 1e-9


def evaluate_trajectory(
    samples: list[dict[str, Any]],
    *,
    min_unique_angles: int = 6,
    max_gap_deg: float = 90.0,
    mad_threshold: float = 3.5,
) -> dict[str, Any]:
    """Fit RGB/depth trajectories and decide whether a radius is publishable."""
    reasons: list[str] = []
    if len(samples) < 3:
        return {
            "quality_status": "invalid",
            "quality_reasons": ["at least three accepted angle samples are required"],
            "recommended_radius_m": None,
            "rgb_fit": None,
            "depth_fit": None,
        }

    rgb_points = np.asarray(
        [sample["rgb_camera_center_m"] for sample in samples], dtype=np.float64
    )
    depth_points = np.asarray(
        [sample["depth_camera_center_m"] for sample in samples], dtype=np.float64
    )
    try:
        rgb_fit = fit_orbit_circle(rgb_points, mad_threshold=mad_threshold)
        depth_fit = fit_orbit_circle(depth_points, mad_threshold=mad_threshold)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {
            "quality_status": "invalid",
            "quality_reasons": [f"trajectory circle fit failed: {exc}"],
            "recommended_radius_m": None,
            "rgb_fit": None,
            "depth_fit": None,
        }
    depth_inliers = np.asarray(depth_fit["inlier_mask"], dtype=bool)
    inlier_angles = [
        float(sample["angle_deg"])
        for sample, is_inlier in zip(samples, depth_inliers)
        if is_inlier
    ]

    if len({round(angle % 360.0, 6) for angle in inlier_angles}) < min_unique_angles:
        reasons.append(f"fewer than {min_unique_angles} unique inlier angles")
    if not has_sufficient_angular_coverage(
        inlier_angles,
        min_unique_angles=min_unique_angles,
        max_gap_deg=max_gap_deg,
    ):
        reasons.append(
            f"insufficient angular coverage (largest allowed circular gap is {max_gap_deg:.1f} deg)"
        )

    valid = not reasons
    return {
        "quality_status": "valid" if valid else "invalid",
        "quality_reasons": reasons,
        "recommended_radius_m": depth_fit["radius_m"] if valid else None,
        "rgb_fit": rgb_fit,
        "depth_fit": depth_fit,
    }


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return matrix


def _mean_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - np.asarray(image_points).reshape(-1, 2), axis=1
    )
    return float(np.mean(errors))


def _marker_local_to_world(marker: dict[str, Any]) -> np.ndarray:
    corners = np.asarray(marker["corners_m"], dtype=np.float64)
    center = corners.mean(axis=0)
    horizontal = corners[1] - corners[0]
    horizontal /= np.linalg.norm(horizontal)
    vertical = corners[0] - corners[3]
    vertical /= np.linalg.norm(vertical)
    local_z = np.cross(horizontal, vertical)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((horizontal, vertical, local_z))
    transform[:3, 3] = center
    return transform


def solve_profile_pose(
    marker_map: dict[str, Any],
    marker_corners,
    marker_ids: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, Any]:
    """Solve fixed-profile world-to-color pose, including one-marker views."""
    markers_by_id = {
        int(marker["id"]): marker for marker in marker_map.get("markers", [])
    }
    known = []
    for corners, raw_id in zip(marker_corners, np.asarray(marker_ids).reshape(-1)):
        marker_id = int(raw_id)
        if marker_id in markers_by_id:
            known.append(
                (
                    marker_id,
                    markers_by_id[marker_id],
                    # ArUco returns float32 corners. Keeping that dtype also
                    # avoids an OpenCV IPPE degeneracy on exactly frontal views.
                    np.asarray(corners, dtype=np.float32).reshape(4, 2),
                )
            )

    empty_result = {
        "ok": False,
        "method": None,
        "used_ids": [item[0] for item in known],
        "world_to_camera": None,
        "rvec": None,
        "tvec": None,
        "reprojection_error_px": None,
    }
    if not known:
        return empty_result

    if len(known) >= 2:
        object_points = np.concatenate(
            [np.asarray(item[1]["corners_m"], dtype=np.float64) for item in known]
        )
        image_points = np.concatenate([item[2] for item in known])
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return empty_result
        world_to_camera = rvec_tvec_to_matrix(rvec, tvec)
        camera_points = (
            world_to_camera[:3, :3] @ object_points.T
            + world_to_camera[:3, 3:4]
        ).T
        if np.any(camera_points[:, 2] <= 0.0):
            return empty_result
        return {
            "ok": True,
            "method": "multi-marker-iterative",
            "used_ids": [item[0] for item in known],
            "world_to_camera": world_to_camera,
            "rvec": rvec,
            "tvec": tvec,
            "reprojection_error_px": _mean_reprojection_error(
                object_points,
                image_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            ),
        }

    marker_id, marker, image_points = known[0]
    marker_size_value = marker.get("size_m", marker_map.get("marker_size_m"))
    if marker_size_value is None:
        return empty_result
    marker_size_m = float(marker_size_value)
    half = marker_size_m / 2.0
    local_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    result = cv2.solvePnPGeneric(
        local_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    ok, rvecs, tvecs = bool(result[0]), result[1], result[2]
    if not ok:
        return empty_result

    marker_to_world = _marker_local_to_world(marker)
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        marker_to_camera = rvec_tvec_to_matrix(rvec, tvec)
        camera_points = (
            marker_to_camera[:3, :3] @ local_points.T
            + marker_to_camera[:3, 3:4]
        ).T
        if np.any(camera_points[:, 2] <= 0.0):
            continue
        error = _mean_reprojection_error(
            local_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        world_to_camera = marker_to_camera @ transform_inverse(marker_to_world)
        camera_center = camera_center_world(world_to_camera)
        marker_center = np.asarray(marker["center_m"], dtype=np.float64)
        outward_normal = np.asarray(marker["outward_normal"], dtype=np.float64)
        if float(np.dot(camera_center - marker_center, outward_normal)) <= 0.0:
            continue
        candidates.append((error, world_to_camera, rvec, tvec, "single-marker-ippe"))
    if not candidates:
        object_points = np.asarray(marker["corners_m"], dtype=np.float64)
        fallback_ok, fallback_rvec, fallback_tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if fallback_ok:
            fallback_pose = rvec_tvec_to_matrix(fallback_rvec, fallback_tvec)
            fallback_center = camera_center_world(fallback_pose)
            marker_center = np.asarray(marker["center_m"], dtype=np.float64)
            outward_normal = np.asarray(marker["outward_normal"], dtype=np.float64)
            if float(np.dot(fallback_center - marker_center, outward_normal)) > 0.0:
                fallback_error = _mean_reprojection_error(
                    object_points,
                    image_points,
                    fallback_rvec,
                    fallback_tvec,
                    camera_matrix,
                    dist_coeffs,
                )
                candidates.append(
                    (
                        fallback_error,
                        fallback_pose,
                        fallback_rvec,
                        fallback_tvec,
                        "single-marker-iterative-fallback",
                    )
                )
    if not candidates:
        return empty_result

    error, world_to_camera, rvec, tvec, method = min(
        candidates, key=lambda item: item[0]
    )
    return {
        "ok": True,
        "method": method,
        "used_ids": [marker_id],
        "world_to_camera": world_to_camera,
        "rvec": rvec,
        "tvec": tvec,
        "reprojection_error_px": error,
    }
