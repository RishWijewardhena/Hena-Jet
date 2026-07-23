#!/usr/bin/env python3
"""Capture continuous ZED VSLAM orbits at multiple measured camera heights."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def normalize_height_offsets_m(values: list[float]) -> tuple[float, ...]:
    """Validate absolute camera-height offsets measured from the first pass."""
    offsets = tuple(float(value) for value in values)
    if len(offsets) < 2:
        raise ValueError("At least two camera height offsets are required")
    if not all(math.isfinite(value) and value >= 0.0 for value in offsets):
        raise ValueError("Camera height offsets must be finite and non-negative")
    if not math.isclose(offsets[0], 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("The first camera height offset must be 0 m")
    if any(current <= previous for previous, current in zip(offsets, offsets[1:])):
        raise ValueError("Camera height offsets must be strictly increasing")
    return offsets


def pass_directory(root: Path, pass_index: int, height_offset_m: float) -> Path:
    """Return the stable output directory for one absolute-height pass."""
    if pass_index < 0 or not math.isfinite(height_offset_m) or height_offset_m < 0.0:
        raise ValueError("Pass index and height offset must be non-negative")
    height_mm = int(round(height_offset_m * 1000.0))
    return Path(root) / f"pass_{pass_index:02d}_height_{height_mm:03d}mm"


def transition_is_ready(
    elapsed_s: float,
    minimum_wait_s: float,
    operator_confirmed: bool,
) -> bool:
    """Require both the settling interval and explicit operator confirmation."""
    if not math.isfinite(elapsed_s) or not math.isfinite(minimum_wait_s):
        raise ValueError("Transition timing must be finite")
    if elapsed_s < 0.0 or minimum_wait_s < 0.0:
        raise ValueError("Transition timing cannot be negative")
    return bool(operator_confirmed and elapsed_s >= minimum_wait_s)


def _validate_rigid_pose(pose: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} must have a rigid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation must have determinant +1")
    return matrix


def lift_pose_metrics(
    before_pose: np.ndarray,
    after_pose: np.ndarray,
    *,
    expected_height_delta_m: float,
    object_up: np.ndarray,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict:
    """Measure a manual lift using consecutive camera-to-world VSLAM poses."""
    before = _validate_rigid_pose(before_pose, "before_pose")
    after = _validate_rigid_pose(after_pose, "after_pose")
    up = np.asarray(object_up, dtype=np.float64)
    if up.shape != (3,) or not np.all(np.isfinite(up)):
        raise ValueError("object_up must be a finite three-vector")
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 0.0:
        raise ValueError("object_up cannot be zero")
    if (
        not math.isfinite(expected_height_delta_m)
        or expected_height_delta_m <= 0.0
        or not math.isfinite(translation_tolerance_m)
        or translation_tolerance_m < 0.0
        or not math.isfinite(rotation_tolerance_deg)
        or rotation_tolerance_deg < 0.0
    ):
        raise ValueError("Lift target and tolerances are invalid")

    up = up / up_norm
    translation = after[:3, 3] - before[:3, 3]
    vertical_translation_m = float(translation @ up)
    lateral = translation - vertical_translation_m * up
    lateral_error_m = float(np.linalg.norm(lateral))
    vertical_error_m = abs(vertical_translation_m - expected_height_delta_m)

    relative_rotation = before[:3, :3].T @ after[:3, :3]
    cosine = (float(np.trace(relative_rotation)) - 1.0) / 2.0
    rotation_error_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, cosine)))
    )
    accepted = bool(
        vertical_error_m <= translation_tolerance_m
        and lateral_error_m <= translation_tolerance_m
        and rotation_error_deg <= rotation_tolerance_deg
    )
    return {
        "accepted": accepted,
        "expected_height_delta_m": float(expected_height_delta_m),
        "vertical_translation_m": vertical_translation_m,
        "vertical_error_m": vertical_error_m,
        "lateral_error_m": lateral_error_m,
        "rotation_error_deg": rotation_error_deg,
    }
