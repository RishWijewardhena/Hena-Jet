"""Shared selection and validation of the rig's measured orbit calibration."""

import json
from pathlib import Path

import numpy as np


DEFAULT_ORBIT_GEOMETRY = (
    Path(__file__).resolve().parents[2]
    / "outputs/test_radius_x100/radius_calibration.json"
)


def select_orbit_geometry(explicit=None, metadata=None):
    """Select an explicit file, a recorded scan file, or the fixed rig file."""
    recorded = (metadata or {}).get("orbit_geometry_source")
    return Path(explicit or recorded or DEFAULT_ORBIT_GEOMETRY).expanduser().resolve()


def load_orbit_geometry(path):
    """Require a valid measured centre, direction, radius, and calibration station."""
    path = Path(path)
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read orbit calibration {path}: {exc}") from exc
    if not isinstance(calibration, dict) or calibration.get("quality_status") != "valid":
        raise ValueError(f"{path}: orbit calibration quality_status must be valid")
    geometry = calibration.get("orbit_geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"{path}: missing orbit_geometry; re-run radius calibration")
    try:
        pivot = np.asarray(geometry["pivot_m"], dtype=float)
        axis = np.asarray(geometry["axis"], dtype=float)
        radius = float(calibration["recommended_radius_m"])
        station = float(calibration["motor"]["x_position_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: incomplete orbit geometry, radius, or X station") from exc
    if (pivot.shape != (3,) or axis.shape != (3,)
            or not np.isfinite(pivot).all() or not np.isfinite(axis).all()
            or not np.isfinite(np.linalg.norm(axis)) or np.linalg.norm(axis) <= 0
            or not np.isfinite(radius) or radius <= 0 or not np.isfinite(station)):
        raise ValueError(f"{path}: orbit geometry must be finite with a positive radius and nonzero axis")
    geometry["axis"] = (axis / np.linalg.norm(axis)).tolist()
    return calibration


def validate_camera_frame(calibration, intrinsics):
    """Measured colour-camera geometry must not be used with depth-camera points."""
    expected = calibration.get("camera", {}).get("pointcloud_coordinate_frame")
    actual = (intrinsics or {}).get("coordinate_frame")
    if expected and actual and expected != actual:
        raise ValueError(f"Calibration coordinate-frame mismatch: {expected!r} vs scan {actual!r}")
