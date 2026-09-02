"""Exact float back-projection of aligned RGB-D frames to coloured points."""

from __future__ import annotations

import numpy as np


def backproject_to_points(
    depth_m: np.ndarray,
    color_rgb: np.ndarray,
    intrinsics: dict,
    *,
    depth_trunc_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project aligned depth and colour without any integer round-trip.

    Open3D's ``RGBDImage`` path requires a uint16 millimetre depth image, which
    truncates every sample to the millimetre below and biases the cloud by
    about -0.5 mm. This keeps full float precision.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    color = np.asarray(color_rgb)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if color.ndim != 3 or color.shape[2] != 3 or color.shape[:2] != depth.shape:
        raise ValueError("color_rgb must be HxWx3 and match the depth shape")

    valid = np.isfinite(depth) & (depth > 0.0)
    if depth_trunc_m is not None:
        valid &= depth <= float(depth_trunc_m)

    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    z = depth[rows, cols]
    x = (cols - float(intrinsics["cx"])) / float(intrinsics["fx"]) * z
    y = (rows - float(intrinsics["cy"])) / float(intrinsics["fy"]) * z
    points = np.column_stack((x, y, z))
    colors = color[rows, cols].astype(np.float64) / 255.0
    return points, colors
