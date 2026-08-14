"""Small, dependency-light geometry primitives used by the experiment."""

from __future__ import annotations

import numpy as np


def backproject_depth(
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_scale_m: float,
    min_depth_m: float,
    max_depth_m: float,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid camera-space points and their ``(u, v)`` pixels."""
    depth = np.asarray(depth)
    if depth.ndim != 2 or stride < 1:
        raise ValueError("depth must be a 2D image and stride must be positive")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if fx <= 0 or fy <= 0 or depth_scale_m <= 0:
        raise ValueError("focal lengths and depth scale must be positive")
    if min_depth_m < 0 or max_depth_m <= min_depth_m:
        raise ValueError("invalid depth range")

    rows, columns = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride].astype(np.float64) * depth_scale_m
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    u = columns[valid].astype(np.float64)
    v = rows[valid].astype(np.float64)
    z = z[valid]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return points, np.column_stack((u.astype(np.int32), v.astype(np.int32)))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("expected Nx3 points and a 4x4 transform")
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Estimate the least-squares transform mapping ``source`` to ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching Nx3 arrays")
    if source.shape[0] < 3:
        raise ValueError("at least three correspondences are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform
