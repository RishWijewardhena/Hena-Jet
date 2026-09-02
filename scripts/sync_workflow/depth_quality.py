"""Pure quality metrics for captured depth and reconstructed point clouds."""

from __future__ import annotations

import numpy as np


def fill_rate(depth_m: np.ndarray) -> float:
    """Fraction of pixels carrying a usable depth measurement."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.size == 0:
        return 0.0
    valid = np.isfinite(depth) & (depth > 0.0)
    return float(np.count_nonzero(valid) / depth.size)


def surface_plane_rms_m(points: np.ndarray) -> float:
    """RMS distance of a local surface patch to its best-fit plane."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("At least three 3D points are required")
    if not np.isfinite(pts).all():
        raise ValueError("Points must be finite")
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]
    return float(np.sqrt(np.mean(((pts - centroid) @ normal) ** 2)))


def cross_view_residual_m(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    max_pair_distance_m: float = 0.008,
    min_pairs: int = 100,
    tree_b=None,
) -> float | None:
    """Median nearest-neighbour distance from *points_a* to *points_b*.

    Only pairs closer than ``max_pair_distance_m`` count, so non-overlapping
    regions do not dominate. Returns ``None`` when the overlap is too small.

    Args:
        points_a: Nx3 query points
        points_b: Nx3 target points (or None if tree_b is provided)
        max_pair_distance_m: Maximum distance threshold for counted pairs
        min_pairs: Minimum number of pairs required to return a value
        tree_b: Optional prebuilt cKDTree over points_b; if None, builds internally
    """
    from scipy.spatial import cKDTree

    a = np.asarray(points_a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("points_a must be Nx3")
    if len(a) == 0:
        return None

    if tree_b is None:
        b = np.asarray(points_b, dtype=np.float64)
        if b.ndim != 2 or b.shape[1] != 3:
            raise ValueError("Both inputs must be Nx3 point arrays")
        if len(b) == 0:
            return None
        tree_b = cKDTree(b)

    distances, _ = tree_b.query(a)
    overlapping = distances[distances < max_pair_distance_m]
    if len(overlapping) < min_pairs:
        return None
    return float(np.median(overlapping))
