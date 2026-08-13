"""Pure NumPy quality controls for close-range ZED + KISS-ICP scanning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilteredCloud:
    points: np.ndarray
    colors: np.ndarray
    mask: np.ndarray
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class FrameQuality:
    accepted: bool
    reason: str
    point_count: int
    valid_coverage: float
    extents_m: tuple[float, float, float]
    eigenvalues: tuple[float, float, float]


@dataclass(frozen=True)
class PoseQuality:
    accepted: bool
    reason: str
    translation_m: float
    rotation_deg: float


def _neighbor_of(mask: np.ndarray) -> np.ndarray:
    """Return pixels touching ``mask`` in an eight-connected neighborhood."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbors |= padded[dy : dy + height, dx : dx + width]
    return neighbors


def _depth_discontinuities(
    depth: np.ndarray,
    valid: np.ndarray,
    threshold_m: float,
) -> np.ndarray:
    edges = np.zeros_like(valid, dtype=bool)
    if threshold_m <= 0:
        return edges

    horizontal = valid[:, :-1] & valid[:, 1:]
    horizontal &= np.abs(depth[:, :-1] - depth[:, 1:]) > threshold_m
    edges[:, :-1] |= horizontal
    edges[:, 1:] |= horizontal

    vertical = valid[:-1, :] & valid[1:, :]
    vertical &= np.abs(depth[:-1, :] - depth[1:, :]) > threshold_m
    edges[:-1, :] |= vertical
    edges[1:, :] |= vertical
    return edges


def filter_organized_cloud(
    cloud: np.ndarray,
    confidence: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    coordinate_system: str,
    confidence_threshold: float,
    edge_threshold_m: float,
    erode_invalid_boundary: bool,
) -> FilteredCloud:
    """Filter an organized ZED XYZRGBA cloud and decode its packed color."""
    cloud = np.asarray(cloud, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if cloud.ndim != 3 or cloud.shape[2] < 4:
        raise ValueError("cloud must have shape (height, width, >=4)")
    if confidence.ndim == 3:
        confidence = confidence[:, :, 0]
    if confidence.shape != cloud.shape[:2]:
        raise ValueError("confidence shape must match cloud height and width")
    if max_depth_m <= min_depth_m:
        raise ValueError("max_depth_m must be greater than min_depth_m")

    xyz = cloud[:, :, :3]
    rgba = cloud[:, :, 3]
    forward_axis = 2 if coordinate_system == "IMAGE" else 0
    depth = xyz[:, :, forward_axis]
    magnitude = np.linalg.norm(xyz, axis=2)

    # RGBA is stored as packed bits in a float channel, so its numeric float
    # interpretation may be NaN even when the color bytes are valid.
    finite = np.isfinite(xyz).all(axis=2) & np.isfinite(confidence)
    in_range = (depth >= min_depth_m) & (depth <= max_depth_m)
    trusted = confidence <= confidence_threshold
    base_valid = finite & (magnitude > 1e-6) & in_range & trusted

    edges = _depth_discontinuities(depth, base_valid, edge_threshold_m)
    invalid_boundary = (
        _neighbor_of(~base_valid)
        if erode_invalid_boundary
        else np.zeros_like(base_valid, dtype=bool)
    )
    mask = base_valid & ~edges & ~invalid_boundary

    flat_rgba = np.ascontiguousarray(rgba[mask], dtype=np.float32)
    packed = flat_rgba.view(np.uint32)
    colors = np.column_stack(
        (
            (packed >> 16) & 0xFF,
            (packed >> 8) & 0xFF,
            packed & 0xFF,
        )
    ).astype(np.uint8)
    points = xyz[mask].astype(np.float32, copy=True)

    pixel_count = int(mask.size)
    finite_confidence = confidence[np.isfinite(confidence)]
    metrics: dict[str, float | int] = {
        "pixel_count": pixel_count,
        "base_valid_points": int(np.count_nonzero(base_valid)),
        "filtered_valid_points": int(points.shape[0]),
        "base_valid_coverage": float(np.count_nonzero(base_valid) / pixel_count),
        "filtered_valid_coverage": float(points.shape[0] / pixel_count),
        "edge_rejected_points": int(np.count_nonzero(base_valid & edges)),
        "confidence_mean": (
            float(np.mean(finite_confidence)) if finite_confidence.size else math.nan
        ),
        "confidence_median": (
            float(np.median(finite_confidence)) if finite_confidence.size else math.nan
        ),
    }
    return FilteredCloud(points=points, colors=colors, mask=mask, metrics=metrics)


def deterministic_voxel_sample(
    points: np.ndarray,
    *,
    voxel_m: float,
    max_points: int,
) -> np.ndarray:
    """Return deterministic voxel centroids, evenly capped when necessary."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] == 0:
        return points.copy()
    if voxel_m <= 0:
        raise ValueError("voxel_m must be positive")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    keys = np.floor(points / voxel_m).astype(np.int64)
    unique_keys, inverse, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inverse, points.astype(np.float64))
    centroids = (sums / counts[:, None]).astype(points.dtype, copy=False)

    if centroids.shape[0] <= max_points:
        return centroids
    selected = np.linspace(0, centroids.shape[0] - 1, max_points, dtype=np.int64)
    return centroids[selected]


def evaluate_frame_quality(
    points: np.ndarray,
    valid_coverage: float,
    *,
    min_points: int = 2_000,
    min_coverage: float = 0.05,
    min_largest_extent_m: float = 0.03,
    min_second_extent_m: float = 0.015,
    min_eigenvalue_ratio: float = 0.002,
    min_thickness_std_m: float = 0.0005,
) -> FrameQuality:
    """Reject sparse or geometrically degenerate point clouds before ICP."""
    points = np.asarray(points, dtype=np.float64)
    empty_extents = (0.0, 0.0, 0.0)
    empty_eigenvalues = (0.0, 0.0, 0.0)
    if points.shape[0] < min_points:
        return FrameQuality(
            False,
            "too_few_points",
            int(points.shape[0]),
            float(valid_coverage),
            empty_extents,
            empty_eigenvalues,
        )
    if valid_coverage < min_coverage:
        return FrameQuality(
            False,
            "low_valid_coverage",
            int(points.shape[0]),
            float(valid_coverage),
            empty_extents,
            empty_eigenvalues,
        )

    extents = np.sort(np.ptp(points, axis=0))[::-1]
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    extents_tuple = tuple(float(value) for value in extents)
    eigenvalues_tuple = tuple(float(value) for value in eigenvalues)

    if extents[0] < min_largest_extent_m or extents[1] < min_second_extent_m:
        return FrameQuality(
            False,
            "insufficient_spatial_extent",
            int(points.shape[0]),
            float(valid_coverage),
            extents_tuple,
            eigenvalues_tuple,
        )

    ratio = eigenvalues[0] / max(eigenvalues[-1], np.finfo(np.float64).eps)
    thickness_std = math.sqrt(eigenvalues[0])
    if ratio < min_eigenvalue_ratio or thickness_std < min_thickness_std_m:
        return FrameQuality(
            False,
            "degenerate_geometry",
            int(points.shape[0]),
            float(valid_coverage),
            extents_tuple,
            eigenvalues_tuple,
        )

    return FrameQuality(
        True,
        "accepted",
        int(points.shape[0]),
        float(valid_coverage),
        extents_tuple,
        eigenvalues_tuple,
    )


def pose_step(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return relative translation in metres and rotation in degrees."""
    delta = np.linalg.inv(previous) @ current
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return translation_m, math.degrees(math.acos(cosine))


def evaluate_pose_quality(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_deg: float,
) -> PoseQuality:
    translation_m, rotation_deg = pose_step(previous, current)
    accepted = (
        translation_m <= max_translation_m and rotation_deg <= max_rotation_deg
    )
    return PoseQuality(
        accepted=accepted,
        reason="accepted" if accepted else "tracking_jump",
        translation_m=translation_m,
        rotation_deg=rotation_deg,
    )


class GlobalVoxelAccumulator:
    """Frame-weighted running averages in a global voxel grid."""

    def __init__(self, voxel_m: float) -> None:
        if voxel_m <= 0:
            raise ValueError("voxel_m must be positive")
        self.voxel_m = float(voxel_m)
        self._voxels: dict[
            tuple[int, int, int], tuple[np.ndarray, np.ndarray, int]
        ] = {}

    def __len__(self) -> int:
        return len(self._voxels)

    def update(self, points: np.ndarray, colors: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points and colors must both have shape (N, 3)")
        if points.shape[0] == 0:
            return

        keys = np.floor(points / self.voxel_m).astype(np.int64)
        unique_keys, inverse, counts = np.unique(
            keys,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        point_sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        color_sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(point_sums, inverse, points)
        np.add.at(color_sums, inverse, colors)
        frame_points = point_sums / counts[:, None]
        frame_colors = color_sums / counts[:, None]

        for key_array, point, color in zip(unique_keys, frame_points, frame_colors):
            key = tuple(int(value) for value in key_array)
            prior = self._voxels.get(key)
            if prior is None:
                self._voxels[key] = (point.copy(), color.copy(), 1)
            else:
                point_sum, color_sum, observations = prior
                self._voxels[key] = (
                    point_sum + point,
                    color_sum + color,
                    observations + 1,
                )

    def to_arrays(
        self,
        *,
        min_observations: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if min_observations <= 0:
            raise ValueError("min_observations must be positive")

        selected = [
            (key, value)
            for key, value in sorted(self._voxels.items())
            if value[2] >= min_observations
        ]
        if not selected:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )

        counts = np.asarray([value[2] for _, value in selected], dtype=np.int32)
        points = np.asarray(
            [value[0] / value[2] for _, value in selected],
            dtype=np.float32,
        )
        colors_float = np.asarray(
            [value[1] / value[2] for _, value in selected],
            dtype=np.float64,
        )
        colors = np.clip(np.rint(colors_float), 0, 255).astype(np.uint8)
        return points, colors, counts
