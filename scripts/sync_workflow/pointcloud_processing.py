"""Full-resolution point-cloud processing with Open3D and Trimesh."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


logger = logging.getLogger(__name__)


def _geometry_libraries():
    try:
        import open3d as o3d
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Reconstruction requires Open3D and Trimesh. "
            "Activate the hena_jet environment."
        ) from exc
    return o3d, trimesh


def _read_colored_cloud(o3d, path: Path):
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Open3D could not read any points from {path}")

    points = np.asarray(cloud.points)
    if not np.isfinite(points).all():
        raise RuntimeError(f"Point cloud contains non-finite coordinates: {path}")
    if not cloud.has_colors() or len(cloud.colors) != len(cloud.points):
        raise RuntimeError(f"Point cloud has no complete RGB data: {path}")
    if not np.isfinite(np.asarray(cloud.colors)).all():
        raise RuntimeError(f"Point cloud contains non-finite RGB values: {path}")
    return cloud


def _write_cloud(o3d, path: Path, cloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    if not written:
        raise RuntimeError(f"Open3D could not write point cloud: {path}")


def _transform_one_cloud(
    source_path: Path,
    transform: np.ndarray,
    transformed_dir: Path,
    matrix_dir: Path,
    *,
    crop_bounds: Optional[tuple[float, float, float, float, float, float]],
    skip_sor: bool,
    sor_neighbors: int,
    sor_sigma: float,
) -> tuple[Path, dict]:
    o3d, _ = _geometry_libraries()
    started = time.perf_counter()
    pose = np.asarray(transform, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid 4x4 transform for {source_path}")

    cloud = _read_colored_cloud(o3d, source_path)
    input_points = len(cloud.points)
    cloud.transform(pose)

    if crop_bounds is not None:
        bounds = np.asarray(crop_bounds, dtype=float)
        if bounds.shape != (6,) or not np.isfinite(bounds).all():
            raise ValueError("crop_bounds must contain six finite values")
        crop = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bounds[:3],
            max_bound=bounds[3:],
        )
        cloud = cloud.crop(crop)
    cropped_points = len(cloud.points)
    if cloud.is_empty():
        raise RuntimeError(f"Transform/crop removed every point from {source_path}")

    if not skip_sor and len(cloud.points) > sor_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=sor_neighbors,
            std_ratio=sor_sigma,
        )
    if cloud.is_empty():
        raise RuntimeError(f"Per-scan SOR removed every point from {source_path}")

    matrix_path = matrix_dir / f"{source_path.stem}_optimized_matrix.txt"
    output_path = transformed_dir / f"{source_path.stem}_transformed.ply"
    np.savetxt(matrix_path, pose, fmt="%.10f")
    _write_cloud(o3d, output_path, cloud)

    stats = {
        "source": source_path.name,
        "output": output_path.name,
        "input_points": input_points,
        "cropped_points": cropped_points,
        "output_points": len(cloud.points),
        "sor_applied": not skip_sor and cropped_points > sor_neighbors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    logger.info(
        "Processed %s: %d -> %d points",
        source_path.name,
        input_points,
        len(cloud.points),
    )
    return output_path, stats


def transform_and_clean_clouds(
    source_paths: Sequence[Path],
    transforms: Sequence[np.ndarray],
    transformed_dir: Path,
    matrix_dir: Path,
    *,
    crop_bounds: Optional[tuple[float, float, float, float, float, float]],
    crop_bounds_by_cloud: Optional[
        Sequence[Optional[tuple[float, float, float, float, float, float]]]
    ] = None,
    skip_sor: bool,
    sor_neighbors: int,
    sor_sigma: float,
    max_workers: int = 4,
) -> tuple[list[Path], list[dict]]:
    """Transform, crop, filter, and save full-resolution scans."""
    if len(source_paths) != len(transforms):
        raise ValueError("source_paths and transforms must have the same length")
    if crop_bounds_by_cloud is not None and len(crop_bounds_by_cloud) != len(source_paths):
        raise ValueError("crop_bounds_by_cloud must match source_paths length")
    if not source_paths:
        raise ValueError("At least one source point cloud is required")
    if sor_neighbors < 1 or sor_sigma <= 0.0:
        raise ValueError("SOR settings must be positive")

    transformed_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(int(max_workers), len(source_paths)))
    effective_crop_bounds = (
        list(crop_bounds_by_cloud)
        if crop_bounds_by_cloud is not None
        else [crop_bounds] * len(source_paths)
    )

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _transform_one_cloud,
                Path(path),
                transform,
                transformed_dir,
                matrix_dir,
                crop_bounds=effective_crop_bounds[index],
                skip_sor=skip_sor,
                sor_neighbors=sor_neighbors,
                sor_sigma=sor_sigma,
            ): index
            for index, (path, transform) in enumerate(zip(source_paths, transforms))
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()

    ordered = [results[index] for index in range(len(source_paths))]
    return [item[0] for item in ordered], [item[1] for item in ordered]


def quantized_unique_indices(points: np.ndarray, *, tolerance_m: float) -> np.ndarray:
    """Return stable representative indices using Trimesh row quantization."""
    _, trimesh = _geometry_libraries()
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be positive")

    digits = max(0, int(math.ceil(-math.log10(tolerance_m))))
    unique, _ = trimesh.grouping.unique_rows(points, digits=digits)
    return np.sort(np.asarray(unique, dtype=int))


def minimum_distance_sample_indices(
    points: np.ndarray,
    *,
    radius_m: float,
) -> np.ndarray:
    """Return a maximal subset separated by at least *radius_m*."""
    _, trimesh = _geometry_libraries()
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")

    _, selected_mask = trimesh.points.remove_close(points, radius_m)
    selected_mask = np.asarray(selected_mask, dtype=bool)

    # Trimesh guarantees separation but intentionally over-culls dense graphs.
    # Refill with points that are not close to anything already selected until
    # the set is maximal, preserving the distance guarantee and more detail.
    while True:
        remaining = np.flatnonzero(~selected_mask)
        if len(remaining) == 0:
            break
        selected = np.flatnonzero(selected_mask)
        distances, _ = cKDTree(points[selected]).query(
            points[remaining],
            workers=-1,
        )
        eligible = remaining[distances >= radius_m]
        if len(eligible) == 0:
            break
        _, refill_mask = trimesh.points.remove_close(points[eligible], radius_m)
        selected_mask[eligible[np.asarray(refill_mask, dtype=bool)]] = True

    return np.flatnonzero(selected_mask)


def _validate_final_artifact(o3d, trimesh, output_path: Path) -> dict:
    cloud = _read_colored_cloud(o3d, output_path)
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    if not cloud.has_normals() or len(normals) != len(points):
        raise RuntimeError(f"Final cloud has no complete normal data: {output_path}")
    if not np.isfinite(normals).all():
        raise RuntimeError(f"Final cloud contains non-finite normals: {output_path}")
    normal_lengths = np.linalg.norm(normals, axis=1)
    if not np.allclose(normal_lengths, 1.0, atol=1e-4):
        raise RuntimeError(f"Final cloud contains non-unit normals: {output_path}")

    trimesh_cloud = trimesh.load(output_path, process=False)
    if not isinstance(trimesh_cloud, trimesh.points.PointCloud):
        raise RuntimeError(f"Trimesh did not load a point cloud from {output_path}")
    if len(trimesh_cloud.vertices) != len(points):
        raise RuntimeError("Open3D and Trimesh disagree on final point count")
    if trimesh_cloud.colors is None or len(trimesh_cloud.colors) != len(points):
        raise RuntimeError("Trimesh did not preserve final point colors")
    if not np.isfinite(np.asarray(trimesh_cloud.vertices)).all():
        raise RuntimeError("Trimesh found non-finite final coordinates")

    return {
        "validated_points": len(points),
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
    }


def merge_and_finalize_clouds(
    transformed_paths: Sequence[Path],
    output_path: Path,
    *,
    pivot: np.ndarray,
    spatial_subsample_m: float,
    duplicate_distance_m: float,
    sor_neighbors: int,
    sor_sigma: float,
    normal_radius_m: float,
    normal_max_neighbors: int,
    normal_mst_neighbors: int,
) -> dict:
    """Merge transformed scans, clean them, orient normals, and validate PLY."""
    o3d, trimesh = _geometry_libraries()
    if not transformed_paths:
        raise ValueError("At least one transformed point cloud is required")
    if spatial_subsample_m <= 0.0 or duplicate_distance_m <= 0.0:
        raise ValueError("Subsample and duplicate distances must be positive")

    started = time.perf_counter()
    merged = o3d.geometry.PointCloud()
    input_points = 0
    for path in transformed_paths:
        cloud = _read_colored_cloud(o3d, Path(path))
        input_points += len(cloud.points)
        merged += cloud

    merged = merged.voxel_down_sample(spatial_subsample_m)
    voxel_points = len(merged.points)
    spatial_indices = minimum_distance_sample_indices(
        np.asarray(merged.points),
        radius_m=spatial_subsample_m,
    )
    merged = merged.select_by_index(spatial_indices.tolist())
    spatially_separated_points = len(merged.points)
    unique_indices = quantized_unique_indices(
        np.asarray(merged.points),
        tolerance_m=duplicate_distance_m,
    )
    merged = merged.select_by_index(unique_indices.tolist())
    deduplicated_points = len(merged.points)

    if len(merged.points) > sor_neighbors:
        merged, _ = merged.remove_statistical_outlier(
            nb_neighbors=sor_neighbors,
            std_ratio=sor_sigma,
        )
    if len(merged.points) < 3:
        raise RuntimeError("Final cleanup left too few points to estimate normals")
    filtered_points = len(merged.points)

    merged.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_m,
            max_nn=normal_max_neighbors,
        )
    )
    orientation_neighbors = min(normal_mst_neighbors, len(merged.points) - 1)
    if orientation_neighbors >= 2:
        merged.orient_normals_consistent_tangent_plane(orientation_neighbors)

    points = np.asarray(merged.points)
    normals = np.asarray(merged.normals)
    center = np.asarray(pivot, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("pivot must contain three finite values")
    outward_scores = np.einsum("ij,ij->i", normals, points - center)
    if float(np.median(outward_scores)) < 0.0:
        normals *= -1.0

    _write_cloud(o3d, output_path, merged)
    validation = _validate_final_artifact(o3d, trimesh, output_path)
    stats = {
        "processing_backend": "open3d+trimesh",
        "open3d_version": o3d.__version__,
        "trimesh_version": trimesh.__version__,
        "input_clouds": len(transformed_paths),
        "input_points": input_points,
        "voxel_points": voxel_points,
        "spatially_separated_points": spatially_separated_points,
        "deduplicated_points": deduplicated_points,
        "filtered_points": filtered_points,
        "elapsed_seconds": time.perf_counter() - started,
        **validation,
    }
    logger.info(
        "Merged %d clouds: %d -> %d validated points",
        len(transformed_paths),
        input_points,
        validation["validated_points"],
    )
    return stats
