#!/usr/bin/env python3
"""Convert a point cloud (.ply) to a smooth, clean 3D triangle mesh (.stl).

Features:
  - Statistical Outlier Removal (SOR) to eliminate flying noisy points.
  - Voxel grid regularization to equalize point density across overlapping scans.
  - Large-radius normal estimation + MST orientation to prevent opposing normal ripples.
  - Screened Poisson reconstruction tuned for organic/smooth biological geometries.
  - Taubin surface smoothing (eliminates high-frequency surface ripples without volume shrinkage).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import open3d as o3d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ply_to_stl")


def clean_and_orient_point_cloud(
    pcd: o3d.geometry.PointCloud,
    *,
    voxel_size_m: float = 0.003,
    sor_neighbors: int = 30,
    sor_std_ratio: float = 1.2,
    normal_radius_m: float = 0.020,
    normal_max_nn: int = 60,
    mst_neighbors: int = 30,
) -> o3d.geometry.PointCloud:
    """Filter noise, regularize point cloud, and compute clean consistent normals."""
    initial_count = len(pcd.points)

    # 1. Statistical Outlier Removal
    if sor_neighbors > 0 and len(pcd.points) > sor_neighbors:
        logger.info("Running Statistical Outlier Removal (k=%d, std=%.2f)...", sor_neighbors, sor_std_ratio)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
        logger.info("Points after SOR: %d (removed %d)", len(pcd.points), initial_count - len(pcd.points))

    # 2. Voxel Regularization
    if voxel_size_m > 0:
        logger.info("Regularizing point spacing with voxel size %.4fm...", voxel_size_m)
        pcd = pcd.voxel_down_sample(voxel_size_m)
        logger.info("Points after voxel regularization: %d", len(pcd.points))

    # 3. Robust Normal Estimation with wider neighborhood
    logger.info("Computing smooth surface normals (radius=%.4fm, max_nn=%d)...", normal_radius_m, normal_max_nn)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_m,
            max_nn=normal_max_nn,
        )
    )

    # 4. Consistent Normal Orientation
    logger.info("Orienting normals consistently (MST k=%d)...", mst_neighbors)
    pcd.orient_normals_consistent_tangent_plane(min(mst_neighbors, max(3, len(pcd.points) - 1)))

    return pcd


def convert_ply_to_stl(
    input_path: Path,
    output_path: Path,
    *,
    depth: int = 7,
    trim_quantile: float = 0.05,
    voxel_size_m: float = 0.003,
    sor_neighbors: int = 30,
    sor_std_ratio: float = 1.2,
    normal_radius_m: float = 0.020,
    smooth_type: str = "taubin",
    smooth_iterations: int = 20,
) -> Path:
    """Load a PLY point cloud and export a smooth, clean STL mesh."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    start_time = time.perf_counter()
    logger.info("Loading point cloud from: %s", input_path)
    pcd = o3d.io.read_point_cloud(str(input_path))

    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {input_path}")

    # Prepare cleaned cloud with consistent normals
    pcd_clean = clean_and_orient_point_cloud(
        pcd,
        voxel_size_m=voxel_size_m,
        sor_neighbors=sor_neighbors,
        sor_std_ratio=sor_std_ratio,
        normal_radius_m=normal_radius_m,
    )

    # Reconstruction
    logger.info("Reconstructing surface using Screened Poisson (depth=%d)...", depth)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean,
        depth=depth,
        linear_fit=True,
    )

    # Density trimming to remove peripheral ghost bubbles
    if trim_quantile > 0:
        densities_arr = np.asarray(densities)
        if len(densities_arr) > 0:
            threshold = float(np.quantile(densities_arr, trim_quantile))
            logger.info("Trimming low-density vertices (threshold=%.4f)...", threshold)
            mesh.remove_vertices_by_mask(densities_arr < threshold)

    # Crop to point cloud bounding box
    bbox = pcd_clean.get_axis_aligned_bounding_box()
    bbox.scale(1.02, bbox.get_center())
    mesh = mesh.crop(bbox)

    # Mesh cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # Smoothing (Taubin or Laplacian)
    if smooth_iterations > 0:
        if smooth_type.lower() == "taubin":
            logger.info("Applying %d iterations of Taubin smoothing (non-shrinking)...", smooth_iterations)
            mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        else:
            logger.info("Applying %d iterations of Laplacian smoothing...", smooth_iterations)
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # Output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving STL mesh to: %s", output_path)
    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_ascii=False,
        compressed=False,
    )
    if not success:
        raise RuntimeError(f"Failed to write STL to {output_path}")

    elapsed = time.perf_counter() - start_time
    extent_mm = (mesh.get_max_bound() - mesh.get_min_bound()) * 1000.0

    logger.info("--- Reconstruction Complete in %.2fs ---", elapsed)
    logger.info("Output STL: %s", output_path)
    logger.info("Vertices: %d | Triangles: %d", len(mesh.vertices), len(mesh.triangles))
    logger.info("Bounding Box (mm): X=%.1f, Y=%.1f, Z=%.1f", extent_mm[0], extent_mm[1], extent_mm[2])

    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a point cloud (.ply) into a clean, smooth 3D printable STL mesh."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input .ply point cloud file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to output .stl file (defaults to same name with .stl extension)",
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=7,
        help="Poisson octree depth (7=smooth organic hand, 8=moderate detail, 9=sharp/noisy; default: 7)",
    )
    parser.add_argument(
        "--trim-quantile", "-t",
        type=float,
        default=0.05,
        help="Quantile threshold for density trimming (default: 0.05)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.003,
        help="Voxel downsampling size in metres (default: 0.003 = 3mm)",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.020,
        help="Search radius in metres for smooth normal estimation (default: 0.020 = 20mm)",
    )
    parser.add_argument(
        "--sor-neighbors",
        type=int,
        default=30,
        help="Number of neighbors for Statistical Outlier Removal (default: 30)",
    )
    parser.add_argument(
        "--smooth-type",
        choices=["taubin", "laplacian"],
        default="taubin",
        help="Smoothing algorithm: 'taubin' (preserves volume while removing ripples) or 'laplacian' (default: taubin)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Number of smoothing iterations (default: 25)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = args.output
    if output_path is None:
        output_path = args.input.with_suffix(".stl")

    convert_ply_to_stl(
        input_path=args.input,
        output_path=output_path,
        depth=args.depth,
        trim_quantile=args.trim_quantile,
        voxel_size_m=args.voxel_size,
        sor_neighbors=args.sor_neighbors,
        normal_radius_m=args.normal_radius,
        smooth_type=args.smooth_type,
        smooth_iterations=args.smooth,
    )


if __name__ == "__main__":
    main()
