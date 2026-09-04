#!/usr/bin/env python3
"""Measure reconstruction accuracy against a certified two-sphere ball bar.

Every other metric in this workflow is internal consistency: plane-RMS,
cross-view residual and ICP fitness all describe how well the pipeline agrees
with itself. A pipeline can be perfectly self-consistent around a wrong orbit
radius, so none of them can detect a scale error.

A ball bar breaks that circularity. Sphere centres are recoverable far more
accurately than the point noise itself, because thousands of points are fitted
to one known radius, so a 1.5 mm-noise sensor still resolves a centre to well
under a millimetre. The centre-to-centre distance is then a direct, traceable
length measurement: a proportional error in it is a proportional error in the
orbit radius.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ball-bar accuracy report")
    parser.add_argument("--merged", type=Path, required=True,
                        help="Merged cloud containing both spheres")
    parser.add_argument("--certified-distance-mm", type=float, required=True,
                        help="Certified centre-to-centre distance of the ball bar")
    parser.add_argument("--sphere-diameter-mm", type=float, required=True,
                        help="Certified sphere diameter")
    parser.add_argument("--diameter-tolerance-mm", type=float, default=8.0,
                        help="Half-width of the accepted fitted-diameter band")
    parser.add_argument("--min-sphere-points", type=int, default=200,
                        help="Reject a cluster with fewer points than this")
    parser.add_argument("--cluster-eps-mm", type=float, default=4.0,
                        help="DBSCAN neighbourhood radius used to separate the spheres")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the report as JSON to this path")
    return parser.parse_args(argv)


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares sphere fit; returns centre, radius, and per-point residuals.

    Solves the linear form 2x*cx + 2y*cy + 2z*cz + t = x^2+y^2+z^2, where
    t = r^2 - |c|^2, which has a closed-form solution and needs no initial
    guess. Residuals are signed distances from the fitted surface, so their
    RMS is the sphere's form error.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Sphere fitting needs an (N, 3) array of points")
    if len(points) < 4:
        raise ValueError("Sphere fitting needs at least four points")

    design = np.hstack([2.0 * points, np.ones((len(points), 1))])
    target = np.sum(points**2, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    centre = solution[:3]
    squared_radius = solution[3] + float(centre @ centre)
    if not np.isfinite(squared_radius) or squared_radius <= 0.0:
        raise ValueError("Sphere fitting produced a non-physical radius")
    radius = float(np.sqrt(squared_radius))
    residuals = np.linalg.norm(points - centre, axis=1) - radius
    return centre, radius, residuals


def largest_clusters(points: np.ndarray, *, eps_m: float, min_points: int, count: int = 2):
    """Return the `count` largest DBSCAN clusters, biggest first."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(cloud.cluster_dbscan(eps=eps_m, min_points=10))
    clusters = []
    for label in range(labels.max() + 1):
        member = points[labels == label]
        if len(member) >= min_points:
            clusters.append(member)
    clusters.sort(key=len, reverse=True)
    return clusters[:count]


def build_report(
    points: np.ndarray,
    *,
    certified_distance_m: float,
    sphere_diameter_m: float,
    diameter_tolerance_m: float,
    min_sphere_points: int,
    cluster_eps_m: float,
) -> dict:
    clusters = largest_clusters(
        points, eps_m=cluster_eps_m, min_points=min_sphere_points,
    )
    if len(clusters) < 2:
        raise RuntimeError(
            f"Found {len(clusters)} sphere candidates, need 2. Crop the cloud to "
            "the ball bar, or relax --cluster-eps-mm / --min-sphere-points."
        )

    spheres = []
    expected_radius_m = sphere_diameter_m / 2.0
    for index, cluster in enumerate(clusters):
        centre, radius, residuals = fit_sphere(cluster)
        diameter_error_m = 2.0 * radius - sphere_diameter_m
        if abs(diameter_error_m) > diameter_tolerance_m:
            raise RuntimeError(
                f"Sphere {index} fitted to a {2000.0 * radius:.3f} mm diameter, but "
                f"the ball bar is {1000.0 * sphere_diameter_m:.3f} mm. That cluster "
                "is probably not a sphere; crop the cloud more tightly."
            )
        spheres.append({
            "point_count": int(len(cluster)),
            "centre_m": centre.tolist(),
            "fitted_radius_m": radius,
            "fitted_diameter_m": 2.0 * radius,
            "diameter_error_m": diameter_error_m,
            "form_error_rms_m": float(np.sqrt(np.mean(residuals**2))),
            "form_error_max_m": float(np.max(np.abs(residuals))),
        })

    separation_m = float(
        np.linalg.norm(
            np.asarray(spheres[0]["centre_m"]) - np.asarray(spheres[1]["centre_m"])
        )
    )
    distance_error_m = separation_m - certified_distance_m
    return {
        "certified_distance_m": certified_distance_m,
        "measured_distance_m": separation_m,
        "distance_error_m": distance_error_m,
        "scale_error_ratio": distance_error_m / certified_distance_m,
        "implied_radius_correction_ratio": 1.0 - distance_error_m / certified_distance_m,
        "expected_sphere_radius_m": expected_radius_m,
        "spheres": spheres,
    }


def log_report(report: dict) -> None:
    logger.info(
        "Ball bar: certified %.3f mm, measured %.3f mm, error %+.3f mm (%+.3f%%)",
        report["certified_distance_m"] * 1000.0,
        report["measured_distance_m"] * 1000.0,
        report["distance_error_m"] * 1000.0,
        report["scale_error_ratio"] * 100.0,
    )
    for index, sphere in enumerate(report["spheres"]):
        logger.info(
            "  sphere %d: %d points, diameter %+.3f mm vs certified, "
            "form error %.3f mm RMS / %.3f mm max",
            index,
            sphere["point_count"],
            sphere["diameter_error_m"] * 1000.0,
            sphere["form_error_rms_m"] * 1000.0,
            sphere["form_error_max_m"] * 1000.0,
        )
    logger.info(
        "Scale error implies the orbit radius should be scaled by %.6f",
        report["implied_radius_correction_ratio"],
    )


def main():
    args = parse_args()
    cloud = o3d.io.read_point_cloud(str(args.merged))
    if cloud.is_empty():
        raise RuntimeError(f"Open3D could not read points from {args.merged}")
    points = np.asarray(cloud.points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]

    report = build_report(
        points,
        certified_distance_m=args.certified_distance_mm / 1000.0,
        sphere_diameter_m=args.sphere_diameter_mm / 1000.0,
        diameter_tolerance_m=args.diameter_tolerance_mm / 1000.0,
        min_sphere_points=args.min_sphere_points,
        cluster_eps_m=args.cluster_eps_mm / 1000.0,
    )
    report["merged_cloud"] = str(args.merged)
    report["input_points"] = int(len(points))
    log_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
