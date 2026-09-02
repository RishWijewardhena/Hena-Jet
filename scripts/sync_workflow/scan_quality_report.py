#!/usr/bin/env python3
"""Report capture/reconstruction quality metrics for one scan directory."""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
import re

import numpy as np
import open3d as o3d

from depth_quality import cross_view_residual_m, surface_plane_rms_m

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANGLE_PATTERN = re.compile(r"y([+-]\d+\.\d)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Scan quality metrics")
    parser.add_argument("--scan-dir", type=Path, required=True,
                        help="Capture directory containing frame_*.ply")
    parser.add_argument("--merged", type=Path, default=None,
                        help="Merged cloud (default: <scan-dir>/reconstruction/merged_cloud.ply)")
    parser.add_argument("--pivot", type=float, nargs=3, default=None,
                        help="Patch centre in metres; default reads orbit radius from scan_metadata.json")
    parser.add_argument("--patch-half-extent-m", type=float, default=0.020)
    parser.add_argument("--subsample", type=int, default=4,
                        help="Keep every Nth point when loading PLY files (default 4)")
    parser.add_argument("--separations-deg", type=float, nargs="+",
                        default=[10.0, 30.0, 90.0, 180.0])
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the metrics as JSON to this path")
    return parser.parse_args(argv)


def load_points(path: Path) -> np.ndarray:
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    return points[np.all(np.isfinite(points), axis=1)]


def patch_near(points: np.ndarray, centre: np.ndarray, half_extent_m: float) -> np.ndarray:
    offsets = np.abs(points - centre)
    inside = np.all(offsets < half_extent_m, axis=1)
    return points[inside]


def resolve_pivot(args) -> np.ndarray:
    if args.pivot is not None:
        return np.asarray(args.pivot, dtype=float)
    metadata_path = args.scan_dir / "scan_metadata.json"
    radius = json.loads(metadata_path.read_text(encoding="utf-8"))["orbit_radius_m"]
    return np.array([0.0, 0.0, float(radius)])


def main(argv=None):
    from scipy.spatial import cKDTree

    args = parse_args(argv)
    pivot = resolve_pivot(args)
    merged_path = args.merged or (args.scan_dir / "reconstruction" / "merged_cloud.ply")

    report: dict[str, object] = {
        "scan_dir": str(args.scan_dir),
        "pivot_m": pivot.tolist(),
        "subsample": args.subsample,
    }

    # Load all transformed clouds once, keyed by angle
    transformed = sorted(glob.glob(str(args.scan_dir / "reconstruction" / "01_transformed" / "*.ply")))
    clouds: dict[float, np.ndarray] = {}
    cloud_trees: dict[float, cKDTree] = {}

    for path in transformed:
        match = ANGLE_PATTERN.search(Path(path).name)
        if match:
            angle = float(match.group(1))
            points = load_points(Path(path))
            # Subsample
            points = points[::args.subsample]
            clouds[angle] = points
            cloud_trees[angle] = cKDTree(points)

    # Compute per-frame metrics from loaded clouds
    per_frame = []
    frame_count = 0
    for angle, points in clouds.items():
        patch = patch_near(points, pivot, args.patch_half_extent_m)
        frame_count += 1
        if len(patch) >= 50:
            per_frame.append(surface_plane_rms_m(patch))

    if per_frame:
        report["single_frame_plane_rms_m"] = float(np.median(per_frame))
        report["single_frame_frames_used"] = len(per_frame)
        logger.info("Single-frame surface plane-RMS (median, %d/%d frames): %.3f mm",
                    len(per_frame), frame_count, report["single_frame_plane_rms_m"] * 1000.0)

    # Compute merged metrics
    if merged_path.is_file():
        merged_points = load_points(merged_path)
        merged_points = merged_points[::args.subsample]
        patch = patch_near(merged_points, pivot, args.patch_half_extent_m)
        if len(patch) >= 50:
            report["merged_plane_rms_m"] = surface_plane_rms_m(patch)
            logger.info("Merged surface plane-RMS: %.3f mm",
                        report["merged_plane_rms_m"] * 1000.0)

    # Compute cross-view residuals reusing cached trees
    separations = {}
    for separation in args.separations_deg:
        values = []
        for angle, points in clouds.items():
            other_angle = angle + separation
            if other_angle not in clouds:
                continue
            other_tree = cloud_trees[other_angle]
            residual = cross_view_residual_m(points, None, max_pair_distance_m=0.008, tree_b=other_tree)
            if residual is not None:
                values.append(residual)
        if values:
            separations[str(separation)] = float(np.mean(values))
            logger.info("Cross-view residual at %5.1f deg: %.3f mm",
                        separation, separations[str(separation)] * 1000.0)
    report["cross_view_residual_m"] = separations

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)
    return report


if __name__ == "__main__":
    main()
