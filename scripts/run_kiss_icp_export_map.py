#!/usr/bin/env python3
"""Run KISS-ICP on a point-cloud folder and export its internal local map."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.datasets.generic import GenericDataset
from kiss_icp.kiss_icp import KissICP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KISS-ICP frame-by-frame and save the voxel local map as a PLY."
    )
    parser.add_argument("data", type=Path, nargs="?", default=Path("captures/zed_m_tsdf_scan"))
    parser.add_argument("--out", type=Path, default=Path("outputs/kiss_icp_local_map.ply"))
    parser.add_argument("--poses-out", type=Path, default=Path("outputs/kiss_icp_local_map_poses.npy"))
    parser.add_argument("--min-range", type=float, default=0.10)
    parser.add_argument("--max-range", type=float, default=0.45)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--max-points-per-voxel", type=int, default=20)
    parser.add_argument("--initial-threshold", type=float, default=0.05)
    parser.add_argument("--min-motion-th", type=float, default=0.005)
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--n-scans", type=int, default=-1)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> KISSConfig:
    config = KISSConfig()
    config.data.min_range = args.min_range
    config.data.max_range = args.max_range
    config.data.deskew = args.deskew
    config.mapping.voxel_size = args.voxel_size
    config.mapping.max_points_per_voxel = args.max_points_per_voxel
    config.adaptive_threshold.initial_threshold = args.initial_threshold
    config.adaptive_threshold.min_motion_th = args.min_motion_th
    return config


def pose_step(previous: np.ndarray | None, current: np.ndarray) -> tuple[float, float]:
    if previous is None:
        return 0.0, 0.0

    delta = np.linalg.inv(previous) @ current
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cos_angle = (float(np.trace(delta[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    rotation_deg = math.degrees(math.acos(cos_angle))
    return translation_m, rotation_deg


def write_xyz_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in points:
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def main() -> None:
    args = parse_args()
    dataset = GenericDataset(args.data)
    config = make_config(args)
    kiss = KissICP(config)

    total_scans = len(dataset) if args.n_scans < 0 else min(args.n_scans, len(dataset))
    poses: list[np.ndarray] = []
    previous_pose: np.ndarray | None = None

    print(f"Processing {total_scans} scans from {args.data}")
    print(
        "KISS config: "
        f"range=[{config.data.min_range}, {config.data.max_range}], "
        f"voxel={config.mapping.voxel_size}, "
        f"deskew={config.data.deskew}, "
        f"threshold={config.adaptive_threshold.initial_threshold}"
    )

    for index in range(total_scans):
        points, timestamps = dataset[index]
        kiss.register_frame(points, timestamps)
        pose = kiss.last_pose.copy()
        poses.append(pose)

        step_m, step_deg = pose_step(previous_pose, pose)
        previous_pose = pose
        map_points = kiss.local_map.point_cloud()
        print(
            f"{index + 1:03d}/{total_scans}: "
            f"input={points.shape[0]} map={map_points.shape[0]} "
            f"step={step_m:.4f}m rot={step_deg:.2f}deg"
        )

    local_map = kiss.local_map.point_cloud()
    if local_map.size == 0:
        raise RuntimeError("KISS-ICP local map is empty; no PLY was written.")

    write_xyz_ply(args.out, local_map)
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, np.stack(poses, axis=0))

    print(f"Saved local map: {args.out} ({local_map.shape[0]} points)")
    print(f"Saved poses: {args.poses_out}")


if __name__ == "__main__":
    main()
