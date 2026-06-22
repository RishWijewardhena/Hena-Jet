#!/usr/bin/env python3
"""Run a small TSDF calibration sweep over radius, yaw, and center offsets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse options for generating many TSDF calibration meshes.

    --capture-dir: RGB-D capture folder to fuse repeatedly.
    --out-dir: Folder where calibration meshes are written.
    --radius-m: Radius value to hold fixed during the sweep.
    --yaw-values: Camera yaw corrections in degrees to test.
    --center-offset-values-m: XY center offsets in meters to test one axis at a
        time. The script creates x-only and y-only offset runs.
    """
    parser = argparse.ArgumentParser(description="Sweep TSDF calibration parameters.")
    parser.add_argument("--capture-dir", type=Path, default=Path("captures/zed_m_tsdf_scan"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tsdf_calibration"))
    parser.add_argument("--radius-m", type=float, default=0.24)
    parser.add_argument(
        "--yaw-values",
        type=float,
        nargs="+",
        default=[-20.0, -10.0, -5.0, 5.0, 10.0, 20.0],
    )
    parser.add_argument(
        "--center-offset-values-m",
        type=float,
        nargs="+",
        default=[-0.03, -0.02, -0.01, 0.01, 0.02, 0.03],
    )
    return parser.parse_args()


def output_stem(value: float) -> str:
    """Format a signed numeric value into a filename-safe token."""
    sign = "p" if value >= 0 else "m"
    return f"{sign}{abs(value):.3f}".replace(".", "p")


def run_fusion(command: list[str]) -> None:
    """Run one fusion command and fail immediately if Open3D reports an error."""
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    """Generate meshes for yaw and center-offset calibration comparisons."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_command = [
        sys.executable,
        "scripts/fuse_tsdf_scan.py",
        "--capture-dir",
        str(args.capture_dir),
        "--voxel-length-m",
        "0.001",
        "--sdf-trunc-m",
        "0.005",
        "--max-depth-over-radius-m",
        "0.03",
        "--min-world-z",
        "0.02",
        "--max-world-z",
        "0.16",
        "--max-world-radius-m",
        "0.09",
        "--override-radius-m",
        str(args.radius_m),
    ]

    for yaw in args.yaw_values:
        stem = output_stem(yaw)
        run_fusion(
            base_command
            + [
                "--mesh-out",
                str(args.out_dir / f"radius_{args.radius_m:.3f}_yaw_{stem}.ply"),
                "--camera-yaw-deg",
                str(yaw),
            ]
        )

    for offset in args.center_offset_values_m:
        stem = output_stem(offset)
        run_fusion(
            base_command
            + [
                "--mesh-out",
                str(args.out_dir / f"radius_{args.radius_m:.3f}_center_x_{stem}.ply"),
                "--center-offset-x-m",
                str(offset),
            ]
        )
        run_fusion(
            base_command
            + [
                "--mesh-out",
                str(args.out_dir / f"radius_{args.radius_m:.3f}_center_y_{stem}.ply"),
                "--center-offset-y-m",
                str(offset),
            ]
        )


if __name__ == "__main__":
    main()
