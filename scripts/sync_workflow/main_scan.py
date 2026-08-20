#!/usr/bin/env python3
"""Main orchestrator for 360-degree motor-controlled scanning.

Captures aligned RGB-D frames at each motor angle, saves per-angle PLY files,
and optionally invokes reconstruct_pipeline.py for CloudCompare-based
registration, merging, and meshing.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

from motor_controller import MotorController
from camera_controller import CameraController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="360 Motor-Controlled Scanner")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Motor controller serial port")
    parser.add_argument("--baud", type=int, default=250000, help="Baud rate")
    parser.add_argument("--step-deg", type=float, default=10.0, help="Degrees to step per capture")
    parser.add_argument("--disparity", type=str, default="256", choices=["128", "256"], help="Disparity search range")
    parser.add_argument("--width", type=int, default=848, help="Camera width resolution")
    parser.add_argument("--height", type=int, default=530, help="Camera height resolution")
    parser.add_argument("--fps", type=int, default=30, help="Camera framerate")
    parser.add_argument("--radius-m", type=float, default=0.12007, help="Radius from camera to object center")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scan"), help="Output directory for PLY files")
    parser.add_argument("--dry-run", action="store_true", help="Print sequence without moving or capturing")
    parser.add_argument("--no-home", action="store_true", help="Skip the homing sequence (use only if already homed)")
    parser.add_argument("--reconstruct", action="store_true", help="Auto-run reconstruct_pipeline.py after capture")

    return parser.parse_args()


def generate_angle_sequence(step_deg):
    """Generates the scan sequence: 0 to 180, then back to 0, then to -180."""
    sequence = []

    # 0 to 180
    pos_angles = np.arange(0.0, 180.0 + step_deg, step_deg)
    for ang in pos_angles:
        sequence.append({"angle": ang, "capture": True})

    # Return to 0 without capturing
    sequence.append({"angle": 0.0, "capture": False})

    # -step to -180
    neg_angles = np.arange(-step_deg, -180.0 - step_deg, -step_deg)
    for ang in neg_angles:
        sequence.append({"angle": ang, "capture": True})

    return sequence


def color_frame_to_rgb(color_frame):
    """Convert Orbbec color frame to numpy RGB array."""
    width = color_frame.get_width()
    height = color_frame.get_height()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape((height, width, 3))
    return data


def depth_frame_to_meters(depth_frame):
    """Convert Orbbec depth frame to meters as float32."""
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    depth_scale_mm = float(depth_frame.get_depth_scale())
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_m = depth_raw.astype(np.float32) * (depth_scale_mm / 1000.0)
    return depth_m


def save_frame_as_ply(color_rgb, depth_m, intrinsics, angle_deg, output_dir):
    """Create a colored point cloud from RGBD and save as PLY."""
    color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image((depth_m * 1000.0).astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics["width"],
        height=intrinsics["height"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)

    out_path = output_dir / f"frame_{angle_deg:.1f}.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    logger.info("Saved %d points -> %s", len(pcd.points), out_path.name)
    return out_path


def main():
    args = parse_args()

    # Ensure output dir exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    angle_sequence = generate_angle_sequence(args.step_deg)

    logger.info("Initializing hardware...")

    saved_files = []

    with MotorController(port=args.port, baud=args.baud) as motor, \
         CameraController(width=args.width, height=args.height, fps=args.fps, disparity=args.disparity) as camera:

        if not args.no_home and not args.dry_run:
            motor.home_all()
        elif args.dry_run:
            logger.info("[DRY RUN] Would home motors here.")

        # Save intrinsics once
        if not args.dry_run and camera.intrinsics:
            intrinsics_path = args.output_dir / "intrinsics.json"
            intrinsics_path.write_text(
                json.dumps(camera.intrinsics, indent=2), encoding="utf-8",
            )
            logger.info("Saved intrinsics to %s", intrinsics_path)

        logger.info("Starting scan with %d steps.", len(angle_sequence))

        for step in angle_sequence:
            angle = step["angle"]
            should_capture = step["capture"]

            logger.info("Moving Y to %.1f degrees...", angle)
            motor.move_y(angle, feedrate=500, dry_run=args.dry_run)

            # Brief pause to let camera settle after movement
            if not args.dry_run:
                time.sleep(0.5)

            if should_capture:
                if args.dry_run:
                    logger.info("[DRY RUN] Would capture frame at %.1f degrees.", angle)
                else:
                    logger.info("Capturing frame at %.1f degrees...", angle)
                    color_frame, depth_frame = camera.capture_aligned_rgbd(timeout_ms=2000)

                    if color_frame and depth_frame:
                        color_rgb = color_frame_to_rgb(color_frame)
                        depth_m = depth_frame_to_meters(depth_frame)
                        ply_path = save_frame_as_ply(
                            color_rgb, depth_m, camera.intrinsics, angle, args.output_dir,
                        )
                        saved_files.append(ply_path)
                    else:
                        logger.error("Failed to capture frame at %.1f degrees. Skipping.", angle)
            else:
                logger.info("Returning to %.1f degrees (no capture).", angle)

    if args.dry_run:
        logger.info("[DRY RUN] Sequence complete.")
        return

    logger.info("Capture complete. Saved %d point clouds to %s", len(saved_files), args.output_dir)

    # Optionally invoke reconstruction
    if args.reconstruct and len(saved_files) > 0:
        logger.info("Launching reconstruction pipeline...")
        script_dir = Path(__file__).parent
        cmd = [
            sys.executable,
            str(script_dir / "reconstruct_pipeline.py"),
            "--input-dir", str(args.output_dir),
            "--orbit-radius-m", str(args.radius_m),
        ]

        logger.info("$ %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    elif len(saved_files) > 0:
        logger.info(
            "To reconstruct, run:\n"
            "  python %s/reconstruct_pipeline.py --input-dir %s --orbit-radius-m %s",
            Path(__file__).parent, args.output_dir, args.radius_m,
        )
    else:
        logger.warning("No frames were captured.")


if __name__ == "__main__":
    main()
