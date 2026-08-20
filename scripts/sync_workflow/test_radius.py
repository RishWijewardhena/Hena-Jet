#!/usr/bin/env python3
"""Capture a rigid target from multiple angles for orbit-radius calibration."""

import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np

from main_scan import capture_fused_rgbd, save_frame_as_ply
from motor_controller import MotorController
from camera_controller import CameraController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture an asymmetric rigid target around the camera orbit"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Motor controller serial port")
    parser.add_argument("--baud", type=int, default=250000, help="Baud rate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_radius"), help="Output directory")
    parser.add_argument("--x-pos", type=float, default=400.0, help="X position (radius offset)")
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0, -180.0],
        help="Motor angles used to observe the rigid calibration target",
    )
    parser.add_argument("--frames-per-angle", type=int, default=5)
    parser.add_argument("--depth-min-m", type=float, default=0.02)
    parser.add_argument("--depth-max-m", type=float, default=0.35)
    return parser.parse_args(argv)


def capture_and_save(camera, angle, args):
    logger.info("Capturing frame at %.1f degrees...", angle)
    color_rgb, depth_m, captured_count = capture_fused_rgbd(
        camera,
        frames_per_angle=args.frames_per_angle,
        timeout_ms=2000,
        min_depth_m=args.depth_min_m,
        max_depth_m=args.depth_max_m,
    )

    if color_rgb is not None and depth_m is not None:
        logger.info("Combined %d/%d frames.", captured_count, args.frames_per_angle)
        ply_path = save_frame_as_ply(
            color_rgb,
            depth_m,
            camera.intrinsics,
            angle,
            args.output_dir,
            depth_trunc_m=args.depth_max_m,
        )

        # Save a 2D image (.png) for quick visual inspection
        try:
            import cv2
            # Normalize depth to 8-bit for visualization
            depth_vis = np.clip(
                depth_m / args.depth_max_m * 255.0, 0, 255
            ).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Convert RGB to BGR for OpenCV
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            
            # Stack horizontally for easy viewing
            vis = np.hstack((color_bgr, depth_colormap))
            img_path = args.output_dir / f"frame_{angle:.1f}_visual.png"
            cv2.imwrite(str(img_path), vis)
            logger.info("Saved visual representation to %s", img_path.name)
        except ImportError:
            logger.warning("OpenCV not installed, skipping 2D image save.")
        return ply_path
    else:
        logger.error("Failed to capture frame at %.1f degrees.", angle)
        return None

def main():
    args = parse_args()
    if args.frames_per_angle < 1:
        raise ValueError("--frames-per-angle must be at least 1.")
    if args.depth_min_m < 0.0 or args.depth_max_m <= args.depth_min_m:
        raise ValueError("Depth range must satisfy 0 <= min < max.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing hardware...")

    with MotorController(port=args.port, baud=args.baud) as motor, \
         CameraController() as camera:

        logger.info("Homing motors...")
        motor.home_all()

        logger.info("Moving X to %.1f (M400 blocking move)...", args.x_pos)
        motor.move_x(args.x_pos, feedrate=1000)

        saved_angles = []
        previous_angle = None
        for angle in args.angles:
            if previous_angle is not None and previous_angle > 0.0 > angle:
                logger.info("Returning Y to 0.0 before the negative sweep...")
                motor.move_y(0.0, feedrate=500)
                time.sleep(0.5)
            logger.info("Moving Y to %.1f...", angle)
            motor.move_y(angle, feedrate=500)
            time.sleep(0.5)
            if capture_and_save(camera, angle, args) is not None:
                saved_angles.append(float(angle))
            previous_angle = angle

    manifest = {
        "purpose": "orbit-radius calibration capture",
        "target_requirement": "rigid asymmetric target fixed at the orbit center",
        "x_position": args.x_pos,
        "angles_deg": saved_angles,
        "frames_per_angle": args.frames_per_angle,
        "depth_range_m": [args.depth_min_m, args.depth_max_m],
        "note": (
            "Fit camera poses/circle from this rigid target. Do not treat the "
            "nearest hand-surface depth as the orbit radius."
        ),
    }
    (args.output_dir / "calibration_capture.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Test complete. Check the %s folder.", args.output_dir)

if __name__ == "__main__":
    main()
