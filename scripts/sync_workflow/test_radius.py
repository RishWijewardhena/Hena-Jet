#!/usr/bin/env python3
"""Targeted script to test camera capture perspective at X=400, Y=0 and Y=180."""

import argparse
import logging
from pathlib import Path
import time
import numpy as np

# We can reuse the existing helper functions from main_scan
from main_scan import color_frame_to_rgb, depth_frame_to_meters, save_frame_as_ply
from motor_controller import MotorController
from camera_controller import CameraController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test Radius Scanner (-55 and 125 degrees at X=400)")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Motor controller serial port")
    parser.add_argument("--baud", type=int, default=250000, help="Baud rate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_radius"), help="Output directory")
    parser.add_argument("--x-pos", type=float, default=400.0, help="X position (radius offset)")
    return parser.parse_args()

def capture_and_save(camera, angle, output_dir):
    logger.info("Capturing frame at %.1f degrees...", angle)
    # The camera buffer flush is already built into camera.capture_aligned_rgbd()
    color_frame, depth_frame = camera.capture_aligned_rgbd(timeout_ms=2000)

    if color_frame and depth_frame:
        color_rgb = color_frame_to_rgb(color_frame)
        depth_m = depth_frame_to_meters(depth_frame)
        
        # Save point cloud (.ply) using the existing helper
        ply_path = save_frame_as_ply(color_rgb, depth_m, camera.intrinsics, angle, output_dir)
        
        # Save a 2D image (.png) for quick visual inspection
        try:
            import cv2
            # Normalize depth to 8-bit for visualization
            depth_vis = np.clip(depth_m / 0.5 * 255.0, 0, 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Convert RGB to BGR for OpenCV
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            
            # Stack horizontally for easy viewing
            vis = np.hstack((color_bgr, depth_colormap))
            img_path = output_dir / f"frame_{angle:.1f}_visual.png"
            cv2.imwrite(str(img_path), vis)
            logger.info("Saved visual representation to %s", img_path.name)
        except ImportError:
            logger.warning("OpenCV not installed, skipping 2D image save.")
            
    else:
        logger.error("Failed to capture frame at %.1f degrees.", angle)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing hardware...")

    with MotorController(port=args.port, baud=args.baud) as motor, \
         CameraController() as camera:

        logger.info("Homing motors...")
        motor.home_all()

        logger.info("Moving X to %.1f (M400 blocking move)...", args.x_pos)
        motor.move_x(args.x_pos, feedrate=1000)

        # ---------------------------------------------------------
        # Capture at Y=-55
        # ---------------------------------------------------------
        angle = -55.0
        logger.info("Moving Y to %.1f...", angle)
        motor.move_y(angle, feedrate=500)
        time.sleep(0.5) # Let the rig settle
        capture_and_save(camera, angle, args.output_dir)

        # ---------------------------------------------------------
        # Capture at Y=125
        # ---------------------------------------------------------
        angle = 125.0
        logger.info("Moving Y to %.1f...", angle)
        motor.move_y(angle, feedrate=500)
        time.sleep(0.5) # Let the rig settle
        capture_and_save(camera, angle, args.output_dir)

    logger.info("Test complete. Check the %s folder.", args.output_dir)

if __name__ == "__main__":
    main()
