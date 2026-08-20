#!/usr/bin/env python3
"""Main orchestrator for 360-degree motor-controlled scanning.

Captures aligned RGB-D frames at each motor angle, saves per-angle PLY files,
and optionally invokes reconstruct_pipeline.py for CloudCompare-based
registration and merging.
"""

import argparse
import json
import logging
import subprocess
import sys
import time

import numpy as np
from pathlib import Path

from motor_controller import MotorController
from camera_controller import CameraController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="360 Motor-Controlled Scanner")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Motor controller serial port")
    parser.add_argument("--baud", type=int, default=250000, help="Baud rate")
    parser.add_argument("--step-deg", type=float, default=10.0, help="Degrees to step per capture")
    parser.add_argument("--disparity", type=str, default="256", choices=["128", "256"], help="Disparity search range")
    parser.add_argument("--width", type=int, default=848, help="Camera width resolution")
    parser.add_argument("--height", type=int, default=530, help="Camera height resolution")
    parser.add_argument("--fps", type=int, default=30, help="Camera framerate")
    parser.add_argument("--radius-m", type=float, default=None,
                        help="Radius from camera optical center to orbit center in metres "
                             "(required with --reconstruct)")
    parser.add_argument("--orbit-axis", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="Orbit axis in camera coordinates (default: 1 0 0)")
    parser.add_argument("--registration-mode", choices=("motor", "guarded-icp"),
                        default="motor", help="Reconstruction pose source (default: motor)")
    parser.add_argument("--frames-per-angle", type=int, default=5,
                        help="Fresh RGB-D frames to median-combine at each angle")
    parser.add_argument("--depth-min-m", type=float, default=0.02,
                        help="Discard depth closer than this distance")
    parser.add_argument("--depth-max-m", type=float, default=0.35,
                        help="Discard depth farther than this distance")
    parser.add_argument("--crop-radius-m", type=float, default=0.15,
                        help="Reconstruction crop-cube half-extent around the orbit center")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scan"), help="Output directory for PLY files")
    parser.add_argument("--dry-run", action="store_true", help="Print sequence without moving or capturing")
    parser.add_argument("--no-home", action="store_true", help="Skip the homing sequence (use only if already homed)")
    parser.add_argument("--reconstruct", action="store_true", help="Auto-run reconstruct_pipeline.py after capture")

    return parser.parse_args(argv)


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


def fuse_depth_frames(depth_frames, *, min_depth_m, max_depth_m):
    """Median-combine valid depth samples and mask points outside the work area."""
    if not depth_frames:
        raise ValueError("At least one depth frame is required.")
    if min_depth_m < 0.0 or max_depth_m <= min_depth_m:
        raise ValueError("Depth range must satisfy 0 <= min_depth_m < max_depth_m.")

    shapes = {np.asarray(frame).shape for frame in depth_frames}
    if len(shapes) != 1:
        raise ValueError("All depth frames must have the same shape.")

    stack = np.stack(depth_frames).astype(np.float32, copy=False)
    valid = (
        np.isfinite(stack)
        & (stack >= min_depth_m)
        & (stack <= max_depth_m)
    )
    masked = np.ma.array(stack, mask=~valid)
    fused = np.ma.median(masked, axis=0).filled(0.0).astype(np.float32)
    return fused


def capture_fused_rgbd(
    camera,
    *,
    frames_per_angle,
    timeout_ms,
    min_depth_m,
    max_depth_m,
):
    """Capture a fresh burst and return the latest color plus median depth."""
    if frames_per_angle < 1:
        raise ValueError("frames_per_angle must be at least 1.")

    depth_frames = []
    latest_color = None
    for _ in range(frames_per_angle):
        color_frame, depth_frame = camera.capture_aligned_rgbd(timeout_ms=timeout_ms)
        if color_frame is None or depth_frame is None:
            continue
        latest_color = color_frame_to_rgb(color_frame).copy()
        depth_frames.append(depth_frame_to_meters(depth_frame))

    if latest_color is None or not depth_frames:
        return None, None, 0

    fused_depth = fuse_depth_frames(
        depth_frames,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    return latest_color, fused_depth, len(depth_frames)


def build_scan_metadata(args, *, active_disparity, captured_angles):
    """Create the portable capture contract consumed by reconstruction."""
    return {
        "schema_version": 1,
        "orbit_radius_m": args.radius_m,
        "orbit_axis": [float(value) for value in args.orbit_axis],
        "step_deg": float(args.step_deg),
        "registration_mode": args.registration_mode,
        "capture": {
            "width": int(args.width),
            "height": int(args.height),
            "fps": int(args.fps),
            "disparity": int(active_disparity),
            "frames_per_angle": int(args.frames_per_angle),
            "depth_range_m": [float(args.depth_min_m), float(args.depth_max_m)],
        },
        "reconstruction": {
            "crop_radius_m": float(args.crop_radius_m),
        },
        "captured_angles_deg": [float(value) for value in captured_angles],
    }


def save_frame_as_ply(
    color_rgb,
    depth_m,
    intrinsics,
    angle_deg,
    output_dir,
    *,
    depth_trunc_m=3.0,
):
    """Create a colored point cloud from RGBD and save as PLY."""
    import open3d as o3d

    color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image((depth_m * 1000.0).astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,
        depth_trunc=depth_trunc_m,
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

    if args.frames_per_angle < 1:
        raise ValueError("--frames-per-angle must be at least 1.")
    if args.depth_min_m < 0.0 or args.depth_max_m <= args.depth_min_m:
        raise ValueError("Depth range must satisfy 0 <= min < max.")
    if args.radius_m is not None and args.radius_m <= 0.0:
        raise ValueError("--radius-m must be positive.")
    if args.reconstruct and args.radius_m is None:
        raise ValueError("--radius-m is required when --reconstruct is enabled.")

    # Ensure output dir exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    angle_sequence = generate_angle_sequence(args.step_deg)

    if args.dry_run:
        logger.info("[DRY RUN] Would home motors here.")
        for step in angle_sequence:
            angle = step["angle"]
            logger.info("[DRY RUN] Would move Y to %.1f degrees.", angle)
            if step["capture"]:
                logger.info("[DRY RUN] Would capture frame at %.1f degrees.", angle)
            else:
                logger.info("[DRY RUN] Would return without capturing.")
        logger.info("[DRY RUN] Sequence complete.")
        return

    logger.info("Initializing hardware...")

    saved_files = []
    captured_angles = []
    metadata_path = args.output_dir / "scan_metadata.json"

    with MotorController(port=args.port, baud=args.baud) as motor, \
         CameraController(width=args.width, height=args.height, fps=args.fps, disparity=args.disparity) as camera:

        if not args.no_home:
            motor.home_all()

        # Save intrinsics once
        if camera.intrinsics:
            intrinsics_path = args.output_dir / "intrinsics.json"
            intrinsics_path.write_text(
                json.dumps(camera.intrinsics, indent=2), encoding="utf-8",
            )
            logger.info("Saved intrinsics to %s", intrinsics_path)

        metadata_path.write_text(
            json.dumps(
                build_scan_metadata(
                    args,
                    active_disparity=camera.active_disparity,
                    captured_angles=captured_angles,
                ),
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved scan settings to %s", metadata_path)

        logger.info("Starting scan with %d steps.", len(angle_sequence))

        for step in angle_sequence:
            angle = step["angle"]
            should_capture = step["capture"]

            logger.info("Moving Y to %.1f degrees...", angle)
            motor.move_y(angle, feedrate=500)

            # Brief pause to let camera settle after movement
            time.sleep(0.5)

            if should_capture:
                logger.info("Capturing frame at %.1f degrees...", angle)
                color_rgb, depth_m, captured_count = capture_fused_rgbd(
                    camera,
                    frames_per_angle=args.frames_per_angle,
                    timeout_ms=2000,
                    min_depth_m=args.depth_min_m,
                    max_depth_m=args.depth_max_m,
                )

                if color_rgb is not None and depth_m is not None:
                    valid_percent = 100.0 * np.count_nonzero(depth_m) / depth_m.size
                    logger.info(
                        "Combined %d/%d frames; %.1f%% valid depth in %.3f-%.3f m",
                        captured_count,
                        args.frames_per_angle,
                        valid_percent,
                        args.depth_min_m,
                        args.depth_max_m,
                    )
                    ply_path = save_frame_as_ply(
                        color_rgb,
                        depth_m,
                        camera.intrinsics,
                        angle,
                        args.output_dir,
                        depth_trunc_m=args.depth_max_m,
                    )
                    saved_files.append(ply_path)
                    captured_angles.append(float(angle))
                    metadata_path.write_text(
                        json.dumps(
                            build_scan_metadata(
                                args,
                                active_disparity=camera.active_disparity,
                                captured_angles=captured_angles,
                            ),
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                    )
                else:
                    logger.error(
                        "Failed to capture any complete frame at %.1f degrees. Skipping.",
                        angle,
                    )
            else:
                logger.info("Returning to %.1f degrees (no capture).", angle)

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
            "--registration-mode", args.registration_mode,
            "--orbit-axis", *(str(value) for value in args.orbit_axis),
            "--crop-radius-m", str(args.crop_radius_m),
        ]

        logger.info("$ %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    elif len(saved_files) > 0:
        if args.radius_m is None:
            logger.info(
                "Capture has no calibrated orbit radius. Reconstruct with:\n"
                "  python %s/reconstruct_pipeline.py --input-dir %s "
                "--orbit-radius-m <CALIBRATED_METRES>",
                Path(__file__).parent,
                args.output_dir,
            )
        else:
            logger.info(
                "To reconstruct, run:\n"
                "  python %s/reconstruct_pipeline.py --input-dir %s",
                Path(__file__).parent,
                args.output_dir,
            )
    else:
        logger.warning("No frames were captured.")


if __name__ == "__main__":
    main()
