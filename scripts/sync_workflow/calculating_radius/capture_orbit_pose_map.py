#!/usr/bin/env python3
"""Measure a full aligned-pointcloud pose at each angle using a fixed ArUco bar."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Running this file directly puts only ``calculating_radius`` on sys.path.
# Add the sync_workflow directory so its camera, motor, and scan modules remain
# importable from the documented repository-root command.
SYNC_WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(SYNC_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from camera_controller import CameraController
from calculating_radius.orbit_pose_map import (
    aggregate_world_to_camera_poses,
    build_orbit_pose_map,
)
from main_scan import capture_filename, generate_scan_sequence, save_frame_as_ply
from motor_controller import MotorController
from calculating_radius.radius_calibration import orbbec_extrinsic_to_matrix
from calculating_radius.test_radius import (
    DEFAULT_MARKER_MAP,
    _draw_detection_overlay,
    _json_pose,
    capture_calibration_burst,
    estimate_frame_pose,
    load_marker_map,
    make_detector,
    rgb_camera_calibration,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Capture a reusable aligned-pointcloud pose map from the fixed four-face "
            "ArUco profile"
        )
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=250000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/orbit_pose_map"))
    parser.add_argument("--marker-map", type=Path, default=DEFAULT_MARKER_MAP)
    parser.add_argument("--x-pos", type=float, default=150.0)
    parser.add_argument("--step-deg", type=float, default=10.0)
    parser.add_argument("--reference-angle-deg", type=float, default=0.0)
    parser.add_argument("--frames-per-angle", type=int, default=10)
    parser.add_argument("--min-valid-poses-per-angle", type=int, default=6)
    parser.add_argument("--pose-mad-threshold", type=float, default=3.5)
    parser.add_argument("--max-reprojection-error-px", type=float, default=1.5)
    parser.add_argument("--depth-min-m", type=float, default=0.02)
    parser.add_argument("--depth-max-m", type=float, default=0.35)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=530)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--disparity", choices=("128", "256"), default="256")
    parser.add_argument("--pose-axis-length-m", type=float, default=0.025)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if not np.isfinite(args.x_pos):
        raise ValueError("--x-pos must be finite")
    if not (0.0 < args.step_deg <= 180.0):
        raise ValueError("--step-deg must be within (0, 180]")
    if args.frames_per_angle < 1:
        raise ValueError("--frames-per-angle must be at least 1")
    if not (1 <= args.min_valid_poses_per_angle <= args.frames_per_angle):
        raise ValueError(
            "--min-valid-poses-per-angle must be between 1 and --frames-per-angle"
        )
    if args.pose_mad_threshold <= 0.0:
        raise ValueError("--pose-mad-threshold must be positive")
    if args.max_reprojection_error_px <= 0.0:
        raise ValueError("--max-reprojection-error-px must be positive")
    if args.depth_min_m < 0.0 or args.depth_max_m <= args.depth_min_m:
        raise ValueError("Depth range must satisfy 0 <= min < max")
    if args.settle_seconds < 0.0:
        raise ValueError("--settle-seconds cannot be negative")


def _capture_angle(camera, angle: float, args, calibration: dict) -> dict:
    colors, depth_m = capture_calibration_burst(camera, args)
    angle_token = f"{angle:+06.1f}"
    result = {
        "angle_deg": float(angle),
        "pose_valid": False,
        "captured_frames": len(colors),
        "accepted_pose_frames": 0,
        "frames": [],
    }
    if not colors or depth_m is None:
        result["rejection_reason"] = "no complete RGB-D frame captured"
        return result

    ply_path = save_frame_as_ply(
        colors[-1],
        depth_m,
        camera.intrinsics,
        angle,
        args.output_dir,
        depth_trunc_m=args.depth_max_m,
        filename=capture_filename(0, args.x_pos, angle),
    )
    result["ply"] = str(ply_path)
    pose_dir = args.output_dir / "pose_diagnostics"
    pose_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    latest_overlay = None
    for frame_index, color_rgb in enumerate(colors):
        pose, corners, ids = estimate_frame_pose(
            color_rgb,
            calibration["marker_map"],
            calibration["detector"],
            calibration["camera_matrix"],
            calibration["dist_coeffs"],
            calibration["depth_to_color"],
            args.max_reprojection_error_px,
        )
        overlay = _draw_detection_overlay(
            color_rgb,
            corners,
            ids,
            pose,
            calibration["camera_matrix"],
            calibration["dist_coeffs"],
            args.pose_axis_length_m,
        )
        overlay_path = pose_dir / f"angle_{angle_token}_burst_{frame_index:02d}.png"
        cv2.imwrite(str(overlay_path), overlay)
        latest_overlay = overlay
        result["frames"].append(_json_pose(pose, frame_index, overlay_path))
        if pose["accepted"]:
            accepted.append(pose)

    result["accepted_pose_frames"] = len(accepted)
    if latest_overlay is not None:
        depth_vis = np.clip(depth_m / args.depth_max_m * 255.0, 0, 255).astype(np.uint8)
        visual = np.hstack((latest_overlay, cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)))
        visual_path = args.output_dir / f"angle_{angle_token}_visual.png"
        cv2.imwrite(str(visual_path), visual)
        result["visual"] = str(visual_path)

    if len(accepted) < args.min_valid_poses_per_angle:
        result["rejection_reason"] = (
            f"only {len(accepted)} accepted PnP poses; "
            f"need {args.min_valid_poses_per_angle}"
        )
        return result

    try:
        pose_key = (
            "world_to_camera"
            if calibration["pointcloud_coordinate_frame"] == "color"
            else "world_to_depth"
        )
        aggregate = aggregate_world_to_camera_poses(
            [pose[pose_key] for pose in accepted],
            min_inliers=args.min_valid_poses_per_angle,
            mad_threshold=args.pose_mad_threshold,
        )
    except ValueError as exc:
        result["rejection_reason"] = str(exc)
        return result

    for frame_result, is_inlier in zip(
        [frame for frame in result["frames"] if frame["accepted"]],
        aggregate["inlier_mask"],
    ):
        frame_result["pose_aggregate_inlier"] = bool(is_inlier)
    result.update(
        {
            "pose_valid": True,
            "world_to_pointcloud": aggregate["world_to_camera"],
            "pointcloud_to_world": aggregate["camera_to_world"],
            "pose_inlier_mask": aggregate["inlier_mask"],
            "translation_residuals_m": aggregate["translation_residuals_m"],
            "rotation_residuals_deg": aggregate["rotation_residuals_deg"],
            "translation_spread_m": aggregate["translation_spread_m"],
            "rotation_spread_deg": aggregate["rotation_spread_deg"],
        }
    )
    return result


def _write_capture_diagnostics(path: Path, args, angle_results: list[dict], camera: dict) -> None:
    payload = {
        "schema_version": 1,
        "purpose": "raw fixed-profile full-pose calibration diagnostics",
        "marker_map": str(args.marker_map),
        "motor": {"x_position_mm": args.x_pos, "step_deg": args.step_deg},
        "camera": camera,
        "acceptance": {
            "frames_per_angle": args.frames_per_angle,
            "min_valid_poses_per_angle": args.min_valid_poses_per_angle,
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "pose_mad_threshold": args.pose_mad_threshold,
        },
        "angles": angle_results,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if not args.marker_map.is_file():
        raise FileNotFoundError(f"Marker map not found: {args.marker_map}")
    marker_map = load_marker_map(args.marker_map)
    sequence = generate_scan_sequence(args.step_deg, [args.x_pos])
    if args.dry_run:
        logger.info("[DRY RUN] Home motors: %s", "no" if args.no_home else "yes")
        logger.info("[DRY RUN] Select G90 absolute positioning.")
        for step in sequence:
            if step["kind"] == "move_x":
                logger.info("[DRY RUN] Move X to %.1f mm while Y=0.", args.x_pos)
            else:
                logger.info(
                    "[DRY RUN] Move Y to %+.1f deg%s.",
                    step["angle_deg"],
                    " and capture profile" if step["capture"] else " without capture",
                )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.output_dir / "orbit_pose_capture_diagnostics.json"
    angle_results = []
    camera_report = {}
    with MotorController(port=args.port, baud=args.baud) as motor, CameraController(
        width=args.width,
        height=args.height,
        fps=args.fps,
        disparity=args.disparity,
    ) as camera:
        if not args.no_home:
            logger.info("Homing motors...")
            motor.home_all()
        if not motor.send_command("G90"):
            raise RuntimeError("Failed to select absolute positioning with G90")

        rgb_intrinsics, camera_matrix, dist_coeffs = rgb_camera_calibration(
            camera.camera_param
        )
        depth_to_color = orbbec_extrinsic_to_matrix(camera.camera_param.transform)
        calibration = {
            "marker_map": marker_map,
            "detector": make_detector(marker_map["dictionary"]),
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "depth_to_color": depth_to_color,
            "pointcloud_coordinate_frame": camera.pointcloud_coordinate_frame,
        }
        camera_report = {
            "active_disparity": camera.active_disparity,
            "pointcloud_coordinate_frame": camera.intrinsics["coordinate_frame"],
            "pointcloud_intrinsics": camera.intrinsics,
            "rgb_intrinsics": rgb_intrinsics,
            "rgb_distortion": dist_coeffs.reshape(-1).tolist(),
            "depth_to_color": depth_to_color.tolist(),
        }

        for step in sequence:
            if step["kind"] == "move_x":
                logger.info("Moving X to %.1f mm at Y=0...", args.x_pos)
                motor.move_x(args.x_pos, feedrate=500)
                time.sleep(args.settle_seconds)
                continue
            angle = float(step["angle_deg"])
            motor.move_y(angle, feedrate=500)
            time.sleep(args.settle_seconds)
            if not step["capture"]:
                logger.info("Returned Y to %+.1f deg without capture.", angle)
                continue
            logger.info("Capturing fixed profile at Y=%+.1f deg...", angle)
            result = _capture_angle(camera, angle, args, calibration)
            angle_results.append(result)
            _write_capture_diagnostics(
                diagnostics_path, args, angle_results, camera_report
            )
            if result["pose_valid"]:
                logger.info(
                    "• Y=%+.1f: full pose valid, spread %.3f mm / %.3f deg",
                    angle,
                    result["translation_spread_m"] * 1000.0,
                    result["rotation_spread_deg"],
                )
            else:
                logger.warning(
                    "• Y=%+.1f: pose invalid (%s)",
                    angle,
                    result.get("rejection_reason", "unknown reason"),
                )

    valid_results = [result for result in angle_results if result["pose_valid"]]
    if not valid_results:
        raise RuntimeError(f"No valid full poses; inspect {diagnostics_path}")
    reference_angle = args.reference_angle_deg
    if not any(abs(result["angle_deg"] - reference_angle) <= 1e-6 for result in valid_results):
        reference_angle = valid_results[0]["angle_deg"]
        logger.warning(
            "Requested reference angle was invalid; using Y=%+.1f for diagnostics. "
            "The map remains invalid until all object-scan angles have poses.",
            reference_angle,
        )
    pose_map = build_orbit_pose_map(
        angle_results,
        reference_angle_deg=reference_angle,
        x_position_mm=args.x_pos,
        marker_map_path=str(args.marker_map),
        pointcloud_coordinate_frame=camera_report["pointcloud_coordinate_frame"],
    )
    pose_map["camera"] = camera_report
    pose_map["capture_diagnostics"] = str(diagnostics_path)
    pose_map_path = args.output_dir / "orbit_pose_map.json"
    pose_map_path.write_text(json.dumps(pose_map, indent=2) + "\n", encoding="utf-8")

    valid_count = sum(result["pose_valid"] for result in angle_results)
    logger.info("• Valid full poses: %d/%d angles", valid_count, len(angle_results))
    logger.info("• Reference camera angle: %+.1f deg", pose_map["reference_angle_deg"])
    if pose_map.get("orbit_fit_profile_frame"):
        logger.info(
            "• Diagnostic orbit radius: %.3f mm",
            pose_map["orbit_fit_profile_frame"]["radius_m"] * 1000.0,
        )
    logger.info("• Pose map: %s", pose_map_path)
    if pose_map["quality_status"] != "valid":
        logger.warning("Pose map is incomplete: %s", "; ".join(pose_map["quality_reasons"]))


if __name__ == "__main__":
    main()
