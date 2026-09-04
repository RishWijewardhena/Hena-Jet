#!/usr/bin/env python3
"""Capture ArUco poses around an orbit and estimate the depth-camera radius."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Keep the existing sibling-module imports working when this script is run
# directly from the calculating_radius subdirectory.
SYNC_WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(SYNC_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from camera_controller import CameraController
from main_scan import (
    color_frame_to_rgb,
    depth_frame_to_meters,
    fuse_depth_frames,
    save_frame_as_ply,
)
from motor_controller import MotorController
from calculating_radius.radius_calibration import (
    build_detector_parameters,
    build_fit_samples,
    camera_center_world,
    classify_pose,
    convert_world_to_color_to_world_to_depth,
    DEFAULT_MAX_FIT_RESIDUAL_M,
    DEFAULT_MAX_FIT_RMSE_M,
    DEFAULT_MAX_REPROJECTION_ERROR_PX,
    DEFAULT_MAX_SINGLE_MARKER_REPROJECTION_ERROR_PX,
    DEFAULT_MIN_IPPE_ERROR_RATIO,
    evaluate_trajectory,
    orbit_geometry_in_camera_frame,
    orbbec_extrinsic_to_matrix,
    solve_profile_pose,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MARKER_MAP = Path("outputs/radius_markers/profile_marker_map.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the depth-camera orbit radius from a fixed four-face ArUco profile"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Motor controller serial port")
    parser.add_argument("--baud", type=int, default=250000, help="Baud rate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_radius"))
    parser.add_argument("--marker-map", type=Path, default=DEFAULT_MARKER_MAP)
    parser.add_argument("--x-pos", type=float, default=80.0, help="Absolute X calibration position in mm")
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[
            0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
            -30.0, -60.0, -90.0, -120.0, -150.0, -180.0,
        ],
        help="Motor angles used to observe the fixed marker profile "
        "(positive 0..180 sweep, then negative -30..-180 sweep)",
    )
    parser.add_argument("--frames-per-angle", type=int, default=10)
    parser.add_argument("--depth-min-m", type=float, default=0.02)
    parser.add_argument("--depth-max-m", type=float, default=0.35)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=DEFAULT_MAX_REPROJECTION_ERROR_PX,
        help="Reprojection limit for a multi-marker pose, where the measure is "
        "meaningful because the markers over-constrain the fit",
    )
    parser.add_argument(
        "--max-single-marker-reprojection-error-px",
        type=float,
        default=DEFAULT_MAX_SINGLE_MARKER_REPROJECTION_ERROR_PX,
        help="Reprojection limit for a single-marker pose; IPPE solves four "
        "coplanar points almost exactly, so this is a sanity check, not a "
        "quality gate",
    )
    parser.add_argument(
        "--no-prefer-multi-marker",
        dest="prefer_multi_marker",
        action="store_false",
        help="Feed single-marker frames to the fit even at angles that also saw "
        "two markers, instead of preferring the better-conditioned poses",
    )
    parser.add_argument(
        "--min-markers-per-pose",
        type=int,
        default=1,
        help="Minimum mapped ArUco markers a frame pose must use to enter the radius "
        "fit; the four-face profile shows a second marker only near corner-on views, "
        "so ambiguity is gated by --min-ippe-error-ratio instead",
    )
    parser.add_argument(
        "--min-ippe-error-ratio",
        type=float,
        default=DEFAULT_MIN_IPPE_ERROR_RATIO,
        help="Reject a single-marker pose unless the discarded IPPE solution fits at "
        "least this many times worse, since two near-equal solutions differ mostly "
        "out of plane",
    )
    parser.add_argument(
        "--max-fit-rmse-mm",
        type=float,
        default=DEFAULT_MAX_FIT_RMSE_M * 1000.0,
        help="Reject the calibration when the inlier circle-fit RMSE exceeds this",
    )
    parser.add_argument(
        "--max-fit-residual-mm",
        type=float,
        default=DEFAULT_MAX_FIT_RESIDUAL_M * 1000.0,
        help="Reject the calibration when any inlier circle-fit residual exceeds this",
    )
    parser.add_argument(
        "--per-angle-median",
        action="store_true",
        help="Fit the orbit circle to one median camera center per angle instead of "
        "every accepted frame pose",
    )
    parser.add_argument("--min-valid-poses-per-angle", type=int, default=6)
    parser.add_argument("--min-unique-angles", type=int, default=6)
    parser.add_argument("--max-angle-gap-deg", type=float, default=90.0)
    parser.add_argument("--pose-axis-length-m", type=float, default=0.025)
    return parser.parse_args(argv)


def load_marker_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        marker_map = json.load(file)
    required = {"dictionary", "markers"}
    missing = required - marker_map.keys()
    if missing:
        raise ValueError(f"Marker map is missing fields: {', '.join(sorted(missing))}")
    has_global_size = marker_map.get("marker_size_m") is not None
    if not has_global_size and any(
        marker.get("size_m") is None for marker in marker_map["markers"]
    ):
        raise ValueError(
            "Marker map must provide marker_size_m or size_m for every marker"
        )
    return marker_map


def rgb_camera_calibration(camera_param):
    intrinsic = camera_param.rgb_intrinsic
    distortion = camera_param.rgb_distortion
    intrinsics = {
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
    }
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(
        [
            distortion.k1,
            distortion.k2,
            distortion.p1,
            distortion.p2,
            distortion.k3,
            distortion.k4,
            distortion.k5,
            distortion.k6,
        ],
        dtype=np.float64,
    ).reshape(-1, 1)
    return intrinsics, camera_matrix, dist_coeffs


def make_detector(dictionary_name: str):
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.ArucoDetector(dictionary, build_detector_parameters())


def capture_calibration_burst(camera, args):
    colors = []
    depths = []
    for _ in range(args.frames_per_angle):
        color_frame, depth_frame = camera.capture_aligned_rgbd(timeout_ms=args.timeout_ms)
        if color_frame is None or depth_frame is None:
            continue
        colors.append(color_frame_to_rgb(color_frame).copy())
        depths.append(depth_frame_to_meters(depth_frame))
    if not colors or not depths:
        return [], None
    fused_depth = fuse_depth_frames(
        depths,
        min_depth_m=args.depth_min_m,
        max_depth_m=args.depth_max_m,
    )
    return colors, fused_depth


def estimate_frame_pose(
    color_rgb,
    marker_map,
    detector,
    camera_matrix,
    dist_coeffs,
    depth_to_color,
    max_reprojection_error_px,
    min_markers=1,
    min_ippe_error_ratio=DEFAULT_MIN_IPPE_ERROR_RATIO,
    max_single_marker_reprojection_error_px=(
        DEFAULT_MAX_SINGLE_MARKER_REPROJECTION_ERROR_PX
    ),
):
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        ids = np.empty((0, 1), dtype=np.int32)
    pose = solve_profile_pose(marker_map, corners, ids, camera_matrix, dist_coeffs)
    pose["detected_ids"] = ids.reshape(-1).astype(int).tolist()
    accepted, rejection_reason = classify_pose(
        pose,
        max_reprojection_error_px=max_reprojection_error_px,
        min_markers=min_markers,
        min_ippe_error_ratio=min_ippe_error_ratio,
        max_single_marker_reprojection_error_px=(
            max_single_marker_reprojection_error_px
        ),
    )
    pose["accepted"] = accepted
    pose["rejection_reason"] = rejection_reason
    # Store the camera centre for every solved pose, accepted or not, so a
    # threshold can be re-evaluated against a recorded run instead of
    # re-scanning the rig to find out what a rejected pose would have added.
    if pose.get("ok"):
        world_to_color = pose["world_to_camera"]
        world_to_depth = convert_world_to_color_to_world_to_depth(
            world_to_color, depth_to_color
        )
        pose["world_to_depth"] = world_to_depth
        pose["rgb_camera_center_m"] = camera_center_world(world_to_color)
        pose["depth_camera_center_m"] = camera_center_world(world_to_depth)
    return pose, corners, ids


def _draw_detection_overlay(
    color_rgb,
    corners,
    ids,
    pose,
    camera_matrix,
    dist_coeffs,
    axis_length_m,
):
    overlay = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    if ids is not None and len(ids):
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
    if pose["ok"]:
        cv2.drawFrameAxes(
            overlay,
            camera_matrix,
            dist_coeffs,
            pose["rvec"],
            pose["tvec"],
            axis_length_m,
        )
    status = "accepted" if pose.get("accepted") else pose.get("rejection_reason", "pose failed")
    error = pose.get("reprojection_error_px")
    label = status if error is None else f"{status} | error {error:.2f}px"
    color = (0, 220, 0) if pose.get("accepted") else (0, 0, 255)
    cv2.putText(overlay, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return overlay


def _json_pose(pose, frame_index, overlay_path):
    result = {
        "frame_index": frame_index,
        "detected_ids": pose["detected_ids"],
        "used_ids": pose["used_ids"],
        "pose_ok": bool(pose["ok"]),
        "accepted": bool(pose["accepted"]),
        "method": pose["method"],
        "reprojection_error_px": pose["reprojection_error_px"],
        "rejection_reason": pose["rejection_reason"],
        "overlay": str(overlay_path),
    }
    if pose["accepted"]:
        result.update(
            {
                "world_to_color": pose["world_to_camera"].tolist(),
                "world_to_depth": pose["world_to_depth"].tolist(),
                "rgb_camera_center_m": pose["rgb_camera_center_m"].tolist(),
                "depth_camera_center_m": pose["depth_camera_center_m"].tolist(),
            }
        )
    return result


def capture_and_save(camera, angle, args, calibration):
    logger.info("Capturing frame at %.1f degrees...", angle)
    color_frames, depth_m = capture_calibration_burst(camera, args)
    if not color_frames or depth_m is None:
        logger.error("Failed to capture frame at %.1f degrees.", angle)
        return None

    logger.info("Combined %d/%d frames.", len(color_frames), args.frames_per_angle)
    color_rgb = color_frames[-1]
    ply_path = save_frame_as_ply(
        color_rgb,
        depth_m,
        camera.intrinsics,
        angle,
        args.output_dir,
        depth_trunc_m=args.depth_max_m,
    )

    angle_token = f"{angle:+06.1f}"
    pose_dir = args.output_dir / "pose_diagnostics"
    pose_dir.mkdir(parents=True, exist_ok=True)
    frame_results = []
    accepted_poses = []
    latest_overlay = None
    for frame_index, pose_color in enumerate(color_frames):
        pose, corners, ids = estimate_frame_pose(
            pose_color,
            calibration["marker_map"],
            calibration["detector"],
            calibration["camera_matrix"],
            calibration["dist_coeffs"],
            calibration["depth_to_color"],
            args.max_reprojection_error_px,
            min_markers=args.min_markers_per_pose,
            min_ippe_error_ratio=args.min_ippe_error_ratio,
            max_single_marker_reprojection_error_px=(
                args.max_single_marker_reprojection_error_px
            ),
        )
        overlay = _draw_detection_overlay(
            pose_color,
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
        frame_results.append(_json_pose(pose, frame_index, overlay_path))
        if pose["accepted"]:
            accepted_poses.append(pose)

    depth_vis = np.clip(depth_m / args.depth_max_m * 255.0, 0, 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    if latest_overlay is None:
        latest_overlay = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    visual = np.hstack((latest_overlay, depth_colormap))
    visual_path = args.output_dir / f"frame_{angle:.1f}_visual.png"
    cv2.imwrite(str(visual_path), visual)

    angle_result = {
        "angle_deg": float(angle),
        "ply": str(ply_path),
        "visual": str(visual_path),
        "captured_frames": len(color_frames),
        "accepted_pose_frames": len(accepted_poses),
        "pose_valid": len(accepted_poses) >= args.min_valid_poses_per_angle,
        "frames": frame_results,
    }
    if angle_result["pose_valid"]:
        angle_result["rgb_camera_center_m"] = np.median(
            [pose["rgb_camera_center_m"] for pose in accepted_poses], axis=0
        ).tolist()
        angle_result["depth_camera_center_m"] = np.median(
            [pose["depth_camera_center_m"] for pose in accepted_poses], axis=0
        ).tolist()
        logger.info(
            "• Angle %.1f: accepted %d/%d marker poses",
            angle,
            len(accepted_poses),
            len(color_frames),
        )
    else:
        logger.warning(
            "• Angle %.1f omitted from radius fit: only %d/%d valid marker poses",
            angle,
            len(accepted_poses),
            args.min_valid_poses_per_angle,
        )
    return angle_result


def write_camera_centers_ply(path: Path, samples: list[dict], trajectory: dict) -> None:
    if not samples or trajectory.get("depth_fit") is None:
        return
    inliers = trajectory["depth_fit"]["inlier_mask"]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(samples)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for sample, is_inlier in zip(samples, inliers):
        x, y, z = sample["depth_camera_center_m"]
        color = (0, 220, 0) if is_inlier else (255, 0, 0)
        lines.append(f"{x:.9f} {y:.9f} {z:.9f} {color[0]} {color[1]} {color[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_angle_csv(path: Path, samples: list[dict], trajectory: dict) -> None:
    inliers = trajectory["depth_fit"]["inlier_mask"] if trajectory.get("depth_fit") else [False] * len(samples)
    residuals = trajectory["depth_fit"]["residuals_m"] if trajectory.get("depth_fit") else [None] * len(samples)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["angle_deg", "depth_x_m", "depth_y_m", "depth_z_m", "fit_inlier", "residual_m"]
        )
        for sample, is_inlier, residual in zip(samples, inliers, residuals):
            writer.writerow(
                [sample["angle_deg"], *sample["depth_camera_center_m"], bool(is_inlier), residual]
            )


def validate_args(args) -> None:
    if args.frames_per_angle < 1:
        raise ValueError("--frames-per-angle must be at least 1.")
    if args.min_valid_poses_per_angle < 1:
        raise ValueError("--min-valid-poses-per-angle must be at least 1.")
    if args.min_valid_poses_per_angle > args.frames_per_angle:
        raise ValueError("--min-valid-poses-per-angle cannot exceed --frames-per-angle.")
    if args.depth_min_m < 0.0 or args.depth_max_m <= args.depth_min_m:
        raise ValueError("Depth range must satisfy 0 <= min < max.")
    if args.max_reprojection_error_px <= 0.0:
        raise ValueError("--max-reprojection-error-px must be positive.")
    if args.max_single_marker_reprojection_error_px <= 0.0:
        raise ValueError(
            "--max-single-marker-reprojection-error-px must be positive."
        )
    if args.min_ippe_error_ratio < 1.0:
        raise ValueError("--min-ippe-error-ratio must be at least 1.0.")
    if args.max_fit_rmse_mm <= 0.0:
        raise ValueError("--max-fit-rmse-mm must be positive.")
    if args.max_fit_residual_mm <= 0.0:
        raise ValueError("--max-fit-residual-mm must be positive.")
    if not 1 <= args.min_markers_per_pose <= 4:
        raise ValueError("--min-markers-per-pose must be within [1, 4].")
    if args.min_unique_angles < 3:
        raise ValueError("--min-unique-angles must be at least 3.")
    if not (0.0 < args.max_angle_gap_deg <= 360.0):
        raise ValueError("--max-angle-gap-deg must be within (0, 360].")


def main():
    args = parse_args()
    validate_args(args)
    if not args.marker_map.is_file():
        raise FileNotFoundError(
            f"Marker map not found: {args.marker_map}. Generate it with: "
            "python scripts/sync_workflow/calculating_radius/generate_radius_markers.py"
        )
    marker_map = load_marker_map(args.marker_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing hardware...")
    angle_results = []
    calibration = None
    active_disparity = None
    with MotorController(port=args.port, baud=args.baud) as motor, CameraController() as camera:
        active_disparity = camera.active_disparity
        rgb_intrinsics, camera_matrix, dist_coeffs = rgb_camera_calibration(camera.camera_param)
        depth_to_color = orbbec_extrinsic_to_matrix(camera.camera_param.transform)
        calibration = {
            "marker_map": marker_map,
            "detector": make_detector(marker_map["dictionary"]),
            "rgb_intrinsics": rgb_intrinsics,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "depth_to_color": depth_to_color,
            "pointcloud_coordinate_frame": camera.pointcloud_coordinate_frame,
        }

        logger.info("Homing motors...")
        motor.home_all()
        logger.info("Moving X to %.1f (absolute, M400 blocking move)...", args.x_pos)
        if not motor.send_command("G90"):
            raise RuntimeError("Failed to select absolute motor positioning.")
        motor.move_x(args.x_pos, feedrate=1000)

        previous_angle = None
        for angle in args.angles:
            if previous_angle is not None and previous_angle > 0.0 > angle:
                logger.info("Returning Y to 0.0 before the negative sweep...")
                motor.move_y(0.0, feedrate=500)
                time.sleep(0.5)
            logger.info("Moving Y to %.1f...", angle)
            motor.move_y(angle, feedrate=500)
            time.sleep(0.5)
            result = capture_and_save(camera, angle, args, calibration)
            if result is not None:
                angle_results.append(result)
            previous_angle = angle
        motor.move_y(0.0, feedrate=500)

    valid_samples = build_fit_samples(
        angle_results,
        use_all_frames=not args.per_angle_median,
        prefer_multi_marker=args.prefer_multi_marker,
    )
    trajectory = evaluate_trajectory(
        valid_samples,
        min_unique_angles=args.min_unique_angles,
        max_gap_deg=args.max_angle_gap_deg,
        max_fit_rmse_m=args.max_fit_rmse_mm / 1000.0,
        max_fit_residual_m=args.max_fit_residual_mm / 1000.0,
    )
    pointcloud_fit_key = (
        "rgb_fit"
        if calibration["pointcloud_coordinate_frame"] == "color"
        else "depth_fit"
    )
    if trajectory["quality_status"] == "valid":
        trajectory["recommended_radius_m"] = trajectory[pointcloud_fit_key]["radius_m"]
        reference_sample = next(
            (sample for sample in valid_samples if abs(float(sample["angle_deg"])) < 1e-9),
            None,
        )
        reference_pose = None
        reference_pose_key = (
            "world_to_color"
            if calibration["pointcloud_coordinate_frame"] == "color"
            else "world_to_depth"
        )
        for angle_result in angle_results:
            if abs(float(angle_result["angle_deg"])) >= 1e-9:
                continue
            for frame in angle_result.get("frames", []):
                if frame.get("accepted"):
                    reference_pose = np.asarray(
                        frame[reference_pose_key], dtype=np.float64
                    )
                    break
        if reference_sample is not None and reference_pose is not None:
            trajectory["orbit_geometry"] = orbit_geometry_in_camera_frame(
                trajectory[pointcloud_fit_key]["center_m"],
                trajectory[pointcloud_fit_key]["axis"],
                reference_pose,
            )
    report = {
        "schema_version": 1,
        "purpose": "four-face ArUco depth-camera orbit-radius calibration",
        "quality_status": trajectory["quality_status"],
        "quality_reasons": trajectory["quality_reasons"],
        "recommended_radius_m": trajectory["recommended_radius_m"],
        "rgb_fit": trajectory["rgb_fit"],
        "depth_fit": trajectory["depth_fit"],
        "orbit_geometry": trajectory.get("orbit_geometry"),
        "marker_map": str(args.marker_map),
        "marker_geometry": marker_map,
        "motor": {"x_position_mm": args.x_pos, "angles_deg": [float(a) for a in args.angles]},
        "acceptance": {
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "min_markers_per_pose": args.min_markers_per_pose,
            "min_ippe_error_ratio": args.min_ippe_error_ratio,
            "max_single_marker_reprojection_error_px": (
                args.max_single_marker_reprojection_error_px
            ),
            "prefer_multi_marker": bool(args.prefer_multi_marker),
            "per_angle_median": bool(args.per_angle_median),
            "min_valid_poses_per_angle": args.min_valid_poses_per_angle,
            "min_unique_angles": args.min_unique_angles,
            "max_angle_gap_deg": args.max_angle_gap_deg,
            "max_fit_rmse_m": args.max_fit_rmse_mm / 1000.0,
            "max_fit_residual_m": args.max_fit_residual_mm / 1000.0,
        },
        "camera": {
            "active_disparity": active_disparity,
            "pointcloud_coordinate_frame": calibration["pointcloud_coordinate_frame"],
            "rgb_intrinsics": calibration["rgb_intrinsics"],
            "rgb_distortion": calibration["dist_coeffs"].reshape(-1).tolist(),
            "depth_to_color": calibration["depth_to_color"].tolist(),
        },
        "angles": angle_results,
    }
    report_path = args.output_dir / "radius_calibration.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_angle_csv(args.output_dir / "radius_angle_summary.csv", valid_samples, trajectory)
    write_camera_centers_ply(
        args.output_dir / "depth_camera_trajectory.ply", valid_samples, trajectory
    )

    capture_manifest = {
        "purpose": "orbit-radius calibration capture",
        "target_requirement": "fixed four-face ArUco profile",
        "marker_map": str(args.marker_map),
        "x_position": args.x_pos,
        "angles_deg": [result["angle_deg"] for result in angle_results],
        "frames_per_angle": args.frames_per_angle,
        "depth_range_m": [args.depth_min_m, args.depth_max_m],
        "radius_calibration": str(report_path),
    }
    (args.output_dir / "calibration_capture.json").write_text(
        json.dumps(capture_manifest, indent=2) + "\n", encoding="utf-8"
    )

    if trajectory["quality_status"] == "valid":
        radius = trajectory["recommended_radius_m"]
        logger.info("• Estimated RGB-camera orbit radius: %.3f mm", trajectory["rgb_fit"]["radius_m"] * 1000.0)
        logger.info("• Estimated depth-camera orbit radius: %.3f mm", trajectory["depth_fit"]["radius_m"] * 1000.0)
        logger.info(
            "• Point-cloud coordinate frame: %s camera",
            calibration["pointcloud_coordinate_frame"],
        )
        logger.info("• Depth trajectory RMSE: %.3f mm", trajectory["depth_fit"]["rmse_m"] * 1000.0)
        depth_ci_low = trajectory["depth_fit"].get("radius_ci_low_m")
        depth_ci_high = trajectory["depth_fit"].get("radius_ci_high_m")
        if depth_ci_low is not None:
            logger.info(
                "• Depth radius 95%% CI: [%.3f, %.3f] mm (std %.3f mm)",
                depth_ci_low * 1000.0,
                depth_ci_high * 1000.0,
                trajectory["depth_fit"]["radius_std_m"] * 1000.0,
            )
        logger.info("• Recommended reconstruction argument: --radius-m %.6f", radius)
    else:
        logger.error("• Radius calibration is INVALID; no radius is recommended.")
        for reason in trajectory["quality_reasons"]:
            logger.error("• %s", reason)
    logger.info("Diagnostics: %s", report_path)


if __name__ == "__main__":
    main()
