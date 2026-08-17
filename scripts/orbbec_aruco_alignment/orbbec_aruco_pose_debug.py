#!/usr/bin/env python3
"""Capture Orbbec RGB frames and debug ArUco table-board pose estimation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from aruco_common import (
    camera_matrix_from_intrinsics,
    camera_params_from_orbbec,
    color_frame_to_rgb,
    detect_markers,
    draw_pose_overlay,
    extrinsic_from_rvec_tvec,
    load_board,
    match_board_corners,
    pose_summary,
    reprojection_error_px,
    solve_board_pose,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Orbbec ArUco table-board pose.")
    parser.add_argument(
        "--board-json",
        type=Path,
        default=Path("outputs/aruco_table_board/aruco_table_board.json"),
        help="Path to generated ArUco board JSON specification.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/orbbec_aruco_debug"),
        help="Output directory for debug overlays and pose json.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Color stream width.")
    parser.add_argument("--height", type=int, default=800, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream framerate.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup frames.")
    parser.add_argument("--frames", type=int, default=1, help="Number of test frames to process.")
    parser.add_argument(
        "--min-markers",
        type=int,
        default=2,
        help="Minimum visible markers required to accept pose.",
    )
    parser.add_argument(
        "--axis-length-m",
        type=float,
        default=0.05,
        help="Length of drawn 3D coordinate axes in meters.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="Frame wait timeout in milliseconds.",
    )
    return parser.parse_args()


def open_orbbec_color_pipeline(args: argparse.Namespace):
    try:
        from pyorbbecsdk import (  # type: ignore
            Config,
            Context,
            OBFormat,
            OBSensorType,
            Pipeline,
        )
    except ImportError as exc:
        raise SystemExit(
            "pyorbbecsdk is required; run this script in the hena_jet conda environment."
        ) from exc

    ctx = Context()
    devices = ctx.query_devices()
    if devices.get_count() == 0:
        raise SystemExit("No Orbbec camera detected.")

    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = None
    # Try RGB format first, then fallback to MJPG / default
    for fmt in (OBFormat.RGB, OBFormat.MJPG, OBFormat.BGR):
        try:
            color_profile = profile_list.get_video_stream_profile(
                args.width, args.height, fmt, args.fps
            )
            if color_profile is not None:
                break
        except Exception:
            continue

    if color_profile is None:
        try:
            color_profile = profile_list.get_default_video_stream_profile()
            print(
                f"Warning: Requested {args.width}x{args.height}@{args.fps} not found, "
                f"falling back to default color profile."
            )
        except Exception as exc:
            raise SystemExit(f"Failed to find a suitable color stream profile: {exc}") from exc

    config.enable_stream(color_profile)
    pipeline.start(config)
    return pipeline


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.min_markers <= 0:
        raise ValueError("--min-markers must be positive")

    board = load_board(args.board_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = open_orbbec_color_pipeline(args)

    try:
        print(f"Warming up ({args.warmup} frames)...")
        for _ in range(args.warmup):
            frames = pipeline.wait_for_frames(args.timeout_ms)
            if frames is None:
                raise RuntimeError("Timeout while waiting for camera frames during warmup")

        camera_param = pipeline.get_camera_param()
        intrinsics, dist_coeffs = camera_params_from_orbbec(camera_param)
        camera_matrix = camera_matrix_from_intrinsics(intrinsics)

        results = []
        for frame_index in range(args.frames):
            frames = pipeline.wait_for_frames(args.timeout_ms)
            if frames is None:
                print(f"Frame {frame_index}: timeout waiting for frame")
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                print(f"Frame {frame_index}: no color frame received")
                continue

            color_rgb = color_frame_to_rgb(color_frame)

            corners, ids, rejected = detect_markers(color_rgb, board)
            object_points, image_points, used_ids = match_board_corners(board, corners, ids)
            ok, rvec, tvec = solve_board_pose(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                args.min_markers,
                used_ids,
            )

            world_to_camera = None
            error_px = None
            if ok:
                world_to_camera = extrinsic_from_rvec_tvec(rvec, tvec)
                error_px = reprojection_error_px(
                    object_points,
                    image_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )
                print(
                    f"Frame {frame_index}: pose OK, visible markers={used_ids}, "
                    f"mean reprojection error={error_px:.2f}px"
                )
            else:
                detected_flat = ids.reshape(-1).tolist() if ids is not None else []
                print(
                    f"Frame {frame_index}: pose failed, detected marker ids={detected_flat}"
                )

            overlay = draw_pose_overlay(
                color_rgb,
                corners,
                ids,
                camera_matrix,
                dist_coeffs,
                rvec if ok else None,
                tvec if ok else None,
                args.axis_length_m,
            )
            overlay_path = args.out_dir / f"pose_debug_{frame_index:04d}.png"
            cv2.imwrite(str(overlay_path), overlay)

            detected_flat = ids.reshape(-1).astype(int).tolist() if ids is not None else []
            result = {
                "frame_index": frame_index,
                "captured_at_s": time.time(),
                "detected_ids": detected_flat,
                "used_ids": used_ids,
                "pose_ok": ok,
                "mean_reprojection_error_px": error_px,
                "overlay": str(overlay_path),
            }
            if world_to_camera is not None:
                result.update(pose_summary(world_to_camera))
            results.append(result)

        write_json(
            args.out_dir / "pose_debug.json",
            {
                "board_json": str(args.board_json),
                "camera_intrinsics": intrinsics,
                "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
                "frames": results,
            },
        )
        print(f"Debug outputs written to: {args.out_dir}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
