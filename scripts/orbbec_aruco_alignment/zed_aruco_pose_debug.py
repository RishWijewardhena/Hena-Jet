#!/usr/bin/env python3
"""Capture ZED-M RGB frames and debug ArUco table-board pose estimation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

from aruco_common import (
    camera_matrix_from_intrinsics,
    color_image_to_rgb,
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


RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug ZED-M ArUco table-board pose.")
    parser.add_argument(
        "--board-json",
        type=Path,
        default=Path("outputs/aruco_table_board/aruco_table_board.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/zed_aruco_debug"))
    parser.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="HD720")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--min-markers", type=int, default=2)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    return parser.parse_args()


def camera_intrinsics_from_zed(zed: sl.Camera, image_shape: tuple[int, int]) -> dict[str, float]:
    camera_info = zed.get_camera_information()
    calibration = camera_info.camera_configuration.calibration_parameters
    left = calibration.left_cam
    height, width = image_shape
    return {
        "fx": float(left.fx),
        "fy": float(left.fy),
        "cx": float(left.cx),
        "cy": float(left.cy),
        "width": int(width),
        "height": int(height),
    }


def open_zed(args: argparse.Namespace) -> sl.Camera:
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[args.resolution]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Could not open ZED camera")
    return zed


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.min_markers <= 0:
        raise ValueError("--min-markers must be positive")

    board = load_board(args.board_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zed = open_zed(args)
    runtime = sl.RuntimeParameters()
    color_mat = sl.Mat()
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    try:
        print(f"Warming up ({args.warmup} frames)...")
        for _ in range(args.warmup):
            zed.grab(runtime)

        results = []
        intrinsics = None
        for frame_index in range(args.frames):
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                print(f"Frame {frame_index}: grab failed")
                continue
            zed.retrieve_image(color_mat, sl.VIEW.LEFT)
            color_rgb = color_image_to_rgb(color_mat.get_data())
            if intrinsics is None:
                intrinsics = camera_intrinsics_from_zed(zed, color_rgb.shape[:2])
            camera_matrix = camera_matrix_from_intrinsics(intrinsics)

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
                    f"Frame {frame_index}: pose ok, markers={used_ids}, "
                    f"mean reprojection={error_px:.2f}px"
                )
            else:
                print(f"Frame {frame_index}: pose failed, detected ids={ids.reshape(-1).tolist()}")

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

            result = {
                "frame_index": frame_index,
                "captured_at_s": time.time(),
                "detected_ids": ids.reshape(-1).astype(int).tolist(),
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
        print(f"Debug outputs: {args.out_dir}")
    finally:
        zed.close()


if __name__ == "__main__":
    main()
