#!/usr/bin/env python3
"""ZED-M RGB-D TSDF scan aligned by a printed ArUco table board."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl

from aruco_common import (
    apply_roi_mask,
    camera_matrix_from_intrinsics,
    clean_depth_image,
    color_image_to_rgb,
    detect_markers,
    draw_pose_overlay,
    extrinsic_from_rvec_tvec,
    load_board,
    make_rgbd,
    mask_marker_depth,
    match_board_corners,
    open3d_intrinsic_from_dict,
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
DEPTH_MODES = {
    name: getattr(sl.DEPTH_MODE, name)
    for name in ("PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS")
    if hasattr(sl.DEPTH_MODE, name)
}


@dataclass
class AcceptedFrame:
    index: int
    captured_at_s: float
    used_ids: list[int]
    detected_ids: list[int]
    valid_depth_px: int
    mean_reprojection_error_px: float
    world_to_camera: np.ndarray
    overlay_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZED-M ArUco table-board TSDF scanner.")
    parser.add_argument(
        "--board-json",
        type=Path,
        default=Path("outputs/aruco_table_board/aruco_table_board.json"),
    )
    parser.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="HD720")
    parser.add_argument(
        "--depth-mode",
        choices=sorted(DEPTH_MODES),
        default="NEURAL" if "NEURAL" in DEPTH_MODES else "NEURAL_LIGHT",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=0.21)
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=(0.0, 0.0, 1.0, 1.0),
        help="Normalized reconstruction crop. Marker detection still uses the full RGB image.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--min-markers", type=int, default=2)
    parser.add_argument("--min-valid-depth-px", type=int, default=5000)
    parser.add_argument("--voxel-length-m", type=float, default=0.002)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.012)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--marker-mask-padding-px", type=int, default=8)
    parser.add_argument("--no-mask-markers", action="store_true")
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/aruco_scans/zed_aruco_debug"))
    parser.add_argument("--mesh-out", type=Path, default=Path("outputs/aruco_scans/zed_aruco_tsdf_mesh.ply"))
    parser.add_argument("--cloud-out", type=Path, default=Path("outputs/aruco_scans/zed_aruco_tsdf_cloud.ply"))
    parser.add_argument("--poses-out", type=Path, default=Path("outputs/aruco_scans/zed_aruco_poses.npy"))
    parser.add_argument("--scan-json", type=Path, default=Path("outputs/aruco_scans/zed_aruco_scan.json"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.roi = tuple(args.roi)
    if args.min_depth_m <= 0.0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m must be greater than --min-depth-m")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or positive")
    if args.min_markers <= 0:
        raise ValueError("--min-markers must be positive")
    if args.min_valid_depth_px < 0:
        raise ValueError("--min-valid-depth-px must be zero or positive")
    if args.voxel_length_m <= 0.0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")
    if args.marker_mask_padding_px < 0:
        raise ValueError("--marker-mask-padding-px must be zero or positive")


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
    init.depth_mode = DEPTH_MODES[args.depth_mode]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m
    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Could not open ZED camera")
    return zed


def make_tsdf_volume(args: argparse.Namespace) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def prompt_for_capture() -> bool:
    raw_value = input("Press Enter to capture, or q to finish: ").strip().lower()
    return raw_value not in {"q", "quit", "exit"}


def save_outputs(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    accepted: list[AcceptedFrame],
    intrinsics: dict[str, float],
    board_json: Path,
    args: argparse.Namespace,
) -> None:
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
    print(f"Mesh: {args.mesh_out}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")

    cloud = volume.extract_point_cloud()
    args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.cloud_out), cloud)
    print(f"Cloud: {args.cloud_out}  ({len(cloud.points)} pts)")

    poses = np.stack([frame.world_to_camera for frame in accepted])
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)
    print(f"Poses: {args.poses_out}  ({len(poses)} poses)")

    write_json(
        args.scan_json,
        {
            "board_json": str(board_json),
            "camera_intrinsics": intrinsics,
            "coordinate_system": "ZED/Open3D image coordinates: +X right, +Y down, +Z forward",
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "roi": list(args.roi),
            "voxel_length_m": args.voxel_length_m,
            "sdf_trunc_m": args.sdf_trunc_m,
            "marker_mask_padding_px": 0 if args.no_mask_markers else args.marker_mask_padding_px,
            "frames": [
                {
                    "index": frame.index,
                    "captured_at_s": frame.captured_at_s,
                    "used_ids": frame.used_ids,
                    "detected_ids": frame.detected_ids,
                    "valid_depth_px": frame.valid_depth_px,
                    "mean_reprojection_error_px": frame.mean_reprojection_error_px,
                    "overlay": frame.overlay_path,
                    **pose_summary(frame.world_to_camera),
                }
                for frame in accepted
            ],
        },
    )
    print(f"Scan JSON: {args.scan_json}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    board = load_board(args.board_json)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    zed = open_zed(args)
    runtime = sl.RuntimeParameters()
    color_mat = sl.Mat()
    depth_mat = sl.Mat()
    volume = make_tsdf_volume(args)
    accepted: list[AcceptedFrame] = []
    intrinsics = None
    open3d_intrinsic = None
    started_at = time.perf_counter()

    if args.save_overlays:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Warming up ({args.warmup} frames)...")
        for _ in range(args.warmup):
            zed.grab(runtime)

        print("ZED ArUco TSDF scanner ready.")
        print("ArUco pose is the alignment source; mechanical angle is not required.")
        while True:
            try:
                if not prompt_for_capture():
                    break
            except KeyboardInterrupt:
                print("\nInterrupted; saving current reconstruction.")
                break

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                print("Grab failed; skipping.")
                continue

            zed.retrieve_image(color_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            color_rgb = color_image_to_rgb(color_mat.get_data())
            raw_depth = depth_mat.get_data()

            if intrinsics is None:
                intrinsics = camera_intrinsics_from_zed(zed, color_rgb.shape[:2])
                open3d_intrinsic = open3d_intrinsic_from_dict(intrinsics)
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
            if not ok:
                print(f"Pose failed; detected ids={ids.reshape(-1).astype(int).tolist()}")
                continue

            depth = clean_depth_image(raw_depth, args.min_depth_m, args.max_depth_m, args.roi)
            if not args.no_mask_markers:
                depth = mask_marker_depth(depth, corners, args.marker_mask_padding_px)
            valid_depth_px = int(np.count_nonzero(depth))
            if valid_depth_px < args.min_valid_depth_px:
                print(
                    f"Skipping frame: valid_depth_px={valid_depth_px} "
                    f"< {args.min_valid_depth_px}"
                )
                continue

            color_for_integration = apply_roi_mask(color_rgb, args.roi, fill_value=0)
            world_to_camera = extrinsic_from_rvec_tvec(rvec, tvec)
            rgbd = make_rgbd(color_for_integration, depth, args.max_depth_m)
            volume.integrate(rgbd, open3d_intrinsic, world_to_camera)

            error_px = reprojection_error_px(
                object_points,
                image_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            overlay_path = None
            if args.save_overlays:
                overlay = draw_pose_overlay(
                    color_rgb,
                    corners,
                    ids,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    args.axis_length_m,
                )
                overlay_path = str(args.debug_dir / f"scan_{len(accepted):04d}.png")
                cv2.imwrite(overlay_path, overlay)

            frame = AcceptedFrame(
                index=len(accepted),
                captured_at_s=time.perf_counter() - started_at,
                used_ids=used_ids,
                detected_ids=ids.reshape(-1).astype(int).tolist(),
                valid_depth_px=valid_depth_px,
                mean_reprojection_error_px=error_px,
                world_to_camera=world_to_camera,
                overlay_path=overlay_path,
            )
            accepted.append(frame)
            print(
                f"KF {frame.index:03d}  markers={used_ids}  "
                f"valid_px={valid_depth_px}  reproj={error_px:.2f}px"
            )
    finally:
        zed.close()

    if intrinsics is None or open3d_intrinsic is None or not accepted:
        print("No accepted frames; no TSDF outputs written.")
        return

    save_outputs(volume, accepted, intrinsics, args.board_json, args)


if __name__ == "__main__":
    main()
