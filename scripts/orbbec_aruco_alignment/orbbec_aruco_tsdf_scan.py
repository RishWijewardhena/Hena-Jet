#!/usr/bin/env python3
"""Orbbec RGB-D TSDF scan aligned by a printed ArUco table board."""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from aruco_common import (
    apply_roi_mask,
    camera_matrix_from_intrinsics,
    camera_params_from_orbbec,
    clean_depth_image,
    color_frame_to_rgb,
    depth_frame_to_meters,
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

STOP_COMMANDS = {"q", "quit", "exit", "done"}


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
    parser = argparse.ArgumentParser(description="Orbbec ArUco table-board TSDF scanner.")
    parser.add_argument(
        "--board-json",
        type=Path,
        default=Path("outputs/aruco_table_board/aruco_table_board.json"),
        help="Path to generated ArUco board JSON specification.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Stream width.")
    parser.add_argument("--height", type=int, default=800, help="Stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream framerate.")
    parser.add_argument(
        "--hw-d2c",
        action="store_true",
        help="Use hardware Depth-to-Color alignment instead of software AlignFilter.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=1.00)
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
    parser.add_argument(
        "--hole-fill-size-m",
        type=float,
        default=0.003,
        help="Maximum hole size to fill in the extracted mesh, in meters.",
    )
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip final statistical point cleanup and small mesh fragment removal.",
    )
    parser.add_argument(
        "--cleanup-outlier-neighbors",
        type=int,
        default=20,
        help="Neighbor count used by statistical outlier removal on the final point cloud.",
    )
    parser.add_argument(
        "--cleanup-outlier-std-ratio",
        type=float,
        default=2.0,
        help="Std-dev ratio used by statistical outlier removal on the final point cloud.",
    )
    parser.add_argument(
        "--cleanup-min-cluster-fraction",
        type=float,
        default=0.02,
        help="Remove mesh triangle clusters smaller than this fraction of the largest cluster.",
    )
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--marker-mask-padding-px", type=int, default=8)
    parser.add_argument("--no-mask-markers", action="store_true")
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("outputs/aruco_scans/orbbec_aruco_debug"),
    )
    parser.add_argument(
        "--mesh-out",
        type=Path,
        default=Path("outputs/aruco_scans/orbbec_aruco_tsdf_mesh.ply"),
    )
    parser.add_argument(
        "--cloud-out",
        type=Path,
        default=Path("outputs/aruco_scans/orbbec_aruco_tsdf_cloud.ply"),
    )
    parser.add_argument(
        "--poses-out",
        type=Path,
        default=Path("outputs/aruco_scans/orbbec_aruco_poses.npy"),
    )
    parser.add_argument(
        "--scan-json",
        type=Path,
        default=Path("outputs/aruco_scans/orbbec_aruco_scan.json"),
    )
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        help="Automatically capture frames without user input.",
    )
    parser.add_argument(
        "--auto-capture-interval-s",
        type=float,
        default=0.5,
        help="Seconds to wait between capture attempts when --auto-capture is enabled.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many accepted frames. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="Frame wait timeout in milliseconds.",
    )
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
    if args.hole_fill_size_m <= 0.0:
        raise ValueError("--hole-fill-size-m must be positive")
    if args.cleanup_outlier_neighbors <= 0:
        raise ValueError("--cleanup-outlier-neighbors must be positive")
    if args.cleanup_outlier_std_ratio <= 0.0:
        raise ValueError("--cleanup-outlier-std-ratio must be positive")
    if not 0.0 < args.cleanup_min_cluster_fraction <= 1.0:
        raise ValueError("--cleanup-min-cluster-fraction must be between 0 and 1")
    if args.marker_mask_padding_px < 0:
        raise ValueError("--marker-mask-padding-px must be zero or positive")
    if args.auto_capture_interval_s <= 0.0:
        raise ValueError("--auto-capture-interval-s must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be zero or positive")


def open_orbbec_pipeline(args: argparse.Namespace):
    try:
        from pyorbbecsdk import (  # type: ignore
            AlignFilter,
            Config,
            Context,
            OBAlignMode,
            OBFormat,
            OBFrameAggregateOutputMode,
            OBSensorType,
            OBStreamType,
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

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = None
    for fmt in (OBFormat.RGB, OBFormat.MJPG, OBFormat.BGR):
        try:
            color_profile = color_profiles.get_video_stream_profile(
                args.width, args.height, fmt, args.fps
            )
            if color_profile is not None:
                break
        except Exception:
            continue

    if color_profile is None:
        try:
            color_profile = color_profiles.get_default_video_stream_profile()
        except Exception as exc:
            raise SystemExit(f"Failed to get color stream profile: {exc}") from exc

    align_filter = None
    if args.hw_d2c:
        try:
            hw_depth_profiles = pipeline.get_d2c_depth_profile_list(
                color_profile, OBAlignMode.HW_MODE
            )
            if len(hw_depth_profiles) > 0:
                config.enable_stream(hw_depth_profiles[0])
                config.enable_stream(color_profile)
                config.set_align_mode(OBAlignMode.HW_MODE)
                print("Hardware Depth-to-Color alignment enabled.")
            else:
                raise RuntimeError("No matching hardware D2C depth profiles found.")
        except Exception as exc:
            print(f"HW D2C setup failed ({exc}); falling back to software AlignFilter.")
            args.hw_d2c = False

    if not args.hw_d2c:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = None
        try:
            depth_profile = depth_profiles.get_video_stream_profile(
                args.width, args.height, OBFormat.Y16, args.fps
            )
        except Exception:
            try:
                depth_profile = depth_profiles.get_default_video_stream_profile()
            except Exception as exc:
                raise SystemExit(f"Failed to get depth stream profile: {exc}") from exc

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        print("Software Depth-to-Color alignment (AlignFilter) enabled.")

    try:
        pipeline.enable_frame_sync()
    except Exception as exc:
        print(f"Hardware frame synchronization warning: {exc}")

    pipeline.start(config)
    return pipeline, align_filter


def make_tsdf_volume(args: argparse.Namespace) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def prompt_for_capture() -> bool:
    raw_value = input("Press Enter to capture, or q to finish: ").strip().lower()
    return raw_value not in STOP_COMMANDS


def fill_mesh_holes(
    mesh: o3d.geometry.TriangleMesh,
    hole_size_m: float,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    return tensor_mesh.fill_holes(hole_size=hole_size_m).to_legacy()


def clean_mesh_fragments(
    mesh: o3d.geometry.TriangleMesh,
    min_cluster_fraction: float,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    keep_threshold = max(1, int(cluster_n_triangles.max() * min_cluster_fraction))
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < keep_threshold
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def clean_point_cloud(
    cloud: o3d.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0:
        return cloud
    cleaned, _ = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return cleaned


def save_outputs(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    accepted: list[AcceptedFrame],
    intrinsics: dict[str, float],
    board_json: Path,
    args: argparse.Namespace,
) -> None:
    mesh = volume.extract_triangle_mesh()
    if not args.no_fill_holes:
        before = len(mesh.triangles)
        mesh = fill_mesh_holes(mesh, args.hole_fill_size_m)
        print(
            f"Filled mesh holes up to {args.hole_fill_size_m:.4f} m "
            f"({before} -> {len(mesh.triangles)} tris)"
        )
    if not args.no_cleanup:
        before = len(mesh.triangles)
        mesh = clean_mesh_fragments(mesh, args.cleanup_min_cluster_fraction)
        print(f"Removed small mesh fragments ({before} -> {len(mesh.triangles)} tris)")
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
    print(f"Mesh: {args.mesh_out}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")

    cloud = volume.extract_point_cloud()
    if not args.no_cleanup:
        before = len(cloud.points)
        cloud = clean_point_cloud(
            cloud,
            args.cleanup_outlier_neighbors,
            args.cleanup_outlier_std_ratio,
        )
        print(f"Removed statistical point outliers ({before} -> {len(cloud.points)} pts)")
    args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.cloud_out), cloud)
    print(f"Cloud: {args.cloud_out}  ({len(cloud.points)} pts)")

    if len(accepted) > 0:
        poses = np.stack([frame.world_to_camera for frame in accepted])
    else:
        poses = np.empty((0, 4, 4), dtype=np.float64)
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)
    print(f"Poses: {args.poses_out}  ({len(poses)} poses)")

    write_json(
        args.scan_json,
        {
            "board_json": str(board_json),
            "camera_intrinsics": intrinsics,
            "coordinate_system": "Orbbec/Open3D image coordinates: +X right, +Y down, +Z forward",
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "roi": list(args.roi),
            "voxel_length_m": args.voxel_length_m,
            "sdf_trunc_m": args.sdf_trunc_m,
            "hole_fill_size_m": None if args.no_fill_holes else args.hole_fill_size_m,
            "cleanup": None
            if args.no_cleanup
            else {
                "outlier_neighbors": args.cleanup_outlier_neighbors,
                "outlier_std_ratio": args.cleanup_outlier_std_ratio,
                "min_cluster_fraction": args.cleanup_min_cluster_fraction,
            },
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

    pipeline, align_filter = open_orbbec_pipeline(args)
    volume = make_tsdf_volume(args)
    accepted: list[AcceptedFrame] = []
    intrinsics = None
    open3d_intrinsic = None
    dist_coeffs = None
    camera_matrix = None
    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        if not stop_requested:
            print("\nCtrl+C received; saving current reconstruction...")
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)

    if args.save_overlays:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Warming up ({args.warmup} frames)...")
        for _ in range(args.warmup):
            frames = pipeline.wait_for_frames(args.timeout_ms)
            if frames is None:
                raise RuntimeError("Timeout while waiting for camera frames during warmup")

        camera_param = pipeline.get_camera_param()
        intrinsics, dist_coeffs = camera_params_from_orbbec(camera_param)
        camera_matrix = camera_matrix_from_intrinsics(intrinsics)
        open3d_intrinsic = open3d_intrinsic_from_dict(intrinsics)

        print("\nOrbbec ArUco TSDF scanner ready.")
        print(f"Stream: {intrinsics['width']}x{intrinsics['height']}, fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        print("ArUco table board provides the global 3D world coordinate frame.")
        if args.auto_capture:
            print(
                f"Auto-capture enabled. Interval: {args.auto_capture_interval_s:.2f}s."
            )
            print("Press Ctrl+C to stop and save.")
            if args.max_frames:
                print(f"Stopping after {args.max_frames} accepted frames.")

        frame_attempt = 0
        while True:
            try:
                if stop_requested:
                    break
                if args.auto_capture:
                    if args.max_frames and len(accepted) >= args.max_frames:
                        print(f"Reached --max-frames={args.max_frames}; saving reconstruction.")
                        break
                    time.sleep(args.auto_capture_interval_s)
                elif not prompt_for_capture():
                    break
            except KeyboardInterrupt:
                print("\nInterrupted; saving current reconstruction.")
                break

            if stop_requested:
                break

            frames = pipeline.wait_for_frames(args.timeout_ms)
            if frames is None:
                print(f"Attempt {frame_attempt}: timeout waiting for frames")
                frame_attempt += 1
                continue

            if align_filter is not None:
                aligned_frames = align_filter.process(frames)
                if aligned_frames is not None:
                    frames = aligned_frames

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                print(f"Attempt {frame_attempt}: incomplete frame pair (color/depth missing)")
                frame_attempt += 1
                continue

            color_rgb = color_frame_to_rgb(color_frame)
            depth_m = depth_frame_to_meters(depth_frame)

            corners, ids, _ = detect_markers(color_rgb, board)
            object_points, image_points, used_ids = match_board_corners(board, corners, ids)
            ok, rvec, tvec = solve_board_pose(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                args.min_markers,
                used_ids,
            )

            detected_flat = ids.reshape(-1).tolist() if ids is not None else []
            if not ok:
                print(
                    f"Attempt {frame_attempt}: rejected (insufficient marker matches: "
                    f"detected={detected_flat}, required>={args.min_markers})"
                )
                frame_attempt += 1
                continue

            world_to_camera = extrinsic_from_rvec_tvec(rvec, tvec)
            error_px = reprojection_error_px(
                object_points,
                image_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )

            # Process depth
            proc_depth = depth_m.copy()
            if not args.no_mask_markers and len(corners) > 0:
                proc_depth = mask_marker_depth(
                    proc_depth,
                    corners,
                    args.marker_mask_padding_px,
                )

            cleaned_depth = clean_depth_image(
                proc_depth,
                args.min_depth_m,
                args.max_depth_m,
                args.roi,
            )

            valid_px = int(np.count_nonzero(cleaned_depth > 0))
            if valid_px < args.min_valid_depth_px:
                print(
                    f"Attempt {frame_attempt}: rejected (valid depth px {valid_px} "
                    f"< threshold {args.min_valid_depth_px})"
                )
                frame_attempt += 1
                continue

            rgbd = make_rgbd(color_rgb, cleaned_depth, depth_trunc_m=args.max_depth_m)
            # ScalableTSDFVolume integrate expects world_to_camera extrinsic
            volume.integrate(rgbd, open3d_intrinsic, world_to_camera)

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
                overlay_file = args.debug_dir / f"frame_{len(accepted):04d}.png"
                cv2.imwrite(str(overlay_file), overlay)
                overlay_path = str(overlay_file)

            accepted_frame = AcceptedFrame(
                index=len(accepted),
                captured_at_s=time.time(),
                used_ids=used_ids,
                detected_ids=detected_flat,
                valid_depth_px=valid_px,
                mean_reprojection_error_px=error_px,
                world_to_camera=world_to_camera,
                overlay_path=overlay_path,
            )
            accepted.append(accepted_frame)
            print(
                f"Accepted frame {accepted_frame.index:03d} | "
                f"markers={used_ids} | valid_depth={valid_px}px | "
                f"reproj={error_px:.2f}px | total_accepted={len(accepted)}"
            )
            frame_attempt += 1

        if intrinsics is not None:
            print("\nSaving reconstruction outputs...")
            save_outputs(volume, accepted, intrinsics, args.board_json, args)
        else:
            print("No frames captured; exiting without saving.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
