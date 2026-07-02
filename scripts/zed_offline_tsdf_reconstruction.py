#!/usr/bin/env python3
"""Offline ZED RGB-D capture and Open3D TSDF reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pyzed.sl as sl


WARMUP_FRAMES = 20
CAPTURE_SETTLE_FRAMES = 3
ODOMETRY_DEPTH_DIFF_MAX_M = 0.08
MAX_ODOMETRY_STEP_TRANSLATION_M = 0.08
MAX_ODOMETRY_STEP_ROTATION_DEG = 25.0
LOOP_MIN_KEYFRAMES = 12
LOOP_ICP_MIN_FITNESS = 0.35
LOOP_ICP_MAX_RMSE = 0.012
EDGE_PRUNE_THRESHOLD = 0.25


@dataclass
class CapturedFrame:
    index: int
    color: np.ndarray
    depth: np.ndarray
    npz_path: Path
    json_path: Path


@dataclass
class ReconstructionFrame:
    capture: CapturedFrame
    pose: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture ZED RGB-D frames, then reconstruct offline with Open3D TSDF."
    )
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=0.30)
    parser.add_argument("--voxel-length-m", type=float, default=0.001)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.006)
    parser.add_argument("--session-dir", type=Path, default=Path("outputs/offline_foot_session"))
    parser.add_argument("--mesh-out", type=Path, default=Path("outputs/offline_foot_mesh.ply"))
    parser.add_argument("--cloud-out", type=Path, default=Path("outputs/offline_foot_cloud.ply"))
    parser.add_argument("--posegraph-out", type=Path, default=Path("outputs/offline_foot_posegraph.json"))
    parser.add_argument("--poses-out", type=Path, default=Path("outputs/offline_foot_poses.npy"))
    parser.add_argument("--capture-interval-s", type=float, default=0.3)
    parser.add_argument(
        "--resolution",
        choices=["HD720", "HD1080", "HD2K"],
        default="HD720",
    )
    parser.add_argument(
        "--depth-mode",
        choices=["NEURAL_LIGHT", "NEURAL", "NEURAL_PLUS", "ULTRA", "QUALITY"],
        default="NEURAL_LIGHT",
    )
    parser.add_argument("--icp-voxel-m", type=float, default=0.002)
    loop_group = parser.add_mutually_exclusive_group()
    loop_group.add_argument("--loop-closure", dest="loop_closure", action="store_true")
    loop_group.add_argument("--no-loop-closure", dest="loop_closure", action="store_false")
    parser.set_defaults(loop_closure=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m must be greater than --min-depth-m")
    if args.voxel_length_m <= 0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")
    if args.icp_voxel_m <= 0:
        raise ValueError("--icp-voxel-m must be positive")
    if args.capture_interval_s <= 0:
        raise ValueError("--capture-interval-s must be positive")


def resolution_from_name(name: str) -> sl.RESOLUTION:
    return {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K,
    }[name]


def depth_mode_from_name(name: str) -> sl.DEPTH_MODE:
    return {
        "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
        "NEURAL": sl.DEPTH_MODE.NEURAL,
        "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
    }[name]


def color_image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        return image[:, :, :3][:, :, ::-1].astype(np.uint8)
    raise ValueError(f"Unsupported ZED color image shape: {image.shape}")


def clean_depth_image(
    depth_image: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = depth[valid]
    return cleaned


def camera_intrinsics_dict(zed: sl.Camera) -> dict:
    camera_info = zed.get_camera_information()
    calib = camera_info.camera_configuration.calibration_parameters
    left = calib.left_cam
    return {
        "fx": float(left.fx),
        "fy": float(left.fy),
        "cx": float(left.cx),
        "cy": float(left.cy),
    }


def open3d_intrinsic(
    intrinsics: dict,
    image_shape: tuple[int, int],
) -> o3d.camera.PinholeCameraIntrinsic:
    h, w = image_shape
    return o3d.camera.PinholeCameraIntrinsic(
        int(w),
        int(h),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )


def make_rgbd(color: np.ndarray, depth: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def make_odometry_options(args: argparse.Namespace) -> o3d.pipelines.odometry.OdometryOption:
    try:
        return o3d.pipelines.odometry.OdometryOption(
            depth_diff_max=ODOMETRY_DEPTH_DIFF_MAX_M,
            depth_min=args.min_depth_m,
            depth_max=args.max_depth_m,
        )
    except TypeError:
        options = o3d.pipelines.odometry.OdometryOption()
        options.depth_diff_max = ODOMETRY_DEPTH_DIFF_MAX_M
        options.depth_min = args.min_depth_m
        options.depth_max = args.max_depth_m
        return options


def make_tsdf_volume(args: argparse.Namespace) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def pose_delta(prev_pose: np.ndarray, curr_pose: np.ndarray) -> tuple[float, float]:
    rel = np.linalg.inv(prev_pose) @ curr_pose
    translation_m = float(np.linalg.norm(rel[:3, 3]))
    cos_angle = (float(np.trace(rel[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return translation_m, math.degrees(math.acos(cos_angle))


def capture_rgbd(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(CAPTURE_SETTLE_FRAMES):
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            return None

    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    zed.retrieve_image(color_mat, sl.VIEW.LEFT)
    depth = clean_depth_image(depth_mat.get_data(), args.min_depth_m, args.max_depth_m)
    color = color_image_to_rgb(color_mat.get_data())
    return color.astype(np.uint8), depth


def save_capture(
    color: np.ndarray,
    depth: np.ndarray,
    index: int,
    intrinsics: dict,
    args: argparse.Namespace,
) -> CapturedFrame:
    args.session_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.session_dir / f"frame_{index:04d}.npz"
    json_path = args.session_dir / f"frame_{index:04d}.json"
    np.savez_compressed(
        npz_path,
        color=color.astype(np.uint8, copy=False),
        depth=depth.astype(np.float32, copy=False),
    )
    metadata = {
        "index": index,
        "npz": npz_path.name,
        "resolution": args.resolution,
        "depth_mode": args.depth_mode,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "camera_intrinsics": intrinsics,
        "shape": {"height": int(depth.shape[0]), "width": int(depth.shape[1])},
        "valid_depth_pixels": int(np.count_nonzero(depth)),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return CapturedFrame(index, color, depth, npz_path, json_path)


def open_zed(args: argparse.Namespace) -> sl.Camera:
    init = sl.InitParameters()
    init.camera_resolution = resolution_from_name(args.resolution)
    init.depth_mode = depth_mode_from_name(args.depth_mode)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Could not open ZED camera")
    return zed


def capture_session(args: argparse.Namespace) -> tuple[list[CapturedFrame], dict]:
    zed = open_zed(args)
    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()
    color_mat = sl.Mat()
    frames: list[CapturedFrame] = []

    try:
        print(f"Warming up ({WARMUP_FRAMES} frames)...")
        for _ in range(WARMUP_FRAMES):
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("Warmup failed")

        intrinsics = camera_intrinsics_dict(zed)
        session_metadata = {
            "resolution": args.resolution,
            "depth_mode": args.depth_mode,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "capture_interval_s": args.capture_interval_s,
            "camera_intrinsics": intrinsics,
        }
        args.session_dir.mkdir(parents=True, exist_ok=True)
        (args.session_dir / "session.json").write_text(
            json.dumps(session_metadata, indent=2),
            encoding="utf-8",
        )

        print(f"Capturing automatically every {args.capture_interval_s:.2f}s.")
        print("Move smoothly around the object. Type q then Enter to reconstruct.")
        next_capture_at = time.monotonic()
        while True:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                command = sys.stdin.readline().strip().lower()
                if command in {"q", "quit", "done"}:
                    break
                if command:
                    print("Unknown command. Type q then Enter to reconstruct.")

            now = time.monotonic()
            if now < next_capture_at:
                time.sleep(min(0.02, next_capture_at - now))
                continue

            captured = capture_rgbd(zed, runtime, depth_mat, color_mat, args)
            next_capture_at = time.monotonic() + args.capture_interval_s
            if captured is None:
                print("Capture failed; trying next interval.")
                continue

            color, depth = captured
            valid_pixels = int(np.count_nonzero(depth))
            if valid_pixels == 0:
                print("Captured frame has no valid depth; trying next interval.")
                continue

            frame = save_capture(color, depth, len(frames), intrinsics, args)
            frames.append(frame)
            print(f"Saved {frame.npz_path}  valid_depth={valid_pixels}")

    finally:
        zed.close()

    return frames, intrinsics


def point_cloud_for_registration(
    frame: CapturedFrame,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        make_rgbd(frame.color, frame.depth, args.max_depth_m),
        intrinsic,
    )
    cloud = cloud.voxel_down_sample(args.icp_voxel_m)
    if len(cloud.points) > 0:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(args.icp_voxel_m * 4.0, 0.008),
                max_nn=30,
            )
        )
    return cloud


def compute_pose_graph(
    captures: list[CapturedFrame],
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> tuple[list[ReconstructionFrame], o3d.pipelines.registration.PoseGraph]:
    if not captures:
        raise ValueError("No captured frames to reconstruct")

    options = make_odometry_options(args)
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    pg = o3d.pipelines.registration.PoseGraph()
    reconstruction_frames = [
        ReconstructionFrame(captures[0], np.eye(4, dtype=np.float64))
    ]
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(reconstruction_frames[0].pose))

    print("Estimating offline odometry...")
    for candidate in captures[1:]:
        prev = reconstruction_frames[-1]
        success, transform, information = o3d.pipelines.odometry.compute_rgbd_odometry(
            make_rgbd(prev.capture.color, prev.capture.depth, args.max_depth_m),
            make_rgbd(candidate.color, candidate.depth, args.max_depth_m),
            intrinsic,
            np.eye(4, dtype=np.float64),
            jacobian,
            options,
        )
        if not success:
            print(f"  skip frame {candidate.index}: odometry failed")
            continue

        pose = prev.pose @ np.linalg.inv(transform)
        move_m, rot_deg = pose_delta(prev.pose, pose)
        if move_m > MAX_ODOMETRY_STEP_TRANSLATION_M or rot_deg > MAX_ODOMETRY_STEP_ROTATION_DEG:
            print(
                f"  skip frame {candidate.index}: jump {move_m:.4f}m / {rot_deg:.2f}deg"
            )
            continue

        target_index = len(reconstruction_frames)
        reconstruction_frames.append(ReconstructionFrame(candidate, pose))
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
        pg.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                target_index - 1,
                target_index,
                transform,
                information,
                uncertain=False,
            )
        )
        print(
            f"  edge {target_index - 1}->{target_index}  "
            f"capture={candidate.index}  move={move_m:.4f}m  rot={rot_deg:.2f}deg"
        )

    return reconstruction_frames, pg


def run_loop_closure(
    frames: list[ReconstructionFrame],
    pg: o3d.pipelines.registration.PoseGraph,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> None:
    if not args.loop_closure or len(frames) < LOOP_MIN_KEYFRAMES:
        return

    print("Trying loop closure: first frame -> final frame")
    first = frames[0]
    last = frames[-1]
    source = point_cloud_for_registration(first.capture, intrinsic, args)
    target = point_cloud_for_registration(last.capture, intrinsic, args)
    if len(source.points) < 100 or len(target.points) < 100:
        print("  loop rejected: not enough registration points")
        return

    init = np.linalg.inv(last.pose) @ first.pose
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
    coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        args.icp_voxel_m * 20.0,
        init,
        estimation,
        criteria,
    )
    fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        args.icp_voxel_m * 8.0,
        coarse.transformation,
        estimation,
        criteria,
    )
    fitness = float(fine.fitness)
    rmse = float(fine.inlier_rmse)
    if fitness < LOOP_ICP_MIN_FITNESS or rmse > LOOP_ICP_MAX_RMSE:
        print(f"  loop rejected  fit={fitness:.3f}  rmse={rmse:.4f}")
        return

    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        args.icp_voxel_m * 8.0,
        fine.transformation,
    )
    pg.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(
            0,
            len(frames) - 1,
            fine.transformation,
            information,
            uncertain=True,
        )
    )
    print(f"  loop accepted  fit={fitness:.3f}  rmse={rmse:.4f}")


def optimize_pose_graph(pg: o3d.pipelines.registration.PoseGraph, args: argparse.Namespace) -> None:
    if len(pg.nodes) < 2:
        return
    print("Optimizing pose graph...")
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.icp_voxel_m * 8.0,
            edge_prune_threshold=EDGE_PRUNE_THRESHOLD,
            reference_node=0,
        ),
    )


def integrate_frames(
    frames: list[ReconstructionFrame],
    pg: o3d.pipelines.registration.PoseGraph,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> None:
    print("Integrating RGB-D frames into final TSDF...")
    volume = make_tsdf_volume(args)
    for i, (frame, node) in enumerate(zip(frames, pg.nodes), start=1):
        frame.pose = np.asarray(node.pose, dtype=np.float64)
        rgbd = make_rgbd(frame.capture.color, frame.capture.depth, args.max_depth_m)
        # Matches Open3D's documented reconstruction-system pattern:
        # integrate each RGB-D frame with the inverse camera pose.
        # https://www.open3d.org/docs/latest/tutorial/reconstruction_system/integrate_scene.html#integrate-rgbd-frames
        volume.integrate(rgbd, intrinsic, np.linalg.inv(frame.pose))
        print(f"  {i}/{len(frames)}", end="\r")
    print()

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
    print(f"Mesh: {args.mesh_out}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")

    cloud = volume.extract_point_cloud()
    args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.cloud_out), cloud)
    print(f"Cloud: {args.cloud_out}  ({len(cloud.points)} pts)")

    args.posegraph_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_pose_graph(str(args.posegraph_out), pg)
    print(f"PoseGraph: {args.posegraph_out}")

    poses = np.stack([np.asarray(node.pose, dtype=np.float64) for node in pg.nodes])
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)
    print(f"Poses: {args.poses_out}  ({len(poses)} poses)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    captures, intrinsics = capture_session(args)
    if len(captures) < 2:
        print("Need at least two captured frames for reconstruction.")
        return

    intrinsic = open3d_intrinsic(intrinsics, captures[0].depth.shape)
    frames, pg = compute_pose_graph(captures, intrinsic, args)
    if len(frames) < 2:
        print("Not enough odometry-linked frames for reconstruction.")
        return

    run_loop_closure(frames, pg, intrinsic, args)
    print(f"{len(frames)} reconstruction frames, {len(pg.edges)} pose graph edges.")
    optimize_pose_graph(pg, args)
    integrate_frames(frames, pg, intrinsic, args)


if __name__ == "__main__":
    main()
