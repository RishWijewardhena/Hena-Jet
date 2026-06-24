#!/usr/bin/env python3
"""Live ZED RGB-D mapping with Open3D PoseGraph and TSDF fusion."""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pyzed.sl as sl


STAGE_NAMES = {
    "1": "outer_circle",
    "2": "translation",
    "3": "inner_circle",
}

# FIX: coordinate system flip from ZED IMAGE convention to Open3D convention.
# ZED IMAGE:  +X right, +Y DOWN,  +Z forward
# Open3D:     +X right, +Y UP,    +Z backward
# Without this the mesh Y axis is inverted and the object appears twisted.
ZED_IMAGE_TO_OPEN3D = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


@dataclass
class Keyframe:
    index: int
    frame_index: int
    stage: str
    pose: np.ndarray
    packet: bytes


@dataclass
class ScanControl:
    stage: str = "outer_circle"
    stop_requested: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live ZED M RGB-D TSDF mapper with Open3D PoseGraph."
    )
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=0.35)
    parser.add_argument("--roi", type=float, nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=(0.0, 0.0, 1.0, 1.0))
    parser.add_argument("--resolution",
        choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD720")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--scan-mode",
        choices=["single-circle", "guided"], default="single-circle")
    parser.add_argument("--no-auto-stop-on-loop", action="store_true")
    parser.add_argument("--keyframe-every-n", type=int, default=2)
    parser.add_argument("--min-keyframe-translation-m", type=float, default=0.003)
    parser.add_argument("--min-keyframe-rotation-deg", type=float, default=0.8)
    parser.add_argument("--max-keyframe-translation-m", type=float, default=0.04)
    parser.add_argument("--max-keyframe-rotation-deg", type=float, default=12.0)
    parser.add_argument("--tracking-loss-policy",
        choices=["stop", "skip"], default="skip")
    parser.add_argument("--voxel-length-m", type=float, default=0.002)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.010)
    parser.add_argument("--preview", choices=["cloud", "mesh"], default="cloud")
    parser.add_argument("--viz-every-n-keyframes", type=int, default=5)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--loop-closure-every-n", type=int, default=15)
    parser.add_argument("--loop-closure-min-separation", type=int, default=30)
    parser.add_argument("--loop-closure-search-radius-m", type=float, default=0.20)
    parser.add_argument("--max-loop-closure-candidates", type=int, default=4)
    parser.add_argument("--icp-voxel-m", type=float, default=0.004)
    # Loop closure ICP uses a wider correspondence distance than odometry
    # because pose drift means the initial alignment may be off by several cm.
    parser.add_argument("--icp-max-correspondence-m", type=float, default=0.04)
    parser.add_argument("--icp-min-fitness", type=float, default=0.25)
    parser.add_argument("--icp-max-rmse", type=float, default=0.015)
    # Correction gates are now bypassed for fit>=0.95, so these only
    # block low-confidence corrections that might be spurious matches.
    parser.add_argument("--loop-max-correction-translation-m", type=float, default=0.08)
    parser.add_argument("--loop-max-correction-rotation-deg", type=float, default=20.0)
    # FIX: raised default from 0.03 to 0.08 — 3cm rejected all depth edges at
    # cylinder boundaries, leaving almost no valid odometry pixels.
    parser.add_argument("--odometry-depth-diff-max-m", type=float, default=0.08)
    parser.add_argument("--max-odometry-failures", type=int, default=20)
    parser.add_argument("--edge-prune-threshold", type=float, default=0.25)
    parser.add_argument("--mesh-out", type=Path,
        default=Path("outputs/zed_live_posegraph_tsdf_mesh.ply"))
    parser.add_argument("--cloud-out", type=Path,
        default=Path("outputs/zed_live_posegraph_tsdf_cloud.ply"))
    parser.add_argument("--posegraph-out", type=Path,
        default=Path("outputs/zed_live_posegraph.json"))
    parser.add_argument("--poses-out", type=Path,
        default=Path("outputs/zed_live_poses.npy"))
    return parser.parse_args()


def resolution_from_name(name: str) -> sl.RESOLUTION:
    return {"HD2K": sl.RESOLUTION.HD2K, "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720, "VGA": sl.RESOLUTION.VGA}[name]


def color_image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        return image[:, :, :3][:, :, ::-1].astype(np.uint8)
    raise ValueError(f"Unsupported ZED color image shape: {image.shape}")


def validate_roi(roi: tuple) -> tuple:
    x_min, y_min, x_max, y_max = roi
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError("--roi must satisfy 0<=X_MIN<X_MAX<=1 and 0<=Y_MIN<Y_MAX<=1")
    return x_min, y_min, x_max, y_max


def apply_roi_mask(
    image: np.ndarray,
    roi: tuple,
    fill_value: float | int = 0,
) -> np.ndarray:
    """Zero pixels outside the ROI. Applied identically to depth AND colour.

    FIX: original code cropped depth but left colour untouched. Mismatched
    active regions cause wrong odometry transforms and fuse background colour
    from outside the ROI into the object geometry.
    """
    if tuple(roi) == (0.0, 0.0, 1.0, 1.0):
        return image
    x_min, y_min, x_max, y_max = roi
    h, w = image.shape[:2]
    x0 = int(round(x_min * w))
    x1 = int(round(x_max * w))
    y0 = int(round(y_min * h))
    y1 = int(round(y_max * h))
    result = np.full_like(image, fill_value)
    result[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    return result


def clean_depth_image(
    depth_image: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    roi: tuple,
) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = depth[valid]
    return apply_roi_mask(cleaned, roi, fill_value=0)


def camera_intrinsic_from_zed(
    zed: sl.Camera,
    image_shape: tuple,
) -> o3d.camera.PinholeCameraIntrinsic:
    camera_info = zed.get_camera_information()
    calib = camera_info.camera_configuration.calibration_parameters
    lc = calib.left_cam
    h, w = image_shape
    return o3d.camera.PinholeCameraIntrinsic(
        int(w), int(h), float(lc.fx), float(lc.fy), float(lc.cx), float(lc.cy))


def make_rgbd(color: np.ndarray, depth: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def encode_keyframe(color: np.ndarray, depth: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf,
        color=color.astype(np.uint8, copy=False),
        depth=depth.astype(np.float32, copy=False))
    return buf.getvalue()


def decode_keyframe(packet: bytes) -> tuple[np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(packet)) as data:
        return data["color"].astype(np.uint8, copy=True), data["depth"].astype(np.float32, copy=True)


def keyframe_rgbd(kf: Keyframe, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    color, depth = decode_keyframe(kf.packet)
    return make_rgbd(color, depth, depth_trunc_m)


def make_tsdf_volume(args: argparse.Namespace) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def pose_delta(prev_pose: np.ndarray, curr_pose: np.ndarray) -> tuple[float, float]:
    rel = np.linalg.inv(prev_pose) @ curr_pose
    t = float(np.linalg.norm(rel[:3, 3]))
    cos_a = max(-1.0, min(1.0, (float(np.trace(rel[:3, :3])) - 1.0) / 2.0))
    return t, math.degrees(math.acos(cos_a))


def point_cloud_for_registration(
    kf: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    rgbd = keyframe_rgbd(kf, args.max_depth_m)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    if args.icp_voxel_m > 0:
        cloud = cloud.voxel_down_sample(args.icp_voxel_m)
    if len(cloud.points) > 0:
        nr = max(args.icp_max_correspondence_m * 2.0, args.icp_voxel_m * 3.0)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=30))
    return cloud


def point_cloud_for_preview(
    kf: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    rgbd = keyframe_rgbd(kf, args.max_depth_m)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    if args.voxel_length_m > 0:
        cloud = cloud.voxel_down_sample(args.voxel_length_m)
    cloud.transform(kf.pose)
    return cloud


def run_loop_closure_icp(
    source: Keyframe, target: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> tuple[bool, np.ndarray, np.ndarray, float, float]:
    """Multi-scale coarse-to-fine ICP for loop closure.

    A coarse pass at 4x correspondence distance handles large pose drift
    (which causes fitness=0.000 at tight distances). A fine pass then
    refines the result. This is the standard approach when the initial
    pose estimate has significant accumulated drift.
    """
    sc = point_cloud_for_registration(source, intrinsic, args)
    tc = point_cloud_for_registration(target, intrinsic, args)
    if len(sc.points) < 100 or len(tc.points) < 100:
        return False, np.eye(4), np.eye(6), 0.0, float("inf")

    init = np.linalg.inv(target.pose) @ source.pose
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)

    # Coarse pass: wide correspondence distance to bridge pose drift gap
    result_coarse = o3d.pipelines.registration.registration_icp(
        sc, tc, args.icp_max_correspondence_m * 4.0, init, estimation, criteria)

    # Fine pass: tighten from coarse result
    result = o3d.pipelines.registration.registration_icp(
        sc, tc, args.icp_max_correspondence_m,
        result_coarse.transformation, estimation, criteria)

    fitness = float(result.fitness)
    rmse = float(result.inlier_rmse)

    # Quality gates
    quality_ok = fitness >= args.icp_min_fitness
    if args.icp_max_rmse > 0:
        quality_ok = quality_ok and rmse <= args.icp_max_rmse

    # Correction size gates — how much the ICP moved from the initial guess
    ct, cr = pose_delta(init, result.transformation)
    correction_ok = True
    if args.loop_max_correction_translation_m > 0:
        correction_ok = correction_ok and ct <= args.loop_max_correction_translation_m
    if args.loop_max_correction_rotation_deg > 0:
        correction_ok = correction_ok and cr <= args.loop_max_correction_rotation_deg

    # CRITICAL FIX: fit=1.000 means every point found a match — ICP is certain.
    # When fitness is near-perfect, the correction gate is irrelevant because
    # the alignment is geometrically correct regardless of how large the drift
    # was. Bypassing the correction gate for high-confidence results allows
    # loop closures that were previously rejected with fit=1.000 rmse=0.003
    # to be accepted and pull the pose graph into alignment.
    high_confidence = fitness >= 0.95 and (args.icp_max_rmse <= 0 or rmse <= args.icp_max_rmse * 1.5)

    ok = quality_ok and (correction_ok or high_confidence)
    if not ok:
        return False, result.transformation, np.eye(6), fitness, rmse
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        sc, tc, args.icp_max_correspondence_m, result.transformation)
    return True, result.transformation, info, float(result.fitness), float(result.inlier_rmse)


def select_loop_candidates(
    keyframes: list[Keyframe], current: Keyframe, args: argparse.Namespace,
) -> list[Keyframe]:
    candidates = []
    for prev in keyframes[:-1]:
        if current.index - prev.index < args.loop_closure_min_separation:
            continue
        same_ring = current.stage == prev.stage and current.stage != "translation"
        cross_ring = {current.stage, prev.stage} == {"outer_circle", "inner_circle"}
        if not same_ring and not cross_ring:
            continue
        d = float(np.linalg.norm(current.pose[:3, 3] - prev.pose[:3, 3]))
        if d <= args.loop_closure_search_radius_m:
            candidates.append((d, prev))
    candidates.sort(key=lambda x: x[0])
    return [x[1] for x in candidates[:args.max_loop_closure_candidates]]


class LivePreview:
    def __init__(self, control: ScanControl, preview: str, scan_mode: str) -> None:
        self.control = control
        self.preview = preview
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        name = ("ZED Live TSDF PoseGraph  [1 outer | 2 translation | 3 inner | Q save]"
                if scan_mode == "guided" else
                "ZED Live TSDF PoseGraph  [single circle | Q save]")
        self.vis.create_window(window_name=name, width=1280, height=720)
        self.geometry = None
        self._has_reset_view = False
        self._setup_callbacks(scan_mode)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 1.5

    def _setup_callbacks(self, scan_mode: str) -> None:
        def make_stage(s):
            def cb(_): self.control.stage = s; print(f"\nStage: {s}"); return False
            return cb
        if scan_mode == "guided":
            self.vis.register_key_callback(ord("1"), make_stage("outer_circle"))
            self.vis.register_key_callback(ord("2"), make_stage("translation"))
            self.vis.register_key_callback(ord("3"), make_stage("inner_circle"))
        def stop(_): self.control.stop_requested = True; print("\nStopping..."); return False
        self.vis.register_key_callback(ord("Q"), stop)
        self.vis.register_key_callback(ord("q"), stop)

    def is_open(self) -> bool:
        return self.vis.poll_events()

    @staticmethod
    def _size(g) -> int:
        if isinstance(g, o3d.geometry.PointCloud): return len(g.points)
        if isinstance(g, o3d.geometry.TriangleMesh): return len(g.vertices)
        return 0

    def update_geometry(self, g) -> None:
        if self._size(g) == 0:
            self.vis.poll_events(); self.vis.update_renderer(); return
        reset = not self._has_reset_view
        if self.geometry is not None:
            self.vis.remove_geometry(self.geometry, reset_bounding_box=False)
        self.geometry = g
        self.vis.add_geometry(g, reset_bounding_box=reset)
        self._has_reset_view = True
        self.vis.poll_events(); self.vis.update_renderer()

    def update(self, volume) -> bool:
        g = (volume.extract_triangle_mesh() if self.preview == "mesh"
             else volume.extract_point_cloud())
        if self.preview == "mesh":
            g.compute_vertex_normals()
        if self._size(g) == 0:
            self.vis.poll_events(); self.vis.update_renderer(); return False
        self.update_geometry(g)
        return True

    def destroy(self) -> None:
        self.vis.destroy_window()


def integrate_keyframe(
    volume, kf: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> None:
    rgbd = keyframe_rgbd(kf, args.max_depth_m)
    # FIX: apply coordinate flip so TSDF receives extrinsics in Open3D convention.
    # ZED IMAGE has +Y down / +Z forward; Open3D expects +Y up / +Z backward.
    extrinsic = np.linalg.inv(kf.pose) @ ZED_IMAGE_TO_OPEN3D
    volume.integrate(rgbd, intrinsic, extrinsic)


def optimize_pose_graph(pg, args: argparse.Namespace) -> None:
    if len(pg.nodes) < 2:
        return
    print("Optimizing pose graph...")
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.icp_max_correspondence_m,
            edge_prune_threshold=args.edge_prune_threshold,
            reference_node=0,
        ),
    )


def save_outputs(keyframes, pg, intrinsic, args) -> None:
    for kf, node in zip(keyframes, pg.nodes):
        kf.pose = np.asarray(node.pose, dtype=np.float64)
    print("Rebuilding final TSDF from optimized poses...")
    vol = make_tsdf_volume(args)
    for i, kf in enumerate(keyframes):
        integrate_keyframe(vol, kf, intrinsic, args)
        print(f"  {i+1}/{len(keyframes)}", end="\r")
    print()
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
    print(f"Mesh: {args.mesh_out}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")
    if args.cloud_out:
        cloud = vol.extract_point_cloud()
        args.cloud_out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.cloud_out), cloud)
        print(f"Cloud: {args.cloud_out}  ({len(cloud.points)} pts)")
    args.posegraph_out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_pose_graph(str(args.posegraph_out), pg)
    poses = np.stack([np.asarray(n.pose, dtype=np.float64) for n in pg.nodes])
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)
    print(f"Poses: {args.poses_out}  ({len(poses)} poses)")


def main() -> None:
    args = parse_args()
    args.roi = validate_roi(tuple(args.roi))
    if args.keyframe_every_n <= 0:
        raise ValueError("--keyframe-every-n must be positive")
    if args.voxel_length_m <= 0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be > --voxel-length-m")

    zed_init = sl.InitParameters()
    zed_init.camera_resolution = resolution_from_name(args.resolution)
    zed_init.depth_mode = sl.DEPTH_MODE.NEURAL
    zed_init.coordinate_units = sl.UNIT.METER
    zed_init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    zed_init.depth_minimum_distance = args.min_depth_m
    zed_init.depth_maximum_distance = args.max_depth_m
    zed = sl.Camera()
    if zed.open(zed_init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Could not open ZED camera")

    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()
    color_mat = sl.Mat()
    intrinsic = None
    odo_opt = o3d.pipelines.odometry.OdometryOption()
    odo_opt.depth_diff_max = args.odometry_depth_diff_max_m
    odo_opt.depth_min = args.min_depth_m
    odo_opt.depth_max = args.max_depth_m
    odo_jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    control = ScanControl()
    preview = None if args.no_viz else LivePreview(control, args.preview, args.scan_mode)

    keyframes: list[Keyframe] = []
    pg = o3d.pipelines.registration.PoseGraph()
    preview_vol = make_tsdf_volume(args)
    consecutive_failures = 0
    frame_index = 0
    skipped_frames = 0   # silent low-depth counter

    print(f"Warming up ({args.warmup} frames)...")
    try:
        for i in range(args.warmup):
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Warmup frame {i+1} failed")

        if args.scan_mode == "guided":
            print("Controls: 1=outer 2=translation 3=inner Q=save")
        else:
            print("Single-circle: orbit once, auto-stops on loop closure. Q saves early.")
        print("Mapping started.")

        while True:
            if preview is not None and not preview.is_open():
                print("\nViewer closed.")
                break
            if control.stop_requested:
                break

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            frame_index += 1
            if frame_index % args.keyframe_every_n != 0 and keyframes:
                continue

            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            zed.retrieve_image(color_mat, sl.VIEW.LEFT)

            depth = clean_depth_image(
                depth_mat.get_data(), args.min_depth_m, args.max_depth_m, args.roi)

            # FIX: apply same ROI mask to colour as to depth
            color = apply_roi_mask(
                color_image_to_rgb(color_mat.get_data()), args.roi, fill_value=0
            ).astype(np.uint8)

            if intrinsic is None:
                intrinsic = camera_intrinsic_from_zed(zed, depth.shape)

            # ── silent sparse-frame handling ──────────────────────────────────
            valid_px = int(np.count_nonzero(depth))
            if valid_px < 500:
                skipped_frames += 1
                if skipped_frames % 30 == 0:
                    print(f"  [{skipped_frames} low-depth frames skipped — "
                          f"move closer or widen --roi]")
                continue
            if skipped_frames >= 5:
                print(f"  [recovered after {skipped_frames} skipped frames]")
            skipped_frames = 0

            rgbd = make_rgbd(color, depth, args.max_depth_m)

            if not keyframes:
                pose = np.eye(4, dtype=np.float64)
                kf = Keyframe(0, frame_index, control.stage, pose, encode_keyframe(color, depth))
                keyframes.append(kf)
                pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
                integrate_keyframe(preview_vol, kf, intrinsic, args)
                if preview is not None:
                    if not preview.update(preview_vol):
                        preview.update_geometry(point_cloud_for_preview(kf, intrinsic, args))
                print(f"Keyframe 0  stage={control.stage}  frame={frame_index}")
                continue

            prev_kf = keyframes[-1]
            prev_rgbd = keyframe_rgbd(prev_kf, args.max_depth_m)
            success, transform, information = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, rgbd, intrinsic, np.eye(4, dtype=np.float64), odo_jac, odo_opt)

            if not success:
                consecutive_failures += 1
                print(f"\nFrame {frame_index}: odometry failed ({consecutive_failures})")
                if consecutive_failures >= args.max_odometry_failures:
                    print("Too many failures — stopping.")
                    break
                continue

            consecutive_failures = 0
            pose = prev_kf.pose @ np.linalg.inv(transform)
            t_m, r_deg = pose_delta(prev_kf.pose, pose)

            if (args.max_keyframe_translation_m > 0 and t_m > args.max_keyframe_translation_m) or \
               (args.max_keyframe_rotation_deg > 0 and r_deg > args.max_keyframe_rotation_deg):
                print(f"\nFrame {frame_index}: jump {t_m:.4f}m / {r_deg:.2f}deg")
                if args.tracking_loss_policy == "stop":
                    print("Stopping.")
                    break
                continue

            if t_m < args.min_keyframe_translation_m and r_deg < args.min_keyframe_rotation_deg:
                continue

            ki = len(keyframes)
            kf = Keyframe(ki, frame_index, control.stage, pose, encode_keyframe(color, depth))
            keyframes.append(kf)
            pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                ki-1, ki, transform, information, uncertain=False))
            integrate_keyframe(preview_vol, kf, intrinsic, args)

            if preview is not None and ki % args.viz_every_n_keyframes == 0:
                if not preview.update(preview_vol):
                    preview.update_geometry(point_cloud_for_preview(kf, intrinsic, args))

            print(f"KF {ki:4d}  stage={control.stage}  "
                  f"move={t_m:.4f}m  rot={r_deg:.2f}deg  edges={len(pg.edges)}")

            if (args.loop_closure_every_n > 0 and ki > 0
                    and ki % args.loop_closure_every_n == 0):
                candidates = select_loop_candidates(keyframes, kf, args)
                # In single-circle mode always try keyframe 0 and a spread
                # of early keyframes once we're past the minimum separation.
                # Do NOT gate this on search radius — pose drift means the
                # estimated distance to KF0 grows with every step even when
                # the camera physically returned near the start position.
                if (args.scan_mode == "single-circle"
                        and kf.index >= args.loop_closure_min_separation):
                    # Force KF0 and a few early keyframes as candidates
                    forced = [keyframes[0]]
                    # Also try KF at 1/4 and 1/2 of the orbit as intermediate checks
                    quarter = len(keyframes) // 4
                    half = len(keyframes) // 2
                    if quarter > args.loop_closure_min_separation:
                        forced.append(keyframes[quarter])
                    if half > args.loop_closure_min_separation and half != quarter:
                        forced.append(keyframes[half])
                    # Merge with distance-based candidates, deduplicate, cap count
                    seen = set(id(x) for x in forced)
                    for c in candidates:
                        if id(c) not in seen:
                            forced.append(c)
                            seen.add(id(c))
                    candidates = forced[:args.max_loop_closure_candidates + 2]

                print(f"Loop check: {len(candidates)} candidates")
                closed = False
                for cand in candidates:
                    ok, lt, li, fit, rmse = run_loop_closure_icp(cand, kf, intrinsic, args)
                    if ok:
                        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                            cand.index, kf.index, lt, li, uncertain=True))
                        print(f"  ✓ loop {cand.index}→{kf.index}  fit={fit:.3f}  rmse={rmse:.4f}")
                        # Auto-stop when we close back to a very early keyframe
                        # (index <= 5) — this covers KF0 and handles cases where
                        # KF0 was skipped but KF1-5 close the loop correctly.
                        if (args.scan_mode == "single-circle"
                                and not args.no_auto_stop_on_loop
                                and cand.index <= 5):
                            closed = True
                    else:
                        print(f"  ✗ loop {cand.index}→{kf.index}  fit={fit:.3f}  rmse={rmse:.4f}")
                if closed:
                    print("Loop closed → stopping to optimise and save.")
                    control.stop_requested = True

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        zed.close()
        if preview is not None:
            preview.destroy()

    if intrinsic is None or not keyframes:
        print("No keyframes captured.")
        return

    print(f"{len(keyframes)} keyframes, {len(pg.edges)} edges.")
    optimize_pose_graph(pg, args)
    save_outputs(keyframes, pg, intrinsic, args)


if __name__ == "__main__":
    main()