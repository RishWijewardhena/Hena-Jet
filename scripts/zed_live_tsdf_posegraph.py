#!/usr/bin/env python3
"""Single-circle live ZED RGB-D mapper using Open3D odometry and TSDF."""

from __future__ import annotations

import argparse
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pyzed.sl as sl


WARMUP_FRAMES = 20
DEFAULT_RESOLUTION = "HD720"
DEFAULT_DEPTH_MODE = "NEURAL_LIGHT"
DEFAULT_KEYFRAME_EVERY_N = 3
DEFAULT_PREVIEW_VOXEL_LENGTH_M = 0.004
DEFAULT_PREVIEW_SDF_TRUNC_M = 0.016
DEFAULT_VIZ_EVERY_N_KEYFRAMES = 15
DEFAULT_MIN_VALID_DEPTH_PX = 5000

RESOLUTIONS = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}
DEPTH_MODES = {
    name: getattr(sl.DEPTH_MODE, name)
    for name in ("PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT")
    if hasattr(sl.DEPTH_MODE, name)
}

MIN_KEYFRAME_TRANSLATION_M = 0.004
MIN_KEYFRAME_ROTATION_DEG = 1.0
MAX_KEYFRAME_TRANSLATION_M = 0.08
MAX_KEYFRAME_ROTATION_DEG = 12.0
MAX_ODOMETRY_FAILURES = 20
MAX_TRACKING_JUMP_FAILURES = 20

LOOP_EVERY_N = 40
LOOP_MIN_SEPARATION = 50
ICP_VOXEL_M = 0.004
ICP_MAX_CORRESPONDENCE_M = 0.04
ICP_MIN_FITNESS = 0.40
ICP_MAX_RMSE = 0.010
ODOMETRY_DEPTH_DIFF_MAX_M = 0.08
EDGE_PRUNE_THRESHOLD = 0.25

# ZED with COORDINATE_SYSTEM.IMAGE uses +X right, +Y down, +Z forward — the
# standard OpenCV pinhole convention.  Open3D's ScalableTSDFVolume.integrate()
# uses the same convention (+Z is the depth axis), so no coordinate-frame
# rotation is needed between the two.


@dataclass
class Keyframe:
    index: int
    frame_index: int
    pose: np.ndarray
    packet: bytes


@dataclass
class ScanControl:
    stop_requested: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live ZED single-circle TSDF mapper."
    )
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=0.35)
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=(0.0, 0.0, 1.0, 1.0),
        help="Normalized crop applied to both color and depth.",
    )
    parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTIONS),
        default=DEFAULT_RESOLUTION,
        help="ZED camera resolution. Lower this to VGA if live FPS is too low.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=sorted(DEPTH_MODES),
        default=DEFAULT_DEPTH_MODE if DEFAULT_DEPTH_MODE in DEPTH_MODES else "NEURAL",
        help="ZED depth mode. PERFORMANCE is faster; NEURAL/NEURAL_LIGHT are cleaner.",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_FRAMES)
    parser.add_argument(
        "--keyframe-every-n",
        type=int,
        default=DEFAULT_KEYFRAME_EVERY_N,
        help="Only run RGB-D odometry/integration every N captured frames.",
    )
    parser.add_argument(
        "--min-keyframe-translation-m",
        type=float,
        default=MIN_KEYFRAME_TRANSLATION_M,
    )
    parser.add_argument(
        "--min-keyframe-rotation-deg",
        type=float,
        default=MIN_KEYFRAME_ROTATION_DEG,
    )
    parser.add_argument(
        "--max-keyframe-translation-m",
        type=float,
        default=MAX_KEYFRAME_TRANSLATION_M,
    )
    parser.add_argument(
        "--max-keyframe-rotation-deg",
        type=float,
        default=MAX_KEYFRAME_ROTATION_DEG,
    )
    parser.add_argument(
        "--max-odometry-failures",
        type=int,
        default=MAX_ODOMETRY_FAILURES,
    )
    parser.add_argument(
        "--max-tracking-jump-failures",
        type=int,
        default=MAX_TRACKING_JUMP_FAILURES,
    )
    parser.add_argument("--voxel-length-m", type=float, default=0.002)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.010)
    parser.add_argument(
        "--preview-voxel-length-m",
        type=float,
        default=DEFAULT_PREVIEW_VOXEL_LENGTH_M,
        help="Coarser live TSDF voxel size. Final save still uses --voxel-length-m.",
    )
    parser.add_argument(
        "--preview-sdf-trunc-m",
        type=float,
        default=DEFAULT_PREVIEW_SDF_TRUNC_M,
        help="Live preview TSDF truncation distance.",
    )
    parser.add_argument(
        "--viz-every-n-keyframes",
        type=int,
        default=DEFAULT_VIZ_EVERY_N_KEYFRAMES,
        help="Refresh the Open3D preview every N accepted keyframes.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable the Open3D live viewer and preview TSDF for maximum FPS.",
    )
    parser.add_argument(
        "--min-valid-depth-px",
        type=int,
        default=DEFAULT_MIN_VALID_DEPTH_PX,
        help="Skip frames with fewer nonzero depth pixels than this.",
    )
    parser.add_argument(
        "--compress-keyframes",
        action="store_true",
        help="Compress stored keyframes to save RAM at the cost of live FPS.",
    )
    parser.add_argument(
        "--loop-every-n",
        type=int,
        default=LOOP_EVERY_N,
        help="Try loop closure every N keyframes; 0 disables loop checks.",
    )
    parser.add_argument(
        "--loop-min-separation",
        type=int,
        default=LOOP_MIN_SEPARATION,
    )
    parser.add_argument(
        "--loop-max-transform-m",
        type=float,
        default=0.08,
        help="Reject loop closures with relative translation larger than this.",
    )
    parser.add_argument(
        "--loop-max-rotation-deg",
        type=float,
        default=30.0,
        help="Reject loop closures with relative rotation (degrees) larger than this.",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=0,
        help="Stop after this many keyframes. 0 means no explicit cap.",
    )
    parser.add_argument(
        "--mesh-out",
        type=Path,
        default=Path("outputs/zed_live_posegraph_tsdf_mesh.ply"),
    )
    parser.add_argument(
        "--cloud-out",
        type=Path,
        default=Path("outputs/zed_live_posegraph_tsdf_cloud.ply"),
    )
    parser.add_argument(
        "--posegraph-out",
        type=Path,
        default=Path("outputs/zed_live_posegraph.json"),
    )
    parser.add_argument(
        "--poses-out",
        type=Path,
        default=Path("outputs/zed_live_poses.npy"),
    )
    return parser.parse_args()


def validate_roi(roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = roi
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError("--roi must satisfy 0<=X_MIN<X_MAX<=1 and 0<=Y_MIN<Y_MAX<=1")
    return roi


def validate_args(args: argparse.Namespace) -> None:
    args.roi = validate_roi(tuple(args.roi))
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m must be greater than --min-depth-m")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or positive")
    if args.keyframe_every_n <= 0:
        raise ValueError("--keyframe-every-n must be positive")
    if args.max_odometry_failures <= 0:
        raise ValueError("--max-odometry-failures must be positive")
    if args.max_tracking_jump_failures <= 0:
        raise ValueError("--max-tracking-jump-failures must be positive")
    if args.voxel_length_m <= 0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")
    if args.preview_voxel_length_m <= 0:
        raise ValueError("--preview-voxel-length-m must be positive")
    if args.preview_sdf_trunc_m <= args.preview_voxel_length_m:
        raise ValueError("--preview-sdf-trunc-m must be greater than --preview-voxel-length-m")
    if args.viz_every_n_keyframes <= 0:
        raise ValueError("--viz-every-n-keyframes must be positive")
    if args.min_valid_depth_px < 0:
        raise ValueError("--min-valid-depth-px must be zero or positive")
    if args.loop_every_n < 0:
        raise ValueError("--loop-every-n must be zero or positive")
    if args.loop_min_separation < 0:
        raise ValueError("--loop-min-separation must be zero or positive")
    if args.loop_max_transform_m <= 0:
        raise ValueError("--loop-max-transform-m must be positive")
    if args.loop_max_rotation_deg <= 0:
        raise ValueError("--loop-max-rotation-deg must be positive")
    if args.max_keyframes < 0:
        raise ValueError("--max-keyframes must be zero or positive")


def color_image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        return image[:, :, :3][:, :, ::-1].astype(np.uint8)
    raise ValueError(f"Unsupported ZED color image shape: {image.shape}")


def apply_roi_mask(
    image: np.ndarray,
    roi: tuple[float, float, float, float],
    fill_value: float | int = 0,
) -> np.ndarray:
    if roi == (0.0, 0.0, 1.0, 1.0):
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
    roi: tuple[float, float, float, float],
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
    image_shape: tuple[int, int],
) -> o3d.camera.PinholeCameraIntrinsic:
    camera_info = zed.get_camera_information()
    calib = camera_info.camera_configuration.calibration_parameters
    left = calib.left_cam
    h, w = image_shape
    return o3d.camera.PinholeCameraIntrinsic(
        int(w), int(h), float(left.fx), float(left.fy), float(left.cx), float(left.cy)
    )


def make_rgbd(color: np.ndarray, depth: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def encode_keyframe(color: np.ndarray, depth: np.ndarray, compressed: bool) -> bytes:
    buffer = io.BytesIO()
    save = np.savez_compressed if compressed else np.savez
    save(
        buffer,
        color=color.astype(np.uint8, copy=False),
        depth=depth.astype(np.float32, copy=False),
    )
    return buffer.getvalue()


def decode_keyframe(packet: bytes) -> tuple[np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(packet)) as data:
        color = data["color"].astype(np.uint8, copy=True)
        depth = data["depth"].astype(np.float32, copy=True)
    return color, depth


def keyframe_rgbd(kf: Keyframe, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    color, depth = decode_keyframe(kf.packet)
    return make_rgbd(color, depth, depth_trunc_m)


def make_tsdf_volume(
    args: argparse.Namespace,
    voxel_length_m: float | None = None,
    sdf_trunc_m: float | None = None,
) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m if voxel_length_m is None else voxel_length_m,
        sdf_trunc=args.sdf_trunc_m if sdf_trunc_m is None else sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def pose_delta(prev_pose: np.ndarray, curr_pose: np.ndarray) -> tuple[float, float]:
    rel = np.linalg.inv(prev_pose) @ curr_pose
    translation_m = float(np.linalg.norm(rel[:3, 3]))
    cos_angle = (float(np.trace(rel[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return translation_m, math.degrees(math.acos(cos_angle))


def point_cloud_for_registration(
    kf: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        keyframe_rgbd(kf, args.max_depth_m), intrinsic
    )
    cloud = cloud.voxel_down_sample(ICP_VOXEL_M)
    if len(cloud.points) > 0:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(ICP_MAX_CORRESPONDENCE_M * 2.0, ICP_VOXEL_M * 3.0),
                max_nn=30,
            )
        )
    return cloud


def point_cloud_for_preview(
    kf: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        keyframe_rgbd(kf, args.max_depth_m), intrinsic
    )
    cloud = cloud.voxel_down_sample(args.preview_voxel_length_m)
    cloud.transform(kf.pose)
    return cloud


def run_loop_closure_icp(
    first: Keyframe,
    current: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> tuple[bool, np.ndarray, np.ndarray, float, float]:
    source = point_cloud_for_registration(first, intrinsic, args)
    target = point_cloud_for_registration(current, intrinsic, args)
    if len(source.points) < 100 or len(target.points) < 100:
        return False, np.eye(4), np.eye(6), 0.0, float("inf")

    init = np.linalg.inv(current.pose) @ first.pose
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)

    coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_MAX_CORRESPONDENCE_M * 4.0,
        init,
        estimation,
        criteria,
    )
    fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_MAX_CORRESPONDENCE_M,
        coarse.transformation,
        estimation,
        criteria,
    )

    fitness = float(fine.fitness)
    rmse = float(fine.inlier_rmse)
    if fitness < ICP_MIN_FITNESS or rmse > ICP_MAX_RMSE:
        return False, fine.transformation, np.eye(6), fitness, rmse

    # Phase 1a: geometric validation — reject loop closures with implausible relative transforms
    trans = float(np.linalg.norm(fine.transformation[:3, 3]))
    cos_angle = (float(np.trace(fine.transformation[:3, :3])) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    rot_deg = math.degrees(math.acos(cos_angle))
    if trans > args.loop_max_transform_m or rot_deg > args.loop_max_rotation_deg:
        return False, fine.transformation, np.eye(6), fitness, rmse

    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, ICP_MAX_CORRESPONDENCE_M, fine.transformation
    )
    return True, fine.transformation, information, fitness, rmse


class LivePreview:
    def __init__(self, control: ScanControl) -> None:
        self.control = control
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="ZED Single Circle TSDF [Q save]", width=1280, height=720)
        self.geometry = None
        self.has_reset_view = False
        self.vis.register_key_callback(ord("q"), self._stop)
        self.vis.register_key_callback(ord("Q"), self._stop)
        options = self.vis.get_render_option()
        options.background_color = np.array([0.05, 0.05, 0.05])
        options.point_size = 1.5

    def _stop(self, _) -> bool:
        self.control.stop_requested = True
        print("\nStopping...")
        return False

    def is_open(self) -> bool:
        return self.vis.poll_events()

    def update_geometry(self, geometry: o3d.geometry.Geometry) -> None:
        if len(geometry.points) == 0:
            self.vis.poll_events()
            self.vis.update_renderer()
            return
        if self.geometry is not None:
            self.vis.remove_geometry(self.geometry, reset_bounding_box=False)
        reset = not self.has_reset_view
        self.geometry = geometry
        self.vis.add_geometry(geometry, reset_bounding_box=reset)
        self.has_reset_view = True
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_from_volume(self, volume) -> bool:
        cloud = volume.extract_point_cloud()
        if len(cloud.points) == 0:
            self.vis.poll_events()
            self.vis.update_renderer()
            return False
        self.update_geometry(cloud)
        return True

    def destroy(self) -> None:
        self.vis.destroy_window()


def integrate_keyframe(
    volume,
    kf: Keyframe,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> None:
    rgbd = keyframe_rgbd(kf, args.max_depth_m)
    # FIX: extrinsic is world-to-camera.  ZED IMAGE convention (+X right,
    # +Y down, +Z forward) is identical to what Open3D TSDF integration
    # expects, so inv(pose) is all that is needed — no extra rotation matrix.
    extrinsic = np.linalg.inv(kf.pose)
    volume.integrate(rgbd, intrinsic, extrinsic)


def optimize_pose_graph(pg: o3d.pipelines.registration.PoseGraph) -> None:
    if len(pg.nodes) < 2:
        return
    print("Optimizing pose graph...")
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=ICP_MAX_CORRESPONDENCE_M,
            edge_prune_threshold=EDGE_PRUNE_THRESHOLD,
            reference_node=0,
        ),
    )


def save_outputs(
    keyframes: list[Keyframe],
    pg: o3d.pipelines.registration.PoseGraph,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    args: argparse.Namespace,
) -> None:
    for kf, node in zip(keyframes, pg.nodes):
        kf.pose = np.asarray(node.pose, dtype=np.float64)

    print("Rebuilding final TSDF from optimized poses...")
    volume = make_tsdf_volume(args)
    for i, kf in enumerate(keyframes, start=1):
        integrate_keyframe(volume, kf, intrinsic, args)
        print(f"  {i}/{len(keyframes)}", end="\r")
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


def retrieve_rgbd(
    zed: sl.Camera,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    zed.retrieve_image(color_mat, sl.VIEW.LEFT)

    depth = clean_depth_image(
        depth_mat.get_data(),
        args.min_depth_m,
        args.max_depth_m,
        args.roi,
    )
    color = apply_roi_mask(color_image_to_rgb(color_mat.get_data()), args.roi, fill_value=0)
    return color.astype(np.uint8), depth


def add_first_keyframe(
    keyframes: list[Keyframe],
    pg: o3d.pipelines.registration.PoseGraph,
    preview_volume,
    preview: LivePreview | None,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    frame_index: int,
    color: np.ndarray,
    depth: np.ndarray,
    args: argparse.Namespace,
) -> None:
    pose = np.eye(4, dtype=np.float64)
    kf = Keyframe(0, frame_index, pose, encode_keyframe(color, depth, args.compress_keyframes))
    keyframes.append(kf)
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
    if preview_volume is not None:
        integrate_keyframe(preview_volume, kf, intrinsic, args)
        if preview is not None and not preview.update_from_volume(preview_volume):
            preview.update_geometry(point_cloud_for_preview(kf, intrinsic, args))
    print(f"Keyframe 0  frame={frame_index}")


def accept_keyframe(
    keyframes: list[Keyframe],
    pg: o3d.pipelines.registration.PoseGraph,
    preview_volume,
    preview: LivePreview | None,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    frame_index: int,
    pose: np.ndarray,
    transform: np.ndarray,
    information: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    args: argparse.Namespace,
) -> Keyframe:
    index = len(keyframes)
    kf = Keyframe(index, frame_index, pose, encode_keyframe(color, depth, args.compress_keyframes))
    keyframes.append(kf)
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
    pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
        index - 1, index, transform, information, uncertain=False
    ))
    if preview_volume is not None:
        integrate_keyframe(preview_volume, kf, intrinsic, args)
        if preview is not None and index % args.viz_every_n_keyframes == 0:
            if not preview.update_from_volume(preview_volume):
                preview.update_geometry(point_cloud_for_preview(kf, intrinsic, args))
    return kf


def main() -> None:
    args = parse_args()
    validate_args(args)

    zed = open_zed(args)
    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()
    color_mat = sl.Mat()
    control = ScanControl()
    preview = None if args.no_viz else LivePreview(control)

    intrinsic = None
    keyframes: list[Keyframe] = []
    pg = o3d.pipelines.registration.PoseGraph()
    preview_volume = None if args.no_viz else make_tsdf_volume(
        args,
        args.preview_voxel_length_m,
        args.preview_sdf_trunc_m,
    )
    frame_index = 0
    odometry_failures = 0
    tracking_jump_failures = 0
    sparse_depth_frames = 0
    scan_started_at = 0.0

    # CUDA Open3D builds expose constructor kwargs as depth_diff_max/depth_min/
    # depth_max. Do not pass iteration_number_per_pyramid_level here: CUDA
    # pybind expects an IntVector, not a normal Python list.
    try:
        odo_options = o3d.pipelines.odometry.OdometryOption(
            depth_diff_max=ODOMETRY_DEPTH_DIFF_MAX_M,
            depth_min=args.min_depth_m,
            depth_max=args.max_depth_m,
        )
    except TypeError:
        odo_options = o3d.pipelines.odometry.OdometryOption()
        odo_options.depth_diff_max = ODOMETRY_DEPTH_DIFF_MAX_M
        odo_options.depth_min = args.min_depth_m
        odo_options.depth_max = args.max_depth_m
    odo_jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    print(f"Warming up ({args.warmup} frames)...")
    try:
        for _ in range(args.warmup):
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("Warmup failed")

        if preview is not None:
            print("Single-circle scan. Orbit once around the object; Q saves early.")
        else:
            print("Single-circle scan. Headless mode; press Ctrl+C to save early.")
        print("Mapping started.")
        scan_started_at = time.perf_counter()
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

            color, depth = retrieve_rgbd(zed, depth_mat, color_mat, args)
            valid_depth_px = int(np.count_nonzero(depth))
            if valid_depth_px < args.min_valid_depth_px:
                sparse_depth_frames += 1
                if sparse_depth_frames % 30 == 0:
                    print(
                        f"  [{sparse_depth_frames} sparse frames skipped; "
                        f"valid_depth_px={valid_depth_px} < {args.min_valid_depth_px}]"
                    )
                continue
            if sparse_depth_frames >= 5:
                print(f"  [recovered after {sparse_depth_frames} sparse frames]")
            sparse_depth_frames = 0

            if intrinsic is None:
                intrinsic = camera_intrinsic_from_zed(zed, depth.shape)

            if not keyframes:
                add_first_keyframe(
                    keyframes, pg, preview_volume, preview, intrinsic,
                    frame_index, color, depth, args
                )
                continue

            prev_kf = keyframes[-1]
            success, transform, information = o3d.pipelines.odometry.compute_rgbd_odometry(
                keyframe_rgbd(prev_kf, args.max_depth_m),
                make_rgbd(color, depth, args.max_depth_m),
                intrinsic,
                np.eye(4, dtype=np.float64),
                odo_jacobian,
                odo_options,
            )
            if not success:
                odometry_failures += 1
                print(f"\nFrame {frame_index}: odometry failed ({odometry_failures})")
                if odometry_failures >= args.max_odometry_failures:
                    print("Too many odometry failures; stopping and saving.")
                    break
                continue
            odometry_failures = 0

            pose = prev_kf.pose @ np.linalg.inv(transform)
            move_m, rot_deg = pose_delta(prev_kf.pose, pose)
            if move_m > args.max_keyframe_translation_m or rot_deg > args.max_keyframe_rotation_deg:
                tracking_jump_failures += 1
                print(f"\nFrame {frame_index}: tracking jump {move_m:.4f}m / {rot_deg:.2f}deg")
                if tracking_jump_failures >= args.max_tracking_jump_failures:
                    print("Too many tracking jumps; stopping before the shape is distorted.")
                    break
                continue
            if move_m < args.min_keyframe_translation_m and rot_deg < args.min_keyframe_rotation_deg:
                continue

            tracking_jump_failures = 0
            kf = accept_keyframe(
                keyframes, pg, preview_volume, preview, intrinsic,
                frame_index, pose, transform, information, color, depth, args
            )
            elapsed = max(time.perf_counter() - scan_started_at, 1e-9)
            fps = frame_index / elapsed
            print(
                f"KF {kf.index:4d}  frame={frame_index:4d}  "
                f"move={move_m:.4f}m  rot={rot_deg:.2f}deg  "
                f"valid_px={valid_depth_px}  fps={fps:.1f}  edges={len(pg.edges)}"
            )

            if args.max_keyframes > 0 and len(keyframes) >= args.max_keyframes:
                print(f"Reached --max-keyframes={args.max_keyframes}; stopping and saving.")
                break

            if (
                args.loop_every_n > 0
                and kf.index >= args.loop_min_separation
                and kf.index % args.loop_every_n == 0
            ):
                ok, loop_transform, loop_info, fitness, rmse = run_loop_closure_icp(
                    keyframes[0], kf, intrinsic, args
                )
                # Compute translation and rotation for logging
                loop_trans = float(np.linalg.norm(loop_transform[:3, 3]))
                loop_cos = (float(np.trace(loop_transform[:3, :3])) - 1.0) / 2.0
                loop_cos = max(-1.0, min(1.0, loop_cos))
                loop_rot = math.degrees(math.acos(loop_cos))
                if ok:
                    pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        0, kf.index, loop_transform, loop_info, uncertain=True
                    ))
                    print(
                        f"  loop 0->{kf.index} accepted  fit={fitness:.3f}  "
                        f"rmse={rmse:.4f}  move={loop_trans:.4f}m  rot={loop_rot:.2f}deg"
                    )
                    print("Loop closed; stopping to optimize and save.")
                    break
                print(
                    f"  loop 0->{kf.index} rejected  fit={fitness:.3f}  "
                    f"rmse={rmse:.4f}  move={loop_trans:.4f}m  rot={loop_rot:.2f}deg"
                )

    except KeyboardInterrupt:
        print("\nInterrupted; saving current map.")
    finally:
        zed.close()
        if preview is not None:
            preview.destroy()

    if intrinsic is None or not keyframes:
        print("No keyframes captured.")
        return

    print(f"{len(keyframes)} keyframes, {len(pg.edges)} edges.")
    optimize_pose_graph(pg)
    save_outputs(keyframes, pg, intrinsic, args)


if __name__ == "__main__":
    main()
