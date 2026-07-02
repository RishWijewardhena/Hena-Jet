#!/usr/bin/env python3
"""Manual-angle OAK-D RGB-D scanner using circular poses and Open3D TSDF."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import depthai as dai
import numpy as np
import open3d as o3d


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 800
DEFAULT_FPS = 15.0
DEFAULT_WARMUP_FRAMES = 20
DEFAULT_MIN_DEPTH_M = 0.10
DEFAULT_MAX_DEPTH_M = 0.35
DEFAULT_VOXEL_LENGTH_M = 0.002
DEFAULT_SDF_TRUNC_M = 0.012
DEFAULT_MIN_VALID_DEPTH_PX = 5000


@dataclass
class Keyframe:
    index: int
    angle_deg: float
    corrected_angle_deg: float
    pose: np.ndarray
    valid_depth_px: int
    captured_at_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture OAK-D RGB-D frames at manual angles and fuse a TSDF mesh."
    )
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--height-m", type=float, default=0.0)
    parser.add_argument("--min-depth-m", type=float, default=DEFAULT_MIN_DEPTH_M)
    parser.add_argument("--max-depth-m", type=float, default=DEFAULT_MAX_DEPTH_M)
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=(0.0, 0.0, 1.0, 1.0),
        help="Normalized crop applied to both color and depth.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--voxel-length-m", type=float, default=DEFAULT_VOXEL_LENGTH_M)
    parser.add_argument("--sdf-trunc-m", type=float, default=DEFAULT_SDF_TRUNC_M)
    parser.add_argument(
        "--min-valid-depth-px",
        type=int,
        default=DEFAULT_MIN_VALID_DEPTH_PX,
        help="Reject captures with fewer nonzero depth pixels than this.",
    )
    parser.add_argument(
        "--no-left-right-check",
        action="store_true",
        help="Disable stereo LR-check. Default keeps LR-check enabled.",
    )
    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="Disable subpixel disparity. Default keeps subpixel enabled.",
    )
    parser.add_argument(
        "--extended-disparity",
        action="store_true",
        help="Enable closer-range extended disparity. This disables subpixel.",
    )
    parser.add_argument("--invert-angles", action="store_true")
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--center-offset-x-m", type=float, default=0.0)
    parser.add_argument("--center-offset-y-m", type=float, default=0.0)
    parser.add_argument(
        "--mesh-out",
        type=Path,
        default=Path("outputs/oakd_circle_tsdf_mesh.ply"),
    )
    parser.add_argument(
        "--cloud-out",
        type=Path,
        default=Path("outputs/oakd_circle_tsdf_cloud.ply"),
    )
    parser.add_argument(
        "--poses-out",
        type=Path,
        default=Path("outputs/oakd_circle_poses.npy"),
    )
    parser.add_argument(
        "--keyframes-out",
        type=Path,
        default=Path("outputs/oakd_circle_keyframes.json"),
    )
    parser.add_argument(
        "--save-keyframes",
        action="store_true",
        help="Save accepted RGB/depth frames as compressed .npz files.",
    )
    parser.add_argument(
        "--keyframe-dir",
        type=Path,
        default=Path("outputs/oakd_circle_keyframes"),
    )
    parser.add_argument("--no-viz", action="store_true")
    return parser.parse_args()


def validate_roi(roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = roi
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError("--roi must satisfy 0<=X_MIN<X_MAX<=1 and 0<=Y_MIN<Y_MAX<=1")
    return roi


def validate_args(args: argparse.Namespace) -> None:
    args.roi = validate_roi(tuple(args.roi))
    if args.radius_m <= 0.0:
        raise ValueError("--radius-m must be positive")
    if args.min_depth_m <= 0.0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("--max-depth-m must be greater than --min-depth-m")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or positive")
    if args.voxel_length_m <= 0.0:
        raise ValueError("--voxel-length-m must be positive")
    if args.sdf_trunc_m <= args.voxel_length_m:
        raise ValueError("--sdf-trunc-m must be greater than --voxel-length-m")
    if args.min_valid_depth_px < 0:
        raise ValueError("--min-valid-depth-px must be zero or positive")


def mono_resolution_from_size(width: int, height: int) -> dai.MonoCameraProperties.SensorResolution:
    if width <= 640 and height <= 480:
        return dai.MonoCameraProperties.SensorResolution.THE_400_P
    if width <= 1280 and height <= 800:
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    return dai.MonoCameraProperties.SensorResolution.THE_1200_P


def create_pipeline(args: argparse.Namespace) -> tuple[dai.Pipeline, object, object]:
    pipeline = dai.Pipeline()

    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb.setPreviewSize(args.width, args.height)
    rgb.setPreviewKeepAspectRatio(False)
    rgb.setInterleaved(False)
    rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    rgb.setFps(args.fps)

    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_resolution = mono_resolution_from_size(args.width, args.height)
    left.setResolution(mono_resolution)
    right.setResolution(mono_resolution)
    left.setFps(args.fps)
    right.setFps(args.fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setLeftRightCheck(not args.no_left_right_check)
    stereo.setExtendedDisparity(args.extended_disparity)
    stereo.setSubpixel((not args.no_subpixel) and (not args.extended_disparity))
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(args.width, args.height)
    stereo.setOutputKeepAspectRatio(False)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    rgb_queue = rgb.preview.createOutputQueue(maxSize=4, blocking=False)
    depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    return pipeline, rgb_queue, depth_queue


def camera_intrinsic_from_device(
    device: dai.Device,
    width: int,
    height: int,
) -> o3d.camera.PinholeCameraIntrinsic:
    calibration = device.readCalibration2()
    matrix = np.asarray(
        calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height),
        dtype=np.float64,
    )
    return o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )


def intrinsic_to_dict(intrinsic: o3d.camera.PinholeCameraIntrinsic) -> dict:
    matrix = intrinsic.intrinsic_matrix
    return {
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
    }


def resize_color_nearest(color: np.ndarray, height: int, width: int) -> np.ndarray:
    if color.shape[:2] == (height, width):
        return color
    row_indices = np.minimum(
        (np.arange(height) * color.shape[0] / height).astype(np.int64),
        color.shape[0] - 1,
    )
    col_indices = np.minimum(
        (np.arange(width) * color.shape[1] / width).astype(np.int64),
        color.shape[1] - 1,
    )
    return color[row_indices[:, None], col_indices]


def color_bgr_to_rgb(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    color = np.asarray(frame)
    if color.ndim == 2:
        color = np.repeat(color[:, :, None], 3, axis=2)
    if color.shape[2] > 3:
        color = color[:, :, :3]
    color = resize_color_nearest(color, height, width)
    return color[:, :, ::-1].astype(np.uint8)


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
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = depth[valid]
    return apply_roi_mask(cleaned, roi, fill_value=0)


def make_rgbd(color: np.ndarray, depth: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def corrected_angle(angle_deg: float, invert_angles: bool, angle_offset_deg: float) -> float:
    signed = -angle_deg if invert_angles else angle_deg
    return signed + angle_offset_deg


def camera_pose_from_circle(
    angle_deg: float,
    radius_m: float,
    height_m: float,
    center_offset_x_m: float,
    center_offset_y_m: float,
) -> np.ndarray:
    """Return camera-to-world pose for Open3D image camera axes.

    Camera-local axes are Open3D pinhole image coordinates:
        +X right, +Y down, +Z forward.
    """
    theta = math.radians(angle_deg)
    center = np.array([center_offset_x_m, center_offset_y_m, 0.0], dtype=np.float64)
    radial_out = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
    forward = -radial_out
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, radial_out)
    right /= np.linalg.norm(right)
    down = -world_up

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = center + np.array(
        [radius_m * math.cos(theta), radius_m * math.sin(theta), height_m],
        dtype=np.float64,
    )
    return pose


def make_tsdf_volume(args: argparse.Namespace) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length_m,
        sdf_trunc=args.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def integrate_frame(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    color: np.ndarray,
    depth: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    pose: np.ndarray,
    args: argparse.Namespace,
) -> None:
    rgbd = make_rgbd(color, depth, args.max_depth_m)
    volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))


def latest_frames(rgb_queue, depth_queue, warmup: int) -> tuple[np.ndarray, np.ndarray]:
    rgb_msg = None
    depth_msg = None
    for _ in range(max(warmup, 1)):
        rgb_msg = rgb_queue.get()
        depth_msg = depth_queue.get()
    if rgb_msg is None or depth_msg is None:
        raise RuntimeError("Could not read RGB/depth frames from OAK-D")

    depth_m = np.asarray(depth_msg.getFrame(), dtype=np.float32) / 1000.0
    color_rgb = color_bgr_to_rgb(rgb_msg.getCvFrame(), depth_m.shape[0], depth_m.shape[1])
    return color_rgb, depth_m


def prompt_for_angle() -> float | None:
    raw_value = input("Angle degrees to integrate, or q to finish: ").strip()
    if raw_value.lower() in {"q", "quit", "exit"}:
        return None
    if not raw_value:
        return prompt_for_angle()
    return float(raw_value)


class LivePreview:
    def __init__(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="OAK-D Manual Circle TSDF", width=1280, height=720)
        self.geometry = None
        self.has_reset_view = False
        options = self.vis.get_render_option()
        options.background_color = np.array([0.05, 0.05, 0.05])
        options.point_size = 1.5

    def update(self, volume: o3d.pipelines.integration.ScalableTSDFVolume) -> None:
        cloud = volume.extract_point_cloud()
        if len(cloud.points) == 0:
            self.vis.poll_events()
            self.vis.update_renderer()
            return
        if self.geometry is not None:
            self.vis.remove_geometry(self.geometry, reset_bounding_box=False)
        reset = not self.has_reset_view
        self.geometry = cloud
        self.vis.add_geometry(cloud, reset_bounding_box=reset)
        self.has_reset_view = True
        self.vis.poll_events()
        self.vis.update_renderer()

    def destroy(self) -> None:
        self.vis.destroy_window()


def save_keyframe_npz(
    args: argparse.Namespace,
    keyframe: Keyframe,
    color: np.ndarray,
    depth: np.ndarray,
) -> str | None:
    if not args.save_keyframes:
        return None
    args.keyframe_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kf_{keyframe.index:04d}_angle_{keyframe.angle_deg:07.2f}".replace(".", "p")
    path = args.keyframe_dir / f"{stem}.npz"
    np.savez_compressed(
        path,
        color=color.astype(np.uint8, copy=False),
        depth=depth.astype(np.float32, copy=False),
        pose=keyframe.pose,
    )
    return str(path)


def save_outputs(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    keyframes: list[Keyframe],
    keyframe_files: list[str | None],
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
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

    poses = np.stack([kf.pose for kf in keyframes]) if keyframes else np.empty((0, 4, 4))
    args.poses_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.poses_out, poses)
    print(f"Poses: {args.poses_out}  ({len(poses)} poses)")

    metadata = {
        "camera_model": "OAK-D S2 FF",
        "coordinate_system": "Open3D image coordinates: +X right, +Y down, +Z forward",
        "radius_m": args.radius_m,
        "height_m": args.height_m,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "voxel_length_m": args.voxel_length_m,
        "sdf_trunc_m": args.sdf_trunc_m,
        "roi": list(args.roi),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "left_right_check": not args.no_left_right_check,
        "subpixel": (not args.no_subpixel) and (not args.extended_disparity),
        "extended_disparity": args.extended_disparity,
        "invert_angles": args.invert_angles,
        "angle_offset_deg": args.angle_offset_deg,
        "center_offset_x_m": args.center_offset_x_m,
        "center_offset_y_m": args.center_offset_y_m,
        "camera_intrinsics": intrinsic_to_dict(intrinsic),
        "keyframes": [
            {
                "index": kf.index,
                "angle_deg": kf.angle_deg,
                "corrected_angle_deg": kf.corrected_angle_deg,
                "valid_depth_px": kf.valid_depth_px,
                "captured_at_s": kf.captured_at_s,
                "pose": kf.pose.tolist(),
                "frame_file": keyframe_files[i],
            }
            for i, kf in enumerate(keyframes)
        ],
    }
    args.keyframes_out.parent.mkdir(parents=True, exist_ok=True)
    args.keyframes_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Keyframes: {args.keyframes_out}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.extended_disparity and not args.no_subpixel:
        print("Note: --extended-disparity is enabled, so subpixel is disabled.")

    pipeline, rgb_queue, depth_queue = create_pipeline(args)
    volume = make_tsdf_volume(args)
    preview = None if args.no_viz else LivePreview()
    keyframes: list[Keyframe] = []
    keyframe_files: list[str | None] = []

    with pipeline:
        device = pipeline.getDefaultDevice()
        intrinsic = camera_intrinsic_from_device(device, args.width, args.height)
        pipeline.start()

        print("OAK-D manual-angle TSDF scanner started.")
        print(f"Depth/RGB output: {args.width}x{args.height}")
        print(f"Depth range: {args.min_depth_m:.3f}m to {args.max_depth_m:.3f}m")
        print(f"Radius: {args.radius_m:.3f}m, height: {args.height_m:.3f}m")
        print(f"Warming up ({args.warmup} frames)...")
        latest_frames(rgb_queue, depth_queue, args.warmup)

        started_at = time.perf_counter()
        while True:
            try:
                angle_deg = prompt_for_angle()
            except ValueError as exc:
                print(f"Invalid angle input: {exc}")
                continue
            except KeyboardInterrupt:
                print("\nInterrupted; saving current reconstruction.")
                break

            if angle_deg is None:
                break

            color, raw_depth = latest_frames(rgb_queue, depth_queue, 2)
            depth = clean_depth_image(raw_depth, args.min_depth_m, args.max_depth_m, args.roi)
            color = apply_roi_mask(color, args.roi, fill_value=0)
            valid_depth_px = int(np.count_nonzero(depth))
            if valid_depth_px < args.min_valid_depth_px:
                print(
                    f"Skipped angle {angle_deg:g}: valid_depth_px={valid_depth_px} "
                    f"< {args.min_valid_depth_px}"
                )
                continue

            fixed_angle = corrected_angle(angle_deg, args.invert_angles, args.angle_offset_deg)
            pose = camera_pose_from_circle(
                fixed_angle,
                args.radius_m,
                args.height_m,
                args.center_offset_x_m,
                args.center_offset_y_m,
            )
            integrate_frame(volume, color, depth, intrinsic, pose, args)
            kf = Keyframe(
                index=len(keyframes),
                angle_deg=angle_deg,
                corrected_angle_deg=fixed_angle,
                pose=pose,
                valid_depth_px=valid_depth_px,
                captured_at_s=time.perf_counter() - started_at,
            )
            keyframes.append(kf)
            keyframe_files.append(save_keyframe_npz(args, kf, color, depth))
            print(
                f"KF {kf.index:03d}  angle={angle_deg:.2f}  "
                f"corrected={fixed_angle:.2f}  valid_px={valid_depth_px}"
            )

            if preview is not None:
                preview.update(volume)

    if preview is not None:
        preview.destroy()

    if not keyframes:
        print("No keyframes captured; no outputs written.")
        return

    save_outputs(volume, keyframes, keyframe_files, intrinsic, args)


if __name__ == "__main__":
    main()
