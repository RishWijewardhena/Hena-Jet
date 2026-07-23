#!/usr/bin/env python3
"""Capture one ZED RGB-D and point cloud frame at a known scanner angle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from queue import Empty, Queue
import re # re is used for parsing serial messages
import time
from threading import Event, Thread
from concurrent.futures import Future, ThreadPoolExecutor
import numpy as np
import pyzed.sl as sl


FULL_REVOLUTION_DEGREES = 360.0
SEGMENT_ACKNOWLEDGEMENT = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+degree\s+ok",
    re.IGNORECASE,
)
CONTINUOUS_ANGLE_EVENT = re.compile(
    r"angle_ok,([+-]?(?:\d+(?:\.\d*)?|\.\d+)),(\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContinuousAngleEvent:
    angle_deg: float
    pulse_count: int
    received_monotonic_ns: int



def parse_segment_acknowledgement(line: str) -> float | None:
    """Return the relative segment angle from a ``<angle> degree ok`` line."""
    match = SEGMENT_ACKNOWLEDGEMENT.fullmatch(line.strip())
    if match is None:
        return None
    return float(match.group(1))


def parse_continuous_angle_event(line: str) -> tuple[float, int] | None:
    """Parse one cumulative angle crossing emitted during continuous motion."""
    match = CONTINUOUS_ANGLE_EVENT.fullmatch(line.strip())
    if match is None:
        return None
    return float(match.group(1)), int(match.group(2))


def continuous_capture_angles(step_degrees: float) -> tuple[float, ...]:
    """Return capture thresholds through 360 degrees, excluding runout."""
    if not math.isfinite(step_degrees) or step_degrees <= 0.0:
        raise ValueError("Continuous capture step must be positive")
    angles = []
    angle = step_degrees
    while angle < FULL_REVOLUTION_DEGREES - 1e-9:
        angles.append(angle)
        angle += step_degrees
    angles.append(FULL_REVOLUTION_DEGREES)
    return tuple(angles)


def pop_due_angle_event(
    pending: list[ContinuousAngleEvent], frame_ready_monotonic_ns: int
) -> ContinuousAngleEvent | None:
    """Select the sole angle event due for this completed camera frame."""
    due_count = sum(
        event.received_monotonic_ns <= frame_ready_monotonic_ns
        for event in pending
    )
    if due_count > 1:
        raise RuntimeError(
            "Multiple continuous angle events arrived before one camera frame; "
            "the scan cannot associate frames reliably"
        )
    if due_count == 0:
        return None
    return pending.pop(0)


def advance_capture_angle(current_angle_deg: float, segment_angle_deg: float) -> float:
    """Advance a cumulative scanner angle without allowing a revolution overrun."""
    if not math.isfinite(segment_angle_deg) or segment_angle_deg <= 0:
        raise ValueError(f"Invalid segment angle from controller: {segment_angle_deg}")

    next_angle = current_angle_deg + segment_angle_deg
    if next_angle > FULL_REVOLUTION_DEGREES + 1e-6:
        raise ValueError(
            f"Controller angle overrun: {current_angle_deg:g} + "
            f"{segment_angle_deg:g} exceeds 360 degrees"
        )
    if math.isclose(next_angle, FULL_REVOLUTION_DEGREES, abs_tol=1e-6):
        return FULL_REVOLUTION_DEGREES
    return next_angle


def serial_start_command(
    step_degrees: float,
    pulses_per_revolution: int,
    *,
    synchronized: bool = False,
    continuous: bool = False,
    motor_rpm: float | None = None,
    direction: str | None = None,
) -> bytes:
    """Format the motor firmware's ``start,<degrees>,<ppr>`` command."""
    step = float(step_degrees)
    ppr = int(pulses_per_revolution)
    if synchronized and continuous:
        raise ValueError("Synchronized and continuous modes are mutually exclusive")
    if not math.isfinite(step) or step <= 0 or step > FULL_REVOLUTION_DEGREES:
        raise ValueError(f"Invalid serial step angle: {step_degrees}")
    if ppr <= 0 or ppr > 100_000:
        raise ValueError(f"Invalid pulses per revolution: {pulses_per_revolution}")
    step_text = f"{step:.6f}".rstrip("0").rstrip(".")
    if continuous:
        if motor_rpm is None or not math.isfinite(float(motor_rpm)):
            raise ValueError("Continuous mode requires a finite motor RPM")
        rpm = float(motor_rpm)
        if rpm < 0.05 or rpm > 2.0:
            raise ValueError("Continuous motor RPM must be between 0.05 and 2")
        if direction not in {None, "forward", "reverse"}:
            raise ValueError("Continuous direction must be 'forward' or 'reverse'")
        rpm_text = f"{rpm:.6f}".rstrip("0").rstrip(".")
        suffix = "" if direction is None else f",{direction}"
        return (
            f"start_continuous,{step_text},{ppr},{rpm_text}{suffix}\n"
        ).encode("ascii")
    if motor_rpm is not None:
        raise ValueError("Motor RPM is only valid in continuous mode")
    if direction is not None:
        raise ValueError("Direction is only valid in continuous mode")
    prefix = "start_sync" if synchronized else "start"
    return f"{prefix},{step_text},{ppr}\n".encode("ascii")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one ZED point-cloud capture.

    --angle-deg: Optional camera angle around the scanner center, in degrees.
        If omitted, the script stays open and prompts for angles until you type q.
        This tells the merge step where this capture sits on the circular path.
    --radius-m: Required distance from the scanner center to the camera, in meters.
        This is used to place the camera around the object during merging.
    --height-m: Optional vertical camera offset, in meters. The default is 0.0.
        Use this only if the camera moves up or down between captures.
    --out-dir: Optional output folder for the .npz point cloud and .json metadata.
        The default is captures/zed_m_first_scan.
    --max-depth-m: Optional far depth cutoff from the camera, in meters.
        Points farther than this are discarded before saving.
    --min-depth-m: Optional near depth cutoff from the camera, in meters.
        The ZED SDK may clamp this upward if the requested value is too close.
    --warmup: Optional number of camera frames to skip before saving.
        This helps the camera stabilize exposure and depth.
    --angle-warmup: Optional number of extra frames to grab between interactive
        captures. This flushes stale buffered frames before each new angle.
    --resolution: Optional ZED camera resolution. Allowed values are HD2K,
        HD1080, HD720, and VGA. The default is HD720.
    --coordinate-system: ZED point-cloud coordinate system. IMAGE is the ZED
        image/depth convention: +X right, +Y down, +Z forward.
    --serial-port: Optional motor-controller port, for example /dev/ttyACM0.
        When supplied, this script owns the port, starts the 360-degree sequence,
        and captures from serial angle events. Continuous mode uses `angle_ok,<cumulative_angle>,<pulse_count>` without stopping the motor.
    """
    parser = argparse.ArgumentParser(
        description="Capture one RGB/depth point cloud from a ZED camera."
    )
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--height-m", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("captures/zed_m_first_scan"))
    parser.add_argument("--max-depth-m", type=float, default=1.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--angle-warmup", type=int, default=3)
    parser.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD720")
    parser.add_argument(
        "--coordinate-system",
        choices=["IMAGE", "RIGHT_HANDED_Z_UP_X_FWD"],
        default="IMAGE",
    )
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--serial-baud", type=int, default=115200)
    parser.add_argument("--step-deg", type=float, default=None)
    parser.add_argument("--pulses-per-revolution", type=int, default=None)
    parser.add_argument("--serial-timeout-s", type=float, default=30.0)
    parser.add_argument("--serial-reset-delay-s", type=float, default=2.0)
    parser.add_argument(
        "--serial-motion-mode",
        choices=["synchronized", "continuous"],
        default="synchronized",
        help="Use continuous to capture while the motor moves without pauses.",
    )
    parser.add_argument(
        "--motor-rpm",
        type=float,
        default=0.5,
        help="Continuous-motion speed in RPM (default: 0.5).",
    )
    parser.add_argument("--vslam", action="store_true")
    parser.add_argument(
        "--vslam-use-imu",
        action="store_true",
        help=(
            "Fuse the ZED IMU into GEN_3 positional tracking. "
            "By default --vslam remains camera-only."
        ),
    )
    parser.add_argument("--camera-yaw-deg", type=float, default=-6.0)
    parser.add_argument("--tracking-ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--tracking-capture-timeout-s", type=float, default=2.0)
    parser.add_argument("--writer-queue-size", type=int, default=2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments and raise early with a clear message on errors.

    FIX: Added validation for radius_m. A zero or negative radius would silently
    produce garbage camera transforms in the merge step.
    """
    if args.radius_m <= 0:
        raise ValueError(f"--radius-m must be positive, got {args.radius_m}")
    if args.min_depth_m <= 0:
        raise ValueError(f"--min-depth-m must be positive, got {args.min_depth_m}")
    if args.max_depth_m <= args.min_depth_m:
        raise ValueError(
            f"--max-depth-m ({args.max_depth_m}) must be greater than "
            f"--min-depth-m ({args.min_depth_m})"
        )
    if args.serial_motion_mode == "continuous" and not args.vslam:
        raise ValueError("Continuous serial motion currently requires --vslam")
    if args.serial_motion_mode == "continuous" and args.serial_port is None:
        raise ValueError("Continuous serial motion requires --serial-port")
    if args.serial_motion_mode == "continuous" and (
        not math.isfinite(args.motor_rpm)
        or args.motor_rpm < 0.05
        or args.motor_rpm > 2.0
    ):
        raise ValueError("--motor-rpm must be between 0.05 and 2")
    if args.vslam_use_imu and not args.vslam:
        raise ValueError("--vslam-use-imu requires --vslam")
    if args.vslam and args.serial_port is None:
        raise ValueError("--vslam requires --serial-port")
    if args.angle_deg is not None and args.serial_port is not None:
        raise ValueError("--angle-deg and --serial-port cannot be used together")
    if args.serial_port is not None:
        if args.step_deg is None:
            raise ValueError("--step-deg is required with --serial-port")
        if args.pulses_per_revolution is None:
            raise ValueError(
                "--pulses-per-revolution is required with --serial-port"
            )
        if args.serial_motion_mode == "continuous":
            serial_start_command(
                args.step_deg,
                args.pulses_per_revolution,
                continuous=True,
                motor_rpm=args.motor_rpm,
            )
        else:
            serial_start_command(args.step_deg, args.pulses_per_revolution)
        if args.serial_baud <= 0:
            raise ValueError("--serial-baud must be positive")
        if args.serial_timeout_s <= 0:
            raise ValueError("--serial-timeout-s must be positive")
        if args.serial_reset_delay_s < 0:
            raise ValueError("--serial-reset-delay-s cannot be negative")
        if args.vslam and args.coordinate_system != "IMAGE":
            raise ValueError("--vslam currently requires --coordinate-system IMAGE")
        if args.tracking_ready_timeout_s <= 0 or args.tracking_capture_timeout_s <= 0:
            raise ValueError("VSLAM tracking timeouts must be positive")
        if args.writer_queue_size < 1:
            raise ValueError("--writer-queue-size must be at least 1")


def resolution_from_name(name: str) -> sl.RESOLUTION:
    """Convert the user-facing resolution name into the ZED SDK enum value.

    argparse restricts the input to HD2K, HD1080, HD720, or VGA. The ZED SDK
    needs the matching sl.RESOLUTION enum instead of the string, so this helper
    performs that translation before camera initialization.
    """
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }[name]


def make_tracking_parameters(use_imu: bool) -> sl.PositionalTrackingParameters:
    """Configure GEN_3 tracking while preserving the initial camera world frame."""
    tracking = sl.PositionalTrackingParameters()
    tracking.mode = sl.POSITIONAL_TRACKING_MODE.GEN_3
    tracking.enable_imu_fusion = bool(use_imu)
    tracking.set_gravity_as_origin = False
    tracking.enable_area_memory = True
    return tracking


def coordinate_system_from_name(name: str) -> sl.COORDINATE_SYSTEM:
    """Convert the user-facing coordinate-system name into a ZED SDK enum."""
    return {
        "IMAGE": sl.COORDINATE_SYSTEM.IMAGE,
        "RIGHT_HANDED_Z_UP_X_FWD": sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
    }[name]


def forward_depth_from_points(points: np.ndarray, coordinate_system: str) -> np.ndarray:
    """Return camera-forward depth from XYZ points for the selected ZED axes."""
    if coordinate_system == "IMAGE":
        return points[:, 2]
    if coordinate_system == "RIGHT_HANDED_Z_UP_X_FWD":
        return points[:, 0]
    raise ValueError(f"Unsupported coordinate system: {coordinate_system}")


def confidence_image_to_uint8(confidence_image: np.ndarray) -> np.ndarray:
    """Convert the ZED 0-best/100-worst confidence map to compact uint8."""
    confidence = np.asarray(confidence_image, dtype=np.float32)
    if confidence.ndim == 3:
        confidence = confidence[:, :, 0]
    confidence = np.nan_to_num(confidence, nan=100.0, posinf=100.0, neginf=0.0)
    return np.clip(np.rint(confidence), 0, 100).astype(np.uint8)


def object_center_from_orbit(radius_m: float, camera_yaw_deg: float) -> np.ndarray:
    """Return the pivot in the initial IMAGE camera frame."""
    yaw = math.radians(camera_yaw_deg)
    return np.array(
        [-math.sin(yaw) * radius_m, 0.0, math.cos(yaw) * radius_m],
        dtype=np.float64,
    )


def vslam_pose_is_acceptable(
    camera_to_world: np.ndarray,
    tracking_state: str,
    odometry_status: str,
    covariance: np.ndarray,
) -> bool:
    """Validate the status and numerical shape of one camera-to-world pose."""
    transform = np.asarray(camera_to_world, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if tracking_state != "OK" or odometry_status != "OK":
        return False
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return False
    if covariance.size == 0 or not np.all(np.isfinite(covariance)):
        return False
    rotation = transform[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
        and np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    )


def rgba_float_to_rgb(rgba_values: np.ndarray) -> np.ndarray:
    """Decode ZED XYZRGBA color values stored in the fourth float channel.

    FIX: The ZED SDK packs color as BGRA, not RGBA. The original code labelled
    the extracted bytes as R, G, B but was actually reading B, G, R order,
    producing swapped red and blue channels in all saved point clouds.
    Corrected byte extraction:
        byte 0 (& 0xFF)        → Blue
        byte 1 (>> 8  & 0xFF)  → Green
        byte 2 (>> 16 & 0xFF)  → Red
    """
    rgba_uint32 = rgba_values.view(np.uint32)
    b = rgba_uint32 & 0xFF
    g = (rgba_uint32 >> 8) & 0xFF
    r = (rgba_uint32 >> 16) & 0xFF
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def camera_intrinsics_from_zed(zed: sl.Camera, image_shape: tuple[int, int]) -> dict:
    """Read left-camera pinhole intrinsics from the ZED SDK.

    Open3D TSDF fusion needs fx, fy, cx, cy, width, and height so it can
    unproject every depth pixel into a 3D point. The width and height are taken
    from the actual retrieved image shape because that is the safest match for
    the arrays saved in the .npz file.
    """
    camera_info = zed.get_camera_information()
    calibration = camera_info.camera_configuration.calibration_parameters
    left_camera = calibration.left_cam
    height, width = image_shape
    return {
        "fx": float(left_camera.fx),
        "fy": float(left_camera.fy),
        "cx": float(left_camera.cx),
        "cy": float(left_camera.cy),
        "width": int(width),
        "height": int(height),
    }


def color_image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a ZED left camera image array into an RGB uint8 image.

    FIX: The ZED SDK returns images in BGRA channel order. The original code
    took the first three channels directly, which produced a BGR image instead
    of RGB. The channel order is now reversed (::-1) to give correct RGB output
    for Open3D TSDF fusion and any downstream viewer.
    """
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    if image.shape[2] >= 3:
        # ZED returns BGRA — take first 3 channels then reverse to get RGB
        return image[:, :, :3][:, :, ::-1].astype(np.uint8)
    raise ValueError(f"Unsupported color image shape: {image.shape}")


def clean_depth_image(
    depth_image: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Prepare a meter-scale depth image for Open3D RGB-D integration.

    Open3D treats zero depth as invalid. This function keeps finite depth
    values inside the requested ZED range and changes all invalid/background
    pixels to 0.0 while preserving the original image shape.
    """
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = depth[valid]
    return cleaned


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write one captured colored point cloud to an ASCII PLY preview file.

    The .npz file remains the raw processing format for merging, while this .ply
    file is a viewer-friendly copy that can be opened directly in MeshLab or
    CloudCompare to inspect one capture before running the merge step.
    """
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def flush_frames(zed: sl.Camera, runtime: sl.RuntimeParameters, count: int) -> None:
    """Grab and discard a number of frames to flush the ZED buffer.

    FIX: In interactive capture mode the camera runs continuously between angle
    inputs. Without flushing, the grabbed frame may be stale (captured before
    the user finished positioning the scanner). Discarding a small number of
    frames gives the sensor time to settle at each new angle.
    """
    for i in range(count):
        status = zed.grab(runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Frame flush {i + 1}/{count} failed: {status}")


def read_vslam_sample(zed: sl.Camera, pose: sl.Pose) -> tuple[dict, bool]:
    """Read one WORLD-frame left-camera pose and its quality status."""
    tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
    status = zed.get_positional_tracking_status()
    camera_to_world = np.array(pose.pose_data().m, dtype=np.float64, copy=True)
    covariance = np.array(pose.pose_covariance, dtype=np.float64, copy=True).reshape(6, 6)
    sample = {
        "timestamp_ns": int(pose.timestamp.get_nanoseconds()),
        "camera_to_vslam_world": camera_to_world.tolist(),
        "pose_covariance": covariance.tolist(),
        "pose_confidence": int(pose.pose_confidence),
        "pose_confidence_authoritative": False,
        "tracking_state": str(tracking_state),
        "odometry_status": str(status.odometry_status),
        "spatial_memory_status": str(status.spatial_memory_status),
        "tracking_fusion_status": str(status.tracking_fusion_status),
    }
    acceptable = vslam_pose_is_acceptable(
        camera_to_world,
        sample["tracking_state"],
        sample["odometry_status"],
        covariance,
    )
    return sample, acceptable


def write_capture_files(
    npz_path: Path,
    ply_path: Path,
    meta_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    depth_image: np.ndarray,
    color_image: np.ndarray,
    confidence_image: np.ndarray,
    metadata: dict,
) -> None:
    """Persist one detached capture payload from the writer thread."""
    np.savez_compressed(
        npz_path,
        points=points,
        colors=colors,
        depth_image_m=depth_image,
        color_image=color_image,
        confidence_image=confidence_image,
    )
    write_ascii_ply(ply_path, points, colors)
    meta_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def capture_angle(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    point_cloud: sl.Mat,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    confidence_mat: sl.Mat,
    args: argparse.Namespace,
    angle_deg: float,
    *,
    flush: bool = False,
    already_grabbed: bool = False,
    vslam_metadata: dict | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> Future | None:
    """Capture one RGB-D frame, optionally queueing persistence."""
    if flush and not already_grabbed:
        flush_frames(zed, runtime, args.angle_warmup)
    if not already_grabbed:
        status = zed.grab(runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Could not grab frame at angle {angle_deg}: {status}")

    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    zed.retrieve_measure(confidence_mat, sl.MEASURE.CONFIDENCE)
    zed.retrieve_image(color_mat, sl.VIEW.LEFT)

    cloud = point_cloud.get_data()
    depth_image = clean_depth_image(
        depth_mat.get_data(),
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    color_image = color_image_to_rgb(color_mat.get_data())
    confidence_image = confidence_image_to_uint8(confidence_mat.get_data())
    intrinsics = camera_intrinsics_from_zed(zed, depth_image.shape)

    xyz = cloud[:, :, :3].reshape(-1, 3)
    rgba = cloud[:, :, 3].reshape(-1)
    finite = np.isfinite(xyz).all(axis=1)
    depth = forward_depth_from_points(xyz, args.coordinate_system)
    in_range = (depth >= args.min_depth_m) & (depth <= args.max_depth_m)
    valid = finite & in_range
    points = xyz[valid].astype(np.float32)
    colors = rgba_float_to_rgb(rgba[valid])

    stem = f"angle_{angle_deg:07.2f}".replace(".", "p")
    npz_path = args.out_dir / f"{stem}.npz"
    meta_path = args.out_dir / f"{stem}.json"
    ply_path = args.out_dir / f"{stem}.ply"
    if npz_path.exists() or meta_path.exists() or ply_path.exists():
        print(f"Warning: overwriting existing files for angle {angle_deg:g}")

    metadata = {
        "angle_deg": angle_deg,
        "radius_m": args.radius_m,
        "height_m": args.height_m,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "resolution": args.resolution,
        "coordinate_system": args.coordinate_system,
        "camera_intrinsics": intrinsics,
        "point_count": int(points.shape[0]),
        "depth_confidence": {
            "minimum": int(confidence_image.min()),
            "maximum": int(confidence_image.max()),
            "mean": float(confidence_image.mean()),
            "median": float(np.median(confidence_image)),
            "sdk_confidence_threshold": int(runtime.confidence_threshold),
            "sdk_texture_confidence_threshold": int(runtime.texture_confidence_threshold),
        },
    }
    if vslam_metadata is not None:
        metadata.update(vslam_metadata)

    arguments = (
        npz_path, ply_path, meta_path, points, colors, depth_image,
        color_image, confidence_image, metadata,
    )
    future = None if executor is None else executor.submit(write_capture_files, *arguments)
    if executor is None:
        write_capture_files(*arguments)
    print(f"Saved angle {angle_deg:g} with {points.shape[0]} points")
    print(npz_path)
    print(meta_path)
    print(ply_path)
    return future


def prompt_for_angle() -> float | None:
    """Read one angle from stdin; q exits the interactive loop.

    FIX: The original implementation called itself recursively on empty input,
    which would eventually hit Python's recursion limit if the user kept pressing
    Enter. Replaced with an explicit while loop that retries until valid input or
    a quit command is received.
    """
    while True:
        raw_value = input("Angle degrees to capture, or q to quit: ").strip()
        if raw_value.lower() in {"q", "quit", "exit"}:
            return None
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except ValueError:
            print(f"Invalid input '{raw_value}'. Please enter a number or q.")


def grab_vslam_frame(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    pose: sl.Pose,
    trajectory: list[dict],
) -> tuple[dict, bool]:
    """Grab one tracking frame and append its pose/status sample."""
    status = zed.grab(runtime)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"VSLAM frame grab failed: {status}")
    sample, acceptable = read_vslam_sample(zed, pose)
    trajectory.append(sample)
    return sample, acceptable


def wait_for_valid_vslam_pose(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    pose: sl.Pose,
    trajectory: list[dict],
    timeout_s: float,
) -> dict:
    deadline = time.monotonic() + timeout_s
    last_sample = None
    while time.monotonic() < deadline:
        last_sample, acceptable = grab_vslam_frame(zed, runtime, pose, trajectory)
        if acceptable:
            return last_sample
    state = "none" if last_sample is None else last_sample["tracking_state"]
    odometry = "none" if last_sample is None else last_sample["odometry_status"]
    raise RuntimeError(
        f"VSLAM did not produce a valid pose within {timeout_s:g}s "
        f"(tracking={state}, odometry={odometry})"
    )


def save_vslam_session(args: argparse.Namespace, trajectory: list[dict]) -> None:
    trajectory_path = args.out_dir / "vslam_trajectory.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(sample, allow_nan=False) + "\n" for sample in trajectory),
        encoding="utf-8",
    )
    center = object_center_from_orbit(args.radius_m, args.camera_yaw_deg)
    pose_timestamps = [
        int(sample["timestamp_ns"])
        for sample in trajectory
        if int(sample.get("timestamp_ns", 0)) > 0
    ]
    trajectory_duration_s = (
        (pose_timestamps[-1] - pose_timestamps[0]) / 1_000_000_000.0
        if len(pose_timestamps) > 1
        else 0.0
    )
    effective_tracking_fps = (
        (len(pose_timestamps) - 1) / trajectory_duration_s
        if trajectory_duration_s > 0.0
        else 0.0
    )
    event_latencies_ms = [
        float(sample["continuous_capture_event"]["event_to_frame_latency_ms"])
        for sample in trajectory
        if "continuous_capture_event" in sample
    ]
    valid_count = sum(
        sample["tracking_state"] == "OK" and sample["odometry_status"] == "OK"
        for sample in trajectory
    )
    session = {
        "schema_version": 1,
        "zed_sdk_version": sl.Camera().get_sdk_version(),
        "tracking_mode": "GEN_3",
        "imu_fusion": bool(args.vslam_use_imu),
        "area_memory": True,
        "reference_frame": "WORLD",
        "world_origin": "initial_left_camera",
        "coordinate_system": args.coordinate_system,
        "radius_m": args.radius_m,
        "camera_yaw_deg": args.camera_yaw_deg,
        "object_center_vslam_world_m": center.tolist(),
        "object_up_vslam_world": [0.0, -1.0, 0.0],
        "depth_confidence_scale": {"best": 0, "worst": 100},
        "trajectory_samples": len(trajectory),
        "trajectory_duration_s": trajectory_duration_s,
        "effective_tracking_fps": effective_tracking_fps,
        "continuous_capture_count": len(event_latencies_ms),
        "event_to_frame_latency_ms": {
            "mean": (
                sum(event_latencies_ms) / len(event_latencies_ms)
                if event_latencies_ms
                else None
            ),
            "maximum": max(event_latencies_ms) if event_latencies_ms else None,
        },
        "valid_tracking_samples": valid_count,
        "serial_motion_mode": args.serial_motion_mode,
        "motor_rpm": (
            args.motor_rpm if args.serial_motion_mode == "continuous" else None
        ),
        "continuous_runout_deg": (
            args.step_deg if args.serial_motion_mode == "continuous" else None
        ),
        "valid_tracking_fraction": (
            valid_count / len(trajectory) if trajectory else 0.0
        ),
    }
    (args.out_dir / "scan_session.json").write_text(
        json.dumps(session, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def capture_vslam_synchronized_from_serial(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    point_cloud: sl.Mat,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    confidence_mat: sl.Mat,
    args: argparse.Namespace,
) -> None:
    """Track continuously while the motor pauses at synchronized angles."""
    import serial

    try:
        connection = serial.Serial(
            args.serial_port,
            args.serial_baud,
            timeout=0.01,
            write_timeout=1,
        )
    except serial.SerialException as exc:
        raise RuntimeError(
            f"Could not open motor controller {args.serial_port}: {exc}"
        ) from exc

    trajectory: list[dict] = []
    pose = sl.Pose()
    pending: list[Future] = []
    current_angle_deg = 0.0
    capture_count = 0
    command = serial_start_command(
        args.step_deg,
        args.pulses_per_revolution,
        synchronized=True,
    )

    def request_stop() -> None:
        try:
            connection.write(b"stop\n")
            connection.flush()
            print("TX  stop")
        except (serial.SerialException, serial.SerialTimeoutException, OSError):
            pass

    def check_writes() -> None:
        nonlocal pending
        active = []
        for future in pending:
            if future.done():
                future.result()
            else:
                active.append(future)
        pending = active

    try:
        wait_for_valid_vslam_pose(
            zed,
            runtime,
            pose,
            trajectory,
            args.tracking_ready_timeout_s,
        )
        if args.serial_reset_delay_s:
            time.sleep(args.serial_reset_delay_s)
        connection.reset_input_buffer()
        connection.write(command)
        connection.flush()
        print(f"TX  {command.decode("ascii").strip()}")
        last_message_at = time.monotonic()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-writer") as executor:
            while True:
                latest_sample, _ = grab_vslam_frame(zed, runtime, pose, trajectory)
                check_writes()
                raw_line = connection.readline()
                if not raw_line:
                    if time.monotonic() - last_message_at > args.serial_timeout_s:
                        raise RuntimeError(
                            f"No motor-controller message for {args.serial_timeout_s:g} seconds"
                        )
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                last_message_at = time.monotonic()
                print(f"RX  {line}")
                segment_angle_deg = parse_segment_acknowledgement(line)
                if segment_angle_deg is not None:
                    expected_segment = min(
                        args.step_deg,
                        FULL_REVOLUTION_DEGREES - current_angle_deg,
                    )
                    if not math.isclose(
                        segment_angle_deg,
                        expected_segment,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise RuntimeError(
                            f"Expected {expected_segment:g} degrees, received "
                            f"{segment_angle_deg:g} degrees"
                        )
                    current_angle_deg = advance_capture_angle(
                        current_angle_deg,
                        segment_angle_deg,
                    )
                    for _ in range(args.angle_warmup):
                        latest_sample, _ = grab_vslam_frame(
                            zed, runtime, pose, trajectory
                        )
                    latest_sample = wait_for_valid_vslam_pose(
                        zed,
                        runtime,
                        pose,
                        trajectory,
                        args.tracking_capture_timeout_s,
                    )
                    while len(pending) >= args.writer_queue_size:
                        grab_vslam_frame(zed, runtime, pose, trajectory)
                        check_writes()
                    future = capture_angle(
                        zed,
                        runtime,
                        point_cloud,
                        depth_mat,
                        color_mat,
                        confidence_mat,
                        args,
                        current_angle_deg,
                        already_grabbed=True,
                        vslam_metadata=latest_sample,
                        executor=executor,
                    )
                    if future is not None:
                        pending.append(future)
                    capture_count += 1
                    if current_angle_deg < FULL_REVOLUTION_DEGREES:
                        connection.write(b"next\n")
                        connection.flush()
                        print("TX  next")
                    continue

                normalized = line.lower()
                if normalized == "completed":
                    if not math.isclose(current_angle_deg, FULL_REVOLUTION_DEGREES, abs_tol=1e-6):
                        raise RuntimeError(
                            f"Controller completed at {current_angle_deg:g}, expected 360"
                        )
                    for future in pending:
                        future.result()
                    save_vslam_session(args, trajectory)
                    print(f"VSLAM serial scan completed with {capture_count} captures.")
                    return
                if normalized in {"stopped", "alert"} or normalized.startswith("error,"):
                    raise RuntimeError(f"Motor controller stopped the scan: {line}")
    except KeyboardInterrupt:
        request_stop()
        print("Serial scan interrupted.")
    except Exception:
        request_stop()
        raise
    finally:
        connection.close()


def capture_vslam_continuous_from_serial(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    point_cloud: sl.Mat,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    confidence_mat: sl.Mat,
    args: argparse.Namespace,
) -> None:
    """Capture the first complete tracked frame after each moving angle event."""
    import serial

    try:
        connection = serial.Serial(
            args.serial_port, args.serial_baud, timeout=0.01, write_timeout=1
        )
    except serial.SerialException as exc:
        raise RuntimeError(
            f"Could not open motor controller {args.serial_port}: {exc}"
        ) from exc

    serial_lines: Queue[tuple[int, str]] = Queue()
    reader_stop = Event()

    def read_serial_lines() -> None:
        while not reader_stop.is_set():
            try:
                raw_line = connection.readline()
            except (serial.SerialException, OSError) as exc:
                if not reader_stop.is_set():
                    serial_lines.put(
                        (time.monotonic_ns(), f"__serial_error__:{exc}")
                    )
                return
            if raw_line:
                serial_lines.put((
                    time.monotonic_ns(),
                    raw_line.decode("utf-8", errors="replace").strip(),
                ))

    reader = Thread(
        target=read_serial_lines, name="motor-serial-reader", daemon=True
    )
    trajectory: list[dict] = []
    pose = sl.Pose()
    pending_writes: list[Future] = []
    pending_events: list[ContinuousAngleEvent] = []
    expected_angles = continuous_capture_angles(args.step_deg)
    received_event_count = 0
    captured_count = 0
    last_pulse_count = -1
    completed_received = False
    command = serial_start_command(
        args.step_deg, args.pulses_per_revolution,
        continuous=True, motor_rpm=args.motor_rpm,
    )

    def request_stop() -> None:
        try:
            connection.write(b"stop\n")
            connection.flush()
            print("TX  stop")
        except (serial.SerialException, serial.SerialTimeoutException, OSError):
            pass

    def reap_writes() -> None:
        nonlocal pending_writes
        active = []
        for future in pending_writes:
            if future.done():
                future.result()
            else:
                active.append(future)
        pending_writes = active

    try:
        wait_for_valid_vslam_pose(
            zed, runtime, pose, trajectory, args.tracking_ready_timeout_s
        )
        if args.serial_reset_delay_s:
            time.sleep(args.serial_reset_delay_s)
        connection.reset_input_buffer()
        reader.start()
        connection.write(command)
        connection.flush()
        print(f"TX  {command.decode('ascii').strip()}")
        last_message_at_ns = time.monotonic_ns()

        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="capture-writer"
        ) as executor:
            while True:
                latest_sample, acceptable = grab_vslam_frame(
                    zed, runtime, pose, trajectory
                )
                frame_ready_ns = time.monotonic_ns()
                reap_writes()

                while True:
                    try:
                        received_ns, line = serial_lines.get_nowait()
                    except Empty:
                        break
                    if not line:
                        continue
                    if line.startswith("__serial_error__:"):
                        raise RuntimeError(line.split(":", 1)[1])
                    last_message_at_ns = received_ns
                    print(f"RX  {line}")

                    parsed_event = parse_continuous_angle_event(line)
                    if parsed_event is not None:
                        angle_deg, pulse_count = parsed_event
                        if received_event_count >= len(expected_angles):
                            raise RuntimeError(
                                f"Unexpected extra continuous angle event: {line}"
                            )
                        expected_angle = expected_angles[received_event_count]
                        if not math.isclose(
                            angle_deg, expected_angle, rel_tol=0.0, abs_tol=1e-6
                        ):
                            raise RuntimeError(
                                f"Expected continuous angle {expected_angle:g}, "
                                f"received {angle_deg:g}"
                            )
                        if pulse_count <= last_pulse_count:
                            raise RuntimeError(
                                "Continuous angle pulse counts are not increasing"
                            )
                        pending_events.append(ContinuousAngleEvent(
                            angle_deg, pulse_count, received_ns
                        ))
                        received_event_count += 1
                        last_pulse_count = pulse_count
                        continue

                    normalized = line.lower()
                    if normalized == "completed":
                        completed_received = True
                    elif normalized in {"stopped", "alert"} or normalized.startswith(
                        "error,"
                    ):
                        raise RuntimeError(
                            f"Motor controller stopped the scan: {line}"
                        )

                capture_event = pop_due_angle_event(
                    pending_events, frame_ready_ns
                )
                if capture_event is not None:
                    if not acceptable:
                        raise RuntimeError(
                            f"VSLAM pose was invalid at "
                            f"{capture_event.angle_deg:g} degrees"
                        )
                    reap_writes()
                    if len(pending_writes) >= args.writer_queue_size:
                        raise RuntimeError(
                            "Capture writer could not keep up with continuous "
                            "motion; refusing to mislabel a later frame"
                        )
                    latency_ms = (
                        frame_ready_ns - capture_event.received_monotonic_ns
                    ) / 1_000_000.0
                    event_metadata = {
                        "angle_deg": capture_event.angle_deg,
                        "pulse_count": capture_event.pulse_count,
                        "serial_received_monotonic_ns": (
                            capture_event.received_monotonic_ns
                        ),
                        "frame_ready_monotonic_ns": frame_ready_ns,
                        "event_to_frame_latency_ms": latency_ms,
                    }
                    latest_sample["continuous_capture_event"] = event_metadata
                    capture_metadata = dict(latest_sample)
                    future = capture_angle(
                        zed, runtime, point_cloud, depth_mat, color_mat,
                        confidence_mat, args, capture_event.angle_deg,
                        already_grabbed=True,
                        vslam_metadata=capture_metadata, executor=executor,
                    )
                    if future is not None:
                        pending_writes.append(future)
                    captured_count += 1

                if completed_received:
                    if pending_events:
                        raise RuntimeError(
                            "Motor completed before the final angle event could "
                            "be associated with a moving camera frame"
                        )
                    if received_event_count != len(expected_angles):
                        raise RuntimeError(
                            f"Motor completed after {received_event_count} angle "
                            f"events; expected {len(expected_angles)}"
                        )
                    if captured_count != len(expected_angles):
                        raise RuntimeError(
                            f"Captured {captured_count} moving frames; expected "
                            f"{len(expected_angles)}"
                        )
                    for future in pending_writes:
                        future.result()
                    save_vslam_session(args, trajectory)
                    print(
                        f"Continuous VSLAM scan completed with "
                        f"{captured_count} captures."
                    )
                    return

                elapsed_s = (
                    time.monotonic_ns() - last_message_at_ns
                ) / 1_000_000_000.0
                if elapsed_s > args.serial_timeout_s:
                    raise RuntimeError(
                        f"No motor-controller message for "
                        f"{args.serial_timeout_s:g} seconds"
                    )
    except KeyboardInterrupt:
        request_stop()
        print("Continuous serial scan interrupted.")
    except Exception:
        request_stop()
        raise
    finally:
        reader_stop.set()
        if reader.is_alive():
            reader.join(timeout=0.5)
        connection.close()


def capture_vslam_from_serial(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    point_cloud: sl.Mat,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    confidence_mat: sl.Mat,
    args: argparse.Namespace,
) -> None:
    """Dispatch to the selected VSLAM motor synchronization mode."""
    capture_function = (
        capture_vslam_continuous_from_serial
        if args.serial_motion_mode == "continuous"
        else capture_vslam_synchronized_from_serial
    )
    capture_function(
        zed, runtime, point_cloud, depth_mat, color_mat, confidence_mat, args
    )


def capture_from_serial(
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    point_cloud: sl.Mat,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    confidence_mat: sl.Mat,
    args: argparse.Namespace,
) -> None:
    """Run a full scan, capturing once after every completed motor segment.

    This mode owns the serial port. The Tkinter motor GUI must be disconnected
    while it runs because two processes cannot safely consume the same device
    messages.
    """
    import serial

    try:
        connection = serial.Serial(
            args.serial_port,
            args.serial_baud,
            timeout=0.2,
            write_timeout=1,
        )
    except serial.SerialException as exc:
        raise RuntimeError(
            f"Could not open motor controller {args.serial_port}: {exc}"
        ) from exc

    command = serial_start_command(args.step_deg, args.pulses_per_revolution)
    current_angle_deg = 0.0
    capture_count = 0
    last_message_at = time.monotonic()

    def request_stop() -> None:
        try:
            connection.write(b"stop\n")
            connection.flush()
            print("TX  stop")
        except (serial.SerialException, serial.SerialTimeoutException, OSError):
            pass

    try:
        if args.serial_reset_delay_s:
            time.sleep(args.serial_reset_delay_s)
        connection.reset_input_buffer()
        connection.write(command)
        connection.flush()
        print(f"TX  {command.decode('ascii').strip()}")

        while True:
            raw_line = connection.readline()
            if not raw_line:
                if time.monotonic() - last_message_at > args.serial_timeout_s:
                    raise RuntimeError(
                        f"No motor-controller message for "
                        f"{args.serial_timeout_s:g} seconds"
                    )
                continue

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            last_message_at = time.monotonic()
            print(f"RX  {line}")

            segment_angle_deg = parse_segment_acknowledgement(line)
            if segment_angle_deg is not None:
                expected_segment = min(
                    args.step_deg,
                    FULL_REVOLUTION_DEGREES - current_angle_deg,
                )
                if not math.isclose(
                    segment_angle_deg,
                    expected_segment,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        f"Expected a {expected_segment:g}-degree acknowledgement, "
                        f"received {segment_angle_deg:g} degrees"
                    )

                current_angle_deg = advance_capture_angle(
                    current_angle_deg,
                    segment_angle_deg,
                )
                capture_angle(
                    zed,
                    runtime,
                    point_cloud,
                    depth_mat,
                    color_mat,
                    confidence_mat,
                    args,
                    current_angle_deg,
                    flush=True,
                )
                capture_count += 1
                last_message_at = time.monotonic()
                continue

            normalized = line.lower()
            if normalized == "completed":
                if not math.isclose(
                    current_angle_deg,
                    FULL_REVOLUTION_DEGREES,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        f"Controller completed at cumulative angle "
                        f"{current_angle_deg:g}, expected 360"
                    )
                print(f"Serial scan completed with {capture_count} captures.")
                return
            if normalized in {"stopped", "alert"} or normalized.startswith("error,"):
                raise RuntimeError(f"Motor controller stopped the scan: {line}")
    except KeyboardInterrupt:
        request_stop()
        print("Serial scan interrupted.")
    except Exception:
        request_stop()
        raise
    finally:
        connection.close()


def main() -> None:
    """Capture one filtered XYZRGBA point cloud and save RGB-D data plus metadata.

    The function opens the ZED camera with the requested resolution and depth
    limits, skips warmup frames, grabs one synchronized frame, saves the
    image-shaped depth/color arrays needed by Open3D TSDF fusion, and also
    keeps the older filtered point/color arrays plus a .ply preview for quick
    MeshLab inspection.
    """
    args = parse_args()
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.camera_resolution = resolution_from_name(args.resolution)
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = coordinate_system_from_name(args.coordinate_system)
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 100
    runtime.texture_confidence_threshold = 100
    point_cloud = sl.Mat()
    depth_mat = sl.Mat()
    color_mat = sl.Mat()
    confidence_mat = sl.Mat()
    tracking_enabled = False

    try:
        for index in range(args.warmup):
            status = zed.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Warmup frame {index + 1} failed: {status}")

        if args.vslam:
            tracking = make_tracking_parameters(args.vslam_use_imu)
            status = zed.enable_positional_tracking(tracking)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Could not enable ZED positional tracking: {status}")
            tracking_enabled = True
            imu_state = "enabled" if args.vslam_use_imu else "disabled"
            print(f"ZED GEN_3 tracking enabled; IMU fusion {imu_state}.")

        if args.serial_port is not None:
            serial_capture = capture_vslam_from_serial if args.vslam else capture_from_serial
            serial_capture(
                zed,
                runtime,
                point_cloud,
                depth_mat,
                color_mat,
                confidence_mat,
                args,
            )
            return

        if args.angle_deg is not None:
            # Single-angle mode: warmup already done above, no extra flush needed
            capture_angle(
                zed,
                runtime,
                point_cloud,
                depth_mat,
                color_mat,
                confidence_mat,
                args,
                args.angle_deg,
                flush=False,
            )
            return

        print("Interactive capture mode. Type an angle and press Enter.")
        print("Type q to quit.")
        while True:
            angle_deg = prompt_for_angle()
            if angle_deg is None:
                print("Capture session finished.")
                break
            # FIX: flush stale frames accumulated while the user was positioning
            capture_angle(
                zed,
                runtime,
                point_cloud,
                depth_mat,
                color_mat,
                confidence_mat,
                args,
                angle_deg,
                flush=True,
            )
    finally:
        if tracking_enabled:
            zed.disable_positional_tracking()
        zed.close()


if __name__ == "__main__":
    main()
