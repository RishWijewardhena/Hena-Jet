#!/usr/bin/env python3
"""Capture continuous ZED VSLAM orbits at multiple measured camera heights."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import time
from typing import Callable

import numpy as np
import pyzed.sl as sl

try:
    from scripts.capture_zed_angle import (
        ContinuousAngleEvent,
        capture_angle,
        continuous_capture_angles,
        grab_vslam_frame,
        make_tracking_parameters,
        object_center_from_orbit,
        parse_continuous_angle_event,
        pop_due_angle_event,
        resolution_from_name,
        serial_start_command,
        wait_for_valid_vslam_pose,
    )
except ModuleNotFoundError:
    from capture_zed_angle import (  # type: ignore[no-redef]
        ContinuousAngleEvent,
        capture_angle,
        continuous_capture_angles,
        grab_vslam_frame,
        make_tracking_parameters,
        object_center_from_orbit,
        parse_continuous_angle_event,
        pop_due_angle_event,
        resolution_from_name,
        serial_start_command,
        wait_for_valid_vslam_pose,
    )


@dataclass
class ContinuousPassProgress:
    """Validate serial angle events and capture counts for one revolution."""

    step_deg: float
    expected_angles: tuple[float, ...] = field(init=False)
    pending_events: list[ContinuousAngleEvent] = field(default_factory=list)
    received_event_count: int = 0
    captured_count: int = 0
    last_pulse_count: int = -1
    completed_received: bool = False

    def __post_init__(self) -> None:
        self.expected_angles = continuous_capture_angles(self.step_deg)

    def accept_serial_line(self, line: str, *, received_ns: int) -> None:
        parsed = parse_continuous_angle_event(line)
        if parsed is not None:
            if self.completed_received:
                raise RuntimeError("Continuous angle event arrived after completion")
            angle_deg, pulse_count = parsed
            if self.received_event_count >= len(self.expected_angles):
                raise RuntimeError(f"Unexpected extra continuous angle event: {line}")
            expected_angle = self.expected_angles[self.received_event_count]
            if not math.isclose(
                angle_deg, expected_angle, rel_tol=0.0, abs_tol=1e-6
            ):
                raise RuntimeError(
                    f"Expected continuous angle {expected_angle:g}, "
                    f"received {angle_deg:g}"
                )
            if pulse_count <= self.last_pulse_count:
                raise RuntimeError("Continuous angle pulse counts are not increasing")
            self.pending_events.append(
                ContinuousAngleEvent(angle_deg, pulse_count, received_ns)
            )
            self.received_event_count += 1
            self.last_pulse_count = pulse_count
            return

        normalized = line.strip().lower()
        if normalized == "completed":
            self.completed_received = True
        elif normalized in {"stopped", "alert"} or normalized.startswith("error,"):
            raise RuntimeError(f"Motor controller stopped the scan: {line}")

    def pop_due_event(self, *, frame_ready_ns: int) -> ContinuousAngleEvent | None:
        return pop_due_angle_event(self.pending_events, frame_ready_ns)

    def mark_captured(self) -> None:
        self.captured_count += 1

    def validate_completed(self) -> None:
        expected = len(self.expected_angles)
        if self.pending_events:
            raise RuntimeError("Motor completed with an unassociated angle event")
        if self.received_event_count != expected:
            raise RuntimeError(
                f"Motor completed after {self.received_event_count} angle events; "
                f"expected {expected}"
            )
        if self.captured_count != expected:
            raise RuntimeError(
                f"Captured {self.captured_count} moving frames; expected {expected}"
            )
        if not self.completed_received:
            raise RuntimeError("Motor completion acknowledgement has not arrived")


@dataclass(frozen=True)
class PassResult:
    """Completed capture statistics for one camera-height revolution."""

    pass_index: int
    height_offset_m: float
    motor_direction: str
    output_directory: str
    capture_count: int
    first_capture_index: int
    final_capture_index: int
    start_camera_to_vslam_world: list[list[float]]
    final_capture_camera_to_vslam_world: list[list[float]]
    end_camera_to_vslam_world: list[list[float]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the dedicated continuous multilevel capture command."""
    parser = argparse.ArgumentParser(
        description=(
            "Capture continuous ZED VSLAM revolutions at measured camera heights "
            "without restarting positional tracking."
        )
    )
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("captures/zed_m_vslam_multilevel"))
    parser.add_argument(
        "--height-offsets-m",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.04],
    )
    parser.add_argument("--between-pass-wait-s", type=float, default=30.0)
    parser.add_argument("--lift-translation-tolerance-m", type=float, default=0.002)
    parser.add_argument("--lift-rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--max-depth-m", type=float, default=1.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--resolution",
        choices=["HD2K", "HD1080", "HD720", "VGA"],
        default="HD720",
    )
    parser.add_argument("--serial-port", required=True)
    parser.add_argument("--serial-baud", type=int, default=115200)
    parser.add_argument("--step-deg", type=float, default=5.0)
    parser.add_argument("--pulses-per-revolution", type=int, required=True)
    parser.add_argument("--motor-rpm", type=float, default=0.25)
    parser.add_argument(
        "--first-pass-direction",
        choices=["forward", "reverse"],
        default="forward",
        help=(
            "Motor direction for pass 1. Later passes automatically alternate "
            "to prevent cable winding."
        ),
    )
    parser.add_argument("--serial-timeout-s", type=float, default=30.0)
    parser.add_argument("--serial-reset-delay-s", type=float, default=2.0)
    parser.add_argument("--camera-yaw-deg", type=float, default=-6.0)
    parser.add_argument("--tracking-ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--writer-queue-size", type=int, default=2)
    parser.add_argument(
        "--vslam-use-imu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ZED IMU fusion (default); use --no-vslam-use-imu to disable.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[float, ...]:
    """Validate hardware and scan settings before opening either device."""
    offsets = normalize_height_offsets_m(args.height_offsets_m)
    if not math.isfinite(args.radius_m) or args.radius_m <= 0.0:
        raise ValueError("--radius-m must be positive")
    if args.min_depth_m <= 0.0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("Depth limits are invalid")
    if args.warmup < 0 or args.writer_queue_size < 1:
        raise ValueError("Warmup and writer queue settings are invalid")
    if args.serial_baud <= 0 or args.serial_timeout_s <= 0.0:
        raise ValueError("Serial baud and timeout must be positive")
    if args.serial_reset_delay_s < 0.0:
        raise ValueError("Serial reset delay cannot be negative")
    if args.between_pass_wait_s < 30.0:
        raise ValueError("--between-pass-wait-s must be at least 30 seconds")
    if args.lift_translation_tolerance_m <= 0.0:
        raise ValueError("Lift translation tolerance must be positive")
    if args.lift_rotation_tolerance_deg <= 0.0:
        raise ValueError("Lift rotation tolerance must be positive")
    if args.tracking_ready_timeout_s <= 0.0:
        raise ValueError("Tracking ready timeout must be positive")
    serial_start_command(
        args.step_deg,
        args.pulses_per_revolution,
        continuous=True,
        motor_rpm=args.motor_rpm,
        direction=args.first_pass_direction,
    )
    return offsets


def direction_for_pass(pass_index: int, first_direction: str) -> str:
    """Alternate direction after every revolution to unwind camera cables."""
    if pass_index < 0:
        raise ValueError("Pass index must be non-negative")
    if first_direction not in {"forward", "reverse"}:
        raise ValueError("First pass direction must be 'forward' or 'reverse'")
    if pass_index % 2 == 0:
        return first_direction
    return "reverse" if first_direction == "forward" else "forward"


def normalize_height_offsets_m(values: list[float]) -> tuple[float, ...]:
    """Validate absolute camera-height offsets measured from the first pass."""
    offsets = tuple(float(value) for value in values)
    if len(offsets) < 2:
        raise ValueError("At least two camera height offsets are required")
    if not all(math.isfinite(value) and value >= 0.0 for value in offsets):
        raise ValueError("Camera height offsets must be finite and non-negative")
    if not math.isclose(offsets[0], 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("The first camera height offset must be 0 m")
    if any(current <= previous for previous, current in zip(offsets, offsets[1:])):
        raise ValueError("Camera height offsets must be strictly increasing")
    return offsets


def pass_directory(root: Path, pass_index: int, height_offset_m: float) -> Path:
    """Return the stable output directory for one absolute-height pass."""
    if pass_index < 0 or not math.isfinite(height_offset_m) or height_offset_m < 0.0:
        raise ValueError("Pass index and height offset must be non-negative")
    height_mm = int(round(height_offset_m * 1000.0))
    return Path(root) / f"pass_{pass_index:02d}_height_{height_mm:03d}mm"


def transition_is_ready(
    elapsed_s: float,
    minimum_wait_s: float,
    operator_confirmed: bool,
) -> bool:
    """Require both the settling interval and explicit operator confirmation."""
    if not math.isfinite(elapsed_s) or not math.isfinite(minimum_wait_s):
        raise ValueError("Transition timing must be finite")
    if elapsed_s < 0.0 or minimum_wait_s < 0.0:
        raise ValueError("Transition timing cannot be negative")
    return bool(operator_confirmed and elapsed_s >= minimum_wait_s)


def _validate_rigid_pose(pose: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} must have a rigid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation must have determinant +1")
    return matrix


def lift_pose_metrics(
    before_pose: np.ndarray,
    after_pose: np.ndarray,
    *,
    expected_height_delta_m: float,
    object_up: np.ndarray,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict:
    """Measure a manual lift using consecutive camera-to-world VSLAM poses."""
    before = _validate_rigid_pose(before_pose, "before_pose")
    after = _validate_rigid_pose(after_pose, "after_pose")
    up = np.asarray(object_up, dtype=np.float64)
    if up.shape != (3,) or not np.all(np.isfinite(up)):
        raise ValueError("object_up must be a finite three-vector")
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 0.0:
        raise ValueError("object_up cannot be zero")
    if (
        not math.isfinite(expected_height_delta_m)
        or expected_height_delta_m <= 0.0
        or not math.isfinite(translation_tolerance_m)
        or translation_tolerance_m < 0.0
        or not math.isfinite(rotation_tolerance_deg)
        or rotation_tolerance_deg < 0.0
    ):
        raise ValueError("Lift target and tolerances are invalid")

    up = up / up_norm
    translation = after[:3, 3] - before[:3, 3]
    vertical_translation_m = float(translation @ up)
    lateral = translation - vertical_translation_m * up
    lateral_error_m = float(np.linalg.norm(lateral))
    vertical_error_m = abs(vertical_translation_m - expected_height_delta_m)

    relative_rotation = before[:3, :3].T @ after[:3, :3]
    cosine = (float(np.trace(relative_rotation)) - 1.0) / 2.0
    rotation_error_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, cosine)))
    )
    accepted = bool(
        vertical_error_m <= translation_tolerance_m
        and lateral_error_m <= translation_tolerance_m
        and rotation_error_deg <= rotation_tolerance_deg
    )
    return {
        "accepted": accepted,
        "expected_height_delta_m": float(expected_height_delta_m),
        "vertical_translation_m": vertical_translation_m,
        "vertical_error_m": vertical_error_m,
        "lateral_error_m": lateral_error_m,
        "rotation_error_deg": rotation_error_deg,
    }


def capture_metadata_for_pass(
    vslam_sample: dict,
    *,
    pass_index: int,
    height_offset_m: float,
    capture_index: int,
    motor_direction: str,
) -> dict:
    """Attach multilevel identity without mutating the trajectory sample."""
    if pass_index < 0 or capture_index < 0:
        raise ValueError("Pass and capture indices must be non-negative")
    if not math.isfinite(height_offset_m) or height_offset_m < 0.0:
        raise ValueError("Height offset must be finite and non-negative")
    if motor_direction not in {"forward", "reverse"}:
        raise ValueError("Motor direction must be 'forward' or 'reverse'")
    metadata = dict(vslam_sample)
    metadata.update(
        {
            "multilevel_pass_index": int(pass_index),
            "height_offset_m": float(height_offset_m),
            "multilevel_capture_index": int(capture_index),
            "motor_direction": motor_direction,
        }
    )
    return metadata


def wait_for_validated_lift(
    *,
    before_pose: np.ndarray,
    expected_height_delta_m: float,
    object_up: np.ndarray,
    minimum_wait_s: float,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
    grab_sample: Callable[[], tuple[dict, bool]],
    operator_confirmed: Callable[[], bool],
    elapsed_s: Callable[[], float],
    reset_confirmation: Callable[[dict], None],
) -> tuple[dict, dict]:
    """Keep tracking during a manual lift until its measured pose is accepted."""
    while True:
        sample, acceptable = grab_sample()
        if not acceptable:
            raise RuntimeError(
                "VSLAM tracking became invalid during the camera-height transition; "
                "restart the complete multilevel scan"
            )
        if not transition_is_ready(
            elapsed_s(), minimum_wait_s, operator_confirmed()
        ):
            continue
        if "camera_to_vslam_world" not in sample:
            raise RuntimeError("Validated VSLAM sample has no camera pose")
        metrics = lift_pose_metrics(
            before_pose,
            np.asarray(sample["camera_to_vslam_world"], dtype=np.float64),
            expected_height_delta_m=expected_height_delta_m,
            object_up=object_up,
            translation_tolerance_m=translation_tolerance_m,
            rotation_tolerance_deg=rotation_tolerance_deg,
        )
        if metrics["accepted"]:
            return sample, metrics
        reset_confirmation(metrics)


def _confirmation_prompt(target_height_m: float) -> Event:
    confirmed = Event()

    def wait_for_enter() -> None:
        try:
            input(
                f"Move the camera to absolute height offset "
                f"{target_height_m * 1000.0:.1f} mm, then press Enter: "
            )
        except EOFError:
            return
        confirmed.set()

    Thread(
        target=wait_for_enter,
        name="multilevel-height-confirmation",
        daemon=True,
    ).start()
    return confirmed


def _reap_writes(pending: list[Future]) -> list[Future]:
    active = []
    for future in pending:
        if future.done():
            future.result()
        else:
            active.append(future)
    return active


def capture_continuous_pass(
    *,
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    point_cloud: sl.Mat,
    depth_mat: sl.Mat,
    color_mat: sl.Mat,
    confidence_mat: sl.Mat,
    connection: object,
    serial_lines: Queue,
    args: argparse.Namespace,
    pass_index: int,
    height_offset_m: float,
    trajectory: list[dict],
    executor: ThreadPoolExecutor,
    first_capture_index: int,
) -> tuple[PassResult, int]:
    """Capture one continuously moving revolution without restarting tracking."""
    pass_dir = pass_directory(args.out_dir, pass_index, height_offset_m)
    pass_dir.mkdir(parents=True, exist_ok=True)
    pass_values = vars(args).copy()
    pass_values.update(
        {
            "out_dir": pass_dir,
            "height_m": height_offset_m,
            "coordinate_system": "IMAGE",
        }
    )
    pass_args = argparse.Namespace(**pass_values)
    motor_direction = direction_for_pass(
        pass_index, args.first_pass_direction
    )
    progress = ContinuousPassProgress(args.step_deg)
    pending_writes: list[Future] = []
    pose = sl.Pose()
    start_sample, start_acceptable = grab_vslam_frame(
        zed, runtime, pose, trajectory
    )
    start_sample["multilevel_phase"] = "pass_start"
    start_sample["multilevel_pass_index"] = pass_index
    start_sample["height_offset_m"] = height_offset_m
    if not start_acceptable:
        raise RuntimeError(
            f"VSLAM tracking was invalid before pass {pass_index + 1} started"
        )
    command = serial_start_command(
        args.step_deg,
        args.pulses_per_revolution,
        continuous=True,
        motor_rpm=args.motor_rpm,
        direction=motor_direction,
    )
    connection.write(command)
    connection.flush()
    print(
        f"Pass {pass_index + 1}: TX {command.decode('ascii').strip()} "
        f"at {height_offset_m * 1000.0:.1f} mm ({motor_direction})"
    )
    last_message_ns = time.monotonic_ns()
    latest_sample: dict | None = None
    latest_acceptable = False
    final_capture_pose: list[list[float]] | None = None

    while True:
        latest_sample, latest_acceptable = grab_vslam_frame(
            zed, runtime, pose, trajectory
        )
        latest_sample["multilevel_phase"] = "orbit"
        latest_sample["multilevel_pass_index"] = pass_index
        latest_sample["height_offset_m"] = height_offset_m
        frame_ready_ns = time.monotonic_ns()
        pending_writes = _reap_writes(pending_writes)

        while True:
            try:
                received_ns, line = serial_lines.get_nowait()
            except Empty:
                break
            if not line:
                continue
            if line.startswith("__serial_error__:"):
                raise RuntimeError(line.split(":", 1)[1])
            last_message_ns = received_ns
            print(f"Pass {pass_index + 1}: RX {line}")
            progress.accept_serial_line(line, received_ns=received_ns)

        capture_event = progress.pop_due_event(frame_ready_ns=frame_ready_ns)
        if capture_event is not None:
            if not latest_acceptable:
                raise RuntimeError(
                    f"VSLAM pose was invalid at pass {pass_index + 1}, "
                    f"angle {capture_event.angle_deg:g} degrees"
                )
            pending_writes = _reap_writes(pending_writes)
            if len(pending_writes) >= args.writer_queue_size:
                raise RuntimeError(
                    "Capture writer could not keep up with continuous motion; "
                    "refusing to associate a later frame with this angle"
                )
            event_metadata = {
                "angle_deg": capture_event.angle_deg,
                "pulse_count": capture_event.pulse_count,
                "serial_received_monotonic_ns": capture_event.received_monotonic_ns,
                "frame_ready_monotonic_ns": frame_ready_ns,
                "event_to_frame_latency_ms": (
                    frame_ready_ns - capture_event.received_monotonic_ns
                )
                / 1_000_000.0,
            }
            latest_sample["continuous_capture_event"] = event_metadata
            global_capture_index = first_capture_index + progress.captured_count
            capture_metadata = capture_metadata_for_pass(
                latest_sample,
                pass_index=pass_index,
                height_offset_m=height_offset_m,
                capture_index=global_capture_index,
                motor_direction=motor_direction,
            )
            future = capture_angle(
                zed,
                runtime,
                point_cloud,
                depth_mat,
                color_mat,
                confidence_mat,
                pass_args,
                capture_event.angle_deg,
                already_grabbed=True,
                vslam_metadata=capture_metadata,
                executor=executor,
            )
            if future is not None:
                pending_writes.append(future)
            if math.isclose(
                capture_event.angle_deg,
                360.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                final_capture_pose = latest_sample["camera_to_vslam_world"]
            progress.mark_captured()

        if progress.completed_received:
            progress.validate_completed()
            if not latest_acceptable or latest_sample is None:
                raise RuntimeError(
                    f"VSLAM tracking was invalid when pass {pass_index + 1} completed"
                )
            if final_capture_pose is None:
                raise RuntimeError(
                    f"Pass {pass_index + 1} has no saved 360-degree pose"
                )
            for future in pending_writes:
                future.result()
            final_capture_index = first_capture_index + progress.captured_count - 1
            result = PassResult(
                pass_index=pass_index,
                height_offset_m=height_offset_m,
                motor_direction=motor_direction,
                output_directory=str(pass_dir),
                capture_count=progress.captured_count,
                first_capture_index=first_capture_index,
                final_capture_index=final_capture_index,
                start_camera_to_vslam_world=start_sample[
                    "camera_to_vslam_world"
                ],
                final_capture_camera_to_vslam_world=final_capture_pose,
                end_camera_to_vslam_world=latest_sample[
                    "camera_to_vslam_world"
                ],
            )
            print(
                f"Pass {pass_index + 1} completed with "
                f"{progress.captured_count} captures."
            )
            return result, final_capture_index + 1

        elapsed_s = (time.monotonic_ns() - last_message_ns) / 1_000_000_000.0
        if elapsed_s > args.serial_timeout_s:
            raise RuntimeError(
                f"No motor-controller message for {args.serial_timeout_s:g} seconds"
            )


def perform_height_transition(
    *,
    zed: sl.Camera,
    runtime: sl.RuntimeParameters,
    args: argparse.Namespace,
    trajectory: list[dict],
    from_result: PassResult,
    target_pass_index: int,
    target_height_m: float,
) -> dict:
    """Track continuously while the operator raises and verifies the camera."""
    before_pose = np.asarray(
        from_result.end_camera_to_vslam_world, dtype=np.float64
    )
    expected_delta_m = target_height_m - from_result.height_offset_m
    transition_started = time.monotonic()
    confirmation = _confirmation_prompt(target_height_m)
    pose = sl.Pose()

    def grab_transition_sample() -> tuple[dict, bool]:
        sample, acceptable = grab_vslam_frame(
            zed, runtime, pose, trajectory
        )
        sample["multilevel_phase"] = "height_transition"
        sample["multilevel_pass_index"] = target_pass_index
        sample["height_offset_m"] = target_height_m
        return sample, acceptable

    def reset_confirmation(metrics: dict) -> None:
        nonlocal confirmation
        print(
            "Lift validation failed: "
            f"vertical={metrics['vertical_translation_m'] * 1000.0:.2f} mm, "
            f"vertical error={metrics['vertical_error_m'] * 1000.0:.2f} mm, "
            f"lateral error={metrics['lateral_error_m'] * 1000.0:.2f} mm, "
            f"rotation={metrics['rotation_error_deg']:.2f} deg."
        )
        print("Adjust the camera mount; tracking remains active.")
        confirmation = _confirmation_prompt(target_height_m)

    sample, metrics = wait_for_validated_lift(
        before_pose=before_pose,
        expected_height_delta_m=expected_delta_m,
        object_up=np.array([0.0, -1.0, 0.0]),
        minimum_wait_s=args.between_pass_wait_s,
        translation_tolerance_m=args.lift_translation_tolerance_m,
        rotation_tolerance_deg=args.lift_rotation_tolerance_deg,
        grab_sample=grab_transition_sample,
        operator_confirmed=lambda: confirmation.is_set(),
        elapsed_s=lambda: time.monotonic() - transition_started,
        reset_confirmation=reset_confirmation,
    )
    print(
        f"Lift accepted: {metrics['vertical_translation_m'] * 1000.0:.2f} mm "
        f"vertical, {metrics['lateral_error_m'] * 1000.0:.2f} mm lateral, "
        f"{metrics['rotation_error_deg']:.2f} deg."
    )
    return {
        "from_pass_index": from_result.pass_index,
        "to_pass_index": target_pass_index,
        "target_height_offset_m": target_height_m,
        "minimum_wait_s": args.between_pass_wait_s,
        "accepted_pose_timestamp_ns": int(sample.get("timestamp_ns", 0)),
        "metrics": metrics,
    }


def save_session_files(
    *,
    args: argparse.Namespace,
    offsets: tuple[float, ...],
    trajectory: list[dict],
    passes: list[PassResult],
    transitions: list[dict],
    status: str,
    failure_reason: str | None,
    sdk_version: str,
) -> None:
    """Persist one top-level trajectory and multilevel session manifest."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "vslam_trajectory.jsonl").write_text(
        "".join(
            json.dumps(sample, allow_nan=False) + "\n" for sample in trajectory
        ),
        encoding="utf-8",
    )
    valid_count = sum(
        sample.get("tracking_state") == "OK"
        and sample.get("odometry_status") == "OK"
        for sample in trajectory
    )
    timestamps = [
        int(sample["timestamp_ns"])
        for sample in trajectory
        if int(sample.get("timestamp_ns", 0)) > 0
    ]
    duration_s = (
        (timestamps[-1] - timestamps[0]) / 1_000_000_000.0
        if len(timestamps) > 1
        else 0.0
    )
    center = object_center_from_orbit(args.radius_m, args.camera_yaw_deg)
    manifest = {
        "schema_version": 2,
        "capture_mode": "continuous_multilevel_vslam",
        "status": status,
        "failure_reason": failure_reason,
        "zed_sdk_version": sdk_version,
        "tracking_mode": "GEN_3",
        "imu_fusion": bool(args.vslam_use_imu),
        "area_memory": True,
        "reference_frame": "WORLD",
        "world_origin": "initial_left_camera",
        "coordinate_system": "IMAGE",
        "radius_m": args.radius_m,
        "camera_yaw_deg": args.camera_yaw_deg,
        "object_center_vslam_world_m": center.tolist(),
        "object_up_vslam_world": [0.0, -1.0, 0.0],
        "height_offsets_m": list(offsets),
        "between_pass_wait_s": args.between_pass_wait_s,
        "lift_translation_tolerance_m": args.lift_translation_tolerance_m,
        "lift_rotation_tolerance_deg": args.lift_rotation_tolerance_deg,
        "step_deg": args.step_deg,
        "motor_rpm": args.motor_rpm,
        "first_pass_direction": args.first_pass_direction,
        "direction_policy": "alternate_each_pass",
        "captures_per_pass": len(continuous_capture_angles(args.step_deg)),
        "capture_count": sum(result.capture_count for result in passes),
        "passes": [asdict(result) for result in passes],
        "height_transitions": transitions,
        "trajectory_samples": len(trajectory),
        "trajectory_duration_s": duration_s,
        "effective_tracking_fps": (
            (len(timestamps) - 1) / duration_s if duration_s > 0.0 else 0.0
        ),
        "valid_tracking_samples": valid_count,
        "valid_tracking_fraction": (
            valid_count / len(trajectory) if trajectory else 0.0
        ),
        "depth_confidence_scale": {"best": 0, "worst": 100},
        "sdk_confidence_threshold": 100,
        "sdk_texture_confidence_threshold": 100,
    }
    (args.out_dir / "scan_session.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    """Run all height passes in one uninterrupted ZED tracking session."""
    args = parse_args(argv)
    offsets = validate_args(args)
    existing_captures = list(args.out_dir.glob("pass_*/angle_*.json"))
    if existing_captures:
        raise RuntimeError(
            f"{args.out_dir} already contains multilevel captures; use a new output directory"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.camera_resolution = resolution_from_name(args.resolution)
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.depth_minimum_distance = args.min_depth_m
    init.depth_maximum_distance = args.max_depth_m

    zed = sl.Camera()
    open_status = zed.open(init)
    if open_status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {open_status}")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 100
    runtime.texture_confidence_threshold = 100
    point_cloud = sl.Mat()
    depth_mat = sl.Mat()
    color_mat = sl.Mat()
    confidence_mat = sl.Mat()
    trajectory: list[dict] = []
    passes: list[PassResult] = []
    transitions: list[dict] = []
    status = "failed"
    failure_reason: str | None = None
    tracking_enabled = False
    connection = None
    reader_stop = Event()
    reader: Thread | None = None
    serial_lines: Queue[tuple[int, str]] = Queue()
    sdk_version = str(zed.get_sdk_version())

    def request_stop() -> None:
        if connection is None:
            return
        try:
            connection.write(b"stop\n")
            connection.flush()
            print("TX stop")
        except Exception:
            pass

    try:
        for index in range(args.warmup):
            warmup_status = zed.grab(runtime)
            if warmup_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(
                    f"Warmup frame {index + 1} failed: {warmup_status}"
                )

        tracking = make_tracking_parameters(args.vslam_use_imu)
        tracking_status = zed.enable_positional_tracking(tracking)
        if tracking_status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Could not enable ZED positional tracking: {tracking_status}"
            )
        tracking_enabled = True
        print(
            "ZED GEN_3 tracking enabled; IMU fusion "
            + ("enabled." if args.vslam_use_imu else "disabled.")
        )
        wait_for_valid_vslam_pose(
            zed,
            runtime,
            sl.Pose(),
            trajectory,
            args.tracking_ready_timeout_s,
        )
        for sample in trajectory:
            sample.setdefault("multilevel_phase", "tracking_startup")

        import serial

        connection = serial.Serial(
            args.serial_port,
            args.serial_baud,
            timeout=0.01,
            write_timeout=1,
        )
        if args.serial_reset_delay_s:
            time.sleep(args.serial_reset_delay_s)
        connection.reset_input_buffer()

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
                    serial_lines.put(
                        (
                            time.monotonic_ns(),
                            raw_line.decode("utf-8", errors="replace").strip(),
                        )
                    )

        reader = Thread(
            target=read_serial_lines,
            name="multilevel-motor-serial-reader",
            daemon=True,
        )
        reader.start()

        next_capture_index = 0
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="multilevel-capture-writer"
        ) as executor:
            for pass_index, height_offset_m in enumerate(offsets):
                if pass_index > 0:
                    transition = perform_height_transition(
                        zed=zed,
                        runtime=runtime,
                        args=args,
                        trajectory=trajectory,
                        from_result=passes[-1],
                        target_pass_index=pass_index,
                        target_height_m=height_offset_m,
                    )
                    transitions.append(transition)
                result, next_capture_index = capture_continuous_pass(
                    zed=zed,
                    runtime=runtime,
                    point_cloud=point_cloud,
                    depth_mat=depth_mat,
                    color_mat=color_mat,
                    confidence_mat=confidence_mat,
                    connection=connection,
                    serial_lines=serial_lines,
                    args=args,
                    pass_index=pass_index,
                    height_offset_m=height_offset_m,
                    trajectory=trajectory,
                    executor=executor,
                    first_capture_index=next_capture_index,
                )
                passes.append(result)
        status = "completed"
        print(
            f"Multilevel scan completed: {len(passes)} passes, "
            f"{sum(result.capture_count for result in passes)} captures."
        )
    except BaseException as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        request_stop()
        raise
    finally:
        reader_stop.set()
        if reader is not None and reader.is_alive():
            reader.join(timeout=0.5)
        if connection is not None:
            connection.close()
        try:
            save_session_files(
                args=args,
                offsets=offsets,
                trajectory=trajectory,
                passes=passes,
                transitions=transitions,
                status=status,
                failure_reason=failure_reason,
                sdk_version=sdk_version,
            )
        except Exception as manifest_error:
            print(f"Could not save multilevel session diagnostics: {manifest_error}")
        if tracking_enabled:
            zed.disable_positional_tracking()
        zed.close()


if __name__ == "__main__":
    main()
