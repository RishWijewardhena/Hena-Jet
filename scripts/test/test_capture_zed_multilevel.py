import math
import unittest
from pathlib import Path

import numpy as np

from scripts.capture_zed_multilevel import (
    ContinuousPassProgress,
    capture_metadata_for_pass,
    direction_for_pass,
    lift_pose_metrics,
    normalize_height_offsets_m,
    parse_args,
    pass_directory,
    transition_is_ready,
    wait_for_validated_lift,
)


class CaptureZedMultilevelTests(unittest.TestCase):
    def test_accepts_three_strictly_increasing_absolute_height_offsets(self):
        self.assertEqual(
            normalize_height_offsets_m([0.0, 0.02, 0.04]),
            (0.0, 0.02, 0.04),
        )

    def test_rejects_invalid_height_sequences(self):
        invalid_sequences = (
            [0.02, 0.04],
            [0.0],
            [0.0, 0.02, 0.02],
            [0.0, -0.02],
            [0.0, math.nan],
        )
        for offsets in invalid_sequences:
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                normalize_height_offsets_m(offsets)

    def test_builds_stable_pass_directory_names_in_millimetres(self):
        root = Path("captures/hand")
        self.assertEqual(
            pass_directory(root, 0, 0.0),
            root / "pass_00_height_000mm",
        )
        self.assertEqual(
            pass_directory(root, 2, 0.04),
            root / "pass_02_height_040mm",
        )

    def test_transition_requires_both_minimum_wait_and_operator_confirmation(self):
        self.assertFalse(transition_is_ready(30.0, 30.0, False))
        self.assertFalse(transition_is_ready(29.999, 30.0, True))
        self.assertTrue(transition_is_ready(30.0, 30.0, True))

    def test_accepts_exact_vertical_lift_with_unchanged_orientation(self):
        before = np.eye(4)
        after = np.eye(4)
        after[:3, 3] = [0.0, -0.02, 0.0]

        metrics = lift_pose_metrics(
            before,
            after,
            expected_height_delta_m=0.02,
            object_up=np.array([0.0, -1.0, 0.0]),
            translation_tolerance_m=0.002,
            rotation_tolerance_deg=1.0,
        )

        self.assertTrue(metrics["accepted"])
        self.assertAlmostEqual(metrics["vertical_translation_m"], 0.02)
        self.assertAlmostEqual(metrics["vertical_error_m"], 0.0)
        self.assertAlmostEqual(metrics["lateral_error_m"], 0.0)
        self.assertAlmostEqual(metrics["rotation_error_deg"], 0.0)

    def test_rejects_excess_vertical_lateral_or_rotation_error(self):
        before = np.eye(4)
        cases = []

        vertical = np.eye(4)
        vertical[:3, 3] = [0.0, -0.017, 0.0]
        cases.append(vertical)

        lateral = np.eye(4)
        lateral[:3, 3] = [0.003, -0.02, 0.0]
        cases.append(lateral)

        rotated = np.eye(4)
        angle = math.radians(1.1)
        rotated[:3, :3] = [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
        rotated[:3, 3] = [0.0, -0.02, 0.0]
        cases.append(rotated)

        for candidate in cases:
            with self.subTest(candidate=candidate):
                metrics = lift_pose_metrics(
                    before,
                    candidate,
                    expected_height_delta_m=0.02,
                    object_up=np.array([0.0, -1.0, 0.0]),
                    translation_tolerance_m=0.002,
                    rotation_tolerance_deg=1.0,
                )
                self.assertFalse(metrics["accepted"])

    def test_keeps_grabbing_while_waiting_for_delay_and_confirmation(self):
        before = np.eye(4)
        after = np.eye(4)
        after[:3, 3] = [0.0, -0.02, 0.0]
        times = iter([0.0, 10.0, 20.0, 30.0])
        grabs = []

        def grab_sample():
            grabs.append(len(grabs))
            return {"camera_to_vslam_world": after.tolist()}, True

        sample, metrics = wait_for_validated_lift(
            before_pose=before,
            expected_height_delta_m=0.02,
            object_up=np.array([0.0, -1.0, 0.0]),
            minimum_wait_s=30.0,
            translation_tolerance_m=0.002,
            rotation_tolerance_deg=1.0,
            grab_sample=grab_sample,
            operator_confirmed=lambda: True,
            elapsed_s=lambda: next(times),
            reset_confirmation=lambda _: None,
        )

        self.assertEqual(len(grabs), 4)
        self.assertTrue(metrics["accepted"])
        np.testing.assert_allclose(sample["camera_to_vslam_world"], after)

    def test_aborts_lift_transition_on_any_invalid_tracking_sample(self):
        with self.assertRaisesRegex(RuntimeError, "tracking became invalid"):
            wait_for_validated_lift(
                before_pose=np.eye(4),
                expected_height_delta_m=0.02,
                object_up=np.array([0.0, -1.0, 0.0]),
                minimum_wait_s=30.0,
                translation_tolerance_m=0.002,
                rotation_tolerance_deg=1.0,
                grab_sample=lambda: ({"tracking_state": "UNAVAILABLE"}, False),
                operator_confirmed=lambda: False,
                elapsed_s=lambda: 0.0,
                reset_confirmation=lambda _: None,
            )

    def test_rejected_lift_requests_reposition_and_keeps_tracking(self):
        before = np.eye(4)
        too_low = np.eye(4)
        too_low[:3, 3] = [0.0, -0.015, 0.0]
        corrected = np.eye(4)
        corrected[:3, 3] = [0.0, -0.02, 0.0]
        samples = iter((too_low, corrected))
        rejected = []

        sample, metrics = wait_for_validated_lift(
            before_pose=before,
            expected_height_delta_m=0.02,
            object_up=np.array([0.0, -1.0, 0.0]),
            minimum_wait_s=30.0,
            translation_tolerance_m=0.002,
            rotation_tolerance_deg=1.0,
            grab_sample=lambda: (
                {"camera_to_vslam_world": next(samples).tolist()},
                True,
            ),
            operator_confirmed=lambda: True,
            elapsed_s=lambda: 30.0,
            reset_confirmation=rejected.append,
        )

        self.assertEqual(len(rejected), 1)
        self.assertTrue(metrics["accepted"])
        np.testing.assert_allclose(sample["camera_to_vslam_world"], corrected)

    def test_default_pipeline_mode_accepts_after_three_settling_frames(self):
        before = np.eye(4)
        measured = np.eye(4)
        measured[:3, 3] = [0.011, -0.011, 0.0]
        grabs = []

        _, metrics = wait_for_validated_lift(
            before_pose=before,
            expected_height_delta_m=0.02,
            object_up=np.array([0.0, -1.0, 0.0]),
            minimum_wait_s=0.0,
            translation_tolerance_m=0.002,
            rotation_tolerance_deg=1.0,
            grab_sample=lambda: (
                grabs.append(len(grabs))
                or {"camera_to_vslam_world": measured.tolist()},
                True,
            ),
            operator_confirmed=lambda: True,
            elapsed_s=lambda: 0.0,
            reset_confirmation=lambda _: self.fail(
                "Pipeline mode must not request repositioning"
            ),
            validate_pose=False,
            settling_frames=3,
        )

        self.assertEqual(len(grabs), 3)
        self.assertTrue(metrics["accepted"])
        self.assertTrue(metrics["validation_skipped"])
        self.assertFalse(metrics["pose_validation_passed"])

    def test_pipeline_mode_still_aborts_on_tracking_loss(self):
        with self.assertRaisesRegex(RuntimeError, "tracking became invalid"):
            wait_for_validated_lift(
                before_pose=np.eye(4),
                expected_height_delta_m=0.02,
                object_up=np.array([0.0, -1.0, 0.0]),
                minimum_wait_s=0.0,
                translation_tolerance_m=0.002,
                rotation_tolerance_deg=1.0,
                grab_sample=lambda: ({"tracking_state": "UNAVAILABLE"}, False),
                operator_confirmed=lambda: True,
                elapsed_s=lambda: 0.0,
                reset_confirmation=lambda _: None,
                validate_pose=False,
                settling_frames=3,
            )

    def test_capture_metadata_records_pass_and_absolute_height(self):
        metadata = capture_metadata_for_pass(
            {"tracking_state": "OK"},
            pass_index=2,
            height_offset_m=0.04,
            capture_index=144,
            motor_direction="forward",
        )
        self.assertEqual(
            metadata,
            {
                "tracking_state": "OK",
                "multilevel_pass_index": 2,
                "height_offset_m": 0.04,
                "multilevel_capture_index": 144,
                "motor_direction": "forward",
            },
        )

    def test_cli_defaults_to_three_heights_continuous_motion_and_imu(self):
        args = parse_args(
            [
                "--radius-m",
                "0.192",
                "--serial-port",
                "/dev/ttyACM0",
                "--pulses-per-revolution",
                "10000",
            ]
        )
        self.assertEqual(args.height_offsets_m, [0.0, 0.02, 0.04])
        self.assertEqual(args.step_deg, 5.0)
        self.assertEqual(args.between_pass_wait_s, 30.0)
        self.assertTrue(args.vslam_use_imu)
        self.assertEqual(args.first_pass_direction, "forward")
        self.assertFalse(args.validate_lift_pose)
        self.assertEqual(args.lift_settling_frames, 3)

        strict_args = parse_args(
            [
                "--radius-m",
                "0.192",
                "--serial-port",
                "/dev/ttyACM0",
                "--pulses-per-revolution",
                "10000",
                "--validate-lift-pose",
            ]
        )
        self.assertTrue(strict_args.validate_lift_pose)

    def test_alternates_motor_direction_to_unwind_the_camera_cable(self):
        self.assertEqual(
            [direction_for_pass(index, "forward") for index in range(4)],
            ["forward", "reverse", "forward", "reverse"],
        )
        self.assertEqual(direction_for_pass(0, "reverse"), "reverse")
        self.assertEqual(direction_for_pass(1, "reverse"), "forward")

    def test_pass_progress_associates_one_ordered_event_with_next_frame(self):
        progress = ContinuousPassProgress(5.0)
        progress.accept_serial_line("angle_ok,5,139", received_ns=100)

        self.assertIsNone(progress.pop_due_event(frame_ready_ns=99))
        event = progress.pop_due_event(frame_ready_ns=100)
        self.assertEqual(event.angle_deg, 5.0)
        progress.mark_captured()
        self.assertEqual(progress.captured_count, 1)

    def test_pass_progress_rejects_wrong_angle_and_nonincreasing_pulses(self):
        progress = ContinuousPassProgress(5.0)
        with self.assertRaisesRegex(RuntimeError, "Expected continuous angle 5"):
            progress.accept_serial_line("angle_ok,10,139", received_ns=100)

        progress.accept_serial_line("angle_ok,5,139", received_ns=100)
        with self.assertRaisesRegex(RuntimeError, "pulse counts are not increasing"):
            progress.accept_serial_line("angle_ok,10,139", received_ns=200)

    def test_pass_progress_requires_every_event_and_capture_before_completion(self):
        progress = ContinuousPassProgress(180.0)
        progress.accept_serial_line("angle_ok,180,5000", received_ns=100)
        progress.pop_due_event(frame_ready_ns=100)
        progress.mark_captured()
        with self.assertRaisesRegex(RuntimeError, "expected 2"):
            progress.validate_completed()

        progress.accept_serial_line("angle_ok,360,10000", received_ns=300)
        progress.pop_due_event(frame_ready_ns=300)
        progress.mark_captured()
        progress.accept_serial_line("completed", received_ns=400)
        progress.validate_completed()


if __name__ == "__main__":
    unittest.main()
