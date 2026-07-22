import unittest

from scripts.capture_zed_angle import (
    confidence_image_to_uint8,
    ContinuousAngleEvent,
    continuous_capture_angles,
    make_tracking_parameters,
    object_center_from_orbit,
    vslam_pose_is_acceptable,
    write_capture_files,
    advance_capture_angle,
    parse_segment_acknowledgement,
    parse_continuous_angle_event,
    pop_due_angle_event,
    serial_start_command,
)


class CaptureZedAngleSerialTests(unittest.TestCase):
    def test_parses_segment_acknowledgement_from_device(self):
        self.assertEqual(parse_segment_acknowledgement("5 degree ok"), 5.0)
        self.assertEqual(parse_segment_acknowledgement("  2.5 DEGREE OK  "), 2.5)

    def test_ignores_non_segment_messages(self):
        for line in ("ready", "started,5,10000", "completed", "stopped", "alert"):
            with self.subTest(line=line):
                self.assertIsNone(parse_segment_acknowledgement(line))

    def test_accumulates_relative_segment_angles(self):
        self.assertEqual(advance_capture_angle(0.0, 5.0), 5.0)
        self.assertEqual(advance_capture_angle(355.0, 5.0), 360.0)

    def test_rejects_invalid_or_overrunning_segment_angles(self):
        with self.assertRaises(ValueError):
            advance_capture_angle(10.0, 0.0)
        with self.assertRaises(ValueError):
            advance_capture_angle(358.0, 5.0)

    def test_formats_firmware_start_command(self):
        self.assertEqual(serial_start_command(5.0, 10000), b"start,5,10000\n")
        self.assertEqual(serial_start_command(2.5, 52100), b"start,2.5,52100\n")

    def test_formats_synchronized_firmware_start_command(self):
        self.assertEqual(
            serial_start_command(5.0, 10000, synchronized=True),
            b"start_sync,5,10000\n",
        )

    def test_formats_continuous_firmware_start_command(self):
        self.assertEqual(
            serial_start_command(
                5.0, 10000, continuous=True, motor_rpm=0.5
            ),
            b"start_continuous,5,10000,0.5\n",
        )

    def test_parses_cumulative_continuous_angle_event(self):
        self.assertEqual(
            parse_continuous_angle_event("angle_ok,125,3472"),
            (125.0, 3472),
        )
        self.assertIsNone(parse_continuous_angle_event("5 degree ok"))

    def test_continuous_capture_angles_include_360_but_not_runout(self):
        self.assertEqual(continuous_capture_angles(5.0), tuple(range(5, 361, 5)))
        self.assertEqual(continuous_capture_angles(7.0)[-1], 360.0)

    def test_selects_one_first_frame_event_and_rejects_backlog(self):
        pending = [ContinuousAngleEvent(5.0, 139, 100)]
        self.assertIsNone(pop_due_angle_event(pending, 99))
        self.assertEqual(pop_due_angle_event(pending, 100).angle_deg, 5.0)
        self.assertEqual(pending, [])

        backlog = [
            ContinuousAngleEvent(5.0, 139, 100),
            ContinuousAngleEvent(10.0, 278, 110),
        ]
        with self.assertRaises(RuntimeError):
            pop_due_angle_event(backlog, 120)

    def test_tracking_parameters_toggle_imu_without_changing_world_origin(self):
        camera_only = make_tracking_parameters(False)
        visual_inertial = make_tracking_parameters(True)

        self.assertFalse(camera_only.enable_imu_fusion)
        self.assertTrue(visual_inertial.enable_imu_fusion)
        self.assertFalse(camera_only.set_gravity_as_origin)
        self.assertFalse(visual_inertial.set_gravity_as_origin)

    def test_converts_confidence_map_to_bounded_uint8(self):
        import numpy as np

        values = np.array([[0.0, 42.4, 100.0, np.nan, 120.0]], dtype=np.float32)
        result = confidence_image_to_uint8(values)
        np.testing.assert_array_equal(result, [[0, 42, 100, 100, 100]])
        self.assertEqual(result.dtype, np.uint8)

    def test_derives_object_center_in_initial_image_camera_frame(self):
        import numpy as np

        center = object_center_from_orbit(0.192, -6.0)
        np.testing.assert_allclose(center, [0.020069, 0.0, 0.190948], atol=1e-6)

    def test_vslam_pose_requires_ok_status_finite_covariance_and_rigid_matrix(self):
        import numpy as np

        pose = np.eye(4)
        covariance = np.eye(6)
        self.assertTrue(vslam_pose_is_acceptable(pose, "OK", "OK", covariance))
        self.assertFalse(vslam_pose_is_acceptable(pose, "UNAVAILABLE", "OK", covariance))
        bad_pose = pose.copy()
        bad_pose[0, 0] = 2.0
        self.assertFalse(vslam_pose_is_acceptable(bad_pose, "OK", "OK", covariance))
        covariance[0, 0] = np.nan
        self.assertFalse(vslam_pose_is_acceptable(pose, "OK", "OK", covariance))

    def test_writer_persists_confidence_and_json_safe_metadata(self):
        import json
        import numpy as np
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_capture_files(
                root / "capture.npz",
                root / "capture.ply",
                root / "capture.json",
                np.array([[0.0, 0.0, 0.2]], dtype=np.float32),
                np.array([[255, 255, 255]], dtype=np.uint8),
                np.array([[0.2]], dtype=np.float32),
                np.array([[[255, 255, 255]]], dtype=np.uint8),
                np.array([[12]], dtype=np.uint8),
                {"tracking_state": "OK"},
            )
            with np.load(root / "capture.npz") as data:
                np.testing.assert_array_equal(data["confidence_image"], [[12]])
            self.assertEqual(
                json.loads((root / "capture.json").read_text())["tracking_state"],
                "OK",
            )


if __name__ == "__main__":
    unittest.main()
