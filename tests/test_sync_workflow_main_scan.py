from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

import main_scan
from calculating_radius import test_radius


class DepthFusionTests(unittest.TestCase):
    def test_capture_profile_defaults_match_controller(self):
        from camera_controller import CameraController
        args = main_scan.parse_args([])
        camera = CameraController()
        self.assertEqual((args.width, args.height, args.disparity), (1280, 800, "128"))
        self.assertEqual((camera.width, camera.height, camera.disparity), (1280, 800, "128"))

    def test_uses_median_valid_depth_and_masks_the_working_range(self):
        frames = [
            np.array([[0.0, 0.11], [0.20, 0.10]], dtype=np.float32),
            np.array([[0.10, 0.12], [0.21, 0.50]], dtype=np.float32),
            np.array([[0.11, 0.90], [0.22, 0.60]], dtype=np.float32),
        ]

        fused = main_scan.fuse_depth_frames(
            frames,
            min_depth_m=0.08,
            max_depth_m=0.35,
        )

        expected = np.array([[0.105, 0.115], [0.21, 0.10]], dtype=np.float32)
        np.testing.assert_allclose(fused, expected, atol=1e-6)

    def test_capture_burst_combines_fresh_rgbd_frames(self):
        class FakeFrame:
            def __init__(self, values, depth_scale_mm=None):
                self.values = np.ascontiguousarray(values)
                self.depth_scale_mm = depth_scale_mm

            def get_width(self):
                return self.values.shape[1]

            def get_height(self):
                return self.values.shape[0]

            def get_data(self):
                return self.values.tobytes()

            def get_depth_scale(self):
                return self.depth_scale_mm

        class FakeCamera:
            def __init__(self):
                color = np.zeros((1, 2, 3), dtype=np.uint8)
                self.frames = [
                    (FakeFrame(color + index), FakeFrame(depth, 1.0))
                    for index, depth in enumerate(
                        (
                            np.array([[100, 200]], dtype=np.uint16),
                            np.array([[110, 210]], dtype=np.uint16),
                            np.array([[120, 220]], dtype=np.uint16),
                        )
                    )
                ]

            def capture_aligned_rgbd(self, timeout_ms):
                self.last_timeout_ms = timeout_ms
                return self.frames.pop(0)

        color, depth, count = main_scan.capture_fused_rgbd(
            FakeCamera(),
            frames_per_angle=3,
            timeout_ms=2000,
            min_depth_m=0.05,
            max_depth_m=0.30,
        )

        self.assertEqual(count, 3)
        np.testing.assert_array_equal(color, np.full((1, 2, 3), 2, dtype=np.uint8))
        np.testing.assert_allclose(depth, [[0.11, 0.21]], atol=1e-6)


class ScanMetadataTests(unittest.TestCase):
    def test_x_positions_default_to_the_existing_single_station(self):
        args = main_scan.parse_args([])

        self.assertEqual(args.x_positions_mm, [150.0])

    def test_registration_crop_defaults_to_10cm(self):
        args = main_scan.parse_args([])

        self.assertEqual(args.registration_crop_radius_m, 0.10)

    def test_automatic_registration_crop_does_not_exceed_the_final_crop(self):
        args = main_scan.parse_args(["--crop-radius-m", "0.075"])

        self.assertEqual(args.registration_crop_radius_m, 0.075)

    def test_explicit_registration_crop_is_preserved(self):
        args = main_scan.parse_args(
            [
                "--crop-radius-m", "0.075",
                "--registration-crop-radius-m", "0.09",
            ]
        )

        self.assertEqual(args.registration_crop_radius_m, 0.09)

    def test_two_station_sequence_captures_a_full_orbit_at_each_x_position(self):
        sequence = main_scan.generate_scan_sequence(10.0, [200.0, 280.0])

        captures = [step for step in sequence if step["capture"]]
        self.assertEqual(len(captures), 74)
        self.assertEqual(
            {step["station_index"] for step in captures},
            {0, 1},
        )
        self.assertEqual(
            sum(step["station_index"] == 0 for step in captures),
            37,
        )
        self.assertEqual(
            sum(step["station_index"] == 1 for step in captures),
            37,
        )

        current_y = 0.0
        for step in sequence:
            if step["kind"] == "move_x":
                self.assertEqual(current_y, 0.0)
            else:
                current_y = step["angle_deg"]

    def test_station_filename_keeps_same_angle_captures_unique(self):
        first = main_scan.capture_filename(0, 200.0, 0.0)
        second = main_scan.capture_filename(1, 280.0, 0.0)

        self.assertEqual(first, "frame_s00_x200.0_y+000.0.ply")
        self.assertEqual(second, "frame_s01_x280.0_y+000.0.ply")
        self.assertNotEqual(first, second)

    def test_records_geometry_and_depth_capture_settings(self):
        args = main_scan.parse_args(
            [
                "--radius-m",
                "0.1175",
                "--step-deg",
                "10",
                "--frames-per-angle",
                "5",
                "--depth-min-m",
                "0.05",
                "--depth-max-m",
                "0.30",
                "--x-positions-mm",
                "200",
            ]
        )

        metadata = main_scan.build_scan_metadata(
            args,
            active_disparity=256,
            captured_angles=[0.0, 10.0],
            captures=[
                {
                    "filename": "frame_s00_x200.0_y+000.0.ply",
                    "station_index": 0,
                    "x_position_mm": 200.0,
                    "x_offset_m": 0.0,
                    "angle_deg": 0.0,
                }
            ],
        )

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["orbit_radius_m"], 0.1175)
        self.assertEqual(metadata["orbit_axis"], [1.0, 0.0, 0.0])
        self.assertEqual(metadata["capture"]["frames_per_angle"], 5)
        self.assertEqual(metadata["capture"]["depth_range_m"], [0.05, 0.3])
        self.assertEqual(metadata["reconstruction"]["crop_radius_m"], 0.15)
        self.assertEqual(
            metadata["reconstruction"]["registration_crop_radius_m"],
            0.10,
        )
        self.assertEqual(metadata["captured_angles_deg"], [0.0, 10.0])
        self.assertEqual(metadata["x_stage"]["positions_mm"], [200.0])
        self.assertEqual(metadata["captures"][0]["x_offset_m"], 0.0)


class RadiusCalibrationCaptureTests(unittest.TestCase):
    def test_default_capture_observes_a_rigid_target_from_multiple_angles(self):
        args = test_radius.parse_args([])

        self.assertGreaterEqual(len(args.angles), 3)
        self.assertIn(0.0, args.angles)
        self.assertEqual(args.frames_per_angle, 10)
        self.assertEqual(args.min_valid_poses_per_angle, 6)
        self.assertEqual(
            args.marker_map,
            Path("outputs/radius_markers/profile_marker_map.json"),
        )

    def test_loads_schema_two_map_with_per_marker_sizes(self):
        marker_map = {
            "schema_version": 2,
            "dictionary": "DICT_5X5_100",
            "marker_sizes_m": [0.018, 0.014, 0.018, 0.014],
            "markers": [{"id": 0, "size_m": 0.018, "corners_m": []}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile_marker_map.json"
            path.write_text(json.dumps(marker_map), encoding="utf-8")

            loaded = test_radius.load_marker_map(path)

        self.assertEqual(loaded["schema_version"], 2)

    def test_rejects_pose_above_the_reprojection_error_limit(self):
        class FakeDetector:
            def detectMarkers(self, _gray):
                return [], np.empty((0, 1), dtype=np.int32), []

        solved_pose = {
            "ok": True,
            "method": "multi-marker-iterative",
            "used_ids": [0, 1],
            "world_to_camera": np.eye(4),
            "rvec": np.zeros((3, 1)),
            "tvec": np.array([[0.0], [0.0], [0.15]]),
            "reprojection_error_px": 2.0,
        }
        with mock.patch.object(test_radius, "solve_profile_pose", return_value=solved_pose):
            pose, _, _ = test_radius.estimate_frame_pose(
                np.zeros((10, 10, 3), dtype=np.uint8),
                {"markers": []},
                FakeDetector(),
                np.eye(3),
                np.zeros((8, 1)),
                np.eye(4),
                max_reprojection_error_px=1.5,
            )

        self.assertFalse(pose["accepted"])
        self.assertIn("exceeds 1.50px", pose["rejection_reason"])

    def _single_marker_pose(self, **overrides):
        """Run estimate_frame_pose over one solved single-marker pose."""
        class FakeDetector:
            def detectMarkers(self, _gray):
                return [], np.empty((0, 1), dtype=np.int32), []

        solved_pose = {
            "ok": True,
            "method": "single-marker-ippe",
            "used_ids": [0],
            "world_to_camera": np.eye(4),
            "rvec": np.zeros((3, 1)),
            "tvec": np.array([[0.0], [0.0], [0.15]]),
            "reprojection_error_px": 0.3,
        }
        kwargs = {"max_reprojection_error_px": 1.5}
        for key in ("min_markers", "min_ippe_error_ratio"):
            if key in overrides:
                kwargs[key] = overrides.pop(key)
        solved_pose.update(overrides)
        with mock.patch.object(test_radius, "solve_profile_pose", return_value=solved_pose):
            pose, _, _ = test_radius.estimate_frame_pose(
                np.zeros((10, 10, 3), dtype=np.uint8),
                {"markers": []},
                FakeDetector(),
                np.eye(3),
                np.zeros((8, 1)),
                np.eye(4),
                **kwargs,
            )
        return pose

    def test_accepts_an_unambiguous_single_marker_pose(self):
        """The four-face profile shows one marker at most angles, so one must do."""
        pose = self._single_marker_pose(ippe_error_ratio=6.0)
        self.assertTrue(pose["accepted"], pose["rejection_reason"])

    def test_rejects_an_ambiguous_single_marker_pose(self):
        pose = self._single_marker_pose(ippe_error_ratio=1.05)
        self.assertFalse(pose["accepted"])
        self.assertIn("ambiguous single-marker pose", pose["rejection_reason"])

    def test_two_marker_faces_can_still_be_demanded_explicitly(self):
        pose = self._single_marker_pose(ippe_error_ratio=6.0, min_markers=2)
        self.assertFalse(pose["accepted"])
        self.assertIn("fewer than 2 mapped markers", pose["rejection_reason"])
        self.assertIn("fewer than 2 mapped markers", pose["rejection_reason"])

    def test_requires_enough_valid_pose_frames_per_angle(self):
        args = test_radius.parse_args(
            ["--frames-per-angle", "2", "--min-valid-poses-per-angle", "3"]
        )

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            test_radius.validate_args(args)


class FuseDepthFramesTests(unittest.TestCase):
    def test_requires_a_minimum_number_of_valid_samples(self):
        frames = [
            np.array([[0.25, 0.25]], dtype=np.float32),
            np.array([[0.25, 0.00]], dtype=np.float32),
            np.array([[0.25, 0.00]], dtype=np.float32),
        ]

        fused = main_scan.fuse_depth_frames(
            frames, min_depth_m=0.02, max_depth_m=0.35, min_valid_samples=2
        )

        np.testing.assert_allclose(fused, [[0.25, 0.0]])

    def test_defaults_to_accepting_a_single_valid_sample(self):
        frames = [
            np.array([[0.25]], dtype=np.float32),
            np.array([[0.00]], dtype=np.float32),
        ]

        fused = main_scan.fuse_depth_frames(frames, min_depth_m=0.02, max_depth_m=0.35)

        np.testing.assert_allclose(fused, [[0.25]])


if __name__ == "__main__":
    unittest.main()


class RegistrationModeDefaultTests(unittest.TestCase):
    def test_capture_defaults_to_guarded_icp(self):
        """Motor mode writes no edges, so ICP is on for its diagnostics."""
        self.assertEqual(main_scan.parse_args([]).registration_mode, "guarded-icp")

    def test_motor_can_still_be_requested(self):
        args = main_scan.parse_args(["--registration-mode", "motor"])
        self.assertEqual(args.registration_mode, "motor")

    def test_the_mode_reaches_scan_metadata(self):
        args = main_scan.parse_args([])
        metadata = main_scan.build_scan_metadata(
            args, active_disparity=256, captured_angles=[0.0],
        )
        self.assertEqual(metadata["registration_mode"], "guarded-icp")
