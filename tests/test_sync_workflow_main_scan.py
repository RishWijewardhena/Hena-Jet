from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

import main_scan
import test_radius


class DepthFusionTests(unittest.TestCase):
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
            ]
        )

        metadata = main_scan.build_scan_metadata(
            args,
            active_disparity=256,
            captured_angles=[0.0, 10.0],
        )

        self.assertEqual(metadata["orbit_radius_m"], 0.1175)
        self.assertEqual(metadata["orbit_axis"], [1.0, 0.0, 0.0])
        self.assertEqual(metadata["capture"]["frames_per_angle"], 5)
        self.assertEqual(metadata["capture"]["depth_range_m"], [0.05, 0.3])
        self.assertEqual(metadata["captured_angles_deg"], [0.0, 10.0])


class RadiusCalibrationCaptureTests(unittest.TestCase):
    def test_default_capture_observes_a_rigid_target_from_multiple_angles(self):
        args = test_radius.parse_args([])

        self.assertGreaterEqual(len(args.angles), 3)
        self.assertIn(0.0, args.angles)
        self.assertEqual(args.frames_per_angle, 5)


if __name__ == "__main__":
    unittest.main()
