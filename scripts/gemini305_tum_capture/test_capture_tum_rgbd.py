from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from capture_tum_rgbd import (  # noqa: E402
    TumRgbdCapture,
    color_frame_to_rgb,
    depth_frame_to_uint16,
    parse_args,
)


class TumRgbdCaptureTests(unittest.TestCase):
    def test_writes_surfelmeshing_compatible_capture(self) -> None:
        depth = np.array([[0, 700], [701, 702]], dtype=np.uint16)
        color = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [12, 128, 254]],
            ],
            dtype=np.uint8,
        )
        pose = np.eye(4)
        pose[:3, 3] = [0.01, -0.02, 0.03]
        pose[:3, :3] = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            capture_dir = Path(directory) / "capture"
            capture = TumRgbdCapture(
                capture_dir,
                intrinsics=(600.0, 601.0, 423.5, 264.5),
                depth_scale_mm=0.1,
                metadata={"camera": "Orbbec Gemini 305"},
            )
            capture.record(1_700_000_000.25, depth, color, pose)
            capture.finalize()

            saved_depth = cv2.imread(
                str(capture_dir / "depth" / "1700000000.250000.png"),
                cv2.IMREAD_UNCHANGED,
            )
            saved_color_bgr = cv2.imread(
                str(capture_dir / "rgb" / "1700000000.250000.png"),
                cv2.IMREAD_COLOR,
            )
            calibration = (capture_dir / "calibration.txt").read_text()
            associations = (capture_dir / "associated.txt").read_text()
            trajectory = (capture_dir / "trajectory.txt").read_text()
            capture_info = (capture_dir / "capture_info.txt").read_text()

        np.testing.assert_array_equal(saved_depth, depth)
        np.testing.assert_array_equal(saved_color_bgr[..., ::-1], color)
        self.assertEqual(calibration, "600 601 423.5 264.5\n")
        self.assertIn(
            "1700000000.250000 rgb/1700000000.250000.png "
            "1700000000.250000 depth/1700000000.250000.png\n",
            associations,
        )
        self.assertIn(
            "1700000000.250000 0.01 -0.02 0.03 "
            "0 0 0.707106781 0.707106781\n",
            trajectory,
        )
        self.assertIn("surfelmeshing_depth_scaling=10000\n", capture_info)
        self.assertIn("camera=Orbbec Gemini 305\n", capture_info)

    def test_extracts_uint16_depth_and_rgb_pixels_from_aligned_frames(self) -> None:
        class FakeFrame:
            def __init__(self, image: np.ndarray) -> None:
                self.image = image

            def get_width(self) -> int:
                return self.image.shape[1]

            def get_height(self) -> int:
                return self.image.shape[0]

            def get_data(self) -> bytes:
                return self.image.tobytes()

        depth = np.array([[700, 701]], dtype=np.uint16)
        color = np.array([[[12, 128, 254], [255, 0, 1]]], dtype=np.uint8)

        np.testing.assert_array_equal(depth_frame_to_uint16(FakeFrame(depth)), depth)
        np.testing.assert_array_equal(color_frame_to_rgb(FakeFrame(color)), color)

    def test_refuses_to_overwrite_a_nonempty_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture_dir = Path(directory) / "capture"
            capture_dir.mkdir()
            (capture_dir / "existing.txt").write_text("keep me")

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                TumRgbdCapture(
                    capture_dir,
                    intrinsics=(600.0, 601.0, 423.5, 264.5),
                    depth_scale_mm=0.1,
                    metadata={},
                )

    def test_refuses_to_replace_a_file_with_a_capture_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture_path = Path(directory) / "capture"
            capture_path.write_text("keep me")

            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                TumRgbdCapture(
                    capture_path,
                    intrinsics=(600.0, 601.0, 423.5, 264.5),
                    depth_scale_mm=0.1,
                    metadata={},
                )

    def test_cli_rejects_an_output_path_that_is_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture_path = Path(directory) / "capture"
            capture_path.write_text("keep me")

            with patch.object(
                sys,
                "argv",
                ["capture_tum_rgbd.py", "--out", str(capture_path)],
            ):
                with self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
