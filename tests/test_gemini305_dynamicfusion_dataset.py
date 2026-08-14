from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.gemini305_dynamicfusion.dataset import TumRgbdDataset


class TumRgbdDatasetTests(unittest.TestCase):
    def test_loads_associated_frames_and_calibration_without_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            cv2.imwrite(str(root / "rgb/1.000000.png"), np.zeros((2, 3, 3), np.uint8))
            cv2.imwrite(str(root / "depth/1.000000.png"), np.full((2, 3), 1200, np.uint16))
            (root / "associated.txt").write_text(
                "# timestamp rgb depth_timestamp depth\n"
                "1.000000 rgb/1.000000.png 1.000000 depth/1.000000.png\n",
                encoding="utf-8",
            )
            (root / "calibration.txt").write_text("100 101 1 0.5\n", encoding="utf-8")
            (root / "capture_info.txt").write_text(
                "depth_scale_mm=0.1\npose_source=KISS-ICP\n", encoding="utf-8"
            )

            dataset = TumRgbdDataset(root)
            frame = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.intrinsics, (100.0, 101.0, 1.0, 0.5))
            self.assertAlmostEqual(dataset.depth_scale_m, 0.0001)
            self.assertEqual(frame.depth.shape, (2, 3))
            self.assertEqual(frame.color_rgb.shape, (2, 3, 3))
            self.assertFalse(hasattr(frame, "pose"))

    def test_rejects_unequal_rgb_and_depth_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "associated.txt").write_text(
                "1.000000 rgb/a.png 1.100000 depth/a.png\n", encoding="utf-8"
            )
            (root / "calibration.txt").write_text("100 100 1 1\n", encoding="utf-8")
            (root / "capture_info.txt").write_text(
                "depth_scale_mm=1\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "timestamps"):
                TumRgbdDataset(root)

    def test_allows_native_rgb_and_depth_to_have_different_resolutions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            cv2.imwrite(str(root / "rgb/a.png"), np.zeros((4, 5, 3), np.uint8))
            cv2.imwrite(str(root / "depth/a.png"), np.ones((2, 3), np.uint16))
            (root / "associated.txt").write_text(
                "1 rgb/a.png 1 depth/a.png\n", encoding="utf-8"
            )
            (root / "calibration.txt").write_text("100 100 1 1\n", encoding="utf-8")
            (root / "capture_info.txt").write_text(
                "depth_scale_mm=1\n", encoding="utf-8"
            )

            frame = TumRgbdDataset(root)[0]

            self.assertEqual(frame.depth.shape, (2, 3))
            self.assertEqual(frame.color_rgb.shape, (4, 5, 3))


if __name__ == "__main__":
    unittest.main()
