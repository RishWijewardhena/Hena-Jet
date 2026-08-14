from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.gemini305_dynamicfusion.capture_rgbd import (
    RawRgbdWriter,
    color_frame_to_rgb,
    depth_frame_to_uint16,
)


class FakeFrame:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.ascontiguousarray(values)

    def get_width(self) -> int:
        return self.values.shape[1]

    def get_height(self) -> int:
        return self.values.shape[0]

    def get_data(self) -> bytes:
        return self.values.tobytes()


class FrameConversionTests(unittest.TestCase):
    def test_copies_raw_depth_buffer(self) -> None:
        expected = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        actual = depth_frame_to_uint16(FakeFrame(expected))
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.shares_memory(actual, expected))

    def test_copies_rgb_buffer(self) -> None:
        expected = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        actual = color_frame_to_rgb(FakeFrame(expected))
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.shares_memory(actual, expected))


class RawRgbdWriterTests(unittest.TestCase):
    def test_writes_raw_trajectory_free_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scan"
            writer = RawRgbdWriter(
                root,
                depth_intrinsics=(100.0, 101.0, 1.0, 0.5),
                depth_scale_mm=0.1,
                camera_metadata={"camera": "Gemini 305", "serial": "test"},
            )

            writer.record(
                1.25,
                np.full((2, 3), 1234, dtype=np.uint16),
                np.zeros((4, 5, 3), dtype=np.uint8),
            )
            writer.finalize()

            info = (root / "capture_info.txt").read_text(encoding="utf-8")
            camera = json.loads((root / "camera.json").read_text(encoding="utf-8"))
            self.assertIn("pose_source=none", info)
            self.assertIn("depth_processing=raw", info)
            self.assertIn("frame_count=1", info)
            self.assertEqual(camera["depthScaleMm"], 0.1)
            self.assertTrue((root / "depth/1.250000.png").is_file())
            self.assertTrue((root / "rgb/1.250000.png").is_file())
            self.assertFalse((root / "trajectory.txt").exists())

    def test_refuses_to_write_into_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "existing.txt").write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                RawRgbdWriter(
                    root,
                    depth_intrinsics=(100.0, 100.0, 1.0, 1.0),
                    depth_scale_mm=1.0,
                    camera_metadata={},
                )

    def test_requires_strictly_increasing_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = RawRgbdWriter(
                Path(directory) / "scan",
                depth_intrinsics=(100.0, 100.0, 1.0, 1.0),
                depth_scale_mm=1.0,
                camera_metadata={},
            )
            depth = np.ones((2, 2), dtype=np.uint16)
            color = np.zeros((2, 2, 3), dtype=np.uint8)
            writer.record(2.0, depth, color)

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                writer.record(2.0, depth, color)


if __name__ == "__main__":
    unittest.main()
