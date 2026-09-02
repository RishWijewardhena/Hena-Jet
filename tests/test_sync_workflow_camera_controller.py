from __future__ import annotations

import sys
import unittest
from pathlib import Path


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from camera_controller import (
    CameraController,
    aligned_pointcloud_intrinsics,
    configure_disparity_search_range,
)


class FakeDevice:
    def __init__(self, current_mode: int, readback_mode: int | None = None):
        self.current_mode = current_mode
        self.readback_mode = readback_mode
        self.set_values: list[int] = []

    def get_int_property(self, _property_id) -> int:
        if self.set_values and self.readback_mode is not None:
            return self.readback_mode
        return self.current_mode

    def set_int_property(self, _property_id, value: int) -> None:
        self.set_values.append(value)
        self.current_mode = value


class TestDisparitySearchRange(unittest.TestCase):
    def test_256_pixels_uses_sdk_mode_2_and_verifies_readback(self):
        device = FakeDevice(current_mode=1)

        active_pixels = configure_disparity_search_range(device, "256")

        self.assertEqual(device.set_values, [2])
        self.assertEqual(active_pixels, 256)

    def test_128_pixels_uses_sdk_mode_1(self):
        device = FakeDevice(current_mode=2)

        active_pixels = configure_disparity_search_range(device, "128")

        self.assertEqual(device.set_values, [1])
        self.assertEqual(active_pixels, 128)

    def test_matching_mode_does_not_rewrite_device_property(self):
        device = FakeDevice(current_mode=2)

        active_pixels = configure_disparity_search_range(device, "256")

        self.assertEqual(device.set_values, [])
        self.assertEqual(active_pixels, 256)

    def test_mismatched_readback_is_a_hard_failure(self):
        device = FakeDevice(current_mode=1, readback_mode=1)

        with self.assertRaisesRegex(RuntimeError, "requested 256.*reported 128"):
            configure_disparity_search_range(device, "256")


class TestAlignedPointCloudIntrinsics(unittest.TestCase):
    def test_depth_to_color_alignment_uses_color_intrinsics(self):
        class Intrinsic:
            width = 848
            height = 530
            fx = 910.0
            fy = 911.0
            cx = 423.0
            cy = 264.0

        class DifferentDepthIntrinsic(Intrinsic):
            fx = 780.0
            fy = 781.0

        class CameraParameters:
            rgb_intrinsic = Intrinsic()
            depth_intrinsic = DifferentDepthIntrinsic()

        result = aligned_pointcloud_intrinsics(CameraParameters())

        self.assertEqual(result["coordinate_frame"], "color")
        self.assertEqual(result["fx"], 910.0)
        self.assertEqual(result["fy"], 911.0)

    def test_capture_explicitly_processes_each_frameset_through_d2c_alignment(self):
        raw_frames = object()
        aligned_color = object()
        aligned_depth = object()

        class AlignedFrames:
            def as_frame_set(self):
                return self

            def get_color_frame(self):
                return aligned_color

            def get_depth_frame(self):
                return aligned_depth

        aligned_frames = AlignedFrames()

        class FakeAlignFilter:
            def __init__(self):
                self.inputs = []

            def process(self, frames):
                self.inputs.append(frames)
                return aligned_frames

        class FakePipeline:
            def wait_for_frames(self, timeout_ms):
                return None if timeout_ms == 10 else raw_frames

        camera = CameraController()
        camera.pipeline = FakePipeline()
        camera.align_filter = FakeAlignFilter()

        color, depth = camera.capture_aligned_rgbd(timeout_ms=2000)

        self.assertIs(color, aligned_color)
        self.assertIs(depth, aligned_depth)
        self.assertEqual(camera.align_filter.inputs, [raw_frames])


if __name__ == "__main__":
    unittest.main()
