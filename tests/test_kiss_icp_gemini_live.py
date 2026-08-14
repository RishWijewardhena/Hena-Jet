from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "kiss_ICP"
sys.path.insert(0, str(SCRIPT_DIR))

from run_kiss_icp_export_map import (  # noqa: E402
    ColoredVoxelMap,
    colored_point_cloud_from_frame,
    configure_point_cloud_scale,
    parse_args,
    write_xyzrgb_ply,
)


class GeminiScaleTests(unittest.TestCase):
    def test_default_normalizes_gemini_tenth_millimetres_to_metres(self) -> None:
        with patch.object(sys, "argv", ["run_kiss_icp_export_map.py"]):
            args = parse_args()

        raw_records = np.array(
            [[0.0, 0.0, 700.0, 12.0, 128.0, 254.0]],
            dtype=np.float32,
        )

        class FakeDepthFrame:
            def get_depth_scale(self) -> float:
                return 0.1

        class FakePointsFrame:
            def __init__(self, points: np.ndarray) -> None:
                self.points = points

            def get_data(self) -> bytes:
                return self.points.tobytes()

        class FakePointCloudFrame:
            def __init__(self, points: np.ndarray) -> None:
                self.points = points

            def as_points_frame(self) -> FakePointsFrame:
                return FakePointsFrame(self.points)

        class FakePointCloudFilter:
            def set_position_data_scaled(self, scale: float) -> None:
                self.position_scale = scale

            def process(self, depth_frame: object) -> FakePointCloudFrame:
                scaled = raw_records.copy()
                scaled[:, :3] *= self.position_scale
                return FakePointCloudFrame(scaled)

        depth_frame = FakeDepthFrame()
        point_cloud_filter = FakePointCloudFilter()
        depth_scale_mm = configure_point_cloud_scale(point_cloud_filter, depth_frame)
        points, colors = colored_point_cloud_from_frame(
            point_cloud_filter,
            depth_frame,
            point_unit_m=args.point_unit_m,
        )

        self.assertEqual(depth_scale_mm, 0.1)
        np.testing.assert_allclose(points, [[0.0, 0.0, 0.07]])
        np.testing.assert_array_equal(colors, [[12, 128, 254]])


class ColoredPointCloudTests(unittest.TestCase):
    def test_extracts_corresponding_xyz_and_rgb_values(self) -> None:
        raw = np.array(
            [
                [0.0, 0.0, 0.0, 255.0, 255.0, 255.0],
                [10.0, -20.0, 70.0, 12.0, 128.0, 254.0],
            ],
            dtype=np.float32,
        )

        class FakePointsFrame:
            def get_data(self) -> bytes:
                return raw.tobytes()

        class FakePointCloudFrame:
            def as_points_frame(self) -> FakePointsFrame:
                return FakePointsFrame()

        class FakePointCloudFilter:
            def process(self, frame: object) -> FakePointCloudFrame:
                return FakePointCloudFrame()

        points, colors = colored_point_cloud_from_frame(
            FakePointCloudFilter(),
            object(),
            point_unit_m=0.001,
        )

        np.testing.assert_allclose(points, [[0.01, -0.02, 0.07]])
        np.testing.assert_array_equal(colors, [[12, 128, 254]])
        self.assertEqual(colors.dtype, np.uint8)

    def test_writes_rgb_properties_and_values_to_ply(self) -> None:
        points = np.array([[0.01, -0.02, 0.07]], dtype=np.float64)
        colors = np.array([[12, 128, 254]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "colored.ply"
            write_xyzrgb_ply(output, points, colors)
            contents = output.read_text(encoding="utf-8")

        self.assertIn("property uchar red\n", contents)
        self.assertIn("property uchar green\n", contents)
        self.assertIn("property uchar blue\n", contents)
        self.assertTrue(contents.endswith("0.010000 -0.020000 0.070000 12 128 254\n"))


class ColoredVoxelMapTests(unittest.TestCase):
    def test_transforms_points_and_averages_samples_in_each_voxel(self) -> None:
        voxel_map = ColoredVoxelMap(voxel_size_m=0.01)
        identity = np.eye(4)
        voxel_map.update(
            np.array([[0.001, 0.001, 0.001], [0.003, 0.001, 0.001]]),
            np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8),
            identity,
        )
        translated_pose = np.eye(4)
        translated_pose[0, 3] = 0.02
        voxel_map.update(
            np.array([[0.001, 0.001, 0.001]]),
            np.array([[0, 255, 0]], dtype=np.uint8),
            translated_pose,
        )

        points, colors = voxel_map.point_cloud()

        np.testing.assert_allclose(
            points,
            [[0.002, 0.001, 0.001], [0.021, 0.001, 0.001]],
        )
        np.testing.assert_array_equal(colors, [[128, 0, 128], [0, 255, 0]])

    def test_preserves_sample_weights_across_updates(self) -> None:
        voxel_map = ColoredVoxelMap(voxel_size_m=0.01)
        pose = np.eye(4)
        voxel_map.update(
            np.array([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0]]),
            np.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            pose,
        )
        voxel_map.update(
            np.array([[0.003, 0.0, 0.0]]),
            np.array([[255, 0, 0]], dtype=np.uint8),
            pose,
        )

        points, colors = voxel_map.point_cloud()

        np.testing.assert_allclose(points, [[0.002, 0.0, 0.0]])
        np.testing.assert_array_equal(colors, [[85, 0, 0]])


if __name__ == "__main__":
    unittest.main()
