from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "kiss_ICP"
sys.path.insert(0, str(SCRIPT_DIR))

from run_kiss_icp_export_map import (  # noqa: E402
    configure_point_cloud_scale,
    parse_args,
    point_cloud_from_depth_frame,
    prepare_points_meters,
)


class PreparePointsMetersTests(unittest.TestCase):
    def test_default_normalizes_gemini_tenth_millimetres_to_metres(self) -> None:
        with patch.object(sys, "argv", ["run_kiss_icp_export_map.py"]):
            args = parse_args()

        raw_points = np.array([[0.0, 0.0, 700.0]], dtype=np.float32)

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
                return FakePointCloudFrame(raw_points * self.position_scale)

        depth_frame = FakeDepthFrame()
        point_cloud_filter = FakePointCloudFilter()
        depth_scale_mm = configure_point_cloud_scale(point_cloud_filter, depth_frame)
        points = point_cloud_from_depth_frame(
            point_cloud_filter,
            depth_frame,
            point_unit_m=args.point_unit_m,
        )

        self.assertEqual(depth_scale_mm, 0.1)
        np.testing.assert_allclose(points, [[0.0, 0.0, 0.07]])

    def test_converts_orbbec_millimetres_and_removes_invalid_points(self) -> None:
        raw_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, -20.0, 70.0],
                [np.nan, 1.0, 2.0],
            ],
            dtype=np.float32,
        )

        points = prepare_points_meters(raw_points, point_unit_m=0.001)

        np.testing.assert_allclose(points, [[0.01, -0.02, 0.07]])
        self.assertEqual(points.dtype, np.float64)

    def test_supports_explicit_millimetres_divided_by_ten_units(self) -> None:
        raw_points = np.array([[1.0, -2.0, 7.0]], dtype=np.float32)

        points = prepare_points_meters(raw_points, point_unit_m=0.01)

        np.testing.assert_allclose(points, [[0.01, -0.02, 0.07]])

    def test_rejects_non_positive_unit_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "point_unit_m must be positive"):
            prepare_points_meters(np.ones((1, 3)), point_unit_m=0.0)

    def test_rejects_non_xyz_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape .*N, 3"):
            prepare_points_meters(np.ones((2, 4)), point_unit_m=0.001)


class PointCloudFromDepthFrameTests(unittest.TestCase):
    def test_extracts_sdk_point_frame_and_converts_it_to_metres(self) -> None:
        raw_points = np.array(
            [[0.0, 0.0, 0.0], [10.0, -20.0, 70.0]], dtype=np.float32
        )

        class FakePointsFrame:
            def get_data(self) -> bytes:
                return raw_points.tobytes()

        class FakePointCloudFrame:
            def as_points_frame(self) -> FakePointsFrame:
                return FakePointsFrame()

        class FakePointCloudFilter:
            def process(self, depth_frame: object) -> FakePointCloudFrame:
                self.depth_frame = depth_frame
                return FakePointCloudFrame()

        depth_frame = object()
        point_cloud_filter = FakePointCloudFilter()

        points = point_cloud_from_depth_frame(
            point_cloud_filter,
            depth_frame,
            point_unit_m=0.001,
        )

        np.testing.assert_allclose(points, [[0.01, -0.02, 0.07]])
        self.assertIs(point_cloud_filter.depth_frame, depth_frame)

    def test_rejects_malformed_sdk_point_buffer(self) -> None:
        class FakePointsFrame:
            def get_data(self) -> bytes:
                return np.array([1.0, 2.0], dtype=np.float32).tobytes()

        class FakePointCloudFrame:
            def as_points_frame(self) -> FakePointsFrame:
                return FakePointsFrame()

        class FakePointCloudFilter:
            def process(self, depth_frame: object) -> FakePointCloudFrame:
                return FakePointCloudFrame()

        with self.assertRaisesRegex(RuntimeError, "multiple of three"):
            point_cloud_from_depth_frame(
                FakePointCloudFilter(),
                object(),
                point_unit_m=0.001,
            )


if __name__ == "__main__":
    unittest.main()
