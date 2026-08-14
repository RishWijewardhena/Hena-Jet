from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "kiss_ICP"
sys.path.insert(0, str(SCRIPT_DIR))

from run_kiss_icp_export_map import prepare_points_meters  # noqa: E402


class PreparePointsMetersTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
