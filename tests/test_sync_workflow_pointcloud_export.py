from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SYNC_WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from pointcloud_export import backproject_to_points  # noqa: E402


INTRINSICS = {"width": 4, "height": 3, "fx": 400.0, "fy": 400.0, "cx": 1.5, "cy": 1.0}


class BackprojectTests(unittest.TestCase):
    def test_preserves_sub_millimetre_depth_exactly(self):
        # 0.2503771 m would truncate to 0.250 m through a uint16 millimetre image.
        depth = np.full((3, 4), 0.2503771, dtype=np.float32)
        color = np.zeros((3, 4, 3), dtype=np.uint8)

        points, _ = backproject_to_points(depth, color, INTRINSICS)

        self.assertEqual(len(points), 12)
        np.testing.assert_allclose(points[:, 2], 0.2503771, rtol=0, atol=1e-7)

    def test_principal_ray_maps_to_the_optical_axis(self):
        depth = np.zeros((3, 4), dtype=np.float32)
        depth[1, 1] = 0.30  # pixel (u=1, v=1); cx=1.5, cy=1.0
        color = np.zeros((3, 4, 3), dtype=np.uint8)

        points, _ = backproject_to_points(depth, color, INTRINSICS)

        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(
            points[0], [(1 - 1.5) / 400.0 * 0.30, 0.0, 0.30], atol=1e-12
        )

    def test_drops_invalid_and_truncated_depth(self):
        depth = np.array(
            [[0.0, 0.25, np.nan, 0.9], [0.25, -1.0, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            dtype=np.float32,
        )
        color = np.zeros((3, 4, 3), dtype=np.uint8)

        points, _ = backproject_to_points(depth, color, INTRINSICS, depth_trunc_m=0.5)

        self.assertEqual(len(points), 8)

    def test_returns_colors_normalised_and_aligned_to_points(self):
        depth = np.zeros((3, 4), dtype=np.float32)
        depth[0, 0] = 0.25
        depth[2, 3] = 0.25
        color = np.zeros((3, 4, 3), dtype=np.uint8)
        color[0, 0] = (255, 0, 0)
        color[2, 3] = (0, 0, 255)

        points, colors = backproject_to_points(depth, color, INTRINSICS)

        self.assertEqual(len(points), 2)
        np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(colors[1], [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
