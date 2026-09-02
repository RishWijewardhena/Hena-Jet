from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SYNC_WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

from depth_quality import (  # noqa: E402
    cross_view_residual_m,
    fill_rate,
    surface_plane_rms_m,
)


class FillRateTests(unittest.TestCase):
    def test_counts_only_positive_finite_depth(self):
        depth = np.array([[0.0, 0.25], [np.nan, 0.30]], dtype=np.float32)

        self.assertAlmostEqual(fill_rate(depth), 0.5)


class SurfacePlaneRmsTests(unittest.TestCase):
    def test_exact_plane_has_zero_thickness(self):
        rng = np.random.default_rng(0)
        xy = rng.uniform(-0.02, 0.02, size=(500, 2))
        points = np.column_stack((xy, np.full(len(xy), 0.25)))

        self.assertLess(surface_plane_rms_m(points), 1e-12)

    def test_recovers_known_gaussian_thickness(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(-0.02, 0.02, size=(20000, 2))
        z = 0.25 + rng.normal(scale=0.001, size=len(xy))
        points = np.column_stack((xy, z))

        self.assertAlmostEqual(surface_plane_rms_m(points), 0.001, places=4)

    def test_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            surface_plane_rms_m(np.zeros((2, 3)))


class CrossViewResidualTests(unittest.TestCase):
    def test_identical_clouds_have_zero_residual(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform(-0.02, 0.02, size=(1000, 3))

        self.assertAlmostEqual(cross_view_residual_m(pts, pts), 0.0)

    def test_detects_a_known_uniform_offset(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-0.02, 0.02, size=(2000, 3))
        shifted = pts + np.array([0.002, 0.0, 0.0])

        self.assertAlmostEqual(cross_view_residual_m(pts, shifted), 0.002, places=3)

    def test_returns_none_when_clouds_do_not_overlap(self):
        rng = np.random.default_rng(4)
        pts = rng.uniform(-0.02, 0.02, size=(500, 3))
        far = pts + np.array([1.0, 0.0, 0.0])

        self.assertIsNone(cross_view_residual_m(pts, far))


if __name__ == "__main__":
    unittest.main()
