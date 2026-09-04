from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

import scan_quality_report  # noqa: E402


def curved_patch(*, count=4000, noise_m=0.0, seed=0):
    """A gently curved surface patch, so rigid alignment is well constrained."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.03, 0.03, count)
    y = rng.uniform(-0.03, 0.03, count)
    z = 0.14 + 0.4 * (x**2 - 0.5 * y**2)
    points = np.column_stack([x, y, z])
    if noise_m:
        points = points + rng.normal(scale=noise_m, size=points.shape)
    return points


class NearestNeighbourStatsTests(unittest.TestCase):
    def test_identical_clouds_have_zero_distance(self):
        points = curved_patch()
        stats = scan_quality_report.nearest_neighbour_stats(
            points, points, max_distance_m=0.008,
        )
        self.assertAlmostEqual(stats["median_m"], 0.0, places=9)
        self.assertAlmostEqual(stats["paired_fraction"], 1.0, places=9)

    def test_a_known_offset_is_recovered(self):
        points = curved_patch()
        shifted = points + np.array([0.001, 0.0, 0.0])
        stats = scan_quality_report.nearest_neighbour_stats(
            shifted, points, max_distance_m=0.008,
        )
        self.assertLess(stats["median_m"], 0.0011)

    def test_disjoint_clouds_are_an_error(self):
        with self.assertRaises(RuntimeError):
            scan_quality_report.nearest_neighbour_stats(
                curved_patch(),
                curved_patch() + np.array([1.0, 0.0, 0.0]),
                max_distance_m=0.008,
            )


class RepeatabilityReportTests(unittest.TestCase):
    def test_a_repeatable_scan_reports_small_residuals(self):
        report = scan_quality_report.repeatability_report(
            curved_patch(noise_m=0.0002, seed=1),
            curved_patch(noise_m=0.0002, seed=2),
            max_distance_m=0.008,
        )
        self.assertLess(report["as_reconstructed"]["median_m"], 0.001)
        self.assertLess(report["after_rigid_alignment"]["median_m"], 0.001)

    def test_a_frame_shift_shows_up_before_alignment_and_not_after(self):
        """The gap between the two numbers is the point: shape repeats, frame drifts.

        The offset is applied along the surface normal. A tangential shift of the
        same size would barely register, because nearest-neighbour distance
        measures the distance to the *surface*, not to the corresponding point --
        which is also why ICP slides so freely on smooth geometry like a hand.
        """
        base = curved_patch(noise_m=0.0002, seed=3)
        shifted = curved_patch(noise_m=0.0002, seed=4) + np.array([0.0, 0.0, 0.002])
        report = scan_quality_report.repeatability_report(
            shifted, base, max_distance_m=0.008,
        )
        # The floor after alignment is the sampling spacing, not the noise, so
        # assert the drop rather than an absolute residual.
        self.assertGreater(report["as_reconstructed"]["median_m"], 0.0015)
        self.assertLess(
            report["after_rigid_alignment"]["median_m"],
            report["as_reconstructed"]["median_m"] / 2.0,
        )
        self.assertAlmostEqual(report["alignment_translation_m"], 0.002, delta=0.0003)


if __name__ == "__main__":
    unittest.main()
