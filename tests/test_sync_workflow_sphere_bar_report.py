from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SYNC_WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "sync_workflow"
)
sys.path.insert(0, str(SYNC_WORKFLOW_DIR))

import sphere_bar_report  # noqa: E402


def sphere_points(centre, radius_m, *, count=2000, noise_m=0.0, seed=0):
    """Sample a sphere surface with optional isotropic Gaussian noise."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius_m + rng.normal(scale=noise_m, size=count) if noise_m else radius_m
    return np.asarray(centre, dtype=float) + directions * np.reshape(radii, (-1, 1))


class FitSphereTests(unittest.TestCase):
    def test_recovers_a_noiseless_sphere(self):
        centre, radius, residuals = sphere_bar_report.fit_sphere(
            sphere_points([0.01, -0.02, 0.14], 0.0125)
        )
        np.testing.assert_allclose(centre, [0.01, -0.02, 0.14], atol=1e-9)
        self.assertAlmostEqual(radius, 0.0125, places=9)
        self.assertLess(np.abs(residuals).max(), 1e-9)

    def test_averaging_beats_the_point_noise(self):
        """The whole point of a ball bar: the centre is better than the points."""
        centre, radius, _ = sphere_bar_report.fit_sphere(
            sphere_points([0.0, 0.0, 0.14], 0.0125, noise_m=0.0015, seed=7)
        )
        centre_error_m = float(np.linalg.norm(centre - np.array([0.0, 0.0, 0.14])))
        self.assertLess(centre_error_m, 0.0002)
        self.assertAlmostEqual(radius, 0.0125, delta=0.0002)

    def test_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            sphere_bar_report.fit_sphere(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            sphere_bar_report.fit_sphere(np.zeros((10, 2)))


class BuildReportTests(unittest.TestCase):
    CERTIFIED_M = 0.100
    DIAMETER_M = 0.025

    def _bar(self, measured_distance_m, *, noise_m=0.0, seed=1):
        return np.vstack([
            sphere_points([0.0, 0.0, 0.14], self.DIAMETER_M / 2, noise_m=noise_m, seed=seed),
            sphere_points(
                [measured_distance_m, 0.0, 0.14],
                self.DIAMETER_M / 2,
                noise_m=noise_m,
                seed=seed + 1,
            ),
        ])

    def _report(self, points):
        return sphere_bar_report.build_report(
            points,
            certified_distance_m=self.CERTIFIED_M,
            sphere_diameter_m=self.DIAMETER_M,
            diameter_tolerance_m=0.008,
            min_sphere_points=200,
            cluster_eps_m=0.004,
        )

    def test_a_correct_bar_reports_no_scale_error(self):
        report = self._report(self._bar(self.CERTIFIED_M))
        self.assertAlmostEqual(report["measured_distance_m"], self.CERTIFIED_M, places=6)
        self.assertAlmostEqual(report["distance_error_m"], 0.0, places=6)
        self.assertEqual(len(report["spheres"]), 2)

    def test_a_one_percent_scale_error_is_detected(self):
        """A 1% orbit-radius error shows up as a 1% length error."""
        report = self._report(self._bar(self.CERTIFIED_M * 1.01))
        self.assertAlmostEqual(report["distance_error_m"], 0.001, places=5)
        self.assertAlmostEqual(report["scale_error_ratio"], 0.01, places=5)
        self.assertAlmostEqual(
            report["implied_radius_correction_ratio"], 0.99, places=5,
        )

    def test_submillimetre_error_survives_sensor_grade_noise(self):
        """1.5 mm point noise must not mask a 0.5 mm length error."""
        report = self._report(
            self._bar(self.CERTIFIED_M + 0.0005, noise_m=0.0015, seed=11)
        )
        self.assertAlmostEqual(report["distance_error_m"], 0.0005, delta=0.0002)
        for sphere in report["spheres"]:
            self.assertGreater(sphere["form_error_rms_m"], 0.001)

    def test_a_single_sphere_is_an_error(self):
        points = sphere_points([0.0, 0.0, 0.14], self.DIAMETER_M / 2)
        with self.assertRaises(RuntimeError) as raised:
            self._report(points)
        self.assertIn("need 2", str(raised.exception))

    def test_a_cluster_that_is_not_the_right_sphere_is_rejected(self):
        points = np.vstack([
            sphere_points([0.0, 0.0, 0.14], self.DIAMETER_M / 2),
            # Sampled densely enough to survive clustering at the same spacing.
            sphere_points([self.CERTIFIED_M, 0.0, 0.14], 0.045, count=26000),
        ])
        with self.assertRaises(RuntimeError) as raised:
            self._report(points)
        self.assertIn("probably not a sphere", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
